#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Benchmark driver for torch_bsa.blitz_sparse_attention.

Measures:
*  latency (usec) - NPU timer
*  correctness comparison with a reference PyTorch implementation

Sweeps every `(B, H, S, D, sparsity)` configuration across every entry in
`BLOCK_SHAPES`. Default is `[(128, 128), (128, 512)]` — the two sabi
granularities the kernel supports at runtime via the `block_shape` op-attr.
Each sweep row is labelled with the active block shape; narrow the list to a
single pair to benchmark only one granularity.
"""

import math
import random
import sys
import itertools
import logging
from collections import namedtuple
from typing import Callable, List, Tuple
import torch
import torch_npu
import torch_bsa

FrameWidths = namedtuple('FrameWidths', ['left_cols', 'right_cols', 'top_rows', 'bottom_rows'])

# Bundle of inputs for `create_attention_mask` — keeps the helper under the
# 5-arg cap (G.FNM.03). Combines the shape dims, block-tile sizes, target
# sparsity, and frame border into one named value.
MaskSpec = namedtuple(
    "MaskSpec",
    ["b", "h", "s_q", "s_kv", "d", "block_size_q", "block_size_kv", "sparsity", "frame"],
)

# Bundle of outputs from `create_attention_mask`. `bsa_atten_mask` is always
# None for the block-sparse pattern (per-tile masking comes from sabi); the
# field is kept for symmetry with the launcher signature.
AttentionMask = namedtuple(
    "AttentionMask",
    ["pfa_atten_mask", "bsa_atten_mask", "sabi", "sparse_mode", "scale", "pre_tokens", "next_tokens"],
)

# Configure logging — write to stdout (not the default stderr) so the result
# table can be piped into `plot.py`:
#   python benchmark.py | python plot.py
#       → produce the figure only; table is consumed by plot.py.
#   python benchmark.py | tee bench.log /dev/tty | python plot.py
#       → produce the figure, also save the table to `bench.log`, AND keep
#         printing it to the terminal (/dev/tty is an extra tee output).
logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Global Definitions
DEVICE = 'npu'
N_REPEATS = 10
N_WARMUP = 2
BLOCK_MASK_SEED = 1234



# ===== Begin of Parameter Sweep Definitions =====

DTYPE = torch.bfloat16
INPUT_LAYOUT = "BNSD"  # [batch_size, num_heads, seq_len, head_dim]
B_VALS = [1]
H_VALS = [3]
S_VALS = [118_806]  # S_q = S_kv
D_VALS = [128]   # head dimension

# Fraction of attention blocks zeroed out per row. 0.0 = dense, 0.9 = ~10 % kept.
SPARSITY_VALS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# For the block mask. BLOCK_SHAPES lists every (BLOCK_SIZE_Q, BLOCK_SIZE_KV) pair
# the benchmark sweeps over. The kernel accepts any pair from
# {128, 256, 512, 1024} × {128, 256, 512, 1024} at runtime via the `block_shape`
# op-attr. The single-value BLOCK_SIZE_Q / BLOCK_SIZE_KV constants are kept as
# informational defaults; the real driver of the sabi-stride choice is the
# per-pair value passed to torch_bsa.blitz_sparse_attention via block_shape.
BLOCK_SHAPES = [(bsq, bskv)
                for bsq in (128, 256, 512, 1024)
                for bskv in (128, 256, 512, 1024)]

# When True, each sabi row is generated with kept block indices sorted ascending
# (followed by pad_value). When False, the kept indices are shuffled within the row
# (padding still at the end). The kernel's sparse-loop assumes sorted sabi
# (`FirstGreaterEqual` linear scan + chunkIdx forward-walk), so setting this to
# False breaks correctness — use only for layout/perf experiments where the kernel
# has been adapted accordingly.
SABI_SORTED = True

# Force a frame of blocks to be always selected (still requires to be in the
# sparsity budget). The frame is defined in *token* terms and then rounded up
# to whole blocks, because `left_cols / right_cols` are in BLOCK_SIZE_KV-token
# units and `top_rows / bottom_rows` are in BLOCK_SIZE_Q-token units.
#
# The chosen footprint targets a typical generative-model sink/registers/recent
# pattern: keep the first ~3600 attention rows and the first ~3600 attention
# columns (these are the "always-attend" sink/system tokens), plus the last 6
# rows and last 6 columns (the most-recent / streaming-tail tokens). 6 tokens
# < 1 block in every direction, so right_cols / bottom_rows round up to 1.
#
# Per-pair entry math, where each value is rounded up to whole blocks:
#   top_rows    covers 3600 query tokens, giving 29, 15, 8, 4 blocks for BLOCK_SIZE_Q  of 128, 256, 512, 1024
#   left_cols   covers 3600 key   tokens, giving 29, 15, 8, 4 blocks for BLOCK_SIZE_KV of 128, 256, 512, 1024
#   right_cols  covers    6 key   tokens, which always rounds up to 1 block
#   bottom_rows covers    6 query tokens, which always rounds up to 1 block
#
# So the same conceptual "first 3600 + last 6 tokens" frame produces a forced-
# token footprint of ≈ 3600–4096 tokens per side at every block_shape pair,
# keeping the forced density comparable across granularities. At extreme shapes
# like 1024×1024 the frame alone is already ~8 % density, so the sparsity-0.9
# budget can be very tight and the host-side feasibility check may reject the
# configuration; `generate_sparse_blocks_by_row` raises ValueError in that case.
_LEFT_COLS_BY_KV = {128: 29, 256: 15, 512: 8, 1024: 4}
_TOP_ROWS_BY_Q = {128: 29, 256: 15, 512: 8, 1024: 4}
FRAMES_BY_BLOCK_SHAPE = {
    (bsq, bskv): FrameWidths(
        left_cols=_LEFT_COLS_BY_KV[bskv],
        right_cols=1,
        top_rows=_TOP_ROWS_BY_Q[bsq],
        bottom_rows=1,
    )
    for bsq in _TOP_ROWS_BY_Q
    for bskv in _LEFT_COLS_BY_KV
}

# Print tensors for manual comparisons
PRINT_OUTPUTS = False
PRINT_MASK = False

# For printing tensor differneces in blocks
PRINT_BLOCK_EQUALITY = False
PRINT_HEIGHT = 128
PRINT_WIDTH = 8

# True <--> enables accuracy (correctness) comparison
RUN_REFERENCE = False  

# Which reference kernel to compare/benchmark against. One of:
#  "custom"                          - our pythonic ref_blitz_sparse_attention_fp32
#  "npu_fusion_attention"            - torch_npu's npu_fusion_attention (PFA)
#  "npu_fused_infer_attention_score" - torch_npu's npu_fused_infer_attention_score (FIAS)
TORCH_REFERENCE = "npu_fusion_attention"

# ===== End of Parameter Sweep Definitions =====

torch.set_printoptions(
    threshold=100_000_000,
    linewidth=400,
    edgeitems=3,
    precision=4,
    sci_mode=False
)


def ref_blitz_sparse_attention_launcher(
    torch_reference: str,
    pfa_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float, torch.Tensor, str],
    force_dense_sm: bool,
    return_lse: bool = False,
):
    """
    runs a reference attention kernel, for correctness comparisons and for baseline time measurements.

    torch_reference: which reference to launch
      "custom"                          - launch our pythonic model `ref_blitz_sparse_attention_fp32`
      "npu_fusion_attention"            - launch `torch_npu.npu_fusion_attention` (PFA)
      "npu_fused_infer_attention_score" - launch `torch_npu.npu_fused_infer_attention_score` (FIAS)
    force_dense_sm - only consulted for the npu_* references:
      True  - apply sparse_mode=0 (dense) and don't use the provided atten_mask
      False - apply sparse_mode=1 (sparse token mask) and use the provided atten_mask
    return_lse - if True, return (attention_out, softmax_lse[B,H,S]).  Only the FIAS
                 reference natively produces softmax_lse; "custom" and
                 "npu_fusion_attention" raise ValueError when asked for it.
                 If False (default), return only attention_out.
    """
    q, k, v, head_num, scale, atten_mask, input_layout = pfa_inputs
    if torch_reference == "custom":
        if return_lse:
            raise ValueError("return_lse=True is not supported by the 'custom' reference "
                             "(ref_blitz_sparse_attention_fp32 does not compute softmax_lse).")
        return ref_blitz_sparse_attention_fp32(q, k, v, scale, atten_mask=atten_mask)
    if torch_reference == "npu_fusion_attention":
        if return_lse:
            raise ValueError("return_lse=True is not supported by 'npu_fusion_attention'; "
                             "use 'npu_fused_infer_attention_score' instead.")
        return torch_npu.npu_fusion_attention(q, k, v, head_num=head_num, input_layout=input_layout,
                                              scale=scale, pre_tockens=0, next_tockens=0,
                                              atten_mask=atten_mask,
                                              sparse_mode=0 if force_dense_sm else 1)[0]
    if torch_reference == "npu_fused_infer_attention_score":
        # FIAS is only supported by this launcher in dense mode (sparse_mode=0).
        # Reason: FIAS does not accept our atten_mask layout — its sparse_mode=1
        # path requires a [B, 1, S, S] mask (H broadcast), whereas this launcher's
        # `pfa_inputs` carries an atten_mask shaped per the caller's needs (often
        # [B, H, S, S]) and would silently mis-apply or hang the kernel. Callers
        # that need a sparse FIAS reference must invoke
        # torch_npu.npu_fused_infer_attention_score(...) directly so they own the
        # mask-shape constraint at the call site.
        if not force_dense_sm:
            raise ValueError(
                "torch_reference='npu_fused_infer_attention_score' requires "
                "force_dense_sm=True. FIAS does not support our atten_mask layout "
                "(sparse_mode=1 needs [B, 1, S, S]); call "
                "torch_npu.npu_fused_infer_attention_score(...) directly for sparse runs."
            )
        # FIAS expects per-batch actual sequence lengths; derive them from q/k shape according to input_layout.
        if input_layout == "BNSD":
            batch, s_q, s_kv = q.shape[0], q.shape[2], k.shape[2]
        elif input_layout in ("BSND", "BSH"):
            batch, s_q, s_kv = q.shape[0], q.shape[1], k.shape[1]
        else:
            raise ValueError(f"Unsupported input_layout for FIAS reference: {input_layout!r}")
        actual_seq_q = [s_q] * batch
        actual_seq_kv = [s_kv] * batch
        out, lse = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=head_num, scale=scale,
            input_layout=input_layout, num_key_value_heads=head_num,
            actual_seq_lengths=actual_seq_q, actual_seq_lengths_kv=actual_seq_kv,
            softmax_lse_flag=return_lse,
        )
        if return_lse:
            # FIAS emits lse as [B, H, S, 1]; squeeze to [B, H, S].
            return out, lse.squeeze(-1)
        return out
    raise ValueError(
        f"Unknown torch_reference value: {torch_reference!r}. "
        f"Expected one of: 'custom', 'npu_fusion_attention', 'npu_fused_infer_attention_score'."
    )


def _frame_exceeds_dims(frame, n_block_rows, n_block_cols):
    """True if any frame side is wider/taller than the available block grid."""
    cols_overflow = max(frame.left_cols, frame.right_cols) > n_block_cols
    rows_overflow = max(frame.top_rows, frame.bottom_rows) > n_block_rows
    return cols_overflow or rows_overflow


def _compute_frame_sets(frame, n_block_rows, n_block_cols):
    """Return (sparse_forced_col_ids, sparse_candidate_col_ids, dense_row_ids).

    `frame=None` means no border — all rows are sparse and all cols are
    candidates with nothing forced. Otherwise the left/right cols are forced
    in every sparse row, and the top/bottom rows are forced fully dense.
    """
    if frame is None:
        return set(), list(range(n_block_cols)), set()
    if _frame_exceeds_dims(frame, n_block_rows, n_block_cols):
        raise ValueError(
            f"Frame configuration exceeds block dimensions: "
            f"{frame=}, {n_block_rows=}, {n_block_cols=}"
        )
    # Forced column indices by the frame (left and right borders) for EVERY sparse row:
    sparse_forced_left = set(range(min(frame.left_cols, n_block_cols)))
    sparse_forced_right = set(range(max(n_block_cols - frame.right_cols, 0), n_block_cols))
    sparse_forced_col_ids = sparse_forced_left | sparse_forced_right
    sparse_candidate_col_ids = sorted(set(range(n_block_cols)) - sparse_forced_col_ids)
    # Row indices forced fully dense by the frame (top and bottom borders):
    dense_top = set(range(min(frame.top_rows, n_block_rows)))
    dense_bottom = set(range(max(n_block_rows - frame.bottom_rows, 0), n_block_rows))
    dense_row_ids = dense_top | dense_bottom
    return sparse_forced_col_ids, sparse_candidate_col_ids, dense_row_ids


def _build_sparse_row(rng, frame_sets, n_kept_per_row, n_block_cols, pad_value):
    """Pick the cols for a single sparse row: forced frame cols + `n_kept_per_row`
    random candidates, sorted, then padded out to width n_block_cols."""
    forced, candidates, _ = frame_sets
    if n_kept_per_row <= 0:
        cols = sorted(forced)
    else:
        sparse_selected = rng.sample(candidates, n_kept_per_row)
        cols = sorted(list(forced) + sparse_selected)
    return cols + [pad_value] * (n_block_cols - len(cols))


def _shuffle_row(rng, cols, pad_value):
    """Shuffle non-pad entries in-place (for sort_within_row=False)."""
    kept = [c for c in cols if c != pad_value]
    rng.shuffle(kept)
    return kept + [pad_value] * (len(cols) - len(kept))


def generate_sparse_blocks_by_row(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    seed: int,
    frame: FrameWidths,
    pad_value: int = -1,
    sort_within_row: bool = True,
) -> list[list[int]]:
    """
    Builds the sabi blocks (sparse blocks selection) of a single head, with a "frame" of
    always-included blocks and the rest of the blocks randomly selected to meet the target
    sparsity

    Note: blocks from the frame + sparsely selected blocks are both taken into account when computing the sparsity.

    Output:
      - shape [n_block_rows][n_block_cols]
      - each row: selected cols, then pad_value to the end. Cols are sorted ascending
        when sort_within_row=True (default); otherwise shuffled deterministically per seed
        (kernel sparse-loop assumes sorted — see SABI_SORTED).
    """
    sparsity = max(0.0, min(1.0, float(sparsity)))

    n_block_rows = math.ceil(s_q / block_size_q)
    n_block_cols = math.ceil(s_kv / block_size_kv)
    n_total_blocks = n_block_rows * n_block_cols
    if n_block_rows == 0 or n_block_cols == 0:
        raise ValueError(f"Invalid block configuration: {s_q=}, {s_kv=}, {block_size_q=}, {block_size_kv=}")

    frame_sets = _compute_frame_sets(frame, n_block_rows, n_block_cols)
    sparse_forced_col_ids, _, dense_row_ids = frame_sets

    n_dense_rows = len(dense_row_ids)
    n_sparse_rows = max(n_block_rows - n_dense_rows, 0)
    n_frame_forced_blocks = len(sparse_forced_col_ids) * n_sparse_rows + n_block_cols * n_dense_rows
    if n_frame_forced_blocks / n_total_blocks > (1.0 - sparsity):
        raise ValueError(
            f"Target sparsity violation: frame forces {n_frame_forced_blocks} blocks out of "
            f"{n_total_blocks} total attention blocks, implying a minimum density of "
            f"{n_frame_forced_blocks / n_total_blocks:.2%}, which violates the target sparsity "
            f"of {sparsity:.2%}. And this is even before allocating the random blocks inside the "
            f"frame. Consider adjusting frame sizes or target sparsity."
        )

    n_max_kept_blocks = int(math.floor(n_total_blocks * (1.0 - sparsity)))
    n_max_kept_blocks_excluding_frame = n_max_kept_blocks - n_frame_forced_blocks
    n_kept_row_blocks_per_row = (
        math.ceil(n_max_kept_blocks_excluding_frame / n_sparse_rows) if n_sparse_rows > 0 else 0
    )

    rng = random.Random(seed)
    rows: list[list[int]] = []
    for r in range(n_block_rows):
        if r in dense_row_ids:
            cols = list(range(n_block_cols))
        else:
            cols = _build_sparse_row(rng, frame_sets, n_kept_row_blocks_per_row, n_block_cols, pad_value)
        if not sort_within_row:
            cols = _shuffle_row(rng, cols, pad_value)
        rows.append(cols)

    return rows


BlockGrid = namedtuple("BlockGrid", ["s_q", "s_kv", "block_size_q", "block_size_kv"])


def is_block_sparse_pattern_feasible(
    grid: BlockGrid, frame, sparsity: float,
    *, require_nonempty_per_row: bool = True,
) -> bool:
    """Return True iff `generate_sparse_blocks_by_row` is satisfiable for the given grid.

    `grid` bundles (s_q, s_kv, block_size_q, block_size_kv); see `BlockGrid`.
    The check returns True iff `generate_sparse_blocks_by_row` will (a) not raise
    ValueError and (b) produce a sabi where every Q-row gets at least one kept
    block (when `require_nonempty_per_row`). Pure arithmetic — no tensor
    allocation — so it is safe to call from pytest parametrize lists at module
    import time.

    The check mirrors the math inside `generate_sparse_blocks_by_row`:

      * frame must fit the sparsity budget:
          n_forced_blocks  <=  floor(n_total * (1 - sparsity))
      * if `require_nonempty_per_row` is True (the default — BSA's per-Q-row
        LSE sentinel is the cliff we are avoiding), every sparse row must end
        up with `>= 1` kept block:
          n_max_kept - n_forced  >=  1   when  n_sparse_rows > 0

    Set `require_nonempty_per_row=False` if you intentionally want to test the
    "row fully masked" path (and accept BSA returning its sentinel).
    """
    sparsity = max(0.0, min(1.0, float(sparsity)))
    n_block_rows = math.ceil(grid.s_q / grid.block_size_q)
    n_block_cols = math.ceil(grid.s_kv / grid.block_size_kv)
    if n_block_rows == 0 or n_block_cols == 0:
        return False
    n_total = n_block_rows * n_block_cols

    if frame is None:
        n_forced = 0
        n_dense_rows = 0
        n_sparse_rows = n_block_rows
    else:
        # Frame configuration must fit inside the block grid (mirrors the
        # explicit check in generate_sparse_blocks_by_row).
        if _frame_exceeds_dims(frame, n_block_rows, n_block_cols):
            return False
        sparse_forced_left = set(range(min(frame.left_cols, n_block_cols)))
        sparse_forced_right = set(range(max(n_block_cols - frame.right_cols, 0), n_block_cols))
        n_forced_cols = len(sparse_forced_left | sparse_forced_right)
        dense_top = set(range(min(frame.top_rows, n_block_rows)))
        dense_bottom = set(range(max(n_block_rows - frame.bottom_rows, 0), n_block_rows))
        n_dense_rows = len(dense_top | dense_bottom)
        n_sparse_rows = max(n_block_rows - n_dense_rows, 0)
        n_forced = n_forced_cols * n_sparse_rows + n_block_cols * n_dense_rows

    n_max_kept = math.floor(n_total * (1.0 - sparsity))
    # (a) frame must fit:
    if n_forced > n_max_kept:
        return False
    # (b) every sparse row must get at least one kept block, otherwise BSA's
    #     uninitialised LSE sentinel (~1e12) ends up in the output and diverges
    #     from the reference's softmax-over-empty-set semantics (FIAS gives
    #     -inf, the Python ref gives -inf, BSA gives the sentinel).
    if require_nonempty_per_row and n_sparse_rows > 0 and n_max_kept - n_forced < 1:
        return False
    return True


def make_block_mask(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row: list[list[int]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, 1, s_q, s_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    n_block_rows = math.ceil(s_q / block_size_q)
    n_block_cols = math.ceil(s_kv / block_size_kv)

    mask = torch.ones((s_q, s_kv), dtype=torch.bool, device=device)

    rows_to_process = min(n_block_rows, len(sparse_blocks_by_row))
    for r in range(rows_to_process):
        row_start = r * block_size_q
        row_end = min(row_start + block_size_q, s_q)

        for c in sparse_blocks_by_row[r]:
            if 0 <= c < n_block_cols:
                col_start = c * block_size_kv
                col_end = min(col_start + block_size_kv, s_kv)
                mask[row_start:row_end, col_start:col_end] = False

    return mask.unsqueeze(0).unsqueeze(0)


def generate_sparse_blocks_by_row_per_head(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparsity: float,
    num_heads: int,
    frame: FrameWidths,
    base_seed: int,
    sort_within_row: bool = True,
) -> list[list[list[int]]]:
    """
    Returns sparse blocks per head.

    Output shape:
      [num_heads][n_block_rows][num_kept_elems]
    """
    return [
        generate_sparse_blocks_by_row(
            s_q=s_q,
            s_kv=s_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparsity=sparsity,
            seed=base_seed + h,
            frame=frame,
            sort_within_row=sort_within_row,
        )
        for h in range(num_heads)
    ]


def make_block_mask_per_head(
    s_q: int,
    s_kv: int,
    block_size_q: int,
    block_size_kv: int,
    sparse_blocks_by_row_per_head: list[list[list[int]]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Builds a dense boolean mask [1, num_heads, s_q, s_kv] from sparse block columns.

    True  = masked
    False = allowed
    """
    head_masks = [
        make_block_mask(
            s_q=s_q,
            s_kv=s_kv,
            block_size_q=block_size_q,
            block_size_kv=block_size_kv,
            sparse_blocks_by_row=sparse_blocks_by_row,
            device=device,
        ).squeeze(0).squeeze(0)  # [s_q, s_kv]
        for sparse_blocks_by_row in sparse_blocks_by_row_per_head
    ]

    return torch.stack(head_masks, dim=0).unsqueeze(0)


def block_allclose_map(
    out: torch.Tensor,
    ref: torch.Tensor,
    block_h: int,
    block_w: int,
    rtol: float = 0.01,
    atol: float = 0.001,
    print_map: bool = False,
) -> torch.Tensor:
    """
    Compare two 4D tensors [dim0, dim1, dim2, dim3] in (block_h x block_w) blocks over the last two dims.
    Returns a boolean tensor of shape [dim0, dim1, nby, nbx] where each entry indicates whether the
    entire block is allclose. Optionally prints a 0/1 block matrix per (batch, head).

    - First two dims: batch and head (iterated and printed as headers)
    - Last two dims: "drawn" dimensions (split into blocks)

    Blocks at the edges can be smaller if dim2 or dim3 is not divisible by block sizes.
    """
    if out.shape != ref.shape:
        raise ValueError(f"Shape mismatch: out {tuple(out.shape)} vs ref {tuple(ref.shape)}")
    if out.ndim != 4:
        raise ValueError(f"Expected 4D tensors [dim0, dim1, dim2, dim3], got out.ndim={out.ndim}")

    if block_h <= 0 or block_w <= 0:
        raise ValueError("block_h and block_w must be positive integers")

    dim0, dim1, dim2, dim3 = out.shape
    nby = (dim2 + block_h - 1) // block_h
    nbx = (dim3 + block_w - 1) // block_w

    # Elementwise closeness map: [dim0, dim1, dim2, dim3] boolean
    close = torch.isclose(out, ref, rtol=rtol, atol=atol)

    # Blockwise result: [dim0, dim1, nby, nbx]
    block_ok = close.new_empty((dim0, dim1, nby, nbx), dtype=torch.bool)

    for b, h, by, bx in itertools.product(range(dim0), range(dim1), range(nby), range(nbx)):
        y0, x0 = by * block_h, bx * block_w
        y1, x1 = min(y0 + block_h, dim2), min(x0 + block_w, dim3)
        block_ok[b, h, by, bx] = close[b, h, y0:y1, x0:x1].all()

    if print_map:
        for b, h in itertools.product(range(dim0), range(dim1)):
            logger.info(f"batch={b}, head={h}")
            # Print as 0/1 grid
            grid = block_ok[b, h].to(dtype=torch.int32)
            for by in range(nby):
                row = " ".join(str(int(v)) for v in grid[by].tolist())
                logger.info(row)
            logger.info()  # blank line between heads

    return block_ok


def ref_blitz_sparse_attention_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale_value: float,
    atten_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reference implementation for correctness comparisons.
    Scaled dot-product attention in float32 throughout.

    Assumes:
    - input_layout == "BNSD"
    - atten_mask (if provided) is bool broadcastable
      to [batch_size, num_heads, s_q, s_kv], where True means "masked out".
    """

    # Convert to fp32 for computation
    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    #    [batch_size, num_heads, s_q, head_dim] x [batch_size, num_heads, head_dim, s_kv]
    # -> [batch_size, num_heads, s_q, s_kv]
    attn_scores = torch.matmul(q_f, k_f.transpose(-1, -2))
    attn_scores = attn_scores * scale_value

    if atten_mask is not None:
        if atten_mask.dtype == torch.bool:
            attn_scores = attn_scores.masked_fill(
                atten_mask, torch.finfo(attn_scores.dtype).min
            )
        elif atten_mask.dtype in (torch.int8, torch.uint8):
            attn_scores = attn_scores.masked_fill(
                atten_mask == 1, torch.finfo(attn_scores.dtype).min
            )

    # Softmax in fp32
    attn_probs = torch.softmax(attn_scores, dim=-1)

    #    [batch_size, num_heads, s_q, s_kv] x [batch_size, num_heads, s_kv, head_dim] 
    # -> [batch_size, num_heads, s_q, head_dim]
    out = torch.matmul(attn_probs, v_f)

    # Cast back to original dtype for comparison
    return out.to(dtype=q.dtype)


def gen_pfa_inputs(
    batch_size: int, num_heads: int, s_q: int, s_kv: int, head_dim: int,
    device: str = "npu:0", dtype: torch.dtype = DTYPE
):
    """
    Generate common inputs for blitz_sparse_attention or prompt_flash_attention.
    """
    q = torch.randn(batch_size, num_heads, s_q, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch_size, num_heads, s_kv, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch_size, num_heads, s_kv, head_dim, dtype=dtype, device=device)

    # For now we assume all sequences are full-length
    actseqlen = [s_q] * batch_size
    actseqlenkv = [s_kv] * batch_size

    return q, k, v, actseqlen, actseqlenkv


def create_attention_mask(spec: MaskSpec, device: str, emit_atten_mask: bool = True) -> AttentionMask:
    """
    Build the block-sparse `sabi` tensor (and the matching token-level
    `pfa_atten_mask` if `emit_atten_mask`) for the batched block-sparse pattern
    that BSA expects. Each batch entry gets an independent random sabi seeded
    by `BLOCK_MASK_SEED + bidx`; `spec.frame` forces a fixed sink/streaming
    border on top of the sparsity budget.
    """
    scale = 1.0 / math.sqrt(float(spec.d))
    pre_tok, post_tok = 2147483647, 0
    bsa_atten_mask = None
    sm = 0

    per_batch_head_block_indices = [
        generate_sparse_blocks_by_row_per_head(
            spec.s_q, spec.s_kv, spec.block_size_q, spec.block_size_kv,
            spec.sparsity, spec.h, spec.frame, BLOCK_MASK_SEED + bidx,
            sort_within_row=SABI_SORTED,
        )
        for bidx in range(spec.b)
    ]
    sabi = torch.tensor(per_batch_head_block_indices, dtype=torch.uint16, device=device)
    pfa_atten_mask = None
    if emit_atten_mask and spec.sparsity > 0:
        pfa_atten_mask = torch.cat([
            make_block_mask_per_head(
                spec.s_q, spec.s_kv, spec.block_size_q, spec.block_size_kv,
                bi, device=device,
            )
            for bi in per_batch_head_block_indices
        ], dim=0)

    return AttentionMask(
        pfa_atten_mask=pfa_atten_mask, bsa_atten_mask=bsa_atten_mask, sabi=sabi,
        sparse_mode=sm, scale=scale, pre_tokens=pre_tok, next_tokens=post_tok,
    )


def _run_timed(kernel_fn: Callable, input_sets: List, n_warmup: int, n_repeat: int):
    """
    Utility function to run a kernel multiple times on the provided inputs 
    and report the average latency in microseconds. Warmup is appplied first (not timed).
    """
    for args in input_sets[:n_warmup]:
        kernel_fn(*args)
    torch.npu.synchronize()

    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for args in input_sets[n_warmup:]:
        kernel_fn(*args)
    end.record()
    torch.npu.synchronize()

    return start.elapsed_time(end) / n_repeat * 1000.0  # ms -> μs


def _check_correctness(our_fn: Callable, ref_fn: Callable, b: int, h: int, s_q: int, s_kv: int, d: int, device: str
    ) -> str:
    """Run both implementations on shared inputs and return 'yes'/'no'."""
    q, k, v, seq, seqkv = gen_pfa_inputs(b, h, s_q, s_kv, d, device=device, dtype=DTYPE)
    out_our = our_fn(q, k, v, seq, seqkv).cpu()
    out_ref = ref_fn(q, k, v, seq, seqkv).cpu()

    if PRINT_OUTPUTS:
        logger.info("OURS: ", out_our.shape)
        logger.info(out_our)
        logger.info("REF:  ", out_ref.shape)
        logger.info(out_ref)

    equal = torch.allclose(out_our, out_ref, rtol=0.01, atol=0.001)
    if not equal and PRINT_BLOCK_EQUALITY:
        block_allclose_map(out_our, out_ref,
                           block_h=PRINT_HEIGHT, block_w=PRINT_WIDTH,
                           rtol=0.01, atol=0.001, print_map=True)
    return "yes" if equal else "no"


def _fmt_or_na(value, width, spec=".2f"):
    """Format a number with given width/spec, or right-align 'N/A' if None."""
    if value is None:
        return f"{'N/A':>{width}}"
    return f"{value:{width}{spec}}"


def _make_our_fn(mask: AttentionMask, h, block_shape):
    def fn(q, k, v, seq, seqkv):
        # Returns (attention_out, softmax_lse); benchmark only needs attention_out
        out, _ = torch_bsa.blitz_sparse_attention(
            q, k, v,
            sabi=mask.sabi, actual_seq_lengths=seq,
            actual_seq_lengths_kv=seqkv, num_heads=h, num_key_value_heads=h,
            input_layout=INPUT_LAYOUT, scale_value=mask.scale,
            atten_mask=mask.bsa_atten_mask, sparse_mode=mask.sparse_mode,
            pre_tokens=mask.pre_tokens, next_tokens=mask.next_tokens,
            softmax_lse_flag=False, # False = don't comppute the 2nd LSE output
            block_shape=list(block_shape),
        )
        return out
    return fn


def _make_ref_fn(h, scale, atten_mask, run_ref_sparsity_0):
    def fn(q, k, v, seq, seqkv):
        return ref_blitz_sparse_attention_launcher(
                torch_reference=TORCH_REFERENCE, 
                pfa_inputs=(q, k, v, h, scale, atten_mask, INPUT_LAYOUT),
                force_dense_sm=run_ref_sparsity_0,
            )
    return fn


_BenchCell = namedtuple(
    "_BenchCell",
    ["bs_label", "block_shape", "frame_for_shape", "b", "h", "s_q", "s_kv", "d", "sparsity"],
)
_BenchCfg = namedtuple("_BenchCfg", ["run_our", "run_ref", "n_warmup", "n_repeat"])


def _format_frame_str(frame, sparsity):
    """Compact "(L,R,T,B)" or "-" when frame is unset or sparsity is dense."""
    if frame is None or sparsity == 0:
        return "-"
    return (f"({frame.left_cols},{frame.right_cols},"
            f"{frame.top_rows},{frame.bottom_rows})")


def _run_one_benchmark_cell(cell: _BenchCell, cfg: _BenchCfg):
    """Build the mask, time our/ref kernels, and emit one results row."""
    mask = create_attention_mask(
        MaskSpec(b=cell.b, h=cell.h, s_q=cell.s_q, s_kv=cell.s_kv, d=cell.d,
                 block_size_q=cell.block_shape[0], block_size_kv=cell.block_shape[1],
                 sparsity=cell.sparsity, frame=cell.frame_for_shape),
        device=DEVICE, emit_atten_mask=cfg.run_ref,
    )
    if PRINT_MASK and mask.pfa_atten_mask is not None:
        logger.info(mask.pfa_atten_mask.int())
        logger.info(mask.pfa_atten_mask.shape)

    # At sparsity 0 always run the reference as a dense sanity check.
    run_ref_sparsity_0 = cell.sparsity == 0
    our_fn = _make_our_fn(mask, cell.h, cell.block_shape)
    ref_fn = _make_ref_fn(cell.h, mask.scale, mask.pfa_atten_mask, run_ref_sparsity_0)

    are_equal_ref = "N/A"
    if cfg.run_our and cfg.run_ref or run_ref_sparsity_0:
        are_equal_ref = _check_correctness(
            our_fn, ref_fn, cell.b, cell.h, cell.s_q, cell.s_kv, cell.d, device=DEVICE,
        )

    inputs = [gen_pfa_inputs(cell.b, cell.h, cell.s_q, cell.s_kv, cell.d, device=DEVICE, dtype=DTYPE)
              for _ in range(cfg.n_warmup + cfg.n_repeat)]
    our_duration = _run_timed(our_fn, inputs, cfg.n_warmup, cfg.n_repeat) if cfg.run_our else None
    ref_duration = (_run_timed(ref_fn, inputs, cfg.n_warmup, cfg.n_repeat)
                    if cfg.run_ref or run_ref_sparsity_0 else None)

    frame_str = _format_frame_str(cell.frame_for_shape, cell.sparsity)
    logger.info(
        f"{cell.bs_label:>13} {cell.h:>3} {cell.b:>3} {cell.s_q:>6} {cell.s_kv:>6} {cell.d:>4} "
        f"{frame_str:>15} {cell.sparsity:>9.2f} "
        f"{are_equal_ref:>15} {_fmt_or_na(ref_duration, 18)} {_fmt_or_na(our_duration, 18)}"
    )


def benchmark_blitz_sparse_attention():
    cfg = _BenchCfg(run_our=True, run_ref=RUN_REFERENCE,
                    n_warmup=N_WARMUP, n_repeat=N_REPEATS)
    if not cfg.run_our and not cfg.run_ref:
        logger.info("Nothing to run, must set run_our=True or run_ref=True")
        return

    logger.info("=" * 120)
    logger.info(f"  {DTYPE=}  {INPUT_LAYOUT=}  {SABI_SORTED=}  {TORCH_REFERENCE=}")
    logger.info("=" * 120)
    logger.info(
        f"{'block_shape':>13} {'H':>3} {'B':>3} {'s_q':>6} {'s_kv':>6} {'D':>4} "
        f"{'frame(L,R,T,B)':>15} {'sparsity':>9} "
        f"{'Outputs_equal':>15} {'Ref_Latency_[usec]':>18} {'Our_Latency_[usec]':>18}"
    )
    logger.info("-" * 120)

    for block_shape in BLOCK_SHAPES:
        bs_q, bs_kv = block_shape
        # Scale the frame per (BLOCK_SIZE_Q, BLOCK_SIZE_KV) so the forced-token
        # footprint stays comparable across block shapes.
        frame_for_shape = FRAMES_BY_BLOCK_SHAPE.get(block_shape, None)
        for b, h, s_kv, d, sparsity in itertools.product(B_VALS, H_VALS, S_VALS, D_VALS, SPARSITY_VALS):
            _run_one_benchmark_cell(
                _BenchCell(bs_label=f"{bs_q}x{bs_kv}", block_shape=block_shape,
                           frame_for_shape=frame_for_shape,
                           b=b, h=h, s_q=s_kv, s_kv=s_kv, d=d, sparsity=sparsity),
                cfg,
            )

    logger.info("=" * 120)


if __name__ == "__main__":
    torch.npu.set_device(DEVICE)
    benchmark_blitz_sparse_attention()
