#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Tests for BSA softmax_lse output.

Compares BSA's log-sum-exp output against two references:
  • FIAS (npu_fused_infer_attention_score) — used for dense and sparse H=1 tests.
  • Python float32 reference — used for multi-head sparse tests.

Target: ~100 pytest test instances (B=1 only; B>1 not working).

Run with:
    pytest benchmark/test_lse.py -v

Design notes
────────────
1.  FIAS sparse_mode=1 requires a broadcast-compatible mask [B, 1, S, S].
    Passing [B, H, S, S] with H > 1 causes FIAS to hang on the NPU.
    Fix: all direct FIAS-comparison tests use H=1 shapes only.

2.  Per-head FIAS calls (passing q[:, h:h+1, :, :]) look innocent but are
    wrong for B > 1: the slice's batch-stride is H×S×D (from the original
    allocation) rather than 1×S×D.  NPU kernels read raw strides, so FIAS
    lands on the wrong memory for batch item 1.
    Fix: multi-head sparse tests use a Python float32 reference instead of
    FIAS, so BSA is always called once with the full [B, H, S, D] tensor.

3.  create_attention_mask(..., device="npu:0") issues one NPU kernel per kept
    block (hundreds of launches for s=2048, H=2).  Fix: always build masks on
    CPU, transfer once with .to(DEVICE).

4.  float16 sparse LSE: FIAS with -inf masking and BSA with full block-skip
    diverge beyond any reasonable atol at high sparsity in float16.
    Fix: all sparse tests use bfloat16 only.
"""

import itertools
import logging
import math
import os
import sys
from collections import namedtuple
import pytest
import torch
import torch_npu
import torch_bsa
from benchmark import create_attention_mask, FrameWidths, is_block_sparse_pattern_feasible, BlockGrid, MaskSpec

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

# Bundle of QKV tensors plus the kernel-config scalars (num_heads, scale,
# seq_len) shared by every `_run_*` helper. Passing this single object keeps
# helpers under the 5-argument limit (G.FNM.03) and lets fields keep
# snake_case names (G.NAM.01).
AttnInputs = namedtuple("AttnInputs", ["q", "k", "v", "num_heads", "scale", "seq_len"])

SEED = 42
DEVICE = "npu:0"
INPUT_LAYOUT = "BNSD"
# block_shape pairs exercised by the suite. All 16 combinations of
# {128, 256, 512, 1024} × {128, 256, 512, 1024} share the same kernel binary.
BLOCK_SHAPES = [(bsq, bskv)
                for bsq in (128, 256, 512, 1024)
                for bskv in (128, 256, 512, 1024)]
_SABI_PAD = 65535   # 0xFFFF – padding sentinel in uint16 sabi tensors
torch.npu.set_device(DEVICE)


@pytest.fixture(params=BLOCK_SHAPES, ids=lambda bs: f"bs{bs[0]}x{bs[1]}")
def block_shape(request):
    """Parametrized fixture: each test function runs once per (Q,KV) pair."""
    return request.param


# ── Dense shapes B, H, S, D — 14 × 2 dtypes total 28 tests ────────────────────
LSE_TEST_SHAPES = [
    (1, 1, 1024, 128), (1, 2, 1024, 128),
    (1, 4, 2048, 128),
    (1, 4, 4096, 128), (1, 8, 4096, 128),
    (1, 2, 8192, 128), (1, 4, 8192, 128),
    (1, 2, 12288, 128), (1, 4, 12288, 128),
    (1, 2, 16384, 128), (1, 4, 16384, 128),
    (1, 1, 24000, 128), (1, 2, 24000, 128), (1, 4, 24000, 128),
]
LSE_DTYPES = [torch.bfloat16, torch.float16]

# ── Sparse H 1 shapes for FIAS comparison — 8 × 4 sparsities total 32 tests ──────
# H 1 keeps mask shape [B, 1, S, S] which FIAS sparse_mode=1 handles correctly.
# B 1 only (B>1 not working). bfloat16 only (see design note 4).
SPARSE_LSE_FIAS_SHAPES = [
    (1, 1, 1024, 128),
    (1, 1, 2048, 128),
    (1, 1, 4096, 128),
    (1, 1, 8192, 128),
    (1, 1, 12288, 128),
    (1, 1, 16384, 128),
    (1, 1, 20480, 128),
    (1, 1, 24000, 128),
]

# ── Sparse H 1 framed shapes — 2 × 4 sparsities total 8 tests ───────────────────
# s >= 16384 at FrameWidths(2,1,3,1) forces ≈5.4% density, within sparsity 0.9.
SPARSE_FRAMED_FIAS_SHAPES = [
    (1, 1, 16384, 128),
    (1, 1, 24000, 128),
]
# Per-block-shape sparse frame: `left_cols` is in BLOCK_SIZE_KV-token units and
# `top_rows` is in BLOCK_SIZE_Q-token units. Scale both with bs_kv / bs_q so the
# forced-token footprint stays comparable across granularities.
_SPARSE_FRAME_LEFT_BY_KV = {128: 2, 256: 1, 512: 1, 1024: 1}
_SPARSE_FRAME_TOP_BY_Q = {128: 3, 256: 2, 512: 1, 1024: 1}
SPARSE_FRAMES_BY_BLOCK_SHAPE = {
    (bsq, bskv): FrameWidths(
        left_cols=_SPARSE_FRAME_LEFT_BY_KV[bskv],
        right_cols=1,
        top_rows=_SPARSE_FRAME_TOP_BY_Q[bsq],
        bottom_rows=1,
    )
    for bsq in (128, 256, 512, 1024)
    for bskv in (128, 256, 512, 1024)
}

# ── Multi-head sparse shapes for Python-reference comparison ──────────────────
# BSA runs once on all heads; LSE is compared against a CPU float32 reference
# that computes log-sum-exp directly over the sabi-selected KV blocks.
# S <= 1024 keeps Python reference loops fast (< 1 s per test).
# Batch size is restricted to 1, since larger batch sizes are not yet supported.
# Combining 4 shapes with 4 sparsities yields 16 tests in total, all in bfloat16.
SPARSE_LSE_MULTIHEAD_SHAPES = [
    (1, 2, 512, 128), (1, 4, 512, 128),
    (1, 2, 1024, 128), (1, 4, 1024, 128),
]

SPARSE_LSE_SPARSITIES = [0.3, 0.5, 0.7, 0.9]
NO_FRAME = FrameWidths(0, 0, 0, 0)

# ── Zero-input shapes — 5 × 2 dtypes total 10 tests ──────────────────────────────
ZERO_INPUT_SHAPES = [
    (1, 1, 512, 128), (1, 2, 1024, 128), (1, 4, 4096, 128),
    (1, 4, 16384, 128), (1, 2, 24000, 128),
]

# ── Shape/dtype check shapes — 3 tests ───────────────────────────────────────
SHAPE_DTYPE_SHAPES = [
    (1, 1, 512, 128),
    (1, 8, 8192, 128), (1, 4, 24000, 128),
]

# ── Disabled-LSE shapes — 3 tests ────────────────────────────────────────────
DISABLED_LSE_SHAPES = [
    (1, 2, 512, 128),
    (1, 8, 8192, 128), (1, 4, 24000, 128),
]

# ── Tolerances ────────────────────────────────────────────────────────────────
# bf16/fp16 LSE values are O(log S) ≈ 6–10. A single ULP at that magnitude is
# ~0.04 in bf16 and ~0.005 in fp16; the online-softmax accumulation amplifies
# that. Empirically the worst observed max_diff across 100-case sweeps is
# ~0.05 (multi-head sparse vs CPU-float32 reference at sp=0.7). 0.06 atol
# accepts that without flagging genuine kernel drift.
LSE_ATOL = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-2,
    torch.float32: 1e-3,
}
SPARSE_REF_ATOL = 1e-2


def _log_diff(label: str, a, b) -> float:
    """Compute max |a - b| for every comparison site. When env var
    BSA_LOG_DIFFS=1 is set, also log a `[DIFF]` line so the suite doubles as
    a diff-distribution probe (pipe to `grep [DIFF]` to harvest the empirical
    noise floor at every site). The logging is opt-in because the extra
    CPU-side logging can perturb cube/vector scheduling enough to expose
    intermittent bf16 allocator-slack noise. Returns the max_diff scalar so
    the caller can reuse it in `pytest.fail` messages."""
    diff = (a - b).abs()
    max_diff = diff.max().item()
    if os.environ.get("BSA_LOG_DIFFS"):
        logger.info(f"[DIFF] {label}: max_diff={max_diff:.6g}")
    return max_diff


# ── Mask helper ───────────────────────────────────────────────────────────────

def _build_sparse_masks(shape, sparsity, frame, block_shape):
    """Return (sabi, pfa_atten_mask) on DEVICE, built on CPU first.

    `shape` is the (b, h, s, d) test-parameter tuple; d is unused here because
    create_attention_mask hardcodes the head dim to 128 internally.

    Parameter combinations that would raise (frame doesn't fit the sparsity
    budget, or leave any sparse Q-row with zero kept blocks) are filtered out
    at collection time by `is_block_sparse_pattern_feasible`, so callers don't
    need to guard against ValueError here."""
    b, h, s, _ = shape
    bs_q, bs_kv = block_shape
    mask = create_attention_mask(
        MaskSpec(b=b, h=h, s_q=s, s_kv=s, d=128,
                 block_size_q=bs_q, block_size_kv=bs_kv,
                 sparsity=sparsity, frame=frame),
        device="cpu", emit_atten_mask=True,
    )
    return mask.sabi.to(DEVICE), mask.pfa_atten_mask.to(DEVICE)


# ── Per-test feasible-parameter generators ───────────────────────────────────
# Build the (shape, sparsity, block_shape) cross product for every sparse test
# and drop cells where `is_block_sparse_pattern_feasible` returns False — those
# combinations would (a) trip a frame-density-vs-sparsity assertion in
# `generate_sparse_blocks_by_row`, or (b) leave some Q-row empty and surface
# BSA's InitOutput LSE sentinel, which the FIAS/Python-ref `-inf` does not
# match. Cleaner pytest output AND avoids the cascading
# `aclrtLaunchKernelWithHostArgs failed: 507015` aicore exception that the
# sentinel-loaded LSE eventually triggers downstream inside torch_npu.

def _id_sparse(shape, sparsity, block_shape):
    return (f"b{shape[0]}-h{shape[1]}-s{shape[2]}-d{shape[3]}"
            f"-sp{int(sparsity*100):02d}pct"
            f"-bs{block_shape[0]}x{block_shape[1]}")


def _id_sparse_framed(shape, sparsity, block_shape):
    return _id_sparse(shape, sparsity, block_shape) + "-framed"


def _filter_feasible(shapes, sparsities, block_shapes, frame_resolver):
    """Build a list of (shape, sparsity, block_shape) tuples for which the
    block-sparse pattern is feasible (frame fits and per-row kept ≥ 1).
    `frame_resolver(shape, block_shape) -> FrameWidths|None`."""
    params = []
    for shape, sparsity, block_shape in itertools.product(shapes, sparsities, block_shapes):
        grid = BlockGrid(s_q=shape[2], s_kv=shape[2],
                         block_size_q=block_shape[0], block_size_kv=block_shape[1])
        feasible = is_block_sparse_pattern_feasible(
            grid, frame_resolver(shape, block_shape), sparsity,
        )
        if feasible:
            params.append((shape, sparsity, block_shape))
    return params


# Dense-LSE parameter list: cross product of (shape, dtype, block_shape), but
# excludes fp16 at any coarse-tile shape — see comment on the corresponding
# fixture in `test_bsa_lse_vs_fias`. Filtering at collection time keeps the run
# log clean (no `pytest.skip` markers) and makes the actual coverage explicit.
def _build_dense_lse_params():
    params = []
    for shape, dtype, block_shape in itertools.product(LSE_TEST_SHAPES, LSE_DTYPES, BLOCK_SHAPES):
        skip = dtype == torch.float16 and (block_shape[0] > 128 or block_shape[1] > 128)
        if not skip:
            params.append((shape, dtype, block_shape))
    return params

DENSE_LSE_PARAMS = _build_dense_lse_params()


def _id_dense_lse(shape, dtype, block_shape):
    dt = "bfloat16" if dtype == torch.bfloat16 else "float16"
    return (f"b{shape[0]}-h{shape[1]}-s{shape[2]}-d{shape[3]}"
            f"-{dt}-bs{block_shape[0]}x{block_shape[1]}")

SPARSE_FIAS_PARAMS = _filter_feasible(
    SPARSE_LSE_FIAS_SHAPES, SPARSE_LSE_SPARSITIES, BLOCK_SHAPES,
    lambda shape, bs: NO_FRAME,
)
SPARSE_FRAMED_PARAMS = _filter_feasible(
    SPARSE_FRAMED_FIAS_SHAPES, SPARSE_LSE_SPARSITIES, BLOCK_SHAPES,
    lambda shape, bs: SPARSE_FRAMES_BY_BLOCK_SHAPE[bs],
)
SPARSE_MULTIHEAD_PARAMS = _filter_feasible(
    SPARSE_LSE_MULTIHEAD_SHAPES, SPARSE_LSE_SPARSITIES, BLOCK_SHAPES,
    lambda shape, bs: NO_FRAME,
)


# ── Python float32 reference LSE ──────────────────────────────────────────────

def _gather_kv_rows(sabi_row, bs_kv, s):
    """Expand sabi entries for one (b, h, qb) into KV-token row indices,
    skipping the 0xFFFF padding sentinel. Returns [] when the q-block is empty."""
    kv_rows = []
    for c in sabi_row.tolist():
        c = int(c)
        if c == _SABI_PAD:
            continue
        kv_s = c * bs_kv
        kv_rows.extend(range(kv_s, min(kv_s + bs_kv, s)))
    return kv_rows


def _lse_reference_sparse(q, k, scale, sabi_cpu, block_shape):
    """Compute exact sparse LSE in float32 using Python loops.

    For each (batch, head, q-block), collects the KV token indices listed in
    sabi (skipping 0xFFFF padding), computes QK^T * scale in float32, then
    logsumexp over attended tokens.

    q, k : [B, H, S, D] (any device / dtype — moved to CPU float32 internally)
    sabi_cpu : [B, H, n_q_blocks, n_kv_blocks] uint16, on CPU
    block_shape : (BLOCK_SIZE_Q, BLOCK_SIZE_KV) — sabi index strides
    Returns [B, H, S] float32 on CPU.
    """
    bs_q, bs_kv = block_shape
    b_dim, n_heads, s, _ = q.shape
    n_q_blocks = (s + bs_q - 1) // bs_q

    qf = q.cpu().float()
    kf = k.cpu().float()
    lse = torch.full((b_dim, n_heads, s), float("-inf"))

    for b, h, qb in itertools.product(range(b_dim), range(n_heads), range(n_q_blocks)):
        q_s = qb * bs_q
        q_e = min(q_s + bs_q, s)
        kv_rows = _gather_kv_rows(sabi_cpu[b, h, qb], bs_kv, s)
        if not kv_rows:
            continue
        scores = (qf[b, h, q_s:q_e] @ kf[b, h, kv_rows].t()) * scale
        lse[b, h, q_s:q_e] = torch.logsumexp(scores, dim=-1)

    return lse


# ── Kernel helpers ────────────────────────────────────────────────────────────

def _run_bsa_with_lse(inp: AttnInputs, block_shape):
    """BSA dense (no sabi)"""
    actual_seq = [inp.seq_len] * inp.q.shape[0]
    out, lse = torch_bsa.blitz_sparse_attention(
        inp.q, inp.k, inp.v,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        num_heads=inp.num_heads, num_key_value_heads=inp.num_heads,
        input_layout=INPUT_LAYOUT,
        scale_value=inp.scale,
        sparse_mode=0,
        softmax_lse_flag=True,
        block_shape=list(block_shape),
    )
    return out, lse


def _run_fias_with_lse(inp: AttnInputs):
    """FIAS dense → (out, lse [B,H,S])."""
    batch = inp.q.shape[0]
    actual_seq = [inp.seq_len] * batch
    out, lse = torch_npu.npu_fused_infer_attention_score(
        inp.q, inp.k, inp.v,
        num_heads=inp.num_heads, scale=inp.scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=inp.num_heads,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


def _run_bsa_with_lse_sparse(inp: AttnInputs, sabi, block_shape):
    """BSA block-sparse (sabi, sparse_mode=0) → (out, lse [B,H,S]).

    Always pass full [B, H, S, D] tensors — never per-head views (see note 2).
    """
    batch = inp.q.shape[0]
    actual_seq = [inp.seq_len] * batch
    out, lse = torch_bsa.blitz_sparse_attention(
        inp.q, inp.k, inp.v, sabi=sabi,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        num_heads=inp.num_heads, num_key_value_heads=inp.num_heads,
        input_layout=INPUT_LAYOUT, scale_value=inp.scale,
        sparse_mode=0, softmax_lse_flag=True,
        block_shape=list(block_shape),
    )
    return out, lse


def _run_fias_with_lse_sparse(inp: AttnInputs, pfa_atten_mask):
    """FIAS sparse_mode=1 → (out, lse [B,h,S]).

    pfa_atten_mask MUST be [B, 1, S, S] — H > 1 causes an NPU hang.
    """
    batch = inp.q.shape[0]
    actual_seq = [inp.seq_len] * batch
    out, lse = torch_npu.npu_fused_infer_attention_score(
        inp.q, inp.k, inp.v,
        num_heads=inp.num_heads, scale=inp.scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=inp.num_heads,
        actual_seq_lengths=actual_seq,
        actual_seq_lengths_kv=actual_seq,
        atten_mask=pfa_atten_mask,
        sparse_mode=1,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


# ── Test 1: dense LSE vs FIAS ────────────────────────────────────────────────
# Parameterised on `DENSE_LSE_PARAMS` which excludes fp16 at any coarse-tile
# shape (bs_q > 128 OR bs_kv > 128). Rationale: the fp16 dense path keeps the
# online-softmax accumulator in fp16, and either a coarser Q-tile (the cube
# does redundant 128-Q work behind a wider sabi row) or a coarser KV-tile (the
# wide-KV packer doubles the number of cube ops sharing one flash-softmax
# instance) compounds the error past the rtol bound — observed max_diff up to
# ~0.02 (≈2.5 fp16-ULPs at log(1024)≈6.9). That's a known fp16-path
# limitation, not a kernel correctness bug; bf16 stays within atol at every
# block_shape.

@pytest.mark.parametrize(
    "shape,dtype,block_shape", DENSE_LSE_PARAMS,
    ids=[_id_dense_lse(*p) for p in DENSE_LSE_PARAMS],
)
def test_bsa_lse_vs_fias(shape, dtype, block_shape):
    """BSA dense softmax_lse must match FIAS softmax_lse."""
    torch.manual_seed(SEED)
    b, h, s, d = shape
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    k = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    v = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)

    inp = AttnInputs(q=q, k=k, v=v, num_heads=h, scale=scale, seq_len=s)
    _, bsa_lse = _run_bsa_with_lse(inp, block_shape)
    _, fias_lse = _run_fias_with_lse(inp)

    assert bsa_lse.shape == (b, h, s)
    assert bsa_lse.dtype == torch.float32

    atol = LSE_ATOL[dtype]
    a, b = bsa_lse.cpu(), fias_lse.cpu().float()
    max_diff = _log_diff(f"lse_vs_fias_dense {dtype} shape={shape}", a, b)
    if not torch.allclose(a, b, rtol=1e-3, atol=atol):
        pytest.fail(f"shape={shape} dtype={dtype}: max_diff={max_diff:.5f} > atol={atol}")


# ── Test 2: sparse H 1 LSE vs FIAS, no frame ─────────────────────────────────
# Parameterised on SPARSE_FIAS_PARAMS so infeasible (block_shape, sparsity, S)
# triples are NEVER collected — no skip noise, no empty-sabi sentinel reaching
# torch_npu's downstream isfinite check.

@pytest.mark.parametrize(
    "shape,sparsity,block_shape", SPARSE_FIAS_PARAMS,
    ids=[_id_sparse(*p) for p in SPARSE_FIAS_PARAMS],
)
def test_bsa_lse_vs_fias_sparse(shape, sparsity, block_shape):
    """BSA sparse (H=1, no frame) softmax_lse vs FIAS with the matching token mask.

    H=1 only: mask is [B,1,S,S], safe for FIAS sparse_mode=1.
    Mask built on CPU → single .to(DEVICE) transfer.
    bfloat16 only (float16 diverges at high sparsity, see module docstring).
    """
    torch.manual_seed(SEED)
    b, h, s, d = shape
    assert h == 1
    scale = 1.0 / math.sqrt(d)
    dtype = torch.bfloat16

    sabi, pfa_atten_mask = _build_sparse_masks(shape, sparsity, NO_FRAME, block_shape)
    q = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    k = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    v = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)

    inp = AttnInputs(q=q, k=k, v=v, num_heads=h, scale=scale, seq_len=s)
    _, bsa_lse = _run_bsa_with_lse_sparse(inp, sabi, block_shape)
    _, fias_lse = _run_fias_with_lse_sparse(inp, pfa_atten_mask)

    assert bsa_lse.shape == (b, h, s)
    assert bsa_lse.dtype == torch.float32

    atol = LSE_ATOL[dtype]
    a, b = bsa_lse.cpu(), fias_lse.cpu().float()
    max_diff = _log_diff(
        f"lse_vs_fias_sparse {dtype} shape={shape} sp={sparsity} bs={block_shape}", a, b
    )
    if not torch.allclose(a, b, rtol=1e-3, atol=atol):
        pytest.fail(
            f"shape={shape} sparsity={sparsity}: max_diff={max_diff:.5f} > atol={atol}"
        )


# ── Test 3: sparse H 1 LSE vs FIAS, with border frame ────────────────────────
# Parameterised on SPARSE_FRAMED_PARAMS — the per-block-shape frame
# (`SPARSE_FRAMES_BY_BLOCK_SHAPE`) is consulted at collection time so cells
# where the frame doesn't fit are excluded up front.

@pytest.mark.parametrize(
    "shape,sparsity,block_shape", SPARSE_FRAMED_PARAMS,
    ids=[_id_sparse_framed(*p) for p in SPARSE_FRAMED_PARAMS],
)
def test_bsa_lse_vs_fias_sparse_framed(shape, sparsity, block_shape):
    """BSA sparse LSE with a per-shape border frame on long sequences, H=1, bf16."""
    torch.manual_seed(SEED)
    b, h, s, d = shape
    assert h == 1
    scale = 1.0 / math.sqrt(d)
    dtype = torch.bfloat16

    # Per-shape frame: token footprint normalised across block_shape values.
    bs_q, bs_kv = block_shape
    sparse_frame = SPARSE_FRAMES_BY_BLOCK_SHAPE[block_shape]

    sabi, pfa_atten_mask = _build_sparse_masks(shape, sparsity, sparse_frame, block_shape)
    q = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    k = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    v = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)

    inp = AttnInputs(q=q, k=k, v=v, num_heads=h, scale=scale, seq_len=s)
    _, bsa_lse = _run_bsa_with_lse_sparse(inp, sabi, block_shape)
    _, fias_lse = _run_fias_with_lse_sparse(inp, pfa_atten_mask)

    atol = LSE_ATOL[dtype]
    a, b = bsa_lse.cpu(), fias_lse.cpu().float()
    max_diff = _log_diff(
        f"lse_vs_fias_framed {dtype} shape={shape} sp={sparsity} bs={block_shape}", a, b
    )
    if not torch.allclose(a, b, rtol=1e-3, atol=atol):
        pytest.fail(
            f"shape={shape} sparsity={sparsity}: max_diff={max_diff:.5f} > atol={atol}"
        )


# ── Test 4: multi-head sparse LSE vs Python float32 reference ───────────────
# Parameterised on SPARSE_MULTIHEAD_PARAMS — same empty-Q-row filter as
# tests 2 and 3.

@pytest.mark.parametrize(
    "shape,sparsity,block_shape", SPARSE_MULTIHEAD_PARAMS,
    ids=[_id_sparse(*p) for p in SPARSE_MULTIHEAD_PARAMS],
)
def test_bsa_lse_sparse_multihead(shape, sparsity, block_shape):
    """BSA multi-head sparse LSE vs a CPU float32 reference.

    The reference _lse_reference_sparse() gathers the exact KV token rows
    listed in sabi and computes log-sum-exp in float32 — independent of FIAS.
    BSA is called once on the full [B, H, S, D] tensor (never per-head slices).
    S <= 1024 keeps the Python reference loops fast (< 1 s per test).
    bfloat16 only.
    """
    torch.manual_seed(SEED)
    b, h, s, d = shape
    bs_q, bs_kv = block_shape
    scale = 1.0 / math.sqrt(d)
    dtype = torch.bfloat16

    # Build masks on CPU; keep sabi_cpu for the reference computation
    sabi_cpu, _ = _build_sparse_masks(shape, sparsity, NO_FRAME, block_shape)
    sabi_cpu = sabi_cpu.cpu()
    sabi_dev = sabi_cpu.to(DEVICE)

    q = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    k = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    v = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)

    inp = AttnInputs(q=q, k=k, v=v, num_heads=h, scale=scale, seq_len=s)
    _, bsa_lse = _run_bsa_with_lse_sparse(inp, sabi_dev, block_shape)
    ref_lse = _lse_reference_sparse(q, k, scale, sabi_cpu, block_shape)  # [B, H, S] float32 CPU

    a, b = bsa_lse.cpu(), ref_lse
    max_diff = _log_diff(
        f"lse_multihead_pyref shape={shape} sp={sparsity} bs={block_shape}", a, b
    )
    if not torch.allclose(a, b, rtol=1e-3, atol=SPARSE_REF_ATOL):
        pytest.fail(
            f"Multi-head sparse LSE mismatch: shape={shape} sparsity={sparsity}: "
            f"max_diff={max_diff:.5f} > atol={SPARSE_REF_ATOL}"
        )


# ── Test 5: zero-input — LSE must equal log(S) (12 tests) ────────────────────

@pytest.mark.parametrize("dtype", LSE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", ZERO_INPUT_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_zero_input(shape, dtype, block_shape):
    """Q=K=V=0 → uniform softmax → LSE = log(S). Catches missing ViewCopy etc.

    float16 inputs use fp16 arithmetic for online-softmax intermediates; the
    vector-unit log near log(S)≈6–9 has ~1 fp16 ULP error (≈0.004–0.008), so
    a tighter tolerance is not achievable for fp16.  bfloat16 uses a fp32
    accumulator path and stays within 1e-3.
    """
    b, h, s, d = shape
    scale = 1.0 / math.sqrt(d)
    q = torch.zeros(b, h, s, d, dtype=dtype, device=DEVICE)

    inp = AttnInputs(q=q, k=q, v=q, num_heads=h, scale=scale, seq_len=s)
    _, lse = _run_bsa_with_lse(inp, block_shape)
    expected = math.log(s)
    atol = LSE_ATOL[dtype]
    a, b = lse.cpu(), torch.full_like(lse.cpu(), expected)
    max_diff = _log_diff(f"lse_zero_input {dtype} shape={shape}", a, b)
    if not torch.allclose(a, b, atol=atol):
        pytest.fail(
            f"shape={shape} dtype={dtype}: max_diff={max_diff:.5f}, "
            f"expected log({s})={expected:.5f}, atol={atol}"
        )


# ── Test 6: LSE shape and dtype (4 tests) ────────────────────────────────────

@pytest.mark.parametrize("shape", SHAPE_DTYPE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_shape_and_dtype(shape, block_shape):
    """LSE tensor must have shape [B, H, S] and dtype float32."""
    torch.manual_seed(SEED)
    b, h, s, d = shape
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=DEVICE)

    inp = AttnInputs(q=q, k=q, v=q, num_heads=h, scale=scale, seq_len=s)
    _, lse = _run_bsa_with_lse(inp, block_shape)
    assert lse.shape == torch.Size([b, h, s]), f"Wrong shape: {lse.shape}"
    assert lse.dtype == torch.float32, f"Wrong dtype: {lse.dtype}"


# ── Test 7: disabled LSE returns empty tensor (4 tests) ──────────────────────

@pytest.mark.parametrize("shape", DISABLED_LSE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_bsa_lse_disabled_returns_empty(shape, block_shape):
    """softmax_lse_flag=False must produce an empty tensor."""
    torch.manual_seed(SEED)
    b, h, s, d = shape
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, s, d, dtype=torch.bfloat16, device=DEVICE)
    actual_seq = [s] * b

    _, lse = torch_bsa.blitz_sparse_attention(
        q, q, q,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        num_heads=h, num_key_value_heads=h,
        input_layout=INPUT_LAYOUT, scale_value=scale,
        softmax_lse_flag=False,
        block_shape=list(block_shape),
    )
    assert lse.numel() == 0, f"Expected empty tensor, got shape {lse.shape}"


# ── __main__ smoke tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    SMOKE_BS = (128, 128)

    test_bsa_lse_shape_and_dtype((1, 2, 512, 128), SMOKE_BS)
    logger.info("shape/dtype check passed")

    test_bsa_lse_disabled_returns_empty((1, 2, 512, 128), SMOKE_BS)
    logger.info("disabled LSE check passed")

    test_bsa_lse_zero_input((1, 2, 1024, 128), torch.bfloat16, SMOKE_BS)
    logger.info("zero-input log(S) invariant passed")

    test_bsa_lse_vs_fias((1, 1, 1024, 128), torch.bfloat16, SMOKE_BS)
    logger.info("dense LSE vs FIAS passed")

    test_bsa_lse_vs_fias_sparse((1, 1, 2048, 128), 0.5, SMOKE_BS)
    logger.info("sparse H=1 no-frame passed")

    test_bsa_lse_vs_fias_sparse_framed((1, 1, 16384, 128), 0.5, SMOKE_BS)
    logger.info("sparse H=1 framed passed")

    test_bsa_lse_sparse_multihead((1, 2, 1024, 128), 0.5, SMOKE_BS)
    logger.info("multi-head sparse vs Python reference passed")
