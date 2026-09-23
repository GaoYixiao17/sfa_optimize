#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
test_joint.py – verifies attention_out and softmax_lse from the *same* BSA call.

Every test makes a single blitz_sparse_attention(..., softmax_lse_flag=True) and
asserts both outputs simultaneously against FIAS:

  • dense  (sparse_mode=0, no mask)       any H, bfloat16 + float16   → 14 tests
  • sparse (sparse_mode=0, sabi)  H=1     bfloat16 only               →  6 tests

Sparse tests are limited to H=1 because FIAS sparse_mode=1 requires a
[B,1,S,S] mask; H>1 shapes would cause an NPU hang.
float16 is excluded from sparse tests because fp16 sparse LSE diverges beyond
the tolerance at high sparsity (see test_lse.py design notes).
Masks are built on CPU to avoid hundreds of per-block NPU launches.

Run with:
    pytest benchmark/test_joint.py -v
"""

import logging
import os
import math
import sys
from collections import namedtuple
import pytest
import torch
import torch_npu
import torch_bsa
from benchmark import create_attention_mask, FrameWidths, MaskSpec

# Bundle of QKV tensors plus the kernel-config scalars (num_heads, scale,
# seq_len) shared by every `_run_*` helper. Passing this single object keeps
# helpers under the 5-argument limit (G.FNM.03) and lets fields keep
# snake_case names (G.NAM.01) without exposing dim-letter args like `H` / `S`.
AttnInputs = namedtuple("AttnInputs", ["q", "k", "v", "num_heads", "scale", "seq_len"])

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = "npu:0"
INPUT_LAYOUT = "BNSD"
# block_shape pairs exercised by the suite. All 16 combinations of
# {128, 256, 512, 1024} × {128, 256, 512, 1024} share the same kernel binary.
BLOCK_SHAPES = [(bsq, bskv)
                for bsq in (128, 256, 512, 1024)
                for bskv in (128, 256, 512, 1024)]
torch.npu.set_device(DEVICE)


@pytest.fixture(params=BLOCK_SHAPES, ids=lambda bs: f"bs{bs[0]}x{bs[1]}")
def block_shape(request):
    """Parametrized fixture: each test runs once per (Q,KV) pair."""
    return request.param

# Per-dtype attention_out tolerance. bfloat16 uses an fp32 accumulator path
# and stays within 1e-3 against FIAS. float16 keeps the accumulation in fp16:
# a single attention-output element accumulates across thousands of KV tokens
# at ~5e-4 per-token noise floor, so the realistic atol is ~5e-3 (max-diff
# observed across the sweep: ≈ 0.005–0.009). Mirrors how test_lse.py handles
# the same fp16-noise pattern.
OUT_ATOL = {
    torch.bfloat16: 1e-3,
    torch.float16: 1e-2,
}
SPARSE_OUT_ATOL = 1e-3   # sparse attention_out vs PFA: slight headroom over 0.01
                         # for block-boundary rounding (BSA skips blocks; PFA masks with -inf).
                         # Only bfloat16 is used for sparse tests (see module docstring).
# Per-dtype softmax_lse tolerance. bfloat16's online-softmax accumulator runs
# in fp32, so the BSA vs FIAS diff is at the float32-rounding floor
# (max observed ≈ 1e-6). float16 keeps the accumulator in fp16; max observed
# across the sweep is ≈ 0.005.
LSE_ATOL = {
    torch.bfloat16: 1e-5,
    torch.float16: 5e-3,
}


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


NO_FRAME = FrameWidths(0, 0, 0, 0)

# Dense: varied H (1,2,4,8) and S (1024–16384); both dtypes → 14 tests
DENSE_SHAPES = [
    (1, 1, 1024, 128),
    (1, 2, 2048, 128),
    (1, 4, 4096, 128),
    (1, 8, 4096, 128),   # high head count
    (1, 4, 8192, 128),   # longer sequence, moderate H
    (1, 2, 12288, 128),   # matches test_lse.py shapes
    (1, 8, 16384, 128),   # high H + long sequence
]
DENSE_DTYPES = [torch.bfloat16, torch.float16]

# Sparse cases are restricted to a single head and bfloat16.
# Cases are explicit shape-and-sparsity pairs rather than a cross-product, so that
# the number of kept KV blocks (sequence length S over 128 multiplied by one minus
# sparsity) stays at or above 15 in every case.  With fewer than about 15 kept KV
# blocks the softmax denominator becomes tiny and BSA vs PFA precision differences
# exceed SPARSE_OUT_ATOL — not a bug in either kernel, just a regime that
# test_attn.py sidesteps by only testing S of 24000 or above.
#
# The cases below list batch, head, sequence-length, head-dim, the requested
# sparsity, total KV blocks, and floor of kept KV blocks:
#   1,  1,  4096, 128, sparsity 0.3, 32 blocks, 22 kept — included
#   1,  1,  4096, 128, sparsity 0.5, 32 blocks, 16 kept — included
#   1,  1,  4096, 128, sparsity 0.7, 32 blocks, 10 kept — excluded
#   1,  1,  8192, 128, sparsity 0.3, 64 blocks, 45 kept — included
#   1,  1,  8192, 128, sparsity 0.5, 64 blocks, 32 kept — included
#   1,  1,  8192, 128, sparsity 0.7, 64 blocks, 19 kept — included
#   1,  1, 16384, 128, sparsity 0.7, 128 blocks, 38 kept — included
#   1,  1, 16384, 128, sparsity 0.9, 128 blocks, 13 kept — excluded
SPARSE_CASES = [
    ((1, 1, 4096, 128), 0.3),
    ((1, 1, 4096, 128), 0.5),
    ((1, 1, 8192, 128), 0.3),
    ((1, 1, 8192, 128), 0.5),
    ((1, 1, 8192, 128), 0.7),
    ((1, 1, 16384, 128), 0.7),
]


# ── Kernel helpers ────────────────────────────────────────────────────────────

def _run_bsa(inp: AttnInputs, block_shape, sabi=None):
    batch = inp.q.shape[0]
    actual_seq = [inp.seq_len] * batch
    return torch_bsa.blitz_sparse_attention(
        inp.q, inp.k, inp.v,
        sabi=sabi,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        num_heads=inp.num_heads, num_key_value_heads=inp.num_heads,
        input_layout=INPUT_LAYOUT, scale_value=inp.scale,
        sparse_mode=0, softmax_lse_flag=True,
        block_shape=list(block_shape),
    )


def _run_fias_dense(inp: AttnInputs):
    batch = inp.q.shape[0]
    actual_seq = [inp.seq_len] * batch
    out, lse = torch_npu.npu_fused_infer_attention_score(
        inp.q, inp.k, inp.v,
        num_heads=inp.num_heads, scale=inp.scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=inp.num_heads,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)   # [B,H,S,1] → [B,H,S]


def _run_pfa_sparse(inp: AttnInputs, pfa_atten_mask):
    """npu_fusion_attention (PFA) with sparse_mode=1 — same reference as test_attn.py."""
    return torch_npu.npu_fusion_attention(
        inp.q, inp.k, inp.v,
        head_num=inp.num_heads, input_layout=INPUT_LAYOUT, scale=inp.scale,
        pre_tockens=0, next_tockens=0,
        atten_mask=pfa_atten_mask, sparse_mode=1,
    )[0]


def _run_fias_sparse(inp: AttnInputs, pfa_atten_mask):
    """FIAS sparse_mode=1 — only used for the LSE reference."""
    batch = inp.q.shape[0]
    actual_seq = [inp.seq_len] * batch
    out, lse = torch_npu.npu_fused_infer_attention_score(
        inp.q, inp.k, inp.v,
        num_heads=inp.num_heads, scale=inp.scale,
        input_layout=INPUT_LAYOUT, num_key_value_heads=inp.num_heads,
        actual_seq_lengths=actual_seq, actual_seq_lengths_kv=actual_seq,
        atten_mask=pfa_atten_mask, sparse_mode=1,
        softmax_lse_flag=True,
    )
    return out, lse.squeeze(-1)


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", DENSE_DTYPES, ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("shape", DENSE_SHAPES,
                          ids=lambda s: f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}")
def test_joint_dense(shape, dtype, block_shape):
    """Dense BSA: attention_out and softmax_lse both match FIAS in one call."""
    torch.manual_seed(SEED)
    b, h, s, d = shape
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    k = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    v = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)

    inp = AttnInputs(q=q, k=k, v=v, num_heads=h, scale=scale, seq_len=s)
    bsa_out, bsa_lse = _run_bsa(inp, block_shape)
    ref_out, ref_lse = _run_fias_dense(inp)

    assert bsa_lse.shape == torch.Size([b, h, s]), f"lse shape wrong: {bsa_lse.shape}"
    assert bsa_lse.dtype == torch.float32, f"lse dtype wrong: {bsa_lse.dtype}"

    atol = OUT_ATOL[dtype]
    a, b = bsa_out.cpu(), ref_out.cpu()
    max_diff_out = _log_diff(
        f"joint_dense_out {dtype} shape={shape} bs={block_shape}", a, b
    )
    if not torch.allclose(a, b, rtol=0.01, atol=atol):
        pytest.fail(f"attention_out: shape={shape} dtype={dtype} "
                    f"max_diff={max_diff_out:.5f} > atol={atol}")

    a, b = bsa_lse.cpu(), ref_lse.cpu().float()
    max_diff_lse = _log_diff(
        f"joint_dense_lse {dtype} shape={shape} bs={block_shape}", a, b
    )
    lse_atol = LSE_ATOL[dtype]
    if not torch.allclose(a, b, rtol=1e-3, atol=lse_atol):
        pytest.fail(f"softmax_lse:  shape={shape} dtype={dtype} "
                    f"max_diff={max_diff_lse:.5f} > atol={lse_atol}")


@pytest.mark.parametrize("shape,sparsity", SPARSE_CASES,
                          ids=[f"b{s[0]}-h{s[1]}-s{s[2]}-d{s[3]}-sp{int(sp*100):02d}pct"
                               for s, sp in SPARSE_CASES])
def test_joint_sparse(shape, sparsity, block_shape):
    """Sparse BSA (H=1, bf16): attention_out and softmax_lse both match FIAS sparse_mode=1."""
    torch.manual_seed(SEED)
    b, h, s, d = shape
    assert h == 1
    dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(d)
    bs_q, bs_kv = block_shape

    try:
        mask = create_attention_mask(
            MaskSpec(b=b, h=h, s_q=s, s_kv=s, d=d,
                     block_size_q=bs_q, block_size_kv=bs_kv,
                     sparsity=sparsity, frame=NO_FRAME),
            device="cpu", emit_atten_mask=True,
        )
    except ValueError as e:
        pytest.skip(f"sparsity {sparsity} unsatisfiable for {block_shape}: {e}")
    sabi = mask.sabi.to(DEVICE)
    pfa_atten_mask = mask.pfa_atten_mask.to(DEVICE)

    q = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    k = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)
    v = torch.randn(b, h, s, d, dtype=dtype, device=DEVICE)

    inp = AttnInputs(q=q, k=k, v=v, num_heads=h, scale=scale, seq_len=s)
    bsa_out, bsa_lse = _run_bsa(inp, block_shape, sabi=sabi)
    # attention_out: compare against npu_fusion_attention (PFA) — same reference
    # as test_attn.py; PFA and BSA share implementation lineage so their outputs
    # FIAS uses a different tiling
    # and diverges from BSA when the softmax denominator is small.
    pfa_out = _run_pfa_sparse(inp, pfa_atten_mask)
    # softmax_lse: compare against FIAS — same reference as test_lse.py.
    _, fias_lse = _run_fias_sparse(inp, pfa_atten_mask)

    assert bsa_lse.shape == torch.Size([b, h, s]), f"lse shape wrong: {bsa_lse.shape}"
    assert bsa_lse.dtype == torch.float32, f"lse dtype wrong: {bsa_lse.dtype}"

    a, b = bsa_out.cpu(), pfa_out.cpu()
    max_diff_out = _log_diff(
        f"joint_sparse_out shape={shape} sp={sparsity} bs={block_shape}", a, b
    )
    if not torch.allclose(a, b, rtol=0.01, atol=SPARSE_OUT_ATOL):
        pytest.fail(f"attention_out: shape={shape} sparsity={sparsity} "
                    f"max_diff={max_diff_out:.5f} > atol={SPARSE_OUT_ATOL}")

    a, b = bsa_lse.cpu(), fias_lse.cpu().float()
    max_diff = _log_diff(
        f"joint_sparse_lse shape={shape} sp={sparsity} bs={block_shape}", a, b
    )
    # sparse tests are bfloat16-only (see module docstring), so look up the
    # bf16 entry directly.
    lse_atol = LSE_ATOL[torch.bfloat16]
    if not torch.allclose(a, b, rtol=1e-3, atol=lse_atol):
        max_diff = (bsa_lse.cpu() - fias_lse.cpu().float()).abs().max().item()
        pytest.fail(f"softmax_lse:  shape={shape} sparsity={sparsity} "
                    f"max_diff={max_diff:.5f} > atol={lse_atol}")


# ── __main__ smoke tests ──────────────────────────────────────────────────────

if __name__ == "__main__":
    SMOKE_BS = (128, 128)

    test_joint_dense((1, 2, 2048, 128), torch.bfloat16, SMOKE_BS)
    logger.info("dense joint bf16 passed")

    test_joint_dense((1, 2, 2048, 128), torch.float16, SMOKE_BS)
    logger.info("dense joint fp16 passed")

    test_joint_sparse((1, 1, 4096, 128), 0.5, SMOKE_BS)
    logger.info("sparse joint H=1 bf16 sp=50% passed")
