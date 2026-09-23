# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# MatmulSwiglu golden reference (numpy)
import numpy as np

def silu(x):
    return x / (1.0 + np.exp(-x))

def matmul_swiglu_golden(x, weight, bias=None, transpose_weight=False):
    """
    x:      [M, K]
    weight: [K, 2N]  (transpose_weight=True -> [2N, K])
    bias:   [2N] or None
    return y: [M, N], y = SiLU(gate) * up, [gate|up] = x@weight (+bias)
    """
    w = weight.T if transpose_weight else weight
    mm = x.astype(np.float32) @ w.astype(np.float32)   # [M, 2N], fp32 accumulate
    if bias is not None:
        mm = mm + bias.astype(np.float32)
    two_n = mm.shape[-1]
    n = two_n // 2
    gate = mm[..., :n]
    up = mm[..., n:]
    y = silu(gate) * up
    return y

def gen_data(M=32, K=128, N=64, dtype=np.float16, seed=0, with_bias=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((M, K)).astype(dtype)
    w = rng.standard_normal((K, 2 * N)).astype(dtype)
    b = rng.standard_normal((2 * N,)).astype(dtype) if with_bias else None
    y = matmul_swiglu_golden(x, w, b).astype(dtype)
    return x, w, b, y

if __name__ == "__main__":
    # 自检: 与"打包权重一次GEMM + swi_glu(dim=-1)"基线等价
    x, w, b, y = gen_data(dtype=np.float32, with_bias=True)
    mm = x @ w + b
    n = w.shape[1] // 2
    base = silu(mm[:, :n]) * mm[:, n:]
    err = np.abs(base - y).max()
    print("self-consistency max abs err:", err)
    assert err < 1e-5
    print("golden shapes:", x.shape, w.shape, y.shape, "OK")
