"""
HiFloat8 向量化解码 (numpy), 与原版 generate_hifloat8_data.py 接口一致.
编码复用原版标量函数 (已验证正确), 解码使用向量化实现 (~5x 加速).
torch.Tensor 输入/输出接口.
"""
import math
import numpy as np
import torch

# 从原版导入标量编码函数 (已验证正确, 不做重复实现)
from generate_hifloat8_data import (
    _get_hif8_fraction_bits_number,
    _fp32_ta_round_to_hif8,
    _fp32_ssr_round_to_hif8,
    _fp16_ta_round_to_hif8,
    _fp16_ssr_round_to_hif8,
    cvt_float32_to_hifuint8,
    cvt_float16_to_hifuint8,
)


# ============================================================
# torch.Tensor 封装: 编码 (标量函数, 与原版一致)
# ============================================================

def trans_np_float_tensor_to_hifuint8(in_tensor, round_mode="round", over_mode=True):
    """输入: torch.Tensor/numpy, 输出同类型 (uint8)"""
    is_torch = isinstance(in_tensor, torch.Tensor)
    if is_torch:
        shape = in_tensor.shape
        x_np = in_tensor.detach().cpu().numpy()
    else:
        x_np = np.asarray(in_tensor)
        shape = x_np.shape
    multi_shape = int(np.prod(shape))
    out = np.zeros(multi_shape, dtype=np.float64)
    x_flat = x_np.ravel()
    is_f32 = (x_np.dtype == np.float32)
    for i in range(multi_shape):
        if is_f32:
            out[i] = cvt_float32_to_hifuint8(float(x_flat[i]), round_mode, over_mode)
        else:
            out[i] = cvt_float16_to_hifuint8(float(x_flat[i]), round_mode, over_mode)
    out = out.astype(np.uint8).reshape(shape)
    return torch.from_numpy(out.copy()) if is_torch else out


# ============================================================
# 向量化解码: hif8 (uint8) -> float32
# ============================================================

def _cvt_hif8_to_f32_vec(x_np, over_mode=True):
    """向量化 hif8 -> f32 解码, 与原版 cvt_hifuint8_to_float 逐元素等价"""
    x = x_np.ravel().astype(np.uint8)
    N = len(x)
    out = np.zeros(N, dtype=np.float32)

    if over_mode:
        m_zero, m_infp, m_infn, m_nan = (x == 0), (x == 111), (x == 239), (x == 128)
    else:
        m_zero, m_infp, m_infn, m_nan = (x == 0), (x == 110), (x == 238), (x == 128)
    out[m_zero] = np.float32(0.0)
    out[m_infp] = np.float32(np.inf)
    out[m_infn] = np.float32(-np.inf)
    out[m_nan] = np.float32(np.nan) if over_mode else np.float32(0.0)
    m_special = m_zero | m_infp | m_infn | m_nan

    m_norm = ~m_special
    if not m_norm.any():
        return out.reshape(x_np.shape)

    xn = x[m_norm]
    sgn = np.where(xn >= 128, -1.0, 1.0).astype(np.float32)
    xn7 = xn & 0x7F
    dot = xn7 >> 3

    m_dml = (dot == 0)
    m_d0 = (dot == 1)
    m_d1 = (dot == 2) | (dot == 3)
    m_d2 = (dot >= 4) & (dot <= 7)
    m_d3 = (dot >= 8) & (dot <= 11)
    m_d4 = (dot >= 12) & (dot <= 15)

    exp_raw = np.zeros_like(xn, dtype=np.int32)
    frac_raw = np.zeros_like(xn, dtype=np.int32)
    exp_bw = np.zeros_like(xn, dtype=np.int32)
    frac_bw = np.zeros_like(xn, dtype=np.int32)
    exp_bias = np.zeros_like(xn, dtype=np.int32)

    exp_raw[m_dml] = (xn7[m_dml] & 7).astype(np.int32)
    exp_bw[m_dml] = 3

    frac_raw[m_d0] = (xn7[m_d0] & 7).astype(np.int32)
    frac_bw[m_d0] = 3

    exp_raw[m_d1] = ((xn7[m_d1] >> 3) & 1).astype(np.int32)
    frac_raw[m_d1] = (xn7[m_d1] & 7).astype(np.int32)
    exp_bw[m_d1] = 1
    exp_bias[m_d1] = 1
    frac_bw[m_d1] = 3

    exp_raw[m_d2] = ((xn7[m_d2] >> 3) & 3).astype(np.int32)
    frac_raw[m_d2] = (xn7[m_d2] & 7).astype(np.int32)
    exp_bw[m_d2] = 2
    exp_bias[m_d2] = 2
    frac_bw[m_d2] = 3

    exp_raw[m_d3] = ((xn7[m_d3] >> 2) & 7).astype(np.int32)
    frac_raw[m_d3] = (xn7[m_d3] & 3).astype(np.int32)
    exp_bw[m_d3] = 3
    exp_bias[m_d3] = 4
    frac_bw[m_d3] = 2

    exp_raw[m_d4] = ((xn7[m_d4] >> 1) & 15).astype(np.int32)
    frac_raw[m_d4] = (xn7[m_d4] & 1).astype(np.int32)
    exp_bw[m_d4] = 4
    exp_bias[m_d4] = 8
    frac_bw[m_d4] = 1

    exp_val = np.where(m_dml,
                       exp_raw.astype(np.float32) - 23.0,
                       np.where(exp_raw < exp_bias,
                                (exp_raw + exp_bias).astype(np.float32),
                                (-exp_raw).astype(np.float32)))
    exp_val = np.where(exp_bw > 0, exp_val, np.float32(0.0))

    frac_den = np.power(np.float32(2.0), frac_bw.astype(np.float32))
    mant_val = np.float32(1.0) + frac_raw.astype(np.float32) / np.where(frac_bw > 0, frac_den, np.float32(1.0))
    mant_val = np.where(frac_bw > 0, mant_val, np.float32(1.0))
    mant_val[m_dml] = np.float32(1.0)

    pr = np.power(np.float32(2.0), exp_val)
    out[m_norm] = (sgn * pr * mant_val).astype(np.float32)
    return out.reshape(x_np.shape)


def trans_np_hifuint8_tensor_to_float32(in_tensor, over_mode=True):
    """输入: torch.Tensor/numpy, 输出同类型 (float32)"""
    is_torch = isinstance(in_tensor, torch.Tensor)
    if is_torch:
        x_np = in_tensor.detach().cpu().numpy().astype(np.uint8)
    else:
        x_np = np.asarray(in_tensor).astype(np.uint8)
    f32_np = _cvt_hif8_to_f32_vec(x_np, over_mode)
    return torch.from_numpy(f32_np.copy()) if is_torch else f32_np