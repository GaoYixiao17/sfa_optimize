import numpy as np

HIF8_POSITIVE_ZERO = 0x00
HIF8_NAN = 0x80
HIF8_POSITIVE_INF = 0x6F
HIF8_NEGATIVE_INF = 0xEF
HIF8_MAX_POSITIVE_NORMAL = 0x6E
HIF8_MAX_NEGATIVE_NORMAL = 0xEE
HIF8_OVERFLOW_THRESHOLD = (2.0 ** 15) * 1.25

def _as_numpy_uint8_codes(codes):
    arr = np.asarray(codes)
    if arr.dtype == np.bool_ or np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"codes must use an integer dtype, got: {arr.dtype}")
    if arr.size > 0 and arr.dtype != np.uint8:
        min_code = int(arr.min())
        max_code = int(arr.max())
        if min_code < 0 or max_code > 0xFF:
            raise ValueError(f"codes must be in [0, 255], got min={min_code}, max={max_code}")
    return arr.astype(np.uint8, copy=False)


def _decode_normal_payload_numpy(payload, d, mantissa_bits):
    mantissa_mask = (1 << mantissa_bits) - 1
    mantissa = (payload & mantissa_mask).astype(np.float32, copy=False)
    significand = np.float32(1.0) + mantissa / np.float32(1 << mantissa_bits)
    if d == 0:
        return significand

    exponent_payload = payload >> mantissa_bits
    sign_exp = (exponent_payload >> (d - 1)) & 1
    mag_tail_mask = (1 << (d - 1)) - 1
    magnitude = (1 << (d - 1)) | (exponent_payload & mag_tail_mask)
    exponent = np.where(sign_exp == 0, magnitude, -magnitude).astype(np.float32, copy=False)
    return np.power(np.float32(2.0), exponent) * significand

def hif8_to_fp32_numpy(codes):
    bits = _as_numpy_uint8_codes(codes).astype(np.int16, copy=False)
    payload = bits & 0x7F
    sign = (bits & 0x80) != 0
    out = np.zeros(bits.shape, dtype=np.float32)

    nan_mask = bits == HIF8_NAN
    if nan_mask.any():
        out[nan_mask] = np.nan

    inf_mask = payload == HIF8_POSITIVE_INF
    if inf_mask.any():
        out[inf_mask] = np.where(sign[inf_mask], -np.inf, np.inf)
    
    dml_mask = (payload <= 0x07) & ~nan_mask
    if dml_mask.any():
        mantissa = payload[dml_mask]
        nonzero = mantissa != 0
        if nonzero.any():
            values = np.power(np.float32(2.0), mantissa[nonzero].astype(np.float32) - np.float32(23.0))
            dml_values = out[dml_mask]
            dml_values[nonzero] = values
            out[dml_mask] = dml_values
    
    layouts = (
        (0, 0x78, 0x08, 3),
        (1, 0x70, 0x10, 3),
        (2, 0x60, 0x20, 3),
        (3, 0x60, 0x40, 2),
        (4, 0x60, 0x60, 1),
    )

    for d, prefix_mask, prefix_value, mantissa_bits in layouts:
        mask = ((payload & prefix_mask) == prefix_value) & ~inf_mask
        if mask.any():
            out[mask] = _decode_normal_payload_numpy(payload[mask], d, mantissa_bits)
    
    negative_finite = sign & np.isfinite(out) & (out != 0.0)
    if negative_finite.any():
        out[negative_finite] = -out[negative_finite]
    return out

def _positive_finite_codebook_numpy():
    codes = np.arange(128, dtype=np.uint8)
    values = hif8_to_fp32_numpy(codes)
    finite = np.isfinite(values)
    codes = codes[finite]
    values = values[finite]
    order = np.argsort(values)
    return values[order], codes[order]

def fp32_to_hif8_numpy(values, saturate=False, nan_to_zero=False):
    arr = np.asarray(values, dtype=np.float32)
    out = np.empty(arr.shape, dtype=np.uint8)
    
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        out[nan_mask] = HIF8_POSITIVE_ZERO if nan_to_zero else HIF8_NAN

    pos_inf_mask = arr == np.inf
    if pos_inf_mask.any():
        out[pos_inf_mask] = HIF8_MAX_POSITIVE_NORMAL if saturate else HIF8_POSITIVE_INF

    neg_inf_mask = arr == -np.inf
    if neg_inf_mask.any():
        out[neg_inf_mask] = HIF8_MAX_NEGATIVE_NORMAL if saturate else HIF8_NEGATIVE_INF
    
    finite_mask = np.isfinite(arr)
    if not finite_mask.any():
        return out
    
    magnitude = np.abs(arr[finite_mask]).astype(np.float32, copy=False)
    pos_values, pos_codes = _positive_finite_codebook_numpy()
    hi = np.searchsorted(pos_values, magnitude, side="left")
    max_idx = pos_values.size - 1
    hi_clamped = np.minimum(hi, max_idx)
    lo_clamped = np.maximum(hi - 1, 0)

    dist_hi = np.abs(pos_values[hi_clamped] - magnitude)
    dist_lo = np.abs(pos_values[lo_clamped] - magnitude)
    rounded_pos = np.where(dist_hi <= dist_lo, pos_codes[hi_clamped], pos_codes[lo_clamped]).astype(np.uint8, copy=False)

    overflow_mask = magnitude >= np.float32(HIF8_OVERFLOW_THRESHOLD)
    overflow_code = HIF8_MAX_POSITIVE_NORMAL if saturate else HIF8_POSITIVE_INF
    rounded_pos = np.where(overflow_mask, np.uint8(overflow_code), rounded_pos).astype(np.uint8, copy=False)

    negative = arr[finite_mask] < 0
    negative_codes = (rounded_pos.astype(np.int16) | 0x80).astype(np.uint8, copy=False)
    signed_codes = np.where(negative & (rounded_pos != 0), negative_codes, rounded_pos).astype(np.uint8, copy=False)
    out[finite_mask] = signed_codes
    return out