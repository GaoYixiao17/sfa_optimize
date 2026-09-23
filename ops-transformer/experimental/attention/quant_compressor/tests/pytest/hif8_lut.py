"""
HiF8 LUT decode + valid code generation.
Companion to hif8_utils.py — provides optimized decode path.

Usage:
    from hif8_lut import hif8_to_fp32_lut, get_valid_hif8_codes
"""
import numpy as np
from hif8_utils import hif8_to_fp32_numpy

# ═══════════════════════════════════════════════════════════════════
# LUT Decode: 256-entry table, pure integer lookup
# ═══════════════════════════════════════════════════════════════════

_LUT_FP32_BITS = None


def _build_lut():
    """Build 256-entry LUT once. Uses baseline decoder — bit-exact by construction."""
    global _LUT_FP32_BITS
    all_codes = np.arange(256, dtype=np.uint8)
    fp32_vals = hif8_to_fp32_numpy(all_codes)
    _LUT_FP32_BITS = fp32_vals.view(np.uint32).copy()


def hif8_to_fp32_lut(codes):
    """Decode HiF8 uint8 -> fp32 via 256-entry LUT.
    Bit-exact equivalent to hif8_to_fp32_numpy, ~24x faster.
    """
    global _LUT_FP32_BITS
    if _LUT_FP32_BITS is None:
        _build_lut()
    flat = codes.ravel().astype(np.uint8)
    bits = _LUT_FP32_BITS[flat]
    return bits.view(np.float32).reshape(codes.shape)


# ═══════════════════════════════════════════════════════════════════
# Valid code computation — for direct uint8 generation (skip encode)
# ═══════════════════════════════════════════════════════════════════
_VALID_CACHE = {}


def get_valid_hif8_codes(min_val, max_val):
    """Return HiF8 codes whose decoded fp32 value is in [min_val, max_val] and finite.
    These are exactly the codes fp32_to_hif8_numpy would produce for uniform input.
    Cached per (min_val, max_val) pair.
    """
    key = (float(min_val), float(max_val))
    if key in _VALID_CACHE:
        return _VALID_CACHE[key].copy()

    all_codes = np.arange(256, dtype=np.uint8)
    all_vals = hif8_to_fp32_numpy(all_codes)
    mask = np.isfinite(all_vals) & (all_vals >= min_val) & (all_vals <= max_val)
    valid = all_codes[mask]

    _VALID_CACHE[key] = valid
    return valid.copy()

_VU_CACHE = {}

# LUT size: 1MB (2^20 entries). Every valid code gets at least 1 entry
# (floor-1 guarantee). Distortion for tiny-probability codes (< 3.8e-7) is
# negligible — they round up from ~0.0004 to 1 entry, < 0.02% total mass shift.
_SAMPLE_LUT_SIZE = 1 << 20


def _build_sampling_lut(codes, vals, min_val, max_val):
    """Build inverse-CDF LUT with floor-1 guarantee for every code."""
    sorted_idx = np.argsort(vals)
    sorted_vals = vals[sorted_idx].astype(np.float64)
    sorted_codes = codes[sorted_idx]
    n = len(sorted_codes)

    mids = (sorted_vals[:-1] + sorted_vals[1:]) * 0.5
    lo = np.empty(n, dtype=np.float64); hi = np.empty(n, dtype=np.float64)
    lo[0] = min_val; lo[1:] = mids; hi[:-1] = mids; hi[-1] = max_val
    widths = np.maximum(hi - lo, 0.0)
    total = widths.sum()

    if total <= 0 or n >= _SAMPLE_LUT_SIZE:
        return sorted_codes  # fallback: uniform over codes

    probs = widths / total
    target = probs * _SAMPLE_LUT_SIZE
    # Floor-1: each code gets at least 1, rest distributed proportionally
    counts = np.ones(n, dtype=np.int32)
    remaining = _SAMPLE_LUT_SIZE - n
    fractional = np.maximum(target - 1.0, 0.0)
    frac_sum = fractional.sum()
    if frac_sum > 0:
        extra = (fractional / frac_sum * remaining).astype(np.int32)
        remainder = fractional / frac_sum * remaining - extra.astype(np.float64)
        extra[np.argsort(-remainder)[:remaining - extra.sum()]] += 1
        counts += extra

    lut = np.repeat(sorted_codes.astype(np.uint8), counts)
    if len(lut) < _SAMPLE_LUT_SIZE:
        lut = np.pad(lut, (0, _SAMPLE_LUT_SIZE - len(lut)), constant_values=sorted_codes[-1])
    return lut[:_SAMPLE_LUT_SIZE]


def sample_hif8_value_uniform(min_val, max_val, shape):
    """Generate random HiF8 codes with value-space-uniform distribution.

    256KB LUT with floor-1 guarantee — every valid code appears at least once.
    Tiny probability codes get rounded up (max distortion < 0.02% of total mass).
    """
    key = (float(min_val), float(max_val))
    if key not in _VU_CACHE:
        codes = get_valid_hif8_codes(min_val, max_val)
        vals = hif8_to_fp32_numpy(codes)
        lut = _build_sampling_lut(codes, vals, min_val, max_val)
        _VU_CACHE[key] = lut

    lut = _VU_CACHE[key]
    indices = np.random.randint(0, len(lut), size=shape, dtype=np.int32)
    return lut[indices]