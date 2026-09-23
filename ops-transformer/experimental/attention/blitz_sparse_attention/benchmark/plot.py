#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Plot speedup vs sparsity from `benchmark.py` stdout.

Reads the benchmark table (one row per (block_shape, B, H, S, D, sparsity)
config) from stdin, then draws one line per distinct (sequence_length,
block_shape) pair. y = `npu_fusion_attention` (dense reference) latency at
sparsity 0 divided by our_latency at the corresponding sparsity.

Usage
-----
    python benchmark.py | python plot.py
    python plot.py < bench.log
    python plot.py --output bench.png < bench.log

A horizontal y = 1 line marks parity with the dense reference. The reference
is measured only at sparsity 0 in `benchmark.py` (one Ref_Latency cell per
block_shape × seq_len pair), so the same reference baseline is used for every
sparsity of the matching (seq_len, block_shape) line.
"""

import argparse
import logging
import re
import sys
from collections import defaultdict, namedtuple

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stderr)
logger = logging.getLogger(__name__)

# Composite sort key produced by `_sort_key` — wraps the six lexicographic
# fields in a single named value (G.FNM.05). Inherits tuple comparison, so
# `sorted(..., key=_sort_key)` orders identically to the prior tuple return.
_SortKey = namedtuple("_SortKey", ["bskv", "bsq", "s_q", "h", "b", "d"])


# Result row format produced by `benchmark.py`:
# block_shape  H  B    s_q   s_kv    D  frame(L,R,T,B)  sparsity   Outputs_equal Ref_Latency_[usec] Our_Latency_[usec]
#     128x128  3  1 118806 118806  128               -      0.00             yes           160225.13         185475.31
#     128x256  3  1 118806 118806  128     (29,1,29,1)      0.05             N/A                N/A          164822.39
# The frame column is "-" when sparsity == 0 (dense, frame irrelevant) or
# "(L,R,T,B)" otherwise; matched as a single non-whitespace token and discarded.
ROW_RE = re.compile(
    r"^\s*"
    r"(?P<bsq>\d+)x(?P<bskv>\d+)\s+"
    r"(?P<h>\d+)\s+"
    r"(?P<b>\d+)\s+"
    r"(?P<sq>\d+)\s+"
    r"(?P<skv>\d+)\s+"
    r"(?P<d>\d+)\s+"
    r"(?P<frame>\S+)\s+"
    r"(?P<sp>[\d.]+)\s+"
    r"(?P<eq>\S+)\s+"
    r"(?P<ref>\S+)\s+"
    r"(?P<ours>\S+)\s*$"
)


def _to_float(token: str):
    if token == "N/A":
        return None
    try:
        return float(token)
    except ValueError:
        return None


def parse_rows(stream):
    """Yield dicts for each parsable result row; header/divider lines are ignored by the regex."""
    for line in stream:
        m = ROW_RE.match(line.rstrip("\n"))
        if not m:
            continue
        yield {
            "block_shape": (int(m["bsq"]), int(m["bskv"])),
            "b": int(m["b"]),
            "h": int(m["h"]),
            "s_q": int(m["sq"]),
            "s_kv": int(m["skv"]),
            "d": int(m["d"]),
            "sparsity": float(m["sp"]),
            "ref": _to_float(m["ref"]),
            "ours": _to_float(m["ours"]),
        }


def _fmt_s(s: int) -> str:
    """Sequence-length label: full integer, no thousands separator (118806 -> '118806')."""
    return str(s)


def _compute_common_bnsd(groups):
    """Return a dict mapping each of B/H/S/D to its constant value across every line, or None if it varies."""
    bs_vals = {k[0] for k in groups}
    hs_vals = {k[1] for k in groups}
    sqs_vals = {k[2] for k in groups}
    ds_vals = {k[3] for k in groups}
    return {
        "B": next(iter(bs_vals)) if len(bs_vals) == 1 else None,
        "H": next(iter(hs_vals)) if len(hs_vals) == 1 else None,
        "S": next(iter(sqs_vals)) if len(sqs_vals) == 1 else None,
        "D": next(iter(ds_vals)) if len(ds_vals) == 1 else None,
    }


def _compute_line_xy(grows):
    """Build (xs, ys, skip_reason) for one (B,H,S,D,block_shape) group; skip_reason is None on success."""
    ref_baseline = next((r["ref"] for r in grows if r["ref"] is not None), None)
    if ref_baseline is None:
        return [], [], "no Ref_Latency baseline (run sparsity=0 row)"

    grows = sorted(grows, key=lambda r: r["sparsity"])
    xs, ys = [], []
    for r in grows:
        if r["ours"] is None or r["ours"] == 0.0:
            continue
        xs.append(r["sparsity"])
        ys.append(ref_baseline / r["ours"])
    if not xs:
        return [], [], "no Our_Latency cells"
    return xs, ys, None


def _line_label(key, common):
    """Per-line legend label: just block_shape when BNSD is shared across all lines; otherwise prepend it."""
    b, h, s_q, d, (bsq, bskv) = key
    if all(v is not None for v in common.values()):
        return f"block_shape=({bsq},{bskv})"
    return f"BNSD=({b},{h},{_fmt_s(s_q)},{d}), block_shape=({bsq},{bskv})"


# Colour each line by BLOCK_SIZE_KV (the dominant lever for speedup at high
# sparsity per the README sweep); linestyle separates the four BLOCK_SIZE_Q
# values; markers separate sequence lengths (typically a single S in one run,
# so all markers look identical).
_kv_palette = {
    128: "#d62728",   # tab:red
    256: "#ff7f0e",   # tab:orange
    512: "#2ca02c",   # tab:green
    1024: "#1f77b4",   # tab:blue
}
_linestyle_by_q = {128: "-", 256: "--", 512: "-.", 1024: ":"}
_seq_markers = ("o", "s", "^", "D", "v", "P", "X", "*")


def _sort_key(key):
    """Legend ordering: group by KV first, then Q, then varying B/H/S/D."""
    b, h, s_q, d, (bsq, bskv) = key
    return _SortKey(bskv=bskv, bsq=bsq, s_q=s_q, h=h, b=b, d=d)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Plot blitz_sparse_attention speedup vs sparsity "
                    "from benchmark.py stdout.",
    )
    parser.add_argument("--output", "-o", default="benchmark.png",
                        help="output PNG path (default: benchmark.png)")
    parser.add_argument("--title", default="blitz_sparse_attention speedup vs sparsity")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def _load_groups():
    """Read benchmark rows from stdin and group by (B,H,S,D,block_shape).

    Returns (rows, groups) on success, or (None, None) when stdin held no
    parsable rows. The caller — main, the script's entry point — is
    responsible for terminating in that case; per G.ERR.11 we keep
    sys.exit() out of this helper.
    """
    rows = list(parse_rows(sys.stdin))
    if not rows:
        return None, None
    groups = defaultdict(list)
    for r in rows:
        groups[(r["b"], r["h"], r["s_q"], r["d"], r["block_shape"])].append(r)
    return rows, groups


def _build_title(base_title, common):
    """Append a `BNSD=(...)` suffix to the title when all 4 fields are shared across lines."""
    if not all(v is not None for v in common.values()):
        return base_title
    suffix = f"BNSD=({common['B']},{common['H']},{_fmt_s(common['S'])},{common['D']})"
    return f"{base_title}\n{suffix}"


def _plot_lines(ax, groups, common):
    """Draw one speedup line per group; return (lines_drawn, [(s_q, bs, reason), ...] skipped)."""
    seq_lens = sorted({k[2] for k in groups})
    marker_by_seq = dict(zip(seq_lens, _seq_markers))

    lines_drawn = 0
    skipped = []
    for key in sorted(groups, key=_sort_key):
        _, _, s_q, _, bs = key
        bsq, bskv = bs
        xs, ys, skip_reason = _compute_line_xy(groups[key])
        if skip_reason is not None:
            skipped.append((s_q, bs, skip_reason))
            continue
        ax.plot(
            xs, ys,
            color=_kv_palette.get(bskv, "#666666"),
            linestyle=_linestyle_by_q.get(bsq, "-"),
            marker=marker_by_seq.get(s_q, "o"),
            markersize=5, linewidth=1.5,
            label=_line_label(key, common),
        )
        lines_drawn += 1
    return lines_drawn, skipped


def _finalize_figure(fig, ax, title):
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("sparsity (fraction of blocks zeroed)", fontsize=11)
    ax.set_ylabel("speedup  =  Ref_Latency(sparsity=0)  /  Our_Latency", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    # One column keeps the colour-grouped ordering intact (matplotlib's
    # 2-column flow interleaves by row); legend can run tall vertically.
    ax.legend(loc="upper left", fontsize=10, ncol=1, framealpha=0.92)
    fig.tight_layout()


def _report(output_path, lines_drawn, rows_total, skipped):
    logger.info(
        "plot.py: wrote %s  (%d lines, %d rows parsed)",
        output_path, lines_drawn, rows_total,
    )
    for s_q, bs, why in skipped:
        logger.info("plot.py: skipped S=%s %dx%d  (%s)", s_q, bs[0], bs[1], why)


def main():
    args = _parse_args()
    rows, groups = _load_groups()
    if rows is None:
        sys.exit("plot.py: no benchmark rows parsed from stdin "
                 "(pipe `benchmark.py` stdout into this script).")
    common = _compute_common_bnsd(groups)

    # matplotlib is imported lazily so `--help` works on hosts without it.
    import matplotlib
    matplotlib.use("Agg")  # headless backend — no DISPLAY on NPU hosts.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    lines_drawn, skipped = _plot_lines(ax, groups, common)
    _finalize_figure(fig, ax, _build_title(args.title, common))
    fig.savefig(args.output, dpi=args.dpi)
    _report(args.output, lines_drawn, len(rows), skipped)


if __name__ == "__main__":
    main()
