#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_report.py — 汇总 baseline / optimized 两轮 benchmark JSON, 生成 Markdown 对比报告

用法:
    python compare_report.py result_li_baseline.json result_li_optimized.json \
            result_sfa_baseline.json result_sfa_optimized.json -o report.md
    # 或自动发现当前目录所有 result_*.json:
    python compare_report.py -o report.md
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="*", help="result_*.json 文件, 或留空自动发现")
    p.add_argument("-o", "--out", default="report.md")
    args = p.parse_args()

    files = args.inputs
    if not files:
        files = sorted(glob.glob("result_*.json"))
    if len(files) < 2:
        print("至少需要两个 result json (baseline 与 optimized)")
        raise SystemExit(1)

    # (op, case) -> {label: stats}
    table = defaultdict(dict)
    meta = {}
    for f in files:
        data = load(f)
        label = data.get("meta", {}).get("label", os.path.basename(f))
        meta[label] = data.get("meta", {})
        for rec in data["records"]:
            key = (rec["op"], rec["case"])
            table[key][label] = rec

    labels = list(meta.keys())
    base_label, opt_label = labels[0], labels[-1]

    lines = []
    lines.append("# A5 算子优化对比报告 (bs=32/64, seq=128)\n")
    lines.append(f"- baseline: **{base_label}**")
    lines.append(f"- optimized: **{opt_label}**\n")
    lines.append(f"- 设备: {meta[base_label].get('device', {})}\n")

    lines.append("## 汇总\n")
    lines.append("| 算子 | 用例 | baseline (us) | optimized (us) | 加速比 |")
    lines.append("|---|---|---|---|---|")
    sum_base, sum_opt = 0.0, 0.0
    for (op, case), recs in sorted(table.items()):
        rb = recs.get(base_label)
        ro = recs.get(opt_label)
        if not rb or not ro:
            continue
        b, o = rb["mean_us"], ro["mean_us"]
        speedup = b / o if o > 0 else float("inf")
        sum_base += b
        sum_opt += o
        lines.append(f"| {op} | `{case}` | {b:.1f} | {o:.1f} | **{speedup:.2f}x** |")
    if sum_opt > 0:
        lines.append(f"| **合计(mean)** | | {sum_base:.1f} | {sum_opt:.1f} | **{sum_base/sum_opt:.2f}x** |")
    lines.append("")

    lines.append("## 明细 (min / median / max, us)\n")
    lines.append("| 算子 | 用例 | min | median | max |")
    lines.append("|---|---|---|---|---|")
    for (op, case), recs in sorted(table.items()):
        ro = recs.get(opt_label)
        if not ro:
            continue
        lines.append(
            f"| {op} | `{case}` | {ro['min_us']:.1f} | {ro['median_us']:.1f} | {ro['max_us']:.1f} |"
        )
    lines.append("")

    report = "\n".join(lines)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n[report] saved -> {args.out}")


if __name__ == "__main__":
    main()
