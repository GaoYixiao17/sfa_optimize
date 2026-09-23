#!/bin/bash
# ============================================================================
# prepare_sources.sh — 生成 baseline / optimized 两套算子源码 overlay
#
# 用法 (在本仓库根目录):
#   bash scripts/prepare_sources.sh
#
# 产物:
#   build_input/baseline/attention/<op>/    — 从原始 zip 解出的未修改源码
#   build_input/optimized/attention/<op>/   — 本仓库中已优化的源码
#
# 说明: <op> ∈ {lightning_indexer, lightning_indexer_v2, sparse_flash_attention}
# 两个 overlay 均保持 ops-transformer 仓库的 attention/<op> 目录结构,
# 供 build_ops.sh 覆盖到构建框架树中编译。
# ============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_BASE="$REPO_ROOT/build_input"
mkdir -p "$OUT_BASE/baseline/attention" "$OUT_BASE/optimized/attention"

OPS=(lightning_indexer lightning_indexer_v2 sparse_flash_attention)

for op in "${OPS[@]}"; do
    # ---------- optimized: 直接取本仓库 (可能已打补丁) 的源码 ----------
    src_opt="$REPO_ROOT/ops-transformer-9.2.0-attention-${op}/ops-transformer-9.2.0-attention-${op}/attention/${op}"
    if [ -d "$src_opt" ]; then
        rm -rf "$OUT_BASE/optimized/attention/${op}"
        mkdir -p "$OUT_BASE/optimized/attention"
        cp -r "$src_opt" "$OUT_BASE/optimized/attention/${op}"
        echo "[ok] optimized/${op}  <- $src_opt"
    else
        echo "[warn] missing $src_opt"
    fi

    # ---------- baseline: 从原始 zip 重新解压 (保证未修改) ----------
    zip="$REPO_ROOT/ops-transformer-9.2.0-attention-${op}.zip"
    tmp="$OUT_BASE/.tmp_${op}"
    rm -rf "$tmp"; mkdir -p "$tmp"
    if [ -f "$zip" ]; then
        (cd "$tmp" && unzip -q "$zip")
        inner=$(find "$tmp" -type d -name "${op}" -path "*attention/${op}" | head -1)
        if [ -n "$inner" ]; then
            rm -rf "$OUT_BASE/baseline/attention/${op}"
            cp -r "$inner" "$OUT_BASE/baseline/attention/${op}"
            echo "[ok] baseline/${op}  <- $zip"
        else
            echo "[warn] cannot locate attention/${op} inside $zip"
        fi
        rm -rf "$tmp"
    else
        echo "[warn] missing $zip; baseline 将退化为使用当前源码树 (若未修改则等价)"
    fi
done

# 差异摘要, 方便确认 optimized 相对 baseline 的改动范围
echo ""
echo "=== 源码差异 (baseline vs optimized) ==="
for op in "${OPS[@]}"; do
    if [ -d "$OUT_BASE/baseline/attention/${op}" ] && [ -d "$OUT_BASE/optimized/attention/${op}" ]; then
        n=$(diff -rq "$OUT_BASE/baseline/attention/${op}" "$OUT_BASE/optimized/attention/${op}" 2>/dev/null | wc -l)
        echo "  ${op}: ${n} 个文件/条目存在差异"
        diff -rq "$OUT_BASE/baseline/attention/${op}" "$OUT_BASE/optimized/attention/${op}" 2>/dev/null | head -20 || true
    fi
done
echo ""
echo "prepare_sources 完成 -> $OUT_BASE"
