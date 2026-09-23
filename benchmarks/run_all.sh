#!/bin/bash
# ============================================================================
# run_all.sh — A5 算子对比测试一键脚本
#
# 流程:
#   1. 确认当前激活的算子实现 (builtin / baseline / optimized),
#      由 scripts/install_pkg.sh 控制 (需 source, 见 BUILD.md)
#   2. 依次运行 bench_li / bench_sfa / bench_e2e, 保存 result_<op>_<label>.json
#   3. 两个实现各跑一轮后, compare_report.py 生成 report.md
#
# 典型用法:
#   # 第一轮: 激活 baseline (未安装优化包 = CANN 内置算子)
#   source scripts/install_pkg.sh builtin
#   LABEL=baseline ./run_all.sh
#   # 第二轮: 激活优化包
#   source scripts/install_pkg.sh optimized
#   LABEL=optimized ./run_all.sh
#   # 生成对比报告
#   python3 compare_report.py -o report.md
#
# 可用环境变量见 bench_common.py (BENCH_BS / BENCH_SEQ / BENCH_ITERS / ...)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

LABEL="${LABEL:-default}"
BENCH_BS="${BENCH_BS:-32,64}"
export BENCH_BS
export BENCH_LABEL="$LABEL"
OUT_DIR="${OUT_DIR:-.}"

echo "============================================================"
echo " run_all: label=$LABEL  BENCH_BS=$BENCH_BS  BENCH_SEQ=${BENCH_SEQ:-128}"
echo " device : $(python3 -c "import torch,torch_npu;print(torch.npu.get_device_properties(0))" 2>/dev/null || echo 'N/A')"
echo "============================================================"

run_one() {
    local script="$1" tag="$2"
    echo ""
    echo ">>> [$LABEL] $tag"
    BENCH_OUT_JSON="$OUT_DIR/result_${tag}_${LABEL}.json" \
        python3 "$script" || echo "[warn] $tag failed"
}

run_one bench_li.py li
run_one bench_sfa.py sfa
run_one bench_e2e.py e2e

echo ""
echo "============================================================"
echo " done: label=$LABEL"
echo " results: $OUT_DIR/result_*_${LABEL}.json"
echo "============================================================"
