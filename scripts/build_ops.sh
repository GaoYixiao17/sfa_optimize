#!/bin/bash
# ============================================================================
# build_ops.sh — 在 A5 服务器上编译算子包 (baseline / optimized)
#
# 前置条件:
#   1. 已安装 CANN toolkit (Ascend-cann-toolkit 9.x) 且 source set_env.sh
#   2. 构建框架树: 本仓库已自带 ops-transformer/ (gitcode 官方 9.2.0 镜像),
#      默认直接使用; 也可用 --framework 指定其他版本树 (根目录含 build.sh/CMakePresets.json/cmake/)
#   3. 已运行 scripts/prepare_sources.sh 生成 build_input/{baseline,optimized}
#
# 用法:
#   bash scripts/prepare_sources.sh
#   bash scripts/build_ops.sh --impl baseline  --framework ~/ops-transformer-9.2.0
#   bash scripts/build_ops.sh --impl optimized --framework ~/ops-transformer-9.2.0
#
# 可选参数:
#   --framework <dir>   构建框架源码树 (默认: 仓内 ops-transformer/, 缺失时
#                       依次找 $OPS_TRANSFORMER_ROOT / ~/ops-transformer)
#   --impl baseline|optimized|both   编译哪套源码 (默认 both)
#   --soc <soc>         ASCEND_COMPUTE_UNIT (默认 ascend950; A5=Ascend 950PR)
#   --jobs <n>          并行度
#
# 产物: $AB_ROOT/<impl>/build_out/CANN-custom_ops-*.run
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AB_ROOT="${AB_ROOT:-$REPO_ROOT/build_out}"
SOC="${SOC:-ascend950}"
IMPL="both"
# 框架树优先级: OPS_TRANSFORMER_ROOT > 仓内自带 ops-transformer/ > ~/ops-transformer
if [ -f "$REPO_ROOT/ops-transformer/build.sh" ]; then
    FRAMEWORK="${OPS_TRANSFORMER_ROOT:-$REPO_ROOT/ops-transformer}"
else
    FRAMEWORK="${OPS_TRANSFORMER_ROOT:-$HOME/ops-transformer}"
fi
JOBS="$(nproc 2>/dev/null || echo 16)"

while [ $# -gt 0 ]; do
    case "$1" in
        --framework) FRAMEWORK="$2"; shift 2 ;;
        --impl) IMPL="$2"; shift 2 ;;
        --soc) SOC="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

# ---------- 环境检查 ----------
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
    CANN_SET_ENV="$(ls /usr/local/Ascend/ascend-toolkit/latest/bin/set_env.sh 2>/dev/null || \
                    ls ~/Ascend/ascend-toolkit/latest/bin/set_env.sh 2>/dev/null || true)"
    if [ -n "$CANN_SET_ENV" ]; then
        echo "[info] source $CANN_SET_ENV"
        # shellcheck disable=SC1090
        source "$CANN_SET_ENV"
    else
        echo "[error] 未找到 CANN set_env.sh, 请先安装 CANN toolkit 并 source"; exit 1
    fi
fi
echo "[info] ASCEND_HOME_PATH=$ASCEND_HOME_PATH"

if [ ! -f "$FRAMEWORK/build.sh" ] || [ ! -f "$FRAMEWORK/CMakePresets.json" ]; then
    echo "[error] 构建框架不完整: $FRAMEWORK (需要 build.sh 与 CMakePresets.json)"
    echo "        请提供 ops-transformer 9.2.0 源码树, 或 cann-ops-adv 兼容框架,"
    echo "        并用 --framework 指定路径"
    exit 1
fi

if [ ! -d "$REPO_ROOT/build_input" ]; then
    echo "[info] 先运行 prepare_sources.sh"
    bash "$REPO_ROOT/scripts/prepare_sources.sh"
fi

# ---------- 逐套编译 ----------
build_one() {
    local impl="$1"
    local overlay="$REPO_ROOT/build_input/$impl/attention"
    local work="$AB_ROOT/$impl/framework"
    local out="$AB_ROOT/$impl/build_out"

    if [ ! -d "$overlay" ] || [ -z "$(ls -A "$overlay" 2>/dev/null)" ]; then
        echo "[error] overlay 不存在: $overlay"; return 1
    fi

    echo ""
    echo "========================================================"
    echo " build: impl=$impl  soc=$SOC  framework=$FRAMEWORK"
    echo "========================================================"

    # 每次全新复制框架树, 保证两套实现互不污染
    rm -rf "$work"
    mkdir -p "$(dirname "$work")"
    echo "[info] 复制构建框架树..."
    cp -r "$FRAMEWORK" "$work"
    rm -rf "$work/build_out" || true

    # 覆盖三个算子目录
    for op in lightning_indexer lightning_indexer_v2 sparse_flash_attention; do
        if [ -d "$overlay/$op" ]; then
            mkdir -p "$work/attention"
            rm -rf "$work/attention/$op"
            cp -r "$overlay/$op" "$work/attention/$op"
            echo "[info] overlay: attention/$op  ($impl)"
        fi
    done

    # 配置 SOC 与 CANN 路径 (python 修改 CMakePresets.json, 避免手工编辑)
    python3 - "$work/CMakePresets.json" "$SOC" "$ASCEND_HOME_PATH" <<'PYEOF'
import json, sys
path, soc, cann = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    data = json.load(f)
changed = False
for ver in data.get("configurePresets", []):
    cv = ver.get("cacheVariables", {})
    if "ASCEND_COMPUTE_UNIT" in cv:
        cv["ASCEND_COMPUTE_UNIT"] = soc; changed = True
    if "ASCEND_CANN_PACKAGE_PATH" in cv:
        cv["ASCEND_CANN_PACKAGE_PATH"] = cann
for ver in data.get("buildPresets", []):
    cv = ver.get("cacheVariables", {})
    if "ASCEND_COMPUTE_UNIT" in cv:
        cv["ASCEND_COMPUTE_UNIT"] = soc
    if "ASCEND_CANN_PACKAGE_PATH" in cv:
        cv["ASCEND_CANN_PACKAGE_PATH"] = cann
with open(path, "w") as f:
    json.dump(data, f, indent=2)
print(f"[info] CMakePresets: ASCEND_COMPUTE_UNIT={soc}  CANN={cann}  changed={changed}")
PYEOF

    # 编译: 优先逐算子构建 (-n), 失败则回退整仓构建
    mkdir -p "$out"
    (
        cd "$work"
        if bash build.sh -n lightning_indexer -n lightning_indexer_v2 -n sparse_flash_attention 2>&1 | tee "$out/build.log"; then
            :
        else
            echo "[warn] 逐算子构建失败, 尝试整仓构建 (日志: $out/build_full.log)"
            bash build.sh 2>&1 | tee "$out/build_full.log"
        fi
    )

    # 收集产物 run 包
    rm -rf "$out/run_pkgs"; mkdir -p "$out/run_pkgs"
    found=0
    for f in $(find "$work" -name "CANN-custom_ops*.run" -o -name "custom_ops*.run" 2>/dev/null); do
        cp "$f" "$out/run_pkgs/"; found=1
        echo "[ok] 产物: $out/run_pkgs/$(basename "$f")"
    done
    if [ "$found" = "0" ]; then
        echo "[error] 未找到 run 包产物, 请检查 $out/build.log"
        return 1
    fi
}

case "$IMPL" in
    baseline)  build_one baseline ;;
    optimized) build_one optimized ;;
    both)      build_one baseline && build_one optimized ;;
    *) echo "unknown --impl $IMPL"; exit 1 ;;
esac

echo ""
echo "build_ops 完成. run 包位于 $AB_ROOT/<impl>/build_out/run_pkgs/"
echo "下一步: bash scripts/install_pkg.sh  (安装并切换)"
