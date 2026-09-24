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
#   bash scripts/build_ops.sh --impl baseline
#   bash scripts/build_ops.sh --impl optimized
#
# 可选参数:
#   --framework <dir>   构建框架源码树 (默认: 仓内 ops-transformer/, 缺失时
#                       依次找 $OPS_TRANSFORMER_ROOT / ~/ops-transformer)
#   --impl baseline|optimized|both   编译哪套源码 (默认 both)
#   --soc <soc>         ASCEND_COMPUTE_UNIT (默认 ascend950; A5=Ascend 950PR)
#   --jobs <n>          并行度
#
# 公开版框架行为: 只编三个目标算子 (--ops 过滤); 第三方依赖缓存于
# build_out/third_party_cache/ (baseline/optimized 共享, 重跑不重复下载编译)
#
# 环境变量:
#   AB_INSECURE_TLS=1   关闭 cmake 侧第三方包下载的 SSL 证书校验
#                       (网络对 obs.myhuaweicloud.com 拦截/缺 CA 时使用)
#
# 产物: $AB_ROOT/<impl>/build_out/run_pkgs/*.run
#       公开版框架: cann-ops-transformer-*.run; 内网版: CANN-custom_ops-*.run
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

if [ ! -f "$FRAMEWORK/build.sh" ] || [ ! -d "$FRAMEWORK/cmake" ]; then
    echo "[error] 构建框架不完整: $FRAMEWORK (需要 build.sh 与 cmake/)"
    echo "        本仓库自带 ops-transformer/ (gitcode 官方镜像); 或用 --framework 指定其他框架树"
    exit 1
fi

# 识别 build.sh 风格:
#   public   — gitcode 公开版: 用 --ops=.../--soc=.../--pkg 参数, 无 CMakePresets.json
#   internal — 内网版: 用 -n <op> 逐算子, 依赖 CMakePresets.json 传 SOC/CANN
if grep -q -- '--ops=\*)' "$FRAMEWORK/build.sh" 2>/dev/null; then
    BUILD_STYLE="public"
elif [ -f "$FRAMEWORK/CMakePresets.json" ]; then
    BUILD_STYLE="internal"
else
    echo "[error] 无法识别 build.sh 风格: $FRAMEWORK"
    echo "        (既不支持 --ops= 参数, 也没有 CMakePresets.json)"
    exit 1
fi
echo "[info] build.sh 风格: $BUILD_STYLE"

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

    # 可选: 关闭 cmake 侧 HTTPS 下载的证书校验 (makeself/json/eigen 等 OBS 第三方包)
    # A5 网络对 cann-3rd.obs.cn-north-4.myhuaweicloud.com 证书校验失败时使用
    if [ "${AB_INSECURE_TLS:-0}" = "1" ] && ! grep -q "CMAKE_TLS_VERIFY" "$work/CMakeLists.txt"; then
        sed -i '/cmake_minimum_required/a set(CMAKE_TLS_VERIFY OFF CACHE BOOL "" FORCE)' "$work/CMakeLists.txt"
        echo "[info] AB_INSECURE_TLS=1: 已注入 CMAKE_TLS_VERIFY=OFF (cmake file(DOWNLOAD)/第三方包下载不再校验证书)"
    fi

    # 配置 SOC 与 CANN 路径 (内网版: python 改写 CMakePresets.json; 公开版: 由 build.sh 参数传递)
    if [ "$BUILD_STYLE" = "internal" ]; then
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
    else
        echo "[info] 公开版框架: SOC/CANN 由 --soc 参数与环境变量传递"
    fi

    # 编译: 按风格调用
    # - 公开版: --ops 只编三个算子; 失败直接中止 (不整仓回退——那会编全部算子, 耗时数小时且无意义)
    # - 第三方依赖缓存指向持久目录: baseline/optimized 共享, 重建框架副本时不重复下载/编译
    mkdir -p "$out" "$AB_ROOT/third_party_cache"
    OPS_CSV="lightning_indexer,lightning_indexer_v2,sparse_flash_attention"
    (
        cd "$work"
        build_ok=0
        if [ "$BUILD_STYLE" = "public" ]; then
            # 公开版: --ops 选算子, --soc 指定芯片, --pkg 产出 run 包
            if bash build.sh --pkg --soc="$SOC" --ops="$OPS_CSV" \
                    --cann_3rd_lib_path="$AB_ROOT/third_party_cache" \
                    -j"$JOBS" -O3 2>&1 | tee "$out/build.log"; then
                build_ok=1
            fi
        else
            if bash build.sh -n lightning_indexer -n lightning_indexer_v2 -n sparse_flash_attention 2>&1 | tee "$out/build.log"; then
                build_ok=1
            fi
        fi
        if [ "$build_ok" != "1" ]; then
            if [ "$BUILD_STYLE" = "public" ]; then
                echo "[error] 构建失败, 已中止 (公开版不做整仓回退; 整仓=全量算子构建, 耗时极长)"
                echo "-------- $out/build.log 末尾 50 行 --------"
                tail -n 50 "$out/build.log" || true
                echo "----------------------------------------"
                exit 1
            else
                echo "[warn] 逐算子构建失败, 尝试整仓构建 (日志: $out/build_full.log)"
                bash build.sh 2>&1 | tee "$out/build_full.log"
            fi
        fi
    )

    # 收集产物 run 包
    rm -rf "$out/run_pkgs"; mkdir -p "$out/run_pkgs"
    found=0
    for f in $(find "$work" -name "CANN-custom_ops*.run" -o -name "custom_ops*.run" -o -name "cann-ops-transformer*.run" 2>/dev/null); do
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
