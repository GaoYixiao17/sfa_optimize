#!/bin/bash
# ============================================================================
# install_pkg.sh — 安装 baseline / optimized 算子包并管理切换
#
# [安装] (一次性, 需先 build_ops.sh 产出 run 包):
#   bash scripts/install_pkg.sh
#     安装到独立目录: $AB_ROOT/<impl>/install/ascend-toolkit/latest/opp/vendors/<vendor>
#
# [切换] (每次对比前在新终端里 source):
#   source scripts/install_pkg.sh builtin      # CANN 内置算子 (通常=baseline)
#   source scripts/install_pkg.sh baseline     # 重编译的原始版本
#   source scripts/install_pkg.sh optimized    # 优化版本
#   source scripts/install_pkg.sh status       # 查看当前激活状态
#
# 机制说明:
#   CANN 会扫描 $ASCEND_HOME_PATH/opp/vendors/* 下的自定义算子包,
#   同名自定义算子优先于内置算子 (aclnn 实现由 op_api 注册表路由).
#   本脚本把所选实现的 vendors/<vendor> 目录以软链接方式挂入 CANN vendors,
#   并把其 op_api/lib 加入 LD_LIBRARY_PATH. 切换 = 换软链接, 立即对新进程生效.
#
# 验证是否生效:
#   python3 -c "import torch,torch_npu; ..."  # 见 benchmarks/ 下用例
#   1) ab_correctness.py 两轮 save 的输出应一致 (实现等价性)
#   2) benchmark 计时应有差异 (优化生效); 若无差异请检查本脚本 status 输出
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AB_ROOT="${AB_ROOT:-$REPO_ROOT/build_out}"
AB_LINK_NAME="op_optimize_ab"

# ---------------------------------------------------------------------------
find_cann() {
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        echo "$ASCEND_HOME_PATH"; return 0
    fi
    for p in /usr/local/Ascend/ascend-toolkit/latest ~/Ascend/ascend-toolkit/latest; do
        if [ -f "$p/bin/set_env.sh" ]; then
            echo "$p"; return 0
        fi
    done
    return 1
}

# ---------------------------------------------------------------------------
do_install() {
    for impl in baseline optimized; do
        pkgdir="$AB_ROOT/$impl/build_out/run_pkgs"
        inst="$AB_ROOT/$impl/install"
        if [ ! -d "$pkgdir" ] || [ -z "$(ls -A "$pkgdir" 2>/dev/null)" ]; then
            echo "[warn] $impl 无 run 包 ($pkgdir), 跳过 (先运行 build_ops.sh)"
            continue
        fi
        rm -rf "$inst"; mkdir -p "$inst"
        for run in "$pkgdir"/*.run; do
            echo "[info] install $impl: $(basename "$run") -> $inst"
            bash "$run" --quiet --install-path="$inst"
        done
        vendors=$(find "$inst" -type d -name vendors 2>/dev/null | head -1)
        if [ -n "$vendors" ]; then
            echo "[ok] $impl vendors: $vendors"
            ls "$vendors" | sed 's/^/       vendor: /'
        else
            echo "[warn] $impl 未找到 vendors 目录, 请检查安装日志"
        fi
    done
}

# ---------------------------------------------------------------------------
do_activate() {
    local which="$1"
    local cann
    if ! cann=$(find_cann); then
        echo "[error] 找不到 CANN (ASCEND_HOME_PATH 未设置且无默认安装)"; return 1
    fi
    local cann_opp="$cann/opp"
    local link="$cann_opp/vendors/$AB_LINK_NAME"

    # 清除旧链接 (幂等)
    if [ -L "$link" ] || [ -e "$link" ]; then
        rm -rf "$link"
    fi

    case "$which" in
        builtin)
            echo "[activate] -> builtin (CANN 内置算子, 已移除自定义包链接 $link)"
            ;;
        baseline | optimized)
            local inst="$AB_ROOT/$which/install"
            local vendor_dir
            vendor_dir=$(find "$inst" -mindepth 1 -maxdepth 6 -type d -path "*opp/vendors/*" 2>/dev/null | head -1)
            if [ -z "$vendor_dir" ]; then
                echo "[error] $which 未安装, 先运行: bash scripts/install_pkg.sh"; return 1
            fi
            mkdir -p "$cann_opp/vendors"
            ln -sfn "$vendor_dir" "$link"
            # op_api 动态库 (aclnn 路由) 加入搜索路径
            local api_lib
            api_lib=$(dirname "$(find "$vendor_dir" -name "*.so" -path "*op_api*" 2>/dev/null | head -1 || true)")
            if [ -n "$api_lib" ] && [ "$api_lib" != "." ]; then
                export LD_LIBRARY_PATH="$api_lib:${LD_LIBRARY_PATH:-}"
            fi
            echo "[activate] -> $which"
            echo "    link : $link -> $vendor_dir"
            echo "    api  : ${api_lib:-<none>}"
            ;;
        status)
            echo "ASCEND_HOME_PATH=$cann"
            echo "vendors:"; ls -la "$cann_opp/vendors/" 2>/dev/null || echo "  (无)"
            if [ -L "$link" ]; then
                echo "当前激活: $(readlink "$link")"
            else
                echo "当前激活: builtin (无 $AB_LINK_NAME 链接)"
            fi
            ;;
        *)
            echo "usage: source scripts/install_pkg.sh {builtin|baseline|optimized|status}"; return 1 ;;
    esac
    echo "[note] 请在**新进程**中运行 benchmark (激活只对新进程生效)"
}

case "${1:-install}" in
    install) do_install ;;
    builtin | baseline | optimized | status) do_activate "$1" ;;
    *)
        echo "usage:"
        echo "  bash   scripts/install_pkg.sh install"
        echo "  source scripts/install_pkg.sh {builtin|baseline|optimized|status}"
        exit 1 ;;
esac
