# 编译与安装 (BUILD)

> 目标机器: **A5 服务器** (Ascend 950PR, CANN 9.2.0-beta.2, Linux)。
> 本仓库 (Windows/开发机) 只改代码, 编译与运行全部在 A5 上进行。

## 0. 前置条件 (A5 服务器)

```bash
# CANN toolkit 已安装并 source
source /usr/local/Ascend/ascend-toolkit/latest/bin/set_env.sh   # 路径按实际安装位置
python3 -c "import sys; print(sys.version)"        # 需 3.7+
which cmake gcc g++                                # 构建工具链
```

需要两样东西:
1. **本仓库** (`op_optimize/`, 含优化后算子源码 + 脚本), 传到 A5 任意目录;
2. **ops-transformer 9.2.0 完整构建框架源码树** (根目录含 `build.sh`,
   `CMakePresets.json`, `cmake/`, 算子在 `attention/<op>`)。
   该框架树只从内部源码仓获取, 需使用者自行提供 (三个 zip 只是算子子目录, 不含构建框架)。

## 1. 生成 baseline / optimized 两套源码 overlay

```bash
cd op_optimize
bash scripts/prepare_sources.sh
```

- `build_input/baseline/attention/<op>/` ← 从根目录原始 zip 重新解出 (**未修改**);
- `build_input/optimized/attention/<op>/` ← 本仓库当前源码 (含全部优化)。

脚本最后会打印两套源码的差异文件清单, 可用于确认改动范围。

## 2. 编译两套算子包

```bash
bash scripts/build_ops.sh --framework ~/ops-transformer-9.2.0          # 两套都编
# 或分开编:
bash scripts/build_ops.sh --impl baseline  --framework ~/ops-transformer-9.2.0
bash scripts/build_ops.sh --impl optimized --framework ~/ops-transformer-9.2.0
```

可选参数: `--soc ascend950` (默认, A5=Ascend 950PR) / `--jobs <n>`。

脚本行为:
- 每套实现复制一份**干净的框架树**到 `build_out/<impl>/framework/`,
  再用对应 overlay 覆盖 `attention/{lightning_indexer,lightning_indexer_v2,sparse_flash_attention}`;
- 自动改写 `CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT` / `ASCEND_CANN_PACKAGE_PATH`;
- 优先逐算子构建 (`build.sh -n <op> ...`), 失败自动回退整仓构建;
- 产物 (`.run` 包) 汇总到 `build_out/<impl>/build_out/run_pkgs/`。

> **重要 — LI v1 跨目录依赖**: lightning_indexer 的 arch35 kernel 通过相对路径
> `../../../../lightning_indexer_v2/op_kernel/arch35/...` 引用 v2 源码,
> 两个算子目录必须作为兄弟目录同时存在于构建树 (build_ops.sh 已保证这一点)。

> 若逐算子构建报错, 先看 `build_out/<impl>/build_out/build.log`;
> `-Werror` 下常见的坑是未使用变量 (优化中已逐一排查)。

## 3. 安装并切换实现

```bash
bash scripts/install_pkg.sh                 # 一次性安装两套 run 包
source scripts/install_pkg.sh status        # 查看当前激活状态
source scripts/install_pkg.sh builtin       # 激活 CANN 内置算子 (= 通常意义上的 baseline)
source scripts/install_pkg.sh baseline      # 激活重编译的原始版本
source scripts/install_pkg.sh optimized     # 激活优化版本
```

机制: 把所选实现的 `vendors/<vendor>` 目录软链到
`$ASCEND_HOME_PATH/opp/vendors/op_optimize_ab`, 并把其 `op_api` 库加入
`LD_LIBRARY_PATH`。 同名自定义算子优先于内置算子, 切换 = 换软链, **对新进程生效**
(source 后请在新终端/新 python 进程里跑测试)。

## 4. 验证

```bash
# 4.1 正确性 A/B (详见 benchmarks/README.md)
cd benchmarks
# 激活 baseline 时:
python3 ab_correctness.py save --out ab_baseline.pt
source ../scripts/install_pkg.sh optimized     # 新终端里激活 optimized 后:
python3 ab_correctness.py save --out ab_optimized.pt
python3 ab_correctness.py compare --a ab_baseline.pt --b ab_optimized.pt

# 4.2 性能对比
LABEL=baseline  ./run_all.sh     # builtin/baseline 激活状态
LABEL=optimized ./run_all.sh     # optimized 激活状态
python3 compare_report.py -o report.md
```

## 5. 常见问题

| 现象 | 排查 |
|---|---|
| 计时无差异 | `install_pkg.sh status` 确认软链; 确认 benchmark 是 source 之后的新进程; `ldd` 确认 op_api so 来源 |
| 找不到构建框架 | `--framework` 必须指向含 `build.sh` 的 ops-transformer 树根 |
| lightning_indexer 编译报缺头文件 | 检查构建树中 `attention/lightning_indexer_v2` 是否存在 (见第 2 步说明) |
| run 包安装失败 | 单独运行 `bash build_out/<impl>/build_out/run_pkgs/*.run --install-path=...` 看详细日志 |
