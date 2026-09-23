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
2. 构建框架源码树 — **本仓库已自带** `ops-transformer/` (gitcode 官方 9.2.0 镜像,
   commit 3e20cfe0), `build_ops.sh` 默认直接使用, 无需另行获取。
   如需其他版本 (如 9.1.0 配套): `git clone -b 9.1.0 https://gitcode.com/cann/ops-transformer.git`
   后用 `--framework` 指定。

### 什么是"完整构建框架树"? 为什么必须要?

本仓库的三个 `ops-transformer-9.2.0-attention-*/` 目录只是**算子源码目录**
(`attention/<op>` 的内容), 本身无法独立编译:

- 每个算子的 `CMakeLists.txt` 引用的是**仓库级** CMake 函数/变量, 不在这套目录里;
- AscendC kernel 的编译 (ccec)、host 代码编译、打包成 `.run` (vendors 目录结构)
  全部由仓库顶层的构建系统驱动。

`ops-transformer` 是一个大型单体源码仓, 顶层结构:

```
ops-transformer-9.2.0/          ← "完整构建框架树", A5 上需自行提供
├── build.sh                    ← 构建入口
├── CMakePresets.json           ← SOC 版本 / CANN 路径预设
├── cmake/                      ← 全部构建模块
├── scripts/ ...                ← 打包/公共设施
└── attention/
    ├── lightning_indexer/      ← 算子源码住在这里
    └── ...
```

**怎么确认手上的是不是**: 目录根部同时有 `build.sh` + `CMakePresets.json` + `cmake/`,
且存在 `attention/` 目录。**版本优先 9.2.0** (与算子源码同版; 机器 CANN 为 9.1.0 时见第 5 节 FAQ)。
获取渠道是内部源码仓 (Gitee 上的 ops-transformer 为内网仓, 外部访问 404);
cann-ops-adv 兼容框架亦可。

`build_ops.sh --framework` 做的事: 复制整棵框架树到 `build_out/<impl>/framework`,
再用本仓库的算子目录 (baseline=zip 原始版 / optimized=优化版) **覆盖**
`attention/<op>` 后调用框架的 `build.sh` 编译。没有框架树则 baseline/optimized
均无法编译 — `install_pkg.sh builtin` 只是对照 CANN 预装算子, 优化代码必须
经框架编译成 `.run` 包才能生效。

### 在 A5 上如何找到/指定框架树

**本仓库已自带 `ops-transformer/` (gitcode 官方 9.2.0 镜像), `build_ops.sh` 默认优先使用它,
通常无需 `--framework` 参数。** 以下供需要其他版本 (如 9.1.0 配套) 或独立获取时参考:

```bash
# 官方渠道 (国内直连, 无需代理): 按机器 CANN 版本选分支
git clone -b 9.2.0 https://gitcode.com/cann/ops-transformer.git
# 机器上搜是否已有:
ls -d ~/ops-transformer* ~/code/* /data/*/ops-transformer* 2>/dev/null
find ~ /data /work /home -maxdepth 4 -name CMakePresets.json 2>/dev/null

# 从内网仓拉 (分支/tag 选与机器 CANN 匹配的版本):
git clone -b <版本分支> <内网ops-transformer地址> ~/ops-transformer

# 验证: 根部须同时有 build.sh + CMakePresets.json + cmake/ + attention/
ls ~/ops-transformer; ls ~/ops-transformer/attention/ | head

# 指定 (三选一):
bash scripts/build_ops.sh --framework ~/ops-transformer             # ① 显式参数
OPS_TRANSFORMER_ROOT=~/ops-transformer bash scripts/build_ops.sh   # ② 环境变量
# ③ 放默认路径 ~/ops-transformer 即免参数
```

注意:
- `/usr/local/Ascend/...` 是**已安装的 CANN 二进制**, 不是框架源码树, 不能当 `--framework` 用;
- build_ops.sh 只**复制**框架树到 `build_out/` 后在副本上编译, 原树 (可为只读/NFS 共享) 不被修改;
- 机器 CANN 为 9.1.0 时优先取 9.1.0 配套版本的框架树 (见第 5 节 FAQ)。

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
bash scripts/build_ops.sh                       # 两套都编 (框架默认取仓内 ops-transformer/)
# 或分开编:
bash scripts/build_ops.sh --impl baseline
bash scripts/build_ops.sh --impl optimized
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
| 机器 CANN 是 9.1.0, 能编 9.2.0 源码吗 | `build_ops.sh` 用**本机** CANN 编译 (自动探测 `ASCEND_HOME_PATH`), 9.1.0 下可直接试。若 op_host/kernel 用到 9.2 新 API 会**编译报错** (快速失败, 无副作用); 此时取 ops-transformer **9.1.0** 的三算子基线源码, 按 `OPTIMIZATION_NOTES.md` 移植优化 (改动集中 6 个文件, 均为算法层改动) |
| 下了 9.2.0 框架, 机器 CANN 9.1.0 不动, 编完能跑吗 | 大概率能: 脚本永远用**本机 CANN** (9.1.0) 的 ccec/头文件编译, 产物即 9.1.0 原生格式, 与运行时同版本, **无二进制跨版本问题**; 9.2.0 框架只贡献构建脚本与源码, 失配面仅在 API 层 (编不过会响亮报错, 无害)。三种结局: ①编过→直接用, 首次加载留意 op 元数据解析; ②挂框架 cmake 层→换 9.1.0 框架; ③挂算子源码层→按 FAQ 上一条移植。**切忌**另装 9.2.0 CANN 编完放 9.1.0 跑 (二进制跨版本才是真坑) |
| 三算子在 9.1.0 会被版本门槛跳过吗 | 不会 — 三个算子的 CMakeLists 均未调用 `require_cann_version` (无最低 CANN 版本声明), 构建不受该机制影响, 成败取决于真实编译结果 |
| 构建时下载 opbase/catlass/ops-tensor 失败 | 框架 configure 阶段会从 gitcode.com (国内直连) 拉取第三方依赖, A5 需能访问外网; 失败时查看 `build_out/<impl>/build_out/build.log` 中 GIT_REPOSITORY 地址预先准备 |
| 仓内框架 (gitcode 9.2.0) 与 zip 基线有差异? | 有: LI v1/v2 各有 7~9 个文件内容不同 (zip=9.2.0-**beta.2** 内部快照, gitcode=9.2.0 正式版), SFA 完全一致 (逐文件哈希验证)。无害 — build_ops.sh 用本仓算子目录**整体覆盖**框架内同名目录, baseline 也仍从 zip 解出, 对照关系不变 |
| 会替换/影响机器原有算子吗 | **不会写入 CANN 任何原有文件**: 编译/安装产物全在 `build_out/`; 激活 = `opp/vendors/` 下一条软链 + 当前 shell 的 `LD_LIBRARY_PATH` (进程级, 其他终端/用户/业务无感)。`source install_pkg.sh builtin` 删链即恢复原状。注意: 软链存在期间同机其他进程调用**这三个算子**时可能路由到自定义实现 (仅限这三个算子), 测完建议立即切回 builtin |
