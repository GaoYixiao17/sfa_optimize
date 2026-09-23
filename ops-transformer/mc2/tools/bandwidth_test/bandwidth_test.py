#  Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file bandwidth_test.py
#  \brief

import os
import socket
import logging
from math import ceil
from typing import List, Tuple, Dict, Any
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Manager
import pandas as pd
import numpy as np

import npu_ops_transformer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# region 配置参数

# region 测试模式配置
is_precision_test = False
is_bandwidth_test = True
is_single_test = False
graph_type = 3
is_full_core = 1
core_num = 16 if not is_full_core else 32
aic_core_num = core_num
aiv_core_num = core_num * 2
aiv_num = aiv_core_num
# endregion

# region 性能检测配置
loop = 100
repeat = 1
warmup = 5
profiling_path = "./Profiling_Result_bandwidth_test"
# endregion

# region 通信域配置
server_num = 1
is_cloud = 0

if not is_cloud:
    server_index = 0  # 多机每个脚本保持一致并递增index值
    master_ip = (
        "71.20.45.123" if server_num > 1 else "127.0.0.1"
    )  # 如果通信不通，查看一下master_ip是否设对
else:
    task_index = os.environ.get("VC_TASK_INDEX")
    if task_index is None:
        raise EnvironmentError("环境变量 VC_TASK_INDEX 未设置，请检查云环境配置")
    server_index = int(task_index)
    host_urls = os.environ.get("VC_WORKER_HOSTS")
    if host_urls is None or host_urls == "":
        raise EnvironmentError("环境变量 VC_WORKER_HOSTS 未设置，请检查云环境配置")
    host_url_list = host_urls.split(",")
    master_ip = socket.gethostbyname(host_url_list[0])

rank_per_dev = 8
world_size = server_num * rank_per_dev
server_ranks = slice(
    server_index * rank_per_dev, server_index * rank_per_dev + rank_per_dev
)
ranks_list = list(range(world_size))
# endregion


def cal_hccl_buffsize(test_bs, h, ep_world_size):
    MAX_OUT_DTYPE_SIZE = 2
    WIN_ADDR_ALIGN = 512
    UB_ALIGN = 32
    SCALE_EXPAND_IDX_BUFFER = 44
    DOUBLE_DATA_BUFFER = 2

    token_actual_len = (
        (h * MAX_OUT_DTYPE_SIZE + UB_ALIGN - 1) // UB_ALIGN
    ) * UB_ALIGN + SCALE_EXPAND_IDX_BUFFER
    token_need_size_dispatch = (
        (token_actual_len + WIN_ADDR_ALIGN - 1) // WIN_ADDR_ALIGN
    ) * WIN_ADDR_ALIGN
    actual_size = (
        test_bs * token_need_size_dispatch * ep_world_size
    ) * DOUBLE_DATA_BUFFER

    hccl_buffsize = ceil(actual_size / 1024 / 1024 / 2) + 5  # 留出 5MB 空间余量
    return str(hccl_buffsize)


if is_single_test:
    bs = 2
    data_size = 8192
    max_bs = 2
    test_configs = [
        {"bs": bs, "data_size": data_size, "max_bs": max_bs},
    ]

    # 计算 HCCL_BUFFSIZE 大小
    os.environ["HCCL_BUFFSIZE"] = cal_hccl_buffsize(bs, data_size, world_size)
else:
    # region 输入维度配置 - 多组数据大小测试 (2KB 到 32MB, BS和H交替翻倍)
    test_configs = [
        {"bs": 1, "data_size": 1024, "max_bs": 1},  # 2KB
        {"bs": 2, "data_size": 1024, "max_bs": 2},  # 4KB
        {"bs": 2, "data_size": 2048, "max_bs": 2},  # 8KB
        {"bs": 4, "data_size": 2048, "max_bs": 4},  # 16KB
        {"bs": 4, "data_size": 4096, "max_bs": 4},  # 32KB
        {"bs": 8, "data_size": 4096, "max_bs": 8},  # 64KB
        {"bs": 16, "data_size": 4096, "max_bs": 16},  # 128KB
        {"bs": 32, "data_size": 4096, "max_bs": 32},  # 256KB
        {"bs": 32, "data_size": 8192, "max_bs": 32},  # 512KB
        {"bs": 64, "data_size": 8192, "max_bs": 64},  # 1MB
        {"bs": 128, "data_size": 8192, "max_bs": 128},  # 2MB
        {"bs": 256, "data_size": 8192, "max_bs": 256},  # 4MB
        {"bs": 256, "data_size": 16384, "max_bs": 256},  # 8MB
        {"bs": 512, "data_size": 16384, "max_bs": 512},  # 16MB
        {"bs": 1024, "data_size": 16384, "max_bs": 1024},  # 32MB
        {"bs": 2048, "data_size": 16384, "max_bs": 2048},  # 64MB
        {"bs": 4096, "data_size": 16384, "max_bs": 4096},  # 128MB
        {"bs": 8192, "data_size": 8192, "max_bs": 8192},  # 128MB
    ]
    os.environ["HCCL_BUFFSIZE"] = cal_hccl_buffsize(8192, 8192, world_size)
# endregion

# region 数据类型配置
x_dtype = "fp16"
random_seed = 90
input_dtype = torch.bfloat16 if x_dtype == "bf16" else torch.float16
dtype_bytes = torch.empty(0, dtype=x_dtype).element_size()
# endregion

# region 算子模式配置
if is_precision_test:
    mode = 0
else:
    mode = 1
# endregion

comm_alg = "mte"

# region 其他配置
is_save_tensor_to_bin = False
is_check_setting = True
# endregion

# region 工具函数


def save_tensor_to_bin(tensor, bin_name):
    if not os.path.exists("data"):
        os.makedirs("data")
    tensor.cpu().numpy().tofile(f"data/{bin_name}.bin")


def print_setting():
    logger.info(f"{server_num=}")
    logger.info(f"{rank_per_dev=}")
    logger.info(f"{world_size=}")
    logger.info(f"{mode=}")
    logger.info(f"{comm_alg=}")
    logger.info(f"{graph_type=}")
    logger.info(f"{aiv_num=}")
    logger.info(f"{is_precision_test=}")
    logger.info(f"{is_bandwidth_test=}")
    logger.info(f"{test_configs=}")


def check_setting():
    def error_log_and_exit(log_info, exit_code=1):
        logger.error(log_info)
        raise RuntimeError(f"ERROR: {log_info}")

    for i, cfg in enumerate(test_configs):
        if cfg["bs"] < 1:
            error_log_and_exit(f"test_configs[{i}]: bs 应大于等于 1")
        if cfg["data_size"] < 1:
            error_log_and_exit(f"test_configs[{i}]: data_size 应大于等于 1")
        if cfg["max_bs"] < cfg["bs"]:
            error_log_and_exit(f"test_configs[{i}]: max_bs 应大于等于 bs")

    if (server_index < 0) or (server_index >= server_num):
        error_log_and_exit("server_index 取值范围为 [0, server_num)")

    if server_num * rank_per_dev != world_size:
        error_log_and_exit("server_num * rank_per_dev 必须等于 world_size")

    if (graph_type == 1) or (graph_type == 2):
        error_log_and_exit("当前不支持动态图模式, graph_type 合法值仅有 0 和 3")


# endregion

# region 性能采集函数


def get_profiling(num_loop, num_repeat, num_warmup, profiling_output_path):
    active = num_loop // num_repeat - num_warmup

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=num_warmup, active=active, repeat=num_repeat, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            profiling_output_path
        ),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )
    return prof


def clean_previous_profiling_result():
    profiling_result = "./Profiling_Result_"
    os.system(f"rm -rf {profiling_result}*")


def get_bandwidth_performance(
    prof_result_path: str, data_size_bytes: int, num_world_size: int, test_bs: int
):
    """从 profiling 结果中提取性能数据并计算带宽"""
    execution_times = []

    for dirpath, _, filenames in os.walk(prof_result_path):
        for filename in filenames:
            if filename.startswith("op_summary"):
                op_summary = pd.read_csv(os.path.join(dirpath, filename))
                bandwidth_test_info = op_summary[
                    op_summary["OP Type"] == "BandwidthTest"
                ]

                if not bandwidth_test_info.empty:
                    op_execute_time_series = bandwidth_test_info[
                        "Task Duration(us)"
                    ].tail(50)
                    execution_times.extend(op_execute_time_series.tolist())

    if not execution_times:
        logger.warning("No performance data found")
        return None

    execution_times_us = np.array(execution_times)
    mean_time_us = np.mean(execution_times_us)
    min_time_us = np.min(execution_times_us)
    max_time_us = np.max(execution_times_us)

    mean_time_s = mean_time_us / 1e6
    min_time_s = min_time_us / 1e6

    single_rank_data_bytes = (
        test_bs * data_size_bytes * ((num_world_size - 1) / num_world_size)
    )
    total_data_bytes = single_rank_data_bytes * num_world_size
    mean_bandwidth_gbps = (
        (total_data_bytes / mean_time_s) / 1024 / 1024 / 1024 / num_world_size
    )
    max_bandwidth_gbps = (
        (total_data_bytes / min_time_s) / 1024 / 1024 / 1024 / num_world_size
    )
    single_rank_data_kb = single_rank_data_bytes / 1024

    logger.info("=" * 60)
    logger.info("Bandwidth Test Performance Results")
    logger.info("=" * 60)
    logger.info(f"Total data per iteration: {total_data_bytes / 1e6:.2f} MB")
    logger.info(f"World size: {num_world_size}")
    logger.info(f"BS per rank: {test_bs}")
    logger.info(f"Data size (H): {data_size_bytes // dtype_bytes}")
    logger.info("-" * 60)
    logger.info(f"Mean execution time: {mean_time_us:.2f} us")
    logger.info(f"Min execution time:  {min_time_us:.2f} us")
    logger.info("-" * 60)
    logger.info(f"Mean bandwidth: {mean_bandwidth_gbps:.2f} GB/s")
    logger.info(f"Max bandwidth:  {max_bandwidth_gbps:.2f} GB/s")
    logger.info("=" * 60)

    return {
        "mean_time_us": mean_time_us,
        "min_time_us": min_time_us,
        "max_time_us": max_time_us,
        "mean_bandwidth_gbps": mean_bandwidth_gbps,
        "max_bandwidth_gbps": max_bandwidth_gbps,
        "total_data_bytes": total_data_bytes,
        "single_rank_data_kb": single_rank_data_kb,
        "data_size": data_size_bytes // dtype_bytes,
        "bs": test_bs,
    }


# endregion

# region 图模式相关


def compile_model(model):
    if graph_type == 0:
        raise RuntimeError("单算子模式下不应调用compile_model函数")

    logger.info(f"------------------graph_type: {graph_type}------------------")
    import torchair

    from torchair.configs.compiler_config import CompilerConfig

    if is_full_core:
        compiler_config = None
    else:
        compiler_config = CompilerConfig()
        compiler_config.ge_config.aicore_num = f"{aic_core_num}|{aiv_core_num}"
        logger.info(f"当前图模式仅使用: {aic_core_num} aic核, {aiv_core_num} aiv核")
    npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)

    if graph_type == 3:
        logger.info("graph_type=3, 测试场景: aclgraph入图模式")
        os.makedirs("./static_kernel", exist_ok=True)
        compiler_config = CompilerConfig()
        logger.info("启用aclnn静态内核（aclgraph），当前只支持全核")
        compiler_config.mode = "reduce-overhead"
        # 开启静态Kernel编译
        compiler_config.debug.aclgraph.clone_input = False
        compiler_config.experimental_config.aclgraph._aclnn_static_shape_kernel = True
        compiler_config.experimental_config.aclgraph._aclnn_static_shape_kernel_build_dir = "./static_kernel"

        npu_backend = torchair.get_npu_backend(compiler_config=compiler_config)
        compiled_model = torch.compile(
            model, backend=npu_backend, dynamic=False, fullgraph=True
        )

    return compiled_model


class BANDWIDTH_TEST_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x,
        dstrank_id,
        group,
        test_world_size,
        test_max_bs,
        test_mode,
        test_comm_alg,
        test_aiv_num,
    ):
        local_loop = loop if is_bandwidth_test else 1
        input_x = x
        output = None
        for _ in range(local_loop):
            output = npu_ops_transformer.ops.npu_bandwidth_test(
                x=input_x,
                dstrank_id=dstrank_id,
                group=group,
                world_size=test_world_size,
                max_bs=test_max_bs,
                mode=test_mode,
                comm_alg=test_comm_alg,
                aiv_num=test_aiv_num,
            )
            [y, receive_cnt] = output
            input_x = y[:test_max_bs]
        return output


# endregion

# region 通信初始化


def set_device(rank):
    torch_npu.npu.set_device(rank % rank_per_dev)
    logger.info(f"current device set: {torch_npu.npu.current_device()}")


def init_hccl_comm(rank, world_size_local):
    logger.info(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size_local,
        init_method=f"tcp://{master_ip}:50001",
    )
    logger.info(f"device_{rank} init_process_group success")

    logger.info(f"device {rank} 初始化通信域")
    comm_group = dist.new_group(backend="hccl", ranks=ranks_list)

    logger.debug("[DEBUG] device_%s 等待所有进程初始化完成...", rank)
    dist.barrier()
    logger.debug("[DEBUG] device_%s barrier完成", rank)

    torch.npu.synchronize()
    logger.debug("[DEBUG] device_%s NPU同步完成", rank)

    hcomm_info = comm_group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    logger.debug(
        "[DEBUG] device_%s HCCL通信域名称: %s, 长度: %s",
        rank,
        hcomm_info,
        len(hcomm_info) if hcomm_info else 0,
    )

    if not hcomm_info or len(hcomm_info) == 0:
        raise RuntimeError(f"device_{rank} HCCL通信域获取失败")

    dist.barrier()
    torch.npu.synchronize()
    logger.debug("[DEBUG] device_%s 通信域完全就绪", rank)

    return hcomm_info, comm_group


# endregion

# region 输入生成


def generate_inputs(
    test_bs: int, test_data_size: int
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """生成测试输入数据，每个发送方内部的目标rank均衡"""
    torch.manual_seed(random_seed)

    x_per_rank = [
        torch.empty([test_bs, test_data_size], dtype=input_dtype).uniform_(-1, 1)
        for _ in range(world_size)
    ]

    dstrank_id_per_rank = []
    for _ in range(world_size):
        # 计算每个目标 rank 应分配的数量（均衡分配）
        base = test_bs // world_size
        rem = test_bs % world_size
        targets = []
        for r in range(world_size):
            count = base + (1 if r < rem else 0)
            targets.extend([r] * count)
        # 随机打乱顺序，使每个发送方内部的目标顺序随机
        indices = torch.randperm(test_bs).tolist()
        shuffled_targets = [targets[i] for i in indices]
        dstrank_id_per_rank.append(torch.tensor(shuffled_targets, dtype=torch.int32))

    return x_per_rank, dstrank_id_per_rank


# endregion

# region Golden 生成


def gen_golden_data(
    rank: int,
    x_per_rank: List[torch.Tensor],
    dstrank_id_per_rank: List[torch.Tensor],
    test_max_bs: int,
    test_data_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 golden 参考数据"""
    y = x_per_rank[0].new_zeros([test_max_bs * world_size, test_data_size])
    receive_cnt = torch.zeros([world_size], dtype=torch.int32)

    offset = 0
    for src_rank in range(world_size):
        x = x_per_rank[src_rank]
        dstrank_id = dstrank_id_per_rank[src_rank]

        dst_rank = dstrank_id % world_size

        mask = dst_rank == rank
        tokens_for_me = x[mask]
        count = mask.sum().item()

        if count > 0:
            y[offset : offset + count] = tokens_for_me
            offset += count

        receive_cnt[src_rank] = count

    return y, receive_cnt


# endregion

# region 精度对比


def compare_result(
    npu_output,
    golden_output,
    name: str,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    verbose: bool = True,
) -> bool:
    """对比 NPU 输出与 Golden 输出"""
    if isinstance(npu_output, (list, tuple)):
        all_pass = True
        for i, (npu_out, golden_out) in enumerate(zip(npu_output, golden_output)):
            if not compare_result(
                npu_out, golden_out, f"{name}_{i}", rtol, atol, verbose
            ):
                all_pass = False
        return all_pass

    logger.info(f"{'=' * 60}")
    logger.info(f"对比 {name}")
    logger.info(f"NPU shape: {npu_output.shape}, Golden shape: {golden_output.shape}")
    logger.info(f"{'=' * 60}")

    if verbose:
        logger.info("NPU 输出:")
        logger.info(npu_output)
        logger.info("Golden 输出:")
        logger.info(golden_output)

    if torch.equal(npu_output, golden_output):
        logger.info(f"[PASS] {name} 精度对比通过 (完全一致)")
        return True

    diff = torch.abs(npu_output - golden_output)
    max_diff = diff.max().item()

    if torch.is_floating_point(npu_output):
        mean_diff = diff.mean().item()
        rel_diff = diff / (torch.abs(golden_output) + 1e-8)
        max_rel_diff = rel_diff.max().item()

        if verbose:
            logger.info("差异:")
            logger.info(diff)
            logger.info("相对差异:")
            logger.info(rel_diff)

        if max_rel_diff > rtol or max_diff > atol:
            logger.error(f"[ERROR] {name} 精度不满足要求:")
            logger.error(f"  最大相对误差: {max_rel_diff}")
            logger.error(f"  容忍范围: rtol={rtol}, atol={atol}")
            return False
        else:
            logger.info(f"[PASS] {name} 精度对比通过 (误差在容忍范围内)")
            logger.info(f"  最大绝对误差: {max_diff}, 最大相对误差: {max_rel_diff:.2e}")
    else:
        mean_diff = diff.float().mean().item()

        if verbose:
            logger.info("差异:")
            logger.info(diff)

        if max_diff > atol:
            logger.error(f"[ERROR] {name} 精度不满足要求:")
            logger.error(f"  容忍范围: atol={atol}")
            return False
        else:
            logger.info(f"[PASS] {name} 精度对比通过 (误差在容忍范围内)")
            logger.info(f"  最大绝对误差: {max_diff}")

    return True


# endregion

# region NPU 执行


def run_bandwidth_test_npu(
    queue,
    rank,
    x,
    dstrank_id,
    hcomm_info,
    test_max_bs=None,
    test_mode=None,
    test_world_size=None,
    is_perf_mode=False,
    destroy_comm=True,
):
    """执行 bandwidth_test 算子"""
    set_device(rank)

    if hcomm_info is None:
        hcomm_info, _ = init_hccl_comm(rank, test_world_size)

    if is_save_tensor_to_bin:
        save_tensor_to_bin(x.to(torch.float32), "x_in_" + str(rank))
        save_tensor_to_bin(dstrank_id.to(torch.int32), "dstrank_id_" + str(rank))

    logger.info(f"[INFO] device_{rank} 构造bandwidth_test算子输入数据")

    x_npu = x.to(input_dtype).npu()
    dstrank_id_npu = dstrank_id.to(torch.int32).npu()
    torch.npu.synchronize()
    logger.debug("[DEBUG] device_%s 开始调用算子", rank)

    # 根据模式执行
    if graph_type == 0:
        bandwidth_test_kwargs = {
            "x": x_npu,
            "dstrank_id": dstrank_id_npu,
            "group": hcomm_info,
            "world_size": test_world_size,
            "max_bs": test_max_bs,
            "mode": test_mode,
            "comm_alg": comm_alg,
            "aiv_num": aiv_num,
        }

        if not is_perf_mode:
            y, receive_cnt = npu_ops_transformer.ops.npu_bandwidth_test(
                **bandwidth_test_kwargs
            )
        else:
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(loop, repeat, warmup, profiling_path) as prof:
                for i in range(loop):
                    torch.npu.synchronize()
                    y, receive_cnt = npu_ops_transformer.ops.npu_bandwidth_test(
                        **bandwidth_test_kwargs
                    )
                    if i == 1:
                        for _ in range(100):
                            torch.exp(tensor)
                    prof.step()
    else:
        bandwidth_test_kwargs = {
            "x": x_npu,
            "dstrank_id": dstrank_id_npu,
            "group": hcomm_info,
            "test_world_size": test_world_size,
            "test_max_bs": test_max_bs,
            "test_mode": test_mode,
            "test_comm_alg": comm_alg,
            "test_aiv_num": aiv_num,
        }
        if not is_perf_mode:
            model = compile_model(BANDWIDTH_TEST_GRAPH_Model())
            y, receive_cnt = model(**bandwidth_test_kwargs)
        else:
            torch._dynamo.reset()
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            model = compile_model(BANDWIDTH_TEST_GRAPH_Model())
            loop_time = 2
            repeat_time = 1
            warm_time = 1
            with get_profiling(
                loop_time, repeat_time, warm_time, profiling_path
            ) as prof:
                for i in range(loop_time):
                    y, receive_cnt = model(**bandwidth_test_kwargs)
                    if i == 1:
                        for _ in range(100):
                            torch.exp(tensor)
                    prof.step()

    torch.npu.synchronize()
    torch.npu.empty_cache()

    if destroy_comm:
        dist.destroy_process_group()

    logger.info(f"rank {rank} npu finished! \n")

    if is_save_tensor_to_bin:
        save_tensor_to_bin(y.to(torch.float32), "y_out_" + str(rank))
        save_tensor_to_bin(receive_cnt, "receive_cnt_" + str(rank))

    queue.put([rank, [y.cpu(), receive_cnt.cpu()]])


def run_bandwidth_test_npu_bandwidth(
    queue, rank, x, dstrank_id, test_max_bs, test_mode, test_world_size
):
    """专门用于带宽测试的执行函数"""
    run_bandwidth_test_npu(
        queue,
        rank,
        x,
        dstrank_id,
        None,
        test_max_bs,
        test_mode,
        test_world_size,
        is_perf_mode=True,
        destroy_comm=True,
    )


def run_bandwidth_test_npu_multi_config(queue, rank, configs):
    """多配置带宽测试，共享通信域

    Args:
        queue: 多进程通信队列
        rank: 当前进程rank
        configs: 配置列表，每个元素为 (x, dstrank_id, test_max_bs, test_data_size, config_index) 元组
    """
    set_device(rank)

    if not configs:
        queue.put([rank, []])
        return

    _, _, _, _, test_world_size, _ = configs[0]

    logger.info(f"[INFO] device_{rank} 创建通信域（多配置共享）")
    hcomm_info, _ = init_hccl_comm(rank, test_world_size)

    results = []

    for _, (
        x,
        dstrank_id,
        test_max_bs,
        test_data_size,
        test_world_size,
        config_index,
    ) in enumerate(configs):
        test_bs = x.size(0)
        logger.info(f"{'=' * 60}")
        logger.info(
            f"[Rank {rank}] Config [{config_index}] bs={test_bs}, data_size={test_data_size},"
            f"max_bs={test_max_bs}"
        )
        logger.info(f"{'=' * 60}")

        logger.info(f"[INFO] device_{rank} 构造bandwidth_test算子输入数据")

        x_npu = x.to(input_dtype).npu()
        dstrank_id_npu = dstrank_id.to(torch.int32).npu()

        torch.npu.synchronize()
        logger.debug("[DEBUG] device_%s 开始调用算子", rank)
        dist.barrier()

        config_profiling_path = f"{profiling_path}_config_{config_index}"

        if graph_type == 0:
            bandwidth_test_kwargs = {
                "x": x_npu,
                "dstrank_id": dstrank_id_npu,
                "group": hcomm_info,
                "world_size": test_world_size,
                "max_bs": test_max_bs,
                "mode": mode,
                "comm_alg": comm_alg,
                "aiv_num": aiv_num,
            }
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            with get_profiling(loop, repeat, warmup, config_profiling_path) as prof:
                for j in range(loop):
                    torch.npu.synchronize()
                    y, receive_cnt = npu_ops_transformer.ops.npu_bandwidth_test(
                        **bandwidth_test_kwargs
                    )
                    if j == 1:
                        for _ in range(100):
                            torch.exp(tensor)
                    prof.step()
        else:
            bandwidth_test_kwargs = {
                "x": x_npu,
                "dstrank_id": dstrank_id_npu,
                "group": hcomm_info,
                "test_world_size": test_world_size,
                "test_max_bs": test_max_bs,
                "test_mode": mode,
                "test_comm_alg": comm_alg,
                "test_aiv_num": aiv_num,
            }
            torch._dynamo.reset()
            tensor = torch.ones(10000, 10000).to(torch.float16).npu()
            model = compile_model(BANDWIDTH_TEST_GRAPH_Model())
            loop_time = 2
            repeat_time = 1
            warm_time = 1
            with get_profiling(
                loop_time, repeat_time, warm_time, config_profiling_path
            ) as prof:
                for i in range(loop_time):
                    y, receive_cnt = model(**bandwidth_test_kwargs)
                    if i == 1:
                        for _ in range(100):
                            torch.exp(tensor)
                    prof.step()

        dist.barrier()
        torch.npu.synchronize()
        torch.npu.empty_cache()

        if is_save_tensor_to_bin:
            save_tensor_to_bin(
                y.to(torch.float32), f"y_out_{rank}_config{config_index}"
            )
            save_tensor_to_bin(receive_cnt, f"receive_cnt_{rank}_config{config_index}")

        results.append(
            {
                "config_index": config_index,
                "y": y.cpu(),
                "receive_cnt": receive_cnt.cpu(),
                "test_bs": test_bs,
                "test_data_size": test_data_size,
            }
        )

        logger.info(f"[INFO] device_{rank} Config [{config_index}] 完成")

    dist.destroy_process_group()

    queue.put([rank, results])


# endregion

# region 主测试函数


def run_precision_test(
    x_per_rank: List[torch.Tensor],
    dstrank_id_per_rank: List[torch.Tensor],
    test_max_bs: int,
    test_data_size: int,
) -> bool:
    """运行精度测试"""
    logger.info("=" * 60)
    logger.info("Precision Test")
    logger.info("=" * 60)

    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        p = Process(
            target=run_bandwidth_test_npu,
            args=(
                result_queue,
                rank,
                x_per_rank[rank],
                dstrank_id_per_rank[rank],
                None,
                test_max_bs,
                mode,
                world_size,
                False,
                True,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    npu_outputs = {}

    if result_queue.qsize() == world_size:
        while not result_queue.empty():
            rank, output = result_queue.get()
            if output is not None:
                npu_outputs[rank] = output
            else:
                logger.error(f"[ERROR] Rank {rank} 执行失败")
                return False
    else:
        logger.error("Get result_queue Failed")
        return False

    logger.info("-" * 60)
    logger.info("Precision Verification:")
    logger.info("-" * 60)

    all_pass = True
    for rank in range(world_size):
        logger.info(f"\n[Rank {rank}] 开始精度对比...")

        golden_y, golden_receive_cnt = gen_golden_data(
            rank, x_per_rank, dstrank_id_per_rank, test_max_bs, test_data_size
        )

        npu_y, npu_receive_cnt = npu_outputs[rank]

        y_pass = compare_result(npu_y, golden_y, f"rank_{rank}_y")
        receive_cnt_pass = compare_result(
            npu_receive_cnt, golden_receive_cnt, f"rank_{rank}_receive_cnt"
        )

        if not (y_pass and receive_cnt_pass):
            all_pass = False

    return all_pass


def run_bandwidth_test(
    x_per_rank: List[torch.Tensor],
    dstrank_id_per_rank: List[torch.Tensor],
    test_max_bs: int,
    test_data_size: int,
) -> Dict[str, Any]:
    """运行带宽测试"""
    logger.info("=" * 60)
    logger.info("Bandwidth Test")
    logger.info("=" * 60)

    clean_previous_profiling_result()

    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        p = Process(
            target=run_bandwidth_test_npu_bandwidth,
            args=(
                result_queue,
                rank,
                x_per_rank[rank],
                dstrank_id_per_rank[rank],
                test_max_bs,
                mode,
                world_size,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    npu_outputs = {}
    while not result_queue.empty():
        rank, output = result_queue.get()
        if output is not None:
            npu_outputs[rank] = output
        else:
            logger.error(f"[ERROR] Rank {rank} 带宽测试执行失败")
            return None

    data_size_bytes = test_data_size * dtype_bytes
    perf_result = get_bandwidth_performance(
        profiling_path, data_size_bytes, world_size, len(x_per_rank[0])
    )

    return perf_result


def run_multi_config_bandwidth_test() -> List[Dict[str, Any]]:
    """运行多配置带宽测试（共享通信域）"""
    all_results = []

    clean_previous_profiling_result()

    all_inputs = []
    for i, cfg in enumerate(test_configs):
        test_bs = cfg["bs"]
        test_data_size = cfg["data_size"]
        test_max_bs = cfg["max_bs"]
        x_per_rank, dstrank_id_per_rank = generate_inputs(test_bs, test_data_size)
        all_inputs.append(
            (x_per_rank, dstrank_id_per_rank, test_max_bs, test_data_size, i + 1)
        )

    manager = Manager()
    result_queue = manager.Queue()

    processes = []
    for rank in range(world_size):
        rank_configs = [
            (
                all_inputs[i][0][rank],
                all_inputs[i][1][rank],
                all_inputs[i][2],
                all_inputs[i][3],
                world_size,
                all_inputs[i][4],
            )
            for i in range(len(all_inputs))
        ]
        p = Process(
            target=run_bandwidth_test_npu_multi_config,
            args=(result_queue, rank, rank_configs),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    npu_all_results = {}
    while not result_queue.empty():
        rank, results = result_queue.get()
        if results is not None:
            npu_all_results[rank] = results
        else:
            logger.error(f"[ERROR] Rank {rank} 多配置测试执行失败")
            return []

    for i, cfg in enumerate(test_configs):
        test_data_size = cfg["data_size"]
        test_bs = cfg["bs"]
        data_size_bytes = test_data_size * dtype_bytes
        config_index = i + 1

        config_profiling_path = f"{profiling_path}_config_{config_index}"
        perf_result = get_bandwidth_performance(
            config_profiling_path, data_size_bytes, world_size, test_bs
        )

        if perf_result:
            perf_result["config_index"] = config_index
            perf_result["config"] = cfg
            all_results.append(perf_result)

    print_summary_table(all_results)

    return all_results


def print_summary_table(results: List[Dict[str, Any]]):
    """打印汇总结果表格"""
    logger.info("=" * 90)
    logger.info("Bandwidth Test Summary - All Configurations")
    logger.info("=" * 90)

    if not results:
        logger.warning("[WARN] No valid test results")
        return

    header = (
        f"{'No.':<5} {'BS':<5} {'H':<10} {'Total Data(KB)':<16} "
        f"{'Mean BW(GB/s)':<16} {'Max BW(GB/s)':<16} {'Mean Time(us)':<14}"
    )
    logger.info(header)
    logger.info("-" * 90)

    max_bandwidth = 0
    max_bw_config = None

    for r in results:
        cfg = r["config"]
        row = (
            f"{r['config_index']:<5} {cfg['bs']:<5} {cfg['data_size']:<10} "
            f"{r['single_rank_data_kb']:<16.4f} {r['mean_bandwidth_gbps']:<16.2f} "
            f"{r['max_bandwidth_gbps']:<16.2f} {r['mean_time_us']:<14.2f}"
        )
        logger.info(row)

        if r["max_bandwidth_gbps"] > max_bandwidth:
            max_bandwidth = r["max_bandwidth_gbps"]
            max_bw_config = r

    logger.info("-" * 90)

    if max_bw_config:
        logger.info(f"\n{'*' * 90}")
        logger.info(
            f"Maximum Bandwidth: {max_bw_config['max_bandwidth_gbps']:.2f} GB/s"
        )
        logger.info(f"  H: {max_bw_config['config']['data_size']}")
        logger.info(f"  Total Data: {max_bw_config['single_rank_data_kb']:.4f} KB")
        logger.info(f"  Mean Time: {max_bw_config['mean_time_us']:.2f} us")
        logger.info(f"{'*' * 90}")


# endregion

# region Main


def main():
    print_setting()
    if is_check_setting:
        check_setting()

    if is_bandwidth_test and len(test_configs) > 1:
        all_results = run_multi_config_bandwidth_test()

        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(
            f"Total configurations tested: {len(all_results)}/{len(test_configs)}"
        )
        logger.info("=" * 60)
    else:
        x_per_rank, dstrank_id_per_rank = generate_inputs(bs, data_size)

        precision_pass = True
        bandwidth_result = None

        if is_precision_test:
            precision_pass = run_precision_test(
                x_per_rank, dstrank_id_per_rank, max_bs, data_size
            )

        if is_bandwidth_test:
            bandwidth_result = run_bandwidth_test(
                x_per_rank, dstrank_id_per_rank, max_bs, data_size
            )

        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)

        if is_precision_test:
            logger.info(f"Precision Test: {'PASS' if precision_pass else 'FAIL'}")

        if is_bandwidth_test:
            if bandwidth_result:
                logger.info("Bandwidth Test: Completed")
                logger.info(
                    f"  Mean Bandwidth: {bandwidth_result['mean_bandwidth_gbps']:.2f} GB/s"
                )
                logger.info(
                    f"  Max Bandwidth:  {bandwidth_result['max_bandwidth_gbps']:.2f} GB/s"
                )
            else:
                logger.error("Bandwidth Test: FAILED")

        logger.info("=" * 60)

        if is_precision_test and not precision_pass:
            exit(1)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
