# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"AllGatherMatmulV2样例测试"

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from torch.multiprocessing import Process, Manager
import torch.multiprocessing as mp


def test_multiprocess(input_list):
    x1_list, x2_list, world_size = input_list
    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method("forkserver", force=True)
    for i in range(world_size):
        proc = Process(
            target=gen_npu, args=(x1_list[i], x2_list[i], world_size, i, result_queue)
        )
        proc.start()
        proc_list.append(proc)
    for proc in proc_list:
        proc.join()
    output_list = [None] * world_size
    gather_out_list = [None] * world_size
    for _ in proc_list:
        rank, output, gather_output = result_queue.get()
        output_list[rank] = output
        gather_out_list[rank] = gather_output
    return output_list, gather_out_list


def gen_golden_data(x1_list, x2_list):
    golden_gather_out = np.concatenate(x1_list, axis=0)
    out_list = []
    for i in range(world_size):
        out = np.matmul(
            golden_gather_out.astype(np.float32), x2_list[i].numpy().astype(np.float32)
        ).astype(np.float16)
        out_list.append(out)

    return golden_gather_out, out_list


def gen_npu(x1, x2, world_size, rank, queue):
    torch_npu.npu.set_device(rank)
    master_ip = "127.0.0.1"
    print(f"[INFO] device_{rank} 创建HCCL通信链路")
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://{master_ip}:50001",
    )
    print(f"[INFO] device_{rank} init_process_group success")
    group = dist.distributed_c10d._get_default_group()
    hcom_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
    x1 = x1.npu()
    x2 = x2.npu()
    output_npu, gather_output_npu = torch_npu.npu_all_gather_base_mm(
        x1, x2, hcom_name, world_size, gather_output=True
    )
    queue.put((rank, output_npu.cpu().numpy(), gather_output_npu.cpu().numpy()))


def cal_relativediff_numpy(data_check, data_exepect, diff_thd):
    a = np.abs(np.subtract(data_check, data_exepect))
    b1 = np.maximum(np.abs(data_check), (np.abs(data_exepect)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result


def data_compare(data_check, data_exepect, diff_thd=0.005, pct_thd=0.005):
    npu_shape = data_check.shape
    expect_shape = data_exepect.shape
    if npu_shape != expect_shape:
        print("[ERROR] ============ out_shape is not equal expect!")
        return False
    data_check = data_check.flatten()
    data_exepect = data_exepect.flatten()
    start = 0
    end = data_check.size - 1
    diff = cal_relativediff_numpy(data_check, data_exepect, diff_thd)
    split_count = int(end - start + 1) if end != start else 1
    lt_num = diff[diff < diff_thd].size
    lt_num = (
        lt_num
        + data_exepect[np.isinf(data_exepect)].size
        + data_exepect[np.isnan(data_exepect)].size
    )
    lt_pct = float(lt_num) / float(split_count) * 100
    pct_thd = (1 - pct_thd) * 100.0
    return lt_pct >= pct_thd


def verify_result(gather_out, out, golden_gather_out, golden_out):
    for i in range(world_size):
        npu_gather_out = gather_out[i]
        npu_out = out[i]
        golden = golden_out[i]

        if not data_compare(npu_gather_out, golden_gather_out):
            print("[ERROR] ============ rank{} gather_out precision check failed", i)
            return False
        if not data_compare(npu_out, golden):
            print("[ERROR] ============ rank{} out precision check failed", i)
            return False

    return True


if __name__ == "__main__":
    # 生成输入数据
    world_size = 4
    m, k, n = 1024, 10240, 5120
    x1_list = []
    x2_list = []
    x1_scale_list = []
    x2_scale_list = []
    for _ in range(world_size):
        x1 = torch.randint(-1, 1, size=(m, k), dtype=torch.float16)
        x2 = torch.randint(-10, 10, size=(k, n), dtype=torch.float16)
        x1_list.append(x1)
        x2_list.append(x2)
    # 生成golden值
    golden_gather_out, golden_out = gen_golden_data(x1_list, x2_list)
    # 执行Npu任务
    output_npu, gather_output_npu = test_multiprocess([x1_list, x2_list, world_size])

    # 结果比对
    if verify_result(gather_output_npu, output_npu, golden_gather_out, golden_out):
        print("[INFO] Precision PASS")
    else:
        print("[ERROR] Precision FAILED!")
