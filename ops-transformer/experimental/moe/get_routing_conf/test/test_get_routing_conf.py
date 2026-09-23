#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import torch
import torch_npu
import ascend_ops

# 基础配置
logging.basicConfig(level=logging.INFO)
BLOCK_DIM = 8
TOKEN_NUM = 16
TOPK = 8  # 只测试 topk=8
LOCAL_EXPERT_NUM = 4
START_EXPERT_ID = 0
end_expert_id1 = LOCAL_EXPERT_NUM
dtype = torch.bfloat16  # 只测试 BF16


# CPU侧参考实现
def get_routing_conf_cpu(indices, scores, local_expert_num, start_expert_id, end_expert_id):
    """
    CPU reference for get_routing_conf
    - init_token_table: 每个expert对应的token索引列表 (flat layout: expert_id * token_num + token_idx)
    - init_token_list: 每个expert的token数量
    - final_score_table: 每个token-expert对的分数 (flat layout: token_id * local_expert_num + expert_id)
    """
    token_num, topk = indices.shape

    # init_token_table 使用flat layout: [expert_id][token_idx] -> expert_id * token_num + token_idx
    init_token_table = torch.full((local_expert_num, token_num), -1, dtype=torch.int32)
    init_token_list = torch.zeros(local_expert_num, dtype=torch.int64)
    final_score_table = torch.zeros((token_num, local_expert_num), dtype=torch.bfloat16)

    for i_token in range(token_num):
        seen_mask = [False] * local_expert_num  # 跟踪每个token已见过的expert，避免重复
        for i_topk in range(topk):
            expert_id = indices[i_token, i_topk].item()
            if expert_id < start_expert_id or expert_id >= end_expert_id:
                continue
            local_expert_id = expert_id % local_expert_num
            if seen[local_expert_id] or init_token_list[local_expert_id] >= token_num:
                continue
            seen_mask[local_expert_id] = True
            init_token_table[local_expert_id, init_token_list[local_expert_id]] = i_token
            init_token_list[local_expert_id] += 1
            final_score_table[i_token, local_expert_id] = scores[i_token, i_topk]

    return init_token_table, init_token_list, final_score_table


def convert_npu_output_to_2d(init_token_table_npu, init_token_list_npu,
                             final_score_table_npu, local_expert_num, token_num):
    """
    将NPU输出的flat tensor转换为2D view
    NPU kernel内部使用flat layout，需要转换为PyTorch的2D view
    """
    # init_token_table: flat (local_expert_num * token_num,) -> 2D (local_expert_num, token_num)
    init_token_table_2d = init_token_table_npu.cpu().view(local_expert_num, token_num)

    # final_score_table: flat (token_num * local_expert_num,) -> 2D (token_num, local_expert_num)
    final_score_table_2d = final_score_table_npu.cpu().view(token_num, local_expert_num)

    return init_token_table_2d, init_token_list_npu.cpu(), final_score_table_2d


# 1. 构造测试数据
torch.manual_seed(42)  # 固定随机种子，保证可复现
indices1 = torch.randint(START_EXPERT_ID, end_expert_id1, (TOKEN_NUM, TOPK), dtype=torch.int32)
scores1 = torch.randn(TOKEN_NUM, TOPK, dtype=dtype)

# 创建flat tensor (与kernel内部layout一致)
init_token_table1 = torch.full((LOCAL_EXPERT_NUM * TOKEN_NUM,), -1, dtype=torch.int32)
init_token_list1 = torch.zeros(LOCAL_EXPERT_NUM, dtype=torch.int64)
final_token_table1 = torch.full((TOKEN_NUM * LOCAL_EXPERT_NUM,), -1, dtype=torch.int32)
final_score_table1 = torch.zeros(TOKEN_NUM * LOCAL_EXPERT_NUM, dtype=dtype)
token_idx_intra_expert1 = torch.full((LOCAL_EXPERT_NUM * TOKEN_NUM,), -1, dtype=torch.int32)

# 2. CPU侧计算结果
init_token_table_cpu, init_token_list_cpu, final_score_table_cpu = get_routing_conf_cpu(
    indices1, scores1, LOCAL_EXPERT_NUM, START_EXPERT_ID, end_expert_id1)

# 3. NPU侧计算结果
# 3.1 数据迁移到NPU
indices_npu = indices1.npu()
scores_npu = scores1.npu()
init_token_table_npu1 = init_token_table1.npu()
init_token_list_npu1 = init_token_list1.npu()
final_token_table_npu1 = final_token_table1.npu()
final_score_table_npu1 = final_score_table1.npu()
token_idx_intra_expert_npu1 = token_idx_intra_expert1.npu()

# 3.2 调用NPU算子
result = torch.ops.ascend_ops.get_routing_conf(
    BLOCK_DIM, indices_npu, scores_npu,
    init_token_table_npu1, init_token_list_npu1, final_token_table_npu1, final_score_table_npu1,
    token_idx_intra_expert_npu1, START_EXPERT_ID, end_expert_id1, LOCAL_EXPERT_NUM, TOKEN_NUM, TOPK)
logging.info(f"NPU operator return code: {result}")

torch.npu.empty_cache()  # 每次循环都释放NPU缓存

# 4. 转换NPU输出为2D view
init_token_table_2d1, init_token_list_npu_2d1, final_score_table_2d1 = \
    convert_npu_output_to_2d(init_token_table_npu1, init_token_list_npu1, final_token_table_npu1,
                             LOCAL_EXPERT_NUM, TOKEN_NUM)

# 5. 结果对比
logging.info(f"------------------------------------------结果对比-----------------------------------------")
logging.info(f"indices tensor: \n{indices1}")
logging.info(f"scores tensor: \n{scores1}")

# init_token_list对比
init_token_list_abs_error = torch.abs(init_token_list_npu_2d - init_token_list_cpu)
logging.info(f"-------------------------------init_token_list的结果对比---------------------------")
logging.info(f"init_token_list cpu result: \n{init_token_list_cpu}")
logging.info(f"init_token_list npu result: \n{init_token_list_npu_2d}")
logging.info(f"init_token_list max absolute error: {init_token_list_abs_error.max().item()}")

# init_token_table对比
init_token_table_abs_error = torch.abs(init_token_table_2d1.float() - init_token_table_cpu.float())
logging.info(f"------------------------init_token_table的结果对比----------------------------------")
logging.info(f"init_token_table cpu result: \n{init_token_table_cpu}")
logging.info(f"init_token_table npu result: \n{init_token_table_2d1}")
logging.info(f"init_token_table max absolute error: {init_token_table_abs_error.max().item()}")
logging.info(f"init_token_table average absolute error: {init_token_table_abs_error.mean().item()}")

# final_score_table对比
final_score_table_abs_error = torch.abs(final_score_table_2d1.float() - final_score_table_cpu.float())
final_score_table_rel_error = final_score_table_abs_error / (torch.abs(final_score_table_cpu.float()) + 1e-12)
logging.info(f"-------------------------final_score_table的结果对比----------------------------------")
logging.info(f"final_score_table cpu result: \n{final_score_table_cpu}")
logging.info(f"final_score_table npu result: \n{final_score_table_2d1}")
logging.info(f"final_score_table max absolute error: {final_score_table_abs_error.max().item():.8f}")
logging.info(f"final_score_table average absolute error: {final_score_table_abs_error.mean().item():.8f}")
logging.info(f"final_score_table max relative error: {final_score_table_rel_error.max().item():.8f}")
logging.info(f"final_score_table average relative error: {final_score_table_rel_error.mean().item():.8f}")