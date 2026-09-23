# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import ascend_ops
import logging

def init_routing_cpu(in_tensor, token_table, token_list, expert_num, token_num, copy_byte):
    """CPU侧实现的InitRouting算子"""
    out_tensor = torch.zeros_like(in_tensor)
    
    for expert_idx in range(expert_num):
        token_idx_start = 0 if expert_idx == 0 else token_list[expert_idx - 1]
        token_idx_end = token_list[expert_idx]
        
        for offset_in_expert in range(token_idx_end - token_idx_start):
            out_token_idx = token_idx_start + offset_in_expert
            original_token_idx = token_table[expert_idx, offset_in_expert]
            out_tensor[out_token_idx] = in_tensor[original_token_idx]
    
    return out_tensor


 # 配置参数
expert_num = 2
token_num = 4
hidden_size = 4
blockDim= 40
dtype = torch.bfloat16
copy_byte = hidden_size * 2  # BFLOAT16占2字节
    
# 构造输入数据
input_tensor = torch.arange(token_num * hidden_size, dtype=dtype).reshape(token_num, hidden_size)
token_table = torch.tensor([[0, 2, -1, -1], [1, 3, -1, -1]], dtype=torch.int32)
token_list = torch.tensor([2, 4], dtype=torch.int64)
    
print("输入数据:")
print(f"input_tensor shape: {input_tensor.shape}")
print(f"token_table shape:{token_table.shape}")
print(f"token_list shape: {token_list.shape}")
print(f"input:\n{input_tensor}")
print(f"token_table:\n{token_table}")
print(f"token_list:\n {token_list}")
print()
    
# 1. CPU侧计算
out_cpu = init_routing_cpu(input_tensor, token_table, token_list, expert_num, token_num, copy_byte)
print("CPU输出:")
print(out_cpu)
print()
    
# 2. NPU侧计算
input_npu = input_tensor.npu()
token_table_npu = token_table.npu()
token_list_npu = token_list.npu()
out_npu=torch.empty_like(input_npu).npu()
    
torch.ops.ascend_ops.init_routing(
        blockDim, input_npu, token_table_npu, token_list_npu,out_npu,
        expert_num, token_num, copy_byte
    )
    
out_npu = out_npu.cpu()
print("NPU输出:")
print(out_npu)
print()
    
# 3. 对比结果
if torch.equal(out_cpu, out_npu):
    print("✓ CPU与NPU结果一致！")
else:
    print("✗ CPU与NPU结果不一致！")
    print(f"CPU:\n{out_cpu}")
    print(f"NPU:\n{out_npu}")
