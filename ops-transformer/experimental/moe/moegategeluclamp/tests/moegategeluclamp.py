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

def moegategeluclamp_cpu(x, is_clamp, clamp_value):
    """CPU侧实现: z = clamp(GELU(x_left) * x_right, -clamp_value, clamp_value)"""
    per_count = x.shape[-1]
    half_dim = per_count // 2
    
    x_left = x[:, :half_dim].float()
    x_right = x[:, half_dim:].float()
    
    x_left = torch.nn.functional.gelu(x_left, approximate='tanh')  # 匹配Ascend C的tanh近似
    z = x_left * x_right
    
    if is_clamp:
        z = torch.clamp(z, -clamp_value, clamp_value)
    
    return z


NUM_ROWS, per_count, per_total_count = 8, 256, 16
is_clamp = True
clamp_value = 10.0
dtype = torch.bfloat16  # 设计文档要求 BFLOAT16
dtype_id = 27  # BFLOAT16 的 dtype_id
block_dim = 40
use_total_rows = False

x = torch.randn(NUM_ROWS, per_count, dtype=dtype)
total_rows = torch.zeros(per_total_count, dtype=torch.int64)

print("=== MOE Gate GELU Clamp 算子测试 ===")
print(f"参数: NUM_ROWS={NUM_ROWS}, per_count={per_count}, is_clamp={is_clamp}, clamp_value={clamp_value}")

print("\n[CPU计算]")
z_cpu = moegategeluclamp_cpu(x, is_clamp, clamp_value)
z_cpu = z_cpu.to(dtype)
print(f"  输出形状: {z_cpu.shape}")
print(f"  前2行示例:\n{z_cpu[:2]}")

print("[NPU计算]")
x_npu = x.npu()
total_rows_npu = total_rows.npu()
z_npu = torch.empty(NUM_ROWS, per_count // 2, dtype=dtype).npu()

torch.ops.ascend_ops.moe_gate_gelu_clamp(
    block_dim, x_npu, total_rows_npu, z_npu,
    NUM_ROWS, per_count, per_total_count, is_clamp, clamp_value, dtype_id, use_total_rows)

z_npu_cpu = z_npu.cpu()
print(f"  输出形状: {z_npu_cpu.shape}")
print(f"  前2行示例:\n{z_npu_cpu[:2]}")

print("\n[结果对比]")
# 使用 allclose 比较浮点结果（允许微小精度差异）
match = torch.allclose(z_cpu, z_npu_cpu, rtol=1e-2, atol=1e-2)

if match:
    print("✓ CPU与NPU结果一致！")
else:
    print("✗ CPU与NPU结果不一致！")
    diff = torch.abs(z_cpu - z_npu_cpu)
    print(f"  最大差异: {diff.max().item()}")