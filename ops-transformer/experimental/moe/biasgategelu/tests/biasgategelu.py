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

def bias_gate_gelu_cpu(in_tensor, bias):
    """CPU侧实现 BiasGateGelu 算子"""
    gbH, gbW = in_tensor.shape
    half_dim = gbW // 2
    
    gate = in_tensor[:, :half_dim]
    value = in_tensor[:, half_dim:]
    
    gate_bias = bias[:half_dim]
    value_bias = bias[half_dim:]
    
    gate = torch.nn.functional.gelu(gate + gate_bias, approximate='tanh')
    value = value + value_bias
    out = gate * value
    
    return out

# 测试逻辑
print("=== BiasGateGelu 算子测试 ===")

gbH, gbW = 8, 256
block_dim = 40
dtype = torch.float16

in_tensor = torch.randn(gbH, gbW, dtype=dtype)
bias = torch.randn(gbW, dtype=dtype)

# CPU计算
print("[CPU计算]")
z_cpu = bias_gate_gelu_cpu(in_tensor, bias)
print(f"  输出形状: {z_cpu.shape}")
print(f"  前2行示例:\n{z_cpu[:2]}")

# NPU计算
print("\n[NPU计算]")
in_tensor_npu = in_tensor.npu()
bias_npu = bias.npu()
out_npu = torch.zeros(gbH, gbW // 2, dtype=dtype).npu()

torch.ops.ascend_ops.bias_gate_gelu(block_dim, in_tensor_npu, bias_npu, out_npu, gbH, gbW)
z_npu_cpu = out_npu.cpu()
print(f"  输出形状: {z_npu_cpu.shape}")
print(f"  前2行示例:\n{z_npu_cpu[:2]}")

# 结果对比
print("\n[结果对比]")
match = torch.allclose(z_cpu, z_npu_cpu, rtol=1e-2, atol=1e-2)
if match:
    print("✓ CPU与NPU结果一致！")
else:
    print("✗ CPU与NPU结果不一致！")
    diff = torch.abs(z_cpu - z_npu_cpu)
    print(f"  最大差异: {diff.max().item()}")
