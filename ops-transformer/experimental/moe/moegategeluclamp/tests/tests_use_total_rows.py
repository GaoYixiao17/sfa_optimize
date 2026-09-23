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

def moegategeluclamp_cpu(x, is_clamp, clamp_value):
    """CPU侧实现: z = clamp(GELU(x_left) * x_right, -clamp_value, clamp_value)
    
    参数:
        x: 输入张量，shape [NUM_ROWS, per_count]
        is_clamp: 是否执行 clamp 操作
        clamp_value: clamp 的边界值
    
    返回:
        z: 输出张量，shape [NUM_ROWS, per_count // 2]
    """
    per_count = x.shape[-1]
    half_dim = per_count // 2
    
    # 分割左右两部分
    x_left = x[:, :half_dim].float()   # [NUM_ROWS, half_dim]
    x_right = x[:, half_dim:].float()   # [NUM_ROWS, half_dim]
    
    # GELU 激活（使用 tanh 近似，与 Ascend C 实现一致）
    x_left = torch.nn.functional.gelu(x_left, approximate='tanh')
    
    # 逐元素相乘
    z = x_left * x_right
    
    # Clamp 操作（如果启用）
    if is_clamp:
        z = torch.clamp(z, -clamp_value, clamp_value)
    
    return z


def test_use_total_rows():
    """测试 use_total_rows=True 的场景
    
    工作原理（NPU侧）：
    1. 从 total_rows 张量读取 per_total_count 个 int64 值
    2. 取最后一个元素 total_rows[per_total_count-1] 作为实际行数 real_rows
    3. 只处理前 real_rows 行数据
    """
    print("=== use_total_rows=True 测试 ===\n")
    
    # 测试参数
    NUM_ROWS = 8           # 输入总行数
    per_count = 256        # 每行的元素数（half_dim = per_count / 2 = 128）
    per_total_count = 16   # total_rows 张量的长度
    is_clamp = True
    clamp_value = 10.0
    dtype = torch.bfloat16
    dtype_id = 27          # BFLOAT16
    block_dim = 40
    use_total_rows = True
    
    # total_rows 数据：最后一个元素决定实际处理的行数
    total_rows_data = [3, 5, 2, 8, 1, 4, 6, 7, 9, 0, 2, 3, 5, 1, 4, 5]
    real_rows = total_rows_data[-1]  # 最后一个元素 = 5
    
    print(f"输入参数:")
    print(f"  NUM_ROWS = {NUM_ROWS}")
    print(f"  per_count = {per_count}")
    print(f"  per_total_count = {per_total_count}")
    print(f"  total_rows_data = {total_rows_data}")
    print(f"  实际处理行数 (total_rows[-1]) = {real_rows}")
    print(f"  is_clamp = {is_clamp}, clamp_value = {clamp_value}")
    print(f"  dtype = {dtype}, dtype_id = {dtype_id}")
    
    # 创建输入数据
    torch.manual_seed(42)  # 固定随机种子，保证可复现
    x = torch.randn(NUM_ROWS, per_count, dtype=dtype)
    total_rows = torch.tensor(total_rows_data, dtype=torch.int64)
    
    print(f"\n输入数据:")
    print(f"  x.shape = {x.shape}")
    print(f"  total_rows.shape = {total_rows.shape}")
    
    # ==================== CPU 侧计算 ====================
    print("\n[CPU 侧计算]")
    
    # 只取前 real_rows 行进行计算
    x_cpu = x[:real_rows]
    print(f"  CPU 处理行数: {real_rows}")
    print(f"  x_cpu.shape = {x_cpu.shape}")
    
    # CPU 计算
    z_cpu = moegategeluclamp_cpu(x_cpu, is_clamp, clamp_value)
    z_cpu = z_cpu.to(dtype)
    print(f"  z_cpu.shape = {z_cpu.shape}")
    
    # 扩展到 NUM_ROWS 行（未处理的行填充0）
    z_cpu_full = torch.zeros(NUM_ROWS, per_count // 2, dtype=dtype)
    z_cpu_full[:real_rows] = z_cpu
    print(f"  z_cpu_full.shape = {z_cpu_full.shape} (填充0用于对比)")
    
    # ==================== NPU 侧计算 ====================
    print("\n[NPU 侧计算]")
    
    x_npu = x.npu()
    total_rows_npu = total_rows.npu()
    z_npu = torch.empty(NUM_ROWS, per_count // 2, dtype=dtype).npu()
    
    torch.ops.ascend_ops.moe_gate_gelu_clamp(
        block_dim, x_npu, total_rows_npu, z_npu,
        NUM_ROWS, per_count, per_total_count, is_clamp, clamp_value, dtype_id, use_total_rows)
    
    z_npu_cpu = z_npu.cpu()
    print(f"  z_npu.shape = {z_npu.shape}")
    print(f"  z_npu 前 {real_rows} 行是否有效（非零）: {(z_npu_cpu[:real_rows] != 0).any()}")
    print(f"  z_npu 后 {NUM_ROWS - real_rows} 行是否为0: {(z_npu_cpu[real_rows:] == 0).all()}")
    
    # ==================== 结果对比 ====================
    print("\n[结果对比]")
    
    # 只对比前 real_rows 行
    z_cpu_compare = z_cpu_full[:real_rows]
    z_npu_compare = z_npu_cpu[:real_rows]
    
    print(f"  对比范围: [0:{real_rows}]")
    
    # 检查是否有 NaN
    has_nan_cpu = torch.isnan(z_cpu_compare).any()
    has_nan_npu = torch.isnan(z_npu_compare).any()
    print(f"  CPU 有 NaN: {has_nan_cpu}")
    print(f"  NPU 有 NaN: {has_nan_npu}")
    
    if has_nan_cpu or has_nan_npu:
        print("✗ 结果包含 NaN，测试失败！")
        return False
    
    # 使用 allclose 对比
    match = torch.allclose(z_cpu_compare, z_npu_compare, rtol=1e-2, atol=1e-2)
    
    if match:
        print(f"✓ CPU 与 NPU 结果一致！")
        print("\n前2行对比:")
        print(f"  CPU: {z_cpu_compare[:2]}")
        print(f"  NPU: {z_npu_compare[:2]}")
        return True
    else:
        print(f"✗ CPU 与 NPU 结果不一致！")
        diff = torch.abs(z_cpu_compare - z_npu_compare)
        print(f"  最大差异: {diff.max().item()}")
        print(f"  平均差异: {diff.mean().item()}")
        
        # 找出差异最大的位置
        max_idx = torch.argmax(diff)
        row = max_idx.item() // z_cpu_compare.shape[1]
        col = max_idx.item() % z_cpu_compare.shape[1]
        print(f"  最大差异位置: row={row}, col={col}")
        print(f"  CPU 值: {z_cpu_compare[row, col]}")
        print(f"  NPU 值: {z_npu_compare[row, col]}")
        return False


if __name__ == "__main__":
    success = test_use_total_rows()
    print(f"\n{'='*60}")
    if success:
        print("测试通过！")
    else:
        print("测试失败！")
