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

def fusedexpert_cpu(expertId, topk_logit, pos, expertNum, shared_expert, sharedId_dp, per_count, start_tokenidx, end_tokenidx):
    ROWS = expertId.shape[0]
    TILE_LENGTH = per_count
    TILE_LENGTH_OUT = per_count + shared_expert
    scalar = pos

    fused_expertId = torch.zeros((ROWS, TILE_LENGTH_OUT), dtype=torch.int32)
    fused_topk_logit = torch.zeros((ROWS, TILE_LENGTH_OUT), dtype=topk_logit.dtype)

    for row_idx in range(ROWS):
        share_expertId = []
        for i in range(shared_expert):
            sid = sharedId_dp + i
            if row_idx < start_tokenidx or row_idx >= end_tokenidx:
                sid += expertNum if sharedId_dp == 0 else -expertNum
            share_expertId.append(sid)

        fused_topk_logit[row_idx, :TILE_LENGTH] = topk_logit[row_idx]
        # 将 1.0 转换为相同的 BFloat16 精度
        one_bf16 = torch.tensor(1.0, dtype=torch.bfloat16)
        fused_topk_logit[row_idx, TILE_LENGTH:] = one_bf16
        fused_topk_logit[row_idx, TILE_LENGTH:] = 1.0

        offset = [((eid >> scalar) + 1) * shared_expert for eid in expertId[row_idx]]
        fused_expertId[row_idx, :TILE_LENGTH] = expertId[row_idx] + torch.tensor(offset, dtype=torch.int32)
        fused_expertId[row_idx, TILE_LENGTH:] = torch.tensor(share_expertId, dtype=torch.int32)

    return fused_expertId, fused_topk_logit


ROWS, per_count, shared_expert = 8, 14, 2
expertNum, pos, sharedId_dp = 18, 5, 0
thread_count, start_tokenidx, end_tokenidx = 8, 0, ROWS
block_dim = 40  # AI CORE 数量，Ascend910B 是 40
dtype = torch.bfloat16
dtype_id = 27

expertId = torch.randint(0, 256, (ROWS, per_count), dtype=torch.int32)
topk_logit = torch.rand(ROWS, per_count, dtype=dtype)

print("=== FusedExpert 算子测试 ===")
print(f"参数: ROWS={ROWS}, per_count={per_count}, shared_expert={shared_expert}, expertNum={expertNum}")

print("\n[CPU计算]")
fused_expertId_cpu, fused_topk_logit_cpu = fusedexpert_cpu(
    expertId, topk_logit, pos, expertNum, shared_expert, sharedId_dp, per_count, start_tokenidx, end_tokenidx)

print("[NPU计算]")
expertId_npu = expertId.npu()
topk_logit_npu = topk_logit.npu()
fused_expertId_npu = torch.empty((ROWS, per_count + shared_expert), dtype=torch.int32).npu()
fused_topk_logit_npu = torch.empty((ROWS, per_count + shared_expert), dtype=dtype).npu()

torch.ops.ascend_ops.fused_expert(
    block_dim, expertId_npu, fused_expertId_npu, topk_logit_npu, fused_topk_logit_npu,
    pos, expertNum, shared_expert, sharedId_dp, thread_count, ROWS, per_count, start_tokenidx, end_tokenidx, dtype_id)

fused_expertId_npu_cpu = fused_expertId_npu.cpu()
fused_topk_logit_npu_cpu = fused_topk_logit_npu.cpu()

print("\n[结果对比]")
expertId_match = torch.equal(fused_expertId_cpu, fused_expertId_npu_cpu)
logit_match = torch.allclose(fused_topk_logit_cpu, fused_topk_logit_npu_cpu, rtol=1e-3, atol=1e-3)

if expertId_match and logit_match:
    print("✓ CPU与NPU结果一致！")
else:
    print("✗ CPU与NPU结果不一致！")
    if not expertId_match:
        print(f"  ExpertId不一致数量: {(fused_expertId_cpu != fused_expertId_npu_cpu).sum().item()}")
    if not logit_match:
        print(f"  TopkLogit不匹配（使用allclose对比）")