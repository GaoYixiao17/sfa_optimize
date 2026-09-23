/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

#include "tiling/platform/platform_ascendc.h"
#include "op_kernel/CustVec.h"
#include "kernel_operator.h"

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkCumsum {

class CubeHandler{
public:
    __aicore__ inline CubeHandler(){}
    __aicore__ inline void Init(GM_ADDR at_mtx, GM_ADDR dt_mtx, GM_ADDR dtbias_mtx, GM_ADDR dtmask_mtx, GM_ADDR out_mtx, GM_ADDR out1_mtx, GM_ADDR out2_mtx, GM_ADDR workspace, int B, int C, int H, int L){
        // scalar computations
        
        // workspace allocation
        int offset = 0;
        
        // kernel tiling
        
        // kernel initialization
    }
    
    __aicore__ inline void Compute(){
    }

private:
    TPipe pipe;
};


class VecHandler{
public:
    __aicore__ inline VecHandler(){}
    __aicore__ inline void Init(GM_ADDR at_mtx, GM_ADDR dt_mtx, GM_ADDR dtbias_mtx, GM_ADDR dtmask_mtx, GM_ADDR out_mtx, GM_ADDR out1_mtx, GM_ADDR out2_mtx, GM_ADDR workspace, int B, int C, int H, int L){
        // scalar computations
        
        // workspace allocation
        int offset = 0;
        
        // kernel tiling
        tilingShapeCustVec(B, C, H, L, vec0shape);
        
        // kernel initialization
        vec0.Init(at_mtx, dt_mtx, dtbias_mtx, dtmask_mtx, out_mtx, out1_mtx, out2_mtx, vec0shape);
    }
    
    __aicore__ inline void Compute(){
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};


__global__ __aicore__ void kernel_cust_chunk_cumsum(GM_ADDR at_mtx, GM_ADDR dt_mtx, GM_ADDR dtbias_mtx, GM_ADDR dtmask_mtx, GM_ADDR out_mtx, GM_ADDR out1_mtx, GM_ADDR out2_mtx, GM_ADDR workspace, int B, int C, int H, int L){
    if ASCEND_IS_AIC{
        CubeHandler cube;
        cube.Init(at_mtx, dt_mtx, dtbias_mtx, dtmask_mtx, out_mtx, out1_mtx, out2_mtx, workspace, B, C, H, L);
        cube.Compute();
        mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy
    }
    if ASCEND_IS_AIV{
        VecHandler vec;
        vec.Init(at_mtx, dt_mtx, dtbias_mtx, dtmask_mtx, out_mtx, out1_mtx, out2_mtx, workspace, B, C, H, L);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy
    }
}  


std::tuple<at::Tensor, at::Tensor, at::Tensor> mambav2_chunk_cumsum(at::Tensor &at, at::Tensor &dt, at::Tensor &dt_bias, at::Tensor &dt_mask){
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDims = 20;
    int devidx = dt.device().index();
    c10_npu::set_device(devidx); 
    int B = dt.size(0);
    int C = dt.size(1);
    int L = dt.size(2);
    int H = dt.size(3);

    // === step 1: convert dtype to make sure the data type is correct, may not be necessary ===
    at::Tensor at_fp32 = at.to(at::kFloat);
    at::Tensor dt_fp16 = dt.to(at::kHalf);
    at::Tensor dt_bias_fp16 = dt_bias.to(at::kHalf);
    at::Tensor dt_mask_fp16 = dt_mask.to(at::kHalf);

    // === step 2: stream ===
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    // === initialize output ===
    auto opts = dt.options().dtype(torch::kFloat32);
    at::Tensor dt_out = torch::empty({B, C, L, H}, opts);
    at::Tensor dacs_out = torch::empty({B, C, L, H}, opts);
    at::Tensor dacs_chunk_out = torch::empty({B, C, 1, H}, opts);

    // === workspace ===
    auto userWorkspaceSize = 1024;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    int64_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto workspaceTensor = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(dt.options().device()));

    // === kernel ===
    auto acl_call = [=]() -> int {
        kernel_cust_chunk_cumsum<<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(at_fp32.storage().data()),
            (GM_ADDR)(dt_fp16.storage().data()),
            (GM_ADDR)(dt_bias_fp16.storage().data()),
            (GM_ADDR)(dt_mask_fp16.storage().data()),
            (GM_ADDR)(dt_out.storage().data()),
            (GM_ADDR)(dacs_out.storage().data()),
            (GM_ADDR)(dacs_chunk_out.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            B, C, H, L
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Mambav2ChunkCumsum", acl_call);
    return std::make_tuple(dt_out, dacs_out, dacs_chunk_out);
}

 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mambav2_chunk_cumsum_meta(at::Tensor &at, at::Tensor &dt, at::Tensor &dt_bias, at::Tensor &dt_mask)
{
    TORCH_CHECK(at.defined(), "Input tensor at must be defined");
    TORCH_CHECK(dt.defined(), "Input tensor dt must be defined");
    TORCH_CHECK(dt_bias.defined(), "Input tensor dt_bias must be defined");
    return std::make_tuple(at, dt, dt_bias);
}

// Register Ascend implementaions for Mambav2CausalConv1d
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("mambav2_chunk_cumsum", mambav2_chunk_cumsum);
}

// Register Meta Function for Mambav2CausalConv1d
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("mambav2_chunk_cumsum", TORCH_FN(mambav2_chunk_cumsum_meta));
}
} // namespace Mambav2ChunkCumsum
} // namespace npu_ops_transformer_ext