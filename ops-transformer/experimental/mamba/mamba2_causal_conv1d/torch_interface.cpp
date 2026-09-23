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
namespace Mambav2CausalConv1d {

class CubeHandler{
public:
    __aicore__ inline CubeHandler(){}
    __aicore__ inline void Init(GM_ADDR xmtx, GM_ADDR wmtx, GM_ADDR bias, GM_ADDR outmtx, GM_ADDR workspace, int B, int D, int S, int W){
        // workspace allocation
        
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
    __aicore__ inline void Init(GM_ADDR xmtx, GM_ADDR wmtx, GM_ADDR bias, GM_ADDR outmtx, GM_ADDR workspace, int B, int D, int S, int W){
        // workspace allocation
        
        // kernel tiling
        tilingShapeCustVec(B, D, S, W, vec0shape);
        
        // kernel initialization
        vec0.Init(xmtx, wmtx, bias, outmtx, vec0shape);
    }
    
    __aicore__ inline void Compute(){
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};

template <typename T>
__global__ __aicore__ void kernel_cust_causal_conv1d(GM_ADDR xmtx, GM_ADDR wmtx, GM_ADDR bias, GM_ADDR outmtx, GM_ADDR workspace, int B, int D, int S, int W){
    if ASCEND_IS_AIC{
        CubeHandler cube;
        cube.Init(xmtx, wmtx, bias, outmtx, workspace, B, D, S, W);
        cube.Compute();
        mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy
    }
    if ASCEND_IS_AIV{
        VecHandler vec;
        vec.Init(xmtx, wmtx, bias, outmtx, workspace, B, D, S, W);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy
    }
}

at::Tensor mambav2_causal_conv1d(at::Tensor &x, at::Tensor &w, at::Tensor &bias){
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDims = 20;
    int devidx = x.device().index();
    c10_npu::set_device(devidx); 
    int B = x.size(0);
    int D = x.size(1);
    int S = x.size(2);
    int W = w.size(-1);

    // === step 1: convert dtype ===
    at::Tensor x_fp32 = x.to(at::kFloat);
    at::Tensor w_fp32 = w.to(at::kFloat);
    at::Tensor bias_fp16 = bias.to(at::kHalf);

    // === step 2: stream ===
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    // === initialize output ===
    at::Tensor output = torch::empty(x_fp32.sizes(), x_fp32.options());

    // === workspace ===
    auto userWorkspaceSize = 1024;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    int64_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto workspaceTensor = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kByte).device(x.options().device()));

    // === kernel ===
    auto acl_call = [=]() -> int {
        kernel_cust_causal_conv1d<bfloat16_t><<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(x_fp32.storage().data()),
            (GM_ADDR)(w_fp32.storage().data()),
            (GM_ADDR)(bias_fp16.storage().data()),
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            B, D, S, W
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Mambav2CausalConv1d", acl_call);
    return output;
}

torch::Tensor mambav2_causal_conv1d_meta(at::Tensor &x, at::Tensor &w, at::Tensor &bias)
{
    TORCH_CHECK(x.defined(), "Input tensor must be defined");
    return x;
}

// Register Ascend implementaions for Mambav2CausalConv1d
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("mambav2_causal_conv1d", mambav2_causal_conv1d);
}

// Register Meta Function for Mambav2CausalConv1d
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("mambav2_causal_conv1d", TORCH_FN(mambav2_causal_conv1d_meta));
}

} // namespace Mambav2CausalConv1d
} // namespace npu_ops_transformer_ext