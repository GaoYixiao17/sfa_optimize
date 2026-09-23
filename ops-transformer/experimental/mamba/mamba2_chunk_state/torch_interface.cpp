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
#include "op_kernel/CustCube.h"
#include "kernel_operator.h"

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkState {

class CubeHandler{
public:
    __aicore__ inline CubeHandler(){}
    __aicore__ inline void Init(GM_ADDR dtout_mtx, GM_ADDR dacs_mtx, GM_ADDR bt_mtx, GM_ADDR xt_mtx, GM_ADDR out_mtx, GM_ADDR workspace, int B, int C, int H, int G, int L, int N, int P, int core_num){
        // workspace allocation
        int offset = 0;
        auto da_out = shiftAddr<float>(workspace, ((core_num * L) * 8), offset);
        auto vec_out = shiftAddr<half>(workspace, ((((core_num * 8) * L) * 64) * 3), offset);
        
        // kernel tiling
        tilingShapeCustCube(B, C, H, G, L, N, P, cube0shape);
        
        // kernel initialization
        cube0.Init(vec_out, xt_mtx, out_mtx, cube0shape);
    }
    
    __aicore__ inline void Compute(){
        cube0.Compute();
    }

private:
    TPipe pipe;
    CustCube cube0;
    CustCubeShapeInfo cube0shape;
};


class VecHandler{
public:
    __aicore__ inline VecHandler(){}
    __aicore__ inline void Init(GM_ADDR dtout_mtx, GM_ADDR dacs_mtx, GM_ADDR bt_mtx, GM_ADDR xt_mtx, GM_ADDR out_mtx, GM_ADDR workspace, int B, int C, int H, int G, int L, int N, int P, int core_num){
        // workspace allocation
        int offset = 0;
        auto da_out = shiftAddr<float>(workspace, ((core_num * L) * 8), offset);
        auto vec_out = shiftAddr<half>(workspace, ((((core_num * 8) * L) * 64) * 3), offset);
        
        // kernel tiling
        tilingShapeCustVec(B, C, H, G, L, N, vec0shape);
        
        // kernel initialization
        vec0.Init(da_out, dtout_mtx, dacs_mtx, bt_mtx, vec_out, vec0shape);
    }
    
    __aicore__ inline void Compute(){
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};


__global__ __aicore__ void kernel_cust_chunk_state(GM_ADDR dtout_mtx, GM_ADDR dacs_mtx, GM_ADDR bt_mtx, GM_ADDR xt_mtx, GM_ADDR out_mtx, GM_ADDR workspace, int B, int C, int H, int G, int L, int N, int P, int core_num){
    if ASCEND_IS_AIC{
        CubeHandler cube;
        cube.Init(dtout_mtx, dacs_mtx, bt_mtx, xt_mtx, out_mtx, workspace, B, C, H, G, L, N, P, core_num);
        cube.Compute();
        mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy
    }
    if ASCEND_IS_AIV{
        VecHandler vec;
        vec.Init(dtout_mtx, dacs_mtx, bt_mtx, xt_mtx, out_mtx, workspace, B, C, H, G, L, N, P, core_num);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy
    }
}

at::Tensor mambav2_chunk_state(const at::Tensor &dt_out, const at::Tensor &dacs, const at::Tensor &bt, const at::Tensor &xt){
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDims = 20;
    int devidx = xt.device().index();
    c10_npu::set_device(devidx); 
    int B = xt.size(0);
    int C = xt.size(1);
    int L = xt.size(2);
    int H = xt.size(3);
    int P = xt.size(4);
    
    int G = bt.size(3);
    int N = bt.size(4);

    int BASEH=8;
    int CBASEM=64;

    // === step 1: convert dtype to make sure data type is correct, may not be necessary ===
    at::Tensor dtout_fp32 = dt_out.to(at::kFloat);
    at::Tensor dacs_fp32 = dacs.to(at::kFloat);
    at::Tensor bt_fp16 = bt.to(at::kHalf);
    at::Tensor xt_fp16 = xt.to(at::kHalf);

    // === step 2: stream ===
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    // === initialize output ===
    at::Tensor output = torch::empty({B,C,H,N,P}, xt.options().dtype(at::kFloat));

    // === workspace ===
    auto userWorkspaceSize = blockDims * (L * BASEH + BASEH * L * CBASEM * 3);
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    int64_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto workspaceTensor = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kFloat).device(xt.options().device()));

    // === kernel ===
    auto acl_call = [=]() -> int {
        kernel_cust_chunk_state<<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(dtout_fp32.storage().data()),
            (GM_ADDR)(dacs_fp32.storage().data()),
            (GM_ADDR)(bt_fp16.storage().data()),
            (GM_ADDR)(xt_fp16.storage().data()),
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            B, C, H, G, L, N, P, blockDims
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Mambav2ChunkState", acl_call);
    return output;
}

torch::Tensor mambav2_chunk_state_meta(const at::Tensor &dt_out, const at::Tensor &dacs, const at::Tensor &bt, const at::Tensor &xt)
{
    TORCH_CHECK(dt_out.defined(), "Input tensor dt_out must be defined");
    return dt_out;
}

// Register Ascend implementaions for Mambav2CausalConv1d
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("mambav2_chunk_state", mambav2_chunk_state);
}

// Register Meta Function for Mambav2CausalConv1d
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("mambav2_chunk_state", TORCH_FN(mambav2_chunk_state_meta));
}

} // Mambav2ChunkState
} // npu_ops_transformer_ext