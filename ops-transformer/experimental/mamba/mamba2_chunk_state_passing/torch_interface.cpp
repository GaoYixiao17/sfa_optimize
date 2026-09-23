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
namespace Mambav2ChunkStatePassing {

class CubeHandler{
public:
    __aicore__ inline CubeHandler(){}
    __aicore__ inline void Init(GM_ADDR dacs, GM_ADDR initmtx, GM_ADDR statemtx, GM_ADDR cmtx, GM_ADDR final_state, GM_ADDR out_bmm, GM_ADDR workspace, int B, int H, int S, int C, int L, int G, int N, int P, int Z, int core_num){
        // workspace allocation
        int offset = 0;
        auto state_fp16 = shiftAddr<half>(workspace, ((core_num * Z) * 3), offset);
        auto tmpmtx = shiftAddr<float>(workspace, ((B * H) * Z), offset);
        
        // kernel tiling
        tilingShapeCustCube(B, H, S, C, L, G, N, P, Z, cube0shape);
        
        // kernel initialization
        cube0.Init(cmtx, state_fp16, out_bmm, cube0shape);
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
    __aicore__ inline void Init(GM_ADDR dacs, GM_ADDR initmtx, GM_ADDR statemtx, GM_ADDR cmtx, GM_ADDR final_state, GM_ADDR out_bmm, GM_ADDR workspace, int B, int H, int S, int C, int L, int G, int N, int P, int Z, int core_num){
        // workspace allocation
        int offset = 0;
        auto state_fp16 = shiftAddr<half>(workspace, ((core_num * Z) * 3), offset);
        auto tmpmtx = shiftAddr<float>(workspace, ((B * H) * Z), offset);
        
        // kernel tiling
        tilingShapeCustVec(B, H, S, C, L, G, N, P, Z, vec0shape);
        
        // kernel initialization
        vec0.Init(dacs, initmtx, statemtx, tmpmtx, state_fp16, final_state, vec0shape);
    }
    
    __aicore__ inline void Compute(){
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};


__global__ __aicore__ void kernel_cust_chunk_state_passing(GM_ADDR dacs, GM_ADDR initmtx, GM_ADDR statemtx, GM_ADDR cmtx, GM_ADDR final_state, GM_ADDR out_bmm, GM_ADDR workspace, int B, int H, int S, int C, int L, int G, int N, int P, int Z, int core_num){
    if ASCEND_IS_AIC{
        CubeHandler cube;
        cube.Init(dacs, initmtx, statemtx, cmtx, final_state, out_bmm, workspace, B, H, S, C, L, G, N, P, Z, core_num);
        cube.Compute();
        mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy
    }
    if ASCEND_IS_AIV{
        VecHandler vec;
        vec.Init(dacs, initmtx, statemtx, cmtx, final_state, out_bmm, workspace, B, H, S, C, L, G, N, P, Z, core_num);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy
    }
}

std::tuple<at::Tensor, at::Tensor> mambav2_chunk_state_passing(const at::Tensor &dacs, const at::Tensor &init_states, const at::Tensor &states, const at::Tensor &ct){
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDims = 20;
    int devidx = states.device().index();
    c10_npu::set_device(devidx); 
    int B = ct.size(0);
    int C = ct.size(1);
    int L = ct.size(2);
    int G = ct.size(3);
    int N = ct.size(4);
    int H = states.size(2);
    int Z = states.size(3);
    int P = Z / N;
    int S = C * L; 

    // === step 1: convert dtype to make sure data type is correct, may not be necessary ===
    at::Tensor dacs_fp32 = dacs.to(at::kFloat);
    at::Tensor initmtx_fp32 = init_states.to(at::kFloat);
    at::Tensor states_fp32 = states.to(at::kFloat);
    at::Tensor ct_fp16 = ct.to(at::kHalf);

    // === step 2: stream ===
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    // === initialize output ===
    at::Tensor final_state = torch::empty({B,H,N,P}, states.options().dtype(at::kFloat));
    at::Tensor output = torch::empty({B,C,H,L,P}, states.options().dtype(at::kFloat));

    // === workspace ===
    auto userWorkspaceSize = B*H*Z + blockDims*Z*3;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    int64_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto workspaceTensor = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kFloat).device(states.options().device()));

    // === kernel ===
    auto acl_call = [=]() -> int {
        kernel_cust_chunk_state_passing<<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(dacs_fp32.storage().data()),
            (GM_ADDR)(initmtx_fp32.storage().data()),
            (GM_ADDR)(states_fp32.storage().data()),
            (GM_ADDR)(ct_fp16.storage().data()),
            (GM_ADDR)(final_state.storage().data()),
            (GM_ADDR)(output.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            B, H, S, C, L, G, N, P, Z, blockDims
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Mambav2ChunkStatePassing", acl_call);
    return std::make_tuple(output, final_state);
}

std::tuple<torch::Tensor, torch::Tensor> mambav2_chunk_state_passing_meta(const at::Tensor &dacs, const at::Tensor &init_states, const at::Tensor &states, const at::Tensor &ct)
{
    TORCH_CHECK(dacs.defined(), "Input tensor at must be defined");
    TORCH_CHECK(states.defined(), "Input tensor dt must be defined");
    return std::make_tuple(dacs, states);
}

// Register Ascend implementaions for Mambav2ChunkStatePassing
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("mambav2_chunk_state_passing", mambav2_chunk_state_passing);
}

// Register Meta Function for Mambav2ChunkStatePassing
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("mambav2_chunk_state_passing", TORCH_FN(mambav2_chunk_state_passing));
}

} // namespace Mambav2ChunkStatePassing
} // namespace npu_ops_transformer_ext