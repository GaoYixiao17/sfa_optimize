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
namespace Mambav2ChunkScan {

class CubeHandler{
public:
    __aicore__ inline CubeHandler(){}
    __aicore__ inline void Init(GM_ADDR cmtx, GM_ADDR bmtx, GM_ADDR xmtx, GM_ADDR dmtx, GM_ADDR statesmtx, GM_ADDR dacsmtx, GM_ADDR dtoutmtx, GM_ADDR cbmtx, GM_ADDR mmtx, GM_ADDR outmtx, GM_ADDR sumoutmtx, GM_ADDR workspace, int B, int C, int H, int G, int L, int N, int P, int core_num){
        // workspace allocation
        int offset = 0;
        auto cb_ws = shiftAddr<float>(workspace, ((core_num * L) * L), offset);
        
        // kernel tiling
        tilingShapeCustCube(B, C, H, G, L, N, P, cube0shape);
        
        // kernel initialization
        cube0.Init(cmtx, bmtx, cb_ws, xmtx, mmtx, outmtx, cbmtx, cube0shape);
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
    __aicore__ inline void Init(GM_ADDR cmtx, GM_ADDR bmtx, GM_ADDR xmtx, GM_ADDR dmtx, GM_ADDR statesmtx, GM_ADDR dacsmtx, GM_ADDR dtoutmtx, GM_ADDR cbmtx, GM_ADDR mmtx, GM_ADDR outmtx, GM_ADDR sumoutmtx, GM_ADDR workspace, int B, int C, int H, int G, int L, int N, int P, int core_num){
        // workspace allocation
        int offset = 0;
        auto cb_ws = shiftAddr<float>(workspace, ((core_num * L) * L), offset);
        
        // kernel tiling
        tilingShapeCustVec(B, C, H, G, L, N, P, vec0shape);
        
        // kernel initialization
        vec0.Init(cb_ws, dacsmtx, dtoutmtx, mmtx, outmtx, statesmtx, xmtx, dmtx, sumoutmtx, vec0shape);
    }
    
    __aicore__ inline void Compute(){
        vec0.Compute();
    }

private:
    TPipe pipe;
    CustVec vec0;
    CustVecShapeInfo vec0shape;
};


__global__ __aicore__ void kernel_cust_chunk_scan(GM_ADDR cmtx, GM_ADDR bmtx, GM_ADDR xmtx, GM_ADDR dmtx, GM_ADDR statesmtx, GM_ADDR dacsmtx, GM_ADDR dtoutmtx, GM_ADDR cbmtx, GM_ADDR mmtx, GM_ADDR outmtx, GM_ADDR sumoutmtx, GM_ADDR workspace, int B, int C, int H, int G, int L, int N, int P, int core_num){
    if ASCEND_IS_AIC{
        CubeHandler cube;
        cube.Init(cmtx, bmtx, xmtx, dmtx, statesmtx, dacsmtx, dtoutmtx, cbmtx, mmtx, outmtx, sumoutmtx, workspace, B, C, H, G, L, N, P, core_num);
        cube.Compute();
        mad((__cc__ float*)0, (__ca__ float*)0, (__cb__ float*)0, 32, 32, 32, 0x0, false, false, true);  // dummy instruction to make compiler happy
    }
    if ASCEND_IS_AIV{
        VecHandler vec;
        vec.Init(cmtx, bmtx, xmtx, dmtx, statesmtx, dacsmtx, dtoutmtx, cbmtx, mmtx, outmtx, sumoutmtx, workspace, B, C, H, G, L, N, P, core_num);
        vec.Compute();
        copy_ubuf_to_ubuf((__ubuf__ float*)0, (__ubuf__ float*)0, 0, 1, 1, 0, 0);  // dummy instruction to make compiler happy
    }
}

at::Tensor mambav2_chunk_scan(
    const at::Tensor &ct, 
    const at::Tensor &bt, 
    const at::Tensor &xt, 
    const at::Tensor &dt,
    const at::Tensor &states, 
    const at::Tensor &dacs, 
    const at::Tensor &dtout
){
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t blockDims = 20;
    int devidx = xt.device().index();
    c10_npu::set_device(devidx); 
    int B = ct.size(0);
    int C = ct.size(1);
    int L = ct.size(2);
    int G = ct.size(3);
    int N = ct.size(4);
    
    int H = states.size(2);
    int P = states.size(4);

    int BASEH=8;
    int CBASEM=64;

    // === step 1: convert dtype to make sure data type is correct, may not be necessary ===
    at::Tensor ct_fp16 = ct.to(at::kHalf);
    at::Tensor bt_fp16 = bt.to(at::kHalf);
    at::Tensor xt_fp16 = xt.to(at::kHalf);
    at::Tensor dt_fp16 = dt.to(at::kHalf);
    at::Tensor states_fp32 = states.to(at::kFloat);
    at::Tensor dacs_fp32 = dacs.to(at::kFloat);
    at::Tensor dtout_fp32 = dtout.to(at::kFloat);

    // === step 2: stream ===
    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream(devidx);
    void* aclstream = stream.stream();

    // === initialize output ===
    at::Tensor out_cb = torch::empty({B,C,G,L,L}, xt.options().dtype(at::kFloat));
    at::Tensor out_mm = torch::empty({B,C,H,L,L}, xt.options().dtype(at::kHalf));
    at::Tensor outmtx = torch::empty({B,C,H,L,P}, xt.options().dtype(at::kFloat));
    at::Tensor final_out = torch::empty({B,C*L,H*P}, xt.options().dtype(at::kFloat));

    // === workspace ===
    auto userWorkspaceSize = blockDims * (L * L);
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    int64_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    auto workspaceTensor = at::empty({workspaceSize}, at::TensorOptions().dtype(at::kFloat).device(xt.options().device()));

    // === kernel ===
    auto acl_call = [=]() -> int {
        kernel_cust_chunk_scan<<<blockDims, nullptr, aclstream>>>(
            (GM_ADDR)(ct_fp16.storage().data()),
            (GM_ADDR)(bt_fp16.storage().data()),
            (GM_ADDR)(xt_fp16.storage().data()),
            (GM_ADDR)(dt_fp16.storage().data()),
            (GM_ADDR)(states_fp32.storage().data()),
            (GM_ADDR)(dacs_fp32.storage().data()),
            (GM_ADDR)(dtout_fp32.storage().data()),
            (GM_ADDR)(out_cb.storage().data()),
            (GM_ADDR)(out_mm.storage().data()),
            (GM_ADDR)(outmtx.storage().data()),
            (GM_ADDR)(final_out.storage().data()),
            (GM_ADDR)(workspaceTensor.storage().data()),
            B, C, H, G, L, N, P, blockDims
        );
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Mambav2ChunkScan", acl_call);
    return final_out;
}

torch::Tensor mambav2_chunk_scan_meta(
    const at::Tensor &ct, 
    const at::Tensor &bt, 
    const at::Tensor &xt, 
    const at::Tensor &dt,
    const at::Tensor &states, 
    const at::Tensor &dacs, 
    const at::Tensor &dtout
){
    TORCH_CHECK(ct.defined(), "Input tensor ct must be defined");
    return ct;
}

// Register Ascend implementaions for Mambav2ChunkScan
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("mambav2_chunk_scan", mambav2_chunk_scan);
}

// Register Meta Function for Mambav2ChunkScan
TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, Meta, m)
{
    m.impl("mambav2_chunk_scan", TORCH_FN(mambav2_chunk_scan_meta));
}

} // namespace Mambav2ChunkScan
} // namespace npu_ops_transformer_ext