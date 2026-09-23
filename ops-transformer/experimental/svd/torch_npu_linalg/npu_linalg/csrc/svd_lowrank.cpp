/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <torch/library.h>
#include "ops_common.h"

namespace npu_linalg {
using namespace at_npu::native;

// torch.svd_lowrank: https://github.com/pytorch/pytorch/blob/main/torch/_lowrank.py#L64
std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_lowrank_npu(
    const at::Tensor& A,
    const c10::optional<at::Tensor>& M,
    const c10::optional<at::Tensor>& omega,
    int64_t q, int64_t niter)
{
    if (M.has_value()) TORCH_WARN("Argument 'M' is not supported.\n");

    auto aShape = A.sizes();
    int64_t nd = aShape.size();
    TORCH_CHECK(nd >= 2, "svd_lowrank: expected A with shape (*, m, n), got ndim=", nd);

    int64_t m = aShape[nd - 2];
    int64_t n = aShape[nd - 1];

    TORCH_CHECK(q >= 0 && q <= std::min(m, n),
                "svd_lowrank: q must be in [0, min(m,n)], got q=", q,
                ", m=", m, ", n=", n);

    auto opts = A.options();

    // random matrix generation
    at::Tensor G;
    if (omega.has_value()) {
        G = omega.value();
    } else {
        G = at::randn({n, q}, opts);
    }

    c10::SmallVector<int64_t, SIZE> yShape(aShape.begin(), aShape.end());
    yShape[nd - 1] = q;
    at::Tensor Y = at::empty(yShape, opts);

    c10::SmallVector<int64_t, SIZE> tempAShape(aShape.begin(), aShape.end());
    tempAShape[nd - 2] = q;
    at::Tensor tempA = at::empty(tempAShape, opts);

    // cubeMathType {0: KEEP_DTYPE, 1: ALLOW_FP32_DOWN_PRECISION, 2: USE_FP16,
    //               3: USE_HF32, 4: FORCE_GRP_ACC_FOR_FP32, 5: USE_HIGH_PREC_MODE}
    int64_t mm_type = 0;
    EXEC_NPU_CMD_V1(aclnnMatmul, A, G, Y, mm_type);

    auto tsqr_output_tensors = construct_tsqr_output_tensor(Y);
    at::Tensor Q  = std::get<0>(tsqr_output_tensors);
    at::Tensor R = std::get<1>(tsqr_output_tensors);
    int64_t block_size = 32;
    
    if (m > 1024) block_size = 1024;
    if (block_size < n) block_size = n * 2;
    EXEC_NPU_CMD_V1(aclnnTsqr, Y, block_size, Q, R);

    auto Qt = Q.transpose(-2, -1);
    EXEC_NPU_CMD_V1(aclnnMatmul, Qt, A, tempA, mm_type);

    auto svd_output_tensors = construct_svd_output_tensor(tempA);
    at::Tensor tempU  = std::get<0>(svd_output_tensors);
    at::Tensor S = std::get<1>(svd_output_tensors);
    at::Tensor V = std::get<2>(svd_output_tensors);
    int64_t num_iterations = 5;
    EXEC_NPU_CMD_V1(aclnnJacobi, tempA, num_iterations, tempU, S, V);

    at::Tensor U = at::empty(yShape, opts);
    auto Ut = tempU.transpose(-2, -1);
    EXEC_NPU_CMD_V1(aclnnMatmul, Q, Ut, U, mm_type);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(U, S, V);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
construct_svd_lowrank_output_tensor(const at::Tensor& A, int64_t q) {
    auto aShape = A.sizes();
    const int64_t nd = aShape.size();

    // U: (*, m, q)
    c10::SmallVector<int64_t, SIZE> uShape(aShape.begin(), aShape.end() - 1);
    uShape.push_back(q);

    // S: (*, q)
    c10::SmallVector<int64_t, SIZE> sShape(aShape.begin(), aShape.end() - 2);
    sShape.push_back(q);

    // V: (*, q, n)
    c10::SmallVector<int64_t, SIZE> vShape(aShape.begin(), aShape.end());
    vShape[nd - 2] = q;

    auto opts = A.options();

    at::Tensor U = at::empty(uShape, opts);
    at::Tensor S = at::empty(sShape, opts);
    at::Tensor V = at::empty(vShape, opts);

    return {U, S, V};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_lowrank_meta(
    const at::Tensor& A,
    const c10::optional<at::Tensor>& M,
    const c10::optional<at::Tensor>& omega,
    int64_t q, int64_t niter)
{
    // construct the output tensor
    auto output_tensors = construct_svd_lowrank_output_tensor(A, q);
    at::Tensor u = std::get<0>(output_tensors);
    at::Tensor s = std::get<1>(output_tensors);
    at::Tensor v = std::get<2>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(u, s, v);
}
} // namespace npu_linalg

TORCH_LIBRARY_IMPL(npu_linalg, PrivateUse1, m) {
    m.impl("svd_lowrank", &npu_linalg::svd_lowrank_npu);
}

TORCH_LIBRARY_IMPL(npu_linalg, Meta, m) {
    m.impl("svd_lowrank", &npu_linalg::svd_lowrank_meta);
}
