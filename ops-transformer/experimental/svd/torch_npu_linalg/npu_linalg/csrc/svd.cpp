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

std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_svd_output_tensor(const at::Tensor& x)
{
    auto xShape = x.sizes();
    int64_t nd = xShape.size();
    int64_t mSize = xShape[nd - 2];
    int64_t nSize = xShape[nd - 1];
    int64_t minDim = std::min(mSize, nSize);

    at::SmallVector<int64_t, SIZE> uShape(xShape.begin(), xShape.end() - 1);
    uShape.push_back(minDim);
    at::SmallVector<int64_t, SIZE> sShape(xShape.begin(), xShape.end() - 2);
    sShape.push_back(minDim);
    at::SmallVector<int64_t, SIZE> vShape(xShape.begin(), xShape.end());
    vShape[nd - 2] = minDim;

    at::Tensor U = at::empty(uShape, x.options().dtype(at::kFloat));
    at::Tensor S = at::empty(sShape, x.options().dtype(at::kFloat));
    at::Tensor V = at::empty(vShape, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(U, S, V);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_npu(const at::Tensor& A, int64_t num_iterations)
{
    // construct the output tensor
    auto output_tensors = construct_svd_output_tensor(A);
    at::Tensor U  = std::get<0>(output_tensors);
    at::Tensor S = std::get<1>(output_tensors);
    at::Tensor V = std::get<2>(output_tensors);

    EXEC_NPU_CMD_V1(aclnnJacobi, A, num_iterations, U, S, V);
    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(U, S, V);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> svd_meta(const at::Tensor& A, int64_t num_iterations)
{
    // construct the output tensor
    auto output_tensors = construct_svd_output_tensor(A);
    at::Tensor U  = std::get<0>(output_tensors);
    at::Tensor S = std::get<1>(output_tensors);
    at::Tensor V = std::get<2>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor>(U, S, V);
}
} // namespace npu_linalg


TORCH_LIBRARY_IMPL(npu_linalg, PrivateUse1, m) {
    m.impl("svd", &npu_linalg::svd_npu);
}


TORCH_LIBRARY_IMPL(npu_linalg, Meta, m) {
    m.impl("svd", &npu_linalg::svd_meta);
}
