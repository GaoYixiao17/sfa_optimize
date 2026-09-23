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

std::tuple<at::Tensor, at::Tensor> construct_tsqr_output_tensor(const at::Tensor& x)
{
    at::SmallVector<int64_t, SIZE> qSize(x.sizes().begin(), x.sizes().end());
    at::SmallVector<int64_t, SIZE> rSize(x.sizes().begin(), x.sizes().end());
    rSize[x.sizes().size() - 2] = rSize[x.sizes().size() - 1];
    at::Tensor Q = at::empty(qSize, x.options().dtype(at::kFloat));
    at::Tensor R = at::empty(rSize, x.options().dtype(at::kFloat));

    return std::tuple<at::Tensor, at::Tensor>(Q, R);
}

std::tuple<at::Tensor, at::Tensor> tsqr_npu(const at::Tensor& A, const c10::optional<int64_t>  block_size)
{
    int64_t blk_size = c10::value_or_else(block_size, [] { return 0; });

    // construct the output tensor
    auto output_tensors = construct_tsqr_output_tensor(A);
    at::Tensor Q  = std::get<0>(output_tensors);
    at::Tensor R = std::get<1>(output_tensors);

    EXEC_NPU_CMD_V1(aclnnTsqr, A, blk_size, Q, R);
    return std::tuple<at::Tensor, at::Tensor>(Q, R);
}

std::tuple<at::Tensor, at::Tensor> tsqr_meta(const at::Tensor& A, const c10::optional<int64_t> block_size)
{
    // construct the output tensor
    auto output_tensors = construct_tsqr_output_tensor(A);
    at::Tensor Q = std::get<0>(output_tensors);
    at::Tensor R = std::get<1>(output_tensors);

    return std::tuple<at::Tensor, at::Tensor>(Q, R);
}
} // namespace npu_linalg


TORCH_LIBRARY_IMPL(npu_linalg, PrivateUse1, m) {
    m.impl("tsqr", &npu_linalg::tsqr_npu);
}


TORCH_LIBRARY_IMPL(npu_linalg, Meta, m) {
    m.impl("tsqr", &npu_linalg::tsqr_meta);
}
