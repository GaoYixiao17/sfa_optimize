/**
* -*- coding: utf-8 -*-
* This file is a part of the CANN Open Software.
* Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
* BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
* Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
*/

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_blitz_sparse_attention.h"

const static int64_t PFA_SPARSE_HIGH_PRECISION_NO_MASK = 10;
const static int64_t PFA_SPARSE_HIGH_PRECISION_BAND = 14;

// Helper function to convert torch dtype to ACL dtype
aclDataType get_acl_dtype(const at::Tensor& tensor) {
    if (tensor.scalar_type() == at::kBFloat16) {
        return aclDataType::ACL_BF16;
    } else if (tensor.scalar_type() == at::kHalf) {
        return aclDataType::ACL_FLOAT16;
    } else if (tensor.scalar_type() == at::kFloat) {
        return aclDataType::ACL_FLOAT;
    } else if (tensor.scalar_type() == at::kUInt16) {
        return aclDataType::ACL_UINT16;
    } else if (tensor.scalar_type() == at::kBool) {
        return aclDataType::ACL_BOOL;
    }
    return aclDataType::ACL_DT_UNDEFINED;
}

// Helper function to create ACL tensor from torch tensor
aclTensor* create_acl_tensor(const at::Tensor& tensor) {
    auto sizes = tensor.sizes();
    std::vector<int64_t> shape(sizes.begin(), sizes.end());

    // Calculate strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    aclDataType dtype = get_acl_dtype(tensor);
    void* data_ptr = tensor.data_ptr();

    return aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                          aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), data_ptr);
}

/**
 * @brief Interface to the `aclnnBlitzSparseAttention` kernel which performs
 *        flash attention computation with prompt support and optional LSE output.
 *
 * @param [in] query Tensor of shape [B, N, S, D] (fp16/bf16)
 * @param [in] key Tensor of shape [B, N, T, D] (fp16/bf16)
 * @param [in] value Tensor of shape [B, N, T, D] (fp16/bf16)
 * @param [in] atten_mask Tensor of shape [B, N, S, T] (bool, optional)
 * @param [in] sabi Tensor of shape [B, N, ceil(S / BLOCK_SIZE_Q), kept_kv_tiles] (uint16, padded with 65535,
                    optional). Each entry indexes a BLOCK_SIZE_KV-wide KV chunk.
 * @param [in] block_shape pair [BLOCK_SIZE_Q, BLOCK_SIZE_KV]. Each value is one of {128, 256, 512, 1024}
                    all 16 combinations supported. Default: [128, 128].
 * @param [in] actual_seq_lengths vector of actual sequence lengths
 * @param [in] actual_seq_lengths_kv vector of actual sequence lengths for key/value
 * @param [in] num_heads number of attention heads
 * @param [in] scale_value scaling factor for attention
 * @param [in] pre_tokens number of previous tokens to consider
 * @param [in] next_tokens number of next tokens to consider
 * @param [in] input_layout output layout string
 * @param [in] num_key_value_heads number of key/value heads
 * @param [in] sparse_mode sparse mode flag
 * @param [in] softmax_lse_flag if true, compute and return softmax LSE (log-sum-exp)
 * @param [out] (first)  attention_out Tensor of shape [B, N, S, D] (fp16/bf16)
 * @param [out] (second) softmax_lse  Tensor of shape [B, N, S] (float32), empty when flag=false
 *
 * @returns tuple (attention_out, softmax_lse)
 */

std::tuple<at::Tensor, at::Tensor> blitz_sparse_attention(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::Tensor &value,
    const c10::optional<at::Tensor> &atten_mask = c10::nullopt,
    const c10::optional<at::Tensor> &sabi = c10::nullopt,
    c10::optional<std::vector<int64_t>> actual_seq_lengths_opt = c10::nullopt,
    c10::optional<std::vector<int64_t>> actual_seq_lengths_kv_opt = c10::nullopt,
    c10::optional<int64_t> num_heads_opt = c10::nullopt,
    c10::optional<float> scale_value_opt = c10::nullopt,
    c10::optional<int64_t> pre_tokens_opt = c10::nullopt,
    c10::optional<int64_t> next_tokens_opt = c10::nullopt,
    c10::optional<std::string> input_layout_opt = c10::nullopt,
    c10::optional<int64_t> num_key_value_heads_opt = c10::nullopt,
    c10::optional<int64_t> sparse_mode_opt = c10::nullopt,
    c10::optional<bool> softmax_lse_flag_opt = c10::nullopt,
    c10::optional<std::vector<int64_t>> block_shape_opt = c10::nullopt
) {
    // Set default values
    int64_t num_heads = num_heads_opt.value_or(1);
    float scale_value = scale_value_opt.value_or(1.0);
    int64_t pre_tokens = pre_tokens_opt.value_or(2147473647);
    int64_t next_tokens = next_tokens_opt.value_or(0);
    std::string input_layout = input_layout_opt.value_or("BSH");
    int64_t num_key_value_heads = num_key_value_heads_opt.value_or(0);
    int64_t sparse_mode = sparse_mode_opt.value_or(0);
    bool softmax_lse_flag = softmax_lse_flag_opt.value_or(false);
    std::vector<int64_t> block_shape = block_shape_opt.value_or(std::vector<int64_t>{128, 128});
    TORCH_CHECK(block_shape.size() == 2,
                "block_shape must be a length-2 list [BLOCK_SIZE_Q, BLOCK_SIZE_KV]");
    TORCH_CHECK(block_shape[0] == 128 || block_shape[0] == 256 ||
                block_shape[0] == 512 || block_shape[0] == 1024,
                "block_shape[0] (BLOCK_SIZE_Q) must be 128, 256, 512, or 1024");
    TORCH_CHECK(block_shape[1] == 128 || block_shape[1] == 256 ||
                block_shape[1] == 512 || block_shape[1] == 1024,
                "block_shape[1] (BLOCK_SIZE_KV) must be 128, 256, 512, or 1024");

    // Validate input tensors
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");

    if (atten_mask.has_value()) {
        TORCH_CHECK(atten_mask.value().scalar_type() == at::kBool, "atten_mask must be bool tensor");
    }

    if (sabi.has_value()) {
        TORCH_CHECK(sabi.value().scalar_type() == at::kUInt16, "sabi tensor must be uint16 tensor");
        TORCH_CHECK(sabi.value().is_contiguous(), "sabi must be contiguous");
    }

    // Create ACL tensors from torch tensors
    aclTensor* query_tensor = create_acl_tensor(query);
    aclTensor* key_tensor = create_acl_tensor(key);
    aclTensor* value_tensor = create_acl_tensor(value);
    aclTensor* atten_mask_tensor = nullptr;
    if (atten_mask.has_value()) {
        atten_mask_tensor = create_acl_tensor(atten_mask.value());
    }
    aclTensor* sabi_tensor = nullptr;
    if (sabi.has_value()) {
        sabi_tensor = create_acl_tensor(sabi.value());
    }

    // Create actual sequence lengths array - handle optional parameters
    aclIntArray* actual_seq_lengths_array = nullptr;
    aclIntArray* actual_seq_lengths_kv_array = nullptr;

    if (actual_seq_lengths_opt.has_value()) {
        const auto& seq_lengths = actual_seq_lengths_opt.value();
        actual_seq_lengths_array = aclCreateIntArray(seq_lengths.data(), seq_lengths.size());
    }

    if (actual_seq_lengths_kv_opt.has_value()) {
        const auto& seq_lengths_kv = actual_seq_lengths_kv_opt.value();
        actual_seq_lengths_kv_array = aclCreateIntArray(seq_lengths_kv.data(), seq_lengths_kv.size());
    }

    aclIntArray* block_shape_array = aclCreateIntArray(block_shape.data(), block_shape.size());

    // Prepare layerOut string
    std::vector<char> layer_out(input_layout.begin(), input_layout.end());
    layer_out.push_back('\0');

    // Create output tensor with same properties as query
    at::Tensor output = torch::empty_like(query);
    aclTensor* output_tensor = create_acl_tensor(output);

    // Allocate LSE output: shape [B, N, S] float32 when flag=true, empty otherwise.
    // Derive B/N/S from the query shape assuming BNSD (the only layout used on 910B2).
    // For other layouts the LSE will be empty regardless of the flag.
    at::Tensor lse_out;
    aclTensor* lse_tensor = nullptr;
    if (softmax_lse_flag) {
        TORCH_CHECK(input_layout == "BNSD" || input_layout == "BSH" || input_layout == "BSND",
                    "softmax_lse_flag=True only supported for BNSD/BSH/BSND layouts");
        int64_t B = 0;
        int64_t N = 0;
        int64_t S = 0;
        if (input_layout == "BNSD") {
            B = query.size(0);
            N = num_heads;
            S = query.size(2);
        } else if (input_layout == "BSND") {
            B = query.size(0);
            N = num_heads;
            S = query.size(1);
        } else {  // BSH
            B = query.size(0);
            N = num_heads;
            S = query.size(1);
        }
        lse_out = torch::empty({B, N, S},
                               torch::dtype(torch::kFloat32).device(query.device()));
        lse_tensor = create_acl_tensor(lse_out);
    } else {
        lse_out = torch::empty({0},
                               torch::dtype(torch::kFloat32).device(query.device()));
    }

    int64_t inner_precise = 1;

    if (sparse_mode >= PFA_SPARSE_HIGH_PRECISION_NO_MASK && sparse_mode <= PFA_SPARSE_HIGH_PRECISION_BAND) {
        // for sparse in range [10,14], set inner calculate mode to high-precision
        inner_precise = 0;
        sparse_mode -= PFA_SPARSE_HIGH_PRECISION_NO_MASK;
    }

    // Get workspace size
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    void* workspace_addr = nullptr;

    int ret = aclnnBlitzSparseAttentionGetWorkspaceSize(
        query_tensor,                 //  const aclTensor   *query,
        key_tensor,                   //  const aclTensor   *key,
        value_tensor,                 //  const aclTensor   *value,
        nullptr,                      //  const aclTensor   *pseShift,
        atten_mask_tensor,            //  const aclTensor   *attenMask,
        sabi_tensor,                  //  const aclTensor   *sabi,
        actual_seq_lengths_array,     //  const aclIntArray *actualSeqLengths,
        actual_seq_lengths_kv_array,  //  const aclIntArray *actualSeqLengthsKv,
        nullptr,                      //  const aclTensor   *deqScale1,
        nullptr,                      //  const aclTensor   *quantScale1,
        nullptr,                      //  const aclTensor   *deqScale2,
        nullptr,                      //  const aclTensor   *quantScale2,
        nullptr,                      //  const aclTensor   *quantOffset2,
        num_heads,                    //  int64_t            numHeads,
        scale_value,                  //  float              scaleValue,
        pre_tokens,                   //  int64_t            preTokens,
        next_tokens,                  //  int64_t            nextTokens,
        layer_out.data(),             //  char              *inputLayout,
        num_key_value_heads,          //  int64_t            numKeyValueHeads,
        sparse_mode,                  //  int64_t            sparseMode,
        inner_precise,                //  int64_t            innerPrecise,
        softmax_lse_flag,             //  bool               softmaxLseFlag,
        block_shape_array,            //  const aclIntArray *blockShape,
        output_tensor,                //  const aclTensor   *attentionOut,
        lse_tensor,                   //  const aclTensor   *softmaxLse,
        &workspace_size,              //  uint64_t          *workspaceSize,
        &executor);                   //  aclOpExecutor     **executor)

    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnBlitzSparseAttentionGetWorkspaceSize failed with error: ", ret);

    // Allocate workspace if needed
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        TORCH_CHECK(ret == ACL_SUCCESS, "aclrtMalloc workspace failed with error: ", ret);
    }

    // Get current NPU stream
    int deviceId;
    aclrtGetDevice(&deviceId);
    auto npuStream = c10_npu::getCurrentNPUStream(deviceId);
    auto aclStream = npuStream.stream(true);

    // Execute the kernel
    ret = aclnnBlitzSparseAttention(workspace_addr, workspace_size, executor, aclStream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclnnBlitzSparseAttention execution failed with error: ", ret);

    // Deallocation of workspace (first make sure it finished)
    ret = aclrtSynchronizeStream(aclStream);
    TORCH_CHECK(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed with error: ", ret);

    // Now safe to cleanup
    if (workspace_addr) aclrtFree(workspace_addr);
    if (actual_seq_lengths_array) aclDestroyIntArray(actual_seq_lengths_array);
    if (actual_seq_lengths_kv_array) aclDestroyIntArray(actual_seq_lengths_kv_array);
    if (block_shape_array) aclDestroyIntArray(block_shape_array);
    if (lse_tensor) aclDestroyTensor(lse_tensor);

    return std::make_tuple(output, lse_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blitz_sparse_attention", &blitz_sparse_attention,
        R"DOC(
        Interface to the `aclnnBlitzSparseAttention` kernel which performs
        flash attention computation with block-sparse support.

        Args:
            query (torch.Tensor): Query tensor of shape [B, N, S, D] (fp16/bf16)
            key (torch.Tensor): Key tensor of shape [B, N, T, D] (fp16/bf16)
            value (torch.Tensor): Value tensor of shape [B, N, T, D] (fp16/bf16)
            atten_mask (torch.Tensor, optional): Attention mask of shape [B, N, S, T] (bool)
            sabi (torch.Tensor, optional): Sparse Attention Block Indices,
                  shape [B, N, ceil(S / BLOCK_SIZE_Q), kept_kv_tiles] (uint16, padded with 65535).
                  Each entry indexes a BLOCK_SIZE_KV-wide KV chunk; pad unused slots with 0xFFFF.
            actual_seq_lengths (list, optional): Actual sequence lengths
            actual_seq_lengths_kv (list, optional): Actual sequence lengths for key/value
            num_heads (int, optional): Number of attention heads. Default: 1
            scale_value (float, optional): Scaling factor for attention. Default: 1.0
            pre_tokens (int, optional): Number of previous tokens to consider. Default: 2147473647
            next_tokens (int, optional): Number of next tokens to consider. Default: 0
            input_layout (str, optional): Output layout string. Default: "BSH"
            num_key_value_heads (int, optional): Number of key/value heads. Default: 0
            sparse_mode (int, optional): sparse mode = 0,1,2,...7 choice. Default: 0
            softmax_lse_flag (bool, optional): If True, compute softmax LSE output. Default: False
            block_shape (list[int], optional): [BLOCK_SIZE_Q, BLOCK_SIZE_KV] sabi-index granularity.
                        Each value is one of {128, 256, 512, 1024}; all 16 combinations supported. Default: [128, 128].

        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - attention_out: shape [B, N, S, D], same dtype as query
                - softmax_lse:   shape [B, N, S], float32 (empty tensor when softmax_lse_flag=False)
        )DOC",
        py::arg("query"),
        py::arg("key"),
        py::arg("value"),
        py::arg("atten_mask") = c10::optional<at::Tensor>(),
        py::arg("sabi") = c10::optional<at::Tensor>(),
        py::arg("actual_seq_lengths") = c10::optional<std::vector<int64_t>>(),
        py::arg("actual_seq_lengths_kv") = c10::optional<std::vector<int64_t>>(),
        py::arg("num_heads") = c10::optional<int64_t>(),
        py::arg("scale_value") = c10::optional<float>(),
        py::arg("pre_tokens") = c10::optional<int64_t>(),
        py::arg("next_tokens") = c10::optional<int64_t>(),
        py::arg("input_layout") = c10::optional<std::string>(),
        py::arg("num_key_value_heads") = c10::optional<int64_t>(),
        py::arg("sparse_mode") = c10::optional<int64_t>(),
        py::arg("softmax_lse_flag") = c10::optional<bool>(),
        py::arg("block_shape") = c10::optional<std::vector<int64_t>>()
    );
}
