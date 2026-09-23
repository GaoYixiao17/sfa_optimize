/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

namespace npu_ops_transformer_ext {
namespace BiasSigmoid {
#include <iostream>
#include <stdio.h>
#include "kernel_operator.h"
#include "dtype_convert.h"
using namespace AscendC;

constexpr int BUFFER_NUM = 2;
#define UB_BUFFER_SIZE 188416
template <typename T>
class bias_sigmoid {
public:
    __aicore__ inline bias_sigmoid()
    {
    }
    __aicore__ inline void Init(const GM_ADDR expertX, GM_ADDR sigmoidOut, GM_ADDR biasOut, GM_ADDR expertBias,
                                int32_t ROWS, int32_t per_count, int32_t block_cnt)
    {
        BASE_SIZE = 32 / sizeof(T);
        BASE_SIZE_FP32 = 32 / sizeof(float);
        core_id_ = GetBlockIdx();
        TILE_LENGTH = per_count;
        per_dim_ = per_count;
        block_cnt_ = block_cnt;

        int32_t rows_per_core = int(ROWS / block_cnt_);
        rows_per_core_per_loop_ = 1;

        align_cnt = DivCeil(per_dim_, BASE_SIZE) * BASE_SIZE;

        if (rows_per_core >= 2) {
            int32_t limit_rows_per_core = UB_BUFFER_SIZE / (align_cnt * BUFFER_NUM * (3 * sizeof(T) + sizeof(float)));
            rows_per_core_per_loop_ = min(rows_per_core, limit_rows_per_core);
        }

        tail_rows_last_loop_per_core_ = rows_per_core_per_loop_;

        /**计算loop_count**/
        int32_t loop_count = ROWS / (rows_per_core_per_loop_ * block_cnt_);

        full_loop_count_ = loop_count;
        full_rows_offset_ = full_loop_count_ * rows_per_core_per_loop_ * block_cnt_ * TILE_LENGTH;
        rows_offset_ = 0;
        int32_t tail_count = ROWS - (full_loop_count_ * block_cnt_ * rows_per_core_per_loop_);
        tail_rows_last_loop_per_core_ = tail_count / block_cnt_;

        tail_count = tail_count % block_cnt_;
        tail_rows_offset_ = tail_count * (tail_rows_last_loop_per_core_ + 1) * TILE_LENGTH;
        if (core_id_ < tail_count || tail_rows_last_loop_per_core_) {
            tail_rows_last_loop_per_core_ += 1;
            loop_count += 1;
        }

        loop_count_ = loop_count;
        tail_count_ = tail_count;

        expertX_Gm.SetGlobalBuffer((__gm__ T *)expertX, rows_per_core_per_loop_ * align_cnt);
        sigmoidOut_Gm.SetGlobalBuffer((__gm__ T *)sigmoidOut, rows_per_core_per_loop_ * align_cnt);
        biasOut_Gm.SetGlobalBuffer((__gm__ T *)biasOut, rows_per_core_per_loop_ * align_cnt);
        expertBias_Gm.SetGlobalBuffer((__gm__ T *)expertBias, align_cnt);

        pipe.InitBuffer(inQueue_expertX, BUFFER_NUM, rows_per_core_per_loop_ * align_cnt * sizeof(T));
        pipe.InitBuffer(inQueue_expertBias, BUFFER_NUM, align_cnt * sizeof(T));
        pipe.InitBuffer(outQueue_expertXout, BUFFER_NUM, rows_per_core_per_loop_ * align_cnt * sizeof(T));
        pipe.InitBuffer(outQueue_biasXout, 1, rows_per_core_per_loop_ * align_cnt * sizeof(T) * 2);

        pipe.InitBuffer(work_buf2, rows_per_core_per_loop_ * align_cnt * sizeof(float));

        pipe.InitBuffer(work_buf1, align_cnt * sizeof(float));
        pipe.InitBuffer(work_buf3, rows_per_core_per_loop_ * align_cnt * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> expertBiasLocal = inQueue_expertBias.AllocTensor<T>();
        DataCopyParams copy_pas{1, (uint16_t)(TILE_LENGTH * sizeof(T)), 0, 0};
        DataCopyPadParams pad_pas;
        DataCopyPad(expertBiasLocal, expertBias_Gm, copy_pas, pad_pas);
        inQueue_expertBias.EnQue(expertBiasLocal);
        expertBiasLocal = inQueue_expertBias.DeQue<T>();
        LocalTensor<float> expertBiasfp32Local = work_buf1.Get<float>();
        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<float> expertBiasLocal_fp32 = expertBiasLocal.template ReinterpretCast<float>();
            Duplicate<float>(expertBiasfp32Local, (float)0.0, align_cnt);
            Adds(expertBiasfp32Local, expertBiasLocal_fp32, (float)(0.0), TILE_LENGTH);
        } else {
            Cast(expertBiasfp32Local, expertBiasLocal, RoundMode::CAST_NONE, align_cnt);
        }
        inQueue_expertBias.FreeTensor(expertBiasLocal);

        broadcast_biasfp32Local_ = work_buf3.Get<float>();
        uint32_t srcshape[2] = {(uint32_t)1, (uint32_t)align_cnt};
        uint32_t dstshape[2] = {(uint32_t)rows_per_core_per_loop_, (uint32_t)align_cnt};
        Broadcast<float, 2, 0>(broadcast_biasfp32Local_, expertBiasfp32Local, dstshape, srcshape);
        work_buf1.FreeTensor(expertBiasfp32Local);

        for (int i = 0; i < loop_count_; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }

        work_buf3.FreeTensor(broadcast_biasfp32Local_);
    }

private:
    __aicore__ inline void CopyIn(int32_t loop_cnt)
    {
        global_core_index_ = loop_cnt * block_cnt_ + core_id_;
        if (loop_cnt < full_loop_count_) {
            rows_offset_ = global_core_index_ * rows_per_core_per_loop_ * TILE_LENGTH;
        } else {
            rows_per_core_per_loop_ = tail_rows_last_loop_per_core_;

            if (core_id_ < tail_count_) {
                rows_offset_ = full_rows_offset_ + core_id_ * rows_per_core_per_loop_ * TILE_LENGTH;
            } else {
                rows_offset_ = full_rows_offset_ + tail_rows_offset_ +
                               (core_id_ - tail_count_) * rows_per_core_per_loop_ * TILE_LENGTH;
            }
        }

        LocalTensor<T> expertXLocal = inQueue_expertX.AllocTensor<T>();

        DataCopyParams copy_pas{(uint16_t)rows_per_core_per_loop_, (uint16_t)(TILE_LENGTH * sizeof(T)), 0, 0};
        DataCopyPadParams pad_pas{true, 0, uint8_t(align_cnt - TILE_LENGTH), 0};
        DataCopyPad(expertXLocal, expertX_Gm[rows_offset_], copy_pas, pad_pas);

        inQueue_expertX.EnQue(expertXLocal);
    }

    __aicore__ inline void Compute(int32_t loop_cnt)
    {
        LocalTensor<T> expertXLocal = inQueue_expertX.DeQue<T>();
        LocalTensor<T> expertXoutLocal = outQueue_expertXout.AllocTensor<T>();
        LocalTensor<T> biasXoutLocal = outQueue_biasXout.AllocTensor<T>();
        LocalTensor<float> sigmoidOutLocal = work_buf2.Get<float>();

        LocalTensor<float> expertXfp32Local = biasXoutLocal.template ReinterpretCast<float>();

        if constexpr (std::is_same<T, float>::value) {
            LocalTensor<float> expertXLocal_fp32 = expertXLocal.template ReinterpretCast<float>();
            LocalTensor<float> expertXoutLocal_fp32 = expertXoutLocal.template ReinterpretCast<float>();
            LocalTensor<float> biasXoutLocal_fp32 = biasXoutLocal.template ReinterpretCast<float>();
            Sigmoid(sigmoidOutLocal, expertXLocal_fp32, rows_per_core_per_loop_ * align_cnt);
            Adds(expertXoutLocal_fp32, sigmoidOutLocal, (float)(0.0), rows_per_core_per_loop_ * align_cnt);
            Add(biasXoutLocal_fp32, sigmoidOutLocal, broadcast_biasfp32Local_, rows_per_core_per_loop_ * align_cnt);
        } else {
            Cast(expertXfp32Local, expertXLocal, RoundMode::CAST_NONE, rows_per_core_per_loop_ * align_cnt);
            Sigmoid(sigmoidOutLocal, expertXfp32Local, rows_per_core_per_loop_ * align_cnt);
            Cast(expertXoutLocal, sigmoidOutLocal, RoundMode::CAST_ROUND, rows_per_core_per_loop_ * align_cnt);
            Add(sigmoidOutLocal, sigmoidOutLocal, broadcast_biasfp32Local_, rows_per_core_per_loop_ * align_cnt);
            Cast(biasXoutLocal, sigmoidOutLocal, RoundMode::CAST_ROUND, rows_per_core_per_loop_ * align_cnt);
        }

        outQueue_expertXout.EnQue<T>(expertXoutLocal);
        outQueue_biasXout.EnQue<T>(biasXoutLocal);

        work_buf2.FreeTensor(sigmoidOutLocal);
        inQueue_expertX.FreeTensor(expertXLocal);
    }

    __aicore__ inline void CopyOut(int32_t loop_cnt)
    {
        LocalTensor<T> expertXoutLocal = outQueue_expertXout.DeQue<T>();
        LocalTensor<T> biasXoutLocal = outQueue_biasXout.DeQue<T>();
        DataCopyParams copyParams{(uint16_t)rows_per_core_per_loop_, (uint16_t)(TILE_LENGTH * sizeof(T)), 0, 0};
        DataCopyPad(sigmoidOut_Gm[rows_offset_], expertXoutLocal, copyParams);

        DataCopyParams copyParams1{(uint16_t)rows_per_core_per_loop_, (uint16_t)(TILE_LENGTH * sizeof(T)), 0, 0};
        DataCopyPad(biasOut_Gm[rows_offset_], biasXoutLocal, copyParams1);

        outQueue_expertXout.FreeTensor(expertXoutLocal);
        outQueue_biasXout.FreeTensor(biasXoutLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_expertX, inQueue_expertBias;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_expertXout, outQueue_biasXout;
    TBuf<QuePosition::VECCALC> work_buf, work_buf1, work_buf2, work_buf3;
    GlobalTensor<T> expertBias_Gm, expertX_Gm, sigmoidOut_Gm, biasOut_Gm;
    LocalTensor<float> broadcast_biasfp32Local_;
    int32_t core_id_;
    int32_t TILE_LENGTH;
    int32_t per_dim_;
    int32_t BASE_SIZE, BASE_SIZE_FP32, block_cnt_;
    int32_t align_cnt;
    int32_t tail_count_, full_rows_offset_, tail_rows_offset_;
    int32_t loop_count_, full_loop_count_, dim_loop_count_, rows_per_core_per_loop_, tail_rows_last_loop_per_core_;
    int32_t global_core_index_, core_rows_, rows_offset_;
};


extern "C" __global__ __aicore__ void compute_bias_sigmoid(GM_ADDR expertX, GM_ADDR sigmoidOut, GM_ADDR biasOut,
                                                           GM_ADDR expertBias, int32_t ROWS, int32_t per_count,
                                                           int32_t block_cnt, int32_t dtype)
{
    TYPE_SWITCH(dtype, T, {
        bias_sigmoid<T> op;
        op.Init(expertX, sigmoidOut, biasOut, expertBias, ROWS, per_count, block_cnt);
        op.Process();
    });
}


void bias_sigmoid_launch(uint8_t *expertX, uint8_t *sigmoidOut, uint8_t *biasOut, uint8_t *expertBias, int32_t ROWS,
                         int32_t per_count, int32_t blockDim, void *stream, int32_t dtype)
{
    int32_t real_blockDim = ROWS < blockDim ? ROWS : blockDim;
    compute_bias_sigmoid<<<real_blockDim, nullptr, stream>>>(expertX, sigmoidOut, biasOut, expertBias, ROWS, per_count,
                                                             real_blockDim, dtype);
}

void bias_sigmoid_npu(int64_t blockDim, torch::Tensor &expertX, torch::Tensor &expertBias, torch::Tensor &sigmoidOut,
                      torch::Tensor biasOut)
{
    TORCH_CHECK(torch_npu::utils::is_npu(expertX), "expertX tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(expertBias), "expertBias tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(sigmoidOut), "sigmoidOut tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(biasOut), "biasOut tensor must be on NPU device");

    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    auto acl_call = [=]() -> int {
        bias_sigmoid_launch((uint8_t *)expertX.data_ptr(), (uint8_t *)sigmoidOut.data_ptr(),
                            (uint8_t *)biasOut.data_ptr(), (uint8_t *)expertBias.data_ptr(), expertX.size(0),
                            expertX.size(1), blockDim, stream, expertX.scalar_type() == at::kHalf ? 1 : 27);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("BiasSigmoid", acl_call);
}

TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
{
    m.impl("bias_sigmoid", bias_sigmoid_npu);
}
} // namespace BiasSigmoid
} // namespace npu_ops_transformer_ext