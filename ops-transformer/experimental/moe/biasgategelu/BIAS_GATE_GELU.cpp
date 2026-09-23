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

namespace ascend_ops {
    namespace BiasGateGelu {
#include "kernel_operator.h"
#include <iostream>
#include <stdio.h>
using namespace AscendC;
constexpr int64_t UB_MAX_BYTES = 184*1024;
constexpr int64_t BUFFER_NUM = 2;
class BIAS_GATE_GELU {
public:
    __aicore__ inline BIAS_GATE_GELU() {}
    __aicore__ inline void Init(const int64_t gbH, const int64_t gbW,
                                GM_ADDR in, GM_ADDR bias, GM_ADDR out)
    {
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();
        gbH_ = gbH;
        gbW_ = gbW;
        bkH_ = 1;
        bkW_ = gbW_ / 2; // in (inOne + inTwo)
        bkAlignW_ = AlignUp(bkW_, 16);
        bkLoop_ = (int64_t)(gbH_ / blockNum_);
        if (gbH_ % blockNum_ != 0) { bkLoop_ += 1; }
        int64_t temp = BUFFER_NUM * bkH_ * sizeof(half) * 3;
        temp += BUFFER_NUM * 2 * sizeof(half);
        tlMaxW_ = UB_MAX_BYTES / temp;
        tlMaxW_ = tlMaxW_ / 64 * 64; // AlignDown
        tlH_ = 1;
        if (bkW_ < tlMaxW_) {
            tlW_ = bkW_;
        }
        else {
            int64_t temp = bkW_ / tlMaxW_;
            temp += (bkW_ % tlMaxW_ == 0) ? 0 : 1;
            tlW_ = bkW_ / temp;
        }
        tlW_ = AlignUp(tlW_, 64);
        tlTailW_ = bkW_ % tlW_;
        tlAlignTailW_ = AlignUp(tlTailW_, 64);
        tlLoop_ = (int64_t)(bkW_ / tlW_);
        if (tlTailW_ != 0) { tlLoop_ += 1; }
        inGm_.SetGlobalBuffer((__gm__ half*)in, gbH_ * gbW_);
        biasGm_.SetGlobalBuffer((__gm__ half*)bias, gbW_);
        outGm_.SetGlobalBuffer((__gm__ half*)out, gbH_ * gbW_ / 2);
        pipe_.InitBuffer(inQueInOne_, BUFFER_NUM, bkH_ * tlW_ * sizeof(half));
        pipe_.InitBuffer(inQueInTwo_, BUFFER_NUM, bkH_ * tlW_ * sizeof(half));
        pipe_.InitBuffer(inQueBias_,  BUFFER_NUM, tlW_ * 2 * sizeof(half));
        pipe_.InitBuffer(outQueOut_,  BUFFER_NUM, bkH_ * tlW_ * sizeof(half));
#ifdef __CCE_KT_TEST__
        if (blockIdx_ == 0) {
            float use_byte = BUFFER_NUM * bkH_ * tlW_ * sizeof(half) * 3;
            use_byte += BUFFER_NUM * tlW_ * 2 * sizeof(half);
            fprintf(stdout, "\033[33m[_AI] Use UB bytes: %f B, %f KB\033[0m\n",
                    use_byte, use_byte/1024);
        }
#endif
    }
    __aicore__ inline void Process()
    {
        for (int64_t j = 0; j < tlLoop_; j++) {
            LocalTensor<half> bias_local = inQueBias_.AllocTensor<half>();
            DataCopyParams copy_pas{1, (uint16_t)(tlW_ * sizeof(half)), 0, 0};
            DataCopyPadParams pad_pas;
            if ((tlTailW_ > 0) && (j == tlLoop_-1)) {
                copy_pas.blockLen = (uint16_t)(tlTailW_ * sizeof(half));
            }
            DataCopyPad(bias_local, biasGm_[j*tlW_], copy_pas, pad_pas);
            DataCopyPad(bias_local[tlW_], biasGm_[bkW_+j*tlW_], copy_pas, pad_pas);
            inQueBias_.EnQue(bias_local);
            biasLm_ = inQueBias_.DeQue<half>();
            for (int64_t i = 0; i < bkLoop_; i++) {
                if (i * blockNum_ + blockIdx_ < gbH_) {
                    if ((tlTailW_ > 0) && (j == tlLoop_-1)) {
                        CopyIn(i, j, tlTailW_, tlAlignTailW_);
                        Compute(i, j, tlTailW_, tlAlignTailW_);
                        CopyOut(i, j, tlTailW_, tlAlignTailW_);
                    }
                    else {
                        CopyIn(i, j, tlW_, tlW_);
                        Compute(i, j, tlW_, tlW_);
                        CopyOut(i, j, tlW_, tlW_);
                    }
                }
            }
            inQueBias_.FreeTensor(bias_local);
        }
    }
private:
    __aicore__ inline void CopyIn(int64_t bkl, int64_t tll, int64_t real_tlW, int64_t real_tlAlignW)
    {
        LocalTensor<half> in_one_local = inQueInOne_.AllocTensor<half>();
        LocalTensor<half> in_two_local = inQueInTwo_.AllocTensor<half>();
        int64_t offset = (bkl * blockNum_ + blockIdx_) * (bkW_ * 2) + (tll * tlW_); // in (inOne + inTwo)
        DataCopyParams copy_pas{1, (uint16_t)(real_tlW * sizeof(half)), 0, 0};
        DataCopyPadParams pad_pas;
        DataCopyPad(in_one_local, inGm_[offset], copy_pas, pad_pas);
        inQueInOne_.EnQue(in_one_local);
        DataCopyPad(in_two_local, inGm_[offset+bkW_], copy_pas, pad_pas);
        inQueInTwo_.EnQue(in_two_local);
    }
    __aicore__ inline void Compute(int64_t bkl, int64_t tll, int64_t real_tlW, int64_t real_tlAlignW)
    {
        LocalTensor<half> in_one_local = inQueInOne_.DeQue<half>();
        LocalTensor<half> out_local = outQueOut_.AllocTensor<half>();
        Add(in_one_local, in_one_local, biasLm_, real_tlAlignW);
        Gelu<half, true, false>(in_one_local, in_one_local, real_tlAlignW);
        LocalTensor<half> in_two_local = inQueInTwo_.DeQue<half>();
        Add(in_two_local, in_two_local, biasLm_[tlW_], real_tlAlignW);
        Mul(out_local, in_one_local, in_two_local, real_tlAlignW);
        inQueInOne_.FreeTensor(in_one_local);
        inQueInTwo_.FreeTensor(in_two_local);
        outQueOut_.EnQue(out_local);
    }
    __aicore__ inline void CopyOut(int64_t bkl, int64_t tll, int64_t real_tlW, int64_t real_tlAlignW)
    {
        LocalTensor<half> out_local = outQueOut_.DeQue<half>();
        int64_t offset = (bkl * blockNum_ + blockIdx_) * bkW_ + (tll * tlW_);
        DataCopyParams copy_pas{1, (uint16_t)(real_tlW * sizeof(half)), 0, 0};
        DataCopyPad(outGm_[offset], out_local, copy_pas);
        outQueOut_.FreeTensor(out_local);
    }
private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueInOne_, inQueInTwo_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueBias_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOut_;
    GlobalTensor<half> inGm_;
    GlobalTensor<half> biasGm_;
    GlobalTensor<half> outGm_;
    LocalTensor<half> biasLm_;
    int64_t blockNum_, blockIdx_;
    int64_t gbH_;
    int64_t gbW_;
    int64_t bkH_;
    int64_t bkW_;
    int64_t bkAlignW_;
    int64_t bkLoop_;
    int64_t tlH_;
    int64_t tlW_, tlTailW_, tlAlignTailW_;
    int64_t tlLoop_;
    int64_t tlMaxW_;
};
extern "C" __global__ __aicore__ void biasGateGelu_kernel(const int64_t gbH, const int64_t gbW,
        GM_ADDR in, GM_ADDR bias, GM_ADDR out)
{
    BIAS_GATE_GELU op;
    op.Init(gbH, gbW, in, bias, out);
    op.Process();
}
void biasGateGelu_lanuch(int64_t blockDim, void* stream,
                         const int64_t gbH, const int64_t gbW, uint8_t* in, uint8_t* bias, uint8_t* out)
{
    if (gbH < blockDim){ blockDim = gbH; }
    biasGateGelu_kernel<<<blockDim, nullptr, stream>>>(gbH, gbW, in, bias, out);
}

int64_t bias_gate_gelu_npu(int64_t block_dim, torch::Tensor &in, torch::Tensor &bias, torch::Tensor &out,
                           int64_t gbH, int64_t gbW)
{
    TORCH_CHECK(in.device().type() == torch::kPrivateUse1, "input must be on NPU");
    TORCH_CHECK(bias.device().type() == torch::kPrivateUse1, "bias must be on NPU");
    TORCH_CHECK(out.device().type() == torch::kPrivateUse1, "out must be on NPU");
    TORCH_CHECK(gbH > 0, "gbH must be positive");
    TORCH_CHECK(gbW > 0, "gbW must be positive");
    TORCH_CHECK(gbW % 2 == 0, "gbW must be even for gate/value split");
    TORCH_CHECK(block_dim > 0, "block_dim must be positive, got ", block_dim);
    TORCH_CHECK(in.sizes() == torch::IntArrayRef({gbH, gbW}), "in tensor shape mismatch");

    c10_npu::NPUStream stream = c10_npu::getCurrentNPUStream();
    auto in_ptr = reinterpret_cast<uint8_t*>(in.data_ptr());
    auto bias_ptr = reinterpret_cast<uint8_t*>(bias.data_ptr());
    auto out_ptr = reinterpret_cast<uint8_t*>(out.data_ptr());
    int64_t blockDim = (int64_t)block_dim;
    biasGateGelu_lanuch(blockDim, stream, gbH, gbW, in_ptr, bias_ptr, out_ptr);
    return 0;
}
    }
    TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
    {
        m.impl("bias_gate_gelu", BiasGateGelu::bias_gate_gelu_npu);
    }
}