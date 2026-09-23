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
    namespace GategeluQuant {
#include <iostream>
#include <stdio.h>
#include "kernel_operator.h"
        using namespace AscendC;

        constexpr int64_t UB_MAX_BYTES = 184*1024;
        constexpr int64_t BUFFER_NUM = 1;

        inline int64_t align_up(const int64_t number, const int64_t alignSize)
        {
            if (number % alignSize == 0) {
                return number;
            }
            return ((number / alignSize + 1) * alignSize);
        }

        class GATEGELU_QUANT2D_SBUF {
        public:
            __aicore__ inline GATEGELU_QUANT2D_SBUF() {}

            __aicore__ inline void Init(const int64_t gbH, const int64_t gbW,
                    GM_ADDR in, GM_ADDR scale,
                    GM_ADDR out, const bool constrait, const float clampValue)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();
                constrait_  = constrait;
                clampValue_ = clampValue;
                gbH_ = gbH;
                gbW_ = gbW;
                bkH_ = 1;
                bkW_ = gbW_ / 2;
                bkAlignW_ = AlignUp(bkW_, 32);
                bkLoop_ = (int64_t)(gbH_ / blockNum_);
                if (gbH_ % blockNum_ != 0) { bkLoop_ += 1; }
                int64_t temp = BUFFER_NUM * bkH_ * sizeof(half) * 2;
                temp += BUFFER_NUM * 2 * sizeof(half);
                temp += BUFFER_NUM * sizeof(float);
                temp += BUFFER_NUM * bkH_ *sizeof(int8_t);
                tlMaxW_ = UB_MAX_BYTES / temp;
                tlMaxW_ = tlMaxW_ / 64 * 64;
                tlH_ = 1;
                if (bkW_ < tlMaxW_) {
                    tlW_ = bkW_;
                } else {
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
                scaleGm_.SetGlobalBuffer((__gm__ float*)scale, gbW_ / 2);
                outGm_.SetGlobalBuffer((__gm__ int8_t*)out, gbH_ * gbW_ / 2);
                pipe_.InitBuffer(inQueIn_, BUFFER_NUM, bkH_ * tlW_ * 2 * sizeof(half));
                pipe_.InitBuffer(inQueScale_, 1, tlW_ * sizeof(float));
                pipe_.InitBuffer(outQueOut_,  BUFFER_NUM, bkH_ * tlW_ * sizeof(int8_t));
            }

            __aicore__ inline void Process()
            {
                for (int64_t i = 0; i < bkLoop_; i++) {
                    if (i * blockNum_ + blockIdx_ < gbH_) {
                        if (tlTailW_ > 0) {
                            CopyIn(i, 0, tlTailW_, tlAlignTailW_);
                            Compute(i, 0, tlTailW_, tlAlignTailW_);
                            CopyOut(i, 0, tlTailW_, tlAlignTailW_);
                        } else {
                            CopyIn(i, 0, tlW_, tlW_);
                            Compute(i, 0, tlW_, tlW_);
                            CopyOut(i, 0, tlW_, tlW_);
                        }
                    }
                }
            }

        private:
            __aicore__ inline void CopyIn(int64_t bkl, int64_t tll, int64_t real_tlW, int64_t real_tlAlignW)
            {
                LocalTensor<half> in_local = inQueIn_.AllocTensor<half>();
                int64_t offset = (bkl * blockNum_ + blockIdx_) * (bkW_ * 2) + (tll * tlW_);
                DataCopyParams copy_pas{1, (uint16_t)(real_tlW * sizeof(half)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(in_local, inGm_[offset], copy_pas, pad_pas);
                DataCopyPad(in_local[tlW_], inGm_[offset+bkW_], copy_pas, pad_pas);
                inQueIn_.EnQue(in_local);
                LocalTensor<float> scale_local = inQueScale_.AllocTensor<float>();
                DataCopyParams copy_pas_float{1, (uint16_t)(real_tlW * sizeof(float)), 0, 0};
                DataCopyPad(scale_local, scaleGm_, copy_pas_float, pad_pas);
                inQueScale_.EnQue(scale_local);
            }

            __aicore__ inline void Compute(int64_t bkl, int64_t tll, int64_t real_tlW, int64_t real_tlAlignW)
            {
                LocalTensor<half> in_local = inQueIn_.DeQue<half>();
                LocalTensor<half> in_one_local = in_local[0];
                LocalTensor<half> in_two_local = in_local[tlW_];
                LocalTensor<float> infloat_local = in_local.ReinterpretCast<float>();
                LocalTensor<int8_t> out_local = outQueOut_.AllocTensor<int8_t>();
                Gelu<half, true, false>(in_one_local, in_one_local, real_tlAlignW);
                Mul(in_two_local, in_one_local, in_two_local, real_tlAlignW);
                Cast(infloat_local, in_two_local, RoundMode::CAST_NONE, real_tlAlignW);
                if (constrait_) {
                    Mins(infloat_local, infloat_local, (float)clampValue_, real_tlAlignW);
                    Maxs(infloat_local, infloat_local, (float)(-clampValue_), real_tlAlignW);
                }
                LocalTensor<float> scale_local = inQueScale_.DeQue<float>();
                Mul(infloat_local, infloat_local, scale_local, real_tlAlignW);
                Cast(infloat_local, infloat_local, RoundMode::CAST_RINT, real_tlAlignW);
                Mins(infloat_local, infloat_local, (float)127.0, real_tlAlignW);
                Maxs(infloat_local, infloat_local, (float)-128.0, real_tlAlignW);
                Cast(in_one_local, infloat_local, RoundMode::CAST_NONE, real_tlAlignW);
                Cast(out_local, in_one_local, RoundMode::CAST_RINT, real_tlAlignW);
                inQueIn_.FreeTensor(in_local);
                inQueScale_.FreeTensor(scale_local);
                outQueOut_.EnQue(out_local);
            }

            __aicore__ inline void CopyOut(int64_t bkl, int64_t tll, int64_t real_tlW, int64_t real_tlAlignW)
            {
                LocalTensor<int8_t> out_local = outQueOut_.DeQue<int8_t>();
                int64_t offset = (bkl * blockNum_ + blockIdx_) * bkW_ + (tll * tlW_);
                DataCopyParams copy_pas{1, (uint16_t)(real_tlW * sizeof(int8_t)), 0, 0};
                DataCopyPad(outGm_[offset], out_local, copy_pas);
                outQueOut_.FreeTensor(out_local);
            }

        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueIn_;
            TQue<QuePosition::VECIN, 1> inQueScale_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOut_;
            GlobalTensor<half> inGm_;
            GlobalTensor<float> scaleGm_;
            GlobalTensor<int8_t> outGm_;
            int64_t blockNum_, blockIdx_;
            int64_t gbH_, gbW_;
            int64_t bkH_;
            int64_t bkW_, bkAlignW_;
            int64_t bkLoop_;
            int64_t tlH_;
            int64_t tlW_, tlTailW_, tlAlignTailW_;
            int64_t tlLoop_;
            int64_t tlMaxW_;
            bool constrait_ = false;
            float clampValue_ = 128.0f;
        };

        extern "C" __global__ __aicore__ void gateGeluQuant2dSBuf_kernel(const int64_t gbH, const int64_t gbW,
                GM_ADDR in, GM_ADDR scale,
                GM_ADDR out, const bool constrait, const float clampValue)
        {
            GATEGELU_QUANT2D_SBUF op;
            op.Init(gbH, gbW, in, scale, out, constrait, clampValue);
            op.Process();
        }

        void gateGeluQuant2dSBuf_lanuch(int64_t blockDim, void* stream,
                                        const int64_t gbH, const int64_t gbW, uint8_t* in, uint8_t* scale,
                                        uint8_t* out, const bool constrait, const float clampValue)
        {
            if (gbH < blockDim) { blockDim = gbH; }
            gateGeluQuant2dSBuf_kernel<<<blockDim, nullptr, stream>>>(gbH, gbW, in, scale,
                    out, constrait, clampValue);
        }

        int gategelu_quant_lanuch(int64_t block_dim, void* stream,
                                  const int64_t gbH, const int64_t gbW,
                                  uint8_t* in, uint8_t* scale,
                                  uint8_t* out, const bool constrait, const float clampValue)
        {
            gateGeluQuant2dSBuf_lanuch(block_dim, stream, gbH, gbW, in, scale,
                                       out, constrait, clampValue);
            return 0;
        }

        int64_t gategelu_quant_npu(int64_t block_dim, torch::Tensor &inTensor,
                                   torch::Tensor &scale, torch::Tensor &out,
                                   bool constrait, double clampValue)
        {
            TORCH_CHECK(torch_npu::utils::is_npu(inTensor), "inTensor tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(scale), "scale tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(out), "out tensor must be on NPU device");
            const int64_t gbH = inTensor.size(0);
            const int64_t gbW = inTensor.size(1);
            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            int launchStatus = 0;
            auto acl_call = [=, &launchStatus]() -> int {
                launchStatus = gategelu_quant_lanuch(block_dim, stream, gbH, gbW,
                                                     (uint8_t *)inTensor.data_ptr(), (uint8_t *)scale.data_ptr(),
                                                     (uint8_t *)out.data_ptr(), constrait, clampValue);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("GategeluQuant", acl_call);
            return launchStatus;
        }

        TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
        {
            m.impl("gategelu_quant", gategelu_quant_npu);
        }
    }
}