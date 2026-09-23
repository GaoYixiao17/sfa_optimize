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

namespace ascend_ops {
    namespace LayernormAddAddQuant {
#include <iostream>
#include <stdio.h>
#include "kernel_operator.h"
        using namespace AscendC;
        constexpr int64_t UB_MAX_BYTES = 184*1024;
        constexpr int64_t BUFFER_NUM = 1;
        constexpr int64_t WORK_LOCAL_SIZE = 64;
        inline int64_t align_up(const int64_t number, const int64_t alignSize)
        {
            if (number % alignSize == 0) {
                return number;
            }
            return ((number / alignSize + 1) * alignSize);
        }

        class LAYERNORM_ADDADD_QUANT2D_SINGLECORE {
        public:
            __aicore__ inline LAYERNORM_ADDADD_QUANT2D_SINGLECORE() {}

            __aicore__ inline void Init(const int64_t gbH, const int64_t gbW, const float epsilon,
                    GM_ADDR inOne, GM_ADDR inTwo, GM_ADDR inThr, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                    GM_ADDR outLynQuant, GM_ADDR outAdd, const bool constrait)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();

                gbH_ = gbH;
                gbW_ = gbW;
                epsilon_ = epsilon;
                invert_ = (float)(1.0) / gbW;
                constrait_ = constrait;

                bkH_ = 1;
                bkW_ = gbW_;
                bkAlignW_ = AlignUp(bkW_, 64);
                bkLoop_ = (int64_t)(gbH_ / blockNum_);
                if (gbH_ % blockNum_ != 0) { bkLoop_ += 1; }

                inOneGm_.SetGlobalBuffer((__gm__ half*)inOne, gbH_ * gbW_);
                inTwoGm_.SetGlobalBuffer((__gm__ half*)inTwo, gbW_);
                inThrGm_.SetGlobalBuffer((__gm__ half*)inThr, gbH_ * gbW_);
                gammaGm_.SetGlobalBuffer((__gm__ half*)gamma, gbW_);
                betaGm_.SetGlobalBuffer((__gm__ half*)beta, gbW_);
                scaleGm_.SetGlobalBuffer((__gm__ float*)scale, gbW_);
                outLynQuantGm_.SetGlobalBuffer((__gm__ int8_t*)outLynQuant, gbH_ * gbW_);
                outAddGm_.SetGlobalBuffer((__gm__ half*)outAdd, gbH_ * gbW_);

                pipe_.InitBuffer(inQueInOneThr_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(half) * 2);
                pipe_.InitBuffer(inQueInTwo_, 1, bkAlignW_ * sizeof(half));
                pipe_.InitBuffer(inQueGamma_, 1, bkAlignW_ * sizeof(half));
                pipe_.InitBuffer(inQueBeta_, 1, bkAlignW_ * sizeof(half));
                pipe_.InitBuffer(inQueScale_, 1, bkAlignW_ * sizeof(float));
                pipe_.InitBuffer(outQueOutLynQuant_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(int8_t));
                pipe_.InitBuffer(outQueOutAdd_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(half));

                pipe_.InitBuffer(bufQueWork_, WORK_LOCAL_SIZE * sizeof(float));
                pipe_.InitBuffer(bufQueSum_, 16*sizeof(float));
            }

            __aicore__ inline void Process()
            {
                LocalTensor<half> gamma_local = inQueGamma_.AllocTensor<half>();
                LocalTensor<half> beta_local = inQueBeta_.AllocTensor<half>();
                LocalTensor<half> in_two_local = inQueInTwo_.AllocTensor<half>();
                LocalTensor<float> scale_local = inQueScale_.AllocTensor<float>();

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(half)), 0, 0};
                DataCopyParams copy_pas_float{1, (uint16_t)(bkW_ * sizeof(float)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(gamma_local, gammaGm_, copy_pas, pad_pas);
                DataCopyPad(beta_local, betaGm_, copy_pas, pad_pas);
                DataCopyPad(in_two_local, inTwoGm_, copy_pas, pad_pas);
                DataCopyPad(scale_local, scaleGm_, copy_pas_float, pad_pas);

                inQueGamma_.EnQue(gamma_local);
                inQueBeta_.EnQue(beta_local);
                inQueInTwo_.EnQue(in_two_local);
                inQueScale_.EnQue(scale_local);
                gammaLm_ = inQueGamma_.DeQue<half>();
                betaLm_ = inQueBeta_.DeQue<half>();
                inTwoLm_ = inQueInTwo_.DeQue<half>();
                scaleLm_ = inQueScale_.DeQue<float>();

                if (bkAlignW_-AlignUp(bkW_, 16) != 0) {
                    Duplicate(in_two_local[AlignUp(bkW_, 16)], (half)0.0, bkAlignW_-AlignUp(bkW_, 16));
                }

                for (int64_t i = 0; i < bkLoop_; i++) {
                    if (i * blockNum_ + blockIdx_ < gbH_) {
                        CopyIn(i);
                        Compute(i);
                        CopyOut(i);
                    }
                }

                inQueGamma_.FreeTensor(gamma_local);
                inQueBeta_.FreeTensor(beta_local);
                inQueInTwo_.FreeTensor(in_two_local);
                inQueScale_.FreeTensor(scale_local);
            }

        private:
            __aicore__ inline void CopyIn(int64_t process)
            {
                LocalTensor<half> in_onethr_local = inQueInOneThr_.AllocTensor<half>();

                int64_t offset = (process * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(half)), 0, 0};
                DataCopyPadParams pad_pas;

                DataCopyPad(in_onethr_local, inOneGm_[offset], copy_pas, pad_pas);
                DataCopyPad(in_onethr_local[bkAlignW_], inThrGm_[offset], copy_pas, pad_pas);

                inQueInOneThr_.EnQue(in_onethr_local);
            }

            __aicore__ inline void Compute(int64_t progress)
            {
                (void)progress;
                LocalTensor<half> in_onethr_local = inQueInOneThr_.DeQue<half>();
                LocalTensor<half> in_one_local   = in_onethr_local[0];
                LocalTensor<half> in_thr_local   = in_onethr_local[bkAlignW_];
                LocalTensor<float> infloat_local = in_onethr_local.ReinterpretCast<float>();

                if (bkAlignW_-AlignUp(bkW_, 16) != 0) {
                    Duplicate<half>(in_one_local[AlignUp(bkW_, 16)], (half)0.0, bkAlignW_-AlignUp(bkW_, 16));
                    Duplicate<half>(in_thr_local[AlignUp(bkW_, 16)], (half)0.0, bkAlignW_-AlignUp(bkW_, 16));
                }

                LocalTensor<int8_t> out_lyn_quant_local = outQueOutLynQuant_.AllocTensor<int8_t>();
                LocalTensor<half> out_add_local = outQueOutAdd_.AllocTensor<half>();

                LocalTensor<float> sum_local     = bufQueSum_.Get<float>();
                LocalTensor<float> work_local    = bufQueWork_.Get<float>();

                Add(in_one_local, in_one_local, inTwoLm_, bkAlignW_);
                Add(out_add_local, in_one_local, in_thr_local, bkAlignW_);

                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkAlignW_);
                {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1, 1, 1, 0, 8, 0};
                    Add(work_local, infloat_local, work_local, 64, bkAlignW_/64, byrt_pas);
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                }

                Muls(sum_local, sum_local, (float)invert_, 1);
                Muls(sum_local, sum_local, (float)(-1.0), 1);
                float cur_mean = sum_local.GetValue(0);

                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkAlignW_);
                Adds(infloat_local, infloat_local, cur_mean, bkW_);
                Mul(infloat_local, infloat_local, infloat_local, bkW_);
                {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1, 1, 1, 0, 8, 0};
                    Add(work_local, infloat_local, work_local, 64, bkAlignW_/64, byrt_pas);
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                }

                Muls(sum_local, sum_local, (float)invert_, 1);
                Adds(sum_local, sum_local, (float)epsilon_, 1);
                Rsqrt(sum_local, sum_local, 1);
                float cur_var = sum_local.GetValue(0);

                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkW_);
                Adds(infloat_local, infloat_local, cur_mean, bkW_);
                Muls(infloat_local, infloat_local, cur_var, bkW_);

                Cast(in_one_local, infloat_local, RoundMode::CAST_NONE, bkW_);
                Mul(in_one_local, in_one_local, gammaLm_, bkW_);
                Add(in_one_local, in_one_local, betaLm_, bkW_);

                Adds(in_thr_local, in_one_local, (half)0.0, bkW_);
                Cast(infloat_local, in_thr_local, RoundMode::CAST_NONE, bkW_);

                if (constrait_) {
                    Mins(infloat_local, infloat_local, (float)128.0, bkW_);
                    Maxs(infloat_local, infloat_local, (float)-128.0, bkW_);
                }

                Mul(infloat_local, infloat_local, scaleLm_, bkW_);
                Cast(infloat_local, infloat_local, RoundMode::CAST_RINT, bkW_);
                Mins(infloat_local, infloat_local, (float)127.0, bkW_);
                Maxs(infloat_local, infloat_local, (float)-128.0, bkW_);
                Cast(in_one_local, infloat_local, RoundMode::CAST_NONE, bkW_);
                Cast(out_lyn_quant_local, in_one_local, RoundMode::CAST_RINT, bkW_);

                inQueInOneThr_.FreeTensor(in_onethr_local);

                outQueOutAdd_.EnQue(out_add_local);
                outQueOutLynQuant_.EnQue(out_lyn_quant_local);
            }

            __aicore__ inline void CopyOut(int64_t progress)
            {
                LocalTensor<half> out_add_local = outQueOutAdd_.DeQue<half>();
                LocalTensor<int8_t> out_lyn_quant_local = outQueOutLynQuant_.DeQue<int8_t>();

                int64_t offset = (progress * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(half)), 0, 0};
                DataCopyParams copy_pas_int8{1, (uint16_t)(bkW_ * sizeof(int8_t)), 0, 0};
                DataCopyPad(outAddGm_[offset], out_add_local, copy_pas);
                DataCopyPad(outLynQuantGm_[offset], out_lyn_quant_local, copy_pas_int8);

                outQueOutAdd_.FreeTensor(out_add_local);
                outQueOutLynQuant_.FreeTensor(out_lyn_quant_local);
            }

        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueInOneThr_;
            TQue<QuePosition::VECIN, 1> inQueGamma_, inQueBeta_, inQueScale_, inQueInTwo_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOutLynQuant_, outQueOutAdd_;

            GlobalTensor<half> inOneGm_, inTwoGm_, inThrGm_, gammaGm_, betaGm_;
            GlobalTensor<float> scaleGm_;
            GlobalTensor<int8_t> outLynQuantGm_;
            GlobalTensor<half> outAddGm_;

            LocalTensor<half> gammaLm_, betaLm_, inTwoLm_;
            LocalTensor<float> scaleLm_;

            TBuf<QuePosition::VECCALC> bufQueWork_;
            TBuf<QuePosition::VECCALC> bufQueSum_;

            int64_t blockNum_, blockIdx_;
            int64_t gbH_, gbW_;

            int64_t bkH_;
            int64_t bkW_, bkAlignW_;
            int64_t bkLoop_;

            float epsilon_ = (float)0.0f;
            float invert_ = (float)0.0f;
            bool constrait_ = false;
        };

        class LAYERNORM_ADDADD_QUANT2D_SINGLECORE_SH {
        public:
            __aicore__ inline LAYERNORM_ADDADD_QUANT2D_SINGLECORE_SH() {}

            __aicore__ inline void Init(const int64_t gbH, const int64_t gbW, const float epsilon,
                    GM_ADDR inOne, GM_ADDR inTwo, GM_ADDR inThr, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                    GM_ADDR outLynQuant, GM_ADDR outAdd, const bool constrait)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();

                gbH_ = gbH;
                gbW_ = gbW;
                epsilon_ = epsilon;
                invert_ = (float)(1.0) / gbW;
                constrait_ = constrait;

                bkH_ = 1;
                bkW_ = gbW_;
                bkAlignW_ = AlignUp(bkW_, 64);
                bkLoop_ = (int64_t)(gbH_ / blockNum_);
                if (gbH_ % blockNum_ != 0) { bkLoop_ += 1; }

                inOneGm_.SetGlobalBuffer((__gm__ half*)inOne, gbH_ * gbW_);
                inTwoGm_.SetGlobalBuffer((__gm__ half*)inTwo, gbW_);
                inThrGm_.SetGlobalBuffer((__gm__ half*)inThr, gbH_ * gbW_);
                gammaGm_.SetGlobalBuffer((__gm__ half*)gamma, gbW_);
                betaGm_.SetGlobalBuffer((__gm__ half*)beta, gbW_);
                scaleGm_.SetGlobalBuffer((__gm__ float*)scale, gbW_);
                outLynQuantGm_.SetGlobalBuffer((__gm__ int8_t*)outLynQuant, gbH_ * gbW_);
                outAddGm_.SetGlobalBuffer((__gm__ half*)outAdd, gbH_ * gbW_);

                pipe_.InitBuffer(inQueInOneThr_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(half) * 2);
                pipe_.InitBuffer(inQueInTwo_, 1, bkAlignW_ * sizeof(half));
                pipe_.InitBuffer(inQueGamma_, 1, bkAlignW_ * sizeof(half));
                pipe_.InitBuffer(inQueBeta_, 1, bkAlignW_ * sizeof(half));
                pipe_.InitBuffer(inQueScale_, 1, bkAlignW_ * sizeof(float));
                pipe_.InitBuffer(outQueOutLynQuant_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(int8_t));
                pipe_.InitBuffer(outQueOutAdd_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(half));

                pipe_.InitBuffer(bufQueWork_, WORK_LOCAL_SIZE * sizeof(float));
                pipe_.InitBuffer(bufQueSum_, 16*sizeof(float));
            }

            __aicore__ inline void Process()
            {
                LocalTensor<half> in_two_local = inQueInTwo_.AllocTensor<half>();
                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(half)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(in_two_local, inTwoGm_, copy_pas, pad_pas);
                inQueInTwo_.EnQue(in_two_local);
                inTwoLm_ = inQueInTwo_.DeQue<half>();

                if (bkAlignW_-AlignUp(bkW_, 16) != 0) {
                    Duplicate(in_two_local[AlignUp(bkW_, 16)], (half)0.0, bkAlignW_-AlignUp(bkW_, 16));
                }

                for (int64_t i = 0; i < bkLoop_; i++) {
                    if (i * blockNum_ + blockIdx_ < gbH_) {
                        CopyIn(i);
                        Compute(i);
                        CopyOut(i);
                    }
                }

                inQueInTwo_.FreeTensor(in_two_local);
            }

        private:
            __aicore__ inline void CopyIn(int64_t process)
            {
                LocalTensor<half> in_onethr_local = inQueInOneThr_.AllocTensor<half>();

                int64_t offset = (process * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(half)), 0, 0};
                DataCopyPadParams pad_pas;

                DataCopyPad(in_onethr_local, inOneGm_[offset], copy_pas, pad_pas);
                DataCopyPad(in_onethr_local[bkAlignW_], inThrGm_[offset], copy_pas, pad_pas);

                inQueInOneThr_.EnQue(in_onethr_local);
            }

            __aicore__ inline void Compute(int64_t progress)
            {
                (void)progress;
                LocalTensor<half> in_onethr_local = inQueInOneThr_.DeQue<half>();
                LocalTensor<half> in_one_local   = in_onethr_local[0];
                LocalTensor<half> in_thr_local   = in_onethr_local[bkAlignW_];
                LocalTensor<float> infloat_local = in_onethr_local.ReinterpretCast<float>();

                {
                    LocalTensor<half> gamma_local = inQueGamma_.AllocTensor<half>();
                    LocalTensor<half> beta_local = inQueBeta_.AllocTensor<half>();
                    LocalTensor<float> scale_local = inQueScale_.AllocTensor<float>();
                    DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(half)), 0, 0};
                    DataCopyParams copy_pas_float{1, (uint16_t)(bkW_ * sizeof(float)), 0, 0};
                    DataCopyPadParams pad_pas;
                    DataCopyPad(gamma_local, gammaGm_, copy_pas, pad_pas);
                    DataCopyPad(beta_local, betaGm_, copy_pas, pad_pas);
                    DataCopyPad(scale_local, scaleGm_, copy_pas_float, pad_pas);
                    inQueGamma_.EnQue(gamma_local);
                    inQueBeta_.EnQue(beta_local);
                    inQueScale_.EnQue(scale_local);
                }

                if (bkAlignW_-AlignUp(bkW_, 16) != 0) {
                    Duplicate<half>(in_one_local[AlignUp(bkW_, 16)], (half)0.0, bkAlignW_-AlignUp(bkW_, 16));
                    Duplicate<half>(in_thr_local[AlignUp(bkW_, 16)], (half)0.0, bkAlignW_-AlignUp(bkW_, 16));
                }

                LocalTensor<int8_t> out_lyn_quant_local = outQueOutLynQuant_.AllocTensor<int8_t>();
                LocalTensor<half> out_add_local = outQueOutAdd_.AllocTensor<half>();

                LocalTensor<float> sum_local     = bufQueSum_.Get<float>();
                LocalTensor<float> work_local    = bufQueWork_.Get<float>();

                Add(in_one_local, in_one_local, inTwoLm_, bkAlignW_);
                Add(out_add_local, in_one_local, in_thr_local, bkAlignW_);

                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkAlignW_);
                {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1, 1, 1, 0, 8, 0};
                    Add(work_local, infloat_local, work_local, 64, bkAlignW_/64, byrt_pas);
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                }

                Muls(sum_local, sum_local, (float)invert_, 1);
                Muls(sum_local, sum_local, (float)(-1.0), 1);
                float cur_mean = sum_local.GetValue(0);

                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkAlignW_);
                Adds(infloat_local, infloat_local, cur_mean, bkW_);
                Mul(infloat_local, infloat_local, infloat_local, bkW_);
                {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1, 1, 1, 0, 8, 0};
                    Add(work_local, infloat_local, work_local, 64, bkAlignW_/64, byrt_pas);
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                }

                Muls(sum_local, sum_local, (float)invert_, 1);
                Adds(sum_local, sum_local, (float)epsilon_, 1);
                Rsqrt(sum_local, sum_local, 1);
                float cur_var = sum_local.GetValue(0);

                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkW_);
                Adds(infloat_local, infloat_local, cur_mean, bkW_);
                Muls(infloat_local, infloat_local, cur_var, bkW_);

                Cast(in_one_local, infloat_local, RoundMode::CAST_NONE, bkW_);
                LocalTensor<half> gamma_local = inQueGamma_.DeQue<half>();
                Mul(in_one_local, in_one_local, gamma_local, bkW_);
                LocalTensor<half> beta_local = inQueBeta_.DeQue<half>();
                Add(in_one_local, in_one_local, beta_local, bkW_);

                Adds(in_thr_local, in_one_local, (half)0.0, bkW_);
                Cast(infloat_local, in_thr_local, RoundMode::CAST_NONE, bkW_);

                if (constrait_) {
                    Mins(infloat_local, infloat_local, (float)128.0, bkW_);
                    Maxs(infloat_local, infloat_local, (float)-128.0, bkW_);
                }

                LocalTensor<float> scale_local = inQueScale_.DeQue<float>();
                Mul(infloat_local, infloat_local, scale_local, bkW_);
                Cast(infloat_local, infloat_local, RoundMode::CAST_RINT, bkW_);
                Mins(infloat_local, infloat_local, (float)127.0, bkW_);
                Maxs(infloat_local, infloat_local, (float)-128.0, bkW_);
                Cast(in_one_local, infloat_local, RoundMode::CAST_NONE, bkW_);
                Cast(out_lyn_quant_local, in_one_local, RoundMode::CAST_RINT, bkW_);

                inQueInOneThr_.FreeTensor(in_onethr_local);
                inQueGamma_.FreeTensor(gamma_local);
                inQueBeta_.FreeTensor(beta_local);
                inQueScale_.FreeTensor(scale_local);

                outQueOutAdd_.EnQue(out_add_local);
                outQueOutLynQuant_.EnQue(out_lyn_quant_local);
            }

            __aicore__ inline void CopyOut(int64_t progress)
            {
                LocalTensor<half> out_add_local = outQueOutAdd_.DeQue<half>();
                LocalTensor<int8_t> out_lyn_quant_local = outQueOutLynQuant_.DeQue<int8_t>();

                int64_t offset = (progress * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(half)), 0, 0};
                DataCopyParams copy_pas_int8{1, (uint16_t)(bkW_ * sizeof(int8_t)), 0, 0};
                DataCopyPad(outAddGm_[offset], out_add_local, copy_pas);
                DataCopyPad(outLynQuantGm_[offset], out_lyn_quant_local, copy_pas_int8);

                outQueOutAdd_.FreeTensor(out_add_local);
                outQueOutLynQuant_.FreeTensor(out_lyn_quant_local);
            }

        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueInOneThr_;
            TQue<QuePosition::VECIN, 1> inQueGamma_, inQueBeta_, inQueScale_, inQueInTwo_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOutLynQuant_, outQueOutAdd_;

            GlobalTensor<half> inOneGm_, inTwoGm_, inThrGm_, gammaGm_, betaGm_;
            GlobalTensor<float> scaleGm_;
            GlobalTensor<int8_t> outLynQuantGm_;
            GlobalTensor<half> outAddGm_;

            LocalTensor<half> inTwoLm_;

            TBuf<QuePosition::VECCALC> bufQueWork_;
            TBuf<QuePosition::VECCALC> bufQueSum_;

            int64_t blockNum_, blockIdx_;
            int64_t gbH_, gbW_;

            int64_t bkH_;
            int64_t bkW_, bkAlignW_;
            int64_t bkLoop_;

            float epsilon_ = (float)0.0f;
            float invert_ = (float)0.0f;
            bool constrait_ = false;
        };

        extern "C" __global__ __aicore__ void layerNormAddAddQuant2dSingleCore_kernel(
                const int64_t gbH, const int64_t gbW,
                const float epsilon, GM_ADDR inOne, GM_ADDR inTwo, GM_ADDR inThr,
                GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale,
                GM_ADDR outLynQuant, GM_ADDR outAdd,
                const bool constrait)
        {
            if (gbH > GetBlockNum()) {
                LAYERNORM_ADDADD_QUANT2D_SINGLECORE op;
                op.Init(gbH, gbW, epsilon, inOne, inTwo, inThr, gamma, beta, scale,
                        outLynQuant, outAdd, constrait);
                op.Process();
            } else {
                LAYERNORM_ADDADD_QUANT2D_SINGLECORE_SH op;
                op.Init(gbH, gbW, epsilon, inOne, inTwo, inThr, gamma, beta, scale,
                        outLynQuant, outAdd, constrait);
                op.Process();
            }
        }

        void layerNormAddAddQuant2dSingleCore_lanuch(int64_t blockDim, void* stream,
                                                     const int64_t gbH, const int64_t gbW, const float epsilon,
                                                     uint8_t* inOne, uint8_t* inTwo, uint8_t* inThr,
                                                     uint8_t* gamma, uint8_t* beta, uint8_t* scale,
                                                     uint8_t* outLynQuant, uint8_t* outAdd,
                                                     const bool constrait)
        {
            if (gbH < blockDim) { blockDim = gbH; }

            layerNormAddAddQuant2dSingleCore_kernel<<<blockDim, nullptr, stream>>>(gbH, gbW, epsilon,
                    inOne, inTwo, inThr, gamma, beta, scale,
                    outLynQuant, outAdd, constrait);
        }

        int judge_layernorm_add_add_quant_lanuch(const int64_t gbH, const int64_t gbW,
                                                 const int64_t ubSize, const int64_t vCores)
        {
            (void)(vCores);

            constexpr int64_t BUFFER_NUM = 1;

            int64_t bkH_ = 1;
            int64_t bkW_ = gbW;
            int64_t bkAlignW_ = align_up(bkW_, 64);

            float use_byte = BUFFER_NUM * bkH_ * bkAlignW_ * sizeof(half) * 2;
            use_byte += bkAlignW_ * sizeof(half) * 3;
            use_byte += bkAlignW_ * sizeof(float);
            use_byte += BUFFER_NUM * bkH_ * bkAlignW_ * sizeof(int8_t);
            use_byte += (16 + WORK_LOCAL_SIZE) * sizeof(float);

            if (use_byte > ubSize) {
                std::cout << __FUNCTION__ << ": " << use_byte/1024 << " KB" << std::endl;
                return 1;
            }
            return 0;
        }

        int layernorm_add_add_quant_lanuch(int64_t block_dim, void* stream,
                                           const int64_t gbH, const int64_t gbW, const float epsilon,
                                           uint8_t* inOne, uint8_t* inTwo, uint8_t* inThr,
                                           uint8_t* gamma, uint8_t* beta, uint8_t* scale,
                                           uint8_t* outLynQuant, uint8_t* outAdd,
                                           const bool constrait)
        {
            int64_t ubSize = 184 * 1024;
            int64_t vCores = 40;

            int ret = judge_layernorm_add_add_quant_lanuch(gbH, gbW, ubSize, vCores);
            if (ret == 0) {
                layerNormAddAddQuant2dSingleCore_lanuch(block_dim, stream, gbH, gbW, epsilon,
                                                        inOne, inTwo, inThr, gamma, beta, scale,
                                                        outLynQuant, outAdd, constrait);
                return 0;
            }

            std::cout << __FUNCTION__ << ": " << "UB size is limited, please check!" << std::endl;
            return 1;
        }

        int64_t layernorm_add_add_quant_npu(int64_t block_dim, torch::Tensor &inOne, torch::Tensor &inTwo,
                                            torch::Tensor &inThr, torch::Tensor &gamma, torch::Tensor &beta,
                                            torch::Tensor &scale, torch::Tensor &outLynQuant, torch::Tensor &outAdd,
                                            double epsilon, bool constrait)
        {
            TORCH_CHECK(torch_npu::utils::is_npu(inOne), "inOne tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(inTwo), "inTwo tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(inThr), "inThr tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(gamma), "gamma tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(beta), "beta tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(scale), "scale tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(outLynQuant), "outLynQuant tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(outAdd), "outAdd tensor must be on NPU device");

            const int64_t gbH = inOne.size(0);
            const int64_t gbW = inOne.size(1);

            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            int launchStatus = 0;
            auto acl_call = [=, &launchStatus]() -> int {
                launchStatus = layernorm_add_add_quant_lanuch(block_dim, stream, gbH, gbW, epsilon,
                                                              (uint8_t *)inOne.data_ptr(), (uint8_t *)inTwo.data_ptr(),
                                                              (uint8_t *)inThr.data_ptr(), (uint8_t *)gamma.data_ptr(),
                                                              (uint8_t *)beta.data_ptr(), (uint8_t *)scale.data_ptr(),
                                                              (uint8_t *)outLynQuant.data_ptr(),
                                                              (uint8_t *)outAdd.data_ptr(), constrait);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("LayernormAddAddQuant", acl_call);
            return launchStatus;
        }

        TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
        {
            m.impl("layernorm_add_add_quant", layernorm_add_add_quant_npu);
        }
    }
}