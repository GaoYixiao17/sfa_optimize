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
    namespace MoeLayernormAddAdd {
#include <iostream>
#include <stdio.h>
#include "dtype_convert.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"
        using namespace AscendC;

        constexpr int64_t UB_MAX_BYTES = 188*1024;
        constexpr int64_t BUFFER_NUM = 1;
        constexpr int64_t WORK_LOCAL_SIZE = 64;

        //===================================================== general singlecore
        template<typename T>
        class MOE_LAYERNORM_ADDADD2D_SINGLECORE {
        public:
            __aicore__ inline MOE_LAYERNORM_ADDADD2D_SINGLECORE() {}

            __aicore__ inline void Init(const int64_t gbH, const int64_t gbW, const float epsilon,
                    GM_ADDR  inOne, GM_ADDR  inTwo, GM_ADDR  inThr, GM_ADDR  gamma, GM_ADDR  beta,
                    GM_ADDR  outLyn, GM_ADDR  outAdd, const bool clamp, int64_t clamp_value, int32_t simplified)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();
                simplified_ = simplified;
                
                gbH_ = gbH;
                gbW_ = gbW;
                epsilon_ = epsilon;
                invert_ = (float)(1.0) / gbW;
                clamp_ = clamp;
                clamp_value_ = clamp_value;
                bkH_ = 1;
                bkW_ = gbW_;
                bkAlignW_ = AlignUp(bkW_, 64);
                bkLoop_ = (int64_t)(gbH_ / blockNum_);
                if (gbH_ % blockNum_ != 0) { bkLoop_ += 1; }

                // tlH_ = 1;
                // // tlW_ = (bkW_ <= MAX_COUNT_PER_TILE) ? bkW_ : MAX_COUNT_PER_TILE;
                // tlW_ = (bkW_ <= MAX_COUNT_PER_TILE) ? bkW_ : bkW_/2;
                // tlW_ = AlignUp(tlW_, 64);
                // tlTailW_ = bkW_ % tlW_;
                // tlAlignTailW_ = AlignUp(tlTailW_, 64);
                // tlLoop_ = (int64_t)(bkW_ / tlW_);
                // if(tlTailW_ != 0){ tlLoop_ += 1; }

                inOneGm_.SetGlobalBuffer((__gm__ T*)inOne, gbH_ * gbW_);
                inTwoGm_.SetGlobalBuffer((__gm__ T*)inTwo, gbW_);
                inThrGm_.SetGlobalBuffer((__gm__ T*)inThr, gbH_ * gbW_);
                gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, gbW_);
                betaGm_.SetGlobalBuffer((__gm__ T*)beta, gbW_);
                outLynGm_.SetGlobalBuffer((__gm__ T*)outLyn, gbH_ * gbW_);
                outAddGm_.SetGlobalBuffer((__gm__ T*)outAdd, gbH_ * gbW_);

                pipe_.InitBuffer(inQueInOneThr_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);  // bkH_ = 1
                pipe_.InitBuffer(inQueInTwo_, 1, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(inQueGamma_, 1, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(inQueBeta_, 1, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(outQueOutLyn_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(outQueOutAdd_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(bufQueWork_, WORK_LOCAL_SIZE * sizeof(float));
                pipe_.InitBuffer(bufQueSum_, 16*sizeof(float));
            }

            __aicore__ inline void Process()
            {
                LocalTensor<T> gamma_local = inQueGamma_.AllocTensor<T>();
                LocalTensor<T> beta_local = inQueBeta_.AllocTensor<T>();
                LocalTensor<T> in_two_local = inQueInTwo_.AllocTensor<T>();

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(gamma_local[bkAlignW_], gammaGm_, copy_pas, pad_pas);
                DataCopyPad(beta_local[bkAlignW_], betaGm_, copy_pas, pad_pas);
                DataCopyPad(in_two_local[bkAlignW_], inTwoGm_, copy_pas, pad_pas);

                inQueGamma_.EnQue(gamma_local);
                inQueBeta_.EnQue(beta_local);
                inQueInTwo_.EnQue(in_two_local);
            
                gammaLm_ = inQueGamma_.DeQue<T>();
                betaLm_ = inQueBeta_.DeQue<T>();
                inTwoLm_ = inQueInTwo_.DeQue<T>();

                gammaFloatLm_ = gammaLm_.template ReinterpretCast<float>();
                betaFloatLm_ = betaLm_.template ReinterpretCast<float>();

                Cast(gammaFloatLm_, gammaLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);
                Cast(betaFloatLm_, betaLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);

                if (bkAlignW_ - AlignUp(bkW_, 16) != 0) {
                    Duplicate(in_two_local[bkAlignW_ + AlignUp(bkW_, 16)], (T)0.0, bkAlignW_ - AlignUp(bkW_, 16));
                }
                inTwoLmfp32_ = inTwoLm_.template ReinterpretCast<float>();
                Cast(inTwoLmfp32_, inTwoLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);

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
            }

        private:
            __aicore__ inline void CopyIn(int64_t process)
            {
                LocalTensor<T> in_onethr_local = inQueInOneThr_.AllocTensor<T>();

                int64_t offset = (process * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                // DataCopyPadParams pad_pas{true, 0, (uint8_t)(bkAlignW_ - bkW_), 0};
                DataCopyPadParams pad_pas;

                DataCopyPad(in_onethr_local, inOneGm_[offset], copy_pas, pad_pas);
                DataCopyPad(in_onethr_local[bkAlignW_], inThrGm_[offset], copy_pas, pad_pas);
                
                inQueInOneThr_.EnQue(in_onethr_local);
            }

            __aicore__ inline void Compute(int64_t progress)
            {
                LocalTensor<T> in_onethr_local = inQueInOneThr_.DeQue<T>();
                LocalTensor<T> in_one_local   = in_onethr_local[0];
                LocalTensor<T> in_thr_local   = in_onethr_local[bkAlignW_];
                LocalTensor<float> infloat_local = in_onethr_local.template ReinterpretCast<float>();
                
                if (bkAlignW_ - AlignUp(bkW_, 16) != 0) {
                    Duplicate<T>(in_one_local[AlignUp(bkW_, 16)], (T)0.0, bkAlignW_ - AlignUp(bkW_, 16));
                    Duplicate<T>(in_thr_local[AlignUp(bkW_, 16)], (T)0.0, bkAlignW_ - AlignUp(bkW_, 16));
                }
                
                LocalTensor<T> out_lyn_local = outQueOutLyn_.AllocTensor<T>();
                LocalTensor<T> out_add_local = outQueOutAdd_.AllocTensor<T>();

                LocalTensor<float> in_one_fp32_local = out_lyn_local.template ReinterpretCast<float>();
                LocalTensor<float> in_thr_fp32_local = out_add_local.template ReinterpretCast<float>();

                Cast(in_one_fp32_local, in_one_local, RoundMode::CAST_NONE, bkAlignW_);
                Cast(in_thr_fp32_local, in_thr_local, RoundMode::CAST_NONE, bkAlignW_);

                LocalTensor<float> sum_local     = bufQueSum_.Get<float>();
                LocalTensor<float> work_local    = bufQueWork_.Get<float>();

                Add(in_one_fp32_local, in_one_fp32_local, inTwoLmfp32_, bkAlignW_);  //  + fc2 bias
                Add(in_one_fp32_local, in_one_fp32_local, in_thr_fp32_local, bkAlignW_); // + attn out
                
                // if(clamp_)
                // {
                //     ClampMax(in_one_fp32_local, in_one_fp32_local,  (float)(clamp_value_), bkW_);
                //     ClampMin(in_one_fp32_local, in_one_fp32_local, (float)(-1 * clamp_value_), bkW_);
                // }
                Cast(out_add_local, in_one_fp32_local, RoundMode::CAST_ROUND, bkAlignW_); // +fc2 + attn out
                // Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkAlignW_);

                float cur_mean = 0.0f;
                if (!simplified_) {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1, 1, 1, 0, 8, 0};
                    Add(work_local, in_one_fp32_local, work_local, 64, bkAlignW_/64, byrt_pas) ;
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                    Muls(sum_local, sum_local, (float)invert_, 1);
                    Muls(sum_local, sum_local, (float)(-1.0), 1);
                    cur_mean = sum_local.GetValue(0);
                }
                              
                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkAlignW_);
                // (x - mean)
                if (!simplified_) {
                    Adds(infloat_local, infloat_local, cur_mean, bkW_);
                }
                // (x - mean) ^ 2
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
                // (x - mean)
                if (!simplified_) {
                    Adds(infloat_local, infloat_local, cur_mean, bkW_);
                }
                // (x - mean) / rsqrt(var)
                Muls(infloat_local, infloat_local, cur_var, bkW_);
                Mul(infloat_local, infloat_local, gammaFloatLm_, bkW_);
                Add(infloat_local, infloat_local, betaFloatLm_, bkW_);
                if (clamp_) {
                    ClampMax(infloat_local, infloat_local,  (float)(clamp_value_), bkW_);
                    ClampMin(infloat_local, infloat_local, (float)(-1 * clamp_value_), bkW_);
                }

                Cast(out_lyn_local, infloat_local, RoundMode::CAST_ROUND, bkW_);
                inQueInOneThr_.FreeTensor(in_onethr_local);

                outQueOutAdd_.EnQue(out_add_local);
                outQueOutLyn_.EnQue(out_lyn_local);
            }

            __aicore__ inline void CopyOut(int64_t progress)
            {
                LocalTensor<T> out_add_local = outQueOutAdd_.DeQue<T>();
                LocalTensor<T> out_lyn_local = outQueOutLyn_.DeQue<T>();

                int64_t offset = (progress * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                DataCopyPad(outAddGm_[offset], out_add_local, copy_pas);
                DataCopyPad(outLynGm_[offset], out_lyn_local, copy_pas);

                outQueOutAdd_.FreeTensor(out_add_local);
                outQueOutLyn_.FreeTensor(out_lyn_local);
            }

        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueInOneThr_;
            TQue<QuePosition::VECIN, 1> inQueGamma_, inQueBeta_, inQueInTwo_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOutLyn_, outQueOutAdd_;

            GlobalTensor<T> inOneGm_, inTwoGm_, inThrGm_, gammaGm_, betaGm_;
            GlobalTensor<T> outLynGm_, outAddGm_;

            LocalTensor<T> gammaLm_, betaLm_, inTwoLm_;
            LocalTensor<float> gammaFloatLm_, betaFloatLm_, inTwoLmfp32_;
            TBuf<QuePosition::VECCALC> bufQueWork_;
            TBuf<QuePosition::VECCALC> bufQueSum_;

            int64_t blockNum_, blockIdx_;
            int64_t gbH_, gbW_;
            
            int64_t bkH_;
            int64_t bkW_, bkAlignW_;
            int64_t bkLoop_;

            // int64_t tlH_;
            // int64_t tlW_, tlTailW_, tlAlignTailW_;
            // int64_t tlLoop_;

            float epsilon_ = (float)0.0f;
            float invert_ = (float)0.0f;
            bool clamp_ = false;
            int64_t clamp_value_ = 65534;
            int32_t simplified_ = 0;
        };

        //===================================================== small gbH (gbH <= AI-CORES number) singlecore
        template<typename T>
        class MOE_LAYERNORM_ADDADD2D_SINGLECORE_SH {
        public:
            __aicore__ inline MOE_LAYERNORM_ADDADD2D_SINGLECORE_SH() {}

            __aicore__  inline void Init(const int64_t gbH, const int64_t gbW, const float epsilon,
                    GM_ADDR inOne, GM_ADDR inTwo, GM_ADDR inThr, GM_ADDR gamma, GM_ADDR beta,
                    GM_ADDR outLyn, GM_ADDR outAdd, const bool clamp, int64_t clamp_value, int32_t simplified)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();

                gbH_ = gbH;
                gbW_ = gbW;
                epsilon_ = epsilon;
                invert_ = (float)(1.0) / gbW;
                clamp_  = clamp;
                clamp_value_ = clamp_value;
                simplified_ = simplified;

                bkH_ = 1;
                bkW_ = gbW_;
                bkAlignW_ = AlignUp(bkW_, 64);
                bkLoop_ = (int64_t)(gbH_ / blockNum_);
                if (gbH_ % blockNum_ != 0) {
                    bkLoop_ += 1;
                }

                // tlH_ = 1;
                // // tlW_ = (bkW_ <= MAX_COUNT_PER_TILE) ? bkW_ : MAX_COUNT_PER_TILE;
                // tlW_ = (bkW_ <= MAX_COUNT_PER_TILE) ? bkW_ : bkW_/2;
                // tlW_ = AlignUp(tlW_, 64);
                // tlTailW_ = bkW_ % tlW_;
                // tlAlignTailW_ = AlignUp(tlTailW_, 64);
                // tlLoop_ = (int64_t)(bkW_ / tlW_);
                // if(tlTailW_ != 0){ tlLoop_ += 1; }

                inOneGm_.SetGlobalBuffer((__gm__ T*)inOne, gbH_ * gbW_);
                inTwoGm_.SetGlobalBuffer((__gm__ T*)inTwo, gbW_);
                inThrGm_.SetGlobalBuffer((__gm__ T*)inThr, gbH_ * gbW_);
                gammaGm_.SetGlobalBuffer((__gm__ T*)gamma, gbW_);
                betaGm_.SetGlobalBuffer((__gm__ T*)beta, gbW_);
                outLynGm_.SetGlobalBuffer((__gm__ T*)outLyn, gbH_ * gbW_);
                outAddGm_.SetGlobalBuffer((__gm__ T*)outAdd, gbH_ * gbW_);

                pipe_.InitBuffer(inQueInOneThr_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(inQueInTwo_, 1, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(inQueGamma_, 1, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(inQueBeta_, 1, bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(outQueOutLyn_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);
                pipe_.InitBuffer(outQueOutAdd_, BUFFER_NUM, bkH_ * bkAlignW_ * sizeof(T) * 2);

                pipe_.InitBuffer(bufQueWork_, WORK_LOCAL_SIZE * sizeof(float));
                pipe_.InitBuffer(bufQueSum_, 16*sizeof(float));
            }

            __aicore__ inline void Process()
            {
                LocalTensor<T> in_two_local = inQueInTwo_.AllocTensor<T>();
                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(in_two_local[bkAlignW_], inTwoGm_, copy_pas, pad_pas);
                inQueInTwo_.EnQue(in_two_local);
                inTwoLm_ = inQueInTwo_.DeQue<T>();

                inTwoLmfp32_ = inTwoLm_.template ReinterpretCast<float>();
                Cast(inTwoLmfp32_, inTwoLm_[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);
                
                if (bkAlignW_ - AlignUp(bkW_, 16) != 0) {
                    Duplicate(in_two_local[bkAlignW_ + AlignUp(bkW_, 16)], (T)0.0, bkAlignW_ - AlignUp(bkW_, 16));
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
                LocalTensor<T> in_onethr_local = inQueInOneThr_.AllocTensor<T>();

                int64_t offset = (process * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                // DataCopyPadParams pad_pas{true, 0, (uint8_t)(bkAlignW_ - bkW_), 0};
                DataCopyPadParams pad_pas;

                DataCopyPad(in_onethr_local, inOneGm_[offset], copy_pas, pad_pas);
                DataCopyPad(in_onethr_local[bkAlignW_], inThrGm_[offset], copy_pas, pad_pas);
                
                inQueInOneThr_.EnQue(in_onethr_local);
            }

            __aicore__ inline void Compute(int64_t progress)
            {
                LocalTensor<T> in_onethr_local = inQueInOneThr_.DeQue<T>();
                LocalTensor<T> in_one_local   = in_onethr_local[0];
                LocalTensor<T> in_thr_local   = in_onethr_local[bkAlignW_];
                LocalTensor<float> infloat_local = in_onethr_local.template ReinterpretCast<float>();

                {
                    LocalTensor<T> gamma_local = inQueGamma_.AllocTensor<T>();
                    LocalTensor<T> beta_local = inQueBeta_.AllocTensor<T>();
                    DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                    DataCopyPadParams pad_pas;
                    DataCopyPad(gamma_local[bkAlignW_], gammaGm_, copy_pas, pad_pas);
                    DataCopyPad(beta_local[bkAlignW_], betaGm_, copy_pas, pad_pas);
                    inQueGamma_.EnQue(gamma_local);
                    inQueBeta_.EnQue(beta_local);
                }
                
                if (bkAlignW_-AlignUp(bkW_, 16) != 0) {
                    Duplicate<T>(in_one_local[AlignUp(bkW_, 16)], (T)0.0, bkAlignW_ - AlignUp(bkW_, 16));
                    Duplicate<T>(in_thr_local[AlignUp(bkW_, 16)], (T)0.0, bkAlignW_ - AlignUp(bkW_, 16));
                }
                
                LocalTensor<T> out_lyn_local = outQueOutLyn_.AllocTensor<T>();
                LocalTensor<T> out_add_local = outQueOutAdd_.AllocTensor<T>();

                LocalTensor<float> in_one_fp32_local = out_lyn_local.template ReinterpretCast<float>();
                LocalTensor<float> in_thr_fp32_local = out_add_local.template ReinterpretCast<float>();
                
                Cast(in_one_fp32_local, in_one_local, RoundMode::CAST_NONE, bkAlignW_);
                Cast(in_thr_fp32_local, in_thr_local, RoundMode::CAST_NONE, bkAlignW_);

                LocalTensor<float> sum_local     = bufQueSum_.Get<float>();
                LocalTensor<float> work_local    = bufQueWork_.Get<float>();
                
                Add(in_one_fp32_local, in_one_fp32_local, inTwoLmfp32_, bkAlignW_);  //  + fc2 bias
                Add(in_one_fp32_local, in_one_fp32_local, in_thr_fp32_local, bkAlignW_); // + attn out

                // if(clamp_){
                //     ClampMax(in_one_fp32_local, in_one_fp32_local,  (float)(clamp_value_), bkW_);
                //     ClampMin(in_one_fp32_local, in_one_fp32_local, (float)(-1 * clamp_value_), bkW_);
                // }
                Cast(out_add_local, in_one_fp32_local, RoundMode::CAST_ROUND, bkAlignW_); // +fc2 + attn out

                float cur_mean = 0.0f;
                if (!simplified_) {
                    Duplicate<float>(work_local, (float)0.0, 64);
                    BinaryRepeatParams byrt_pas = {1, 1, 1, 0, 8, 0};
                    Add(work_local, in_one_fp32_local, work_local, 64, bkAlignW_/64, byrt_pas) ;
                    WholeReduceSum(sum_local, work_local, 64, 1, 1, 1, 8);
                    Muls(sum_local, sum_local, (float)invert_, 1);
                    Muls(sum_local, sum_local, (float)(-1.0), 1);
                    cur_mean = sum_local.GetValue(0);
                }

                Cast(infloat_local, out_add_local, RoundMode::CAST_NONE, bkAlignW_);
                // (x - mean)
                if (!simplified_) {
                    Adds(infloat_local, infloat_local, cur_mean, bkW_);
                }
                // (x - mean) ^ 2
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
                // (x - mean)
                if (!simplified_) {
                    Adds(infloat_local, infloat_local, cur_mean, bkW_);
                }
                // (x - mean) / rsqrt(var)
                Muls(infloat_local, infloat_local, cur_var, bkW_);
                LocalTensor<T> gamma_local = inQueGamma_.DeQue<T>();
                LocalTensor<float> gammaFloatLm = gamma_local.template ReinterpretCast<float>();
                Cast(gammaFloatLm, gamma_local[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);

                Mul(infloat_local, infloat_local, gammaFloatLm, bkW_);
                LocalTensor<T> beta_local = inQueBeta_.DeQue<T>();
                LocalTensor<float> betaFloatLm = beta_local.template ReinterpretCast<float>();
                Cast(betaFloatLm, beta_local[bkAlignW_], RoundMode::CAST_NONE, bkAlignW_);

                Add(infloat_local, infloat_local, betaFloatLm, bkW_);
                if (clamp_) {
                    ClampMax(infloat_local, infloat_local,  (float)(clamp_value_), bkW_);
                    ClampMin(infloat_local, infloat_local, (float)(-1 * clamp_value_), bkW_);
                }
                Cast(out_lyn_local, infloat_local, RoundMode::CAST_ROUND, bkW_);

                inQueInOneThr_.FreeTensor(in_onethr_local);
                inQueGamma_.FreeTensor(gamma_local);
                inQueBeta_.FreeTensor(beta_local);

                outQueOutAdd_.EnQue(out_add_local);
                outQueOutLyn_.EnQue(out_lyn_local);
            }

            __aicore__ inline void CopyOut(int64_t progress)
            {
                LocalTensor<T> out_add_local = outQueOutAdd_.DeQue<T>();
                LocalTensor<T> out_lyn_local = outQueOutLyn_.DeQue<T>();

                int64_t offset = (progress * blockNum_ + blockIdx_) * bkW_;

                DataCopyParams copy_pas{1, (uint16_t)(bkW_ * sizeof(T)), 0, 0};
                DataCopyPad(outAddGm_[offset], out_add_local, copy_pas);
                DataCopyPad(outLynGm_[offset], out_lyn_local, copy_pas);

                outQueOutAdd_.FreeTensor(out_add_local);
                outQueOutLyn_.FreeTensor(out_lyn_local);
            }

        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueInOneThr_;
            TQue<QuePosition::VECIN, 1> inQueGamma_, inQueBeta_, inQueInTwo_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOutLyn_, outQueOutAdd_;

            GlobalTensor<T> inOneGm_, inTwoGm_, inThrGm_, gammaGm_, betaGm_;
            GlobalTensor<T> outLynGm_, outAddGm_;

            // LocalTensor<T> gammaLm_, betaLm_, inTwoLm_;
            LocalTensor<T> inTwoLm_;
            LocalTensor<float> inTwoLmfp32_;
            TBuf<QuePosition::VECCALC> bufQueWork_;
            TBuf<QuePosition::VECCALC> bufQueSum_;

            int64_t blockNum_, blockIdx_;
            int64_t gbH_, gbW_;
            
            int64_t bkH_;
            int64_t bkW_, bkAlignW_;
            int64_t bkLoop_;

            // int64_t tlH_;
            // int64_t tlW_, tlTailW_, tlAlignTailW_;
            // int64_t tlLoop_;

            float epsilon_ = (float)0.0f;
            float invert_ = (float)0.0f;
            bool clamp_ = false;
            int64_t clamp_value_ = 65534;
            int32_t simplified_ = 0;
        };

        extern "C" __global__ __aicore__ void moeLayerNormAddAdd2dSingleCore_kernel(int64_t gbH, int64_t gbW,
                float epsilon, GM_ADDR inOne, GM_ADDR inTwo, GM_ADDR inThr,
                GM_ADDR gamma, GM_ADDR beta, GM_ADDR outLyn, GM_ADDR outAdd,
                bool clamp, int64_t clamp_value, int32_t simplified, int32_t dtype)
        {
            if (gbH > GetBlockNum()) {
                TYPE_SWITCH(dtype, T, {
                MOE_LAYERNORM_ADDADD2D_SINGLECORE<T> op;
                op.Init(gbH, gbW, epsilon, inOne, inTwo, inThr, gamma, beta, outLyn, outAdd,
                        clamp, clamp_value, simplified);
                op.Process();
                });
            }
            else {
                TYPE_SWITCH(dtype, T, {
                MOE_LAYERNORM_ADDADD2D_SINGLECORE_SH<T> op;
                op.Init(gbH, gbW, epsilon, inOne, inTwo, inThr, gamma, beta, outLyn, outAdd,
                        clamp, clamp_value, simplified);
                op.Process();
                });
            }
        }


        void moeLayerNormAddAdd2dSingleCore_lanuch(int64_t blockDim, void* stream,
                                                   const int64_t gbH, const int64_t gbW, const float epsilon,
                                                   uint8_t* inOne, uint8_t* inTwo, uint8_t* inThr,
                                                   uint8_t* gamma, uint8_t* beta, uint8_t* outLyn,
                                                   uint8_t* outAdd, const bool clamp,
                                                   int64_t clamp_value, int32_t simplified, int32_t dtype)
        {
            // NOTE_STDOUT_FILE();
            
            if (gbH < blockDim) {
                blockDim = gbH;
            }

            moeLayerNormAddAdd2dSingleCore_kernel<<<blockDim, nullptr, stream>>>(gbH, gbW, epsilon,
                inOne, inTwo, inThr,
                gamma, beta, outLyn, outAdd, clamp, clamp_value, simplified, dtype);
        }

        inline int64_t alignUp(const int64_t number, const int64_t alignSize)
        {
            if (number % alignSize == 0) {
                return number;
            }
            
            return ((number / alignSize + 1) * alignSize);
        }

        int judge_moeLayerNormAddAdd2dSingleCore(const int64_t gbW, const int64_t ubSize)
        {
            constexpr int64_t BUFFER_NUM = 1;
            constexpr int64_t WORK_LOCAL_SIZE = 64;
            
            int64_t gbW_;
            int64_t bkH_;
            int64_t bkW_;
            int64_t bkAlignW_;
            
            gbW_ = gbW;

            bkH_ = 1;
            bkW_ = gbW_;
            bkAlignW_ = alignUp(bkW_, 64);
            
            float use_byte = BUFFER_NUM * bkH_ * bkAlignW_ * sizeof(half) * 6;
            use_byte += bkAlignW_ * sizeof(half) * 6;
            use_byte += (WORK_LOCAL_SIZE+16) * sizeof(float);
            
            if (use_byte > ubSize) {
                std::cout << __FUNCTION__ << ": " << "bkW_,bkAlignW_, UB = "<< bkW_ << ",
                " << bkAlignW_ << ", " << use_byte/1024 << " KB" << std::endl;
                return 1;
            }

            return 0;
        }


        int moeLayerNormAddAdd2d_lanuch(int64_t blockDim, void* stream,
                                        const int64_t gbH, const int64_t gbW,
                                        const float epsilon, uint8_t* inOne,
                                        uint8_t* inTwo, uint8_t* inThr, uint8_t* gamma, uint8_t* beta,
                                        uint8_t* outLyn, uint8_t* outAdd, const bool clamp,
                                        int64_t clamp_value, int32_t simplified, int32_t dtype)
        {
            int64_t ubSize = 188*1024;
            
            int ret = judge_moeLayerNormAddAdd2dSingleCore(gbW, ubSize);
            if (ret == 0) {
                moeLayerNormAddAdd2dSingleCore_lanuch(blockDim, stream, gbH, gbW, epsilon,
                                                      inOne, inTwo, inThr, gamma, beta, outLyn, outAdd,
                                                      clamp, clamp_value, simplified, dtype);
                return 0;
            }
            
            // NOTE_UB_LIMIT_ERROR();
            std::cout << __FUNCTION__ << ": " << "UB size is limited, please check!" << std::endl;
            return 1;
        }

        int64_t moe_layernorm_add_add_npu(int64_t blockDim, const int64_t gbH, const int64_t gbW,
                                          const double epsilon, torch::Tensor &inOne, torch::Tensor &inTwo,
                                          torch::Tensor &inThr, torch::Tensor &gamma, torch::Tensor &beta,
                                          torch::Tensor &outLyn, torch::Tensor &outAdd, const bool clamp,
                                          int64_t clamp_value, int64_t simplified, int64_t dtype)
                                          {
            TORCH_CHECK(torch_npu::utils::is_npu(inOne), "input one tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(inTwo), "input two tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(inThr), "input three tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(gamma), "gamma tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(beta), "beta tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(outLyn), "outLyn tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(outAdd), "outAdd tensor must be on NPU device");
            TORCH_CHECK(gbW <= 16384, "gbW (hidden_size) exceeds maximum allowed value of 16384, but got ", gbW);

            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            int launchStatus = 0;
            auto acl_call = [=, &launchStatus]() -> int {
                launchStatus = moeLayerNormAddAdd2d_lanuch(blockDim, stream, gbH, gbW, epsilon,
                    (uint8_t *)inOne.data_ptr(), (uint8_t *)inTwo.data_ptr(), (uint8_t *)inThr.data_ptr(),
                    (uint8_t *)gamma.data_ptr(), (uint8_t *)beta.data_ptr(), (uint8_t *)outLyn.data_ptr(),
                    (uint8_t *)outAdd.data_ptr(), clamp, clamp_value, simplified, dtype);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("MoeLayernormAddAdd", acl_call);

            return launchStatus;
        }

        TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
        {
            m.impl("moe_layernorm_add_add", moe_layernorm_add_add_npu);
        }
    }
}