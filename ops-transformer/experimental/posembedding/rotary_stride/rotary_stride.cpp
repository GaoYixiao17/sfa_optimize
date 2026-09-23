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
namespace RotaryStride {
#include <iostream>
#include <stdio.h>
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"
#include "dtype_convert.h"

        using namespace AscendC;

        constexpr int64_t UB_MAX_BYTES = 184*1024;
        constexpr int64_t BUFFER_NUM = 1;

        template<typename T>
        class rotary_stride {
        public:
            __aicore__ inline rotary_stride() {}

            __aicore__ inline void Init(__gm__ void* in, __gm__ void* sin, __gm__ void* cos,
                    __gm__ void* psql, __gm__ void* out,
                    const int64_t gbB, const int64_t gbS, const int64_t gbN, const int64_t gbD, const int64_t gbMAXS,
                    const int64_t stride)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();

                gbB_ = gbB;
                gbS_ = gbS;
                gbN_ = gbN;
                gbD_ = gbD;
                gbMAXS_ = gbMAXS;
                stride_ = stride;
                gbAlignB_ = AlignUp(gbB_, 256/sizeof(int32_t));

                bkB_ = 1;
                bkS_ = 1;
                bkN_ = gbN_;
                bkD_ = gbD_;
                bkHalfD_ = bkD_ / 2;
                bkAlignHalfD_ = AlignUp(bkHalfD_, 256/sizeof(T));
                bkLoop_ = gbB_*gbS_ / blockNum_;
                if (gbB_*gbS_ % blockNum_ != 0){ bkLoop_ += 1; }

                int64_t used_bytes = UB_MAX_BYTES - bkAlignHalfD_ * 2 * sizeof(T) * 2;
                used_bytes -= bkAlignHalfD_ * 2 * sizeof(float) * 2;
                used_bytes -= gbAlignB_ * sizeof(int32_t);

                bkMaxN_ = used_bytes / ((bkAlignHalfD_ * 2 * sizeof(T) * 2) + (bkAlignHalfD_ * 2 * sizeof(float) * 4));

                gmIn_.SetGlobalBuffer((__gm__ T*)(in), gbB_ * gbS_ * gbN_ * gbD_);
                gmSin_.SetGlobalBuffer((__gm__ T*)(sin), gbMAXS_ * gbD_);
                gmCos_.SetGlobalBuffer((__gm__ T*)(cos), gbMAXS_ * gbD_);
                gmPsql_.SetGlobalBuffer((__gm__ int32_t*)(psql), gbB_);
                gmOut_.SetGlobalBuffer((__gm__ T*)(out), gbB_ * gbS_ * gbN_ * gbD_);

                pipe_.InitBuffer(inQueIn_,    BUFFER_NUM, bkMaxN_ * bkAlignHalfD_ * 2 * sizeof(T));
                pipe_.InitBuffer(inQueSin_,   BUFFER_NUM, bkAlignHalfD_ * 2 * sizeof(T));
                pipe_.InitBuffer(inQueCos_,   BUFFER_NUM, bkAlignHalfD_ * 2 * sizeof(T));

                pipe_.InitBuffer(local_in_f_,    bkMaxN_ * bkAlignHalfD_ * 2 * sizeof(float));
                pipe_.InitBuffer(local_out_f_,   bkMaxN_ * bkAlignHalfD_ * 2 * sizeof(float));
                pipe_.InitBuffer(local_sin_f_,   bkAlignHalfD_ * 2 * sizeof(float));
                pipe_.InitBuffer(local_cos_f_,   bkAlignHalfD_ * 2 * sizeof(float));

                pipe_.InitBuffer(outQueOut_,  BUFFER_NUM, bkMaxN_ * bkAlignHalfD_ * 2 * sizeof(T));
                pipe_.InitBuffer(inQuePsql_,  1,          gbAlignB_ * sizeof(int32_t));

                pipe_.InitBuffer(bufSinMul_, bkMaxN_ * bkAlignHalfD_ * 2 * sizeof(float));
                pipe_.InitBuffer(bufCosMul_, bkMaxN_ * bkAlignHalfD_ * 2 * sizeof(float));
            }

            __aicore__ inline void Process()
            {
                lmSinMul_ = bufSinMul_.Get<float>();
                lmCosMul_ = bufCosMul_.Get<float>();

                LocalTensor<int32_t> local_psql = inQuePsql_.AllocTensor<int32_t>();

                DataCopyExtParams copy_pas{1, (uint32_t)(gbB_*sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad_pas;
                DataCopyPad(local_psql, gmPsql_, copy_pas, pad_pas);

                inQuePsql_.EnQue(local_psql);
                lmPsql_ = inQuePsql_.DeQue<int32_t>();

                for (int64_t bkl = 0; bkl < bkLoop_; bkl++) {
                    int64_t task_index = bkl * blockNum_ + blockIdx_;

                    int64_t real_b = task_index / gbS_;
                    int64_t real_s = task_index % gbS_;

                    int32_t psql_value = lmPsql_.GetValue(real_b);
                    if ((task_index < gbB_*gbS_) && ((real_s + psql_value) >= 0)) {
                        int64_t offset_inout  = (stride_ - gbD_) + (real_b*gbS_ + real_s) * (gbN_*stride_);
                        int64_t offset_sincos = (real_s + psql_value) * gbD_;
                        
                        for (int64_t bkn = 0; bkn < bkN_; bkn += bkMaxN_) {
                            int64_t bkn_len = bkMaxN_;
                            if (bkn + bkMaxN_ > bkN_) {
                                bkn_len = bkN_ - bkn;
                            }
                            CopyIn(task_index, offset_inout, offset_sincos, bkn, bkn_len);
                            Compute(task_index, offset_inout, offset_sincos, bkn, bkn_len);
                            CopyOut(task_index, offset_inout, offset_sincos, bkn, bkn_len);
                        }
                    }
                } // bkLoop_

                inQuePsql_.FreeTensor(local_psql);
            }

        private:
            __aicore__ inline void CopyIn(int64_t task_index, int64_t offset_inout, int64_t offset_sincos,
                int64_t bkn_offset, int64_t bkn_len)
                {
                LocalTensor<T> local_in  = inQueIn_.AllocTensor<T>();
                LocalTensor<T> local_sin = inQueSin_.AllocTensor<T>();
                LocalTensor<T> local_cos = inQueCos_.AllocTensor<T>();

                DataCopyExtParams copy_pas;
                copy_pas.blockCount  = (uint16_t)(bkn_len);
                copy_pas.blockLen    = (uint32_t)(bkHalfD_ * sizeof(T));
                copy_pas.srcStride   = (uint32_t)((stride_ - bkHalfD_) * sizeof(T));
                copy_pas.dstStride   = (uint32_t)(2 * (bkAlignHalfD_*sizeof(T)/32) -
                (AlignUp(bkHalfD_*sizeof(T), 32)/32));
                DataCopyPadExtParams<T> pad_pas;

                DataCopyPad(local_in, gmIn_[offset_inout + bkn_offset * stride_], copy_pas, pad_pas);
                DataCopyPad(local_in[bkAlignHalfD_], gmIn_[offset_inout + bkn_offset * stride_ + bkHalfD_],
                    copy_pas, pad_pas);
                
                copy_pas.blockCount  = 2;
                copy_pas.blockLen    = (uint32_t)(bkHalfD_ * sizeof(T));
                copy_pas.srcStride   = 0;
                copy_pas.dstStride   = (uint32_t)((bkAlignHalfD_*sizeof(T)/32) - (AlignUp(bkHalfD_*sizeof(T), 32)/32));
                DataCopyPad(local_sin, gmSin_[offset_sincos], copy_pas, pad_pas);
                DataCopyPad(local_cos, gmCos_[offset_sincos], copy_pas, pad_pas);
                
                inQueIn_.EnQue(local_in);
                inQueSin_.EnQue(local_sin);
                inQueCos_.EnQue(local_cos);
            }

            __aicore__ inline void Compute(int64_t task_index, int64_t offset_inout, int64_t offset_sincos,
                int64_t bkn_offset, int64_t bkn_len)
                {
                LocalTensor<T> local_in  = inQueIn_.DeQue<T>();
                LocalTensor<T> local_sin = inQueSin_.DeQue<T>();
                LocalTensor<T> local_cos = inQueCos_.DeQue<T>();

                LocalTensor<float> local_in_f  = local_in_f_.Get<float>();
                LocalTensor<float> local_sin_f = local_sin_f_.Get<float>();
                LocalTensor<float> local_cos_f = local_cos_f_.Get<float>();

                Cast(local_in_f,  local_in,  RoundMode::CAST_NONE, bkn_len * bkAlignHalfD_ * 2);
                Cast(local_sin_f, local_sin, RoundMode::CAST_NONE, bkAlignHalfD_ * 2);
                Cast(local_cos_f, local_cos, RoundMode::CAST_NONE, bkAlignHalfD_ * 2);

                LocalTensor<T> local_out = outQueOut_.AllocTensor<T>();

                BinaryRepeatParams repeat_params;
                repeat_params.dstBlkStride  = 1;
                repeat_params.src0BlkStride = 1;
                repeat_params.src1BlkStride = 1;
                repeat_params.dstRepStride  = bkAlignHalfD_ * sizeof(float) * 2 / 32;
                repeat_params.src0RepStride = bkAlignHalfD_ * sizeof(float) * 2 / 32;
                repeat_params.src1RepStride = 0;
                uint64_t mask = bkHalfD_;

                for (int i = 0; i < 2; i++) {
                    Mul(lmSinMul_[i*bkAlignHalfD_], local_in_f[i*bkAlignHalfD_], local_sin_f[i*bkAlignHalfD_],
                        mask, (uint8_t)bkn_len, repeat_params);
                    Mul(lmCosMul_[i*bkAlignHalfD_], local_in_f[i*bkAlignHalfD_], local_cos_f[i*bkAlignHalfD_],
                        mask, (uint8_t)bkn_len, repeat_params);
                }

                repeat_params.dstRepStride  = bkAlignHalfD_ * sizeof(float) * 2 / 32;
                repeat_params.src0RepStride = bkAlignHalfD_ * sizeof(float) * 2 / 32;
                repeat_params.src1RepStride = bkAlignHalfD_ * sizeof(float) * 2 / 32;
                
                LocalTensor<float>local_out_f = local_out_f_.Get<float>();

                Sub(local_out_f, lmCosMul_, lmSinMul_[bkAlignHalfD_], mask, (uint8_t)bkn_len, repeat_params);
                Add(local_out_f[bkAlignHalfD_], lmCosMul_[bkAlignHalfD_], lmSinMul_, mask,
                    (uint8_t)bkn_len, repeat_params);
                
                Cast(local_out, local_out_f, RoundMode::CAST_ROUND, bkn_len * bkAlignHalfD_ * 2);

                outQueOut_.EnQue(local_out);
                
                inQueIn_.FreeTensor(local_in);
                inQueSin_.FreeTensor(local_sin);
                inQueCos_.FreeTensor(local_cos);
                local_in_f_.FreeTensor(local_in_f);
                local_sin_f_.FreeTensor(local_sin_f);
                local_cos_f_.FreeTensor(local_cos_f);
                local_out_f_.FreeTensor(local_out_f);
            }

            __aicore__ inline void CopyOut(int64_t task_index, int64_t offset_inout, int64_t offset_sincos,
                int64_t bkn_offset, int64_t bkn_len)
                {
                LocalTensor<T> local_out = outQueOut_.DeQue<T>();

                DataCopyExtParams copy_pas;
                copy_pas.blockCount  = (uint16_t)(bkn_len);
                copy_pas.blockLen    = (uint32_t)(bkHalfD_ * sizeof(T));
                copy_pas.srcStride   = (uint32_t)(2*(bkAlignHalfD_*sizeof(T)/32) -
                (AlignUp(bkHalfD_*sizeof(T), 32)/32));
                copy_pas.dstStride   = (uint32_t)((stride_ - bkHalfD_) * sizeof(T));

                DataCopyPad(gmOut_[offset_inout + bkn_offset * stride_], local_out, copy_pas);
                DataCopyPad(gmOut_[offset_inout + bkn_offset * stride_ + bkHalfD_], local_out[bkAlignHalfD_], copy_pas);

                outQueOut_.FreeTensor(local_out);
            }

        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueIn_, inQueSin_, inQueCos_;
            TQue<QuePosition::VECIN, 1> inQuePsql_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOut_;
            TBuf<QuePosition::VECCALC> bufSinMul_, bufCosMul_;
            TBuf<QuePosition::VECCALC> local_in_f_, local_sin_f_, local_cos_f_, local_out_f_;

            GlobalTensor<T> gmIn_, gmSin_, gmCos_, gmOut_;
            GlobalTensor<int32_t> gmPsql_;

            LocalTensor<int32_t> lmPsql_;
            LocalTensor<float> lmSinMul_, lmCosMul_;

            int64_t blockNum_, blockIdx_;
            int64_t gbB_, gbS_, gbN_, gbD_, gbMAXS_;
            int64_t gbAlignB_;
            
            int64_t bkB_, bkS_, bkN_;
            int64_t bkMaxN_;
            int64_t bkD_, bkHalfD_, bkAlignHalfD_;
            int64_t bkLoop_;

            int64_t stride_;
        };

        extern "C" __global__ __aicore__ void compute_rotary_stride(__gm__ void* in,
                __gm__ void* sin, __gm__ void* cos, __gm__ void* psql, __gm__ void* out,
                const int64_t gbB, const int64_t gbS, const int64_t gbN, const int64_t gbD, const int64_t gbMAXS,
                const int64_t stride, int32_t dtype)
        {
            TYPE_SWITCH(dtype, T, {
                rotary_stride<T> op;
                op.Init(in, sin, cos, psql, out, gbB, gbS, gbN, gbD, gbMAXS, stride);
                op.Process();
            })
        }

        inline int64_t align_up(const int64_t number, const int64_t alignSize)
        {
            if (number % alignSize == 0) {
                return number;
            }
            
            return ((number / alignSize + 1) * alignSize);
        }

        int judge_rotary_stride_launch(const int64_t gbB,  const int64_t gbD, const int64_t ubSize)
        {
            int64_t gbB_;
            int64_t gbD_;
            int64_t gbAlignB_;
            
            int64_t bkD_;
            int64_t bkHalfD_;
            int64_t bkAlignHalfD_;
            int64_t bkMaxN_;
            
            gbB_ = gbB;
            gbD_ = gbD;
            gbAlignB_ = align_up(gbB_, 256/sizeof(int32_t));

            bkD_ = gbD_;
            bkHalfD_ = bkD_ / 2;
            bkAlignHalfD_ = align_up(bkHalfD_, 256/sizeof(uint16_t));

            int64_t used_bytes = ubSize - bkAlignHalfD_ * 2 * sizeof(uint16_t) * 2;
            used_bytes -= bkAlignHalfD_ * 2 * sizeof(float) * 2;
            used_bytes -= gbAlignB_ * sizeof(int32_t);

            bkMaxN_ = used_bytes / ((bkAlignHalfD_ * 2 * sizeof(uint16_t) * 2) +
            (bkAlignHalfD_ * 2 * sizeof(float) * 4));

            if (bkMaxN_ < 1) {
                std::cout << __FUNCTION__ << ": bkMaxN = " << bkMaxN_ << std::endl;
                return 1;
            }

            return 0;
        }

        void rotary_stride_kernel_lanuch(int64_t blockDim, void* stream,
            void* in, void* sin, void* cos, void* psql, void* out,
            const int64_t gbB, const int64_t gbS, const int64_t gbN, const int64_t gbD, const int64_t gbMAXS,
            const int64_t stride, int32_t dtype)
        {
            if (gbB*gbS < blockDim) { blockDim = gbB*gbS; }

            compute_rotary_stride<<<blockDim, nullptr, stream>>>(in, sin, cos, psql, out,
                    gbB, gbS, gbN, gbD, gbMAXS, stride, dtype);
        }

        int rotary_stride_lanuch(int64_t blockDim, void* stream,
            void* in, void* sin, void* cos, void* psql, void* out,
            const int64_t gbB, const int64_t gbS, const int64_t gbN, const int64_t gbD, const int64_t gbMAXS,
            const int64_t stride, int32_t dtype)
        {
            int64_t ubSize = 184 * 1024;
            
            int ret = judge_rotary_stride_launch(gbB, gbD, ubSize);
            if (ret == 0) {
                rotary_stride_kernel_lanuch(blockDim, stream, in, sin, cos, psql, out, gbB, gbS, gbN,
                    gbD, gbMAXS, stride, dtype);
                return 0;
            }

            std::cout << __FUNCTION__ << ": " << "UB size is limited, please check!" << std::endl;
            return 1;
        }

        int64_t rotary_stride_npu(int64_t blockDim, torch::Tensor &in, torch::Tensor &sin, torch::Tensor &cos,
            torch::Tensor &out, const int64_t hiddenDim)
            {
            TORCH_CHECK(torch_npu::utils::is_npu(in), "input tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(sin), "sin tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(cos), "cosin tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(out), "output tensor must be on NPU device");
            TORCH_CHECK(in.scalar_type() == at::kBFloat16 || in.scalar_type() == at::kHalf,
                "dtype of input tensor is invalid, only BF16 or FP16 is supported.");
            TORCH_CHECK(out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf,
                "dtype of output tensor is invalid, only BF16 or FP16 is supported.");
            TORCH_CHECK(sin.scalar_type() == at::kBFloat16 || sin.scalar_type() == at::kHalf,
                "dtype of sin tensor is invalid, only BF16 or FP16 is supported.");
            TORCH_CHECK(cos.scalar_type() == at::kBFloat16 || cos.scalar_type() == at::kHalf,
                "dtype of cos tensor is invalid, only BF16 or FP16 is supported.");

            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            torch::Tensor psql = torch::arange(0, in.size(0) * in.size(1), in.size(1)).to(at::Device("npu"));
            int launchStatus = 0;
            auto acl_call = [=, &launchStatus]() -> int {
                launchStatus = rotary_stride_lanuch(blockDim, stream, in.data_ptr(), sin.data_ptr(), cos.data_ptr(),
                psql.data_ptr(), out.data_ptr(), in.size(0), in.size(1), in.size(2), hiddenDim, sin.size(0),
                in.size(3), in.scalar_type() == at::kHalf ? 1 : 27);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("RotaryStride", acl_call);

            return launchStatus;
        }

        TORCH_LIBRARY_IMPL(npu_ops_transformer_ext, PrivateUse1, m)
        {
            m.impl("rotary_stride", rotary_stride_npu);
        }
    }
}

