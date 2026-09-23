/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/all.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
    namespace MOEGateGeluClamp {
#include <iostream>
#include "kernel_operator.h"
        using namespace AscendC;
        constexpr int32_t BUFFER_NUM = 2;
#define UB_BUFFER_SIZE 192512  // 188KB

        template<typename T>
        class MOE_GATE_GELU_CLAMP {
        public:
            __aicore__ inline MOE_GATE_GELU_CLAMP() {}
            __aicore__ inline void InitTOTAL(int32_t per_total_count, GM_ADDR total_rows, int32_t block_cnt)
            {
                core_id_ = GetBlockIdx();
                real_rows_ = -1;
                BASE_SIZE_TOTAL = 32/sizeof(int64_t); // 32B / 2B = 16 个数
                TILE_LENGTH_TOTAL = per_total_count; // 256 / thread_count
                per_dim_total_ = per_total_count; // 256 / thread_count
                align_total_cnt = DivCeil(per_dim_total_, BASE_SIZE_TOTAL) * BASE_SIZE_TOTAL;
                total_rows_Gm.SetGlobalBuffer((__gm__ int64_t*) total_rows, align_total_cnt);
                pipe.InitBuffer(inQueue_total_rows, 2,  align_total_cnt * sizeof(int64_t));
            }
            __aicore__ inline void Process_Total()
            {
                LocalTensor<int64_t> total_rowsLocal = inQueue_total_rows.AllocTensor<int64_t>();
                DataCopyParams copy_pas{1, (uint16_t)(TILE_LENGTH_TOTAL * sizeof(int64_t)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(total_rowsLocal, total_rows_Gm[0], copy_pas, pad_pas);
                int32_t eventId1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventId1);
                WaitFlag<HardEvent::MTE2_S>(eventId1);
                real_rows_ = total_rowsLocal.GetValue(TILE_LENGTH_TOTAL-1);
                inQueue_total_rows.FreeTensor(total_rowsLocal);
            }
            __aicore__ inline void Init(int32_t ROWS, int32_t per_count, GM_ADDR x, GM_ADDR z, bool is_clamp,
                                        float clamp_value, int32_t block_cnt)
            {
                core_id_ = GetBlockIdx();
                BASE_SIZE = 32/sizeof(T); // 32B / 2B = 16 个数
                ROWS = real_rows_ == -1 ? ROWS : real_rows_;
                block_cnt_ = ROWS < block_cnt ? ROWS: block_cnt;  // 置真实core数
                if (core_id_ < block_cnt_) {
                    TILE_LENGTH = per_count;  // 2048
                    per_dim_ = per_count / 2;   // 1024
                    clamp_value_ = clamp_value;
                    is_clamp_ = is_clamp;
                    int32_t rows_per_core = int(ROWS / block_cnt_);
                    rows_per_core_per_loop_ = 1;
        
                    align_cnt = DivCeil(per_dim_, BASE_SIZE) * BASE_SIZE;
                
                    if (rows_per_core >= 2) {
                        int32_t limit_rows_per_core = UB_BUFFER_SIZE / (align_cnt * BUFFER_NUM * 6 + align_cnt * 8);
                        rows_per_core_per_loop_ = min(rows_per_core, limit_rows_per_core);
                    }

                    tail_rows_last_loop_per_core_ = rows_per_core_per_loop_;
                
                    int32_t loop_count = ROWS / (rows_per_core_per_loop_ * block_cnt_);  // 4578 / 30 * 40 = 3

                    full_loop_count_ = loop_count;
                    full_rows_offset_ = full_loop_count_ * rows_per_core_per_loop_ * block_cnt_ * TILE_LENGTH;
                    rows_offset_ = 0;
                    int32_t tail_count = ROWS - (full_loop_count_ * block_cnt_ * rows_per_core_per_loop_);
                    if (tail_count > 0) {
                        loop_count += 1;
                    }
                    tail_rows_last_loop_per_core_ = tail_count / block_cnt_;  // 24 or 0
                    
                    tail_count = tail_count % block_cnt_; // 18 or 0
                    tail_rows_offset_ = tail_count * (tail_rows_last_loop_per_core_ + 1) * TILE_LENGTH;
                    if (core_id_ < tail_count) {
                        tail_rows_last_loop_per_core_ +=1 ;
                    }
                    
                    loop_count_ = loop_count;
                    tail_count_ = tail_count;

                    X_Gm.SetGlobalBuffer((__gm__ T*) x, rows_per_core_per_loop_ * align_cnt);
                    Z_Gm.SetGlobalBuffer((__gm__ T*) z, rows_per_core_per_loop_ * align_cnt);
                    
                    pipe.InitBuffer(inQueue_X, BUFFER_NUM,  rows_per_core_per_loop_ * align_cnt * sizeof(T));
                    pipe.InitBuffer(inQueue_Y, BUFFER_NUM,  rows_per_core_per_loop_ * align_cnt * sizeof(T));
                    pipe.InitBuffer(outQueue_Z, BUFFER_NUM,  rows_per_core_per_loop_ * align_cnt * sizeof(T));
                    pipe.InitBuffer(work_buf,  rows_per_core_per_loop_ * align_cnt  * sizeof(float));
                    pipe.InitBuffer(work_buf1,  rows_per_core_per_loop_ * align_cnt  * sizeof(float));
                }
            }
            __aicore__ inline void Process()
            {
                if (core_id_ < block_cnt_) {
                    for (int i = 0;i<loop_count_; ++i) {
                            CopyIn(i);
                            Compute(i);
                            CopyOut(i);
                    }
                }
            }

        private:
            __aicore__ inline void CopyIn(int32_t loop_cnt)
            {
                LocalTensor<T> xleftLocal = inQueue_X.AllocTensor<T>();
                LocalTensor<T> xrightLocal = inQueue_Y.AllocTensor<T>();
                
                global_core_index_ = loop_cnt * block_cnt_ + core_id_;
                if (loop_cnt < full_loop_count_) {
                    rows_offset_ = global_core_index_ * rows_per_core_per_loop_ * TILE_LENGTH;
                } else {
                    rows_per_core_per_loop_ = tail_rows_last_loop_per_core_;
                    if (core_id_ < tail_count_) {
                        rows_offset_ = full_rows_offset_ + core_id_ * rows_per_core_per_loop_ * TILE_LENGTH;
                    } else {
                        rows_offset_ = full_rows_offset_ + tail_rows_offset_ + (core_id_ - tail_count_) *
                        rows_per_core_per_loop_ * TILE_LENGTH;
                    }
                }

                DataCopyParams copyParams{(uint16_t)rows_per_core_per_loop_,  (uint16_t)(TILE_LENGTH / 2 * sizeof(T)),
                                        (uint16_t)(TILE_LENGTH / 2 * sizeof(T)), 0};
                DataCopyPadParams padParams;
                DataCopyPad(xleftLocal, X_Gm[rows_offset_], copyParams, padParams);
                DataCopyPad(xrightLocal, X_Gm[rows_offset_ + TILE_LENGTH / 2], copyParams, padParams);
                inQueue_X.EnQue(xleftLocal);
                inQueue_Y.EnQue(xrightLocal);
            }
            __aicore__ inline void Compute(int32_t loop_cnt)
            {
                LocalTensor<T> xleftLocal = inQueue_X.DeQue<T>();
                LocalTensor<T> xrightLocal = inQueue_Y.DeQue<T>();

                LocalTensor<T> zLocal = outQueue_Z.AllocTensor<T>();
                LocalTensor<float> xleftfp32Local = work_buf.Get<float>();
                LocalTensor<float> xrightfp32Local = work_buf1.Get<float>();
                Cast(xleftfp32Local, xleftLocal, RoundMode::CAST_NONE, rows_per_core_per_loop_ * align_cnt);
                Cast(xrightfp32Local, xrightLocal, RoundMode::CAST_NONE, rows_per_core_per_loop_ * align_cnt);
                Gelu<float, false, true>(xleftfp32Local, xleftfp32Local, rows_per_core_per_loop_ * align_cnt);
                Mul(xleftfp32Local, xleftfp32Local, xrightfp32Local, rows_per_core_per_loop_ * align_cnt);
                if (is_clamp_) {
                    Mins(xleftfp32Local, xleftfp32Local, (float)(clamp_value_), rows_per_core_per_loop_ * align_cnt);
                    Maxs(xleftfp32Local, xleftfp32Local, (float)(-1*clamp_value_), rows_per_core_per_loop_ * align_cnt);
                }
                Cast(zLocal, xleftfp32Local, RoundMode::CAST_ROUND, rows_per_core_per_loop_ * align_cnt);

                outQueue_Z.EnQue(zLocal);

                inQueue_X.FreeTensor(xleftLocal);
                inQueue_Y.FreeTensor(xrightLocal);
                work_buf.FreeTensor(xleftfp32Local);
                work_buf1.FreeTensor(xrightfp32Local);
            }
            __aicore__ inline void CopyOut(int32_t loop_cnt)
            {
                LocalTensor<T> zLocal = outQueue_Z.DeQue<T>();
                DataCopyParams copyParams{(uint16_t)1, (uint16_t)(rows_per_core_per_loop_ * align_cnt * sizeof(T)),
                                          0, 0};
                DataCopyPad(Z_Gm[rows_offset_ / 2], zLocal, copyParams);
                outQueue_Z.FreeTensor(zLocal);
            }

        private:
            
            TPipe pipe;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_X, inQueue_Y, inQueue_total_rows;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_Z;
            TBuf<QuePosition::VECCALC> work_buf1, work_buf;
            GlobalTensor<T> X_Gm;
            GlobalTensor<T> Z_Gm;
            GlobalTensor<int64_t> total_rows_Gm;
        
            int32_t core_id_ ;
            int32_t TILE_LENGTH, TILE_LENGTH_TOTAL;
            int32_t per_dim_, per_dim_total_;
            int32_t BASE_SIZE, BASE_SIZE_TOTAL, block_cnt_;
            int32_t align_cnt, align_total_cnt, real_align_cnt;
            int32_t tail_count_, full_rows_offset_, tail_rows_offset_ ;
            int32_t real_rows_;
            float clamp_value_;
            bool is_clamp_;
            int32_t loop_count_, full_loop_count_, dim_loop_count_, rows_per_core_per_loop_,
                    tail_rows_last_loop_per_core_;
            int32_t global_core_index_, core_rows_, rows_offset_;
        };
        extern "C" __global__ __aicore__ void moe_gate_gelu_clamp_kernel(GM_ADDR x, GM_ADDR total_rows, GM_ADDR z,
                                                                         int32_t NUM_ROWS, int32_t per_count,
                                                                         int32_t per_total_count, bool is_clamp,
                                                                         float clamp_value, int32_t block_cnt,
                                                                         int32_t dtype, bool use_total_rows)
        {
            TYPE_SWITCH(dtype, T, {
            MOE_GATE_GELU_CLAMP<T> op;
            op.InitTOTAL(per_total_count, total_rows, block_cnt);
            if (use_total_rows) {
                op.Process_Total();
            }
            op.Init(NUM_ROWS, per_count, x, z, is_clamp, clamp_value, block_cnt);
            op.Process();
            });
        }
        void moe_gate_gelu_clamp_lanuch(unsigned char* x, unsigned char* total_rows, unsigned char* out,
            int32_t NUM_ROWS, int32_t per_count, int32_t per_total_count,  bool is_clamp, float clamp_value,
            int32_t blockDim,  void* stream, int32_t dtype, bool use_total_rows)
        {
            int32_t real_blockDim = NUM_ROWS < blockDim ? NUM_ROWS: blockDim;
            moe_gate_gelu_clamp_kernel<<<real_blockDim, nullptr, stream>>>(x, total_rows, out, NUM_ROWS, per_count,
            per_total_count, is_clamp, clamp_value, real_blockDim, dtype, use_total_rows);
        }  // 根据实际计算量计算gate gelu
        int64_t moe_gate_gelu_clamp_npu(int64_t block_dim, torch::Tensor &x, torch::Tensor &total_rows,
                                        torch::Tensor &out, int64_t NUM_ROWS, int64_t per_count,
                                        int64_t per_total_count, bool is_clamp, double clamp_value, int64_t dtype,
                                        bool use_total_rows)
        {
                TORCH_CHECK(torch_npu::utils::is_npu(x), "x tensor must be on NPU device");
                TORCH_CHECK(torch_npu::utils::is_npu(out), "out tensor must be on NPU device");
                TORCH_CHECK(NUM_ROWS > 0, "NUM_ROWS must be positive");

                if (use_total_rows) {
                TORCH_CHECK(torch_npu::utils::is_npu(total_rows), "total_rows tensor must be on NPU device");
                }
                TORCH_CHECK(per_count > 0 && per_count % 2 == 0, "per_count must be positive and even");
                TORCH_CHECK(per_total_count > 0, "per_total_count must be positive");
                TORCH_CHECK(dtype == 1 || dtype == 27, "unsupported dtype, only half(1) and bfloat16(27) supported");
                TORCH_CHECK(per_count <= 8192, "Hidden dimension exceeds 8192 limit");

                auto stream = c10_npu::getCurrentNPUStream().stream(false);
                auto acl_call = [=]() -> int {
                    moe_gate_gelu_clamp_lanuch((unsigned char *)x.data_ptr(),
                        use_total_rows ? (unsigned char *)total_rows.data_ptr() : nullptr,
                        (unsigned char *)out.data_ptr(),
                        static_cast<int32_t>(NUM_ROWS),
                        static_cast<int32_t>(per_count),
                        static_cast<int32_t>(per_total_count),
                        is_clamp,
                        static_cast<float>(clamp_value),
                        static_cast<int32_t>(block_dim),
                        stream,
                        static_cast<int32_t>(dtype),
                        use_total_rows);
                    return 0;
                };
                at_npu::native::OpCommand::RunOpApi("MOEGateGeluClamp", acl_call);
                return 0;
        }
        TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
        {
                m.impl("moe_gate_gelu_clamp", moe_gate_gelu_clamp_npu);
        }
    }
}