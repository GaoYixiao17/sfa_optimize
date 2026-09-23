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
#include "kernel_operator.h"
#include "moe_routing.h"
#include "dtype_convert.h"

namespace ascend_ops {
    namespace FusedExpert {
        using namespace AscendC;
        const int BUFFER_NUM = 2;
        template<typename T>
        class FUSED_EXPERT {
        public:
            __aicore__ inline KernelExpertId() {}
            __aicore__ inline void Init(const GM_ADDR expertId, GM_ADDR fused_expertId, const GM_ADDR topk_logit,
                                        GM_ADDR fused_topk_logit, int32_t pos, int32_t expertNum,
                                        int32_t shared_expert, int32_t sharedId_dp,  int32_t thread_count,
                                        int32_t ROWS, int32_t per_count, int32_t start_tokenidx, int32_t end_tokenidx,
                                        int32_t block_cnt)
            {
                if (block_cnt <= 0) {
                    return;
                }
                BASE_SIZE = 32/sizeof(int32_t); // 8
                BASE_SIZE_TOPK = 32/sizeof(T); // 16  or T 为float时是 8
                core_id_ = GetBlockIdx();
                TILE_LENGTH = per_count; // 14 或者 7
                TILE_LENGTH_OUT = per_count + shared_expert; //
                per_dim_ = per_count; //  14 或者 7
                per_dim_out_ = per_count + shared_expert;
                block_cnt_ = block_cnt;
                shared_expertNum = expertNum;
                scalar_ = pos;
                thread_count_ = thread_count;
                start_tokenidx_ = start_tokenidx;
                end_tokenidx_ = end_tokenidx;
                int loop_count = int(ROWS / block_cnt_);
                shared_expert_ = shared_expert;
                sharedId_dp_ = sharedId_dp;
                int tail_count = ROWS - (loop_count * block_cnt_); // 0
                if (core_id_ < tail_count) {
                    loop_count++;
                }
                loop_count_ = loop_count;
                align_cnt = DivCeil(per_dim_, BASE_SIZE) * BASE_SIZE;
                align_expert_out_cnt = DivCeil(per_dim_out_, BASE_SIZE) * BASE_SIZE;    // 对齐16 有效数16 or 对齐16 有效数 8
                align_topk_logit_cnt = DivCeil(per_dim_out_, BASE_SIZE_TOPK) * BASE_SIZE_TOPK;
                // 16 or  对齐8 有效数8（float）
                expertId_Gm.SetGlobalBuffer((__gm__ int32_t*) expertId, align_expert_out_cnt);  // NUMROWS, 16
                fused_expertId_Gm.SetGlobalBuffer((__gm__ int32_t*) fused_expertId, align_expert_out_cnt);
                topk_logit_Gm.SetGlobalBuffer((__gm__ T*) topk_logit, align_topk_logit_cnt);
                fused_topk_logit_Gm.SetGlobalBuffer((__gm__ T*) fused_topk_logit, align_topk_logit_cnt);
                pipe.InitBuffer(inQueue_expertId, BUFFER_NUM,  align_expert_out_cnt * sizeof(int32_t));
                pipe.InitBuffer(inQueue_topklogit, BUFFER_NUM,  align_topk_logit_cnt * sizeof(T));
                pipe.InitBuffer(work_buf,  align_cnt * sizeof(int32_t));
                // pipe.InitBuffer(work_buf1,  align_cnt * sizeof(int32_t));
            }
            __aicore__ inline void Process()
            {
                for (int i = 0; i<loop_count_; ++i)
                {
                    CopyOut(i);
                }
            }
        private:
            __aicore__ inline void CopyOut(int32_t loop_cnt)
            {
                global_rows_index_ = loop_cnt * block_cnt_ + core_id_;   // 按token遍历
                int32_t share_expertId[2] = {0}; // 最多支持2共享专家， 即每组2个
                int32_t startId = sharedId_dp_;
                // 初始化共享专家id
                for (int32_t i = 0; i< shared_expert_; i++)
                {
                    share_expertId[i] =  startId + i;   // dp0 : 0、1、、、sharedexpert-1 dp1：

                    if (global_rows_index_ < start_tokenidx_ || global_rows_index_ >= end_tokenidx_)  // 将不在当前dp上的token
                    {
                        if (sharedId_dp_ == 0)
                        {
                            share_expertId[i] += shared_expertNum;
                        }else
                        {
                            share_expertId[i] -= shared_expertNum;
                        }
                    }
                }
                LocalTensor<int32_t> expertIdLocal = inQueue_expertId.AllocTensor<int32_t>();
                LocalTensor<T> topklogitLocal = inQueue_topklogit.AllocTensor<T>();
                LocalTensor<int32_t> offsetLocal = work_buf.Get<int32_t>();
                // index:
                DataCopyParams copy_pas1{1, (uint16_t)(TILE_LENGTH * sizeof(T)), 0, 0};   // 拷入14个数 or 9
                DataCopyPadParams pad_pas1;
                int64_t offset = (int64_t)global_rows_index_ * TILE_LENGTH;
                DataCopyPad(topklogitLocal, topk_logit_Gm[offset], copy_pas1, pad_pas1);
                // 填充到16个数
                DataCopyParams copy_pas{1, (uint16_t)(TILE_LENGTH * sizeof(int32_t)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(expertIdLocal, expertId_Gm[offset], copy_pas, pad_pas);
                // 填充到16个数
                int32_t eventId1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventId1);
                WaitFlag<HardEvent::MTE2_S>(eventId1);
                for (int32_t i = 0; i<shared_expert_; i++)
                {
                    topklogitLocal.SetValue(TILE_LENGTH + i, (T)(1.0));
                }
                int32_t eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventId);
                WaitFlag<HardEvent::S_MTE3>(eventId);
                DataCopyParams copyParams1{1, (uint16_t)(TILE_LENGTH_OUT * sizeof(T)), 0, 0};
                int64_t offset_out = (int64_t)global_rows_index_ * TILE_LENGTH_OUT;
                DataCopyPad(fused_topk_logit_Gm[offset_out], topklogitLocal, copyParams1);
                eventId1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventId1);
                WaitFlag<HardEvent::MTE2_V>(eventId1);
                ShiftRight(offsetLocal, expertIdLocal, (int32_t)scalar_, align_cnt);
                //==divs(expertNum:32) 得到 expertid 属于的 device id
                Adds(offsetLocal, offsetLocal, 1, align_cnt);
                Muls(offsetLocal, offsetLocal, shared_expert_, align_cnt);
                // (device_id + 1) * shared_expert_   路由专家新增共享专家后需要的偏移,实际是前有多少个共享专家
                // 构造offsetLocal
                Add(expertIdLocal, expertIdLocal, offsetLocal, align_cnt);
                eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventId);
                WaitFlag<HardEvent::V_S>(eventId);
                for (int32_t i = 0; i< shared_expert_; i++)
                {
                    expertIdLocal.SetValue(TILE_LENGTH + i, share_expertId[i]);  // 设置共享专家id
                }
                eventId = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                SetFlag<HardEvent::S_MTE3>(eventId);
                WaitFlag<HardEvent::S_MTE3>(eventId);

                DataCopyParams copyParams{1, (uint16_t)(TILE_LENGTH_OUT * sizeof(int32_t)), 0, 0};
                DataCopyPad(fused_expertId_Gm[offset_out], expertIdLocal, copyParams);

                inQueue_expertId.FreeTensor(expertIdLocal);
                inQueue_topklogit.FreeTensor(topklogitLocal);
                work_buf.FreeTensor(offsetLocal);
            }
        private:
            TPipe pipe;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_expertId, inQueue_topklogit;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_out;
            TBuf<QuePosition::VECCALC>  work_buf, work_buf1;
            GlobalTensor<int32_t> expertId_Gm, fused_expertId_Gm;
            GlobalTensor<T> topk_logit_Gm, fused_topk_logit_Gm;
            LocalTensor<int32_t> indexoffsetLocal_;
            int32_t core_id_, block_cnt_, shared_expertNum, shared_expert_, thread_count_;
            int32_t TILE_LENGTH, TILE_LENGTH_OUT;
            int32_t per_dim_, per_dim_out_;
            int32_t BASE_SIZE, BASE_SIZE_TOPK;
            int32_t align_cnt, align_topk_logit_cnt, align_expert_out_cnt;
            int32_t loop_count_, dim_loop_count_;
            int32_t global_rows_index_;
            int32_t scalar_, sharedId_dp_, start_tokenidx_, end_tokenidx_;
        };
        extern "C" __global__ __aicore__ void compute_fused_expert(GM_ADDR expertId, GM_ADDR fused_expertId,
                                                                GM_ADDR topk_logit, GM_ADDR fused_topk_logit,
                                                                int32_t pos, int32_t expertNum, int32_t shared_expert,
                                                                int32_t sharedId_dp, int32_t thread_count, int32_t ROWS,
                                                                int32_t per_count, int32_t start_tokenidx,
                                                                int32_t end_tokenidx, int32_t block_cnt, int32_t dtype)
        {
            KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
            TYPE_SWITCH(dtype, T, {
                KernelExpertId<T> op;
                op.Init(expertId, fused_expertId, topk_logit, fused_topk_logit, pos, expertNum, shared_expert,
                        sharedId_dp, thread_count,  ROWS, per_count, start_tokenidx,  end_tokenidx, block_cnt);
                op.Process();
            });
        }
        void fused_expert_launch(uint8_t* expertId, uint8_t* fused_expertId, uint8_t* topk_logit,
                                 uint8_t* fused_topk_logit, int32_t pos, int32_t expertNum, int32_t shared_expert,
                                 int32_t sharedId_dp, int32_t thread_count, int32_t ROWS, int32_t per_count,
                                 int32_t start_tokenidx, int32_t end_tokenidx, int32_t blockDim,
                                 void* stream, int32_t dtype)
        {
            /*
            分配的共享专家id 为0,1， 但是填充在expertId后面， topklogit同理
            thread_count_: dp域内卡数， 即tp
            per_count = 14 非共享专家数, 或者 7(30B)
            expertNum： 增加共享专家后每张卡分到专家数， 16card: （初始256专家 + 2 * 16 ）/ 16 = 18
            shared_expert = 2 ， 4 共享专家数  , 或者1(30b)
            ROWS: B * S
            支持score 为 float
            */
            int32_t real_blockDim = ROWS < blockDim ? ROWS: blockDim;
            compute_fused_expert<<<real_blockDim, nullptr, stream>>> (expertId, fused_expertId, topk_logit,
                                                                      fused_topk_logit, pos, expertNum, shared_expert,
                                                                      sharedId_dp, thread_count, ROWS, per_count,
                                                                      start_tokenidx, end_tokenidx, real_blockDim,
                                                                      dtype);
        }
        int64_t fused_expert_npu(int64_t block_dim, torch::Tensor &expertId, torch::Tensor &fused_expertId,
                                 torch::Tensor &topk_logit, torch::Tensor &fused_topk_logit, int64_t pos,
                                 int64_t expertNum, int64_t shared_expert, int64_t sharedId_dp,
                                 int64_t thread_count, int64_t ROWS, int64_t per_count,
                                 int64_t start_tokenidx, int64_t end_tokenidx, int64_t dtype)
        {
                TORCH_CHECK(torch_npu::utils::is_npu(expertId), "expertId tensor must be on NPU device");
                TORCH_CHECK(torch_npu::utils::is_npu(fused_expertId), "fused_expertId tensor must be on NPU device");
                TORCH_CHECK(torch_npu::utils::is_npu(topk_logit), "topk_logit tensor must be on NPU device");
                TORCH_CHECK(torch_npu::utils::is_npu(fused_topk_logit), "fused_topk_logit must be on NPU device");
                TORCH_CHECK(dtype == 1 || dtype == 27, "unsupported dtype");
                TORCH_CHECK(expertId.scalar_type() == torch::kInt32, "expertId must be int32");
                TORCH_CHECK(topk_logit.scalar_type() == torch::kBFloat16 || topk_logit.scalar_type() ==
                            torch::kFloat16, "unsupported logit dtype");
                TORCH_CHECK(per_count == 7 || per_count == 8 || per_count == 16, "per_count must be 7/8/16");
                auto stream = c10_npu::getCurrentNPUStream().stream(false);
                auto acl_call = [=]() -> int {
                    fused_expert_launch((uint8_t *)expertId.data_ptr(), (uint8_t *)fused_expertId.data_ptr(),
                                        (uint8_t *)topk_logit.data_ptr(), (uint8_t *)fused_topk_logit.data_ptr(),
                                        static_cast<int32_t>(pos), static_cast<int32_t>(expertNum),
                                        static_cast<int32_t>(shared_expert), static_cast<int32_t>(sharedId_dp),
                                        static_cast<int32_t>(thread_count), static_cast<int32_t>(ROWS),
                                        static_cast<int32_t>(per_count), static_cast<int32_t>(start_tokenidx),
                                        static_cast<int32_t>(end_tokenidx), static_cast<int32_t>(block_dim),
                                        stream, static_cast<int32_t>(dtype));
                    return 0;
                };
                at_npu::native::OpCommand::RunOpApi("FusedExpert", acl_call);
                return 0;
                }
        }  // 关闭 FusedExpert 命名空间

        TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
        {
            m.impl("fused_expert", FusedExpert::fused_expert_npu);
        }
}