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
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"

namespace ascend_ops {
    namespace InitRouting {
#include <iostream>
#include "kernel_operator.h"
        using namespace AscendC;
        constexpr int64_t UB_MAX_BYTES = 184*1024;
        constexpr int64_t BUFFER_NUM = 1;
        class Init_Routing {
        public:
            __aicore__ inline init_routing_kernel() {}
            __aicore__ inline void Init(GM_ADDR in, GM_ADDR token_table, GM_ADDR token_list, GM_ADDR out,
                                        const int64_t expert_num, const int64_t token_num, const int64_t copy_byte)
            {
                blockNum_ = GetBlockNum();
                blockIdx_ = GetBlockIdx();
                gbExpertNum_ = expert_num;
                gbTokenNum_  = token_num;
                gbCopyByte_  = copy_byte;
                gbTokenNumAlign_  = AlignUp(gbTokenNum_, 64 / sizeof(int32_t));
                gbExpertNumAlign_ = AlignUp(gbExpertNum_, 64 / sizeof(int32_t));
                bkExpertNum_ = 1;
                bkTokenNum_  = 1;
                bkCopyByte_  = gbCopyByte_;
                bkCopyByteAlign_ = AlignUp(bkCopyByte_, 64);
                
                gmIn_.SetGlobalBuffer((__gm__ int8_t*)(in), gbTokenNum_ * gbCopyByte_);
                gmTokenTable_.SetGlobalBuffer((__gm__ int32_t*)(token_table), gbExpertNum_ * gbTokenNum_);
                gmTokenList_.SetGlobalBuffer((__gm__ int64_t*)(token_list), gbExpertNum_);
                // 直接设置为可能的最大空间，只要代码逻辑保证不要超过实际空间即可
                gmOut_.SetGlobalBuffer((__gm__ int8_t*)(out), gbExpertNum_ * gbTokenNum_ * gbCopyByte_);
                pipe_.InitBuffer(inQueIn_, BUFFER_NUM, bkCopyByteAlign_);
                pipe_.InitBuffer(inQueTokenTable_, BUFFER_NUM, gbTokenNumAlign_ * sizeof(int32_t));
                pipe_.InitBuffer(inQueTokenList_, BUFFER_NUM, gbExpertNumAlign_ * sizeof(int64_t));
                pipe_.InitBuffer(outQueOut_,  BUFFER_NUM, bkCopyByteAlign_);
            }
            __aicore__ inline void Process()
            {
                LocalTensor<int64_t> local_token_list = inQueTokenList_.AllocTensor<int64_t>();
                DataCopyParams copy_pas{1, (uint16_t)(gbExpertNum_ * sizeof(int64_t)), 0, 0};
                DataCopyPadParams pad_pas;
                DataCopyPad(local_token_list, gmTokenList_, copy_pas, pad_pas);
                inQueTokenList_.EnQue(local_token_list);
                lmTokenList_ = inQueTokenList_.DeQue<int64_t>();
                for (int64_t expert_idx = 0; expert_idx < gbExpertNum_; expert_idx++) {
                    LocalTensor<int32_t> local_token_table = inQueTokenTable_.AllocTensor<int32_t>();
                    copy_pas.blockLen = (uint16_t)(gbTokenNum_ * sizeof(int32_t));
                    DataCopyPad(local_token_table, gmTokenTable_[expert_idx * gbTokenNum_], copy_pas, pad_pas);
                    inQueTokenTable_.EnQue(local_token_table);
                    lmTokenTable_ = inQueTokenTable_.DeQue<int32_t>();
                    int64_t token_idx_start = 0;
                    if (expert_idx != 0) {
                        token_idx_start = lmTokenList_.GetValue(expert_idx-1);
                    }
                    int64_t token_idx_end = lmTokenList_.GetValue(expert_idx);
                    for (int64_t tis = token_idx_start; tis < token_idx_end; tis += blockNum_) {
                        int64_t token_idx = tis + blockIdx_;
                        int64_t token_offset = token_idx - token_idx_start;
                        if (token_idx < token_idx_end) {
                            // AscendC::printf("token_idx, token_offset: %d, %d\n", token_idx, token_offset);
                            CopyIn(token_offset);
                            Compute();
                            CopyOut(token_idx);
                        }
                    }
                    inQueTokenTable_.FreeTensor(local_token_table);
                }
                inQueTokenList_.FreeTensor(local_token_list);
            }
        private:
            __aicore__ inline void CopyIn(int64_t token_offset)
            {
                LocalTensor<int8_t> local_in = inQueIn_.AllocTensor<int8_t>();
                DataCopyExtParams copy_pas{1, (uint32_t)(bkCopyByte_), 0, 0, 0};
                DataCopyPadExtParams<int8_t> pad_pas;
                int64_t offset = lmTokenTable_.GetValue(token_offset) * gbCopyByte_;
                DataCopyPad(local_in, gmIn_[offset], copy_pas, pad_pas);
                inQueIn_.EnQue(local_in);
            }
            __aicore__ inline void Compute()
            {
                LocalTensor<int8_t> local_in = inQueIn_.DeQue<int8_t>();
                LocalTensor<int8_t> local_out = outQueOut_.AllocTensor<int8_t>();
                DataCopy(local_out, local_in, bkCopyByteAlign_);
                outQueOut_.EnQue(local_out);
                inQueIn_.FreeTensor(local_in);
            }
            __aicore__ inline void CopyOut(int64_t token_idx)
            {
                LocalTensor<int8_t> local_out = outQueOut_.DeQue<int8_t>();
                DataCopyExtParams copy_pas{1, (uint32_t)(bkCopyByte_), 0, 0, 0};
                DataCopyPad(gmOut_[token_idx * bkCopyByte_], local_out, copy_pas);
                outQueOut_.FreeTensor(local_out);
            }
        private:
            TPipe pipe_;
            TQue<QuePosition::VECIN, BUFFER_NUM> inQueIn_, inQueTokenTable_, inQueTokenList_;
            TQue<QuePosition::VECOUT, BUFFER_NUM> outQueOut_;
            GlobalTensor<int8_t> gmIn_, gmOut_;
            GlobalTensor<int32_t> gmTokenTable_;
            GlobalTensor<int64_t> gmTokenList_;
            LocalTensor<int32_t> lmTokenTable_;
            LocalTensor<int64_t> lmTokenList_;
            int64_t blockNum_, blockIdx_;
            int64_t gbExpertNum_, gbTokenNum_, gbCopyByte_;
            int64_t gbExpertNumAlign_, gbTokenNumAlign_;
            int64_t bkExpertNum_, bkTokenNum_, bkCopyByte_;
            int64_t bkCopyByteAlign_;
        };
        extern "C" __global__ __aicore__ void compute_init_routing(GM_ADDR in, GM_ADDR token_table,
                                                                   GM_ADDR token_list, GM_ADDR out,
                                                                   const int64_t expert_num, const int64_t token_num,
                                                                   const int64_t copy_byte)
        {
            init_routing_kernel op;
            op.Init(in, token_table, token_list, out, expert_num, token_num, copy_byte);
            op.Process();
        }
        void init_routing_kernel_lanuch(int64_t block_dim, void* stream,
                                        uint8_t* in, uint8_t* token_table, uint8_t* token_list, uint8_t* out,
                                        const int64_t expert_num, const int64_t token_num, const int64_t copy_byte)
        {
            compute_init_routing<<<block_dim, nullptr, stream>>>(in, token_table, token_list, out,
                                                                 expert_num, token_num, copy_byte);
        }
        inline int64_t align_up(const int64_t number, const int64_t alignSize)
        {
            if (number % alignSize == 0) {
                return number;
            }
            return ((number / alignSize + 1) * alignSize);
        }
        int judge_init_routing_lanuch(const int64_t expert_num, const int64_t token_num, const int64_t copy_byte,
                                      const int64_t ubSize, const int64_t vCores)
        {
            (void)(vCores);
            constexpr int64_t BUFFER_NUM = 1;
            int64_t gbExpertNum_;
            int64_t gbTokenNum_;
            int64_t gbCopyByte_;
            int64_t gbExpertNumAlign_;
            int64_t gbTokenNumAlign_;
            int64_t bkCopyByte_;
            int64_t bkCopyByteAlign_;
            gbExpertNum_ = expert_num;
            gbTokenNum_  = token_num;
            gbCopyByte_  = copy_byte;
            gbTokenNumAlign_  = align_up(gbTokenNum_, 64 / sizeof(int32_t));
            gbExpertNumAlign_ = align_up(gbExpertNum_, 64 / sizeof(int32_t));
            bkCopyByte_  = gbCopyByte_;
            bkCopyByteAlign_ = align_up(bkCopyByte_, 64);
            float use_byte = BUFFER_NUM * bkCopyByteAlign_ * 2;
            use_byte += gbTokenNumAlign_ * sizeof(int32_t);
            use_byte += gbExpertNumAlign_ * sizeof(int64_t);
            if (use_byte > ubSize) {
                return 1;
            }
            return 0;
        }
        int init_routing_lanuch(int64_t block_dim, void* stream, uint8_t* in, uint8_t* token_table,
                                uint8_t* token_list, uint8_t* out, const int64_t expert_num, const int64_t token_num,
                                const int64_t copy_byte)
        {
            int64_t ubSize = 184 * 1024;
            int64_t vCores = 40;
            int ret = judge_init_routing_lanuch(expert_num, token_num, copy_byte, ubSize, vCores);
            if (ret == 0) {
                init_routing_kernel_lanuch(block_dim, stream, in, token_table, token_list, out, expert_num,
                                           token_num, copy_byte);
                return 0;
            }
            return 1;
        }
        int64_t init_routing_npu(int64_t block_dim, torch::Tensor &in, torch::Tensor &token_table,
                                 torch::Tensor &token_list, torch::Tensor &out, const int64_t expert_num,
                                 const int64_t token_num, const int64_t copy_byte)
        {
            TORCH_CHECK(torch_npu::utils::is_npu(in), "input tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(token_table), "token table tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(token_list), "token list tensor must be on NPU device");
            TORCH_CHECK(torch_npu::utils::is_npu(out), "output tensor must be on NPU device");
            TORCH_CHECK((copy_byte % 64) == 0, "copy_byte must be aligned to 64 bytes, but got ", copy_byte);
            TORCH_CHECK(expert_num <= 8191, "expert_num exceeds maximum allowed value of 8191, but got ", expert_num);
            TORCH_CHECK(token_num <= 16383, "token_num exceeds maximum allowed value of 16383, but got ", token_num);

            auto stream = c10_npu::getCurrentNPUStream().stream(false);
            int launchStatus = 0;
            auto acl_call = [=, &launchStatus]() -> int {
                launchStatus = init_routing_lanuch(block_dim, stream, (uint8_t *)in.data_ptr(),
                                                   (uint8_t *)token_table.data_ptr(), (uint8_t *)token_list.data_ptr(),
                                                   (uint8_t *)out.data_ptr(), expert_num, token_num, copy_byte);
                return 0;
            };
            at_npu::native::OpCommand::RunOpApi("InitRouting", acl_call);
            return launchStatus;
        }
        TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
        {
            m.impl("init_routing", init_routing_npu);
        }
    }
}