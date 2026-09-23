/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int64_t BUFFER_NUM = 1;
constexpr int64_t MIN_ALIGN_BYTES = 512;

class GRC_INITTABLE_SBUF_V2 {
public:
    __aicore__ inline GRC_INITTABLE_SBUF_V2() {}

    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR scores, GM_ADDR init_token_table, GM_ADDR final_score_table,
                                const int64_t start_expert_id, const int64_t end_expert_id,
                                const int64_t local_expert_num, const int64_t token_num,
                                const int64_t topk, const int64_t UB_MAX_TOKEN)
    {
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        gbStartExpertId_    = start_expert_id;
        gbEndExpertId_      = end_expert_id;
        gbLocalExpertNum_   = local_expert_num;
        gbTokenNum_         = token_num;
        gbTopk_             = topk;

        bkTokenNum_ = gbTokenNum_ / blockNum_;
        bkTokenNumStartIdx_ = bkTokenNum_ * blockIdx_;
        if (blockIdx_ < gbTokenNum_ % blockNum_) {
            bkTokenNum_ += 1;
            bkTokenNumStartIdx_ += blockIdx_;
        } else {
            bkTokenNumStartIdx_ += gbTokenNum_ % blockNum_;
        }
        bkTokenNumAlign_ = AlignUp(bkTokenNum_, 32 / sizeof(int32_t));

        ubMaxToken_ = UB_MAX_TOKEN;
        ubLoopCount_ = (bkTokenNum_ + ubMaxToken_ - 1) / ubMaxToken_;

        bkValidTableStart_ = gbStartExpertId_ * bkTokenNumAlign_;
        bkValidTableCount_ = (gbEndExpertId_ - gbStartExpertId_) * bkTokenNumAlign_;

        gmIndices_.SetGlobalBuffer((__gm__ int32_t*)(indices), gbTokenNum_ * gbTopk_);
        gmScores_.SetGlobalBuffer((__gm__ bfloat16_t*)(scores), gbTokenNum_ * gbTopk_);
        gmInitTokenTable_.SetGlobalBuffer((__gm__ int32_t*)(init_token_table), gbLocalExpertNum_ * gbTokenNum_);
        gmFinalScoreTable_.SetGlobalBuffer((__gm__ bfloat16_t*)(final_score_table), gbTokenNum_ * gbLocalExpertNum_);

        int64_t indices_bytes           = AlignUp(ubMaxToken_ * gbTopk_ * sizeof(int32_t), MIN_ALIGN_BYTES);
        int64_t scores_bytes            = AlignUp(ubMaxToken_ * gbTopk_ * sizeof(bfloat16_t), MIN_ALIGN_BYTES);
        int64_t init_token_table_bytes  = AlignUp(gbLocalExpertNum_ * ubMaxToken_ * sizeof(int32_t), MIN_ALIGN_BYTES);
        int64_t final_score_table_bytes = AlignUp(ubMaxToken_ * gbLocalExpertNum_ * sizeof(bfloat16_t),
                                                  MIN_ALIGN_BYTES);

        pipe_.InitBuffer(inQueIndices_, BUFFER_NUM, indices_bytes);
        pipe_.InitBuffer(inQueScores_, BUFFER_NUM, scores_bytes);
        pipe_.InitBuffer(outQueInitTokenTable_, BUFFER_NUM, init_token_table_bytes);
        pipe_.InitBuffer(outQueFinalScoreTable_, BUFFER_NUM, final_score_table_bytes);
    }

    __aicore__ inline void Process()
    {
        int64_t ubTokenNum = ubMaxToken_;
        for (int64_t iLoop = 0; iLoop < ubLoopCount_; iLoop++) {
            int64_t ubTokenNumAlign = ubTokenNum;
            if (bkTokenNum_ - iLoop * ubMaxToken_ < ubMaxToken_) {
                ubTokenNum = bkTokenNum_ - iLoop * ubMaxToken_;
                ubTokenNumAlign = AlignUp(ubTokenNum, 32 / sizeof(int32_t));
            }

            LocalTensor<int32_t> local_indices = inQueIndices_.AllocTensor<int32_t>();
            {
                DataCopyExtParams params{1, (uint32_t)(ubMaxToken_ * gbTopk_ * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad_params;
                DataCopyPad(local_indices, gmIndices_[(bkTokenNumStartIdx_ + iLoop * ubMaxToken_) * gbTopk_],
                            params, pad_params);
            }

            LocalTensor<bfloat16_t> local_scores = inQueScores_.AllocTensor<bfloat16_t>();
            {
                DataCopyExtParams params{1, (uint32_t)(ubTokenNum * gbTopk_ * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPadExtParams<bfloat16_t> pad_params;
                DataCopyPad(local_scores, gmScores_[(bkTokenNumStartIdx_ + iLoop * ubMaxToken_) * gbTopk_],
                            params, pad_params);
            }

            LocalTensor<int32_t> local_init_token_table = outQueInitTokenTable_.AllocTensor<int32_t>();
            Duplicate<int32_t>(local_init_token_table, (int32_t)(-1), gbLocalExpertNum_ * ubTokenNumAlign);

            LocalTensor<bfloat16_t> local_final_score_table = outQueFinalScoreTable_.AllocTensor<bfloat16_t>();
            Duplicate<bfloat16_t>(local_final_score_table, (bfloat16_t)(0.0), ubTokenNumAlign * gbLocalExpertNum_);

            // 同步indices和scores数据拷贝
            int32_t eventIdMTE2ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

            // 同步local_init_token_table和local_final_score_table初始化
            int32_t eventIdVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            for (int64_t iToken = 0; iToken < ubTokenNum; iToken++) {
                for (int64_t iTopk = 0; iTopk < gbTopk_; iTopk++) {
                    int32_t expert_id = local_indices.GetValue(iToken * gbTopk_ + iTopk);
                    if (expert_id >= gbStartExpertId_ && expert_id < gbEndExpertId_) {
                        int32_t local_expert_id = expert_id % gbLocalExpertNum_;
                        int32_t temp_init_index = local_expert_id * ubTokenNumAlign + iToken;
                        int32_t current_token_idx = local_init_token_table.GetValue(temp_init_index);
                        // 只在第一次出现时设置（通过init_token_table是否为-1来判断）
                        if (current_token_idx == -1) {
                            local_init_token_table.SetValue(temp_init_index,
                                                            bkTokenNumStartIdx_ + iLoop * ubMaxToken_ + iToken);
                            bfloat16_t expert_score = local_scores.GetValue(iToken * gbTopk_ + iTopk);
                            int32_t temp_final_index = iToken * gbLocalExpertNum_ + local_expert_id;
                            local_final_score_table.SetValue(temp_final_index, expert_score);
                        }
                    }
                }
            }

            int32_t eventIdSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);

            {
                DataCopyExtParams params {
                    (uint16_t)(gbLocalExpertNum_),
                    (uint32_t)(ubTokenNum * sizeof(int32_t)),
                    0,
                    (uint32_t)((gbTokenNum_ - ubTokenNum) * sizeof(int32_t)),
                    0};
                DataCopyPad(gmInitTokenTable_[bkTokenNumStartIdx_ + iLoop * ubMaxToken_],
                            local_init_token_table, params);
            }

            {
                DataCopyExtParams params{1, (uint32_t)(ubTokenNum * gbLocalExpertNum_ * sizeof(bfloat16_t)), 0, 0, 0};
                DataCopyPad(gmFinalScoreTable_[(bkTokenNumStartIdx_ + iLoop * ubMaxToken_) * gbLocalExpertNum_],
                            local_final_score_table, params);
            }

            inQueIndices_.FreeTensor(local_indices);
            inQueScores_.FreeTensor(local_scores);
            outQueInitTokenTable_.FreeTensor(local_init_token_table);
            outQueFinalScoreTable_.FreeTensor(local_final_score_table);
        }
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueIndices_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueScores_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueInitTokenTable_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueFinalScoreTable_;

    GlobalTensor<int32_t> gmIndices_;
    GlobalTensor<bfloat16_t> gmScores_;
    GlobalTensor<int32_t> gmInitTokenTable_;
    GlobalTensor<bfloat16_t> gmFinalScoreTable_;
    
    int64_t blockNum_, blockIdx_;

    int64_t gbStartExpertId_, gbEndExpertId_, gbLocalExpertNum_;
    int64_t bkValidTableStart_, bkValidTableCount_;
    int64_t gbTokenNum_, gbTopk_;
    
    int64_t bkTokenNum_, bkTokenNumAlign_;
    int64_t bkTokenNumStartIdx_;

    int64_t ubMaxToken_, ubLoopCount_;
};

class GRC_INITOTHER_SBUF_V2 {
public:
    __aicore__ inline GRC_INITOTHER_SBUF_V2() {}

    __aicore__ inline void Init(GM_ADDR init_token_list, GM_ADDR init_token_table, GM_ADDR token_idx_intra_expert,
                                const int64_t local_expert_num, const int64_t token_num, const int64_t UB_MAX_TOKEN)
    {
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        gbLocalExpertNum_       = local_expert_num;
        gbLocalExpertNumAlign_  = AlignUp(gbLocalExpertNum_, 32 / sizeof(int32_t));
        gbTokenNum_             = token_num;
        gbTokenNumAlign_        = AlignUp(gbTokenNum_, 256 / sizeof(int32_t));

        bkLocalExpertNum_       = gbLocalExpertNum_ / blockNum_;
        bkLocalExpertNumAlign_  = AlignUp(gbLocalExpertNum_, 32 / sizeof(int64_t));
        bkGmStartIdx_           = blockIdx_ * bkLocalExpertNum_ * gbTokenNum_;
        if (blockIdx_ < gbLocalExpertNum_ % blockNum_) {
            bkLocalExpertNum_ += 1;
            bkGmStartIdx_ += blockIdx_ * gbTokenNum_;
        } else {
            bkGmStartIdx_ += (gbLocalExpertNum_ % blockNum_) * gbTokenNum_;
        }

        ubMaxToken_ = UB_MAX_TOKEN;
        ubLoopCount_ = (gbTokenNum_ + ubMaxToken_ - 1) / ubMaxToken_;

        gmInitTokenList_.SetGlobalBuffer((__gm__ int64_t*)(init_token_list), gbLocalExpertNum_);
        gmInitTokenTable_.SetGlobalBuffer((__gm__ int32_t*)(init_token_table), gbLocalExpertNum_ * gbTokenNum_);
        gmTokenIdxIntraExpert_.SetGlobalBuffer((__gm__ int32_t*)(token_idx_intra_expert),
                                               gbLocalExpertNum_ * gbTokenNum_);

        pipe_.InitBuffer(outQueInitTokenList_, BUFFER_NUM, bkLocalExpertNumAlign_ * sizeof(int64_t));
        pipe_.InitBuffer(outQueInitTokenTable_, BUFFER_NUM, ubMaxToken_ * sizeof(int32_t));
        pipe_.InitBuffer(tmpMask_, AlignUp(ubMaxToken_ / 8, 32)); // char = 8bit
        pipe_.InitBuffer(tmpBuffer_, ubMaxToken_ * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (bkLocalExpertNum_ > 0) {
            LocalTensor<int64_t> local_out_list = outQueInitTokenList_.AllocTensor<int64_t>();
            for (int64_t iExpert = 0; iExpert < bkLocalExpertNum_; iExpert++) {
                int64_t ubTokenNum = ubMaxToken_;
                int32_t routingCount = 0;
                for (int64_t iLoop = 0; iLoop < ubLoopCount_; iLoop++) {
                    int64_t ubTokenNumAlign = ubTokenNum;
                    if (gbTokenNum_ - iLoop * ubMaxToken_ < ubMaxToken_) {
                        ubTokenNum = gbTokenNum_ - iLoop * ubMaxToken_;
                        ubTokenNumAlign = AlignUp(ubTokenNum, 256 / sizeof(int32_t));
                    }
                    
                    LocalTensor<int32_t> local_in_table = tmpBuffer_.Get<int32_t>();
                    {
                        DataCopyExtParams params{1, (uint32_t)(ubTokenNum * sizeof(int32_t)), 0, 0, 0};
                        DataCopyPadExtParams<int32_t> pad_params;
                        DataCopyPad(local_in_table,
                                    gmInitTokenTable_[bkGmStartIdx_ + iExpert * gbTokenNum_ + iLoop * ubMaxToken_],
                                    params, pad_params);
                    }

                    LocalTensor<uint8_t> local_mask = tmpMask_.Get<uint8_t>();
                    LocalTensor<uint32_t> local_mask_uint32 = local_mask.ReinterpretCast<uint32_t>();

                    LocalTensor<int32_t> local_out_table = outQueInitTokenTable_.AllocTensor<int32_t>();
                    Duplicate<int32_t>(local_out_table, (int32_t)(-1), ubTokenNumAlign);

                    LocalTensor<float> local_in_table_fp32 = local_out_table.ReinterpretCast<float>();
                    Duplicate<float>(local_in_table_fp32, -1.0f, ubTokenNumAlign);

                    // 同步local_in_table数据拷贝
                    int32_t eventIdMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

                    // 同步Duplicate(local_in_table_fp32)
                    PipeBarrier<PIPE_V>();

                    Cast(local_in_table_fp32, local_in_table, RoundMode::CAST_NONE, ubTokenNum);

                    // 同步Cast(local_in_table_fp32)
                    PipeBarrier<PIPE_V>();

                    CompareScalar(local_mask, local_in_table_fp32, 0.0f, CMPMODE::GE, ubTokenNumAlign);

                    // 同步CompareScalar(local_mask)
                    PipeBarrier<PIPE_V>();

                    uint64_t rsvdCnt = 0;
                    GatherMask(local_out_table, local_in_table, local_mask_uint32, true,
                               ubTokenNumAlign, {1, 1, 0, 0}, rsvdCnt);

                    LocalTensor<int32_t> local_out_intra = tmpBuffer_.Get<int32_t>();
                    Duplicate<int32_t>(local_out_intra, (int32_t)(-1), ubTokenNumAlign);

                    // 同步Duplicate(local_out_intra)，临时借用给initTokenTable
                    int32_t eventIdVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

                    // 因为initTokenTable只拷贝有效值，需要提前清零
                    DataCopyExtParams params{1, (uint32_t)(ubTokenNum * sizeof(int32_t)), 0, 0, 0};
                    DataCopyPad(gmInitTokenTable_[bkGmStartIdx_ + iExpert * gbTokenNum_ + iLoop * ubMaxToken_],
                                local_out_intra, params);

                    int32_t eventIdMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                    SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
                    WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);

                    for (int32_t i = 0; i < (int32_t)rsvdCnt; i++) {
                        int32_t value = local_out_table.GetValue(i);
                        local_out_intra.SetValue(value - iLoop * ubMaxToken_, routingCount + i);
                    }

                    int32_t eventIdSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                    SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
                    WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);

                    params.blockLen = (uint32_t)(ubTokenNum * sizeof(int32_t));
                    DataCopyPad(gmTokenIdxIntraExpert_[bkGmStartIdx_ + iExpert * gbTokenNum_ + iLoop * ubMaxToken_],
                                local_out_intra, params);
                    
                    params.blockLen = (uint32_t)(rsvdCnt * sizeof(int32_t));
                    DataCopyPad(gmInitTokenTable_[bkGmStartIdx_ + iExpert * gbTokenNum_ + routingCount],
                                local_out_table, params);

                    PipeBarrier<PIPE_ALL>();

                    outQueInitTokenTable_.FreeTensor(local_out_table);
                    routingCount += rsvdCnt;
                }
                local_out_list.SetValue(iExpert, routingCount);
            }
            DataCopyExtParams params{1, (uint32_t)(bkLocalExpertNum_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPad(gmInitTokenList_[blockIdx_ * bkLocalExpertNum_], local_out_list, params);
        }
    }

private:
    TPipe pipe_;
    // TQue<QuePosition::VECIN, BUFFER_NUM> inQueInitTokenTable_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueInitTokenList_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueInitTokenTable_;
    // TQue<QuePosition::VECOUT, BUFFER_NUM> outQueTokenIdxIntraExpert_;
    TBuf<TPosition::VECCALC> tmpMask_, tmpBuffer_;

    GlobalTensor<int64_t> gmInitTokenList_;
    GlobalTensor<int32_t> gmInitTokenTable_;
    GlobalTensor<int32_t> gmTokenIdxIntraExpert_;
    
    int64_t blockNum_, blockIdx_;
    int64_t gbLocalExpertNum_, gbLocalExpertNumAlign_;
    int64_t gbTokenNum_, gbTokenNumAlign_;
    int64_t bkLocalExpertNum_, bkLocalExpertNumAlign_;
    int64_t bkGmStartIdx_;
    int64_t ubMaxToken_, ubLoopCount_;
};

class GRC_FINAL_SBUF_V2 {
public:
    __aicore__ inline GRC_FINAL_SBUF_V2() {}

    __aicore__ inline void Init(
        GM_ADDR indices, GM_ADDR init_token_list, GM_ADDR final_token_table, GM_ADDR token_idx_intra_expert,
        const int64_t start_expert_id, const int64_t end_expert_id, const int64_t local_expert_num,
        const int64_t token_num, const int64_t topk, const int64_t UB_MAX_TOKEN)
    {
        blockNum_ = GetBlockNum();
        blockIdx_ = GetBlockIdx();

        gbStartExpertId_        = start_expert_id;
        gbEndExpertId_          = end_expert_id;
        gbLocalExpertNum_       = local_expert_num;
        gbLocalExpertNumAlign_  = AlignUp(gbLocalExpertNum_, 32 / sizeof(int32_t));
        gbTokenNum_             = token_num;
        gbTopk_                 = topk;

        bkTokenNum_ = gbTokenNum_ / blockNum_;
        bkTokenNumStartIdx_ = bkTokenNum_ * blockIdx_;
        if (blockIdx_ < gbTokenNum_ % blockNum_) {
            bkTokenNum_ += 1;
            bkTokenNumStartIdx_ += blockIdx_;
        } else {
            bkTokenNumStartIdx_ += gbTokenNum_ % blockNum_;
        }
        bkTokenNumAlign_ = AlignUp(bkTokenNum_, 32 / sizeof(int32_t));

        ubMaxToken_ = UB_MAX_TOKEN;
        ubLoopCount_ = (bkTokenNum_ + ubMaxToken_ - 1) / ubMaxToken_;
      
        gmIndices_.SetGlobalBuffer((__gm__ int32_t*)(indices), gbTokenNum_ * gbTopk_);
        gmInitTokenList_.SetGlobalBuffer((__gm__ int64_t*)(init_token_list), gbLocalExpertNum_);
        gmTokenIdxIntraExpert_.SetGlobalBuffer((__gm__ int32_t*)(token_idx_intra_expert),
                                               gbLocalExpertNum_ * gbTokenNum_);
        gmFinalTokenTable_.SetGlobalBuffer((__gm__ int32_t*)(final_token_table), gbTokenNum_ * gbLocalExpertNum_);

        int64_t indices_num                 = AlignUp(ubMaxToken_ * gbTopk_, MIN_ALIGN_BYTES / sizeof(int32_t));
        int64_t init_token_list_num         = AlignUp(gbLocalExpertNum_, MIN_ALIGN_BYTES / sizeof(int64_t));
        int64_t final_token_table_num       = AlignUp(ubMaxToken_ * gbLocalExpertNum_,
                                                      MIN_ALIGN_BYTES / sizeof(int32_t));
        int64_t token_idx_intra_expert_num  = AlignUp(gbLocalExpertNum_ * ubMaxToken_,
                                                      MIN_ALIGN_BYTES / sizeof(int32_t));

        pipe_.InitBuffer(inQueIndices_, BUFFER_NUM, indices_num * sizeof(int32_t));
        pipe_.InitBuffer(inQueInitTokenList_, BUFFER_NUM, init_token_list_num * sizeof(int64_t));
        pipe_.InitBuffer(inQueTokenIdxIntraExpert_, BUFFER_NUM, token_idx_intra_expert_num * sizeof(int32_t));
        pipe_.InitBuffer(outQueFinalTokenTable_, BUFFER_NUM, final_token_table_num * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        // 计算前缀和
        LocalTensor<int64_t> local_init_token_list = inQueInitTokenList_.AllocTensor<int64_t>();
        {
            DataCopyExtParams params{1, (uint32_t)(gbLocalExpertNum_ * sizeof(int64_t)), 0, 0, 0};
            DataCopyPadExtParams<int64_t> pad_params;
            DataCopyPad(local_init_token_list, gmInitTokenList_[0], params, pad_params);
        }

        // 同步local_init_token_list数据拷贝
        int32_t eventIdMTE2ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

        for (int64_t iExpert = 1; iExpert < gbLocalExpertNum_; iExpert++) {
            int32_t sum = local_init_token_list.GetValue(iExpert - 1);
            int32_t val = local_init_token_list.GetValue(iExpert);
            local_init_token_list.SetValue(iExpert, sum + val);
        }

        int64_t ubTokenNum = ubMaxToken_;
        for (int64_t iLoop = 0; iLoop < ubLoopCount_; iLoop++) {
            int64_t ubTokenNumAlign = ubTokenNum;
            if (bkTokenNum_ - iLoop * ubMaxToken_ < ubMaxToken_) {
                ubTokenNum = bkTokenNum_ - iLoop * ubMaxToken_;
                ubTokenNumAlign = AlignUp(ubTokenNum, 32 / sizeof(int32_t));
            }

            LocalTensor<int32_t> local_indices = inQueIndices_.AllocTensor<int32_t>();
            {
                DataCopyExtParams params{1, (uint32_t)(ubTokenNum * gbTopk_ * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad_params;
                DataCopyPad(local_indices, gmIndices_[(bkTokenNumStartIdx_ + iLoop * ubMaxToken_) * gbTopk_],
                            params, pad_params);
            }

            LocalTensor<int32_t> local_token_idx_intra_expert = inQueTokenIdxIntraExpert_.AllocTensor<int32_t>();
            {
                DataCopyExtParams params {
                    (uint16_t)(gbLocalExpertNum_),
                    (uint32_t)(ubTokenNum * sizeof(int32_t)),
                    (uint32_t)((gbTokenNum_ - ubTokenNum) * sizeof(int32_t)),
                    0,
                    0};
                DataCopyPadExtParams<int32_t> pad_params;
                DataCopyPad(local_token_idx_intra_expert,
                            gmTokenIdxIntraExpert_[bkTokenNumStartIdx_ + iLoop * ubMaxToken_], params, pad_params);
            }
         
            LocalTensor<int32_t> local_final_token_table = outQueFinalTokenTable_.AllocTensor<int32_t>();
            Duplicate<int32_t>(local_final_token_table, (int32_t)(-1), ubTokenNumAlign * gbLocalExpertNum_);

            // 同步indices和intra的数据拷贝
            int32_t eventIdMTE2ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);

            // 同步Duplicate(local_final_token_table)
            int32_t eventIdVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            for (int64_t iToken = 0; iToken < ubTokenNum; iToken++) {
                for (int64_t iTopk = 0; iTopk < gbTopk_; iTopk++) {
                    int32_t expert_id = local_indices.GetValue(iToken * gbTopk_ + iTopk);
                    if (expert_id >= gbStartExpertId_ && expert_id < gbEndExpertId_) {
                        int32_t local_expert_id = expert_id % gbLocalExpertNum_;
                        int32_t routing_token_offset = 0;
                        if (local_expert_id != 0) {
                            routing_token_offset = local_init_token_list.GetValue(local_expert_id - 1);
                        }
                        int32_t temp_final_value = local_token_idx_intra_expert.GetValue(local_expert_id *
                                                                                         ubTokenNumAlign + iToken);
                        int32_t temp_final_index = iToken * gbLocalExpertNum_ + local_expert_id;
                        local_final_token_table.SetValue(temp_final_index, temp_final_value + routing_token_offset);
                    }
                }
            }

            // 同步local_final_token_table计算
            int32_t eventIdSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);

            DataCopyExtParams copyout_pas{1, (uint32_t)(ubTokenNum * gbLocalExpertNum_ * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(gmFinalTokenTable_[(bkTokenNumStartIdx_ + iLoop * ubMaxToken_) * gbLocalExpertNum_],
                        local_final_token_table, copyout_pas);

            outQueFinalTokenTable_.EnQue(local_final_token_table);
            outQueFinalTokenTable_.DeQue<int32_t>();

            inQueIndices_.FreeTensor(local_indices);
            inQueTokenIdxIntraExpert_.FreeTensor(local_token_idx_intra_expert);
            outQueFinalTokenTable_.FreeTensor(local_final_token_table);
        }

        inQueInitTokenList_.FreeTensor(local_init_token_list);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueIndices_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueInitTokenList_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueTokenIdxIntraExpert_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueFinalTokenTable_;

    GlobalTensor<int32_t> gmIndices_;
    GlobalTensor<int64_t> gmInitTokenList_;
    GlobalTensor<int32_t> gmTokenIdxIntraExpert_;
    GlobalTensor<int32_t> gmFinalTokenTable_;

    int64_t blockNum_, blockIdx_;

    int64_t gbLocalExpertNum_, gbLocalExpertNumAlign_;
    int64_t gbStartExpertId_, gbEndExpertId_;
    int64_t gbTokenNum_, gbTopk_;

    int64_t bkTokenNum_, bkTokenNumAlign_;
    int64_t bkTokenNumStartIdx_;
    int64_t ubMaxToken_, ubLoopCount_;
};

extern "C" __global__ __aicore__ void getRoutingConfigurationsSBufV2_InitTable_kernel(
    GM_ADDR indices, GM_ADDR scores, GM_ADDR init_token_table, GM_ADDR init_token_list,
    GM_ADDR final_token_table, GM_ADDR final_score_table, GM_ADDR token_idx_intra_expert,
    const int64_t start_expert_id, const int64_t end_expert_id, const int64_t local_expert_num,
    const int64_t token_num, const int64_t topk, const int64_t UB_MAX_TOKEN)
{
    GRC_INITTABLE_SBUF_V2 inittable_op;
    inittable_op.Init(indices, scores, init_token_table, final_score_table, start_expert_id,
                      end_expert_id, local_expert_num, token_num, topk, UB_MAX_TOKEN);
    inittable_op.Process();
}

extern "C" __global__ __aicore__ void getRoutingConfigurationsSBufV2_InitOther_kernel(
    GM_ADDR indices, GM_ADDR scores, GM_ADDR init_token_table, GM_ADDR init_token_list,
    GM_ADDR final_token_table, GM_ADDR final_score_table, GM_ADDR token_idx_intra_expert,
    const int64_t start_expert_id, const int64_t end_expert_id, const int64_t local_expert_num,
    const int64_t token_num, const int64_t topk, const int64_t UB_MAX_TOKEN)
{
    GRC_INITOTHER_SBUF_V2 initohter_op;
    initohter_op.Init(init_token_list, init_token_table, token_idx_intra_expert,
                      local_expert_num, token_num, UB_MAX_TOKEN);
    initohter_op.Process();
}

extern "C" __global__ __aicore__ void getRoutingConfigurationsSBufV2_Final_kernel(
    GM_ADDR indices, GM_ADDR scores, GM_ADDR init_token_table, GM_ADDR init_token_list,
    GM_ADDR final_token_table, GM_ADDR final_score_table, GM_ADDR token_idx_intra_expert,
    const int64_t start_expert_id, const int64_t end_expert_id, const int64_t local_expert_num,
    const int64_t token_num, const int64_t topk, const int64_t UB_MAX_TOKEN)
{
    GRC_FINAL_SBUF_V2 final_routing_op;
    final_routing_op.Init(indices, init_token_list, final_token_table, token_idx_intra_expert,
                          start_expert_id, end_expert_id, local_expert_num, token_num, topk, UB_MAX_TOKEN);
    final_routing_op.Process();
}

void getRoutingConfigurationsSBufV2_launch(
    int64_t block_dim, void* stream, uint8_t* indices, uint8_t* scores, uint8_t* init_token_table,
    uint8_t* init_token_list, uint8_t* final_token_table, uint8_t* final_score_table, uint8_t* token_idx_intra_expert,
    const int64_t start_expert_id, const int64_t end_expert_id, const int64_t local_expert_num,
    const int64_t token_num, const int64_t topk, const int64_t UB_MAX_TOKEN)
{
    int64_t inittable_block_dim = token_num > block_dim ? block_dim : token_num;
    getRoutingConfigurationsSBufV2_InitTable_kernel<<<inittable_block_dim, nullptr, stream>>>(
        indices, scores, init_token_table, init_token_list, final_token_table, final_score_table,
        token_idx_intra_expert, start_expert_id, end_expert_id, local_expert_num, token_num, topk, UB_MAX_TOKEN);

    int64_t initother_block_dim = local_expert_num > block_dim ? block_dim : local_expert_num;
    getRoutingConfigurationsSBufV2_InitOther_kernel<<<initother_block_dim, nullptr, stream>>>(
        indices, scores, init_token_table, init_token_list, final_token_table, final_score_table,
        token_idx_intra_expert, start_expert_id, end_expert_id, local_expert_num, token_num, topk, UB_MAX_TOKEN);

    int64_t final_block_dim = token_num > block_dim ? block_dim : token_num;
    getRoutingConfigurationsSBufV2_Final_kernel<<<final_block_dim, nullptr, stream>>>(
        indices, scores, init_token_table, init_token_list, final_token_table, final_score_table,
        token_idx_intra_expert, start_expert_id, end_expert_id, local_expert_num, token_num, topk, UB_MAX_TOKEN);
}

int64_t getRoutingConfigurationsSBufV2(int64_t block_dim, torch::Tensor &indices, torch::Tensor &scores,
                                       torch::Tensor &init_token_table, torch::Tensor &init_token_list,
                                       torch::Tensor &final_token_table, torch::Tensor &final_score_table,
                                       torch::Tensor &token_idx_intra_expert, const int64_t start_expert_id,
                                       const int64_t end_expert_id, const int64_t local_expert_num,
                                       const int64_t token_num, const int64_t topk)
{
    TORCH_CHECK(torch_npu::utils::is_npu(indices), "indices tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(scores), "scores tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(init_token_table), "init_token_table tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(init_token_list), "init_token_list tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(final_token_table), "final_token_table tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(final_score_table), "final_score_table tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(token_idx_intra_expert),
                "token_idx_intra_expert tensor must be on NPU device");

    auto stream = c10_npu::getCurrentNPUStream().stream(false);

    constexpr int64_t MIN_ALIGN_BYTES = 512;
    int64_t UB_MAX_TOKEN = 1024;

    auto acl_call = [=, &indices, &scores, &init_token_table, &init_token_list, &final_token_table,
                    &final_score_table, &token_idx_intra_expert]() -> int {
        getRoutingConfigurationsSBufV2_launch(block_dim, stream,
                                              (uint8_t *)indices.data_ptr(),
                                              (uint8_t *)scores.data_ptr(),
                                              (uint8_t *)init_token_table.data_ptr(),
                                              (uint8_t *)init_token_list.data_ptr(),
                                              (uint8_t *)final_token_table.data_ptr(),
                                              (uint8_t *)final_score_table.data_ptr(),
                                              (uint8_t *)token_idx_intra_expert.data_ptr(),
                                              start_expert_id, end_expert_id, local_expert_num,
                                              token_num, topk, UB_MAX_TOKEN);
        return 0;
    };
    at_npu::native::OpCommand::RunOpApi("getRoutingConfigurationsSBufV2", acl_call);
    return 0;
}

TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("get_routing_conf", getRoutingConfigurationsSBufV2);
}