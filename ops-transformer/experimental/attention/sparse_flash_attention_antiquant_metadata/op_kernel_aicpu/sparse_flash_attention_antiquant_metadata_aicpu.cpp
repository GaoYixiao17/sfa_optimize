/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file sparse_flash_attention_antiquant_metadata_aicpu.cpp
 * \brief
 */

#include "sparse_flash_attention_antiquant_metadata_aicpu.h"
#include "../../sparse_flash_attention_antiquant/op_kernel/sparse_flash_attention_antiquant_metadata.h"
#include "../../common/cpu_context_util.h"
#include <cstdio>
#include <math.h>

namespace aicpu {
uint32_t
SparseFlashAttentionAntiquantMetaDataCpuKernel::Compute(CpuKernelContext &ctx) {
  context_ = &ctx;
  bool success = Prepare(ctx) && BalanceSchedule() && GenMetaData();
  return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool SparseFlashAttentionAntiquantMetaDataCpuKernel::Prepare(
    CpuKernelContext &ctx) {
  // input
  actSeqLenQ_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenQ));
  actSeqLenKV_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenKV));
  sparseSeqLenKV_ = ctx.Input(static_cast<uint32_t>(ParamId::sparseSeqLenKV));
  // output
  metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

  bool requiredAttrs = GetAttrValue(ctx, "batch_size", batchSize_) &&
                       GetAttrValue(ctx, "query_seq_size", querySeqSize_) &&
                       GetAttrValue(ctx, "query_head_num", queryHeadNum_) &&
                       GetAttrValue(ctx, "kv_seq_size", kvSeqSize_) &&
                       GetAttrValue(ctx, "kv_head_num", kvHeadNum_) &&
                       GetAttrValue(ctx, "head_dim", headDim_) &&
                       GetAttrValue(ctx, "topk_size", topKSize_) &&
                       GetAttrValue(ctx, "sparse_block_size", sparseBlockSize_) && 
                       GetAttrValue(ctx, "aic_core_num", aicCoreNum_) && 
                       GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
  if (!requiredAttrs) {
    return false;
  }

  coreNum_ = aicCoreNum_;
  // attributes optional
  GetAttrValueOpt(ctx, "layout_query", layoutQuery_);
  GetAttrValueOpt(ctx, "layout_kv", layoutKV_);
  GetAttrValueOpt(ctx, "sparse_mode", sparseMode_);
  GetAttrValueOpt(ctx, "attention_mode", attentionMode_);
  GetAttrValueOpt(ctx, "rope_head_dim", ropeHeadDim_);
  GetAttrValueOpt(ctx, "sparse_shared_size", sparseSharedSize_);

  return (ParamsCheck() && ParamsInit());
}

bool SparseFlashAttentionAntiquantMetaDataCpuKernel::ParamsCheck() {
  if (actSeqLenQ_ != nullptr) {
    auto shape = actSeqLenQ_->GetTensorShape();
    auto data = actSeqLenQ_->GetData();
    auto dtype = actSeqLenQ_->GetDataType();

    KERNEL_CHECK_NULLPTR(shape, false,
                         "shape of actual_seq_lengths_query is null");
    KERNEL_CHECK_NULLPTR(data, false,
                         "data of actual_seq_lengths_query is null");
    KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                       "dtype %u of actual_seq_lengths_query is not int32 ", dtype);

    KERNEL_CHECK_FALSE(
        (shape->GetDims() == 1 && shape->GetDimSize(0) == batchSize_), false,
        "shape of actual_seq_lengths_query is not {%u,}", batchSize_);
  }

  if (actSeqLenKV_ != nullptr) {
    auto shape = actSeqLenKV_->GetTensorShape();
    auto data = actSeqLenKV_->GetData();
    auto dtype = actSeqLenKV_->GetDataType();

    KERNEL_CHECK_NULLPTR(shape, false,
                         "shape of actual_seq_lengths_kv is null");
    KERNEL_CHECK_NULLPTR(data, false, "data of actual_seq_lengths_kv is null");
    KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                       "dtype of actual_seq_lengths_kv is not int32");

    KERNEL_CHECK_FALSE(
        (shape->GetDims() == 1 && shape->GetDimSize(0) == batchSize_), false,
        "shape of actual_seq_lengths_query date is not {%u,}", batchSize_);
  }

  if (sparseSeqLenKV_ != nullptr) {
    auto shape = sparseSeqLenKV_->GetTensorShape();
    auto data = sparseSeqLenKV_->GetData();
    auto dtype = sparseSeqLenKV_->GetDataType();

    KERNEL_CHECK_NULLPTR(shape, false,
                         "shape of sparse_seq_lengths_kv is null");
    KERNEL_CHECK_NULLPTR(data, false, "data of sparse_seq_lengths_kv is null");
    KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                       "dtype of sparse_seq_lengths_kv is not int32");
    KERNEL_CHECK_FALSE(
        (shape->GetDims() == 1 && shape->GetDimSize(0) == batchSize_), false,
        "shape of sparse_seq_lengths_kv date is not {%u,}", batchSize_);
  }

  KERNEL_CHECK_FALSE((layoutQuery_ == "BSND" || layoutQuery_ == "TND"), false,
                     "layout_query invalid");

  KERNEL_CHECK_FALSE(
      (layoutKV_ == "BSND" || layoutKV_ == "TND" || layoutKV_ == "PA_BSND" || layoutKV_ == "PA_BNSD" || layoutKV_ == "PA_NZ"),
      false, "layout_kv invalid");
  KERNEL_CHECK_FALSE((sparseMode_ == 0 || sparseMode_ == 3), false,
                     "sparse_mode invalid");
  KERNEL_CHECK_FALSE(
      (attentionMode_ == 0 || attentionMode_ == 1 || attentionMode_ == 2),
      false, "attention_mode invalid");

  KERNEL_CHECK_NULLPTR(metaData_, false, "metadata is null");
  auto shape = metaData_->GetTensorShape();
  auto data = metaData_->GetData();
  auto dtype = metaData_->GetDataType();

  KERNEL_CHECK_NULLPTR(shape, false, "shape of metadata is null");
  KERNEL_CHECK_NULLPTR(data, false, "data of metadata is null");
  KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                     "dtype of metadata is not int32");
  KERNEL_CHECK_FALSE((shape->GetDims() == 1 &&
                      shape->GetDimSize(0) == optiling::SFA_META_SIZE),
                     false, "shape of sparse_seq_lengths_kv date is not {%u,}",
                     optiling::SFA_META_SIZE);

  KERNEL_CHECK_FALSE(
      (aicCoreNum_ != 0 && aivCoreNum_ != 0 && aivCoreNum_ % aicCoreNum_ == 0 &&
       aicCoreNum_ <= optiling::CORE_NUM &&
       aivCoreNum_ <= (2 * optiling::CORE_NUM)),
      false, "core num invalid aic:%u aiv:%u", aicCoreNum_,
      aivCoreNum_);  // more limit check with platform-core

  return true;
}

bool SparseFlashAttentionAntiquantMetaDataCpuKernel::ParamsInit() {
    groupSize_ = queryHeadNum_ / kvHeadNum_;
    mBaseSize_ = groupSize_ * sparseSharedSize_;
    s2BaseSize_ = 1024U;
    gS1BaseSizeOfFd_ = 8U;
    return true;
}

uint32_t SparseFlashAttentionAntiquantMetaDataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    if (actSeqLenQ_ == nullptr) {
        return querySeqSize_;
    }
    const int32_t *s1Ptr = (int32_t*)actSeqLenQ_->GetData();
    if (layoutQuery_ == "TND") {
        return (bIdx == 0) ? static_cast<uint32_t>(s1Ptr[bIdx]) :
           static_cast<uint32_t>(s1Ptr[bIdx] - s1Ptr[bIdx - 1U]);
    } else {
        return static_cast<uint32_t>(s1Ptr[bIdx]);
    }
}

uint32_t SparseFlashAttentionAntiquantMetaDataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    uint32_t s2Size = 0;
    if (actSeqLenKV_ == nullptr) {
        s2Size = kvSeqSize_;
    } else {
        const int32_t *s2Ptr = (int32_t*)actSeqLenKV_->GetData();
        if (layoutKV_ == "TND") {
            s2Size = (bIdx == 0) ? static_cast<uint32_t>(s2Ptr[bIdx]) :
            static_cast<uint32_t>(s2Ptr[bIdx] - s2Ptr[bIdx - 1U]);
        } else {
            s2Size = static_cast<uint32_t>(s2Ptr[bIdx]);
        }
    }

    uint32_t sparseSeqSize = GetSparseSeqSize(bIdx);
    if (sparseSeqSize < s2Size) {
        s2Size = sparseSeqSize;
    }
    return s2Size;
}

uint32_t SparseFlashAttentionAntiquantMetaDataCpuKernel::GetSparseSeqSize(uint32_t bIdx)
{
    if (sparseSeqLenKV_ == nullptr) {
        return 0U;
    } else {
        const int32_t *sparseS2Ptr = (int32_t*)sparseSeqLenKV_->GetData();
        if (layoutKV_ == "TND") {
            return (bIdx == 0) ? static_cast<uint32_t>(sparseS2Ptr[bIdx]) :
            static_cast<uint32_t>(sparseS2Ptr[bIdx] - sparseS2Ptr[bIdx - 1U]);
        } else {
            return static_cast<uint32_t>(sparseS2Ptr[bIdx]);
        }
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
{
    // 计算每个batch的切分，统计是否为空batch，记录最后有效batch（每个batch的每个N2切分是一样的）
    SplitInfo &splitInfo = splitContext.splitInfo;
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        uint32_t s1Size = GetS1SeqSize(bIdx);
        uint32_t s2Size = GetS2SeqSize(bIdx);
        splitInfo.s1GBaseNum[bIdx] = (s1Size * groupSize_ + (mBaseSize_ - 1U)) / mBaseSize_;
        splitInfo.s1GTailSize[bIdx] = (s1Size * groupSize_) % mBaseSize_;
        splitInfo.s2BaseNum[bIdx] = (s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        splitInfo.s2TailSize[bIdx] = s2Size % s2BaseSize_;
        if (splitInfo.s1GBaseNum[bIdx] != 0U && splitInfo.s2BaseNum[bIdx] != 0U) {
            splitInfo.isKvSeqAllZero = false;
        }
    }
    return;
}

int64_t SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcPreTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(sparseMode_);
    if (mode == SparseMode::BAND) {
        return static_cast<int64_t>(s1Size) - static_cast<int64_t>(s2Size) + preToken_;
    }
    return preToken_;
}

int64_t SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcNextTokenLeftUp(
    uint32_t s1Size, uint32_t s2Size)
{
    auto mode = static_cast<SparseMode>(sparseMode_);
    switch (mode) {
        case SparseMode::DEFAULT_MASK:
        case SparseMode::ALL_MASK:
        case SparseMode::LEFT_UP_CAUSAL:
            return nextToken_;
        case SparseMode::RIGHT_DOWN_CAUSAL:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size);
        case SparseMode::BAND:
            return static_cast<int64_t>(s2Size) - static_cast<int64_t>(s1Size) + nextToken_;
        default:
            return nextToken_;
    }
}

int64_t SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    float a = 25.04f;
    float b = 2.41f;
    float c = 54.99f;
    return static_cast<int64_t>(a * basicM + b * basicS2 + c);
}

BlockCost<int64_t> SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcCostTable(uint32_t s1NormalSize, 
    uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcS2Range(
    uint32_t s1GIdx, const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    // actual seq == 0
    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    // no mask
    if (!attentionMode_) { //attentionMaskFlag ?
        s2Start = 0U;
        s2End = (batchCache.s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
        return std::make_pair(s2Start, s2End);
    }

    // 1. calc index of s2FirstToken, s2LastToken by index of s1GFirstToken, s1GLastToken
    int64_t s1GFirstToken = static_cast<int64_t>(s1GIdx) * static_cast<int64_t>(mBaseSize_);
    int64_t s1GLastToken = std::min(s1GFirstToken + static_cast<int64_t>(mBaseSize_),
        static_cast<int64_t>(batchCache.s1Size) * static_cast<int64_t>(groupSize_)) - 1;
    
    int64_t s1FirstToken = 0;
    int64_t s1LastToken = 0;
    if (isS1G_) {
        s1FirstToken = s1GFirstToken / static_cast<int64_t>(groupSize_);
        s1LastToken = s1GLastToken / static_cast<int64_t>(groupSize_);
    } else {
        if (s1GFirstToken / batchCache.s1Size == s1GLastToken / batchCache.s1Size) {
            // start and end locate in one G
            s1FirstToken = s1GFirstToken % static_cast<int64_t>(batchCache.s1Size);
            s1LastToken = s1GLastToken % static_cast<int64_t>(batchCache.s1Size);
        } else {
            // start and end locate in tow or more G, but working same as crossing a complete block
            s1FirstToken = 0;
            s1LastToken = batchCache.s1Size;
        }
    }

    int64_t s2FirstToken = s1FirstToken - batchCache.preTokenLeftUp;
    int64_t s2LastToken = s1LastToken + batchCache.nextTokenLeftUp;

    // 2. trans index of token to index of block
    // no valid token
    if (s2FirstToken >= static_cast<int64_t>(batchCache.s2Size) || s2LastToken < 0 || s2LastToken < s2FirstToken) {
        s2Start = 0U;
        s2End = 0U;
        return std::make_pair(s2Start, s2End);
    }
    // get valid range
    s2FirstToken = Clip(s2FirstToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));
    s2LastToken = Clip(s2LastToken, static_cast<int64_t>(0), static_cast<int64_t>(batchCache.s2Size - 1U));

    s2Start = static_cast<uint32_t>(s2FirstToken) / s2BaseSize_;
    s2End = static_cast<uint32_t>(s2LastToken) / s2BaseSize_ + 1U; // end of block index, Right-open interval

    return std::make_pair(s2Start, s2End);
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcBatchCache(
    uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.preTokenLeftUp = CalcPreTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.nextTokenLeftUp = CalcNextTokenLeftUp(batchCache.s1Size, batchCache.s2Size);
    batchCache.typeCost = CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[bIdx],
        splitInfo.s2TailSize[bIdx]);
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcS1GCache(uint32_t s1GIdx, 
    const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;

    if (s1GCache.s2Start >= s1GCache.s2End || splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GBlock = 0;
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
        return;
    }

    // 计算S2方向满块、尾块数量
    s1GCache.s1GBlock = s1GCache.s2End - s1GCache.s2Start;
    uint32_t curTailS2Num = (splitInfo.s2TailSize[batchCache.bIdx] != 0U &&
        s1GCache.s2End == splitInfo.s2BaseNum[batchCache.bIdx]) ? 1U : 0U;
    uint32_t curNormalS2Num = s1GCache.s1GBlock - curTailS2Num;
    if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
        s1GCache.s1GCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[TAIL_BLOCK][TAIL_BLOCK] :
                                 batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[TAIL_BLOCK][NORMAL_BLOCK];
    } else {
        s1GCache.s1GCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK] * curNormalS2Num +
            batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] * curTailS2Num;
        s1GCache.s1GLastBlockCost = curTailS2Num > 0U ? batchCache.typeCost[NORMAL_BLOCK][TAIL_BLOCK] :
                                 batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
        s1GCache.s1GNormalBlockCost = batchCache.typeCost[NORMAL_BLOCK][NORMAL_BLOCK];
    }

    if (s1GCache.s1GBlock == 1) {
        s1GCache.s1GLastBlockCost += V0_COST;
    }
    s1GCache.s1GCost += V0_COST;
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcBatchCost(
    uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    costInfo.bN2CostOfEachBatch[bIdx] = 0;
    costInfo.bN2BlockOfEachBatch[bIdx] = 0U;
    costInfo.bN2LastBlockCostOfEachBatch[bIdx] = 0U;

    if (GetS1SeqSize(bIdx) == 0U || GetS2SeqSize(bIdx) == 0U) {
        return;
    }

    BatchCache bCache;
    S1GCache s1GCache;
    CalcBatchCache(bIdx, splitContext, bCache);
    for (uint32_t s1GIdx = 0; s1GIdx < splitInfo.s1GBaseNum[bIdx]; s1GIdx++) {
        CalcS1GCache(s1GIdx, splitContext, bCache, s1GCache);
        costInfo.bN2CostOfEachBatch[bIdx] += s1GCache.s1GCost;
        costInfo.bN2BlockOfEachBatch[bIdx] += s1GCache.s1GBlock;
        if(s1GCache.s1GBlock > 0){
            costInfo.bN2LastBlockCostOfEachBatch[bIdx] = s1GCache.s1GLastBlockCost;
        }
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcCostInfo(SplitContext &splitContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    CostInfo &costInfo = splitContext.costInfo;

    if (splitInfo.isKvSeqAllZero) {
        costInfo.totalCost = 0;
        costInfo.totalBlockNum = 0U;
        return;
    }

    // 计算batch的负载并记录，用于按batch分配，需要按行计算起止点，统计块数、负载
    for (uint32_t bIdx = 0; bIdx < batchSize_; bIdx++) {
        CalcBatchCost(bIdx, splitContext, costInfo);
        costInfo.totalCost += costInfo.bN2CostOfEachBatch[bIdx] * kvHeadNum_;
        costInfo.totalBlockNum += costInfo.bN2BlockOfEachBatch[bIdx] * kvHeadNum_;
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    const CostInfo &costInfo = splitContext.costInfo;

    bool UpdateS1G = false;
    bool UpdateBatch = false;

    // Update S2
    if (assignContext.curS2Idx >= assignContext.s1GCache.s2End) {    // 边界assignInfo.s2End是取不到的开区间
        assignContext.curS2Idx = 0U;
        assignContext.curS1GIdx++;
        UpdateS1G = true;
    }

    // Update S1G
    if (assignContext.curS1GIdx >= splitInfo.s1GBaseNum[assignContext.curBIdx]) {
        assignContext.curS1GIdx = 0U;
        assignContext.curBN2Idx++;
    }

    // Update Batch
    if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {  // 所有负载全部分配完，设置最后一个核的右开区间，返回
        assignContext.curS1GIdx = 0U;
        assignContext.curS2Idx = 0U;
        assignContext.isFinished = true;
        return;
    }

    if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
        assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
        assignContext.curS1GIdx = 0U;
        UpdateBatch = true;
        UpdateS1G = true;
    }

    // Update Cache
    if (UpdateBatch) {
        CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    }
    if (UpdateS1G) {
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }
    const CostInfo &costInfo = splitContext.costInfo;
    while (assignContext.bN2Cost == 0 || IsWithinTolerance(assignContext.coreCache.costLimit,
        costInfo.bN2LastBlockCostOfEachBatch[assignContext.curBIdx] / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.bN2Cost)) {
        assignContext.coreCache.cost += assignContext.bN2Cost;
        assignContext.coreCache.block += assignContext.bN2Block;
        assignContext.curBN2Idx++;

        // to the end
        if (assignContext.curBN2Idx == batchSize_ * kvHeadNum_) {
            assignContext.curS1GIdx = 0U;
            assignContext.curS2Idx = 0U;
            assignContext.isFinished = true;
            return;
        }

        // next batch
        if (assignContext.curBN2Idx / kvHeadNum_ != assignContext.curBIdx) {
            assignContext.curBIdx = assignContext.curBN2Idx / kvHeadNum_;
            CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
        }

        assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
        assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
        assignContext.curS1GIdx = 0U;
        CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit,
        assignContext.s1GCache.s1GLastBlockCost / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + assignContext.s1GCache.s1GCost)) {
        assignContext.coreCache.cost += assignContext.s1GCache.s1GCost;
        assignContext.coreCache.block += assignContext.s1GCache.s1GBlock;

        // 当前batch被分配一行出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost > assignContext.s1GCache.s1GCost ?
                                assignContext.bN2Cost - assignContext.s1GCache.s1GCost : 0;
        assignContext.bN2Block = assignContext.bN2Block > assignContext.s1GCache.s1GBlock ?
                                 assignContext.bN2Block - assignContext.s1GCache.s1GBlock : 0U;
        // 计算新一行的信息
        do{
            assignContext.curS1GIdx++;
            CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
        }while(assignContext.s1GCache.s1GBlock == 0);
        assignContext.curS2Idx = assignContext.s1GCache.s2Start;
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }
    int64_t realCost = curCost;
    if (assignContext.curS2Idx == 0 && assignContext.s1GCache.s1GBlock != 1) {
        realCost += V0_COST;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, realCost / FA_TOLERANCE_RATIO,
        assignContext.coreCache.cost + realCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += realCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - realCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - realCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;

        realCost = curCost;
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }
    int64_t realCost = curCost;
    if (assignContext.curS2Idx == 0 && assignContext.s1GCache.s1GBlock != 1) {
        realCost += V0_COST;
    }

    assignContext.coreCache.cost += realCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - realCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - realCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext); 
}

bool SparseFlashAttentionAntiquantMetaDataCpuKernel::IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes)
{
    // 切分点大概率不会刚好在行尾，因此滞后处理归约信息的统计，到下一个切分点再判断是否需要归约
    // 核0无需处理
    if (assignContext.curCoreIdx == 0U) {
        return false;
    }
    // 无跨核行，无需处理
    if (assignContext.curKvSplitPart <= 1U) {
        return false;
    }
    // 需要归约的行还未处理完
    if (assignContext.curBN2Idx == splitRes.bN2End[assignContext.curCoreIdx - 1U] &&
        assignContext.curS1GIdx == splitRes.gS1End[assignContext.curCoreIdx - 1U]) {
        return false;
    }
    return true;
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;
    // 需要规约的行是上一个核的切分点所在位置
    uint32_t splitBIdx = result.bN2End[assignContext.curCoreIdx - 1U] / kvHeadNum_;
    uint32_t splitS1GIdx = result.gS1End[assignContext.curCoreIdx - 1U];
    uint32_t s1Size = GetS1SeqSize(splitBIdx);

    // 计算归约数据的FD均衡划分信息
    uint32_t curFdS1gSize = (splitS1GIdx == splitInfo.s1GBaseNum[splitBIdx] - 1U) ?
                            (s1Size * groupSize_ - splitS1GIdx * mBaseSize_) : mBaseSize_;
    uint32_t curFdS1gSplitPart = (curFdS1gSize + gS1BaseSizeOfFd_ - 1U) / gS1BaseSizeOfFd_;
    uint32_t curFdS1gLastPartSize = curFdS1gSize - (gS1BaseSizeOfFd_ * (curFdS1gSplitPart - 1U));
    // 记录
    result.maxS2SplitNum = std::max(result.maxS2SplitNum, assignContext.curKvSplitPart);
    // 若存在头归约，则切分点一定为上一个核结束的位置
    result.fdRes.bN2IdxOfFdHead[result.numOfFdHead] = result.bN2End[assignContext.curCoreIdx - 1U];
    result.fdRes.gS1IdxOfFdHead[result.numOfFdHead] = result.gS1End[assignContext.curCoreIdx - 1U];
    result.fdRes.s2SplitNumOfFdHead[result.numOfFdHead] = assignContext.curKvSplitPart;
    result.fdRes.gS1SplitNumOfFdHead[result.numOfFdHead] = curFdS1gSplitPart;
    result.fdRes.gS1LastPartSizeOfFdHead[result.numOfFdHead] = curFdS1gLastPartSize;
    result.numOfFdHead++;
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::CalcSplitPlan(uint32_t coreNum,
    int64_t costLimit, const SplitContext &splitContext, SplitResult &result)
{
    const CostInfo &costInfo = splitContext.costInfo;

    if (coreNum == 0U) {
        return;
    }
    result.maxCost = 0U;
    result.usedCoreNum = 0U;

    AssignContext assignContext {};
    assignContext.curBIdx = 0U;
    assignContext.curS1GIdx = 0U;
    assignContext.unassignedCost = costInfo.totalCost;
    assignContext.bN2Cost = costInfo.bN2CostOfEachBatch[assignContext.curBIdx];
    assignContext.bN2Block = costInfo.bN2BlockOfEachBatch[assignContext.curBIdx];
    CalcBatchCache(assignContext.curBIdx, splitContext, assignContext.batchCache);
    CalcS1GCache(assignContext.curS1GIdx, splitContext, assignContext.batchCache, assignContext.s1GCache);
    assignContext.curS2Idx = assignContext.s1GCache.s2Start;

    for (uint32_t i = 0; i < coreNum; ++i) {
        if (result.maxCost > costLimit) {
            return;
        }
        if (assignContext.isFinished || assignContext.unassignedCost <= 0) {
            break;
        }

        assignContext.curCoreIdx = i;
        result.fdRes.s2SplitStartIdxOfCore[assignContext.curCoreIdx] = assignContext.curKvSplitPart - 1U;

        assignContext.coreCache = {};
        assignContext.coreCache.costLimit = assignContext.unassignedCost / (coreNum - assignContext.curCoreIdx);

        // 1、按整batch分配
        AssignByBatch(splitContext, assignContext);
        // 2、按行分配
        AssignByRow(splitContext, assignContext);
        // 3、按块分配
        AssignByBlock(splitContext, assignContext);
        // 4、强制分配
        if (assignContext.coreCache.block == 0) {
            ForceAssign(splitContext, assignContext);
        }

        result.bN2End[i] = assignContext.curBN2Idx;
        result.gS1End[i] = assignContext.curS1GIdx;
        result.s2End[i] = assignContext.curS2Idx;
        result.maxCost = std::max(result.maxCost, assignContext.coreCache.cost);

        assignContext.unassignedCost -= assignContext.coreCache.cost;

        // 对之前的归约信息进行记录并清理
        if (IsNeedRecordFDInfo(assignContext, result)) {
            RecordFDInfo(splitContext, assignContext, result);
            assignContext.curKvSplitPart = 1U;
        }

        // 更新S2切分信息
        if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
            assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
            assignContext.curKvSplitPart++;
        }
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;
    splitRes.numOfFdHead = tmpRes.numOfFdHead;
    splitRes.maxS2SplitNum = tmpRes.maxS2SplitNum;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.gS1End[i] = tmpRes.gS1End[i];
        splitRes.s2End[i] = tmpRes.s2End[i];

        splitRes.fdRes.bN2IdxOfFdHead[i] = tmpRes.fdRes.bN2IdxOfFdHead[i];
        splitRes.fdRes.gS1IdxOfFdHead[i] = tmpRes.fdRes.gS1IdxOfFdHead[i];
        splitRes.fdRes.s2SplitNumOfFdHead[i] = tmpRes.fdRes.s2SplitNumOfFdHead[i];
        splitRes.fdRes.s2SplitStartIdxOfCore[i] = tmpRes.fdRes.s2SplitStartIdxOfCore[i];
        splitRes.fdRes.gS1SplitNumOfFdHead[i] = tmpRes.fdRes.gS1SplitNumOfFdHead[i];
        splitRes.fdRes.gS1LastPartSizeOfFdHead[i] = tmpRes.fdRes.gS1LastPartSizeOfFdHead[i];
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::ClearTmpResult(SplitResult &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;
    tmpResult.numOfFdHead = 0U;
    tmpResult.maxS2SplitNum = 0U;
    tmpResult.usedVecNumOfFd = 0U;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.gS1End[i] = 0U;
        tmpResult.s2End[i] = 0U;
        tmpResult.fdRes.bN2IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1IdxOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.s2SplitStartIdxOfCore[i] = 0U;
        tmpResult.fdRes.gS1SplitNumOfFdHead[i] = 0U;
        tmpResult.fdRes.gS1LastPartSizeOfFdHead[i] = 0U;
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::RollBackCursor(const SplitContext &splitContext,
    const CostInfo &costInfo, SplitResult &splitRes)
{
    for (size_t i = 0; i < splitRes.usedCoreNum; ++i) {
        // x, y, z
        if (splitRes.s2End[i] > 0U) {
            splitRes.s2End[i] = splitRes.s2End[i] - 1U;
            continue;
        }
        uint32_t bIdx = splitRes.bN2End[i] / kvHeadNum_;
        // x, y, 0
        if (splitRes.gS1End[i] > 0U) {
            splitRes.gS1End[i] = splitRes.gS1End[i] - 1U;
            splitRes.s2End[i] = splitContext.splitInfo.s2BaseNum[bIdx] - 1U;
            continue;
        }

        // x, 0, 0
        uint32_t bN2Idx = splitRes.bN2End[i] > 0U ? splitRes.bN2End[i] - 1U : 0U;
        bIdx = bN2Idx / kvHeadNum_;

        // last end point
        if (i == splitRes.usedCoreNum - 1U && costInfo.bN2BlockOfEachBatch[bIdx] == 0U) {
            splitRes.bN2End[i] = batchSize_ * kvHeadNum_ - 1;
            splitRes.gS1End[i] = splitContext.splitInfo.s1GBaseNum[bIdx] > 0 ?
                                 splitContext.splitInfo.s1GBaseNum[bIdx] - 1U : 0U;
            splitRes.s2End[i] = splitContext.splitInfo.s2BaseNum[bIdx] > 0 ?
                                splitContext.splitInfo.s2BaseNum[bIdx] - 1U : 0U;
            continue;
        }

        while (bN2Idx > 0U && costInfo.bN2BlockOfEachBatch[bIdx] == 0U) {
            bN2Idx -= 1U;
            bIdx = bN2Idx / kvHeadNum_;
        }

        if (costInfo.bN2BlockOfEachBatch[bIdx] != 0U) {
            splitRes.bN2End[i] = bN2Idx;
            splitRes.gS1End[i] = splitContext.splitInfo.s1GBaseNum[bIdx] - 1U;
            splitRes.s2End[i] = splitContext.splitInfo.s2BaseNum[bIdx] - 1U;
        } else {
            splitRes.bN2End[i] = 0U;
            splitRes.gS1End[i] = 0U;
            splitRes.s2End[i] = 0U;
        }
    }
}

void SparseFlashAttentionAntiquantMetaDataCpuKernel::SplitFD(SplitResult &result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }
    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);  // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = result.fdRes.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < result.fdRes.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector);  // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * FD_TOLERANCE_RATIO) {
                result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector);  // 当前未分配的总负载
                // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    result.usedVecNumOfFd = curCoreIndex + 1;
}

bool SparseFlashAttentionAntiquantMetaDataCpuKernel::BalanceSchedule() {
    SplitContext splitContext(batchSize_);

    // 1、划分基本块，统计信息
    CalcSplitInfo(splitContext);
    // 全空case
    if (splitContext.splitInfo.isKvSeqAllZero) {
        splitRes_.usedCoreNum = 1U;
        splitRes_.bN2End[0] = batchSize_ * kvHeadNum_ - 1;
        splitRes_.gS1End[0] = 0U;
        splitRes_.s2End[0] = 0U;
        return true;
    }
    CalcCostInfo(splitContext);

    // 2、获取每个核的分配方案
    uint32_t maxCore = std::min(coreNum_, splitContext.costInfo.totalBlockNum);
    uint32_t minCore = static_cast<uint32_t>(
        std::sqrt(static_cast<float>(splitContext.costInfo.totalBlockNum) + 0.25f) + 0.5f);
    minCore = std::min(minCore, maxCore);

    splitRes_.maxCost = INT64_MAX;
    splitRes_.usedCoreNum = 1U;
    SplitResult tmpResult {coreNum_, aivCoreNum_ / aicCoreNum_}; // C: V = 1: 2, TODO: C: V = 1: 1 ?
    for (uint32_t i = minCore; i <= maxCore; ++i) {
        CalcSplitPlan(i, splitRes_.maxCost, splitContext, tmpResult);
        if (tmpResult.maxCost < splitRes_.maxCost) {
            CopyTmpResult(tmpResult, splitRes_);
        }
        ClearTmpResult(tmpResult);
    }

    // 3、存在FD任务，对FD进行负载均衡分配
    if (splitRes_.numOfFdHead > 0U) {
        SplitFD(splitRes_);
    }
    splitRes_.usedCoreNum = std::max(splitRes_.usedCoreNum, 1U);  // 至少使用1个core
    RollBackCursor(splitContext, splitContext.costInfo, splitRes_);
    return true;
}

bool SparseFlashAttentionAntiquantMetaDataCpuKernel::GenMetaData() {
    optiling::detail::SfaMetaData* metaDataPtr = (optiling::detail::SfaMetaData*)metaData_->GetData();
    metaDataPtr->usedCoreNum = splitRes_.usedCoreNum;
    metaDataPtr->numOfFdHead = splitRes_.numOfFdHead;
    metaDataPtr->usedVecNumOfFd = splitRes_.usedVecNumOfFd;
    metaDataPtr->mBaseSize = mBaseSize_;
    metaDataPtr->s2BaseSize = s2BaseSize_;
    metaDataPtr->gS1BaseSizeOfFd = gS1BaseSizeOfFd_;

    for (size_t i = 0; i < coreNum_; ++i) {
        metaDataPtr->bN2End[i] = splitRes_.bN2End[i];
        metaDataPtr->gS1End[i] = splitRes_.gS1End[i];
        metaDataPtr->s2End[i] = splitRes_.s2End[i];
        metaDataPtr->fdRes.bN2IdxOfFdHead[i] = splitRes_.fdRes.bN2IdxOfFdHead[i];
        metaDataPtr->fdRes.gS1IdxOfFdHead[i] = splitRes_.fdRes.gS1IdxOfFdHead[i];
        metaDataPtr->fdRes.s2SplitNumOfFdHead[i] = splitRes_.fdRes.s2SplitNumOfFdHead[i];
        metaDataPtr->fdRes.s2SplitStartIdxOfCore[i] = splitRes_.fdRes.s2SplitStartIdxOfCore[i];
        metaDataPtr->fdRes.gS1SplitNumOfFdHead[i] = splitRes_.fdRes.gS1SplitNumOfFdHead[i];
        metaDataPtr->fdRes.gS1LastPartSizeOfFdHead[i] = splitRes_.fdRes.gS1LastPartSizeOfFdHead[i];
    }
    if (splitRes_.numOfFdHead > 0U) {
        for (size_t i = 0; i < coreNum_ * 2U; ++i) {
            metaDataPtr->fdRes.gS1IdxEndOfFdHead[i] = splitRes_.fdRes.gS1IdxEndOfFdHead[i];
            metaDataPtr->fdRes.gS1IdxEndOfFdHeadSplit[i] = splitRes_.fdRes.gS1IdxEndOfFdHeadSplit[i];
        }
    }
    return true;
}

static const char *sfaKernelType = "KvQuantSparseFlashAttentionMetadata";
REGISTER_CPU_KERNEL(sfaKernelType, SparseFlashAttentionAntiquantMetaDataCpuKernel);

}; // namespace aicpu

