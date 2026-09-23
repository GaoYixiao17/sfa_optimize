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
 * \file quant_sals_indexer_metadata_aicpu.cpp
 * \brief
 */

#include "quant_sals_indexer_metadata_aicpu.h"
#include <cstdio>
#include <math.h>
#include "../../quant_sals_indexer/op_kernel/quant_sals_indexer_metadata.h"
#include "../../common/cpu_context_util.h"


namespace aicpu {
uint32_t QuantSalsIndexerMetaDataCpuKernel::Compute(CpuKernelContext &ctx) {
  context_ = &ctx;
  bool success = Prepare(ctx) && BalanceSchedule() && GenMetaData();
  return success ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool QuantSalsIndexerMetaDataCpuKernel::Prepare(CpuKernelContext &ctx) {
  // input
  actSeqLenKV_ = ctx.Input(static_cast<uint32_t>(ParamId::actSeqLenKV));
  // output
  metaData_ = ctx.Output(static_cast<uint32_t>(ParamId::metaData));

  bool requiredAttrs =
      GetAttrValue(ctx, "batch_size", batchSize_) &&
      GetAttrValue(ctx, "key_seq_size", kvSeqSize_) &&
      GetAttrValue(ctx, "key_head_num", kvHeadNum_) &&
      GetAttrValue(ctx, "fixed_tail_count", fixedTailCount_) &&
      GetAttrValue(ctx, "sparse_block_size", sparseBlockSize_) &&
      GetAttrValue(ctx, "aic_core_num", aicCoreNum_) &&
      GetAttrValue(ctx, "aiv_core_num", aivCoreNum_);
  if (!requiredAttrs) {
    return false;
  }
  coreNum_ = aicCoreNum_;
  return (ParamsCheck() && ParamsInit());
}

bool QuantSalsIndexerMetaDataCpuKernel::ParamsCheck() {
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

  KERNEL_CHECK_NULLPTR(metaData_, false, "metadata is null");
  auto shape = metaData_->GetTensorShape();
  auto data = metaData_->GetData();
  auto dtype = metaData_->GetDataType();

  KERNEL_CHECK_NULLPTR(shape, false, "shape of metadata is null");
  KERNEL_CHECK_NULLPTR(data, false, "data of metadata is null");
  KERNEL_CHECK_FALSE((dtype == DataType::DT_INT32), false,
                     "dtype of metadata is not int32");
  KERNEL_CHECK_FALSE((shape->GetDims() == 1 &&
                      shape->GetDimSize(0) == optiling::QSI_META_SIZE),
                     false, "shape of sparse_seq_lengths_kv date is not {%u,}",
                     optiling::QSI_META_SIZE);
  KERNEL_CHECK_FALSE(
      (aicCoreNum_ != 0 && aivCoreNum_ != 0 && aivCoreNum_ % aicCoreNum_ == 0 &&
       aicCoreNum_ <= optiling::CORE_NUM &&
       aivCoreNum_ <= (2 * optiling::CORE_NUM)),
      false, "core num invalid aic:%u aiv:%u", aicCoreNum_,
      aivCoreNum_);  // more limit check with platform-core

  return true;
}

bool QuantSalsIndexerMetaDataCpuKernel::ParamsInit() {
    groupSize_ = 1U;
    mBaseSize_ = 1U;
    s2BaseSize_ = 2048U;
    return true;
}

uint32_t QuantSalsIndexerMetaDataCpuKernel::GetS1SeqSize(uint32_t bIdx)
{
    return 1U;
}

uint32_t QuantSalsIndexerMetaDataCpuKernel::GetS2SeqSize(uint32_t bIdx)
{
    uint32_t s2Size = 0;
    if (actSeqLenKV_ == nullptr) {
        s2Size = kvSeqSize_;
    } else {
        const int32_t *s2Ptr = (int32_t*)actSeqLenKV_->GetData();
        s2Size = static_cast<uint32_t>(s2Ptr[bIdx]);
    }

    // 即是S2为0，QSI依然存在刷-1和搬运动作。如果actual kv中存大量0，需要对刷新和搬运操作计算负载，避免集中在一个vector上操作
    // 而如果存在部分0，刷新和搬运操作理应会被正常计算流水掩盖，所以需要给一个合适小的值。
    // 此处返回512U是计算测试所得经验值。
    int32_t totalNCount = (s2Size + sparseBlockSize_ - 1) / sparseBlockSize_;
    if (totalNCount <= fixedTailCount_) {
        return 512U;         
    }
    return (totalNCount - fixedTailCount_) * sparseBlockSize_;
}

void QuantSalsIndexerMetaDataCpuKernel::CalcSplitInfo(SplitContext &splitContext)
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

int64_t QuantSalsIndexerMetaDataCpuKernel::CalcCost(
    uint32_t basicM, uint32_t basicS2)
{
    uint32_t alignCoefM = 16U;
    uint32_t alignCoefS2 = 64U;
    uint32_t alignBasicM = (basicM + alignCoefM - 1U) >> 4U;      // 按alignCoefM对齐，向上取整，4：移位操作实现除16
    uint32_t alignBasicS2 = (basicS2 + alignCoefS2 - 1U) >> 6U;   // 按alignCoefS2对齐，向上取整，6：移位操作实现除64
    return static_cast<int64_t>(6U * alignBasicM + 10U * alignBasicS2);                 // 6：M轴系数，10：S2轴系数
}

BlockCost<int64_t> QuantSalsIndexerMetaDataCpuKernel::CalcCostTable(uint32_t s1NormalSize, 
    uint32_t s2NormalSize, uint32_t s1GTailSize, uint32_t s2TailSize)
{
    BlockCost<int64_t> typeCost {};
    typeCost[NORMAL_BLOCK][NORMAL_BLOCK] = CalcCost(s1NormalSize, s2NormalSize);
    typeCost[TAIL_BLOCK][NORMAL_BLOCK] = (s1GTailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2NormalSize);
    typeCost[NORMAL_BLOCK][TAIL_BLOCK] = (s2TailSize == 0U) ? 0U : CalcCost(s1NormalSize, s2TailSize);
    typeCost[TAIL_BLOCK][TAIL_BLOCK] = (s1GTailSize == 0U || s2TailSize == 0U) ? 0U : CalcCost(s1GTailSize, s2TailSize);
    return typeCost;
}

Range<uint32_t> QuantSalsIndexerMetaDataCpuKernel::CalcS2Range(
    uint32_t s1GIdx, const BatchCache &batchCache)
{
    uint32_t s2Start = 0U;
    uint32_t s2End = 0U;

    if (batchCache.s1Size == 0U || batchCache.s2Size == 0U) {
        return std::make_pair(s2Start, s2End);
    }

    s2End = (batchCache.s2Size + s2BaseSize_ - 1U) / s2BaseSize_;
    return std::make_pair(s2Start, s2End);
}

void QuantSalsIndexerMetaDataCpuKernel::CalcBatchCache(
    uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    batchCache.bIdx = bIdx;
    batchCache.s1Size = GetS1SeqSize(bIdx);
    batchCache.s2Size = GetS2SeqSize(bIdx);
    batchCache.typeCost = CalcCostTable(mBaseSize_, s2BaseSize_, splitInfo.s1GTailSize[bIdx],
        splitInfo.s2TailSize[bIdx]);
}

void QuantSalsIndexerMetaDataCpuKernel::CalcS1GCache(uint32_t s1GIdx, 
    const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache)
{
    const SplitInfo &splitInfo = splitContext.splitInfo;

    s1GCache.bIdx = batchCache.bIdx;
    s1GCache.s1GIdx = s1GIdx;

    auto s2Range = CalcS2Range(s1GIdx, batchCache);
    s1GCache.s2Start = s2Range.first;
    s1GCache.s2End = s2Range.second;

    if (s1GCache.s2Start >= s1GCache.s2End) {
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
    if (splitInfo.s1GBaseNum[batchCache.bIdx] == 0) {
        s1GCache.s1GCost = 0;
        s1GCache.s1GLastBlockCost = 0;
        s1GCache.s1GNormalBlockCost = 0;
    } else if (s1GIdx == (splitInfo.s1GBaseNum[batchCache.bIdx] - 1U) && splitInfo.s1GTailSize[batchCache.bIdx] != 0U) {
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
}

void QuantSalsIndexerMetaDataCpuKernel::CalcBatchCost(
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

void QuantSalsIndexerMetaDataCpuKernel::CalcCostInfo(SplitContext &splitContext)
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

void QuantSalsIndexerMetaDataCpuKernel::UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext)
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

void QuantSalsIndexerMetaDataCpuKernel::AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext)
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

void QuantSalsIndexerMetaDataCpuKernel::AssignByRow(const SplitContext &splitContext, AssignContext &assignContext)
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

void QuantSalsIndexerMetaDataCpuKernel::AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    while (IsWithinTolerance(assignContext.coreCache.costLimit, curCost / FA_TOLERANCE_RATIO, 
            assignContext.coreCache.cost + curCost)) { // (costLimit - curCostOnCore) * FA_TOLERANCE_RATIO > curCost；至少分配1块
        assignContext.coreCache.cost += curCost;
        assignContext.coreCache.block++;
        assignContext.curS2Idx++;
        // 当前batch被分配一块出去，更新剩余负载
        assignContext.bN2Cost = assignContext.bN2Cost - curCost;
        // 当前行被分配一块出去，更新剩余负载
        assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
        assignContext.bN2Block--;
        assignContext.s1GCache.s1GBlock--;
    }
}

void QuantSalsIndexerMetaDataCpuKernel::ForceAssign(const SplitContext &splitContext, AssignContext &assignContext)
{
    if (assignContext.isFinished) {
        return;
    }

    int64_t curCost = assignContext.s1GCache.s1GNormalBlockCost;
    if (assignContext.curS2Idx == (assignContext.s1GCache.s2End - 1U)) {
        curCost = assignContext.s1GCache.s1GLastBlockCost;
    }

    assignContext.coreCache.cost += curCost;
    assignContext.coreCache.block++;
    assignContext.curS2Idx++;
    // 当前batch被分配一块出去，更新剩余负载
    assignContext.bN2Cost = assignContext.bN2Cost - curCost;
    assignContext.bN2Block--;
    // 当前行被分配一块出去，更新剩余负载
    assignContext.s1GCache.s1GCost = assignContext.s1GCache.s1GCost - curCost;
    assignContext.s1GCache.s1GBlock--;
    UpdateCursor(splitContext, assignContext); 
}

void QuantSalsIndexerMetaDataCpuKernel::CalcSplitPlan(uint32_t coreNum,
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
    }

    result.usedCoreNum = assignContext.curCoreIdx + 1;
}

void QuantSalsIndexerMetaDataCpuKernel::CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes)
{
    uint64_t len = tmpRes.bN2End.size();
    splitRes.usedCoreNum = tmpRes.usedCoreNum;
    splitRes.maxCost = tmpRes.maxCost;

    for (size_t i = 0; i < len; ++i) {
        splitRes.bN2End[i] = tmpRes.bN2End[i];
        splitRes.gS1End[i] = tmpRes.gS1End[i];
        splitRes.s2End[i] = tmpRes.s2End[i];
    }
}

void QuantSalsIndexerMetaDataCpuKernel::ClearTmpResult(SplitResult &tmpResult)
{
    uint64_t len = tmpResult.bN2End.size();
    tmpResult.usedCoreNum = 0U;
    tmpResult.maxCost = 0;

    for (size_t i = 0; i < len; ++i) {
        tmpResult.bN2End[i] = 0U;
        tmpResult.gS1End[i] = 0U;
        tmpResult.s2End[i] = 0U;
    }
}

void QuantSalsIndexerMetaDataCpuKernel::RollBackCursor(const SplitContext &splitContext,
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

bool QuantSalsIndexerMetaDataCpuKernel::BalanceSchedule() {
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

    splitRes_.usedCoreNum = std::max(splitRes_.usedCoreNum, 1U);  // 至少使用1个core
    RollBackCursor(splitContext, splitContext.costInfo, splitRes_);
    if (GetS2SeqSize(batchSize_-1) == 0U) {
        splitRes_.bN2End[splitRes_.usedCoreNum-1] = batchSize_ * kvHeadNum_ - 1;
        splitRes_.gS1End[splitRes_.usedCoreNum-1] = 0U;
        splitRes_.s2End[splitRes_.usedCoreNum-1] = 0U;
    }
    return true;
}

bool QuantSalsIndexerMetaDataCpuKernel::GenMetaData() {
    optiling::detail::QsiMetaData* metaDataPtr = (optiling::detail::QsiMetaData*)metaData_->GetData();
    metaDataPtr->usedCoreNum = splitRes_.usedCoreNum;

    for (size_t i = 0; i < coreNum_; ++i) {
        metaDataPtr->bN2End[i] = splitRes_.bN2End[i];
        metaDataPtr->gS1End[i] = splitRes_.gS1End[i];
        metaDataPtr->s2End[i] = splitRes_.s2End[i];
    }


    return true;
}

static const char *qsiKernelType = "QuantSalsIndexerMetadata";
REGISTER_CPU_KERNEL(qsiKernelType, QuantSalsIndexerMetaDataCpuKernel);

}; // namespace aicpu

