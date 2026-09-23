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
 * \file quant_sals_indexer_metadata_aicpu.h
 * \brief
 */

#ifndef QUANT_SALS_INDEXER_METADATA_AICPU_H
#define QUANT_SALS_INDEXER_METADATA_AICPU_H

#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace aicpu {
constexpr int64_t FA_TOLERANCE_RATIO = 2;
constexpr uint32_t FD_TOLERANCE_RATIO = 2U;

enum BlockType : uint32_t {
    NORMAL_BLOCK = 0,
    TAIL_BLOCK,
    BLOCK_MAX_TYPE
};

template<class T>
using Range = std::pair<T, T>;

template<class T>
using BlockCost = std::array<std::array<T, static_cast<size_t>(BLOCK_MAX_TYPE)>, static_cast<size_t>(BLOCK_MAX_TYPE)>;

template<typename T>
inline bool IsWithinTolerance(T limit, T tolerance, T value)
{
    return limit + tolerance >= value;
}

// 分核功能模块输出：FA阶段的核间分核信息
struct SplitResult {
    uint32_t usedCoreNum { 0U };        // 使用的核数量
    uint32_t vecCubeRatio { 0U };        // vec 与 cube 核数比例
    std::vector<uint32_t> bN2End {};    // 每个核处理数据的BN2结束点
    std::vector<uint32_t> gS1End {};    // 每个核处理数据的GS1结束点
    std::vector<uint32_t> s2End {};     // 每个核处理数据的S2结束点
    int64_t maxCost { 0 };            // 慢核开销

    SplitResult(uint32_t coreNum, uint32_t ratio) :
        bN2End(coreNum),
        vecCubeRatio(ratio),
        gS1End(coreNum),
        s2End(coreNum) {};
};

// 分核功能模块内部使用：记录切分信息
struct SplitInfo {
    std::vector<uint32_t> s1GBaseNum {};                   // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum {};                    // S2方向，切了多少个基本块
    std::vector<uint32_t> s1GTailSize {};                  // S1G方向，尾块size
    std::vector<uint32_t> s2TailSize {};                   // S2方向，尾块size
    bool isKvSeqAllZero { true };

    explicit SplitInfo(uint32_t batchSize) :
        s1GBaseNum(batchSize),
        s2BaseNum(batchSize),
        s1GTailSize(batchSize),
        s2TailSize(batchSize) {}
};

// 分核功能模块内部使用：记录batch的开销信息
struct CostInfo {
    std::vector<int64_t> bN2CostOfEachBatch {};           // 整个batch的开销
    std::vector<uint32_t> bN2BlockOfEachBatch {};          // 整个batch的开销
    std::vector<int64_t> bN2LastBlockCostOfEachBatch {};  // batch最后一块的开销
    uint32_t totalBlockNum { 0U };
    int64_t totalCost { 0 };

    explicit CostInfo(uint32_t batchSize) :
        bN2CostOfEachBatch(batchSize),
        bN2BlockOfEachBatch(batchSize),
        bN2LastBlockCostOfEachBatch(batchSize) {}
};

// 分核功能模块内部使用：分核过程中，case基本信息的上下文信息，组合以减少接口传参数量
struct SplitContext {
    SplitInfo splitInfo { 0U };
    CostInfo costInfo { 0U };

    explicit SplitContext(uint32_t batchSize) :
        splitInfo(batchSize),
        costInfo(batchSize) {}
};

// 分核功能模块内部使用：记录batch相关的临时信息
struct BatchCache {
    uint32_t bIdx { 0U };
    uint32_t s1Size { 0U };
    uint32_t s2Size { 0U };
    int64_t preTokenLeftUp { 0 };
    int64_t nextTokenLeftUp { 0 };
    BlockCost<int64_t> typeCost {};
};

// 分核功能模块内部使用：记录当前行（S1G）的临时信息
struct S1GCache {
    uint32_t bIdx { 0U };
    uint32_t s1GIdx { 0U };
    uint32_t s2Start { 0U };
    uint32_t s2End { 0U };
    int64_t s1GCost { 0 };
    int64_t s1GLastBlockCost { 0 };
    uint32_t s1GBlock { 0U };
    int64_t s1GNormalBlockCost { 0 };
};

// 分核功能模块内部使用：记录分配过程中，当前核的负载信息
struct CoreCache {
    int64_t costLimit { 0 };  // 负载上限
    int64_t cost { 0 };       // 已分配负载
    uint32_t block { 0U };      // 已分配块数
};

// 分核功能模块内部使用：记录分配过程中的上下文信息
struct AssignContext {
    uint32_t curBIdx { 0U };
    uint32_t curBN2Idx { 0U };
    uint32_t curS1GIdx { 0U };
    uint32_t curS2Idx { 0U };
    uint32_t curCoreIdx { 0U };
    int64_t unassignedCost { 0 };
    uint32_t usedCoreNum { 0U };
    uint32_t curKvSplitPart { 1U };

    int64_t bN2Cost { 0 };
    uint32_t bN2Block { 0U };
    bool isFinished { false };
    BatchCache batchCache {};
    S1GCache s1GCache {};
    CoreCache coreCache {};
};

class QuantSalsIndexerMetaDataCpuKernel : public CpuKernel {
public:
    QuantSalsIndexerMetaDataCpuKernel() = default;
    ~QuantSalsIndexerMetaDataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool ParamsCheck();
    bool ParamsInit();
    bool BalanceSchedule();
    bool GenMetaData();

  // util
    uint32_t GetS1SeqSize(uint32_t bIdx);
    uint32_t GetS2SeqSize(uint32_t bIdx);
    uint32_t GetSparseSeqSize(uint32_t bIdx);
    int64_t CalcPreTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    int64_t CalcNextTokenLeftUp(uint32_t s1Size, uint32_t s2Size);
    Range<uint32_t> CalcS2Range(uint32_t s1GIdx,const BatchCache &batchCache);
    int64_t CalcCost(uint32_t basicM, uint32_t basicS2);
    BlockCost<int64_t> CalcCostTable(uint32_t s1NormalSize, uint32_t s2NormalSize, uint32_t s1GTailSize,
        uint32_t s2TailSize);

    // cache calculation
    void CalcBatchCache(uint32_t bIdx, const SplitContext &splitContext, BatchCache &batchCache);
    void CalcS1GCache(uint32_t s1GIdx, const SplitContext &splitContext, const BatchCache &batchCache, S1GCache &s1GCache);
    void CopyTmpResult(SplitResult &tmpRes, SplitResult &splitRes);
    void ClearTmpResult(SplitResult &tmpRes);

    // preprocess
    void CalcSplitInfo(SplitContext &splitContext);
    void CalcBatchCost(uint32_t bIdx, const SplitContext &splitContext, CostInfo &costInfo);
    void CalcCostInfo(SplitContext &splitContext);

    // assign
    void UpdateCursor(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBatch(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByRow(const SplitContext &splitContext, AssignContext &assignContext);
    void AssignByBlock(const SplitContext &splitContext, AssignContext &assignContext);
    void ForceAssign(const SplitContext &splitContext, AssignContext &assignContext);

    // FD
    bool IsNeedRecordFDInfo(const AssignContext &assignContext, const SplitResult &splitRes);
    void RecordFDInfo(const SplitContext &splitContext, const AssignContext &assignContext, SplitResult &result);

    // main
    void SplitFD(SplitResult &result);
    void CalcSplitPlan(uint32_t coreNum, int64_t costLimit, const SplitContext &splitContext, SplitResult &result);
    void SplitCore();
    void RollBackCursor(const SplitContext &splitContext, const CostInfo &costInfo, SplitResult &splitRes);

private:
    CpuKernelContext* context_ = nullptr;

    Tensor *actSeqLenKV_ = nullptr;
    Tensor *metaData_ = nullptr;

    uint32_t coreNum_ = 24U; // new
    uint32_t aicCoreNum_ = 24U;
    uint32_t aivCoreNum_ = 48U;
    uint32_t batchSize_ = 0;
    uint32_t kvSeqSize_ = 0;
    uint32_t kvHeadNum_ = 0;
    uint32_t fixedTailCount_ = 0;
    uint32_t sparseBlockSize_ = 0;

    // SplitParam
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 0;
    uint32_t s2BaseSize_ = 0;
    SplitResult splitRes_ {24, 2};

private:
    enum class ParamId : uint32_t {
        // input
        actSeqLenKV = 0,
        // output
        metaData = 0,
    };
};
} // namespace aicpu
#endif
