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
 * \file quant_sals_indexer_common.h
 * \brief
 */
#ifndef QUANT_SALS_INDEXER_COMMON_H
#define QUANT_SALS_INDEXER_COMMON_H

namespace QSICommon {
constexpr uint32_t MAX_TOPK = 2048;
constexpr uint32_t LD_PARAM_NUM = 16;

// 与tiling的layout保持一致
enum class QSI_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BNSD = 2,
    PA_BSND = 3,
    PA_NZ = 4
};

template <typename Q_T, typename K_T, typename OUT_T, const bool PAGE_ATTENTION = false,
        QSI_LAYOUT K_LAYOUT_T = QSI_LAYOUT::PA_BNSD, typename... Args>
struct QSIType {
    using queryType = Q_T;
    using keyType = K_T;
    using outputType = OUT_T;
    static constexpr bool pageAttention = PAGE_ATTENTION;
    static constexpr QSI_LAYOUT keyLayout = K_LAYOUT_T;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bN2Idx;
    uint32_t bIdx;
    uint32_t n2Idx = 0;
    uint32_t gS1Idx;
    uint32_t s2Idx;

    uint32_t actS1Size = 1;
    uint32_t actS2Size = 1;
    int32_t needProcessS2Size = 0;
    int32_t targetTopKAlign = 0L;
    int32_t targetTopK = 0L;
    int32_t fixedTailCount = 0L;
    uint32_t actMBaseSize;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;

    uint64_t tensorQueryOffset;
    uint64_t tensorKeyOffset;
    uint64_t tensorKeyScaleOffset;
    uint64_t indiceOutOffset;

    float qScale = 0.0f;

    bool isFirstS2InnerLoop;
    bool isLastS2InnerLoop;
    bool isAllLoopEnd = false;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t FIA_SYNC_MODE2 = 2;
    // BUFFER的字节数
    static constexpr uint32_t BUFFER_SIZE_BYTE_32B = 32;
    static constexpr uint32_t BUFFER_SIZE_BYTE_64B = 64;
    static constexpr uint32_t BUFFER_SIZE_BYTE_256B = 256;
    static constexpr uint32_t BUFFER_SIZE_BYTE_512B = 512;
    static constexpr uint32_t BUFFER_SIZE_BYTE_1K = 1024;
    static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2048;
    static constexpr uint32_t BUFFER_SIZE_BYTE_4K = 4096;
    static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8192;
    static constexpr uint32_t BUFFER_SIZE_BYTE_16K = 16384;
    static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32768;
    // 无效索引
    static constexpr int INVALID_IDX = -1;

    // CUBE和VEC的核间同步EventID
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C1 = 0U;

    // 基本块大小
    uint32_t mBaseSize = 1ULL;
    uint32_t s1BaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    uint64_t batchSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kHeadNum;
    uint64_t headDim;
    uint64_t sparseCount;             // topK选取大小
    uint64_t kSeqSize = 0ULL;         // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    uint32_t kCacheBlockSize = 0;     // PA场景的block size
    int32_t sparseBlockSize = 0;
    float sparseRatio = 0.0f;
    int32_t fixedTailCount = 1UL;
    int32_t maxSeqlenKey = 0ULL;
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    QSI_LAYOUT outputLayout;           // 输出的格式
    bool attenMaskFlag = false;

    uint32_t actualLenQDims = 0U; // query的actualSeqLength 的维度
    uint32_t actualLenDims = 0U;  // KV 的actualSeqLength 的维度
    bool isAccumSeqS1 = false;    // 是否累加模式
    bool isAccumSeqS2 = false;    // 是否累加模式
};

struct SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false;     // 当前核是否需要进行Decode归约任务
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

__aicore__ inline uint64_t QSiCeilDiv(uint64_t num, uint64_t rnd)
{
    return rnd == 0 ? 0 : (num + rnd-1) / rnd;
}

template <typename T>
__aicore__ inline T QSiCeilAlign(T num, T rnd)
{
    return rnd == 0 ? 0 : (num + rnd-1) / rnd * rnd;
}
} // namespace QSICommon

#endif // QUANT_SALS_INDEXER_COMMON_H