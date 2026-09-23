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
 * \file sparse_flash_attention_antiquant_common.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_COMMON_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_COMMON_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
// 将isCheckTiling设置为false, 输入输出的max&sum&exp的shape为(m, 1)
constexpr SoftmaxConfig SFAA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};

enum class SFAA_LAYOUT {
    BSND = 0,
    TND = 1,
    PA_BSND = 2,
    PA_BNSD = 3,
    PA_NZ = 4,
};

enum class QUANT_MODE {
    PER_CHANNEL = 0,  // GQA支持
    PER_TOKEN_HEAD = 1, // GQA支持
    PER_TILE = 2,   // MLA支持
};

enum class ATTENTION_MODE {
    GQA_MHA = 0,  // QKV headDim相等
    MLA_NAIVE = 1, // Dn=128, Dr=64
    MLA_ABSORB = 2,   // Dn=512, Dr=64
};

enum class QUANT_SCALE_REPO_MODE {
    SEPARATE = 0,  // 分开存储
    COMBINE = 1, // 合并存储，量化模式是PER_TOKEN_HEAD/PER_TILE时支持COMBINE模式，参数顺序为：Nope+Rope+DequantScale
};

template <typename Q_T, typename KV_T, typename OUT_T, const bool FLASH_DECODE = false,
	  SFAA_LAYOUT LAYOUT_T = SFAA_LAYOUT::BSND, SFAA_LAYOUT KV_LAYOUT_T = SFAA_LAYOUT::BSND,
          const int TEMPLATE_MODE = C_TEMPLATE, const bool MSD_DD = false, typename... Args>
struct SFAAType {
    using queryType = Q_T;
    using kvType = KV_T;
    using kRopeType = Q_T;
    using outputType = OUT_T;
    static constexpr bool isMsdDD = MSD_DD;
    static constexpr bool flashDecode = FLASH_DECODE;
    static constexpr SFAA_LAYOUT layout = LAYOUT_T;
    static constexpr SFAA_LAYOUT kvLayout = KV_LAYOUT_T;
    static constexpr int templateMode = TEMPLATE_MODE;
    static constexpr bool pageAttention = (KV_LAYOUT_T == SFAA_LAYOUT::PA_BSND ||
                                           KV_LAYOUT_T == SFAA_LAYOUT::PA_BNSD ||
                                           KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ);
};

// ================================Util functions==================================
__aicore__ inline uint64_t SFAAAlign(uint64_t num, uint64_t rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd) * (rnd)));
}

__aicore__ inline uint64_t SfaaCeilDiv(uint64_t num, uint64_t rnd)
{
    return rnd == 0 ? 0 : (num + rnd-1) / rnd;
}

template <typename T1, typename T2> __aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T> __aicore__ inline size_t BlockAlign(size_t s)
{
    if constexpr (IsSameType<T, int4b_t>::value) {
        return (s + 63) / 64 * 64;
    }
    size_t n = (32 / sizeof(T));
    return (s + n - 1) / n * n;
}

struct FDparams {
    uint32_t *bN2IdxOfFdHead;
    uint32_t *gS1IdxOfFdHead;
    uint32_t *s2SplitNumOfFdHead;
    uint32_t *gS1SplitNumOfFdHead;
    uint32_t *gS1LastPartSizeOfFdHead;
    uint32_t *gS1IdxEndOfFdHead;
    uint32_t *gS1IdxEndOfFdHeadSplit;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
};

struct RunInfo {
    uint32_t loop;
    uint32_t bIdx;
    uint32_t n2Idx;
    uint32_t gIdx;
    uint32_t s1Idx;
    uint32_t s2Idx;
    uint32_t bN2Idx;
    uint32_t curSInnerLoopTimes;
    uint64_t tndBIdxOffsetQ;
    uint64_t tndBIdxOffsetKV;
    uint64_t tensorAOffset;
    uint64_t tensorBOffset;
    uint64_t tensorARopeOffset;
    uint64_t tensorBRopeOffset;
    uint64_t attenOutOffset;
    uint64_t attenMaskOffset;
    uint64_t topKBaseOffset;
    uint64_t curS2BaseOffset;
    uint64_t nextS2BaseOffset;
    uint32_t actualSingleProcessSInnerSize;
    uint32_t actualSingleProcessSInnerSizeAlign;
    bool isFirstSInnerLoop;
    bool isChangeBatch;
    uint32_t s2BatchOffset;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t mSize;
    uint32_t mSizeV;
    uint32_t aicS2AccessSize;
    uint32_t aivS2AccessSize;
    uint32_t mSizeVStart;
    uint32_t tndIsS2SplitCore;
    uint32_t tndCoreStartKVSplitPos;
    bool isBmm2Output;
    bool isValid = false;

    uint64_t actS1Size = 1;
    uint64_t curActualSeqLenOri = 0ULL;

    uint32_t gS1Idx;
    uint64_t actS2Size = 1;
    uint32_t actMBaseSize;
    bool isLastS2Loop;
    int32_t nextTokensPerBatch = 0;
    int64_t threshold;
    uint32_t sparseBlockCount;
    int64_t topkGmBaseOffset = 0;
    uint32_t maskStart = 0;
    uint32_t maskEnd = 0;
};

struct ConstInfo {
    // CUBE与VEC核间同步的模式
    static constexpr uint32_t SFAA_SYNC_MODE2 = 2;
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
    // FP32的0值和极大值
    static constexpr float FLOAT_ZERO = 0;
    static constexpr float FLOAT_MAX = 3.402823466e+38F;

    // preLoad的总次数
    uint32_t preLoadNum = 0U;
    uint32_t nBufferMBaseSize = 0U;
    // CUBE和VEC的核间同步EventID
    uint32_t syncV1NupdateC2 = 0U;
    uint32_t syncV0C1 = 0U;
    uint32_t syncC1V1 = 0U;
    uint32_t syncV1C2 = 0U;
    uint32_t syncC2V2 = 0U;
    uint32_t syncC2V1 = 0U;

    uint32_t mmResUbSize = 0U;   // Matmul1输出结果GM上的大小
    uint32_t vec1ResUbSize = 0U; // Vector1输出结果GM上的大小
    uint32_t bmm2ResUbSize = 0U; // Matmul2输出结果GM上的大小
    uint64_t batchSize = 0ULL;
    uint64_t gSize = 0ULL;
    uint64_t qHeadNum = 0ULL;
    uint64_t kvHeadNum;
    uint64_t headDim;
    uint64_t headDimRope;
    uint64_t headDimAlign;
    uint64_t combineHeadDim; // quantScaleRepoMode为Combine模式时=headDim+headDimRope, 否则=headDim
    uint64_t kvSeqSize = 0ULL;        // kv最大S长度
    uint64_t qSeqSize = 1ULL;         // q最大S长度
    int64_t kvCacheBlockSize = 0;    // PA场景的block size
    uint32_t maxBlockNumPerBatch = 0; // PA场景的最大单batch block number
    uint32_t splitKVNum = 0U;         // S2核间切分的切分份数
    SFAA_LAYOUT outputLayout;          // 输出的Transpose格式
    uint32_t sparseMode = 0;
    bool needInit = false;

    // FlashDecoding
    uint32_t actualCombineLoopSize = 0U; // FlashDecoding场景, S2在核间切分的最大份数
    uint64_t combineLseOffset = 0ULL;
    uint64_t combineAccumOutOffset = 0ULL;

    uint32_t actualLenDimsQ = 0U;   // query的actualSeqLength 的维度
    uint32_t actualLenDimsKV = 0U;  // KV 的actualSeqLength 的维度
    uint32_t sparseLenDimsKV = 0U;  // TopK的实际K大小的维度

    // TND
    uint32_t s2Start = 0U; // TND场景下，S2的起始位置
    uint32_t s2End = 0U;   // 单核TND场景下S2循环index上限

    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;

    uint32_t tndFDCoreArrLen = 0U;     // TNDFlashDecoding相关分核信息array的长度
    uint32_t coreStartKVSplitPos = 0U; // TNDFlashDecoding kv起始位置

    uint32_t mBaseSize = 1ULL;
    uint32_t s2BaseSize = 1ULL;

    // sparse attr
    int64_t sparseBlockSize = 0;
    uint32_t sparseBlockCount = 0;
    uint32_t sparseShardSize = 0; // 多少个S1共用同一组sparse_indices

    // attention模式与量化模式
    ATTENTION_MODE attentionMode = ATTENTION_MODE::MLA_ABSORB;
    QUANT_MODE keyQuantMode = QUANT_MODE::PER_TILE;
    QUANT_MODE valueQuantMode = QUANT_MODE::PER_TILE;
    QUANT_SCALE_REPO_MODE quantScaleRepoMode = QUANT_SCALE_REPO_MODE::COMBINE;
    uint64_t tileSize = 128ULL;
};

struct MSplitInfo {
    uint32_t nBufferIdx = 0U;
    uint32_t nBufferStartM = 0U;
    uint32_t nBufferDealM = 0U;
    uint32_t vecStartM = 0U;
    uint32_t vecDealM = 0U;
};


// BLOCK和REPEAT的字节数
constexpr uint64_t BYTE_BLOCK = 32UL;
constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
// BLOCK和REPEAT的FP32元素数
constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
// repeat stride不能超过256
constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;
constexpr int64_t HALF_NUM = 2;
constexpr int64_t STRIDE_LENGTH = 8;
constexpr int64_t MAX_VALID_LENGTH = 1024;

template <typename T>
__aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

__aicore__ inline void MatDivsVec(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 1;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 0;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsDiv);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsDiv);
    }
}

__aicore__ inline void RowSub(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsSub;
    repeatParamsSub.src0BlkStride = 1;
    repeatParamsSub.src1BlkStride = 1;
    repeatParamsSub.dstBlkStride = 1;
    repeatParamsSub.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsSub.src1RepStride = 0;
    repeatParamsSub.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Sub(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsSub);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Sub(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsSub);
    }
}

__aicore__ inline void ColMax(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = 0;
    repeatParamsMax.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Max(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsMax);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Max(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsMax);
    }
}

__aicore__ inline void ColAdd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsAdd;
    repeatParamsAdd.src0BlkStride = 1;
    repeatParamsAdd.src1BlkStride = 1;
    repeatParamsAdd.dstBlkStride = 1;
    repeatParamsAdd.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsAdd.src1RepStride = 0;
    repeatParamsAdd.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsAdd);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Add(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount, repeatParamsAdd);
    }
}

#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_COMMON_H
