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
 * \file sparse_flash_attention_antiquant_kernel_gqa.h
 * \brief
 */

#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_KERNEL_GQA_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_KERNEL_GQA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_antiquant_common.h"
#include "sparse_flash_attention_antiquant_service_cube_gqa_msd.h"
#include "sparse_flash_attention_antiquant_service_vector_gqa_msd.h"
#include "sparse_flash_attention_antiquant_service_flashdecode.h"
#include "sparse_flash_attention_antiquant_metadata.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using namespace optiling::detail;

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define SALSK_S2BASEIZE 1024
#define SALSK_S2BASEIZE_1 1023

// 由于S2循环前，RunInfo还没有赋值，使用Bngs1Param临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfoGqa {
    uint32_t bN2Idx = 0U;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint64_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
    uint32_t s2LoopTimes = 0U; // S2方向循环的总次数，无论TND还是BXXD都是等于实际次数，不用减1
    uint64_t curActualSeqLen = 0ULL;
    uint64_t curActualSeqLenOri = 0ULL;
    bool curActSeqLenIsZero = false;
    int32_t nextTokensPerBatch = 0;

    uint64_t actS1Size = 1ULL; // TND场景下当前Batch循环处理的S1轴的大小，非TND场景下不要用这个字段
    uint64_t sparseBlockCount = 1Ull; // TopK的K大小
    uint32_t tndCoreStartKVSplitPos;
    bool tndIsS2SplitCore;
    int32_t threshold = 0;

    uint32_t gS1Idx = 0U;
    uint64_t mBasicSizeTail = 0U; // gS1方向循环的尾基本块大小
    uint32_t sparseBlockActualSeqSizeKv = 0U;
};

template <typename SFAAT> class SparseFlashAttentionAntiquantGqa {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using Q_T = typename SFAAT::queryType;
    using KV_T = typename SFAAT::kvType;
    using OUT_T = typename SFAAT::outputType;
    using Q_ROPE_T = Q_T;
    using K_ROPE_T = typename AscendC::Conditional<SFAAT::isMsdDD, KV_T, half>::type;
    using UPDATE_T = T;
    using MM1_OUT_T = typename AscendC::Conditional<SFAAT::isMsdDD, half, T>::type;
    using MM2_OUT_T = typename AscendC::Conditional<SFAAT::isMsdDD, half, T>::type;

    __aicore__ inline SparseFlashAttentionAntiquantGqa(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale,
                                __gm__ uint8_t* valueScale, __gm__ uint8_t *blockTable,
                                __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *sparseSeqLengthsKv, SfaMetaData *metaData,
                                __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
                                const SparseFlashAttentionAntiquantTilingDataMla *__restrict tiling,
				                __gm__ uint8_t *gmTiling, TPipe *tPipe);

    __aicore__ inline void Process();

private:
    static constexpr bool PAGE_ATTENTION = SFAAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAAT::templateMode;
    static constexpr SFAA_LAYOUT LAYOUT_T = SFAAT::layout;
    static constexpr SFAA_LAYOUT KV_LAYOUT_T = SFAAT::kvLayout;
    bool FLASH_DECODE = false;
    static constexpr bool IS_META = SFAAT::flashDecode;

    static constexpr int64_t fdPrefetchLen = 2;
    static constexpr uint32_t PRELOAD_NUM = 2;
    static constexpr uint32_t N_BUFFER_M_BASIC_SIZE = 256;
    static constexpr uint32_t SFAA_PRELOAD_TASK_CACHE_SIZE = 3;
    static constexpr uint32_t SFAA_KVPREMERGE_CACHE_SIZE = 2;

    static constexpr uint32_t SYNC_V0_C1_FLAG = 6;
    static constexpr uint32_t SYNC_C1_V1_FLAG = 7;
    static constexpr uint32_t SYNC_V1_C2_FLAG = 8;
    static constexpr uint32_t SYNC_C2_V2_FLAG = 9;
    static constexpr uint32_t SYNC_C2_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_NUPDATE_C2_FLAG = 5;

    static constexpr uint64_t SYNC_MM2RES_BUF1_FLAG = 10;
    static constexpr uint64_t SYNC_MM2RES_BUF2_FLAG = 11;
    static constexpr uint64_t SYNC_FDOUTPUT_BUF_FLAG = 12;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = SFAAVectorServiceGqa<SFAAT>::BYTE_BLOCK / sizeof(T);

    static constexpr uint32_t dbWorkspaceRatio = PRELOAD_NUM;

    const SparseFlashAttentionAntiquantTilingDataMla *__restrict tilingData = nullptr;

    TPipe *pipe = nullptr;

    uint64_t mSizeVStart = 0ULL;
    int64_t threshold = 0;
    uint64_t s2BatchBaseOffset = 0;
    uint64_t tensorACoreOffset = 0ULL;
    uint64_t tensorARopeCoreOffset = 0ULL;
    uint64_t tensorBCoreOffset = 0ULL;
    int64_t topkGmBaseOffset = 0;

    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    __gm__ uint8_t *keyPtr = nullptr;
    __gm__ uint8_t *valuePtr = nullptr;
    SfaMetaData *metaDataPtr = nullptr;

    ConstInfo constInfo{};
    TempLoopInfoGqa tempLoopInfo{};

    SFAAMatmulServiceGqaMsd<SFAAT> matmulServiceMsd;
    SFAAVectorServiceGqaMsd<SFAAT> vectorServiceMsd;
    SFAAFlashDecodeServiceGqa<SFAAT> fdService;

    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<T> keyDequantScaleGm;
    GlobalTensor<T> valueDequantScaleGm;

    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<int32_t> sparseSeqLengthsKVGm;

    // workspace
    GlobalTensor<KV_T> queryPreProcessResGm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<K_ROPE_T> vec1ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<K_ROPE_T> keyMergeGm_;
    GlobalTensor<K_ROPE_T> valueMergeGm_;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<T> vec2ResGm;

    GlobalTensor<T> accumOutGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    template <typename T>
    __aicore__ inline T Align(T num, T rnd)
    {
        return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
    }

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData();
    __aicore__ inline void InitMetaData();
    __aicore__ inline void InitCalcParamsEach();
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
                                            __gm__ uint8_t *sparseSeqLengthsKv);
    __aicore__ inline void InitOutputSingleCore();
    // ================================Process functions================================
    __aicore__ inline void ProcessBalance();
    __aicore__ inline void PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                           RunInfo extraInfo[SFAA_PRELOAD_TASK_CACHE_SIZE]);
    __aicore__ inline void FlashDecode();
    // ================================Offset Calc=====================================
    __aicore__ inline void GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx = 0);
    __aicore__ inline void GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
    __aicore__ inline void UpdateInnerLoopCond();
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1StartIdx, uint32_t s1Size, uint32_t n2Idx);
    __aicore__ inline void CalcParams(uint32_t loop, uint64_t s2Start, uint32_t s2LoopIdx, RunInfo &info);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t gS1EndPrev, uint32_t s2EndPrev);
    __aicore__ inline uint64_t GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths, uint32_t bIdx);
    __aicore__ inline uint32_t GetActualSeqLenKV(uint32_t bIdx);
    __aicore__ inline void GetSparseBlockCountAndActualSeqKv(uint32_t bIdx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx, uint32_t &n2Idx);
    __aicore__ inline void UpdateInner(uint32_t &s2End, uint32_t &curS2End, uint32_t s1Idx, bool isEnd);
    __aicore__ inline void GetPreNextTokensLeftUp();
    // ================================Mm1==============================================
    __aicore__ inline void ComputeMm1(const RunInfo &info);
    // ================================Mm2==============================================
    __aicore__ inline void ComputeMm2(const RunInfo &info);
    __aicore__ inline void Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb, uint32_t startRow,
                                           uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx, uint32_t n2Idx);
};

template <typename SFAAT> __aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::InitTilingData()
{
    usedCoreNum = tilingData->singleCoreParams.usedCoreNum;
    constInfo.splitKVNum = tilingData->splitKVParams.s2;
    constInfo.mmResUbSize = tilingData->singleCoreTensorSize.mmResUbSize;
    constInfo.bmm2ResUbSize = tilingData->singleCoreTensorSize.bmm2ResUbSize;
    constInfo.vec1ResUbSize = constInfo.mmResUbSize;

    constInfo.batchSize = tilingData->baseParams.batchSize;
    constInfo.gSize = tilingData->baseParams.nNumOfQInOneGroup;
    constInfo.kvHeadNum = tilingData->baseParams.kvHeadNum;
    constInfo.qHeadNum = constInfo.gSize * tilingData->baseParams.kvHeadNum;
    constInfo.kvSeqSize = tilingData->baseParams.seqSize;
    constInfo.qSeqSize = tilingData->baseParams.qSeqSize;
    constInfo.maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    constInfo.kvCacheBlockSize = tilingData->baseParams.blockSize;
    constInfo.outputLayout = static_cast<SFAA_LAYOUT>(tilingData->baseParams.outputLayout);

    constInfo.sparseBlockSize = tilingData->baseParams.sparseBlockSize;
    constInfo.sparseBlockCount = tilingData->baseParams.sparseBlockCount;
    constInfo.sparseShardSize = tilingData->baseParams.sparseShardSize;
    constInfo.sparseMode = tilingData->baseParams.sparseMode;
    constInfo.attentionMode = static_cast<ATTENTION_MODE>(tilingData->baseParams.attentionMode);
    constInfo.keyQuantMode = static_cast<QUANT_MODE>(tilingData->baseParams.keyQuantMode);
    constInfo.valueQuantMode = static_cast<QUANT_MODE>(tilingData->baseParams.valueQuantMode);
    constInfo.quantScaleRepoMode = static_cast<QUANT_SCALE_REPO_MODE>(tilingData->baseParams.quantScaleRepoMode);
    constInfo.combineHeadDim = tilingData->baseParams.qkHeadDim;
    constInfo.headDimRope = tilingData->baseParams.ropeHeadDim;
    constInfo.headDim = (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::COMBINE) ? 
                        constInfo.combineHeadDim - constInfo.headDimRope : constInfo.combineHeadDim;
    constInfo.headDimAlign = Align(constInfo.headDim, (uint64_t)BYTE_BLOCK);

    constInfo.s2BaseSize = SALSK_S2BASEIZE;
    constInfo.mBaseSize = constInfo.gSize * constInfo.sparseShardSize; 

    constInfo.preLoadNum = PRELOAD_NUM;
    constInfo.nBufferMBaseSize = N_BUFFER_M_BASIC_SIZE; // 256
    constInfo.syncV0C1 = SYNC_V0_C1_FLAG;
    constInfo.syncC1V1 = SYNC_C1_V1_FLAG;
    constInfo.syncV1C2 = SYNC_V1_C2_FLAG;
    constInfo.syncC2V2 = SYNC_C2_V2_FLAG;
    // constInfo.syncC2V1 = SYNC_C2_V1_FLAG;
    constInfo.syncV1NupdateC2 = SYNC_V1_NUPDATE_C2_FLAG;
}

template <typename SFAAT> __aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::InitMetaData()
{
    FLASH_DECODE = metaDataPtr->numOfFdHead > 0U;
    usedCoreNum = metaDataPtr->usedCoreNum;
    constInfo.s2BaseSize = SALSK_S2BASEIZE;
    constInfo.mBaseSize = metaDataPtr->mBaseSize; 
}

template <typename SFAAT> __aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::InitBuffers()
{
    if constexpr (SFAAT::isMsdDD) {
        if ASCEND_IS_AIV {
            vectorServiceMsd.InitBuffers(pipe);
        } else {
            matmulServiceMsd.InitBuffers(pipe);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void
SparseFlashAttentionAntiquantGqa<SFAAT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengthsQ,
                                                          __gm__ uint8_t *actualSeqLengths,
                                                          __gm__ uint8_t *sparseSeqLengthsKv)
{
    constInfo.actualLenDimsQ = tilingData->baseParams.actualLenDimsQ;
    constInfo.actualLenDimsKV = tilingData->baseParams.actualLenDimsKV;
    constInfo.sparseLenDimsKV = tilingData->baseParams.sparseLenDimsKV;
    if (constInfo.actualLenDimsKV != 0) {
        actualSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengths, constInfo.actualLenDimsKV);
    }
    if (constInfo.actualLenDimsQ != 0) {
        actualSeqLengthsQGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQ, constInfo.actualLenDimsQ);
    }
    if (constInfo.sparseLenDimsKV != 0) {
        sparseSeqLengthsKVGm.SetGlobalBuffer((__gm__ int32_t *)sparseSeqLengthsKv, constInfo.sparseLenDimsKV);
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::InitAllZeroOutput(uint32_t bIdx, uint32_t s1Idx,
                                                                                  uint32_t n2Idx)
{
    if (constInfo.outputLayout == SFAA_LAYOUT::TND) {
        uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsQGm.GetValue(bIdx - 1);
        uint32_t s1Count = tempLoopInfo.actS1Size;

        uint64_t attenOutOffset = (tBase + s1Idx) * constInfo.kvHeadNum * constInfo.gSize * constInfo.headDim +   // T轴、s1轴偏移
                                    n2Idx * constInfo.gSize * constInfo.headDim;                        // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * constInfo.headDim, 0);
    } else if (constInfo.outputLayout == SFAA_LAYOUT::BSND) {
        uint64_t attenOutOffset = bIdx * constInfo.qSeqSize * constInfo.kvHeadNum  * constInfo.gSize * constInfo.headDim +
                                    s1Idx * constInfo.kvHeadNum  * constInfo.gSize * constInfo.headDim + // B轴、S1轴偏移
                                    n2Idx * constInfo.gSize * constInfo.headDim;              // N2轴偏移
        matmul::InitOutput<OUT_T>(attentionOutGm[attenOutOffset], constInfo.gSize * constInfo.headDim, 0);
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::InitOutputSingleCore()
{
    uint32_t coreNum = GetBlockNum();
    if (coreNum != 0) {
        uint64_t totalOutputSize = constInfo.batchSize * constInfo.qHeadNum * constInfo.qSeqSize * constInfo.headDim;
        uint64_t singleCoreSize = (totalOutputSize + (2 * coreNum) - 1) / (2 * coreNum);  // 2 means c:v = 1:2
        uint64_t tailSize = totalOutputSize - tmpBlockIdx * singleCoreSize;
        uint64_t singleInitOutputSize = tailSize < singleCoreSize ? tailSize : singleCoreSize;
        if (singleInitOutputSize > 0) {
            matmul::InitOutput<OUT_T>(attentionOutGm[tmpBlockIdx * singleCoreSize], singleInitOutputSize, 0);
        }
        SyncAll();
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::GetActualSeqLen(uint32_t bIdx, uint32_t s1Idx)
{
    tempLoopInfo.curActualSeqLenOri = GetActualSeqLenKV(bIdx);
    tempLoopInfo.actS1Size = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bIdx);
    GetSparseBlockCountAndActualSeqKv(bIdx);
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::GetSparseActualSeqLen(uint32_t bIdx, uint32_t s1Idx,
                                                                                      uint32_t n2Idx)
{
    if (tempLoopInfo.nextTokensPerBatch < 0 && s1Idx < (-tempLoopInfo.nextTokensPerBatch)) { // 存在行无效
        tempLoopInfo.curActualSeqLen = 0;
        return;
    }
    tempLoopInfo.threshold = tempLoopInfo.curActualSeqLenOri;
    if (constInfo.sparseMode == 3) {
        tempLoopInfo.threshold = static_cast<int64_t>(tempLoopInfo.nextTokensPerBatch) + s1Idx + 1;
    }
    tempLoopInfo.curActualSeqLen = (tempLoopInfo.sparseBlockActualSeqSizeKv > tempLoopInfo.threshold) ?
                                tempLoopInfo.threshold : tempLoopInfo.sparseBlockActualSeqSizeKv;
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::GetSparseBlockCountAndActualSeqKv(uint32_t bIdx)
{
    if (constInfo.sparseLenDimsKV == 0) {
        tempLoopInfo.sparseBlockCount = constInfo.sparseBlockCount;
        tempLoopInfo.sparseBlockActualSeqSizeKv = constInfo.sparseBlockCount * constInfo.sparseBlockSize;
    } else {
        tempLoopInfo.sparseBlockActualSeqSizeKv = sparseSeqLengthsKVGm.GetValue(bIdx);
        tempLoopInfo.sparseBlockCount = (tempLoopInfo.sparseBlockActualSeqSizeKv + constInfo.sparseBlockSize - 1) / constInfo.sparseBlockSize;
    }
}

template <typename SFAAT>
__aicore__ inline uint32_t SparseFlashAttentionAntiquantGqa<SFAAT>::GetActualSeqLenKV(uint32_t bIdx)
{
    if (constInfo.actualLenDimsKV == 0) {
        return constInfo.kvSeqSize;
    } else if (constInfo.actualLenDimsKV == 1) {
        return actualSeqLengthsKVGm.GetValue(0);
    } else {
        return actualSeqLengthsKVGm.GetValue(bIdx);
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t s1StartIdx,
                                                                                    uint32_t s1Size, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        for (uint32_t i = 0; i < s1Size; i++) {
            InitAllZeroOutput(bIdx, s1StartIdx + i, n2Idx);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::GetPreNextTokensLeftUp()
{
    if (constInfo.sparseMode == 3) {
        tempLoopInfo.nextTokensPerBatch =
            static_cast<int32_t>(tempLoopInfo.curActualSeqLenOri) - static_cast<int32_t>(tempLoopInfo.actS1Size);
    }
}

template <typename SFAAT> __aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::UpdateInnerLoopCond()
{
    if ((tempLoopInfo.curActualSeqLen == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.curActualSeqLen & (SALSK_S2BASEIZE_1);
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? SALSK_S2BASEIZE : tempLoopInfo.s2BasicSizeTail;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;
    tempLoopInfo.s2LoopTimes = 0;
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::UpdateInner(uint32_t &s2End, uint32_t &curS2End,
                                                                            uint32_t s1Idx, bool isEnd)
{
    uint32_t s1BaseSize = 1;
    int64_t s1Offset = s1BaseSize * s1Idx;
    int64_t s2LastToken = Min(s1Offset + tempLoopInfo.nextTokensPerBatch + s1BaseSize, tempLoopInfo.curActualSeqLenOri);
    s2LastToken = Min(constInfo.sparseBlockSize * tempLoopInfo.sparseBlockCount, s2LastToken);
    curS2End = (s2LastToken + SALSK_S2BASEIZE - 1) / SALSK_S2BASEIZE;
    tempLoopInfo.s2LoopTimes = isEnd ? constInfo.s2End + 1 : curS2End;
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::Init(__gm__ uint8_t *query,
    __gm__ uint8_t *key, __gm__ uint8_t *value, __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale,
    __gm__ uint8_t* valueScale, __gm__ uint8_t *blockTable, __gm__ uint8_t *actualSeqLengthsQ, 
    __gm__ uint8_t *actualSeqLengths, __gm__ uint8_t *sparseSeqLengthsKv, SfaMetaData *metaData, 
    __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace,
    const SparseFlashAttentionAntiquantTilingDataMla *__restrict tiling,
    __gm__ uint8_t *gmTiling, TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // init tiling data
    tilingData = tiling;
    InitTilingData();
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths, sparseSeqLengthsKv);

    // 初始化计算参数
    if constexpr (IS_META) {
        if (metaData != nullptr) {
            metaDataPtr = metaData;
            InitMetaData();
        }
    }
    InitCalcParamsEach();
    pipe = tPipe;
    keyPtr = key;
    valuePtr = value;

    // init global buffer
    queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    keyGm.SetGlobalBuffer((__gm__ KV_T *)keyPtr);
    valueGm.SetGlobalBuffer((__gm__ KV_T *)valuePtr);
    keyDequantScaleGm.SetGlobalBuffer((__gm__ T *)keyScale);
    valueDequantScaleGm.SetGlobalBuffer((__gm__ T *)valueScale);
    keyGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    valueGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    attentionOutGm.SetGlobalBuffer((__gm__ OUT_T *)attentionOut);

    if ASCEND_IS_AIV {
        if (constInfo.needInit && LAYOUT_T != SFAA_LAYOUT::TND) {
            InitOutputSingleCore();
        }
    }

    if constexpr (PAGE_ATTENTION) {
        blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    }
    topKGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);

    // workspace 内存排布
    // |Q--|mm1ResGm(存S)|vec1ResGm(存A1,A2)|mm2ResGm(存O)|vec2ResGm
    // |Core0_Q1-Core0_Q2-Core1_Q1-Core1_Q2....Core32_Q1-Core32_Q2|Core0_mmRes
    uint64_t offset = 0;
    queryPreProcessResGm.SetGlobalBuffer(
        (__gm__ KV_T *)(workspace + offset +
                        aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * 2 * sizeof(KV_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * 2 * sizeof(KV_T);

    mm1ResGm.SetGlobalBuffer(
        (__gm__ MM1_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(MM1_OUT_T);

    vec1ResGm.SetGlobalBuffer(
        (__gm__ K_ROPE_T *)(workspace + offset + aiCoreIdx * dbWorkspaceRatio * constInfo.mmResUbSize * 2 *
                            sizeof(K_ROPE_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.mmResUbSize * sizeof(K_ROPE_T) * 2;

    mm2ResGm.SetGlobalBuffer(
        (__gm__ MM2_OUT_T *)(workspace + offset +
                             aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);
    mm2ResInt32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(mm2ResGm.GetPhyAddr(0)));

    vec2ResGm.SetGlobalBuffer((__gm__ T *)(workspace + offset +
                              aiCoreIdx * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(T)));
    offset += GetBlockNum() * dbWorkspaceRatio * constInfo.bmm2ResUbSize * sizeof(MM2_OUT_T);

    // s2  d bufNum
    // 384 is mergesize
    // 128 is headdim
    // 4 is buffernum
    keyMergeGm_.SetGlobalBuffer((__gm__ K_ROPE_T *)(workspace + offset + aiCoreIdx * 384 * 128 * 4 *
                                sizeof(K_ROPE_T)));
    offset += GetBlockNum() * 384 * 128 * 4 * sizeof(K_ROPE_T);
    valueMergeGm_.SetGlobalBuffer((__gm__ K_ROPE_T *)(workspace + offset + aiCoreIdx * 384 * 128 * 4 *
                        sizeof(K_ROPE_T)));
    offset += GetBlockNum() * 384 * 128 * 4 * sizeof(K_ROPE_T);

    if constexpr (IS_META) {
        if (FLASH_DECODE) {
            accumOutGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
            offset = offset + tilingData->splitKVParams.accumOutSize * sizeof(float);
            lseSumFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
            lseMaxFdGm.SetGlobalBuffer((__gm__ float *)(workspace + offset) + tilingData->splitKVParams.logSumExpSize / 2);
            offset = offset + tilingData->splitKVParams.logSumExpSize * sizeof(float);
        }
    }

    if ASCEND_IS_AIV {        
        if constexpr (SFAAT::isMsdDD) {
            vectorServiceMsd.InitParams(constInfo, tilingData, metaDataPtr);
            vectorServiceMsd.InitVec0GlobalTensor(keyMergeGm_, valueMergeGm_, queryPreProcessResGm, queryGm,
                                           keyGm, valueGm, blockTableGm, keyDequantScaleGm);
            vectorServiceMsd.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, actualSeqLengthsQGm,
                                            actualSeqLengthsKVGm, lseMaxFdGm, lseSumFdGm, topKGm);
            vectorServiceMsd.InitVec2GlobalTensor(valueDequantScaleGm, accumOutGm, mm2ResGm, attentionOutGm);
        }
        if constexpr (IS_META) {
            if (FLASH_DECODE) {
                fdService.InitParams(constInfo);
                fdService.InitGlobalTensor(lseMaxFdGm, lseSumFdGm, accumOutGm, attentionOutGm, 
                                        actualSeqLengthsQGm, sparseSeqLengthsKVGm);
            }
        }
    }

    if ASCEND_IS_AIC {
        if constexpr (SFAAT::isMsdDD) {
            matmulServiceMsd.InitParams(constInfo);
            matmulServiceMsd.InitMm1GlobalTensor(queryGm, keyGm, mm1ResGm);
            matmulServiceMsd.InitMm2GlobalTensor(vec1ResGm, valueGm, mm2ResGm, attentionOutGm);
            matmulServiceMsd.InitPageAttentionInfo(keyMergeGm_, valueMergeGm_, queryPreProcessResGm, blockTableGm, topKGm,
                                                constInfo.kvCacheBlockSize, constInfo.maxBlockNumPerBatch);
        }
    }
    // 要在InitParams之后执行
    if (pipe != nullptr) {
        InitBuffers();
    }
}

template <typename SFAAT> __aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::InitCalcParamsEach()
{
    if constexpr (IS_META) {
        const uint32_t *bN2End = metaDataPtr->bN2End;
        const uint32_t *gS1End = metaDataPtr->gS1End;
        const uint32_t *s2End = metaDataPtr->s2End;
        const uint32_t *s2SplitStartIdxOfCore = metaDataPtr->fdRes.s2SplitStartIdxOfCore;
        constInfo.bN2End = bN2End[aiCoreIdx];
        constInfo.gS1End = gS1End[aiCoreIdx];
        constInfo.s2End = s2End[aiCoreIdx];
        if (aiCoreIdx != 0) {
            GetAxisStartIdx(bN2End[aiCoreIdx - 1], gS1End[aiCoreIdx - 1], s2End[aiCoreIdx - 1]);
        }
        constInfo.coreStartKVSplitPos = s2SplitStartIdxOfCore[aiCoreIdx];
    } else {
#ifdef ASCENDC_CPU_DEBUG
        const uint32_t *bN2End = tilingData->outerSplitParams.bN2End;
        const uint32_t *gS1End = tilingData->outerSplitParams.gS1End;
        const uint32_t *s2End = tilingData->outerSplitParams.s2End;
#else
        uint32_t bN2End[ARRAY_SIZE(tilingData->outerSplitParams.bN2End)];
        uint32_t gS1End[ARRAY_SIZE(tilingData->outerSplitParams.gS1End)];
        uint32_t s2End[ARRAY_SIZE(tilingData->outerSplitParams.s2End)];
        copy_data_align64((uint8_t *)bN2End, (uint8_t *)(tilingData->outerSplitParams.bN2End), sizeof(bN2End));
        copy_data_align64((uint8_t *)gS1End, (uint8_t *)(tilingData->outerSplitParams.gS1End), sizeof(gS1End));
        copy_data_align64((uint8_t *)s2End, (uint8_t *)(tilingData->outerSplitParams.s2End), sizeof(s2End));
#endif
        // TND分核信息
        constInfo.bN2End = bN2End[aiCoreIdx];
        constInfo.gS1End = gS1End[aiCoreIdx];
        constInfo.s2End = s2End[aiCoreIdx];
        if (aiCoreIdx != 0) {
            GetAxisStartIdx(bN2End[aiCoreIdx - 1], gS1End[aiCoreIdx - 1], s2End[aiCoreIdx - 1]);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void
SparseFlashAttentionAntiquantGqa<SFAAT>::Bmm2DataCopyOut(uint64_t attenOutOffset, LocalTensor<OUT_T> &attenOutUb,
                                                         uint32_t startRow, uint32_t dealRowCount,
                                                         uint32_t columnCount, uint32_t actualColumnCount)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRowCount;
    dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
    dataCopyParams.srcStride = (columnCount - actualColumnCount) / (SFAAVectorServiceGqa<SFAAT>::BYTE_BLOCK /
        sizeof(OUT_T));
    dataCopyParams.dstStride = 0;
    DataCopyPad(attentionOutGm[attenOutOffset + (mSizeVStart + startRow) * actualColumnCount], attenOutUb,
                dataCopyParams);
}


template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::CalcParams(uint32_t loop, uint64_t s2Start,
                                                                           uint32_t s2LoopIdx, RunInfo &info)
{
    info.loop = loop;
    info.bN2Idx = tempLoopInfo.bN2Idx;
    info.bIdx = tempLoopInfo.bIdx;
    info.n2Idx = tempLoopInfo.n2Idx;
    info.gS1Idx = tempLoopInfo.gS1Idx;
    info.s2Idx = s2LoopIdx;
    info.curSInnerLoopTimes = tempLoopInfo.s2LoopTimes;
    info.threshold = tempLoopInfo.threshold;
    info.sparseBlockCount = tempLoopInfo.sparseBlockCount;

    info.tndIsS2SplitCore = tempLoopInfo.tndIsS2SplitCore;
    info.tndCoreStartKVSplitPos = tempLoopInfo.tndCoreStartKVSplitPos;
    info.isBmm2Output = false;

    info.actS1Size = tempLoopInfo.actS1Size;
    info.actS2Size = tempLoopInfo.curActualSeqLen;
    info.nextTokensPerBatch =
        static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size);
    
    info.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        info.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    info.isValid = s2LoopIdx < tempLoopInfo.s2LoopTimes;
    info.mSize = info.actMBaseSize;
    if ASCEND_IS_AIV {
        // info.mSizeV = (info.mSize <= 16) ? info.mSize : (((info.mSize + 15) / 16 + 1) / 2 * 16);
        info.mSizeV = (info.mSize + 1) / 2;
        info.mSizeVStart = 0;
        if (tmpBlockIdx % 2 == 1) {
            info.mSizeVStart = info.mSizeV;
            info.mSizeV = info.mSize - info.mSizeV;
        }
    }

    info.isChangeBatch = false;

    info.isFirstSInnerLoop = s2LoopIdx == s2Start;
    info.isLastS2Loop = s2LoopIdx == tempLoopInfo.s2LoopTimes - 1;
    uint64_t actualSeqQPrefixSum;
    if constexpr (LAYOUT_T == SFAA_LAYOUT::TND) {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsQGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqQPrefixSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.qSeqSize;
    }
    info.tndBIdxOffsetQ = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.combineHeadDim;
    uint64_t actualSeqKvSum;
    if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::TND) {
        actualSeqKvSum = (info.bIdx <= 0) ? 0 : actualSeqLengthsKVGm.GetValue(info.bIdx - 1);
    } else {
        actualSeqKvSum = (info.bIdx <= 0) ? 0 : info.bIdx * constInfo.kvSeqSize;
    }
    info.tndBIdxOffsetKV = actualSeqKvSum * constInfo.kvHeadNum * constInfo.combineHeadDim;
    if (info.isFirstSInnerLoop) {
        // 支持BSND/TND
        tensorACoreOffset = info.tndBIdxOffsetQ + info.gS1Idx * constInfo.kvHeadNum * constInfo.combineHeadDim
                            + info.n2Idx * constInfo.gSize * constInfo.combineHeadDim; // info.gS1Idx：前提是不切G，否则会有向下取整的问题
        tensorBCoreOffset = info.tndBIdxOffsetKV + info.n2Idx * constInfo.gSize * constInfo.combineHeadDim;
        topkGmBaseOffset = actualSeqQPrefixSum / constInfo.sparseShardSize * constInfo.kvHeadNum *
                            constInfo.sparseBlockCount + info.gS1Idx / constInfo.mBaseSize *
                            constInfo.sparseBlockCount * constInfo.kvHeadNum +
                            info.n2Idx * constInfo.sparseBlockCount;
    }
    info.topkGmBaseOffset = topkGmBaseOffset;
    info.tensorAOffset = tensorACoreOffset;
    info.tensorBOffset = tensorBCoreOffset;
    info.attenOutOffset = tensorACoreOffset;

    info.curS2BaseOffset = info.s2Idx * SALSK_S2BASEIZE;
    info.nextS2BaseOffset = info.curS2BaseOffset + SALSK_S2BASEIZE;
    info.s2BatchOffset = s2BatchBaseOffset + info.curS2BaseOffset;

    info.curActualSeqLenOri = tempLoopInfo.curActualSeqLenOri;
    // 计算实际基本块size
    if (tempLoopInfo.curActualSeqLen > info.curS2BaseOffset) {
        info.actualSingleProcessSInnerSize = tempLoopInfo.curActualSeqLen - info.curS2BaseOffset;
        info.actualSingleProcessSInnerSize = info.actualSingleProcessSInnerSize > SALSK_S2BASEIZE ?
                                             SALSK_S2BASEIZE : info.actualSingleProcessSInnerSize;
    } else {
        info.actualSingleProcessSInnerSize = 0;
    }
    info.aicS2AccessSize = (info.actualSingleProcessSInnerSize * 3 / 4) / 16 * 16;
    info.aivS2AccessSize = info.actualSingleProcessSInnerSize - info.aicS2AccessSize;
    info.actualSingleProcessSInnerSizeAlign =
        SFAAAlign((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)SFAAVectorServiceGqa<SFAAT>::BYTE_BLOCK);
    info.maskEnd = tempLoopInfo.sparseBlockActualSeqSizeKv - (info.actS1Size - info.gS1Idx / constInfo.mBaseSize * constInfo.sparseShardSize) +
 	         constInfo.sparseShardSize;
    info.maskStart = info.maskEnd - constInfo.sparseShardSize + 1;
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::ComputeMm1(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        if constexpr (SFAAT::isMsdDD) {
            matmulServiceMsd.ComputeMm1(info, mSplitInfo);
        }
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::ComputeMm2(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;
        CrossCoreWaitFlag(constInfo.syncV1C2);
        if constexpr (SFAAT::isMsdDD) {
            matmulServiceMsd.ComputeMm2(info, mSplitInfo);
        }
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC2V2);
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::FlashDecode()
{
    fdService.InitBuffers(pipe);
    AscendC::ICachePreLoad(fdPrefetchLen);
    SyncAll();
    if ASCEND_IS_AIV {
        if constexpr (IS_META) {
            uint32_t *bN2IdxOfFdHead = metaDataPtr->fdRes.bN2IdxOfFdHead;
            uint32_t *gS1IdxOfFdHead = metaDataPtr->fdRes.gS1IdxOfFdHead;
            uint32_t *s2SplitNumOfFdHead = metaDataPtr->fdRes.s2SplitNumOfFdHead;
            uint32_t *gS1IdxEndOfFdHead = metaDataPtr->fdRes.gS1IdxEndOfFdHead;
            uint32_t *gS1IdxEndOfFdHeadSplit = metaDataPtr->fdRes.gS1IdxEndOfFdHeadSplit;
            uint32_t *gS1SplitNumOfFdHead = metaDataPtr->fdRes.gS1SplitNumOfFdHead;
            uint32_t *gS1LastPartSizeOfFdHead = metaDataPtr->fdRes.gS1LastPartSizeOfFdHead;
            FDparams fdParams = {bN2IdxOfFdHead, gS1IdxOfFdHead, s2SplitNumOfFdHead, gS1SplitNumOfFdHead, gS1LastPartSizeOfFdHead,
                    gS1IdxEndOfFdHead, gS1IdxEndOfFdHeadSplit, metaDataPtr->usedVecNumOfFd,
                    metaDataPtr->gS1BaseSizeOfFd};
            fdService.AllocEventID();
            fdService.InitDecodeParams();
            fdService.FlashDecode(fdParams);
            fdService.FreeEventID();
        } else {
#ifdef ASCENDC_CPU_DEBUG
                const uint32_t *bN2IdxOfFdHead = tilingData->fdParams.bN2IdxOfFdHead;
                const uint32_t *gS1IdxOfFdHead = tilingData->fdParams.gS1IdxOfFdHead;
                const uint32_t *s2SplitNumOfFdHead = tilingData->fdParams.s2SplitNumOfFdHead;
                const uint32_t *gS1IdxEndOfFdHead = tilingData->fdParams.gS1IdxEndOfFdHead;
                const uint32_t *gS1IdxEndOfFdHeadSplit = tilingData->fdParams.gS1IdxEndOfFdHeadSplit;
                const uint32_t *gS1SplitNumOfFdHead = tilingData->fdParams.gS1SplitNumOfFdHead;
                const uint32_t *gS1LastPartSizeOfFdHead = tilingData->fdParams.gS1LastPartSizeOfFdHead;
#else
                uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.bN2IdxOfFdHead)];
                uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxOfFdHead)];
                uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.s2SplitNumOfFdHead)];
                uint32_t gS1IdxEndOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxEndOfFdHead)];
                uint32_t gS1IdxEndOfFdHeadSplit[ARRAY_SIZE(tilingData->fdParams.gS1IdxEndOfFdHeadSplit)];
                uint32_t gS1SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1SplitNumOfFdHead)];
                uint32_t gS1LastPartSizeOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1LastPartSizeOfFdHead)];
                copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->fdParams.bN2IdxOfFdHead),
                            sizeof(bN2IdxOfFdHead));
                copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxOfFdHead),
                            sizeof(gS1IdxOfFdHead));
                copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.s2SplitNumOfFdHead),
                            sizeof(s2SplitNumOfFdHead));
                copy_data_align64((uint8_t *)gS1IdxEndOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxEndOfFdHead),
                            sizeof(gS1IdxEndOfFdHead));
                copy_data_align64((uint8_t *)gS1IdxEndOfFdHeadSplit,
                            (uint8_t *)(tilingData->fdParams.gS1IdxEndOfFdHeadSplit),
                            sizeof(gS1IdxEndOfFdHeadSplit));
                copy_data_align64((uint8_t *)gS1SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.gS1SplitNumOfFdHead),
                            sizeof(gS1SplitNumOfFdHead));
                copy_data_align64((uint8_t *)gS1LastPartSizeOfFdHead,
                            (uint8_t *)(tilingData->fdParams.gS1LastPartSizeOfFdHead),
                            sizeof(gS1LastPartSizeOfFdHead));
#endif
            FDparams fdParams = {bN2IdxOfFdHead, gS1IdxOfFdHead, s2SplitNumOfFdHead, gS1SplitNumOfFdHead, gS1LastPartSizeOfFdHead,
                    gS1IdxEndOfFdHead, gS1IdxEndOfFdHeadSplit, tilingData->fdParams.usedVecNumOfFd,
                    tilingData->fdParams.gS1BaseSizeOfFd};
            fdService.AllocEventID();
            fdService.InitDecodeParams();
            fdService.FlashDecode(fdParams);
            fdService.FreeEventID();
        }
    }
}

template <typename SFAAT> __aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::Process()
{
    if (aiCoreIdx < usedCoreNum) {
        if constexpr (SFAAT::isMsdDD) {
            if ASCEND_IS_AIV {
                vectorServiceMsd.AllocEventID();
                vectorServiceMsd.InitSoftmaxDefaultBuffer();
            } else {
                matmulServiceMsd.AllocEventID();
            }
        }
        
        ProcessBalance();

        if constexpr (SFAAT::isMsdDD) {
            if ASCEND_IS_AIV {
                vectorServiceMsd.FreeEventID();
            } else {
                matmulServiceMsd.FreeEventID();
            }
        }
    }
    if constexpr (IS_META) {
        if (FLASH_DECODE) {
            FlashDecode();
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::GetBN2Idx(uint32_t bN2Idx, uint32_t &bIdx,
                                                                          uint32_t &n2Idx)
{
    bIdx = bN2Idx / constInfo.kvHeadNum ;
    n2Idx = bN2Idx % constInfo.kvHeadNum ;
}

template <typename SFAAT> __aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::ProcessBalance()
{
    RunInfo extraInfo[SFAA_PRELOAD_TASK_CACHE_SIZE];
    uint32_t gloop = 0;
    int gS1LoopEnd;
    bool globalLoopStart = true;
    if ASCEND_IS_AIC {
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_MTE2>(3);
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_MTE2>(3);
    }
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx <= constInfo.bN2End; bN2LoopIdx++) {
        GetBN2Idx(bN2LoopIdx, tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        tempLoopInfo.bN2Idx = bN2LoopIdx;
        GetActualSeqLen(tempLoopInfo.bIdx); // 获取actualSeqLength及ActualSeqLengthKV
        GetPreNextTokensLeftUp();
        if (tempLoopInfo.actS1Size == 0 && !(bN2LoopIdx == constInfo.bN2End)) {
            continue;
        }
        uint32_t gS1SplitNum;
        uint32_t s1SizeTail;
        if (tempLoopInfo.actS1Size == 0) {
            gS1SplitNum = 0;
            s1SizeTail = 0;
        } else {
            gS1SplitNum = (tempLoopInfo.actS1Size + constInfo.sparseShardSize - 1) / constInfo.sparseShardSize;
            s1SizeTail = tempLoopInfo.actS1Size - (gS1SplitNum - 1) * constInfo.sparseShardSize;
        }
        gS1LoopEnd = (bN2LoopIdx == constInfo.bN2End) ? constInfo.gS1End : gS1SplitNum - 1;
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx <= gS1LoopEnd; gS1LoopIdx++) {
            tempLoopInfo.gS1Idx = gS1LoopIdx * constInfo.mBaseSize;
            uint32_t s1Size = (gS1LoopIdx == gS1LoopEnd) ? s1SizeTail : constInfo.sparseShardSize;
            // TopK值sparse完后的ActualSeqLengthKV
            uint32_t s1StartIdx = gS1LoopIdx * constInfo.sparseShardSize;
            uint32_t s1EndIdx;
            if (s1Size == 0) {
                s1EndIdx = 0;
            } else {
                s1EndIdx = s1StartIdx + s1Size - 1;
            }
            GetSparseActualSeqLen(tempLoopInfo.bIdx, s1EndIdx, tempLoopInfo.n2Idx);
            UpdateInnerLoopCond();

            if (tempLoopInfo.curActSeqLenIsZero) {
                DealActSeqLenIsZero(tempLoopInfo.bIdx, s1StartIdx, s1Size, tempLoopInfo.n2Idx);
            }
            int s2SplitNum =
                (tempLoopInfo.curActualSeqLen + SALSK_S2BASEIZE - 1) / SALSK_S2BASEIZE; // S2切分份数
            bool isEnd = (bN2LoopIdx == constInfo.bN2End) && (gS1LoopIdx == constInfo.gS1End);
            if constexpr (IS_META) {
                tempLoopInfo.s2LoopTimes = (bN2LoopIdx == constInfo.bN2End && gS1LoopIdx == constInfo.gS1End && !tempLoopInfo.curActSeqLenIsZero) ? constInfo.s2End + 1 : s2SplitNum;
            } else {
                tempLoopInfo.s2LoopTimes = s2SplitNum;
            }
            // 分核修改后需要打开
            // 当前s2是否被切，决定了输出是否要写到attenOut上
            tempLoopInfo.tndIsS2SplitCore =
                ((constInfo.s2Start == 0) && (tempLoopInfo.s2LoopTimes == s2SplitNum)) ? false : true;
            tempLoopInfo.tndCoreStartKVSplitPos = globalLoopStart ? constInfo.coreStartKVSplitPos : 0;
            uint32_t extraLoop = isEnd ? 2 : 0;
            for (int s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopTimes + extraLoop); s2LoopIdx++) {
                // PreloadPipeline loop初始值要求为 PRELOAD_NUM
                PreloadPipeline(gloop, constInfo.s2Start, s2LoopIdx, extraInfo);
                ++gloop;
            }
            globalLoopStart = false;
            constInfo.s2Start = 0;
        }
        constInfo.gS1Start = 0;
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
        CrossCoreWaitFlag(3);
    }
}

template <typename SFAAT>
__aicore__ inline void
SparseFlashAttentionAntiquantGqa<SFAAT>::PreloadPipeline(uint32_t loop, uint64_t s2Start, uint64_t s2LoopIdx,
                                                         RunInfo extraInfo[SFAA_PRELOAD_TASK_CACHE_SIZE])
{
    RunInfo &extraInfo0 = extraInfo[loop % SFAA_PRELOAD_TASK_CACHE_SIZE];       // 本轮任务
    RunInfo &extraInfo2 = extraInfo[(loop + 2) % SFAA_PRELOAD_TASK_CACHE_SIZE]; // 上一轮任务
    RunInfo &extraInfo1 = extraInfo[(loop + 1) % SFAA_PRELOAD_TASK_CACHE_SIZE]; // 上两轮任务

    CalcParams(loop, s2Start, s2LoopIdx, extraInfo0);

    if (extraInfo0.isValid) {
        if ASCEND_IS_AIC {
            CrossCoreWaitFlag(constInfo.syncV0C1);
            ComputeMm1(extraInfo0);
        } else {
            if constexpr (SFAAT::isMsdDD) {
                vectorServiceMsd.ProcessVec0Msd(extraInfo0);
            }
            CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
        }
    }
    if (extraInfo2.isValid) {
        if ASCEND_IS_AIV {
            if constexpr (SFAAT::isMsdDD) {
                vectorServiceMsd.ProcessVec1Msd(extraInfo2);
            }
        }
        if ASCEND_IS_AIC {
            ComputeMm2(extraInfo2);
        }
    }
    if (extraInfo1.isValid) {
        if ASCEND_IS_AIV {
            if constexpr (SFAAT::isMsdDD) {
                vectorServiceMsd.ProcessVec2Msd(extraInfo1);
            }
        }
        extraInfo1.isValid = false;
    }
}

template <typename SFAAT>
__aicore__ inline uint64_t
SparseFlashAttentionAntiquantGqa<SFAAT>::GetBalanceActualSeqLengths(GlobalTensor<int32_t> &actualSeqLengths,
                                                                    uint32_t bIdx)
{
    if constexpr (LAYOUT_T == SFAA_LAYOUT::TND) {
        if (bIdx > 0) {
            return actualSeqLengths.GetValue(bIdx) - actualSeqLengths.GetValue(bIdx - 1);
        } else if (bIdx == 0) {
            return actualSeqLengths.GetValue(0);
        } else {
            return 0;
        }
    } else {
        if (constInfo.actualLenDimsQ == 0) {
            return constInfo.qSeqSize;
        } else if (constInfo.actualLenDimsQ == 1) {
            return actualSeqLengths.GetValue(0);
        } else {
            return actualSeqLengths.GetValue(bIdx);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SparseFlashAttentionAntiquantGqa<SFAAT>::GetAxisStartIdx(uint32_t bN2EndPrev,
                                                                                uint32_t s1GEndPrev,
                                                                                uint32_t s2EndPrev)
{
    if constexpr (IS_META) {
        uint32_t bEndPrev = bN2EndPrev / constInfo.kvHeadNum ;
        uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bEndPrev);
        uint32_t s1GPrevBaseNum = (actualSeqQPrev * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        
        // get s2PrevBaseNum
        uint32_t actualSeqKVPrev = GetActualSeqLenKV(bEndPrev);

        uint32_t sparseBlockActualSeqSizeKv = 0;
        if (constInfo.sparseLenDimsKV == 0) {
            sparseBlockActualSeqSizeKv = constInfo.sparseBlockCount * constInfo.sparseBlockSize;
        } else {
            sparseBlockActualSeqSizeKv = sparseSeqLengthsKVGm.GetValue(bEndPrev);
        }
        uint32_t curActualSeqLen = (sparseBlockActualSeqSizeKv > actualSeqKVPrev) ? actualSeqKVPrev : sparseBlockActualSeqSizeKv;
        uint32_t s2PrevBaseNum = (curActualSeqLen + SALSK_S2BASEIZE - 1) / SALSK_S2BASEIZE;

        constInfo.bN2Start = bN2EndPrev;
        constInfo.gS1Start = s1GEndPrev;
        constInfo.s2Start = s2EndPrev + 1U;

        if (constInfo.s2Start >= s2PrevBaseNum) {
            constInfo.gS1Start++;
            constInfo.s2Start = 0;
        }
        if (constInfo.gS1Start >= s1GPrevBaseNum) {
            constInfo.bN2Start++;
            constInfo.gS1Start = 0;
        }
    } else {
        uint32_t bEndPrev = bN2EndPrev / constInfo.kvHeadNum ;
        uint32_t actualSeqQPrev = GetBalanceActualSeqLengths(actualSeqLengthsQGm, bEndPrev);
        uint32_t s1GPrevBaseNum = (actualSeqQPrev * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        constInfo.bN2Start = bN2EndPrev;
        constInfo.gS1Start = s1GEndPrev;
        
        constInfo.s2Start = 0;
        if (s1GEndPrev >= s1GPrevBaseNum - 1) { // 上个核把S1G处理完了
            constInfo.gS1Start = 0;
            constInfo.bN2Start++;
        } else {
            constInfo.gS1Start++;
        }
    }
}
#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_KERNEL_GQA_H
