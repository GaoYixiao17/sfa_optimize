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
 * \file quant_sals_indexer_kernel.h
 * \brief
 */

#ifndef QUANT_SALS_INDEXER_KERNEL_H
#define QUANT_SALS_INDEXER_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_sals_indexer_common.h"
#include "quant_sals_indexer_service_vector.h"
#include "quant_sals_indexer_service_cube_int4.h"
#include "quant_sals_indexer_metadata.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))
namespace QSIKernel {
using namespace QSICommon;
using namespace optiling::detail;
using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

__aicore__ inline int32_t FloatCeil(float num) {
    int integer_part = static_cast<int>(num);
    if (num > 0 && num != integer_part) {
        integer_part += 1;
    }
    return integer_part;
}

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息；同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    int32_t s2LoopEnd = 0;       // S2方向循环的结束Idx
    uint32_t actS2Size = 0ULL;
    int32_t needProcessS2Size = 0L;
    int32_t targetTopKAlign = 0L;
    int32_t targetTopK = 0L;
    int32_t sparseBlockSize = 0UL;
    int32_t fixedTailCount = 0UL;
    float sparseRatio = 0.0f;
    float qScale = 0.0f;
    bool curActSeqLenIsZero = false;
    bool needCleanSparseCount = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    bool OnlyFixTail = false;
    uint32_t actMBaseSize = 0U;    // m轴(bN2)方向实际大小
    uint32_t s2BasicSizeTail = 0U; // S2方向循环的尾基本块大小
};

template <typename QSIT>
class QSIPreload {
public:
    __aicore__ inline QSIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *queryScale,
                                __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengths,
                                __gm__ uint8_t *blockTable, QsiMetaData *metaData,
                                __gm__ uint8_t *sparseIndices, __gm__ uint8_t *workspace,
                                const QSITilingData *__restrict tiling, TPipe *tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QSIT::queryType;
    using K_T = typename QSIT::keyType;
    using OUT_T = typename QSIT::outputType;
    static constexpr bool PAGE_ATTENTION = QSIT::pageAttention;
    static constexpr QSI_LAYOUT K_LAYOUT_T = QSIT::keyLayout;

    using MM1_OUT_T = int32_t;

    QSIMatmulInt4<QSIT> matmulService;
    QSIVector<QSIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    static constexpr uint32_t M_BASE_SIZE = 1;
    static constexpr uint32_t S2_BASE_SIZE = 2048;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t SI_QUANT_PRELOAD_TASK_CACHE_SIZE = 2;

    static constexpr int64_t LD_PREFETCH_LEN = 2;
    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;

protected:
    TPipe *pipe = nullptr;
    QsiMetaData *metaDataPtr = nullptr;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;

    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;

    GlobalTensor<uint32_t> actualSeqLengthsGm;
    // workspace
    GlobalTensor<MM1_OUT_T> mm1ResGm;  // 存放S
    GlobalTensor<float> vec1ResGm;     // 存放TopK计算中间结果
    GlobalTensor<int64_t> vec1ParamGm; // 存放LD参数信息

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;
    uint32_t usedCoreNum = 0U;

    uint64_t keyCoreOffset = 0ULL;
    uint64_t actualSeqKPrefixSum = 0ULL;
    QSICommon::ConstInfo constInfo{};
    TempLoopInfo tempLoopInfo{};
    QSICommon::SplitCoreInfo splitCoreInfo{};

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QSITilingData *__restrict tilingData);
    __aicore__ inline void InitMetaData();
    __aicore__ inline void InitCalcParamsEach(const QSITilingData *__restrict tilingData,
                                              QSICommon::SplitCoreInfo &info);
    __aicore__ inline void GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev,
                                           uint32_t s2EndPrev, QSICommon::SplitCoreInfo &info);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t *actualSeqLengths);
    // ================================Split Core================================
    __aicore__ inline void SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, QSICommon::SplitCoreInfo &info);
    __aicore__ inline int32_t GetS2NeedProcessSize(uint32_t actS2Size);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, QSICommon::RunInfo &runInfo);
    __aicore__ inline void ProcessDecode();
    __aicore__ inline void ProcessInvalid();
    // ================================Params Calc=====================================
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QSICommon::RunInfo &runInfo);
    __aicore__ inline void CalcRunInfo(QSICommon::RunInfo &runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx);
};

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::InitTilingData(const QSITilingData *__restrict tilingData)
{
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.sparseBlockSize = tilingData->sparseBlockSize;
    constInfo.sparseRatio = tilingData->sparseRatio;
    constInfo.fixedTailCount = tilingData->fixedTailCount;
    constInfo.outputLayout = QSI_LAYOUT::BSND; // 输出和输入形状一致
    constInfo.maxSeqlenKey = tilingData->maxSeqlenKey;

    if (K_LAYOUT_T == QSI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = tilingData->n2Size;
    constInfo.headDim = tilingData->dSize;

    constInfo.mBaseSize = M_BASE_SIZE;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.s1BaseSize = M_BASE_SIZE;
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::InitMetaData()
{
    usedCoreNum = metaDataPtr->usedCoreNum;
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::InitCalcParamsEach(const QSITilingData *__restrict tilingData,
                                                            QSICommon::SplitCoreInfo &info)
{
    // 计算总的基本块
    // 这里是编译器优化写法，定义一个局部数组变量coreSidxEnd(存在栈上)，使用copy_data_align64接口
    // 可以只从ub中拷贝tiling中coreSidxEnd的内容到栈上，而非将整个increFlashAttentionCoreParams
    // 内容拷贝到栈，减少拷贝时间
    if (metaDataPtr != nullptr) {
        const uint32_t *bN2End = metaDataPtr->bN2End;
        const uint32_t *gS1End = metaDataPtr->gS1End;
        const uint32_t *s2End = metaDataPtr->s2End;
        // TND分核信息
        info.bN2End = bN2End[aiCoreIdx];
        info.gS1End = gS1End[aiCoreIdx];
        info.s2End = s2End[aiCoreIdx];
        if (aiCoreIdx != 0) {
            GetAxisStartIdx(bN2End[aiCoreIdx - 1], gS1End[aiCoreIdx - 1], s2End[aiCoreIdx - 1], info);
        }
        // 当前核是否需要进行Decode规约任务
        uint32_t bEnd = info.bN2End / constInfo.kHeadNum;
        uint32_t s2BaseNum = (GetActualSeqLen(bEnd) + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
        info.isLD = (info.s2Start > 0U || info.s2End < s2BaseNum - 1U);
    } else {
#ifdef ASCENDC_CPU_DEBUG
            const uint32_t *bN2End = tilingData->splitParams.bN2End;
            const uint32_t *gS1End = tilingData->splitParams.gS1End;
            const uint32_t *s2End = tilingData->splitParams.s2End;
#else
            uint32_t bN2End[ARRAY_SIZE(tilingData->splitParams.bN2End)];
            uint32_t gS1End[ARRAY_SIZE(tilingData->splitParams.gS1End)];
            uint32_t s2End[ARRAY_SIZE(tilingData->splitParams.s2End)];
            copy_data_align64((uint8_t*)bN2End, (uint8_t*)(tilingData->splitParams.bN2End), sizeof(bN2End));
            copy_data_align64((uint8_t*)gS1End, (uint8_t*)(tilingData->splitParams.gS1End), sizeof(gS1End));
            copy_data_align64((uint8_t*)s2End, (uint8_t*)(tilingData->splitParams.s2End), sizeof(s2End));
#endif
        // TND分核信息
        info.bN2End = bN2End[aiCoreIdx];
        info.gS1End = gS1End[aiCoreIdx];
        info.s2End = s2End[aiCoreIdx];
        if (aiCoreIdx != 0) {
            GetAxisStartIdx(bN2End[aiCoreIdx - 1], gS1End[aiCoreIdx - 1], s2End[aiCoreIdx - 1], info);
        }
        // 当前核是否需要进行Decode规约任务
        info.isLD = false; // 保持兼容性
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::GetAxisStartIdx(uint32_t bN2EndPrev, uint32_t s1GEndPrev,
                                                         uint32_t s2EndPrev, QSICommon::SplitCoreInfo &info)
{
    if (metaDataPtr != nullptr) {
        uint32_t bEndPrev = bN2EndPrev / constInfo.kHeadNum;
        uint32_t actualSeqQPrev = 1;
        uint32_t s1GPrevBaseNum = (actualSeqQPrev + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        uint32_t s2PrevBaseNum = (GetActualSeqLen(bEndPrev) + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
        info.bN2Start = bN2EndPrev;
        info.gS1Start = s1GEndPrev;
        info.s2Start = s2EndPrev + 1U;

        if (info.s2Start >= s2PrevBaseNum) {
            info.gS1Start++;
            info.s2Start = 0;
        }
        if (info.gS1Start >= s1GPrevBaseNum) {
            info.bN2Start++;
            info.gS1Start = 0;
        }
    } else {
        uint32_t bEndPrev = bN2EndPrev / constInfo.kHeadNum;
        uint32_t actualSeqQPrev = 1;
        uint32_t s1GPrevBaseNum = (actualSeqQPrev + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
        info.bN2Start = bN2EndPrev;
        info.gS1Start = s1GEndPrev;

        info.s2Start = 0;
        if (s1GEndPrev >= s1GPrevBaseNum - 1) { // 上个核把S1G处理完了
            info.gS1Start = 0;
            info.bN2Start++;
        } else {
            info.gS1Start++;
        }
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::InitActualSeqLen(__gm__ uint8_t *actualSeqLengths)
{
    // todo 删除
    constInfo.actualLenQDims = constInfo.batchSize;
    if (actualSeqLengths == nullptr) {
        constInfo.actualLenDims = 0;
    } else {
        constInfo.actualLenDims = constInfo.batchSize;
        actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t *)actualSeqLengths, constInfo.actualLenDims);
    }
}

template <typename QSIT>
__aicore__ inline uint32_t QSIPreload<QSIT>::GetActualSeqLen(uint32_t bIdx)
{
    if (constInfo.actualLenDims == 0) {
        return constInfo.kSeqSize;
    } else if (constInfo.isAccumSeqS2 && bIdx > 0) {
        return actualSeqLengthsGm.GetValue(bIdx) - actualSeqLengthsGm.GetValue(bIdx - 1);
    } else {
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QSIT>
__aicore__ inline int32_t QSIPreload<QSIT>::GetS2NeedProcessSize(uint32_t actS2Size)
{
    int32_t totalNCount = (actS2Size + constInfo.sparseBlockSize - 1) / constInfo.sparseBlockSize;
    if (totalNCount <= constInfo.fixedTailCount) {
        return 0;
    }
    return (totalNCount - constInfo.fixedTailCount) * constInfo.sparseBlockSize;
}

template <typename QSIT>
__aicore__ inline uint32_t QSIPreload<QSIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        uint32_t actS2Size = GetActualSeqLen(bIdx);

        totalBlockNum += (GetS2NeedProcessSize(actS2Size)+ constInfo.s2BaseSize - 1) / constInfo.s2BaseSize * constInfo.kHeadNum;
    }
    return totalBlockNum;
}

// 多核版本，双闭区间
template <typename QSIT>
__aicore__ void inline QSIPreload<QSIT>::SplitCore(uint32_t curCoreIdx, uint32_t &coreNum, QSICommon::SplitCoreInfo &info)
{
    // 计算每个核最少处理的块数, 剩余的部分前面的核每个核多处理一块
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;
    uint32_t coreIdx = 0;
    uint32_t lastBN2RemainBlockCnt = 0;
    uint32_t coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
    coreNum = minBlockPerCore == 0 ? deal1MoreBlockCoreNum : coreNum;

    bool findLastCoreEnd = true;
    uint32_t s2BaseNum;
    for (uint32_t bN2Idx = 0; bN2Idx < constInfo.batchSize * constInfo.kHeadNum; bN2Idx++) {
        uint32_t bIdx = bN2Idx / constInfo.kHeadNum;
        if (bN2Idx % constInfo.kHeadNum == 0) {
            uint32_t actS2Size = GetActualSeqLen(bIdx);
            s2BaseNum = (GetS2NeedProcessSize(actS2Size) + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
        }
        if (findLastCoreEnd && s2BaseNum == 0U) {
            info.bN2Start = bN2Idx;
            info.s2Start = 0;
            findLastCoreEnd = false;
        }
        for (uint32_t s2Idx = 0; s2Idx < s2BaseNum;) {
            uint32_t s2RemainBaseNum = s2BaseNum - s2Idx;
            if (findLastCoreEnd) {
                info.bN2Start = bN2Idx;
                info.s2Start = s2Idx;
                findLastCoreEnd = false;
            }
            if (lastBN2RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                info.bN2End = bN2Idx;
                info.s2End = s2Idx + coreDealBlockCnt - lastBN2RemainBlockCnt - 1;

                if (coreIdx == curCoreIdx) {
                    // S2被切N核，那么只有第一个核需要处理LD，其他核不用
                    if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                        info.isLD = true;
                    }
                    // 最后一个核处理的不是最后一个Batch，表明后面的Batch为空块(S2=0), 调整终点坐标以便清理输出
                    if (coreIdx == coreNum - 1 && info.bN2End / constInfo.kHeadNum != constInfo.batchSize - 1) {
                        info.bN2End = (constInfo.batchSize * constInfo.kHeadNum) - 1;
                        info.s2End = 0;
                    }
                    return;
                }
                coreIdx++;
                findLastCoreEnd = true;
                s2Idx = info.s2End + 1;
                lastBN2RemainBlockCnt = 0;
                coreDealBlockCnt = coreIdx < deal1MoreBlockCoreNum ? minBlockPerCore + 1 : minBlockPerCore;
            } else {
                lastBN2RemainBlockCnt += s2RemainBaseNum;
                break;
            }
        }
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx)
{
    if ASCEND_IS_AIV {
        // B,N2,K
        if (GetSubBlockIdx() == 0) {
            return;
        }
        uint64_t indiceOutOffset =
            bIdx * constInfo.kHeadNum * constInfo.sparseCount + n2Idx * constInfo.sparseCount;  // N2轴偏移
        vectorService.CleanInvalidOutput(indiceOutOffset);
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::Init(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                            __gm__ uint8_t *queryScale, __gm__ uint8_t *keyScale, __gm__ uint8_t *actualSeqLengths,
                                            __gm__ uint8_t *blockTable, QsiMetaData *metaData, __gm__ uint8_t *sparseIndices,
                                            __gm__ uint8_t *workspace, const QSITilingData *__restrict tiling,
                                            TPipe *tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(actualSeqLengths);

    // 计算分核
    // SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);
    if (metaData != nullptr) {
        metaDataPtr = metaData;
        InitMetaData();
    }
    InitCalcParamsEach(tiling, splitCoreInfo);

    pipe = tPipe;
    // workspace 内存排布
    // |mm1ResGm(存S)|vec1ResGm(存LD中间结果)|vec1ParamGm(存LD参数)
    // |Core0_mm1ResDB0-Core0_mm1ResDB1-Core1_mm1ResDB0....Core23_mm1ResDB0-Core23_mm1ResDB1|Core0_vec1Res...
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.mBaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T *)(workspace + offset + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    // ld流程需要ws大小: [aicnum, 2, constInfo.mBaseSize, topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    vec1ResGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * MAX_TOPK * sizeof(float);

    // (aic, 8, 2, 16)
    // (aic, s1_cube, 头尾，16ele)
    vec1ParamGm.SetGlobalBuffer((__gm__ int64_t *)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * LD_PARAM_NUM * sizeof(int64_t);

    qScaleGm.SetGlobalBuffer((__gm__ float *)queryScale);
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t *)sparseIndices);
        kScaleGm.SetGlobalBuffer((__gm__ float *)keyScale);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        vectorService.InitVec1GlobalTensor(mm1ResGm, qScaleGm, kScaleGm, blockTableGm, vec1ResGm, vec1ParamGm,
                                           indiceOutGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T *)key);
        keyGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm);
    }
    InitBuffers();
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::CalcS2LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    tempLoopInfo.qScale = qScaleGm.GetValue(tempLoopInfo.bIdx * constInfo.kHeadNum + tempLoopInfo.n2Idx);
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    tempLoopInfo.actS2Size = GetActualSeqLen(bN2LoopIdx / constInfo.kHeadNum);

    tempLoopInfo.needProcessS2Size = GetS2NeedProcessSize(tempLoopInfo.actS2Size);
    tempLoopInfo.s2LoopEnd =(tempLoopInfo.needProcessS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize - 1;
    if (metaDataPtr != nullptr) {
        tempLoopInfo.s2LoopEnd = (bN2LoopIdx == splitCoreInfo.bN2End && !(tempLoopInfo.actS2Size == 0)) ? splitCoreInfo.s2End : tempLoopInfo.s2LoopEnd;
    }
    tempLoopInfo.needCleanSparseCount = (tempLoopInfo.actS2Size < constInfo.maxSeqlenKey);
    int32_t targetTopK = FloatCeil(((tempLoopInfo.needProcessS2Size + constInfo.sparseBlockSize - 1) / (constInfo.sparseBlockSize)) * constInfo.sparseRatio);
    targetTopK = targetTopK >= 2048 ? 2048 : targetTopK;
    tempLoopInfo.OnlyFixTail = (targetTopK == 0);
    if (targetTopK > constInfo.s2BaseSize / constInfo.sparseBlockSize) {
        tempLoopInfo.targetTopKAlign = QSiCeilAlign(static_cast<uint32_t>(targetTopK), 4 * constInfo.s2BaseSize / constInfo.sparseBlockSize);
    } else {
        tempLoopInfo.targetTopKAlign = QSiCeilAlign(targetTopK, 128);
    }
    tempLoopInfo.targetTopK = targetTopK;
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::CalcRunInfo(QSICommon::RunInfo &runInfo)
{
    runInfo.needProcessS2Size = tempLoopInfo.needProcessS2Size;
    runInfo.targetTopKAlign = tempLoopInfo.targetTopKAlign;
    runInfo.targetTopK = tempLoopInfo.targetTopK;
    runInfo.indiceOutOffset = tempLoopInfo.bN2Idx * constInfo.sparseCount;
    int32_t totalNCount = (tempLoopInfo.actS2Size + constInfo.sparseBlockSize - 1) / constInfo.sparseBlockSize;
    runInfo.fixedTailCount = constInfo.fixedTailCount > totalNCount ? totalNCount : constInfo.fixedTailCount;
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QSICommon::RunInfo &runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.n2Idx = tempLoopInfo.n2Idx;

    runInfo.qScale = tempLoopInfo.qScale;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.needProcessS2Size = tempLoopInfo.needProcessS2Size;
    runInfo.targetTopKAlign = tempLoopInfo.targetTopKAlign;
    runInfo.targetTopK = tempLoopInfo.targetTopK;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.fixedTailCount = constInfo.fixedTailCount;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.needProcessS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (s2SplitNum == 0) {
        tempLoopInfo.s2BasicSizeTail = 0;
        runInfo.actualSingleProcessSInnerSize = 0;
    } else {
        tempLoopInfo.s2BasicSizeTail = tempLoopInfo.needProcessS2Size - ((s2SplitNum - 1) * constInfo.s2BaseSize);
        if (runInfo.s2Idx == s2SplitNum - 1) {
            runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
        }
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        QSICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, QSICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == splitCoreInfo.s2Start;
    runInfo.isLastS2InnerLoop = s2LoopIdx == tempLoopInfo.s2LoopEnd;
    runInfo.isAllLoopEnd = (runInfo.bN2Idx == splitCoreInfo.bN2End) && (runInfo.s2Idx == tempLoopInfo.s2LoopEnd);

    if (runInfo.isFirstS2InnerLoop) {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
        uint64_t bsndKeyBIdxOffset = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
        // B,N2,D
        runInfo.tensorQueryOffset = runInfo.bN2Idx * constInfo.headDim;

        // bsnd or pa_bsnd
        keyCoreOffset = bsndKeyBIdxOffset + runInfo.n2Idx * constInfo.headDim;
        // B,N2,k
        runInfo.indiceOutOffset = runInfo.bN2Idx * constInfo.sparseCount;
    }
    runInfo.tensorKeyOffset = keyCoreOffset + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    runInfo.tensorKeyScaleOffset = actualSeqKPrefixSum * constInfo.kHeadNum +
                                   runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum;
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::Process()
{
    if (usedCoreNum == 0) {
        // 没有计算任务，直接清理输出
        ProcessInvalid();
        return;
    }
    ProcessMain();
    ProcessDecode();
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            QSICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize < totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::ProcessMain()
{
    if (aiCoreIdx >= usedCoreNum) {
        // 无任务核直接返回
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QSICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<QSICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    QSICommon::RunInfo runInfo;
    uint32_t loopIdx = 0;
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcS2LoopParams(bN2LoopIdx);
        if (tempLoopInfo.needCleanSparseCount && splitCoreInfo.s2Start == 0) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx);
        }
        if (tempLoopInfo.OnlyFixTail) {
            CalcRunInfo(runInfo);
            if ASCEND_IS_AIV {
                WaitFlag<HardEvent::MTE3_V>(0);
                vectorService.CopyOutResult(runInfo);
                SetFlag<HardEvent::MTE3_V>(0);
            }
            continue;
        }
        for (uint32_t s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= tempLoopInfo.s2LoopEnd; s2LoopIdx++) {
            ProcessBaseBlock(loopIdx, s2LoopIdx, runInfo);
            ++loopIdx;
        }
        splitCoreInfo.s2Start = 0;
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx, QSICommon::RunInfo &runInfo)
{
    CalcRunInfo(loop, s2LoopIdx, runInfo);
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<QSICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<QSICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    }
}

template <typename QSIT>
__aicore__ inline void QSIPreload<QSIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        SyncAll();
        if (splitCoreInfo.isLD) {
            vectorService.InitLDBuffers(pipe);
            ICachePreLoad(LD_PREFETCH_LEN);
            vectorService.ProcessLD();
        }
    }
}
} // namespace QSIKernel
#endif // QUANT_SALS_INDEXER_KERNEL_H