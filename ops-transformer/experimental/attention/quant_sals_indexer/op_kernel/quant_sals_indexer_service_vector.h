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
 * \file quant_sals_indexer_service_vector.h
 * \brief
 */
#ifndef QUANT_SALS_INDEXER_SERVICE_VECTOR_H
#define QUANT_SALS_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_sals_indexer_common.h"
//#include "quant_sals_indexer_vector.h"

namespace QSIKernel {
using namespace AscendC;
using namespace QSICommon;

template <typename QSIT>
class QSIVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using K_T = typename QSIT::keyType;
    static constexpr QSI_LAYOUT LAYOUT_T = QSIT::layout;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = int32_t;

    __aicore__ inline QSIVector(){};
    __aicore__ inline void ProcessVec(const QSICommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QSICommon::ConstInfo &constInfo,
                                      const QSITilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                GlobalTensor<float> qScaleGm,
                                                GlobalTensor<float> kScaleGm,
                                                GlobalTensor<int32_t> blkTableGm,
                                                GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm,
                                                GlobalTensor<int32_t> indiceOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);
    __aicore__ inline void CopyMm1ResultIn(const QSICommon::RunInfo &runInfo);
    __aicore__ inline void GetKeyScale(const QSICommon::RunInfo &runInfo);
    __aicore__ inline void CopyOut(const GlobalTensor<int32_t> &dstGm,
        const LocalTensor<int32_t> &srcUb, int64_t copyCount);
    __aicore__ inline void ComputeLse(const QSICommon::RunInfo &runInfo);
    __aicore__ inline void SortSubTopK(const QSICommon::RunInfo &runInfo);
    __aicore__ inline void SortAll(const LocalTensor<float> &dst, const LocalTensor<float> &src, int64_t logitsNum, int64_t mrgElements);
    __aicore__ inline void SortBasicBlockTopKToSub(const QSICommon::RunInfo &runInfo);
    __aicore__ inline void CopyOutResult(const QSICommon::RunInfo &runInfo);
    __aicore__ inline void CopyOutFdResult(const QSICommon::RunInfo &runInfo);
    __aicore__ inline void ExtractIndex(const LocalTensor<int32_t> &idxULocal, const LocalTensor<int32_t> &sortLocal,
                                    int64_t extractNum);
protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm_;
    GlobalTensor<float> qScaleGm_;
    GlobalTensor<float> kScaleGm_;
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm_;
    GlobalTensor<int32_t> indiceOutGm_;
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // tmp buff for vector1
    TBuf<> outQueue_;
    TBuf<> sortOutBuf_;
    TBuf<> reduceOutBuf_;
    TBuf<> brcBuf_;
    TBuf<> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<float> mmInUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;
    LocalTensor<float> qScaleUb_;
    LocalTensor<float> kScaleUb_;
    LocalTensor<float> qkScaleUb_;

    LocalTensor<float> nBlkMaxUb_;
    LocalTensor<float> nMaxUb_;
    LocalTensor<float> nMaxBrcbUb_;
    LocalTensor<float> nLseUb_;
    LocalTensor<float> nLseTobeSortUb_;
    LocalTensor<float> sortTensor_;
    LocalTensor<float> nSortOutUb_;
    LocalTensor<int32_t> nIdxUb_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int64_t mte2BufIdx_ = 0;
    int64_t mte3BufIdx_ = 0;
    int64_t nSubSortCountOffset_ = 0;
    int64_t nSortCountOffset_ = 0;
    int64_t sortedIdx_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;
    constexpr static uint32_t paramNum_ = 16;

    constexpr static int64_t MTE2_BUF_SIZE = 2048;
    constexpr static int64_t SORT_BUF_SIZE = 64 * 256;
    constexpr static int64_t MAX_TOPK_WITH_ID = 2048 * 2;
    constexpr static int64_t DOUBLE_BUFFER_NUM = 2;
    constexpr static int64_t B32_VEC_BLK_NUM = 8;
    constexpr static int64_t B32_VEC_MAX_MASK = 64;
    constexpr static int64_t SORT_VALUE_ID_SIZE = 8;
    constexpr static int64_t SORT_MASK = 32;

    constexpr static int32_t NEG_INF = 0xFF800000;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    constexpr static uint32_t V_MTE2_EVENT = 0;
    constexpr static uint32_t V_MTE2_EVENT_FD = 2;
    constexpr static uint32_t S_MTE2_EVENT = 0;

    constexpr static uint32_t V_MTE3_EVENT = 0;
    constexpr static uint32_t S_MTE3_EVENT = 0;
    constexpr static uint32_t V_S_EVENT = 0;
    constexpr static uint32_t MTE3_S_EVENT = 0;

    constexpr static uint32_t MTE2_V_EVENT = 0;
    constexpr static uint32_t MTE3_V_EVENT = 0;
    constexpr static uint32_t MTE3_V_EVENT_FD = 2;

    constexpr static int64_t MRG_QUE_0 = 0;
    constexpr static int64_t MRG_QUE_1 = 1;
    constexpr static int64_t MRG_QUE_2 = 2;
    constexpr static int64_t MRG_QUE_3 = 3;
    constexpr static int64_t MRG_BLOCK_2 = 2;
    constexpr static int64_t MRG_BLOCK_3 = 3;
    constexpr static int64_t MRG_BLOCK_4 = 4;
    constexpr static int64_t VALUE_AND_INDEX_NUM = 2;
    static constexpr bool PAGE_ATTENTION = QSIT::pageAttention;
    static constexpr QSI_LAYOUT K_LAYOUT_T = QSIT::keyLayout;

    struct QSICommon::ConstInfo constInfo_;
};

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (MAX_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * constInfo_.s2BaseSize * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;

    // 1, s2 -> 1 * 2048 输入开db 16k
    TBuf<> inputBuf;
    pipe->InitBuffer(inputBuf, DOUBLE_BUFFER_NUM * constInfo_.s2BaseSize * sizeof(float));
    mmInUb_ = inputBuf.Get<float>();

    // topk的排序结果, mrgSort最多4条排序队列，空间预留128k
    TBuf<> sortBuf;
    pipe->InitBuffer(sortBuf,  DOUBLE_BUFFER_NUM * MAX_TOPK_WITH_ID * 4 * sizeof(float));
    sortTensor_ = sortBuf.Get<float>();
    Duplicate(sortTensor_.template ReinterpretCast<int32_t>(), NEG_INF, 2 * SORT_BUF_SIZE);
    // 基本块的排序索引缓存, 最多支持256个待排序索引，空间预留1k
    TBuf<> indexBuf;
    pipe->InitBuffer(indexBuf,  256 * sizeof(int32_t));
    globalTopkIndice_ = indexBuf.Get<int32_t>();

    TBuf<> tmpBuf;
    pipe->InitBuffer(tmpBuf, 46*1024);
    nBlkMaxUb_ = tmpBuf.Get<float>();   //8k
    qScaleUb_ = tmpBuf.Get<float>();    // 8k, 复用nBlkMaxUb_
    qkScaleUb_ = tmpBuf.Get<float>();   // 8k, 复用nBlkMaxUb_
    nMaxBrcbUb_ =tmpBuf.Get<float>()[8 * 256];  //8k;
    kScaleUb_ = tmpBuf.Get<float>()[8 * 256];   // 8k, 复用nMaxBrcbUb_
    nMaxUb_ =tmpBuf.Get<float>()[16 * 256];     //1k;
    nLseTobeSortUb_ =tmpBuf.Get<float>()[17 * 256];     //2k;

    nSortOutUb_ =tmpBuf.Get<float>()[19 * 256];     //16k;
    nIdxUb_ =tmpBuf.Get<int32_t>()[36 * 256];       //8k;
    pipe->InitBuffer(paramBuf_, 1024);

    ArithProgression<int32_t>(globalTopkIndice_, 0, 1,
        QSiCeilDiv(constInfo_.s2BaseSize, static_cast<uint32_t>(constInfo_.sparseBlockSize)));

    LocalTensor<int32_t> tmpfBuff = sortBuf.Get<int32_t>();
    Duplicate(tmpfBuff, -1, 2 * (constInfo_.s1BaseSize / 2) * paramNum_ * 2);
    SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    int64_t wsInfoOffset = blockId_ * constInfo_.s1BaseSize * 2 * paramNum_;
    DataCopyPad(vec1ParamGm_[wsInfoOffset], tmpfBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>(constInfo_.s1BaseSize * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * MAX_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * MAX_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, MAX_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, MAX_TOPK * sizeof(int32_t) + 32);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::InitParams(const struct QSICommon::ConstInfo &constInfo,
                                                 const QSITilingData *__restrict tilingData)
{
    constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    // define N2 para
    kHeadNum_ = constInfo_.kHeadNum;

    // group ub 切分因子当前按照UB空间强制为16
    groupInner_ = 16;

    blockId_ = GetBlockIdx();
}

template <typename QSIT>
__aicore__ inline void
QSIVector<QSIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                      GlobalTensor<float> qScaleGm,
                                      GlobalTensor<float> kScaleGm,
                                      GlobalTensor<int32_t> blkTableGm,
                                      GlobalTensor<float> vec1ResGm,
                                      GlobalTensor<int64_t> vec1ParamGm,
                                      GlobalTensor<int32_t> indiceOutGm)
{
    mm1ResGm_ = mm1ResGm;
    qScaleGm_ = qScaleGm;
    blkTableGm_ = blkTableGm;
    kScaleGm_ = kScaleGm;
    this->vec1ResGm = vec1ResGm;
    vec1ParamGm_ = vec1ParamGm;
    indiceOutGm_ = indiceOutGm;
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + 1);

    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + 1);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + 1);

    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT + 1);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::CopyOut(const GlobalTensor<int32_t> &dstGm,
    const LocalTensor<int32_t> &srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(int32_t);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::CopyMm1ResultIn(const QSICommon::RunInfo &runInfo)
{
    // 将MMout_gmoffset copy到UB上
    AscendC::DataCopyPadExtParams<int32_t> Mm1padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams Mm1dataCopymMoutParams;
    Mm1dataCopymMoutParams.blockCount = 1;
    Mm1dataCopymMoutParams.blockLen = runInfo.actualSingleProcessSInnerSizeAlign * sizeof(int32_t);
    Mm1dataCopymMoutParams.srcStride = 0;
    Mm1dataCopymMoutParams.dstStride = 0;
    Mm1dataCopymMoutParams.rsv = 0;
    int64_t mmGmOffset = (runInfo.loop % 2) * constInfo_.mBaseSize  * constInfo_.s2BaseSize;
    AscendC::DataCopyPad(mmInUb_.template ReinterpretCast<int32_t>()[mte2BufIdx_ % DOUBLE_BUFFER_NUM * MTE2_BUF_SIZE],
                         mm1ResGm_[mmGmOffset], Mm1dataCopymMoutParams, Mm1padParams);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::GetKeyScale(const QSICommon::RunInfo &runInfo)
{
    // 读取kScale
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t startS2 = runInfo.s2Idx * constInfo_.s2BaseSize;
        int32_t curS2Len = runInfo.actualSingleProcessSInnerSize;
        int32_t startBlockTableIdx = startS2 / constInfo_.kCacheBlockSize;
        int32_t startBlockTableOffset = startS2 % constInfo_.kCacheBlockSize;
        int32_t blockTableBatchOffset = runInfo.bIdx * constInfo_.maxBlockNumPerBatch;
        if constexpr (K_LAYOUT_T == QSI_LAYOUT::PA_BNSD || K_LAYOUT_T == QSI_LAYOUT::PA_NZ) {
            dataCopymMoutParams.blockCount = 1;
            dataCopymMoutParams.srcStride = 0;
        } else {
            dataCopymMoutParams.blockCount = 1;
            dataCopymMoutParams.srcStride = 0;
        }
        dataCopymMoutParams.dstStride = 0;
        dataCopymMoutParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                constInfo_.kCacheBlockSize - startBlockTableOffset > curS2Len ? curS2Len :
                constInfo_.kCacheBlockSize - startBlockTableOffset;
            dataCopymMoutParams.blockLen = firstPartLen * sizeof(float);
            uint64_t keyScaleGmOffset = 0;
            if constexpr (K_LAYOUT_T == QSI_LAYOUT::PA_BNSD || K_LAYOUT_T == QSI_LAYOUT::PA_NZ) { // PA_BNSD(blockNum, n2, blockSize)
                keyScaleGmOffset = blkTableGm_.GetValue(blockTableBatchOffset + startBlockTableIdx) *
                                   constInfo_.kHeadNum * constInfo_.kCacheBlockSize +
                                   runInfo.n2Idx * constInfo_.kCacheBlockSize + startBlockTableOffset;
            } else { // PA_BSND(blockNum, blockSize, n2)
                keyScaleGmOffset = blkTableGm_.GetValue(blockTableBatchOffset + startBlockTableIdx) *
                                   constInfo_.kCacheBlockSize * constInfo_.kHeadNum +
                                   startBlockTableOffset * constInfo_.kHeadNum + runInfo.n2Idx;
            }
            SetFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUb_, kScaleGm_[keyScaleGmOffset], dataCopymMoutParams, padParams);
            startBlockTableIdx++;
            curS2Len -= firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = QSiCeilDiv(curS2Len, constInfo_.kCacheBlockSize);
        dataCopymMoutParams.blockLen = constInfo_.kCacheBlockSize * sizeof(float);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                dataCopymMoutParams.blockLen = (curS2Len - i * constInfo_.kCacheBlockSize) * sizeof(float);
            }
            uint64_t keyScaleGmOffset = 0;
            if constexpr (K_LAYOUT_T == QSI_LAYOUT::PA_BNSD || K_LAYOUT_T == QSI_LAYOUT::PA_NZ) { // PA_BNSD(blockNum, n2, blockSize)
                keyScaleGmOffset = blkTableGm_.GetValue(blockTableBatchOffset + startBlockTableIdx + i) *
                                   constInfo_.kHeadNum * constInfo_.kCacheBlockSize +
                                   runInfo.n2Idx * constInfo_.kCacheBlockSize;
            } else { // PA_BSND(blockNum, blockSize, n2)
                keyScaleGmOffset = blkTableGm_.GetValue(blockTableBatchOffset + startBlockTableIdx + i) *
                                   constInfo_.kCacheBlockSize * constInfo_.kHeadNum + runInfo.n2Idx;
            }
            SetFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUb_[resUbBaseOffset + i * constInfo_.kCacheBlockSize], kScaleGm_[keyScaleGmOffset],
                                 dataCopymMoutParams, padParams);
        }
    } else { // B S N
        if (constInfo_.kHeadNum == 1) {
            dataCopymMoutParams.blockCount = 1;
            dataCopymMoutParams.blockLen = runInfo.actualSingleProcessSInnerSize * sizeof(float);
            dataCopymMoutParams.srcStride = 0;
        } else {
            dataCopymMoutParams.blockCount = runInfo.actualSingleProcessSInnerSize;
            dataCopymMoutParams.blockLen = 1 * sizeof(float);
            dataCopymMoutParams.srcStride = (constInfo_.kHeadNum - 1) * sizeof(float);
        }
        dataCopymMoutParams.dstStride = 0;
        dataCopymMoutParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUb_, kScaleGm_[runInfo.tensorKeyScaleOffset], dataCopymMoutParams, padParams);
    }
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::ComputeLse(const QSICommon::RunInfo &runInfo)
{
    LocalTensor<float> mmResUb = mmInUb_[mte2BufIdx_ % DOUBLE_BUFFER_NUM * MTE2_BUF_SIZE];
    int64_t nCount = QSiCeilDiv(runInfo.actualSingleProcessSInnerSize, static_cast<uint32_t>(constInfo_.sparseBlockSize));
    int64_t nRepeatTimes = QSiCeilDiv(nCount, B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
    // input n*sparseBlockSize
    CopyRepeatParams repeatParams;
    repeatParams.dstStride = 1;
    repeatParams.dstRepeatSize = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
    repeatParams.srcStride = QSiCeilDiv(static_cast<int64_t>(constInfo_.sparseBlockSize), B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
    repeatParams.srcRepeatSize = repeatParams.srcStride * (B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
    Copy(nBlkMaxUb_, mmResUb, B32_VEC_MAX_MASK, nRepeatTimes, repeatParams);
    PipeBarrier<PIPE_V>();
    for (int64_t s2BlockOffset = B32_VEC_BLK_NUM; s2BlockOffset < constInfo_.sparseBlockSize;
         s2BlockOffset += B32_VEC_BLK_NUM) {
        BinaryRepeatParams binaryRepeatParams;
        binaryRepeatParams.dstBlkStride = 1;
        binaryRepeatParams.dstRepStride = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
        binaryRepeatParams.src0BlkStride = 1;
        binaryRepeatParams.src0RepStride = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
        binaryRepeatParams.src1BlkStride = QSiCeilDiv(static_cast<int64_t>(constInfo_.sparseBlockSize), B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        binaryRepeatParams.src1RepStride = binaryRepeatParams.src1BlkStride * (B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        AscendC::Max(nBlkMaxUb_, nBlkMaxUb_, mmResUb[s2BlockOffset], B32_VEC_MAX_MASK, nRepeatTimes,
                     binaryRepeatParams);
    }
    PipeBarrier<PIPE_V>();
    BlockReduceMax(nMaxUb_, nBlkMaxUb_, nRepeatTimes, B32_VEC_MAX_MASK, 1, 1, B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);

    PipeBarrier<PIPE_V>();
    BrcbRepeatParams brcbRepeatParams;
    brcbRepeatParams.dstBlkStride = 1;
    brcbRepeatParams.dstRepStride = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
    Brcb(nMaxBrcbUb_, nMaxUb_, nRepeatTimes, brcbRepeatParams);  // n,1 -> n,8

    PipeBarrier<PIPE_V>();
    for (int64_t s2BlockOffset = 0; s2BlockOffset < constInfo_.sparseBlockSize; s2BlockOffset += B32_VEC_BLK_NUM) {
        BinaryRepeatParams binaryRepeatParams;
        binaryRepeatParams.dstBlkStride = QSiCeilDiv(static_cast<int64_t>(constInfo_.sparseBlockSize), B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        binaryRepeatParams.dstRepStride = binaryRepeatParams.dstBlkStride * (B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        binaryRepeatParams.src0BlkStride = QSiCeilDiv(static_cast<int64_t>(constInfo_.sparseBlockSize), B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        binaryRepeatParams.src0RepStride = binaryRepeatParams.dstBlkStride * (B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        binaryRepeatParams.src1BlkStride = 1;
        binaryRepeatParams.src1RepStride = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
        Sub(mmResUb[s2BlockOffset], mmResUb[s2BlockOffset], nMaxBrcbUb_, B32_VEC_MAX_MASK, nRepeatTimes,
            binaryRepeatParams);  // n,sparseBlockSize - n,8
    }

    PipeBarrier<PIPE_V>();
    Exp(mmResUb, mmResUb, runInfo.actualSingleProcessSInnerSize);  // n, sparseBlockSize

    PipeBarrier<PIPE_V>();
    // n,sparseBlockSize -> n, blk
    for (int64_t s2BlockOffset = B32_VEC_BLK_NUM; s2BlockOffset < constInfo_.sparseBlockSize;
         s2BlockOffset += B32_VEC_BLK_NUM) {
        BinaryRepeatParams binaryRepeatParams;
        binaryRepeatParams.dstBlkStride = 1;
        binaryRepeatParams.dstRepStride = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
        binaryRepeatParams.src0BlkStride = QSiCeilDiv(static_cast<int64_t>(constInfo_.sparseBlockSize), B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        binaryRepeatParams.src0RepStride = binaryRepeatParams.src0BlkStride * (B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
        binaryRepeatParams.src1BlkStride = binaryRepeatParams.src0BlkStride;
        binaryRepeatParams.src1RepStride = binaryRepeatParams.src0RepStride;
        Add(mmResUb, mmResUb[s2BlockOffset], mmResUb, B32_VEC_MAX_MASK, nRepeatTimes, binaryRepeatParams);
    }
    PipeBarrier<PIPE_V>();
    // n, blk -> n,1
    BlockReduceSum(mmResUb, mmResUb, nRepeatTimes, B32_VEC_MAX_MASK, 1, 1, B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
    PipeBarrier<PIPE_V>();
    Log(mmResUb, mmResUb, nCount);  // n,1
    Duplicate(nLseTobeSortUb_.template ReinterpretCast<int32_t>(), NEG_INF, constInfo_.s2BaseSize / constInfo_.sparseBlockSize);
    PipeBarrier<PIPE_V>();
    Add(nLseTobeSortUb_, mmResUb, nMaxUb_, nCount);  // n,1
}

/**
  src: logits和索引，前logitsNum为logits，后logitsNum为索引
  dst: src最终存放的目的地址
  logitsNum: 排序的元素个数, 暂只支持[128,256,384,512,1024,1536,2048]
 */
template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::SortAll(const LocalTensor<float> &dst, const LocalTensor<float> &src, int64_t logitsNum, int64_t mrgElements)
{
    int64_t sort32Repeats = logitsNum / mrgElements;
    AscendC::PipeBarrier<PIPE_V>();

    int64_t mrgGroups = sort32Repeats;
    int64_t i = 0;
    AscendC::LocalTensor<float> srcTensor;
    AscendC::LocalTensor<float> dstTensor;
    while (true) {
        if (i % 2 == 0) {
            srcTensor = src;
            dstTensor = dst;
        } else {
            srcTensor = dst;
            dstTensor = src;
        }
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgElements;
        params.elementLengths[MRG_QUE_1] = mrgElements;
        params.elementLengths[MRG_QUE_2] = mrgElements;
        params.elementLengths[MRG_QUE_3] = mrgElements;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = srcTensor[0];
        srcList.src2 = srcTensor[MRG_QUE_1 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src3 = srcTensor[MRG_QUE_2 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src4 = srcTensor[MRG_QUE_3 * VALUE_AND_INDEX_NUM * mrgElements];
        if (mrgGroups <= MRG_BLOCK_4) {
            params.repeatTimes = 1;
            if (mrgGroups == 1) {
                break;
            } else if (mrgGroups == MRG_BLOCK_2) {
                params.validBit = 0b0011;
            } else if (mrgGroups == MRG_BLOCK_3) {
                params.validBit = 0b0111;
            } else if (mrgGroups == MRG_BLOCK_4) {
                params.validBit = 0b1111;
            }
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            break;
        } else {
            params.repeatTimes = mrgGroups / MRG_BLOCK_4;
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            mrgElements = mrgElements * MRG_BLOCK_4;
            mrgGroups = mrgGroups / MRG_BLOCK_4;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (i % 2 == 0) {
        AscendC::DataCopy(dst, src, logitsNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::SortBasicBlockTopKToSub(const QSICommon::RunInfo &runInfo)
{
    int64_t nCount = constInfo_.s2BaseSize / constInfo_.sparseBlockSize;
    int64_t nSortRepeatTimes = QSiCeilDiv(nCount, SORT_MASK);
    int64_t s2Offset = runInfo.s2Idx * constInfo_.s2BaseSize;
    PipeBarrier<PIPE_V>();
    // 生成索引, 按照最大规格，从256之后开始生成索引
    Adds(nLseTobeSortUb_.template ReinterpretCast<int32_t>()[nCount], globalTopkIndice_,
    static_cast<int32_t>(QSiCeilDiv(s2Offset, static_cast<int64_t>(constInfo_.sparseBlockSize))), nCount);

    PipeBarrier<PIPE_V>();
    LocalTensor<float> subSortTensor = sortTensor_[((sortedIdx_ + 1) % DOUBLE_BUFFER_NUM) * SORT_BUF_SIZE + nSortCountOffset_ * VALUE_AND_INDEX_NUM];
    LocalTensor<float> subSortCacheTensor = sortTensor_[(sortedIdx_ % DOUBLE_BUFFER_NUM) * SORT_BUF_SIZE + nSortCountOffset_ * VALUE_AND_INDEX_NUM];
    // 8 * 32 or 4 * 32
    Sort32(subSortCacheTensor[nSubSortCountOffset_ * VALUE_AND_INDEX_NUM], nLseTobeSortUb_, nLseTobeSortUb_[nCount].ReinterpretCast<uint32_t>(), nSortRepeatTimes);

    // src和dst不共地址
    SortAll(subSortTensor[nSubSortCountOffset_ * VALUE_AND_INDEX_NUM],
        subSortCacheTensor[nSubSortCountOffset_ * VALUE_AND_INDEX_NUM], nCount, SORT_MASK);
    nSubSortCountOffset_+= nCount;
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::SortSubTopK(const QSICommon::RunInfo &runInfo)
{
    if (nSubSortCountOffset_ >= runInfo.targetTopKAlign || runInfo.isLastS2InnerLoop) {
        LocalTensor<float> subSortTensor = sortTensor_[((sortedIdx_ + 1) % DOUBLE_BUFFER_NUM) * SORT_BUF_SIZE + nSortCountOffset_ * VALUE_AND_INDEX_NUM];
        LocalTensor<float> sortTensor = sortTensor_[(sortedIdx_ % DOUBLE_BUFFER_NUM) * SORT_BUF_SIZE + nSortCountOffset_ * VALUE_AND_INDEX_NUM];
        // 产生4的倍数个基本块 256->1024->2048/  128->512->1024->1536->2048
        // src和dst非共地址
        SortAll(sortTensor,
                subSortTensor, runInfo.targetTopKAlign, constInfo_.s2BaseSize / constInfo_.sparseBlockSize);
        nSortCountOffset_ += nSubSortCountOffset_;
        nSubSortCountOffset_ = 0;
    }

    if (nSortCountOffset_ >= 4 * runInfo.targetTopKAlign || runInfo.isLastS2InnerLoop) {
        LocalTensor<float> nextSortTensor = sortTensor_[((sortedIdx_ + 1) % DOUBLE_BUFFER_NUM) * SORT_BUF_SIZE];
        LocalTensor<float> sortTensor = sortTensor_[(sortedIdx_ % DOUBLE_BUFFER_NUM) * SORT_BUF_SIZE];
        if (nSortCountOffset_ <= runInfo.targetTopKAlign) {
            CopyRepeatParams repeatParams;
            repeatParams.dstStride = 1;
            repeatParams.dstRepeatSize = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
            repeatParams.srcStride = 1;
            repeatParams.srcRepeatSize = repeatParams.srcStride * (B32_VEC_MAX_MASK / B32_VEC_BLK_NUM);
            Copy(nSortOutUb_, sortTensor,
                B32_VEC_MAX_MASK, (runInfo.targetTopKAlign * 2) / B32_VEC_MAX_MASK, repeatParams);
            sortedIdx_++;
            nSortCountOffset_ = runInfo.isLastS2InnerLoop ? 0 : runInfo.targetTopKAlign;
            PipeBarrier<PIPE_V>();
            Duplicate(sortTensor.template ReinterpretCast<int32_t>(), NEG_INF, SORT_BUF_SIZE);
            return;
        }
        // 只够缓存4条
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = runInfo.targetTopKAlign;
        params.elementLengths[MRG_QUE_1] = runInfo.targetTopKAlign;
        params.elementLengths[MRG_QUE_2] = runInfo.targetTopKAlign;
        params.elementLengths[MRG_QUE_3] = runInfo.targetTopKAlign;
        params.ifExhaustedSuspension = true;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = sortTensor;
        srcList.src2 = sortTensor[MRG_QUE_1 * VALUE_AND_INDEX_NUM * runInfo.targetTopKAlign];
        srcList.src3 = sortTensor[MRG_QUE_2 * VALUE_AND_INDEX_NUM * runInfo.targetTopKAlign];
        srcList.src4 = sortTensor[MRG_QUE_3 * VALUE_AND_INDEX_NUM * runInfo.targetTopKAlign];
        params.repeatTimes = 1;
        params.validBit = (1 << QSiCeilDiv(nSortCountOffset_, static_cast<int64_t>(runInfo.targetTopKAlign))) - 1;
        AscendC::MrgSort<float>(nextSortTensor, srcList, params);
        if (runInfo.isLastS2InnerLoop) {
            PipeBarrier<PIPE_V>();
            DataCopy(nSortOutUb_, nextSortTensor, runInfo.targetTopKAlign * 2);
            PipeBarrier<PIPE_V>();
            Duplicate(nextSortTensor.template ReinterpretCast<int32_t>(), NEG_INF, SORT_BUF_SIZE);
        }
        sortedIdx_ += 1;
        nSortCountOffset_ = runInfo.isLastS2InnerLoop ? 0 : runInfo.targetTopKAlign;
        PipeBarrier<PIPE_V>();
        Duplicate(sortTensor.template ReinterpretCast<int32_t>(), NEG_INF, SORT_BUF_SIZE);
        PipeBarrier<PIPE_V>();
    }
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::ExtractIndex(const LocalTensor<int32_t> &idxULocal, const LocalTensor<int32_t> &sortLocal,
                                    int64_t extractNum)
{
    AscendC::GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = QSiCeilDiv(extractNum * VALUE_AND_INDEX_NUM, B32_VEC_MAX_MASK);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = B32_VEC_MAX_MASK / B32_VEC_BLK_NUM;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2; // 固定模式2,表示筛选出奇数索引的数
    AscendC::GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::CopyOutResult(const QSICommon::RunInfo &runInfo)
{
    if (GetSubBlockIdx() == 0) {
        return;
    }
    // 1.写尾块
    PipeBarrier<PIPE_V>();
    ExtractIndex(nIdxUb_, nSortOutUb_.template ReinterpretCast<int32_t>(), runInfo.targetTopKAlign);

    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = (runInfo.targetTopK + runInfo.fixedTailCount) * sizeof(int32_t);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    SetFlag<HardEvent::V_S>(V_S_EVENT);
    WaitFlag<HardEvent::V_S>(V_S_EVENT);
    for (int i = 0; i < runInfo.fixedTailCount; i++) {
        nIdxUb_.SetValue((runInfo.targetTopK + i),
                           runInfo.needProcessS2Size / constInfo_.sparseBlockSize + i);
    }
    SetFlag<HardEvent::S_MTE3>(S_MTE3_EVENT);
    WaitFlag<HardEvent::S_MTE3>(S_MTE3_EVENT);
    AscendC::DataCopyPad(indiceOutGm_[runInfo.indiceOutOffset],nIdxUb_, dataCopyOutyParams);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::CopyOutFdResult(const QSICommon::RunInfo &runInfo)
{
    // vec1Res Gm = [aic, constInfo_.s1BaseSize, 2, 2, topkOut_] float32
    // vec1Param Gm = [aic, constInfo_.s1BaseSize, 2, 16] int64
    //     16 = [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, TopkAlign, TopK, ......]
    int64_t wsOffset = blockId_ * constInfo_.s1BaseSize * 2 * MAX_TOPK_WITH_ID;
    int64_t wsInfoOffset = blockId_ * constInfo_.s1BaseSize * 2 * paramNum_;

    LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
    SetFlag<HardEvent::MTE3_S>(MTE3_S_EVENT);
    WaitFlag<HardEvent::MTE3_S>(MTE3_S_EVENT);
    tmpiBuff.SetValue(0, static_cast<int64_t>(1));
    tmpiBuff.SetValue(1, static_cast<int64_t>(runInfo.needProcessS2Size));
    tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
    tmpiBuff.SetValue(3, static_cast<int64_t>(runInfo.s2Idx * constInfo_.s2BaseSize + runInfo.actualSingleProcessSInnerSize));
    tmpiBuff.SetValue(4, static_cast<int64_t>((runInfo.s2Idx + 1) * constInfo_.s2BaseSize >= runInfo.needProcessS2Size));
    tmpiBuff.SetValue(5, static_cast<int64_t>(runInfo.bN2Idx));
    tmpiBuff.SetValue(6, static_cast<int64_t>(runInfo.targetTopKAlign));
    tmpiBuff.SetValue(7, static_cast<int64_t>(runInfo.targetTopK));
    tmpiBuff.SetValue(8, static_cast<int64_t>(runInfo.indiceOutOffset));
    // 写入头尾判断
    // [head, tail]
    // head: 与前面规约，与前后规约
    // tail: 与后面规约
    // WS偏移规则 blockS2StartIdx_ != 0
    // 跟前面块做规约 写到0偏移 不用做计算 blockS2StartIdx_ == 0 and !isS2End
    // 跟后面块做规约 写到1偏移  需要 + constInfo_.s1BaseSize, MAX_TOPK*2
    if (blockS2StartIdx_ == 0) {  // S2不是最后结束的数据就需要往后做规约，放入第二块ws
        wsInfoOffset += paramNum_;
        wsOffset += MAX_TOPK_WITH_ID;
    }
    SetFlag<HardEvent::S_MTE3>(S_MTE3_EVENT);
    WaitFlag<HardEvent::S_MTE3>(S_MTE3_EVENT);
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = 16 * sizeof(int64_t);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(vec1ParamGm_[wsInfoOffset], tmpiBuff, dataCopyOutyParams);
    SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);

    dataCopyOutyParams.blockLen = MAX_TOPK_WITH_ID * sizeof(float);
    AscendC::DataCopyPad(vec1ResGm[wsOffset], nSortOutUb_, dataCopyOutyParams);
    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT_FD);
    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT_FD);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    Duplicate(nIdxUb_, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    SetFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(V_MTE3_EVENT);
    CopyOut(indiceOutGm_[invalidS1offset], nIdxUb_, constInfo_.sparseCount);
    SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::ProcessVec(const QSICommon::RunInfo &runInfo)
{
    if (GetSubBlockIdx() == 0) {
        return;
    }

    if (runInfo.isFirstS2InnerLoop) {
        blockS2StartIdx_ = runInfo.s2Idx;
    }

    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + mte2BufIdx_ % DOUBLE_BUFFER_NUM);
    CopyMm1ResultIn(runInfo);
    GetKeyScale(runInfo);

    AscendC::SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
    AscendC::WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
    // 将qScale从1扩展到2048个数
    AscendC::Duplicate(qScaleUb_.template ReinterpretCast<float>(), runInfo.qScale, runInfo.actualSingleProcessSInnerSize);
    // 计算反量化参数Weight = qScale * kScale
    PipeBarrier<PIPE_V>();
    AscendC::Mul(qkScaleUb_, qScaleUb_, kScaleUb_, runInfo.actualSingleProcessSInnerSize);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);

    // bmm1结果从int32转换为fp32
    PipeBarrier<PIPE_V>();
    LocalTensor<int32_t> mmInUbInt = mmInUb_.template ReinterpretCast<int32_t>()[mte2BufIdx_ % DOUBLE_BUFFER_NUM * MTE2_BUF_SIZE];
    LocalTensor<float> mmInUb = mmInUb_[mte2BufIdx_ % DOUBLE_BUFFER_NUM * MTE2_BUF_SIZE];
    AscendC::Cast(mmInUb, mmInUbInt, RoundMode::CAST_NONE, MTE2_BUF_SIZE);
    SetFlag<HardEvent::V_S>(V_S_EVENT);
    WaitFlag<HardEvent::V_S>(V_S_EVENT);
    AscendC::Mul(mmInUb, mmInUb, qkScaleUb_, runInfo.actualSingleProcessSInnerSize);
    PipeBarrier<PIPE_V>();

    ComputeLse(runInfo);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT + mte2BufIdx_ % DOUBLE_BUFFER_NUM);
    mte2BufIdx_++;

    if(runInfo.isLastS2InnerLoop) {
        WaitFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    }
    SortBasicBlockTopKToSub(runInfo);   // 基本块排满一个sub topk空间
    SortSubTopK(runInfo);               // sub空间排满后，做一次整体排序，产生topk
    if (runInfo.isLastS2InnerLoop) {
        if (blockS2StartIdx_ == 0 && (runInfo.s2Idx + 1) * constInfo_.s2BaseSize >= runInfo.needProcessS2Size) {
            CopyOutResult(runInfo);     // 最后一个循环，排序结果拷出
        } else {
            // 触发FD写出
            CopyOutFdResult(runInfo);
        }
        SetFlag<HardEvent::MTE3_V>(MTE3_V_EVENT);
    }
}

template <typename QSIT>
__aicore__ inline void QSIVector<QSIT>::ProcessLD()
{
    if (GetSubBlockIdx() == 0) {
        return;
    }
    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息
    // 开始必然没有头规约，因此从尾规约开始处理，while循环读取下一个核的头规约
    // 存满4个list或者遇到S2结尾，则做merge，直到做完S2
    // 每个核都忽略自己的头规约，因为必然由前面的核做完
    // vec1Res Gm = [aiv, constInfo_.s1BaseSize, 2, 2, topkOut_] float32
    // vec1Param Gm = [aiv, constInfo_.s1BaseSize, 2, 16] int64
    int64_t needFd = vec1ParamGm_.GetValue(blockId_ * constInfo_.s1BaseSize * 2 * paramNum_ + paramNum_);

    if (needFd == 0 || vec1ParamGm_.GetValue(blockId_ * constInfo_.s1BaseSize * 2 * paramNum_ + paramNum_ + 2) != 0) {
        return;
    }

    // 搬入数据
    int64_t wsOffsetInit = blockId_ * constInfo_.s1BaseSize * 2 * MAX_TOPK_WITH_ID + MAX_TOPK_WITH_ID;
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT_FD);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT_FD);
    SetFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
    WaitFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
    DataCopyPad(curValueIdxUb, vec1ResGm[wsOffsetInit],
                {1, static_cast<uint16_t>(MAX_TOPK_WITH_ID * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
    int64_t valueOffset = MAX_TOPK_WITH_ID;
    int64_t acc_list_num = vec1ParamGm_.GetValue(blockId_ * constInfo_.s1BaseSize * 2 * paramNum_ + paramNum_ + 6);
    // 获取下一个核规约信息
    int32_t tmpCubeId = blockId_ + 2;
    int64_t wsInfoOffsetInit = tmpCubeId * constInfo_.s1BaseSize * 2 * paramNum_;
    needFd = vec1ParamGm_.GetValue(wsInfoOffsetInit);
    int64_t isS2End = vec1ParamGm_.GetValue(wsInfoOffsetInit + 4);
    int64_t bN2Idx = vec1ParamGm_.GetValue(wsInfoOffsetInit + 5);
    int64_t targetTopKAlign = vec1ParamGm_.GetValue(wsInfoOffsetInit + 6);

    while (needFd == 1) {
        int64_t wsOffset = tmpCubeId * constInfo_.s1BaseSize * 2 * MAX_TOPK_WITH_ID;
        // 搬入头规约数据
        SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT_FD);
        WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT_FD);
        SetFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
        WaitFlag<HardEvent::S_MTE2>(S_MTE2_EVENT);
        DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(MAX_TOPK_WITH_ID * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        valueOffset += MAX_TOPK_WITH_ID;
        acc_list_num+=targetTopKAlign;
        // 每满4个list，聚合  前2K为mrg结果
        if (acc_list_num >= 4*targetTopKAlign || isS2End == 1) {
            // MrgSort 四条2048的队列，Mrg成一条
            SetFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(MTE2_V_EVENT);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = targetTopKAlign;
            params.elementLengths[1] = targetTopKAlign;
            params.elementLengths[2] = targetTopKAlign;
            params.elementLengths[3] = targetTopKAlign;
            params.ifExhaustedSuspension = true;
            params.validBit = (1 << QSiCeilDiv(acc_list_num, static_cast<int64_t>(targetTopKAlign))) - 1;
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[1 * MAX_TOPK_WITH_ID];
            srcList.src3 = curValueIdxUb[2 * MAX_TOPK_WITH_ID];
            srcList.src4 = curValueIdxUb[3 * MAX_TOPK_WITH_ID];
            MrgSort<float>(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, MAX_TOPK_WITH_ID);
            PipeBarrier<PIPE_V>();
            acc_list_num = targetTopKAlign;
            valueOffset = MAX_TOPK_WITH_ID;
        }
        // reduce到S2末尾，则跳出
        if (isS2End == 1) {
            break;
        }
        tmpCubeId+=2;
        int64_t wsInfoOffset = tmpCubeId * constInfo_.s1BaseSize * 2 * paramNum_;
        needFd = vec1ParamGm_.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm_.GetValue(wsInfoOffset + 4);
    }
    // 搬出
    // 1.写尾块
    int64_t needProcessS2Size = vec1ParamGm_.GetValue(wsInfoOffsetInit + 1);
    int64_t targetTopK = vec1ParamGm_.GetValue(wsInfoOffsetInit + 7);
    int64_t outOffset = vec1ParamGm_.GetValue(wsInfoOffsetInit + 8);
    LocalTensor<int32_t> outIdxUb = ldOutIdxBuf_.Get<int32_t>();
    ExtractIndex(outIdxUb, curValueIdxUb.template ReinterpretCast<int32_t>(), targetTopKAlign);

    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = (targetTopK + constInfo_.fixedTailCount) * sizeof(int32_t);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    SetFlag<HardEvent::V_S>(V_S_EVENT);
    WaitFlag<HardEvent::V_S>(V_S_EVENT);
    for (int i = 0; i < constInfo_.fixedTailCount; i++) {
        outIdxUb.SetValue((targetTopK + i),
                           needProcessS2Size / constInfo_.sparseBlockSize + i);
    }
    SetFlag<HardEvent::S_MTE3>(S_MTE3_EVENT);
    WaitFlag<HardEvent::S_MTE3>(S_MTE3_EVENT);
    AscendC::DataCopyPad(indiceOutGm_[outOffset], outIdxUb, dataCopyOutyParams);
}
} // namespace QSIKernel
#endif