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
 * \file sparse_flash_attention_antiquant_service_vector_mla.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_VECTOR_GQA_MSD_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_VECTOR_GQA_MSD_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_antiquant_common.h"
#include "sparse_flash_attention_antiquant_metadata.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))
#define Q_AMAX_BUF_BYTES (ConstInfo::BUFFER_SIZE_BYTE_2K + ConstInfo::BUFFER_SIZE_BYTE_256B)
#define P_AMAX_BUF_BYTES (ConstInfo::BUFFER_SIZE_BYTE_2K + ConstInfo::BUFFER_SIZE_BYTE_256B)
#define Q_AMAX_BUF_SIZE (Q_AMAX_BUF_BYTES / sizeof(T))
#define SOFTMAX_MAX_BUF_BYTES ConstInfo::BUFFER_SIZE_BYTE_512B
#define SOFTMAX_EXP_BUF_BYTES ConstInfo::BUFFER_SIZE_BYTE_512B
#define SOFTMAX_SUM_BUF_BYTES ConstInfo::BUFFER_SIZE_BYTE_512B
#define SOFTMAX_MAX_BUF_SIZE (SOFTMAX_MAX_BUF_BYTES / sizeof(T))
#define SOFTMAX_EXP_BUF_SIZE (SOFTMAX_EXP_BUF_BYTES / sizeof(T))
#define SOFTMAX_SUM_BUF_SIZE (SOFTMAX_SUM_BUF_BYTES / sizeof(T))
#define SALSV_S2BASEIZE 1024
#define SALSV_MERGESIZE 384
#define SALSV_S2BASEIZE_1 1023
#define SFACV_SOFTMAX_WITHOUT_BRC_PRELOAD
constexpr SoftmaxConfig IFA_SOFTMAX_FLASHV2_CFG = {false}; // 将isCheckTiling设置为false
template <typename SFAAT> class SFAAVectorServiceGqaMsd {
public:
    // 中间计算数据类型为float，高精度模式
    using Q_T = typename SFAAT::queryType;
    using T = float;
    using KV_T = typename SFAAT::kvType;
    using KV_ORIGIN_T = typename SFAAT::queryType;
    using OUT_T = typename SFAAT::outputType;
    using UPDATE_T = half;
    using MM1_OUT_T = half;
    using MM2_OUT_T = half;

    __aicore__ inline SFAAVectorServiceGqaMsd(){};
    __aicore__ inline void ProcessVec0Msd(const RunInfo &info);
    __aicore__ inline void ProcessVec1Msd(const RunInfo &info);
    __aicore__ inline void ProcessVec2Msd(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const SparseFlashAttentionAntiquantTilingDataMla *__restrict tilingData,
                                      SfaMetaData *metaDataPtr);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<KV_T> &keyMergeGm, const GlobalTensor<KV_T> &valueMergeGm,
                                                const GlobalTensor<KV_T> &queryPreProcessResGm, const GlobalTensor<Q_T> &queryGm,
                                                const GlobalTensor<KV_T> &keyGm, const GlobalTensor<KV_T> &valueGm,
                                                const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<T> &keyDequantScaleGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm);
    __aicore__ inline void InitVec2GlobalTensor(const GlobalTensor<T> &valueDequantScaleGm, GlobalTensor<T> accumOutGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();

private:
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMaxForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb, uint32_t dealRowCount,
                                                    uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb, uint32_t dealRowCount, uint32_t columnCount,
                                  uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline void QueryPreProcess(const RunInfo &info);
    __aicore__ inline void DealQueryPreProcessBaseBlock(const RunInfo &info, uint32_t startRow, uint32_t dealRowCount,
                                                        uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiqQuery(LocalTensor<T> &queryCastUb, uint64_t qOffset, uint32_t dealRowCount,
                                          uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMatmulPreProcess(const RunInfo &info, GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb,
                                                     LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb, uint32_t startRow,
                                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb, LocalTensor<T> tmpAUb,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void AntiquantAIterExpand(GlobalTensor<KV_T> &dstGm, LocalTensor<half> &tmpA1, LocalTensor<half> &tmpA2,
                                                uint32_t calcSize, bool isFirst, uint64_t outOffset);
    __aicore__ inline int64_t MergeKv(const RunInfo &runInfo, int64_t s2GmStartOffset, int64_t s2GmLimit,
                                    int64_t topkGmBaseOffset, bool isValue, int64_t mergeMte3Idx, int64_t s2GmStartOffset4Merge);
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo, bool &mask,
                                    int64_t s2GmOffsetArray, int64_t s2GmLimit);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo, bool isValue, bool &needWaitMte3ToMte2);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetInfInBlkHasTail(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                              uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo, bool &mask,
                                          int64_t &s2GmOffsetArray, int64_t s2GmLimit);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount,
                                                uint32_t loopId, bool &needMask, uint32_t &maskStart, uint32_t &maskEnd);
    __aicore__ inline void AntiquantMatmulResCombineDD(const RunInfo &info, LocalTensor<T> bmmResUb, GlobalTensor<MM1_OUT_T> srcGm,
                                                       uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount,
                                                       float scaleC);
    __aicore__ inline void AntiquantSoftmaxResPreProcess(const RunInfo &info, GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb,
                                                         LocalTensor<T> tmpAFloorUb, uint32_t startRow,
                                                         uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void AttentionMaskCompute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t startRow, bool &needMask,
                                                uint32_t &maskStart, uint32_t &maskEnd);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void AntiquantMM2ResCombine(const RunInfo &info, LocalTensor<MM2_OUT_T> bmmResUb,
                                                  GlobalTensor<MM2_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
                                                  uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void CopyAntiquantScale(LocalTensor<T> &castUb, GlobalTensor<T> srcGm, uint64_t offset);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    static constexpr bool PAGE_ATTENTION = SFAAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAAT::templateMode;
    static constexpr bool IS_META = SFAAT::flashDecode;
    bool FLASH_DECODE = SFAAT::flashDecode;
    static constexpr SFAA_LAYOUT LAYOUT_T = SFAAT::layout;
    static constexpr SFAA_LAYOUT KV_LAYOUT_T = SFAAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 7;
    static constexpr uint64_t SYNC_INPUT_BUF2_PONG_FLAG = 8;
    static constexpr uint64_t SYNC_INPUT_DEQUANT_SCALE_FLAG = 6;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_8K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8192
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;
    static constexpr T antiqCoeff1 = 127;
    static constexpr T antiqCoeff2 = 1 / antiqCoeff1;
    static constexpr float scaleC1 = 1024.0 / 127;
    static constexpr float scaleC2 = 1024.0;
    MM1_OUT_T antiquantExpandCoeff = 254;
    static constexpr uint32_t msdIterNum = 2;

    const SparseFlashAttentionAntiquantTilingDataMla *__restrict tilingData;
    SfaMetaData *metaDataPtr;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};
    uint32_t lastBN2Idx = 1 << 31;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<KV_T> keyMergeGm_;
    GlobalTensor<KV_T> valueMergeGm_;
    GlobalTensor<KV_T> queryPreProcessResGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<KV_T> valueGm_;
    GlobalTensor<T> keyDequantScaleGm_;
    GlobalTensor<T> valueDequantScaleGm_;
    GlobalTensor<int32_t> topkGm_;

    // ================================Local Buffer区====================================

    // queue
    TBuf<> inputBuf1; // 32K, inque
    TBuf<> inputBuf2; // 16K, inque
    TBuf<> outputBuf1; // 32K, outque
    TBuf<> outputBuf2; // 8K, outque

    // 临时tbuf
    TBuf<> tmpBuff1; // 32K
    TBuf<> tmpBuff2; // 32K
    TBuf<> tmpBuff3; // 2K

    // 常驻tbuf
    TBuf<> vec2ResBuff; // 16k 伪量化场景
    TBuf<> antiqKeyScaleBuff; // 2K
    TBuf<> antiqValueScaleBuff; // 2K
    TBuf<> qAmaxBuff; // constInfo.preLoadNum * (2K + 256B)
    TBuf<> softmaxResAmaxBuff; // 2K + 256B
    TBuf<> qRowSumBuff; // 2K + 256B
    TBuf<> softmaxResRowSumBuff; // 2K + 256B
    TBuf<> softmaxMaxBuff; // constInfo.preLoadNum * 2K
    TBuf<> softmaxExpBuff; // constInfo.preLoadNum * 2K
    TBuf<> softmaxSumBuff; // constInfo.preLoadNum * 2K
    TBuf<> softmaxMaxDefaultBuff; // 2K
    TBuf<> softmaxSumDefaultBuff; // 2K

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<KV_T> kvMergUb_;

    // antiquant msd
    LocalTensor<T> aMaxBmm1Ub;
    LocalTensor<T> aMaxBmm2Ub;
    LocalTensor<T> softmaxScaleResRowSumUb;
    LocalTensor<half> vec2ResUb;
    LocalTensor<T> antiquantKScaleUb_;
    LocalTensor<T> antiquantVScaleUb_;
    LocalTensor<T> qRowSumUb;
};

__aicore__ inline uint32_t GetMinPowerTwo(uint32_t cap)
{
    uint32_t i = 1;
    while (i < cap) {
        i = i << 1;
    }
    return i;
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::InitBuffers(TPipe *pipe)
{
    // 182.5k
    pipe->InitBuffer(inputBuf1, ConstInfo::BUFFER_SIZE_BYTE_16K * 2); // 32K // mm1 mm2 res
    pipe->InitBuffer(inputBuf2, ConstInfo::BUFFER_SIZE_BYTE_8K * 2); // 16K // kvmerge
    pipe->InitBuffer(outputBuf1, ConstInfo::BUFFER_SIZE_BYTE_16K); // 16K // attentionOut
    pipe->InitBuffer(vec2ResBuff, ConstInfo::BUFFER_SIZE_BYTE_16K); // 16K // flashupdate常驻ub
    pipe->InitBuffer(outputBuf2, ConstInfo::BUFFER_SIZE_BYTE_8K); // 8K

    // tmpBuff
    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K); // 32K
    pipe->InitBuffer(tmpBuff2, ConstInfo::BUFFER_SIZE_BYTE_32K); // 32K
    pipe->InitBuffer(tmpBuff3, ConstInfo::BUFFER_SIZE_BYTE_2K); // 2K

    // 常驻buffer
    pipe->InitBuffer(antiqKeyScaleBuff, ConstInfo::BUFFER_SIZE_BYTE_2K); // 2K // 实际0.5K
    pipe->InitBuffer(antiqValueScaleBuff, ConstInfo::BUFFER_SIZE_BYTE_2K); // 2K
    // 预留空间2K = 64 * 32，支持 gSize = 64
    // brcb 操作每次操作8*32字节输出，startRow接近64时，
    // 输出最多可能超出2k空间7*32字节， 这里预留256B防止越界
    pipe->InitBuffer(qAmaxBuff, Q_AMAX_BUF_BYTES * constInfo.preLoadNum); // 2.25*2 = 4.5K
    pipe->InitBuffer(softmaxMaxBuff, SOFTMAX_MAX_BUF_BYTES * constInfo.preLoadNum); // 0.5*2 = 1K
    pipe->InitBuffer(softmaxExpBuff, SOFTMAX_EXP_BUF_BYTES * constInfo.preLoadNum); // 0.5*2 = 1K
    pipe->InitBuffer(softmaxSumBuff, SOFTMAX_SUM_BUF_BYTES * constInfo.preLoadNum); // 0.5*2 = 1K

    pipe->InitBuffer(softmaxMaxDefaultBuff, SOFTMAX_MAX_BUF_BYTES); // 0.5K
    pipe->InitBuffer(softmaxSumDefaultBuff, SOFTMAX_SUM_BUF_BYTES); // 0.5K

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuf2.Get<KV_T>();

    antiquantKScaleUb_ = antiqKeyScaleBuff.Get<T>();
    antiquantVScaleUb_ = antiqValueScaleBuff.Get<T>();
    vec2ResUb = vec2ResBuff.Get<half>();
    aMaxBmm1Ub = qAmaxBuff.Get<T>();
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqaMsd<SFAAT>::InitParams(const struct ConstInfo &constInfo,
                                     const SparseFlashAttentionAntiquantTilingDataMla *__restrict tilingData,
                                     SfaMetaData *metaDataPtr)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    this->metaDataPtr = metaDataPtr;
    if constexpr (IS_META) {
        FLASH_DECODE = metaDataPtr->numOfFdHead > 0U;
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::InitVec0GlobalTensor(
    const GlobalTensor<KV_T> &keyMergeGm, const GlobalTensor<KV_T> &valueMergeGm,
    const GlobalTensor<KV_T> &queryPreProcessResGm, const GlobalTensor<Q_T> &queryGm, const GlobalTensor<KV_T> &keyGm, const GlobalTensor<KV_T> &valueGm,
    const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<T> &keyDequantScaleGm)
{
    this->keyMergeGm_ = keyMergeGm;
    this->valueMergeGm_ = valueMergeGm;
    this->queryPreProcessResGm_ = queryPreProcessResGm;
    this->queryGm_ = queryGm;
    this->keyGm_ = keyGm;
    this->valueGm_ = valueGm;
    this->blkTableGm_ = blkTableGm;
    this->keyDequantScaleGm_ = keyDequantScaleGm;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::InitVec2GlobalTensor(const GlobalTensor<T> &valueDequantScaleGm,
                                                                            GlobalTensor<T> accumOutGm,
                                                                            GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                            GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
    this->valueDequantScaleGm_ = valueDequantScaleGm;
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET);
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bN2Idx, info.gS1Idx);
    uint64_t offset = (accumTmpOutNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> tmp = outputBuf2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuf2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ElewiseCompute(const RunInfo &info,
                                                                const LocalTensor<T> &mmResUb,
                                                                uint32_t dealRowCount, uint32_t columnCount)
{
}
template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AttentionMaskCompute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                            const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                            uint32_t columnCount, uint32_t startRow, bool &needMask,
                                            uint32_t &maskStart, uint32_t &maskEnd)
{
    uint32_t maskEndCurrentPart = (maskEnd > info.nextS2BaseOffset) ?
        info.nextS2BaseOffset : maskEnd;
    uint32_t maskStartCurrentPart = (maskStart < info.curS2BaseOffset) ?
        info.curS2BaseOffset : maskStart;
    if (!(maskStartCurrentPart < info.actS2Size && needMask) || maskStartCurrentPart >= maskEndCurrentPart) {
        return;
    }
    if (info.nextS2BaseOffset <= maskStartCurrentPart) {
        return;
    }
    uint32_t maskStartSinglePro = (maskStartCurrentPart & SALSV_S2BASEIZE_1);
    uint32_t maskEndSinglePro = (maskEndCurrentPart & SALSV_S2BASEIZE_1)  == 0 ? SALSV_S2BASEIZE : (maskEndCurrentPart & SALSV_S2BASEIZE_1);
    uint32_t gS1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint32_t s1StartIdx = gS1StartIdx / constInfo.gSize;
    uint32_t gStartIdx = gS1StartIdx % constInfo.gSize;
    uint32_t gS1EndIdx = gS1StartIdx + dealRowCount - 1;
    uint32_t s1EndIdx = gS1EndIdx / constInfo.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t headGCount = s1Count > 1 ? (constInfo.gSize - gStartIdx) : dealRowCount;
    uint32_t dstMaskOffset = 0;
    int64_t s2StartCeilAlign = SFAAAlign(maskStartSinglePro, 8);
    int64_t s2MidFloorAlign = maskEndSinglePro / 8 * 8;
    uint64_t maskEndTmp = s2StartCeilAlign >= maskEndSinglePro ? maskEndSinglePro : s2StartCeilAlign;
    uint64_t maskStartTmp = s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign;
    SetInfInBlkHasTail(mmResUb, headGCount, columnCount, maskStartSinglePro, maskEndTmp);
    SetMidInf(mmResUb, headGCount, columnCount, s2StartCeilAlign, s2MidFloorAlign);
    SetInfInBlkHasTail(mmResUb, headGCount, columnCount, maskStartTmp, maskEndSinglePro);
    if ((headGCount < dealRowCount) || ((headGCount == dealRowCount) && (((gS1StartIdx + headGCount) % (constInfo.gSize)) == 0))) {
        maskStart++;
        maskStartCurrentPart = (maskStart < info.curS2BaseOffset) ?
            info.curS2BaseOffset : maskStart;
        if (!(maskStartCurrentPart < info.actS2Size) || maskStartCurrentPart >= maskEndCurrentPart) {
            return;
        }
        if (info.nextS2BaseOffset <= maskStartCurrentPart) {
            return;
        }
        maskStartSinglePro = (maskStartCurrentPart - info.curS2BaseOffset);
    }
    dstMaskOffset += headGCount * columnCount;
    uint32_t remainRowCount = dealRowCount - headGCount;
    uint32_t midS1Count = remainRowCount / constInfo.gSize;
    uint32_t tailGSize = remainRowCount % constInfo.gSize;
    for (uint32_t midIdx = 0; midIdx < midS1Count; ++midIdx) {
        s2StartCeilAlign = SFAAAlign(maskStartSinglePro, 8);
        s2MidFloorAlign = maskEndSinglePro / 8 * 8;
        maskEndTmp = s2StartCeilAlign >= maskEndSinglePro ? maskEndSinglePro : s2StartCeilAlign;
        maskStartTmp = s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign;
        SetInfInBlkHasTail(mmResUb[dstMaskOffset], constInfo.gSize, columnCount, maskStartSinglePro, maskEndTmp);
        SetMidInf(mmResUb[dstMaskOffset], constInfo.gSize, columnCount, s2StartCeilAlign, s2MidFloorAlign);
        SetInfInBlkHasTail(mmResUb[dstMaskOffset], constInfo.gSize, columnCount, maskStartTmp, maskEndSinglePro);
        dstMaskOffset += constInfo.gSize * columnCount;
        maskStart++;
        maskStartCurrentPart = (maskStart < info.curS2BaseOffset) ?
            info.curS2BaseOffset : maskStart;
        if (!(maskStartCurrentPart < info.actS2Size) || maskStartCurrentPart >= maskEndCurrentPart) {
            return;
        }
        if (info.nextS2BaseOffset <= maskStartCurrentPart) {
            return;
        }
        maskStartSinglePro = (maskStartCurrentPart & SALSV_S2BASEIZE_1);
    }
    if (tailGSize > 0) {
        s2StartCeilAlign = SFAAAlign(maskStartSinglePro, 8);
        s2MidFloorAlign = maskEndSinglePro / 8 * 8;
        maskEndTmp = s2StartCeilAlign >= maskEndSinglePro ? maskEndSinglePro : s2StartCeilAlign;
        maskStartTmp = s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign;

        SetInfInBlkHasTail(mmResUb[dstMaskOffset], tailGSize, columnCount, maskStartSinglePro, maskEndTmp);
        SetMidInf(mmResUb[dstMaskOffset], tailGSize, columnCount, s2StartCeilAlign, s2MidFloorAlign);
        SetInfInBlkHasTail(mmResUb[dstMaskOffset], tailGSize, columnCount, maskStartTmp, maskEndSinglePro);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }

    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = (1ULL << (startId - startFloorAlignSize)) - 1;
    uint64_t notComputePostMaskOneBlk = ~((1ULL << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;

    uint64_t maskOneBlk = (~notComputeMaskOneBlk) & 0xFFFFFFFFULL; 
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId += 8) {
        Duplicate(mmResUb[rowId * columnCount + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
                  1, SfaaCeilDiv(columnCount, 8), 0);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::SetInfInBlkHasTail(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                                                       uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = ((1ULL << (startId - startFloorAlignSize)) - 1);
    uint64_t notComputePostMaskOneBlk = ~((1ULL << (endId - startFloorAlignSize)) - 1); 
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;
    uint64_t maskOneBlk = (~notComputeMaskOneBlk) & 0xFFFFFFFFULL; 
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; ++i) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    uint32_t rowLoop = dealRowCount / 8;
    uint32_t rowTail = dealRowCount % 8;
    for (uint64_t rowId = 0; rowId < rowLoop; rowId ++) {
        Duplicate(mmResUb[rowId * columnCount * 8 + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
            1, SfaaCeilDiv(columnCount, 8), 0);
    }

    if (rowTail > 0) {
        mask[0] = maskOneBlk;
        for (int i = 1; i < rowTail; ++i) {
            mask[0] = mask[0] | (maskOneBlk << (i * 8));
        }
        Duplicate(mmResUb[rowLoop * columnCount * 8 + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
            1, SfaaCeilDiv(columnCount, 8), 0);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::SetMidInf(const LocalTensor<T> &mmResUb,
                                                           uint32_t dealRowCount, uint32_t columnCount,
                                                           uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId++) {
        Duplicate(mmResUb[rowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
#ifdef SFACV_SOFTMAX_WITHOUT_BRC_PRELOAD
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
#else
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow * BLOCK_ELEMENT_NUM;
#endif
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
#ifdef SFACV_SOFTMAX_WITHOUT_BRC_PRELOAD
        SoftmaxFlashV2<T, true, true, false, false, SFAA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
#else
        SoftmaxFlashV2<T, true, true, false, false, IFA_SOFTMAX_FLASHV2_CFG>(
            mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
            softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
#endif
    } else {
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCount);
        pipe_barrier(PIPE_V);
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCount);
    }
}

// 由于S在GM上已完成atomatic，此处只需要把数据拷贝出来有fp16转成fp32
template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AntiquantMatmulResCombineDD(const RunInfo &info,
    LocalTensor<T> bmmResUb, GlobalTensor<MM1_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount, float scaleC)
{
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;

    LocalTensor<MM1_OUT_T> tmpMMRes = inputBuf1.Get<MM1_OUT_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(tmpMMRes, srcGm[baseOffset], copySize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    Cast(bmmResUb, tmpMMRes, AscendC::RoundMode::CAST_NONE, copySize);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    Muls(bmmResUb, bmmResUb, scaleC, copySize);
    PipeBarrier<PIPE_V>();
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount, uint32_t loopId, bool &needMask, uint32_t &maskStart, uint32_t &maskEnd)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) * columnCount;
    uint64_t outGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize * 2 +
                           (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) * columnCount;
    LocalTensor<T> mmResUb = tmpBuff1.Get<T>();
    AntiquantMatmulResCombineDD(info, mmResUb, mm1ResGm[inGmOffset], startRow, dealRowCount, columnCount,
        actualColumnCount, scaleC1 * static_cast<T>(tilingData->baseParams.scaleValue));
    LocalTensor<T> aMax = aMaxBmm1Ub[(info.bN2Idx % constInfo.preLoadNum) * Q_AMAX_BUF_SIZE + startRow * BLOCK_ELEMENT_NUM];
    RowMuls<T>(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    if (loopId == 0) {
        maskEnd = maskStart + constInfo.sparseShardSize - 1;
        maskStart += mSplitInfo.vecStartM / constInfo.gSize;
    }
    ElewiseCompute(info, mmResUb, dealRowCount, columnCount);
    pipe_barrier(PIPE_V);
    AttentionMaskCompute(info, mSplitInfo, mmResUb, dealRowCount, columnCount,
        startRow, needMask, maskStart, maskEnd);
    pipe_barrier(PIPE_V);
    LocalTensor<T> tmpAFloorUb = tmpBuff2.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();
    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount,
                          columnCount, actualColumnCount);
    pipe_barrier(PIPE_V);
    AntiquantSoftmaxResPreProcess(info, vec1ResGm[outGmOffset], mmResUb, tmpAFloorUb, startRow, dealRowCount, columnCount,
                                  actualColumnCount);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AntiquantSoftmaxResPreProcess(const RunInfo &info,
    GlobalTensor<KV_T> dstGm, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = info.mSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    Muls(srcUb, srcUb, antiqCoeff1, calcSize); // 127 * A
    PipeBarrier<PIPE_V>();

    Cast(tmpAFloorUb, srcUb, RoundMode::CAST_ROUND, calcSize); // fp32
    PipeBarrier<PIPE_V>();
    LocalTensor<half> tmpAFloorUbFp16 = tmpAFloorUb.template ReinterpretCast<half>();
    tmpAFloorUbFp16.SetSize(tmpAFloorUb.GetSize());
    Cast(tmpAFloorUbFp16, tmpAFloorUb, RoundMode::CAST_ROUND, calcSize); // A1:fp16
    PipeBarrier<PIPE_V>();
    // step5: 将Qtmp转成fp16
    LocalTensor<half> srcUbFp16 = srcUb.template ReinterpretCast<half>();
    srcUbFp16.SetSize(srcUb.GetSize());
    Cast(srcUbFp16, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUbFp16, tmpAFloorUbFp16, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                            const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
#ifdef SFACV_SOFTMAX_WITHOUT_BRC_PRELOAD
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;
#endif

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    uint32_t maskStart = info.maskStart;
    uint32_t maskEnd = info.maskEnd;
    bool needMask = (constInfo.sparseMode == 3);
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize,
            info.actualSingleProcessSInnerSizeAlign, info.actualSingleProcessSInnerSize, i, needMask, maskStart, maskEnd);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + runInfo.s2Idx * SALSV_S2BASEIZE) / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= runInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2Idx * SALSV_S2BASEIZE) % constInfo.sparseBlockSize);
}

template <typename SFAAT>
__aicore__ inline int64_t SFAAVectorServiceGqaMsd<SFAAT>::GetKeyBNBOffset(int64_t realS2Idx,
                                                                    const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset = 0;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_BSND) {
            realKeyBNBOffset = (blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo.kvCacheBlockSize) + blkTableOffset) *
                                static_cast<int64_t>(constInfo.kvHeadNum) + runInfo.n2Idx;
        } else if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_BNSD) { // PA_BNSD (blockNum n2 blockSize)
            realKeyBNBOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                               static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.kvCacheBlockSize) +
                               + runInfo.n2Idx * static_cast<int64_t>(constInfo.kvCacheBlockSize) + blkTableOffset;
        }
    } else {
        realKeyBNBOffset = runInfo.tensorBOffset / constInfo.headDim + realS2Idx * static_cast<int64_t>(constInfo.kvHeadNum);
    }

    return realKeyBNBOffset;
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqaMsd<SFAAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                               int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo, bool &mask,
                                               int64_t &s2GmOffsetArray, int64_t s2GmLimit)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t sparseBlockRealSize = (s2GmLimit - s2GmOffsetArray < constInfo.sparseBlockSize) ? 
                                  (s2GmLimit - s2GmOffsetArray) : constInfo.sparseBlockSize;
    int64_t validS2Count =
        (realS2Idx + sparseBlockRealSize > s2IdLimit ? s2IdLimit - realS2Idx : sparseBlockRealSize);
    // 当前仅支持SEPARATE模式
    if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::SEPARATE) {
        DataCopyExtParams intriParams;
        DataCopyPadExtParams<KV_T> padParams;
        if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ) {
            uint32_t copyFinishRowCnt = 0;
            uint32_t blockElementCnt = 32 / sizeof(KV_T);
            uint32_t ubOffset = mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * blockElementCnt;
            uint32_t kvMergUbRowSize = ConstInfo::BUFFER_SIZE_BYTE_4K / constInfo.headDim;
            while (copyFinishRowCnt < validS2Count) {
                uint64_t blockIdOffset = realS2Idx / constInfo.kvCacheBlockSize; // 获取block table上的索引
                uint64_t reaminRowCnt = realS2Idx % constInfo.kvCacheBlockSize; // 获取在单个块上超出的行数
                uint64_t idInBlockTable = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blockIdOffset); // 从block table上获取编号
                // 计算可以拷贝的行数
                uint32_t copyRowCnt = constInfo.kvCacheBlockSize - reaminRowCnt;
                if (copyFinishRowCnt + copyRowCnt > validS2Count) {
                    copyRowCnt = validS2Count - copyFinishRowCnt;
                }
                uint64_t keyOffset = idInBlockTable * constInfo.kvCacheBlockSize * constInfo.headDim * constInfo.kvHeadNum;
                keyOffset += (uint64_t)(runInfo.n2Idx * constInfo.headDim * constInfo.kvCacheBlockSize) + reaminRowCnt * blockElementCnt;
                intriParams.blockLen = copyRowCnt * blockElementCnt * sizeof(KV_T);
                intriParams.blockCount = constInfo.headDim / blockElementCnt;
                intriParams.dstStride = kvMergUbRowSize - copyRowCnt;
                intriParams.srcStride = (constInfo.kvCacheBlockSize - copyRowCnt) * blockElementCnt * sizeof(KV_T);
                padParams.isPad = false;
                DataCopyPad(kvMergUb_[ubOffset + copyFinishRowCnt * blockElementCnt], keyGm_[keyOffset], intriParams, padParams);
                DataCopyPad(kvMergUb_[ubOffset + ConstInfo::BUFFER_SIZE_BYTE_4K + copyFinishRowCnt * blockElementCnt],
                            valueGm_[keyOffset], intriParams, padParams);
                // 更新循环变量
                copyFinishRowCnt += copyRowCnt;
                realS2Idx += copyRowCnt;
            }
        } else {
            uint32_t ubOffset = mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * constInfo.headDim;
            uint16_t srcStride = 0;
            if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_BNSD) {
                intriParams.blockCount = 1;
                intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
            } else if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_BSND) {
                srcStride = (constInfo.kvHeadNum - 1) * constInfo.headDim * sizeof(KV_T);
                if (unlikely(srcStride == 0)) {
                    intriParams.blockCount = 1;
                    intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
                } else {
                    intriParams.blockCount = validS2Count;
                    intriParams.blockLen = constInfo.headDim * sizeof(KV_T); 
                }
            }
            intriParams.dstStride = 0;
            intriParams.srcStride = srcStride;
            padParams.isPad = false;
            DataCopyPad(kvMergUb_[ubOffset], keyGm_[keyBNBOffset * constInfo.headDim], intriParams, padParams);
            DataCopyPad(kvMergUb_[ubOffset + ConstInfo::BUFFER_SIZE_BYTE_4K], valueGm_[keyBNBOffset * constInfo.headDim], intriParams, padParams);
        }
    }
    mte2Size += validS2Count;
    s2GmOffsetArray += validS2Count;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                                int64_t realS2Idx1, int64_t realS2Idx2, const RunInfo &runInfo,
                                                                bool &mask, int64_t s2GmOffsetArray, int64_t s2GmLimit)
{
    int64_t s2IdLimit = runInfo.threshold;

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t blkTableSrcStride = (keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) :
        (keyBNBOffset2 - keyBNBOffset1)) - constInfo.sparseBlockSize;
    int64_t keySrcStride = blkTableSrcStride * constInfo.headDim;
    if (likely(KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ || constInfo.kvHeadNum > 1 || keySrcStride >= INT32_MAX || keySrcStride < 0 ||
        realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit || realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, runInfo, mask, s2GmOffsetArray, s2GmLimit);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, runInfo, mask, s2GmOffsetArray, s2GmLimit);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持SEPARATE模式
        if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::SEPARATE) {
            intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDim;
            uint32_t headDimAlign = SFAAAlign(constInfo.headDim, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = headDimAlign - constInfo.headDim;
            padParams.paddingValue = 0;
            uint32_t ubOffset = mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * headDimAlign;
            DataCopyPad(kvMergUb_[ubOffset], keyGm_[startGmOffset * constInfo.headDim], intriParams, padParams);
            DataCopyPad(kvMergUb_[ubOffset + ConstInfo::BUFFER_SIZE_BYTE_4K], valueGm_[startGmOffset * constInfo.headDim], intriParams, padParams);
        }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo.sparseBlockSize;
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &runInfo, bool isValue, bool &needWaitMte3ToMte2)
{
    if (mte2Size <= mte3Size) {
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG + mergeMte3Idx % 2);
        return;
    }
    int32_t dealRow = mte2Size - mte3Size;
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    
    SetFlag<AscendC::HardEvent::MTE2_MTE3>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_MTE3>(SYNC_INPUT_BUF2_FLAG);
    DataCopyExtParams dataCopyParams;
    if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ) {
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        dataCopyParams.blockCount = constInfo.headDim / blockElementCnt;
        dataCopyParams.blockLen = 32 * blockElementCnt * sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = (SALSV_MERGESIZE - 32) * blockElementCnt * sizeof(KV_T);
        DataCopyPad(valueMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * SALSV_MERGESIZE * constInfo.headDim +
            (s2GmStartOffset + mte3Size) * blockElementCnt], srcTensor[ConstInfo::BUFFER_SIZE_BYTE_4K], dataCopyParams);
        DataCopyPad(keyMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * SALSV_MERGESIZE * constInfo.headDim +
            (s2GmStartOffset + mte3Size) * blockElementCnt], srcTensor, dataCopyParams);
        
    } else {
        dataCopyParams.blockCount = dealRow;
        dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPad(keyMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * SALSV_MERGESIZE * constInfo.headDim + (s2GmStartOffset + mte3Size) *
            constInfo.headDim], srcTensor, dataCopyParams);
        DataCopyPad(valueMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * SALSV_MERGESIZE * constInfo.headDim + (s2GmStartOffset + mte3Size) *
            constInfo.headDim], srcTensor[ConstInfo::BUFFER_SIZE_BYTE_4K], dataCopyParams);
    }
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG + mergeMte3Idx % 2);
    needWaitMte3ToMte2 = true;
}

// b s1 k
template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ProcessVec0Msd(const RunInfo &info)
{
    QueryPreProcess(info);
    if (info.aivS2AccessSize == 0) {
        return;
    }
    int64_t s2ProcessSize = info.aivS2AccessSize;
    int64_t s2Pair = SfaaCeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = info.topkGmBaseOffset;

    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 
        info.aicS2AccessSize : (s2Pair / 2L) * 2 * constInfo.sparseBlockSize + info.aicS2AccessSize;
    int64_t s2GmStartOffset4Merge = GetSubBlockIdx() == 0 ? 
        0 : (s2Pair / 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? (s2Pair / 2L) * 2 * constInfo.sparseBlockSize + info.aicS2AccessSize :
        s2ProcessSize + info.aicS2AccessSize;
    if (s2GmLimit > s2ProcessSize + info.aicS2AccessSize) {
        s2GmLimit = s2ProcessSize + info.aicS2AccessSize;
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = MergeKv(info, s2GmStartOffset, s2GmLimit, topkGmBaseOffset, false, mergeMte3Idx, s2GmStartOffset4Merge);
    return;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::QueryPreProcess(const RunInfo &info)
{
    if (!info.isFirstSInnerLoop) {
        return;
    }
    if (info.bN2Idx != lastBN2Idx) { // 搬入scaleK
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = constInfo.headDim * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
        uint32_t srcOffset = info.n2Idx * constInfo.headDim;
        DataCopyPad(antiquantKScaleUb_, keyDequantScaleGm_[srcOffset], dataCopyParams, padParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    }
    
    uint32_t headDimAlign = SFAAAlign(constInfo.headDim, ConstInfo::BUFFER_SIZE_BYTE_32B); // headDim对齐32B
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / headDimAlign; // 单次处理query矩阵的行数（8192/128=64）
    if (mSplitSize > info.mSizeV) {
        mSplitSize = info.mSizeV; // 20
    }
    uint32_t loopCount = (info.mSizeV + mSplitSize - 1) / mSplitSize; // 单个vec核Q预处理的循环次数 // 1
    uint32_t tailSplitSize = info.mSizeV - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealQueryPreProcessBaseBlock(info, i * mSplitSize, dealSize, headDimAlign, constInfo.headDim);
    }

    if (info.bN2Idx != lastBN2Idx) {
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
        lastBN2Idx = info.bN2Idx;
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::DealQueryPreProcessBaseBlock(const RunInfo &info,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t bufferId = info.bN2Idx % constInfo.preLoadNum;
    LocalTensor<T> queryUb = tmpBuff1.Get<T>();
    LocalTensor<T> aFloorUb = tmpBuff2.Get<T>();
    uint64_t qOffset = info.tensorAOffset + (info.mSizeVStart + startRow) * actualColumnCount;

    if (constInfo.kvHeadNum != 1) {
        uint32_t startS1Idx = (info.mSizeVStart + startRow) / constInfo.gSize; // V0: 0 V1: 1
        uint32_t startGOfffset = (info.mSizeVStart + startRow) % constInfo.gSize; // V0: 0 V1: 0
        uint32_t endS1Idx = (info.mSizeVStart + startRow + dealRowCount - 1) / constInfo.gSize; // V0: 0 V1: 1
        uint32_t curStartRow = (info.mSizeVStart + startRow); // V0: 0 V1: 2
        uint32_t curDealRowCount = 0;
        uint32_t ubOffset = 0;
        for (uint32_t curS1idx = startS1Idx; curS1idx <= endS1Idx; curS1idx++) {
            qOffset = info.tensorAOffset + curS1idx * constInfo.qHeadNum * constInfo.headDim +
                startGOfffset * constInfo.headDim;
            if (curS1idx != endS1Idx) {
                curDealRowCount = (curS1idx + 1) * constInfo.gSize - curStartRow;
            } else {
                curDealRowCount = info.mSizeVStart + startRow + dealRowCount - curStartRow;
            }
            ubOffset = (curStartRow - info.mSizeVStart) * columnCount;
            LocalTensor<T> curQueryUb = queryUb[ubOffset];
            CopyAntiqQuery(curQueryUb, qOffset, curDealRowCount, columnCount, actualColumnCount);
            PipeBarrier<PIPE_V>();
            curStartRow += curDealRowCount;
            startGOfffset = 0;
        }
    } else {
        CopyAntiqQuery(queryUb, qOffset, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
    }
    // mul scale
    VecMulMat(queryUb, antiquantKScaleUb_, queryUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // A pre process
    size_t dstOffset = bufferId * constInfo.bmm2ResUbSize * 2 + info.mSizeVStart * columnCount;

    AntiquantMatmulPreProcess(info, queryPreProcessResGm_[dstOffset], aMaxBmm1Ub[bufferId * Q_AMAX_BUF_SIZE],
        queryUb, aFloorUb, startRow, dealRowCount, columnCount, actualColumnCount);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::VecMulMat(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // vec mul by row
    // dstUb[i, j] = src0Ub[j] * src1Ub[i, j],
    // src0Ub:[1, columnCount] src1Ub:[dealRowCount, actualColumnCount] dstUb:[dealRowCount, columnCount]
    if (columnCount < REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) { // dstRepStride为0~255,columnCount需要小于2048
        BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        repeatParams.src0RepStride = 0;
        repeatParams.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
        uint32_t mask = FP32_REPEAT_ELEMENT_NUM;
        uint32_t loopCount = actualColumnCount / mask;
        uint32_t remainCount = actualColumnCount % mask;
        uint32_t offset = 0;
        for (int i = 0; i < loopCount; i++) {
            // offset = i * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
            offset += mask;
        }
        if (remainCount > 0) {
            // offset = loopCount * mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
        }
    } else {
        uint32_t offset = 0;
        for (int i = 0; i < dealRowCount; i++) {
            Mul(dstUb[offset], src0Ub, src1Ub[offset], actualColumnCount);
            offset += columnCount;
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::CopyAntiqQuery(LocalTensor<T> &queryCastUb,
    uint64_t qOffset, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(Q_T); // 16 / 2 = 8
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<Q_T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = actualColumnCount * sizeof(Q_T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (columnCount - actualColumnCount) / qTypeElementSize;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % qTypeElementSize;
    copyInPadParams.paddingValue = 0;

    LocalTensor<Q_T> inputUb = inputBuf1.Get<Q_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopyPad(inputUb, queryGm_[qOffset], copyInParams, copyInPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    Cast(queryCastUb, inputUb, RoundMode::CAST_NONE, dealRowCount * columnCount); // 将Query cast到fp32
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AntiquantMatmulPreProcess(const RunInfo &info,
    GlobalTensor<KV_T> dstGm, LocalTensor<T> aMaxResUb, LocalTensor<T> srcUb, LocalTensor<T> tmpAFloorUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t step = info.mSize * columnCount;
    uint32_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    LocalTensor<T> tmpAMaxRes = aMaxResUb[startRow * BLOCK_ELEMENT_NUM];
    AbsRowMax(tmpAMaxRes, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // 128/(1.001*Amax)*A
    Duplicate(tmpAFloorUb, antiqCoeff1, dealRowCount * BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    Div(tmpAFloorUb, tmpAFloorUb, tmpAMaxRes, dealRowCount * BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    RowMuls<T>(srcUb, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // step4: 取出Qtmp整数部分
    Cast(tmpAFloorUb, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> tmpAFloorUbFp16 = tmpAFloorUb.template ReinterpretCast<half>();
    tmpAFloorUbFp16.SetSize(tmpAFloorUb.GetSize());
    Cast(tmpAFloorUbFp16, tmpAFloorUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    // step5: 将Qtmp转成fp16
    LocalTensor<half> srcUbFp16 = srcUb.template ReinterpretCast<half>();
    srcUbFp16.SetSize(srcUb.GetSize());
    Cast(srcUbFp16, srcUb, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < msdIterNum; i++) {
        AntiquantAIterExpand(dstGm, srcUbFp16, tmpAFloorUbFp16, calcSize, (i == 0 ? true : false), step * i + baseOffset);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AbsRowMax(LocalTensor<T> &tmpAMaxRes, LocalTensor<T> &srcUb,
                                                                 LocalTensor<T> tmpAUb, uint32_t dealRowCount,
                                                                 uint32_t columnCount, uint32_t actualColumnCount)
{
    Abs(tmpAUb, srcUb, dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
    LocalTensor<T> tmpRowMaxUb = tmpBuff3.Get<T>();
    RowMaxForLongColumnCount(tmpRowMaxUb, tmpAUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    Brcb(tmpAMaxRes, tmpRowMaxUb, (dealRowCount + 7) / 8, {1, 8});
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::RowMaxForLongColumnCount(LocalTensor<float> &dstUb, LocalTensor<float> srcUb,
                                                                                uint32_t dealRowCount, uint32_t columnCount,
                                                                                uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t newColumnCount = columnCount;
    uint32_t newActualColumnCount = actualColumnCount;
    if (columnCount >= REPEATE_STRIDE_UP_BOUND * FP32_BLOCK_ELEMENT_NUM) {
        uint32_t split = GetMinPowerTwo(actualColumnCount);
        split = split >> 1;

        // deal tail
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Max(srcUb[offset], srcUb[offset], srcUb[offset + split], actualColumnCount - split);
            offset += columnCount;
        }
        PipeBarrier<PIPE_V>();

        uint32_t validLen = split;
        while (validLen > ConstInfo::BUFFER_SIZE_BYTE_1K) {
            uint32_t copyLen = validLen / 2;

            offset = 0;
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Max(srcUb[offset], srcUb[offset], srcUb[offset + copyLen], copyLen);
                offset += columnCount;
            }
            PipeBarrier<PIPE_V>();

            validLen = copyLen;
        }

        for (uint32_t i = 0; i < dealRowCount; i++) {
            DataCopy(srcUb[i * validLen], srcUb[i * columnCount], validLen);
            PipeBarrier<PIPE_V>();
        }

        newColumnCount = validLen;
        newActualColumnCount = validLen;
    }

    RowMax(dstUb, srcUb, dealRowCount, newColumnCount, newActualColumnCount);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::RowMax(LocalTensor<float> &dstUb, LocalTensor<float> &srcUb,
                                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // max by row, 按行求最大值
    // dstUb[i] = max(srcUb[i, :])
    // src0Ub:[dealRowCount, columnCount] dstUb:[1, dealRowCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtypeMask;
    uint32_t remain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtypeMask], remain, dealRowCount, repeatParamsMax);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
        blockCount = (blockCount + 1) / 2;
        for (uint32_t j = 0; j < loopCount; j++) {
            Max(srcUb[j * dtypeMask], srcUb[j * dtypeMask], srcUb[(j + blockCount) * dtypeMask], dtypeMask,
                dealRowCount, repeatParamsMax);
        }
        PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(dstUb, srcUb, (actualColumnCount < dtypeMask) ? actualColumnCount : dtypeMask, dealRowCount, 1, 1,
                   columnCount / FP32_BLOCK_ELEMENT_NUM, ReduceOrder::ORDER_ONLY_VALUE);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AntiquantAIterExpand(GlobalTensor<KV_T> &dstGm, LocalTensor<half> &tmpA1,
                                                                            LocalTensor<half> &tmpA2, uint32_t calcSize,
                                                                            bool isFirst, uint64_t outOffset)
{
    if (!isFirst) {
        Sub(tmpA2, tmpA1, tmpA2, calcSize);
        PipeBarrier<PIPE_V>();
        Muls(tmpA2, tmpA2, antiquantExpandCoeff, calcSize);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<KV_T> aResOutUbI8 = outputBuf1.Get<KV_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    Cast(aResOutUbI8, tmpA2, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(dstGm[outOffset], aResOutUbI8, calcSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline int64_t SFAAVectorServiceGqaMsd<SFAAT>::MergeKv(const RunInfo &runInfo, int64_t s2GmStartOffset,
    int64_t s2GmLimit, int64_t topkGmBaseOffset, bool isValue, int64_t mergeMte3Idx, int64_t s2GmStartOffset4Merge)
{
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    bool mask = false;

    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2 * constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SYNC_INPUT_BUF2_FLAG + mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(s2IdxArray0 < 0)) { // 当前sparseblock已经超过了sparseCount的范围
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset4Merge, mergeMte3Idx, runInfo, isValue, needWaitMte3ToMte2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo, mask, s2GmOffsetArray, s2GmLimit);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 32) || // TODO，这里可以支持到128，但需要循环拷出
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset4Merge, mergeMte3Idx, runInfo, isValue, needWaitMte3ToMte2);
            mte3Size = mte2Size;
            mergeMte3Idx++;
        }
    }
    return mte2Size;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ProcessVec1Msd(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        // mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
        //                                                         (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecDealM = info.mSizeV;
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = info.mSize - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        // move lse for flash decode
        if constexpr (IS_META) {
            if (info.s2Idx == info.curSInnerLoopTimes - 1) {
                if (info.tndIsS2SplitCore) {
                    if (FLASH_DECODE) {
                        uint32_t outIdx = info.loop % (constInfo.preLoadNum);
                        auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
                        auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
                        ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                    }
                }
            }
        }
    }
}

template <typename SFAAT>
__aicore__ inline uint64_t SFAAVectorServiceGqaMsd<SFAAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    uint64_t accumTmpOutNum = 0;
    uint32_t taskId = 0;

    if constexpr (IS_META) {
        const uint32_t *bN2IdxOfFdHead = metaDataPtr->fdRes.bN2IdxOfFdHead;
        const uint32_t *gS1IdxOfFdHead = metaDataPtr->fdRes.gS1IdxOfFdHead;
        const uint32_t *s2SplitNumOfFdHead = metaDataPtr->fdRes.s2SplitNumOfFdHead;
        uint32_t usedCoreNum = metaDataPtr->usedCoreNum;
        while (taskId < usedCoreNum && (bN2IdxOfFdHead[taskId] != bN2Idx || gS1IdxOfFdHead[taskId] * constInfo.mBaseSize != gS1Idx)) {  // 考虑在tiling阶段直接算出accumOut的偏置，则可以省略CalcAccumOffset()
            accumTmpOutNum += s2SplitNumOfFdHead[taskId]; // 计算前面的workspace数
            taskId++;
        }
    } else {
        return 0;
    }
    return accumTmpOutNum;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                      const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim; // 64
    if (gPreSplitSize > mSplitInfo.vecDealM) {
        gPreSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + gPreSplitSize - 1) / gPreSplitSize; // 1
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * gPreSplitSize; // 20

    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * gPreSplitSize, dealSize, constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::AntiquantMM2ResCombine(const RunInfo &info,
    LocalTensor<MM2_OUT_T> bmmResUb, GlobalTensor<MM2_OUT_T> srcGm, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t baseOffset = startRow * columnCount;
    uint32_t copySize = dealRowCount * columnCount;
    LocalTensor<MM2_OUT_T> tmpCInt = inputBuf1.Get<MM2_OUT_T>();
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(tmpCInt, srcGm[baseOffset], copySize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(bmmResUb, tmpCInt, copySize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                      uint32_t startRow, uint32_t dealRowCount,
                                                                      uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
#ifdef SFACV_SOFTMAX_WITHOUT_BRC_PRELOAD
    uint32_t baseOffset = startRow;
#else
    uint32_t baseOffset = startRow * BLOCK_ELEMENT_NUM;
#endif
    size_t batchBase = 0;
    uint64_t inOutBaseOffset = (mSplitInfo.vecStartM + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
    LocalTensor<half> bmm2ResUb = tmpBuff1.Get<half>();
    bmm2ResUb.SetSize(vec2ComputeSize);
    AntiquantMM2ResCombine(info, bmm2ResUb, mm2ResGm[srcGmOffset], startRow, dealRowCount, columnCount, actualColumnCount);
    // 除第一个循环外，均需要更新中间计算结果
    if (!info.isFirstSInnerLoop) {
        //step2: 将softmaxExpUb转换为FP16
#ifdef SFACV_SOFTMAX_WITHOUT_BRC_PRELOAD
        LocalTensor<half> tmpSoftmaxFp16 = tmpBuff3.Get<half>();
        Cast(tmpSoftmaxFp16, softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            AscendC::RoundMode::CAST_ROUND, dealRowCount);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> tmpExpBrcbResUb = tmpBuff2.Get<half>();
        // repeatTime:指令迭代次数，每次迭代完成8个datablock的数据收集
        // repeatParams:{单次迭代内，矢量目的操作数不同datablock间地址步长, 相邻迭代间，矢量目的操作数相同datablock地址步长}
        Brcb(tmpExpBrcbResUb, tmpSoftmaxFp16, (mSplitInfo.vecDealM + 7) / 8, {1, 8}); 
        // step3: Oi = (Oi - 1)*tmpSoftmaxFp16 + Otmp
        PipeBarrier<PIPE_V>();
        RowMuls<half>(vec2ResUb, vec2ResUb, tmpExpBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        LocalTensor<T> tmpSoftmaxFp32 = tmpBuff3.Get<T>();
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = 1;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 1;
        DataCopy(tmpSoftmaxFp32, softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset], dataCopyParams);
        DataCopy(tmpSoftmaxFp32[BLOCK_ELEMENT_NUM], softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            dataCopyParams);
        LocalTensor<half> tmpSoftmaxFp16 = tmpBuff3.Get<half>();
        PipeBarrier<PIPE_V>();
        Cast(tmpSoftmaxFp16, tmpSoftmaxFp32, AscendC::RoundMode::CAST_ROUND, dealRowCount * BLOCK_ELEMENT_NUM * 2);
        PipeBarrier<PIPE_V>();
        RowMuls(vec2ResUb, vec2ResUb, tmpSoftmaxFp16,
            dealRowCount, columnCount, actualColumnCount);
#endif
        PipeBarrier<PIPE_V>();
        Add(bmm2ResUb, bmm2ResUb, vec2ResUb, vec2ComputeSize);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        PipeBarrier<PIPE_V>();
        LocalTensor<T> bmm2ResUbFp32 = tmpBuff2.Get<T>();
        Cast(bmm2ResUbFp32, bmm2ResUb, RoundMode::CAST_NONE, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        uint32_t idx = info.loop % constInfo.preLoadNum;
#ifdef SFACV_SOFTMAX_WITHOUT_BRC_PRELOAD
        LocalTensor<T> tmpSumBrcbResUb = tmpBuff1.Get<T>();
        Brcb(tmpSumBrcbResUb, softmaxSumUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        PipeBarrier<PIPE_V>();
        RowDivs(bmm2ResUbFp32, bmm2ResUbFp32, tmpSumBrcbResUb, dealRowCount, columnCount, actualColumnCount);
#else
        RowDivs(bmm2ResUbFp32, bmm2ResUbFp32, softmaxSumUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            dealRowCount, columnCount, actualColumnCount);
#endif
        pipe_barrier(PIPE_V);
        Muls(bmm2ResUbFp32, bmm2ResUbFp32, scaleC2, vec2ComputeSize);
        PipeBarrier<PIPE_V>();
        CopyAntiquantScale(antiquantVScaleUb_, valueDequantScaleGm_, info.n2Idx * constInfo.headDim);
        PipeBarrier<PIPE_V>();
        // ScaleV * bmm2res
        VecMulMat(bmm2ResUbFp32, antiquantVScaleUb_, bmm2ResUbFp32, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Bmm2ResCopyOut(info, bmm2ResUbFp32, mSplitInfo.vecStartM + startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        PipeBarrier<PIPE_V>();
        DataCopy(vec2ResUb, bmm2ResUb, dealRowCount * columnCount);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::CopyAntiquantScale(LocalTensor<T> &castUb, GlobalTensor<T> srcGm,
                                                                          uint64_t offset)
{
    uint32_t qTypeElementSize = BYTE_BLOCK / sizeof(T);
    uint32_t headDimAlign = SFAAAlign(constInfo.headDim, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = 1;
    copyInParams.blockLen = constInfo.headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlign - constInfo.headDim) / qTypeElementSize;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlign - constInfo.headDim) % qTypeElementSize;
    copyInPadParams.paddingValue = 0;

    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    DataCopyPad(castUb, srcGm[offset], copyInParams, copyInPadParams);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ProcessVec2Msd(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = info.mSizeV;
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = info.mSize - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::ProcessVec2Inner(const RunInfo &info,
                                                                  const MSplitInfo &mSplitInfo,
                                                                  uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}


template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqaMsd<SFAAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqaMsd<SFAAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                            uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                            uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuf1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bN2Idx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.mBaseSize * constInfo.headDim + // taskoffset
                      // 份数offset
                      info.tndCoreStartKVSplitPos * constInfo.mBaseSize * constInfo.headDim +
                      wsMStart * actualColumnCount; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqaMsd<SFAAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                               uint32_t wsMStart, uint32_t dealRowCount,
                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t s1StartIdx = wsMStart / constInfo.gSize;
    uint32_t startGOffset = wsMStart % constInfo.gSize;
    uint32_t s1EndIdx = SfaaCeilDiv(wsMStart + dealRowCount, static_cast<uint32_t>(constInfo.gSize)) - 1;
    uint32_t curStartRow = wsMStart;
    uint32_t curDealRowCount = 0;
    uint32_t ubOffset = 0;
    for (uint32_t curS1idx = s1StartIdx; curS1idx <= s1EndIdx; curS1idx++) {
        uint32_t outOffset = info.attenOutOffset + curS1idx * constInfo.qHeadNum * constInfo.headDim + startGOffset * constInfo.headDim;
        if (curS1idx != s1EndIdx) {
            curDealRowCount = (curS1idx + 1) * constInfo.gSize -curStartRow;
        } else {
            curDealRowCount = wsMStart + dealRowCount - curStartRow;
        }
        ubOffset = (curStartRow - wsMStart) * columnCount;
        if (unlikely((info.nextTokensPerBatch < 0) && curS1idx < (-info.nextTokensPerBatch))) {
            matmul::InitOutput<OUT_T>(attentionOutGm[outOffset], constInfo.gSize * constInfo.headDim, 0);
            curStartRow += curDealRowCount;
            startGOffset = 0;
            continue;
        }
        LocalTensor<OUT_T> curAttenOutUb = attenOutUb[ubOffset];
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(attentionOutGm[outOffset], curAttenOutUb, dataCopyParams);
        curStartRow += curDealRowCount;
        startGOffset = 0;
    }
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqaMsd<SFAAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                             uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuf1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqaMsd<SFAAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                         uint32_t dealRowCount, uint32_t columnCount,
                                         uint32_t actualColumnCount)
{
    if constexpr (IS_META) {
        if (FLASH_DECODE) {
            if (info.tndIsS2SplitCore) {
                Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
            } else {
                Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
            }
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqaMsd<SFAAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_VECTOR_GQA_MSD_H
