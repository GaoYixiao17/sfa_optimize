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
 * \file blitz_sparse_attention_s1s2_bns1_x910.h
 * \brief
 */
#ifndef BLITZ_SPARSE_ATTENTION_S1S2_BNS1_X910_H
#define BLITZ_SPARSE_ATTENTION_S1S2_BNS1_X910_H

#include "blitz_sparse_attention_s1s2_bns1_x910_base.h"


static constexpr uint16_t kPadU16 = 65535; // 0xFFFF

struct U16VecView {
    const __gm__ uint16_t* ptr = nullptr;
    uint32_t len = 0;

    __aicore__ inline bool IsValid() const { return ptr != nullptr && len != 0; }
    __aicore__ inline uint16_t Get(uint32_t i) const { return ptr[i]; }
};

struct U16T3View {
    const __gm__ uint16_t* base = nullptr;
    uint32_t d0 = 0, d1 = 0, d2 = 0;
    uint32_t totalLen = 0;

    __aicore__ inline U16VecView At(uint32_t i0, uint32_t i1) const {
        U16VecView v{};
        if (base == nullptr || d2 == 0 || totalLen == 0) return v;
        if (i0 >= d0 || i1 >= d1) return v;

        const uint64_t idx = (static_cast<uint64_t>(i0) * d1 + i1) * d2;
        const uint64_t end = idx + d2;
        if (end > totalLen) return v;

        v.ptr = base + idx;
        v.len = d2;
        return v;
    }
};

// 4D view: [b][d0][d1][d2] contiguous row-major
struct U16T4View {
    const __gm__ uint16_t* base = nullptr;

    uint32_t b  = 0;
    uint32_t d0 = 0;
    uint32_t d1 = 0;
    uint32_t d2 = 0;

    uint32_t totalLen = 0;

    __aicore__ inline U16VecView At(uint32_t ib, uint32_t i0, uint32_t i1) const {
        U16VecView v{};
        if (base == nullptr || d2 == 0 || totalLen == 0) return v;
        if (ib >= b || i0 >= d0 || i1 >= d1) return v;

        const uint64_t idx =
            ((static_cast<uint64_t>(ib) * d0 + i0) * d1 + i1) * d2;

        const uint64_t end = idx + d2;
        if (end > totalLen) return v;

        v.ptr = base + idx;
        v.len = d2;
        return v;
    }

    __aicore__ inline U16T3View AtBatch(uint32_t ib) const {
        U16T3View v{};
        if (base == nullptr || d2 == 0 || totalLen == 0) return v;
        if (ib >= b) return v;

        const uint64_t batchStride = static_cast<uint64_t>(d0) * d1 * d2;
        const uint64_t idx = static_cast<uint64_t>(ib) * batchStride;
        const uint64_t end = idx + batchStride;
        if (end > totalLen) return v;

        v.base = base + idx;
        v.d0 = d0; v.d1 = d1; v.d2 = d2;
        v.totalLen = static_cast<uint32_t>(batchStride);
        return v;
    }
};

// Linear scan: returns the smallest index i whose value is >= `lower`, stopping at the
// first `kPadU16` padding entry. A binary-search variant was evaluated and is ~38 ms
// slower at 0% sparsity on this benchmark: the jumps break the hardware prefetcher and
// every probe lands in a cold cache line, whereas the sequential scan reads one cache
// line and lets the prefetcher pipeline the rest. In practice the first loop iteration
// hits immediately (startIndex=0 in the causal benchmark), so the linear cost is O(1).
__aicore__ inline int32_t FirstGreaterEqual(const U16VecView& v, uint16_t lower)
{
    if (!v.IsValid()) return -1;

    for (uint32_t i = 0; i < v.len; ++i) {
        const uint16_t x = v.ptr[i];
        if (x == kPadU16) break;        // reached padding
        if (x >= lower) return (int32_t)i;
    }
    return -1;
}


using namespace matmul;
template<typename PFAT>
class BlitzSparseAttentionS1s2Bns1X910 : public BlitzSparseAttentionS1s2Bns1X910Base<PFAT> {
public:
    using FT = float;
    using T = typename PFAT::inputType;
    using KV_T = typename PFAT::kvInputType;
    using U = typename PFAT::maskType;
    using O = typename PFAT::outputType;
    using mmOutputTypeTmp = typename BlitzSparseAttentionTypeTraits<T,PFAT::calcMode>::mmOutputType;
    using computeType = typename BlitzSparseAttentionTypeTraits<T,PFAT::calcMode>::softmaxType;
    using pseShiftType = typename BlitzSparseAttentionTypeTraits<T,PFAT::calcMode>::pseShiftType;
    using pseShiftCastType = typename BlitzSparseAttentionTypeTraits<T,PFAT::calcMode>::pseShiftCastType;

    using mmOutputType = typename AscendC::Conditional<PFAT::msdMode == MsdMode::MSD_ON, int32_t, mmOutputTypeTmp>::type;

    __aicore__ inline BlitzSparseAttentionS1s2Bns1X910() {};
    __aicore__ inline void Process();

protected:
    __aicore__ inline void AllocGlobalResources();

    __aicore__ inline void FreeGlobalResources();

    __aicore__ inline void PseOrMaskCopyIn(int64_t offset, uint32_t souterSize,
        bool isInnerTail, uint32_t alignSInner, uint32_t unalignSInner, uint32_t padSize, bool isMask, uint32_t souterTailMaskCopyInSize);

    __aicore__ inline void SparseBandElewiseCompute(int32_t ubPingpong, uint32_t souterSize, int64_t attenMaskOffsetPre);

    __aicore__ inline void Bmm1VecInputCopyIn();

    __aicore__ inline void Bmm1ResDoVecBmm2Compute();

    __aicore__ inline void ComputeEachCoreSInnerLoop();

    __aicore__ inline void SInnerLoopFunc(int64_t sInnerFirstToken, int64_t sInnerEndToken, int curBatch, int64_t preTokens, int64_t nextTokens, U16VecView rowSabi = {});

    __aicore__ inline void ComputeEachCore(uint32_t coreIdx);

    __aicore__ inline void ComputeEachCoreBalance(uint32_t coreIdx);

    __aicore__ inline void InitEachCoreWorkspace(uint32_t coreIdx, int32_t blockNum);

    __aicore__ inline void ComputeEachCoreSplitSeqOneN(uint32_t coreIdx);

    __aicore__ inline void ProcessLastSouterLoopFinalRes();

    __aicore__ inline void CheckRowInvalid(int64_t preTokens, int64_t nextTokens, PFAComputeParam* params);

    __aicore__ inline int64_t ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue);

    __aicore__ inline void MsdRowSum(LocalTensor<FT>& dstUb, LocalTensor<FT> srcUb,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdRowMax(LocalTensor<FT>& aMaxDstUb, LocalTensor<FT>& srcUb,
                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdAbsRowMax(LocalTensor<FT>& tmpAMaxRes, LocalTensor<FT>& srcUb, LocalTensor<FT> tmpAUb, 
                                        uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdRowMuls(LocalTensor<FT>& dstUb, LocalTensor<FT>& src0Ub, LocalTensor<FT>& src1Ub, 
                                      uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdAntiParamsMulByRow(LocalTensor<FT>& dstUb, LocalTensor<FT>& antiParams, LocalTensor<FT>& src1Ub, 
                                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdFusedMulAddByRow(LocalTensor<FT>& dstLocal, LocalTensor<FT>& src0Local, LocalTensor<FT>& src1Local, 
                                               uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdMulMaxAddSumByRow(LocalTensor<FT>& dstLocal, LocalTensor<FT>& src0RowMax, LocalTensor<FT>& src1RowSum, 
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdVecMulMat(LocalTensor<FT>& dstUb, LocalTensor<FT>& src0Ub, LocalTensor<FT>& src1Ub,
                                        uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm1MsdCopyQueryGm2Ub(LocalTensor<FT> &queryUb, int64_t queryOffset,
                                                 uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdAIterExpand(GlobalTensor<int8_t> dstGm, LocalTensor<FT>& tmpA1, LocalTensor<FT>& tmpA2, 
                                          uint32_t calcSize, bool isFirst, int64_t outOffset, int gmPingpong);

    __aicore__ inline void MsdMatmulPreProcess(PFAComputeParam *params, GlobalTensor<KV_T> dstGm, LocalTensor<FT>& msdMaxResUb, LocalTensor<FT>& srcUb, LocalTensor<FT>& tmpAFloorUb,
                                               uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm1MsdCalcQueryRowSum(LocalTensor<FT> &queryUb, 
                                                  uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm1MsdExpandPertoken(PFAComputeParam *params);

    __aicore__ inline void Bmm1MsdExpand(PFAComputeParam *params);

    __aicore__ inline void MsdMatmulResCombine(LocalTensor<FT> &bmmResUb, GlobalTensor<mmOutputType> srcGm, 
                                               uint32_t expandLineCnt, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void MsdCopyAntiquantParamsPerToken(GlobalTensor<FT> srcGm, int64_t offset, 
                                                          uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void Bmm1MsdSqueezeResMulOffsetKPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, 
                                                               uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm1MsdSqueezeResMulScaleKPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm1MsdSqueezePerToken(PFAComputeParam *params, LocalTensor<FT>& bmm1ResUb, uint32_t startRow, uint32_t dealRowCount, int gmPingpong);

    __aicore__ inline void Bmm1MsdSqueeze(PFAComputeParam *params, LocalTensor<FT>& bmm1ResUb, uint32_t startRow, uint32_t dealRowCount, int ubPingpong, int gmPingpong);

    __aicore__ inline void Bmm2MsdExpandMm1ResMulScaleVPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, 
                                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdExpandResPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, GlobalTensor<KV_T> bmm1ExpandGm,
                                                    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdExpandResMulOffsetVPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb,
                                                              uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdExpandPertoken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, GlobalTensor<KV_T> bmm1ExpandGm, uint32_t startRow, uint32_t dealRowCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdExpand(PFAComputeParam *params, LocalTensor<FT> &mmResUb, GlobalTensor<KV_T> bmm1ExpandGm, uint32_t startRow, uint32_t dealRowCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdSqueezeResMulMax(LocalTensor<FT> &bmm2ResUb,
                                                   uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdSqueezePertoken(PFAComputeParam *params, LocalTensor<FT>& bmm2ResUb, GlobalTensor<mmOutputType> bmm2ResGmDb, int gmPingpong);

    __aicore__ inline void Bmm2MsdSqueeze(PFAComputeParam *params, LocalTensor<computeType>& bmm2ResUb, GlobalTensor<mmOutputType> bmm2ResGmDb, int gmPingpong);

    __aicore__ inline void CopyAntiquantScale(LocalTensor<FT>& castUb, GlobalTensor<T> srcGm, int64_t offset);

    __aicore__ inline void VecAddMat(LocalTensor<FT> dstUb, LocalTensor<FT> src0Ub, LocalTensor<FT> src1Ub, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);

    __aicore__ inline void MsdSoftmaxResPreProcess(PFAComputeParam *params, GlobalTensor<KV_T> dstGm, LocalTensor<FT> srcUb, LocalTensor<FT> tmpAFloorUb, uint32_t startRow,
                                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong);

    __aicore__ inline void Bmm1MsdExpandPerchannel(PFAComputeParam *params);

    __aicore__ inline void Bmm1MsdSqueezePerchannel(PFAComputeParam *params, LocalTensor<FT> &bmm1ResUb, uint32_t startRow, uint32_t dealRowCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdExpandPerchannel(PFAComputeParam *params, LocalTensor<FT> &bmm1ResUb, GlobalTensor<KV_T> &bmm1ExpandGm, uint32_t startRow, uint32_t dealRowCount, int gmPingpong);

    __aicore__ inline void Bmm2MsdSqueezePerchannel(PFAComputeParam *params, LocalTensor<FT> &bmm2ResUb, GlobalTensor<mmOutputType> &srcGM, int gmPingpong);

    __aicore__ inline void Bmm2UpdateDivPerchannel(LocalTensor<FT> &bmm2ResLastUb);

    __aicore__ inline void Bmm2Antiquant(PFAComputeParam *params) {
        int step = this->tilingData->promptAttentionSingleCoreParams.kvAntiquantSInnerSize;
        int kvAntiquantLoopTimes = (params->singleProcessSInnerBmmTail + step -1) / step;
        int headSize = this->tilingData->promptAttentionBaseParams.alignedHeadSize;

        LocalTensor<T> scaleLocal = this->antiquantScaleUb.template Get<T>(headSize);
        LocalTensor<T> offsetLocal = this->antiquantOffsetUb.template Get<T>(headSize);
        if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel && PFAT::MM_TYPE != MatMulType::MM_PA_ANTIQUANT) {
            scaleLocal.SetSize(headSize);
            offsetLocal.SetSize(headSize);
            DataCopy(scaleLocal, this->antiquantScaleGm[((int64_t)this->tilingData->promptAttentionBaseParams.headNumSize + (int64_t)params->batchNOffset) / this->headNumRatio * headSize], headSize);
            if (!this->isAntiquantSymmetric) {
                DataCopy(offsetLocal, this->antiquantOffsetGm[((int64_t)this->tilingData->promptAttentionBaseParams.headNumSize + (int64_t)params->batchNOffset) / this->headNumRatio * headSize], headSize);
            } else {
                Duplicate(offsetLocal, static_cast<T>(0), headSize);    // Symmetric quantization
            }
        }

        if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel && PFAT::MM_TYPE == MatMulType::MM_PA_ANTIQUANT) {
            scaleLocal.SetSize(headSize);
            offsetLocal.SetSize(headSize);
            DataCopy(scaleLocal, this->valueAntiquantScaleGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            if (this->msdIsVOffsetExist) {
                DataCopy(offsetLocal, this->valueAntiquantOffsetGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            } else {
                Duplicate(offsetLocal, static_cast<T>(0), headSize);    // Symmetric quantization
            }
        }

        DataCopyParams kvCopyParam;
        kvCopyParam.blockLen = headSize / 32;   // KV int8 dtype  32 : 32B alignment
        kvCopyParam.dstStride = 0;
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            kvCopyParam.srcStride = 0;  // BNSD
        } else {
            kvCopyParam.srcStride = (this->MultiHeadKV - headSize) / 32;    // BSH  32 : 32B alignment
        }

        for (int loopIdx = 0; loopIdx < kvAntiquantLoopTimes; loopIdx++) {
            int64_t kvComputeSInnerSize = (loopIdx == kvAntiquantLoopTimes - 1) ? (params->singleProcessSInnerBmmTail - loopIdx * step) : step;
            kvComputeSInnerSize = kvComputeSInnerSize > step ? step : kvComputeSInnerSize;
            int64_t vOffset;
            if constexpr (PFAT::layout == PFALayout::BNSD) {
                vOffset = params->valueOffset + loopIdx * step * headSize;  // BNSD
            } else {
                vOffset = params->valueOffset + loopIdx * step * this->MultiHeadKV; // BSH
            }

            LocalTensor<int8_t> srcLocal = this->kvAntiquantSrcQueue.template AllocTensor<int8_t>();
            LocalTensor<T> dstLocal = this->kvAntiquantDstQueue.template AllocTensor<T>();

            if (PFAT::MM_TYPE != MatMulType::MM_PA_ANTIQUANT) {
                kvCopyParam.blockCount = kvComputeSInnerSize;
                if constexpr (PFAT::enablePrefix) {
                    if (params->isPrefixInnerIter) {
                        DataCopy(srcLocal, this->valueSharedPrefixGm[vOffset], kvCopyParam);
                    } else {
                        DataCopy(srcLocal, this->valueGm[vOffset], kvCopyParam);
                    }
                } else {
                    if (this->isKvContinuous == 0) {
                        ListTensorDesc valueListDesc((__gm__ void*)this->value_ptr);
                        __gm__ uint8_t* tempValueGm = (__gm__ uint8_t*)valueListDesc.GetDataPtr<__gm__ uint8_t>(params->taskBatch);
                        this->valueGm.SetGlobalBuffer((__gm__ KV_T*)tempValueGm);
                    }
                    DataCopy(srcLocal, this->valueGm[vOffset], kvCopyParam);
                }
            } else {
                uint32_t kvHeadNum = this->tilingData->promptAttentionBaseParams.headNumSize / this->tilingData->promptAttentionBaseParams.headNumRatio;
                uint32_t blockTableDim2 = this->tilingData->promptAttentionBaseParams.blockTableDim2;
                uint32_t nOffset = params->batchNOffset / this->headNumRatio;
                uint32_t blockSize = this->tilingData->promptAttentionBaseParams.blockSize;
                int64_t i = 0;
                while (i < kvComputeSInnerSize) {
                    uint64_t blockTableIdx = (params->sInnerOffsetDataSize + loopIdx * step + i) / blockSize;
                    uint32_t offsetInBlock = (params->sInnerOffsetDataSize + loopIdx * step + i) % blockSize;
                    uint32_t remainBlockNum = blockSize - offsetInBlock;
                    kvCopyParam.blockCount = remainBlockNum > kvComputeSInnerSize - i ? kvComputeSInnerSize - i : remainBlockNum;
                    uint64_t curOffset = 0;
                    uint32_t blockId = *(reinterpret_cast<__gm__ int32_t*>(this->blocktable_ptr) + params->taskBatch * blockTableDim2 + blockTableIdx);
                    if (this->tilingData->promptAttentionBaseParams.PAlayoutType == 1) {
                        curOffset = blockId * blockSize * kvHeadNum * headSize + offsetInBlock * kvHeadNum * headSize + nOffset * headSize;
                    } else {
                        curOffset = blockId * blockSize * kvHeadNum * headSize + nOffset * blockSize * headSize + offsetInBlock * headSize;
                    }
                    DataCopy(srcLocal[i * headSize], this->valueGm[curOffset], kvCopyParam);
                    i += kvCopyParam.blockCount;
                }
            }
            this->kvAntiquantSrcQueue.EnQue(srcLocal);
            srcLocal = this->kvAntiquantSrcQueue.template DeQue<int8_t>();

            AntiQuantShapeInfo antiquantShape = {static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize),
                                                 static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize)};
            
            dstLocal.SetSize(kvComputeSInnerSize * headSize);
            srcLocal.SetSize(kvComputeSInnerSize * headSize);
            
            if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel) {
                AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, offsetLocal, scaleLocal, kvComputeSInnerSize, antiquantShape);
            } else {
                AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, this->valueAntiquantOffset, this->valueAntiquantScale, kvComputeSInnerSize, antiquantShape);
            }
            this->kvAntiquantDstQueue.EnQue(dstLocal);

            dstLocal = this->kvAntiquantDstQueue.template DeQue<T>();
            DataCopy(this->valueGmAntiquant[static_cast<int64_t>(loopIdx) * step * headSize], dstLocal, kvComputeSInnerSize * headSize);

            this->kvAntiquantSrcQueue.FreeTensor(srcLocal);
            this->kvAntiquantDstQueue.FreeTensor(dstLocal);
        }
        event_t bmmWaitAntiEvt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
        WaitFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
    }

    __aicore__ inline void Bmm1Antiquant(PFAComputeParam *params){
        int step = this->tilingData->promptAttentionSingleCoreParams.kvAntiquantSInnerSize;
        int kvAntiquantLoopTimes = (params->singleProcessSInnerBmmTail + step -1) / step;
        int headSize = this->tilingData->promptAttentionBaseParams.alignedHeadSize;

        LocalTensor<T> scaleLocal = this->antiquantScaleUb.template Get<T>(headSize);
        LocalTensor<T> offsetLocal = this->antiquantOffsetUb.template Get<T>(headSize);
        if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel && PFAT::MM_TYPE != MatMulType::MM_PA_ANTIQUANT) {
            scaleLocal.SetSize(headSize);
            offsetLocal.SetSize(headSize);
            DataCopy(scaleLocal, this->antiquantScaleGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            if (!this->isAntiquantSymmetric) {
                DataCopy(offsetLocal, this->antiquantOffsetGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            } else {
                Duplicate(offsetLocal, static_cast<T>(0), headSize);    // Symmetric quantization
            }
        }

        if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel && PFAT::MM_TYPE == MatMulType::MM_PA_ANTIQUANT) {
            scaleLocal.SetSize(headSize);
            offsetLocal.SetSize(headSize);
            DataCopy(scaleLocal, this->keyAntiquantScaleGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            if (this->msdIsKOffsetExist) {
                DataCopy(offsetLocal, this->keyAntiquantOffsetGm[params->batchNOffset / this->headNumRatio * headSize], headSize);
            } else {
                Duplicate(offsetLocal, static_cast<T>(0), headSize);    // Symmetric quantization
            }
        }

        DataCopyParams kvCopyParam;
        kvCopyParam.blockLen = headSize / 32;   // KV int8 dtype  32 : 32B alignment
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            kvCopyParam.srcStride = 0;  // BNSD
        } else {
            kvCopyParam.srcStride = (this->MultiHeadKV - headSize) / 32;    // BSH  32 : 32B alignment
        }
        kvCopyParam.dstStride = 0;

        for (int loopIdx = 0; loopIdx < kvAntiquantLoopTimes; loopIdx++) {
            Bmm1AntiquantEachLoop(params, scaleLocal, offsetLocal, kvCopyParam, step, kvAntiquantLoopTimes, headSize, loopIdx);
        }
        event_t bmmWaitAntiEvt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
        WaitFlag<HardEvent::MTE3_MTE2>(bmmWaitAntiEvt);
    }

    __aicore__ inline void Bmm1AntiquantEachLoop(PFAComputeParam *params, LocalTensor<T> scaleLocal, LocalTensor<T> offsetLocal, DataCopyParams kvCopyParam, int step, int kvAntiquantLoopTimes, int headSize, int loopIdx){
        int64_t kvComputeSInnerSize = (loopIdx == kvAntiquantLoopTimes - 1) ? (params->singleProcessSInnerBmmTail - loopIdx * step) : step;
        kvComputeSInnerSize = kvComputeSInnerSize > step ? step : kvComputeSInnerSize;
        int64_t kOffset;
        if constexpr (PFAT::layout == PFALayout::BNSD) {
            kOffset = params->tensorBOffset + loopIdx * step * headSize;  // BNSD
        } else {
            kOffset = params->tensorBOffset + loopIdx * step * this->MultiHeadKV; // BSH
        }

        LocalTensor<int8_t> srcLocal = this->kvAntiquantSrcQueue.template AllocTensor<int8_t>();
        LocalTensor<T> dstLocal = this->kvAntiquantDstQueue.template AllocTensor<T>();

        if (PFAT::MM_TYPE != MatMulType::MM_PA_ANTIQUANT) {
            kvCopyParam.blockCount = kvComputeSInnerSize;
            if constexpr (PFAT::enablePrefix) {
                if (params->isPrefixInnerIter) {
                    DataCopy(srcLocal, this->keySharedPrefixGm[kOffset], kvCopyParam);
                } else {
                    DataCopy(srcLocal, this->keyGm[kOffset], kvCopyParam);
                }
            } else {
                if (this->isKvContinuous == 0) {
                    ListTensorDesc keyListDesc((__gm__ void*)this->key_ptr);
                    __gm__ uint8_t* tempKeyGm = (__gm__ uint8_t*)keyListDesc.GetDataPtr<__gm__ uint8_t>(params->taskBatch);
                    this->keyGm.SetGlobalBuffer((__gm__ KV_T*)tempKeyGm);
                }
                DataCopy(srcLocal, this->keyGm[kOffset], kvCopyParam);
            }
        } else {
            uint32_t blockTableDim2 = this->tilingData->promptAttentionBaseParams.blockTableDim2;
            uint32_t kvHeadNum = this->tilingData->promptAttentionBaseParams.headNumSize / this->tilingData->promptAttentionBaseParams.headNumRatio;
            uint32_t blockSize = this->tilingData->promptAttentionBaseParams.blockSize;
            uint32_t nOffset = params->batchNOffset / this->headNumRatio;
            int64_t i = 0;
            while (i < kvComputeSInnerSize) {
                uint64_t blockTableIdx = (params->sInnerOffsetDataSize + loopIdx * step + i) / blockSize;
                uint32_t offsetInBlock = (params->sInnerOffsetDataSize + loopIdx * step + i) % blockSize;
                uint32_t remainBlockNum = blockSize - offsetInBlock;
                kvCopyParam.blockCount = remainBlockNum > kvComputeSInnerSize - i ? kvComputeSInnerSize - i : remainBlockNum;
                uint32_t blockId = *(reinterpret_cast<__gm__ int32_t*>(this->blocktable_ptr) + params->taskBatch * blockTableDim2 + blockTableIdx);
                uint64_t curOffset = 0;
                if (this->tilingData->promptAttentionBaseParams.PAlayoutType == 1) {
                    curOffset = blockId * blockSize * kvHeadNum * headSize + offsetInBlock * kvHeadNum * headSize + nOffset * headSize;
                } else {
                    curOffset = blockId * blockSize * kvHeadNum * headSize + nOffset * blockSize * headSize + offsetInBlock * headSize;
                }
                DataCopy(srcLocal[i * headSize], this->keyGm[curOffset], kvCopyParam);
                i += kvCopyParam.blockCount;
            }
        }
        Bmm1AntiquantEachFlow(params, srcLocal, dstLocal, scaleLocal, offsetLocal, step, kvComputeSInnerSize, headSize, loopIdx);
    }

    __aicore__ inline void Bmm1AntiquantEachFlow(PFAComputeParam *params, LocalTensor<int8_t> srcLocal, LocalTensor<T> dstLocal, LocalTensor<T> scaleLocal, LocalTensor<T> offsetLocal,
        int step, int64_t kvComputeSInnerSize, int headSize, int loopIdx){
        this->kvAntiquantSrcQueue.EnQue(srcLocal);
        srcLocal = this->kvAntiquantSrcQueue.template DeQue<int8_t>();

        AntiQuantShapeInfo antiquantShape = {static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize),
                                                static_cast<uint32_t>(kvComputeSInnerSize), static_cast<uint32_t>(headSize)};
        srcLocal.SetSize(kvComputeSInnerSize * headSize);
        dstLocal.SetSize(kvComputeSInnerSize * headSize);

        if (this->tilingData->promptAttentionBaseParams.isAntiPerchannel) {
            AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, offsetLocal, scaleLocal, kvComputeSInnerSize, antiquantShape);
        } else {
            AscendAntiQuant<KV_T, T, false>(dstLocal, srcLocal, this->keyAntiquantOffset, this->keyAntiquantScale, kvComputeSInnerSize, antiquantShape);
        }
        this->kvAntiquantDstQueue.EnQue(dstLocal);

        dstLocal = this->kvAntiquantDstQueue.template DeQue<T>();
        DataCopy(this->keyGmAntiquant[static_cast<int64_t>(loopIdx) * step * headSize], dstLocal, kvComputeSInnerSize * headSize);

        this->kvAntiquantSrcQueue.FreeTensor(srcLocal);
        this->kvAntiquantDstQueue.FreeTensor(dstLocal); 
    }

    __aicore__ inline void Bmm1ComputeIterate(PFAComputeParam *params) {
       if (this->mm1SingleCoreNPrev != params->mm1SingleCoreN) {
            // Reduce the number of SetOrgShape calls to reduce the frequency of CV communications.
            this->mm.SetOrgShape(this->tilingData->bmm1TilingDataRect.M * this->msdIterNum, this->tilingData->bmm1TilingDataRect.N,
                                 this->tilingData->bmm1TilingDataRect.Ka, this->tilingData->bmm1TilingDataRect.Kb,
                                 params->mm1SingleCoreN);
            this->mm1SingleCoreNPrev = params->mm1SingleCoreN;
        }
        if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
            this->bmm1LocalInfo = this->PABmm1UB.template Get<uint32_t>();
            this->bmm1LocalInfo.SetValue(0, params->taskBatch);
            this->bmm1LocalInfo.SetValue(1, params->batchNOffset / this->tilingData->promptAttentionBaseParams.headNumRatio);
            this->bmm1LocalInfo.SetValue(2, params->sInnerOffsetDataSize);  // 2: Sinner offset
            this->bmm1LocalInfo.SetValue(3, (uint32_t)((reinterpret_cast<int64_t>(this->currentKey)>>32) & 0x00000000ffffffff));  // 3: The high position of the pointer key.  32: Shift right by 32 bits.
            this->bmm1LocalInfo.SetValue(4, (uint32_t)(reinterpret_cast<int64_t>(this->currentKey)));  // 4: The low position of the pointer key.
            this->bmm1LocalInfo.SetValue(5, (uint32_t)((reinterpret_cast<int64_t>(this->blocktable_ptr)>>32) & 0x00000000ffffffff));  // 5: The high position of the pointer blocktable.  32: Shift right by 32 bits.
            this->bmm1LocalInfo.SetValue(6, (uint32_t)(reinterpret_cast<int64_t>(this->blocktable_ptr)));  // 6: The low position of the pointer blocktable.

            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

            DataCopy(this->bmm1CBDataGm[params->gmPingpong], this->bmm1LocalInfo, 8);  // 8: alignment

            event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);

            this->mm.SetSelfDefineData(reinterpret_cast<int64_t>(this->bmm1CBDataPtr[params->gmPingpong]));
        }

        if (this->isKvContinuous == 0) {
            ListTensorDesc keyListDesc((__gm__ void*)this->key_ptr);
            __gm__ uint8_t* tempKeyGm = (__gm__ uint8_t*)keyListDesc.GetDataPtr<__gm__ uint8_t>(params->taskBatch);
            this->keyGm.SetGlobalBuffer((__gm__ KV_T*)tempKeyGm);
        }

        if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
            this->mm.SetTensorA(this->queryMsdExpandGm);
        } else {
            this->mm.SetTensorA(this->queryGm[params->tensorAOffset]);
        }

        // quant:
        if constexpr (IsSameType<T, int8_t>::value) {
            this->mm.SetQuantScalar(this->dequantScale1);
        }

        if (params->isBlockSparse) {
            // True sub-tile gather. Two paths share this code:
            //   1) Contiguous-K fast path: when the packed sabi indices form a
            //      consecutive run [s, s+1, ..., s+packedCount-1], the kept K
            //      positions span one contiguous HBM region of length
            //      packedCount * kvBlockSize. Emit ONE wide matmul of
            //      N = mm1SingleCoreN (= packedCount * kvBlockSize minus tail
            //      truncation) at column offset 0.
            //   2) Gather slow path: when sabis are non-contiguous (mid-sparsity
            //      with sub-blocks skipped within a coarse tile), issue
            //      packedCount narrow N = kvBlockSize matmuls, each writing the
            //      i-th kvBlockSize-wide slice. Intermediate iters use
            //      sync=true because back-to-back IterateAll<false> hangs the
            //      cube on this CANN version regardless of waitIterateAll —
            //      the library drops completion-flag state between iterates.
            const uint32_t souterM = params->singleProcessSOuterSize * this->msdIterNum;
            const int64_t kvTokenStride =
                (PFAT::layout == PFALayout::BNSD)
                ? (int64_t)this->tilingData->promptAttentionBaseParams.headSize
                : (int64_t)this->MultiHeadKV;
            // Per-sabi-entry token stride: 128 for sabi-at-128 mode, 512 for sabi-at-512.
            // Direct member access (tilingData is a plain struct in GM, getters may
            // not inline correctly for raw-pointer-casted access).
            const int64_t kvBlockSize =
                (int64_t)this->tilingData->promptAttentionBaseParams.kvBlockSize;
            // params->tensorBOffset holds the per-batch-head base (tensorBCoreOffset),
            // captured at enqueue time. Reading this->tensorBCoreOffset live would
            // race against multi-head LoopSOuterOffsetInit updates between batches.
            const int64_t kvBase = params->tensorBOffset;
            const uint16_t firstSabi = params->packedSabi[0];
            const uint16_t lastSabi  = params->packedSabi[params->packedCount - 1];
            const bool kContiguous   = ((uint32_t)lastSabi - (uint32_t)firstSabi + 1u
                                        == (uint32_t)params->packedCount);
            // NOTE: the per-sub-tile token offset for KV > basicSInnerSize is
            // pre-baked into params->tensorBOffset by the wide-KV packer branch
            // in SInnerLoopFunc; no extra term needed here. Keeping this code
            // byte-identical for the existing (KV <= basicSInnerSize) path
            // avoids an I-cache layout shift in the rest of this function.
            if (kContiguous) {
                const int64_t kvOffset = kvBase + (int64_t)firstSabi * kvBlockSize * kvTokenStride;
                this->mm.SetTail(souterM, params->mm1SingleCoreN);
                if constexpr (PFAT::msdMode != MsdMode::MSD_ON
                              && (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value)) {
                    this->mm.SetTensorB(this->keyGmAntiquant[kvOffset], true);
                } else if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
                    this->mm.SetTensorB(this->keyGm, true);
                } else {
                    this->mm.SetTensorB(this->keyGm[kvOffset], true);
                }
                this->mm.template IterateAll<false>(
                    this->bmm1ResGmDb[params->gmPingpong], false, false, true, params->fakeMsg);
                return;
            }
            for (int i = 0; i < (int)params->packedCount; ++i) {
                const int64_t kvOffset = kvBase
                    + (int64_t)params->packedSabi[i] * kvBlockSize * kvTokenStride;
                uint32_t cols = (i == (int)params->packedCount - 1)
                    ? (params->mm1SingleCoreN - (uint32_t)i * (uint32_t)kvBlockSize)
                    : (uint32_t)kvBlockSize;
                this->mm.SetTail(souterM, cols);
                if constexpr (PFAT::msdMode != MsdMode::MSD_ON
                              && (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value)) {
                    this->mm.SetTensorB(this->keyGmAntiquant[kvOffset], true);
                } else if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
                    this->mm.SetTensorB(this->keyGm, true);
                } else {
                    this->mm.SetTensorB(this->keyGm[kvOffset], true);
                }
                if (i < (int)params->packedCount - 1) {
                    this->mm.template IterateAll<true>(
                        this->bmm1ResGmDb[params->gmPingpong][(int64_t)i * kvBlockSize],
                        false, false, false, params->fakeMsg);
                } else {
                    this->mm.template IterateAll<false>(
                        this->bmm1ResGmDb[params->gmPingpong][(int64_t)i * kvBlockSize],
                        false, false, true, params->fakeMsg);
                }
            }
            return;
        }

        this->mm.SetTail(params->singleProcessSOuterSize * this->msdIterNum, params->singleProcessSInnerBmmTail);

        if constexpr (PFAT::msdMode != MsdMode::MSD_ON and (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value)) {
            this->mm.SetTensorB(this->keyGmAntiquant, true);
        } else if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
            this->mm.SetTensorB(this->keyGm, true);
        } else if constexpr (PFAT::enablePrefix) {
            if (params->isPrefixInnerIter) {
                this->mm.SetTensorB(this->keySharedPrefixGm[params->tensorBOffset], true);
            } else {
                this->mm.SetTensorB(this->keyGm[params->tensorBOffset], true);
            }
        } else {
            this->mm.SetTensorB(this->keyGm[params->tensorBOffset], true);
        }

        this->mm.template IterateAll<false>(this->bmm1ResGmDb[params->gmPingpong], false, false, true, params->fakeMsg);
    }

    __aicore__ inline void Bmm1GmResCopyInUb(LocalTensor<computeType> &mmResUb, int64_t gmOffset,
        int32_t blockCount, int32_t blockLen, int32_t srcStride, int pingpong, int ubPingpong,
        uint32_t souterSize, bool unalign, uint32_t alignSInner, uint32_t unalignSInner, bool setCopyIn) {
        if (unalign) {
            mmResUb.SetSize(souterSize * alignSInner);
        }
        else {
            mmResUb.SetSize(souterSize * alignSInner);
        }

        this->mm1GmUbCopyParam[ubPingpong].blockCount = blockCount;
        this->mm1GmUbCopyParam[ubPingpong].blockLen = blockLen;
        this->mm1GmUbCopyParam[ubPingpong].srcStride = srcStride;
        this->mm1GmUbCopyParam[ubPingpong].dstStride = 0;

        WaitFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong]);
        DataCopy(mmResUb, this->bmm1ResGmDb[pingpong][gmOffset], this->mm1GmUbCopyParam[ubPingpong]);
        SetFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong]);
    }

    __aicore__ inline void SoftmaxResCopyOut(LocalTensor<computeType> &mmResUb, int64_t gmOffset, int pingpong, int ubPingpong, uint32_t singleProcessSInnerSizeNow) {
        if constexpr (PFAT::calcMode == OptimizationMode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            // The same ub buffer is used before and after cast.
            LocalTensor<T> tmpSoftmaxResUb = mmResUb.template ReinterpretCast<T>();
            PipeBarrier<PIPE_V>();
            Cast(tmpSoftmaxResUb, mmResUb, RoundMode::CAST_ROUND, mmResUb.GetSize());

            // After cast, the valid data length is reduced by half. Therefore, the datacopy length needs to be changed.
            this->mm1GmUbCopyParam[ubPingpong].blockLen = singleProcessSInnerSizeNow / (32 / sizeof(T));   // 32 : 32B alignment
        }

        this->Bmm1Queue.EnQue(mmResUb);  // Can't move forward. It must be placed here, otherwise there will be an accuracy error.
        this->mm1GmUbCopyParam[ubPingpong].dstStride = this->mm1GmUbCopyParam[ubPingpong].srcStride;
        this->mm1GmUbCopyParam[ubPingpong].srcStride = 0;

        mmResUb = this->Bmm1Queue.template DeQue<computeType>();

        PFAComputeParam *params = this->headParams;

        if constexpr (PFAT::calcMode == OptimizationMode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value) {
            // Under compact arrangement, the offset needs to be modified when copying fp16 data to fp32 global memory.
            GlobalTensor<T> tmpBmm1ResGmDb;
            tmpBmm1ResGmDb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->bmm1ResGmDb[pingpong][gmOffset / 2].address_), mmResUb.GetSize());
            LocalTensor<T> tmpSoftmaxResUb = mmResUb.template ReinterpretCast<T>();
            DataCopy(tmpBmm1ResGmDb, tmpSoftmaxResUb, this->mm1GmUbCopyParam[ubPingpong]);
        } else {
            if constexpr (PFAT::MM_TYPE == MatMulType::MM_IBSHARE_NORM && PFAT::calcMode == OptimizationMode::HighPerformance) {
                PFAComputeParam *params = this->headParams;
                if (params->isLastInnerIter && params->singleProcessSInnerBmmTail <= SINGLE_PROCESS_SINNER_BMMTAIL_LIMIT) {
                    this->mm1GmUbCopyParam[ubPingpong].dstStride = (params->mm2SingleKAlign - params->mm1SingleCoreN) * sizeof(T) / MM2_SINGLE_K_ALIGN_SIZE;
                }
            }
            DataCopy(this->bmm1ResGmDb[pingpong][gmOffset], mmResUb, this->mm1GmUbCopyParam[ubPingpong]);
        }

        SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong]);
    }

    // Pre-mask the cube-OOB cols [bmm2K, mm1N) to -10000 before softmax, on the
    // global last batch only. Kept out-of-line and `noinline` so the inlined
    // caller (Res1VecCompute) preserves its instruction layout for the hot
    // path that doesn't need this scrub. This is executed in Two stages:
    //   Stage 1 — partial leading BLOCK at floor(bmm2K/8)*8, bitmask form:
    //     mask[0] = bits [laneOffset, 8) set, where laneOffset = bmm2K % 8.
    //     Skipped when bmm2K is already 8-aligned (laneOffset == 0).
    //   Stage 2 — full aligned BLOCKs from ceil(bmm2K/8)*8 to mm1N, count form
    //     (writes the FIRST `count` lanes of each repeat).
    // After the masked write the vector mask register is restored to "all
    // lanes on" so subsequent softmax / Muls / Add ops see a normal mask state.
    __aicore__ __attribute__((noinline)) void ApplyTailKOOBPreMask(
        int32_t ubPingpong, uint32_t bmm2K, uint32_t mm1N, uint32_t souterSize)
    {
        const uint32_t partialStart = bmm2K & ~uint32_t(7);
        const uint32_t laneOffset   = bmm2K - partialStart;
        const uint32_t alignedStart = partialStart + ((laneOffset > 0u) ? 8u : 0u);
        const uint8_t  repStride    = static_cast<uint8_t>(mm1N / 8u);

        if (laneOffset != 0u) {
            uint64_t partialMask[2] = {
                ((1ULL << 8) - (1ULL << laneOffset)),
                0ULL
            };
            Duplicate<computeType, true>(
                this->mmResUb[ubPingpong][partialStart],
                static_cast<computeType>(-10000.0f),
                partialMask, static_cast<uint8_t>(souterSize), 1, repStride);
        }
        if (alignedStart < mm1N) {
            Duplicate<computeType, true>(
                this->mmResUb[ubPingpong][alignedStart],
                static_cast<computeType>(-10000.0f),
                static_cast<uint64_t>(mm1N - alignedStart),
                static_cast<uint8_t>(souterSize), 1, repStride);
        }
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Res1VecCompute(PFAComputeParam *params) {
        if (params->fakeMsg) {
            return;
        }
        int64_t mm1ResGmOffset = 0;
        int64_t nextMm1ResGmOffset = 0;
        int64_t attenMaskOffset = params->attenMaskOffset;
        int64_t attenMaskOffsetPre = params->attenMaskOffsetPre;
        int64_t pseShiftOffset = params->pseShiftOffset;
        LocalTensor<float> softmaxMaxUbSub;
        LocalTensor<float> softmaxSumUbSub;
        LocalTensor<computeType> softmaxExpUbSub;

        int ubPingpong = 0;
        int64_t nextSouterOffset;
        uint32_t computeSize;
        uint32_t padSize = 0;
        for (int64_t souterOffset = 0; souterOffset < params->singleProcessSOuterSize; souterOffset = nextSouterOffset) {     // Pending rectification
            int64_t leftSouterSize = params->singleProcessSOuterSize - souterOffset;
            int64_t souterSize = (leftSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen : leftSouterSize;
            nextSouterOffset = souterOffset + this->softmaxSouterStepLen;
            bool noLastSoftmaxLoop = (nextSouterOffset < params->singleProcessSOuterSize);
            bool noLastLastSoftmaxLoop = (nextSouterOffset + this->softmaxSouterStepLen < params->singleProcessSOuterSize);
            int64_t nextLeftSouterSize = params->singleProcessSOuterSize - nextSouterOffset;
            int64_t nextSouterSize = (nextLeftSouterSize >= this->softmaxSouterStepLen) ? this->softmaxSouterStepLen : nextLeftSouterSize;
            nextMm1ResGmOffset = mm1ResGmOffset + souterSize * params->mm1SingleCoreN;

            // mm1 + mask*-10000
            softmaxMaxUbSub = this->softmaxMaxUb[souterOffset * 8];  // 8 softmaxShapeArray, The length of the second dimension
            softmaxSumUbSub = this->softmaxSumUb[souterOffset * 8];  // 8 softmaxShapeArray, The length of the second dimension

            // mul scaleValue
            computeSize = souterSize * params->singleProcessSInnerSizeNow;
            WaitFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong]);  // Synchronize CopyIn
            Muls(this->mmResUb[ubPingpong], this->mmResUb[ubPingpong],
                 static_cast<computeType>(this->tilingData->promptAttentionBaseParams.scaleValue), computeSize);
            PipeBarrier<PIPE_V>();

            if (params->usePseShift) {
                this->pseShiftUb = this->tempBmm2Queue.template DeQue<pseShiftType>();
                if constexpr (AscendC::IsSameType<pseShiftCastType, float>::value) {
                    LocalTensor<float> pseShiftCastTensor = this->pseShiftCastUb.template Get<float>(this->pseShiftUbSize);
                    Cast(pseShiftCastTensor, this->pseShiftUb, RoundMode::CAST_NONE, computeSize);
                    PipeBarrier<PIPE_V>();
                    Add(this->mmResUb[ubPingpong], this->mmResUb[ubPingpong], pseShiftCastTensor, computeSize);
                } else {
                    Add(this->mmResUb[ubPingpong], this->mmResUb[ubPingpong], this->pseShiftUb, computeSize);
                }
                PipeBarrier<PIPE_V>();
                this->tempBmm2Queue.FreeTensor(this->pseShiftUb);
                if (params->useMask && params->sparseBandSelect0) {  // mask pre fetch. In non band OptimizationMode, the sparseBandSelect0 is true. Just focus on the previous useMask.
                    if constexpr (PFAT::enablePrefix) {
                        padSize = params->isPrefixInnerIter ? params->padPrefixSize : params->padSize;
                    } else {
                        padSize = params->padSize;
                    }
                    this->PseOrMaskCopyIn(attenMaskOffset, souterSize, params->isInnerTail,
                        params->maskCopyInCol, params->singleProcessSInnerBmmTail, padSize, true, souterSize);
                }
            }

            // T2.A: attentionMaskType ∈ {0, 1} only — band ElewiseCompute removed.
            this->template ElewiseCompute<U>(this->mmResUb[ubPingpong],
                                             souterSize,
                                             params->singleProcessSInnerSizeNow,
                                             params->maskCopyInCol,
                                             params->useMask,
                                             this->bmm1ResCopyInEvent[ubPingpong], 0);

            // With true sub-tile gather, every score-buffer column is a kept K
            // position by construction. The post-matmul Adds(-10000) sub-block
            // correction is no longer needed — bmm1 only writes
            // packedCount * kvBlockSize valid columns, all of them active
            // sabi entries.

            // Scrub the cube-OOB workspace cols
            // [bmm2K, singleProcessSInnerSizeNow) to -10000 *before* softmax on
            // the global last batch. Out-of-line helper to keep the hot
            // Res1VecCompute layout intact (I-cache stability).
            if (params->isBlockSparse && params->bmm2K < params->singleProcessSInnerSizeNow) {
                this->ApplyTailKOOBPreMask(
                    ubPingpong, params->bmm2K,
                    params->mm1SingleCoreN, static_cast<uint32_t>(souterSize));
            }

            bool firstIter = params->isBlockSparse ? params->isFirstComputedIter : params->isFirstInnerIter;
            this->isSoftmaxResNeedUpdate = (firstIter ||
                                            this->softmaxSouterStepLen == 0 ||
                                            souterOffset / this->softmaxSouterStepLen >= MAX_SUBSOUTER_NUM) ?
                                            this->tilingData->promptAttentionBaseParams.isRowInvalid :
                                            this->isSoftmaxNeedUpdate[souterOffset / this->softmaxSouterStepLen];
            if (params->kernelInvalidRow) {
                this->isSoftmaxResNeedUpdate = params->kernelInvalidRow;
            }
            // softmaxflash
            const uint32_t basicSoftmaxSinner = 64;
            const uint32_t basicSoftmaxSouter = 8;
            const uint32_t basicSoftmaxK = 1024;
            if (firstIter) {
                if ((params->singleProcessSInnerBmmTail % basicSoftmaxSinner == 0)
                    && (params->singleProcessSInnerBmmTail <= basicSoftmaxK)
                    && (souterSize % basicSoftmaxSouter == 0)) {
                    this->SoftmaxBasicComputeFirstNoTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                                         softmaxSumUbSub, souterSize);
                } else {
                    this->SoftmaxComputeFirstTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                                  softmaxSumUbSub, souterSize);
                }
            } else {
                softmaxExpUbSub = this->softmaxExpUb[souterOffset * this->softmaxTypeByteNum];
                if ((params->singleProcessSInnerBmmTail % basicSoftmaxSinner == 0)
                    && (params->singleProcessSInnerBmmTail <= basicSoftmaxK)
                    && (souterSize % basicSoftmaxSouter == 0)) {
                    this->SoftmaxBasicComputeNoTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                                    softmaxSumUbSub, softmaxExpUbSub, souterSize);
                } else {
                    this->SoftmaxComputeTail(this->mmResUb[ubPingpong], softmaxMaxUbSub,
                                             softmaxSumUbSub, softmaxExpUbSub, souterSize);
                }
                PipeBarrier<PIPE_V>();
            }
            if (this->softmaxSouterStepLen != 0 && souterOffset / this->softmaxSouterStepLen < MAX_SUBSOUTER_NUM) {
                this->isSoftmaxNeedUpdate[souterOffset / this->softmaxSouterStepLen] = this->isSoftmaxResNeedUpdate;
            }

            if (noLastSoftmaxLoop) {
                // There are reusable functions (Bmm1VecInputCopyIn). Pending rectification
                if (params->useMask) {
                    attenMaskOffset += souterSize * this->attentionMaskStride;
                    attenMaskOffsetPre += souterSize * this->attentionMaskStride;
                }
                if (params->usePseShift) {  // pse pre fetch
                    if constexpr (PFAT::enablePrefix) {
                        padSize = params->isPrefixInnerIter ? params->pseShiftPadPrefixSize : params->pseShiftPadSize;
                    } else {
                        padSize = params->pseShiftPadSize;
                    }
                    pseShiftOffset += souterSize * this->pseShiftStride;
                    this->PseOrMaskCopyIn(pseShiftOffset, nextSouterSize, params->isInnerTail,
                        params->pseShiftCopyInCol, params->singleProcessSInnerBmmTail, padSize, false, nextSouterSize);
                } else if (params->useMask && params->sparseBandSelect0) {  // mask pre fetch. In non band OptimizationMode, the sparseBandSelect0 is true. Just focus on the previous useMask.
                    if constexpr (PFAT::enablePrefix) {
                        padSize = params->isPrefixInnerIter ? params->padPrefixSize : params->padSize;
                    } else {
                        padSize = params->padSize;
                    }
                    this->PseOrMaskCopyIn(attenMaskOffset, nextSouterSize, params->isInnerTail, params->maskCopyInCol,
                        params->singleProcessSInnerBmmTail, padSize, true, nextSouterSize);
                }

                // mm1 result copyIn pre fetch
                bool flag = (souterOffset != 0);
                if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
                    Bmm1MsdSqueeze(params, this->mmResUb[ubPingpong^1], nextSouterOffset, nextSouterSize, ubPingpong^1, params->gmPingpong);
                } else {
                    this->Bmm1GmResCopyInUb(this->mmResUb[ubPingpong^1], nextMm1ResGmOffset,
                        nextSouterSize, params->singleProcessSInnerSizeNow / this->softmaxTypeByteNum,
                        (params->mm1SingleCoreN - params->singleProcessSInnerSizeNow) / this->softmaxTypeByteNum,
                        params->gmPingpong, ubPingpong^1, nextSouterSize, params->isInnerTail,
                        params->singleProcessSInnerSizeNow, params->singleProcessSInnerBmmTail, flag);
                }
            }

            if constexpr (IsSameType<T, int8_t>::value) {
                LocalTensor<int8_t> softmaxQuantResUb;
                softmaxQuantResUb = this->mmResUb[ubPingpong].template ReinterpretCast<int8_t>();
                softmaxQuantResUb.SetSize(this->mmResUb[ubPingpong].GetSize());
                this->QuantCompute(softmaxQuantResUb, this->mmResUb[ubPingpong], this->quantScale1, 0, souterSize * params->singleProcessSInnerSizeNow);
                // Synchronize vector quant calculation before copying, while modifying copy parameters to int8 attribute.
                event_t enQueEvtId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(enQueEvtId);
                WaitFlag<HardEvent::V_MTE3>(enQueEvtId);
                this->mm1GmUbCopyParam[ubPingpong].blockLen = params->singleProcessSInnerSizeNow / this->typeByteNum;
                this->mm1GmUbCopyParam[ubPingpong].dstStride = (params->mm1SingleCoreN - params->singleProcessSInnerSizeNow) / this->typeByteNum;
                this->mm1GmUbCopyParam[ubPingpong].srcStride = 0;
                DataCopy(this->quant1ResGmDb[params->gmPingpong][mm1ResGmOffset], softmaxQuantResUb, this->mm1GmUbCopyParam[ubPingpong]);
                SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[ubPingpong]);
            } else if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value and PFAT::msdMode == MsdMode::MSD_ON) {
                Bmm2MsdExpand(params, this->mmResUb[ubPingpong], this->bmm1ExpandGm[params->gmPingpong], souterOffset, souterSize, params->gmPingpong); 
            } else {
                // softmax res copyOut
                this->SoftmaxResCopyOut(this->mmResUb[ubPingpong], mm1ResGmOffset, params->gmPingpong, ubPingpong, params->singleProcessSInnerSizeNow);
            }

            mm1ResGmOffset = nextMm1ResGmOffset;
            ubPingpong ^= 1; // change ping/pong ub
        }
    }

    __aicore__ inline void Bmm2ComputeIterate(const uint32_t BIdx, const uint32_t NIdx, const uint32_t sInnerOffsetDataSize) {
        PFAComputeParam *params = this->headParams;

        uint32_t mm2KaStride = params->mm1SingleCoreN;
        if constexpr (PFAT::MM_TYPE == MatMulType::MM_IBSHARE_NORM && (IsSameType<T, half>::value && PFAT::calcMode == OptimizationMode::HighPerformance)) {
            mm2KaStride = params->isLastInnerIter && params->singleProcessSInnerBmmTail <= SINGLE_PROCESS_SINNER_BMMTAIL_LIMIT 
                        ? params->mm2SingleKAlign 
                        : params->mm1SingleCoreN;
        }
        if ((this->mm2MStridePrev != params->singleProcessSOuterSize)
            || (this->mm2KaStridePrev != mm2KaStride)) {
            // Reducing the number of SetOrgShape calls can decrease the frequency of CV communication.
            this->bmm2.SetOrgShape(params->singleProcessSOuterSize * this->msdIterNum,  // M stride for trans a
                this->tilingData->bmm2TilingDataRect.N,   // N stride for b
                mm2KaStride,  // Ka stride for a
                this->tilingData->bmm2TilingDataRect.Kb,   // Kb stride for trans b
                this->tilingData->promptAttentionBaseParams.headSize);  // Kc
            this->mm2MStridePrev = params->singleProcessSOuterSize * this->msdIterNum;
            this->mm2KaStridePrev = mm2KaStride;
        }
        if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
            this->bmm2LocalInfo = this->PABmm2UB.template Get<uint32_t>();
            this->bmm2LocalInfo.SetValue(0, BIdx);
            this->bmm2LocalInfo.SetValue(1, NIdx / this->tilingData->promptAttentionBaseParams.headNumRatio);
            this->bmm2LocalInfo.SetValue(2, sInnerOffsetDataSize);  // 2: sinner offset
            this->bmm2LocalInfo.SetValue(3, (uint32_t)((reinterpret_cast<int64_t>(this->currentValue)>>32) & 0x00000000ffffffff));  // 3: The high position of the pointer value.  32: Shift right by 32 bits.
            this->bmm2LocalInfo.SetValue(4, (uint32_t)(reinterpret_cast<int64_t>(this->currentValue)));  // 4: The low position of the pointer value.
            this->bmm2LocalInfo.SetValue(5, (uint32_t)((reinterpret_cast<int64_t>(this->blocktable_ptr)>>32) & 0x00000000ffffffff));  // 5: The high position of the pointer blocktable.  32: Shift right by 32 bits.
            this->bmm2LocalInfo.SetValue(6, (uint32_t)(reinterpret_cast<int64_t>(this->blocktable_ptr)));  // 6:  The low position of the pointer blocktable.

            event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

            DataCopy(this->bmm2CBDataGm[params->gmPingpong], this->bmm2LocalInfo, 8);  // 8: alignment

            event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);

            this->bmm2.SetSelfDefineData(reinterpret_cast<int64_t>(this->bmm2CBDataPtr[params->gmPingpong]));
       }

        if (this->isKvContinuous == 0) {
            ListTensorDesc valueListDesc((__gm__ void*)this->value_ptr);
            __gm__ uint8_t* tempValueGm = (__gm__ uint8_t*)valueListDesc.GetDataPtr<__gm__ uint8_t>(params->taskBatch);
            this->valueGm.SetGlobalBuffer((__gm__ KV_T*)tempValueGm);
        }

        if constexpr (IsSameType<T, int8_t>::value) {
            this->bmm2.SetQuantScalar(this->dequantScale2);
        }

        if (params->isBlockSparse) {
            // Mirrors Bmm1ComputeIterate above.
            //
            // Contiguous-K fast path: when packed sabis are consecutive, the V
            // slice is contiguous → ONE bmm2 IterateAll of K = params->bmm2K
            // (= totalColsActual, unaligned). Cube-OOB workspace cols
            // [bmm2K, mm1SingleCoreN) are pre-masked to -10000 in Res1VecCompute
            // before softmax, so softmax's fast-path runs on the full 64-aligned
            // totalCols. bmm2's V load uses bmm2K = totalColsActual, so V reads
            // stay strictly inside actualSeqLengthKVPerBatch.
            //
            // Gather slow path (non-contiguous sabis): packedCount narrow
            // K = kvBlockSize bmm2 calls, each consuming the matching slice
            // from bmm1ResGmDb and atomic-adding into bmm2ResGmDb.
            // Intermediate iters use sync=true — back-to-back IterateAll<false>
            // hangs the cube on this CANN version, same as bmm1.
            const uint32_t souterM = params->singleProcessSOuterSize * this->msdIterNum;
            const uint32_t headSize = this->tilingData->promptAttentionBaseParams.headSize;
            const int64_t kvTokenStride =
                (PFAT::layout == PFALayout::BNSD)
                ? (int64_t)headSize
                : (int64_t)this->MultiHeadKV;
            // Per-sabi-entry token stride: 128 for sabi-at-128 mode, 512 for sabi-at-512.
            // Direct member access (mirrors Bmm1ComputeIterate).
            const int64_t kvBlockSize =
                (int64_t)this->tilingData->promptAttentionBaseParams.kvBlockSize;
            // params->valueOffset holds the per-batch-head base (valueCoreOffset),
            // captured at enqueue time. See Bmm1ComputeIterate for the same fix.
            const int64_t valBase = params->valueOffset;
            const uint16_t firstSabi = params->packedSabi[0];
            const uint16_t lastSabi  = params->packedSabi[params->packedCount - 1];
            const bool kContiguous   = ((uint32_t)lastSabi - (uint32_t)firstSabi + 1u
                                        == (uint32_t)params->packedCount);
            // Mirror Bmm1ComputeIterate: per-sub-tile offset is pre-baked into
            // params->valueOffset by the wide-KV packer branch; no extra term
            // needed here, keeping the hot path byte-identical.
            if (kContiguous) {
                const int64_t valOffset = valBase + (int64_t)firstSabi * kvBlockSize * kvTokenStride;
                this->bmm2.SetTail(souterM, headSize, params->bmm2K);
                if constexpr (IsSameType<T, int8_t>::value) {
                    this->bmm2.SetTensorA(this->quant1ResGmDb[params->gmPingpong]);
                } else if constexpr (PFAT::msdMode != MsdMode::MSD_ON
                                     && (PFAT::calcMode == OptimizationMode::HighPrecision
                                         || IsSameType<T, bfloat16_t>::value)) {
                    uint64_t gmSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize *
                                      this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
                    GlobalTensor<T> tmpBmm1ResGmDb;
                    tmpBmm1ResGmDb.SetGlobalBuffer(
                        reinterpret_cast<__gm__ T*>(this->bmm1ResGmDb[params->gmPingpong].address_), gmSize);
                    this->bmm2.SetTensorA(tmpBmm1ResGmDb);
                } else if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
                    this->bmm2.SetTensorA(this->bmm1ExpandGm[params->gmPingpong]);
                } else {
                    this->bmm2.SetTensorA(this->bmm1ResGmDb[params->gmPingpong]);
                }
                if constexpr (PFAT::msdMode != MsdMode::MSD_ON
                              && (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value)) {
                    this->bmm2.SetTensorB(this->valueGmAntiquant[valOffset]);
                } else if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
                    this->bmm2.SetTensorB(this->valueGm);
                } else {
                    this->bmm2.SetTensorB(this->valueGm[valOffset]);
                }
                this->bmm2.template IterateAll<false>(
                    this->bmm2ResGmDb[params->gmPingpong], 0, false, true, params->fakeMsg);
                return;
            }
            for (int i = 0; i < (int)params->packedCount; ++i) {
                const int64_t valOffset = valBase
                    + (int64_t)params->packedSabi[i] * kvBlockSize * kvTokenStride;
                // Last iter uses bmm2K (= totalColsActual), not mm1SingleCoreN,
                // to keep V reads inside actualSeqLengthKVPerBatch.
                uint32_t kCols = (i == (int)params->packedCount - 1)
                    ? (params->bmm2K - (uint32_t)i * (uint32_t)kvBlockSize)
                    : (uint32_t)kvBlockSize;
                this->bmm2.SetTail(souterM, headSize, kCols);
                if constexpr (IsSameType<T, int8_t>::value) {
                    this->bmm2.SetTensorA(this->quant1ResGmDb[params->gmPingpong][(int64_t)i * kvBlockSize]);
                } else if constexpr (PFAT::msdMode != MsdMode::MSD_ON
                                     && (PFAT::calcMode == OptimizationMode::HighPrecision
                                         || IsSameType<T, bfloat16_t>::value)) {
                    uint64_t gmSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize *
                                      this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
                    GlobalTensor<T> tmpBmm1ResGmDb;
                    tmpBmm1ResGmDb.SetGlobalBuffer(
                        reinterpret_cast<__gm__ T*>(this->bmm1ResGmDb[params->gmPingpong].address_),
                        gmSize);
                    this->bmm2.SetTensorA(tmpBmm1ResGmDb[(int64_t)i * kvBlockSize]);
                } else if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
                    this->bmm2.SetTensorA(this->bmm1ExpandGm[params->gmPingpong][(int64_t)i * kvBlockSize]);
                } else {
                    this->bmm2.SetTensorA(this->bmm1ResGmDb[params->gmPingpong][(int64_t)i * kvBlockSize]);
                }
                if constexpr (PFAT::msdMode != MsdMode::MSD_ON
                              && (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value)) {
                    this->bmm2.SetTensorB(this->valueGmAntiquant[valOffset]);
                } else if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
                    this->bmm2.SetTensorB(this->valueGm);
                } else {
                    this->bmm2.SetTensorB(this->valueGm[valOffset]);
                }
                bool isLast = (i == (int)params->packedCount - 1);
                uint8_t enAtomic = (i == 0) ? 0 : 1;
                auto &bmm2Buf = this->bmm2ResGmDb[params->gmPingpong];
                if (!isLast) {
                    this->bmm2.template IterateAll<true>(
                        bmm2Buf, enAtomic, false, false, params->fakeMsg);
                } else {
                    this->bmm2.template IterateAll<false>(
                        bmm2Buf, enAtomic, false, true, params->fakeMsg);
                }
            }
            return;
        }

        this->bmm2.SetTail(params->singleProcessSOuterSize * this->msdIterNum,
            this->tilingData->promptAttentionBaseParams.headSize, params->singleProcessSInnerBmmTail);
        if constexpr (IsSameType<T, int8_t>::value) {
            this->bmm2.SetTensorA(this->quant1ResGmDb[params->gmPingpong]);
        } else if constexpr (PFAT::msdMode != MsdMode::MSD_ON and (PFAT::calcMode == OptimizationMode::HighPrecision ||
            IsSameType<T, bfloat16_t>::value)) {
            uint64_t gmSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize *
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
            GlobalTensor<T> tmpBmm1ResGmDb;
            tmpBmm1ResGmDb.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(this->bmm1ResGmDb[params->gmPingpong].address_), gmSize);
            this->bmm2.SetTensorA(tmpBmm1ResGmDb);
        } else if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
            this->bmm2.SetTensorA(this->bmm1ExpandGm[params->gmPingpong]);
        }
        else {
            this->bmm2.SetTensorA(this->bmm1ResGmDb[params->gmPingpong]);
        }

        if constexpr (PFAT::msdMode != MsdMode::MSD_ON and (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value)) {
            this->bmm2.SetTensorB(this->valueGmAntiquant);
        } else if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
            this->bmm2.SetTensorB(this->valueGm);
        } else if constexpr (PFAT::enablePrefix) {
            if (params->isPrefixInnerIter) {
                this->bmm2.SetTensorB(this->valueSharedPrefixGm[params->valueOffset]);
            } else {
                this->bmm2.SetTensorB(this->valueGm[params->valueOffset]);
            }
        } else {
            this->bmm2.SetTensorB(this->valueGm[params->valueOffset]);
        }

        this->bmm2.template IterateAll<false>(this->bmm2ResGmDb[params->gmPingpong], false, false, true, params->fakeMsg);
    }

    __aicore__ inline LocalTensor<computeType> AllocBmm2UbRes(PFAComputeParam *params, bool useTbuf, uint32_t& resShapeSize) {
        LocalTensor<computeType> bmm2ResUb;
        // Optimize for small Q_S using the actual required Q_S.
        uint32_t neededSouterSize = params->singleProcessSOuterSize;
        if (this->tilingData->promptAttentionBaseParams.seqSize < neededSouterSize) {
            neededSouterSize = this->tilingData->promptAttentionBaseParams.seqSize;
        }

        if (useTbuf) {
            bmm2ResUb = this->tempBmm2Ub.template Get<computeType>(this->bmm2ResUbSize);
        } else {
            bmm2ResUb = this->tempBmm2Queue.template AllocTensor<computeType>();
        }

        resShapeSize = neededSouterSize * this->tilingData->promptAttentionBaseParams.headSize;
        return bmm2ResUb;
    }

    __aicore__ inline void CopyParamsAttrOutOfInnerLoop(PFAComputeParam *dst, PFAComputeParam *src) {
        dst->isFirstInnerIter = src->isFirstInnerIter;
        dst->isSecondInnerIter = src->isSecondInnerIter;
        dst->useMask = src->useMask;
        dst->usePseShift = src->usePseShift;
        dst->singleProcessSOuterSize = src->singleProcessSOuterSize;
        dst->singleProcessSInnerSize = src->singleProcessSInnerSize;
        dst->singleProcessSInnerSizeTail = src->singleProcessSInnerSizeTail;
        dst->maskCopyInCol = src->maskCopyInCol;
        dst->maskInnerTailAlign = src->maskInnerTailAlign;
        dst->padSize = src->padSize;
        dst->pseShiftCopyInCol = src->pseShiftCopyInCol;
        dst->pseShiftInnerTailAlign = src->pseShiftInnerTailAlign;
        dst->pseShiftPadSize = src->pseShiftPadSize;

        // For block sparsity
        dst->isBlockSparse = src->isBlockSparse;
        dst->isFirstComputedIter = src->isFirstComputedIter;
        dst->isSecondComputedIter = src->isSecondComputedIter;
        dst->isLastComputedIter = src->isLastComputedIter;
        dst->packedSabi[0] = src->packedSabi[0];
        dst->packedSabi[1] = src->packedSabi[1];
        dst->packedSabi[2] = src->packedSabi[2];
        dst->packedSabi[3] = src->packedSabi[3];
        dst->packedCount = src->packedCount;

        dst->unalignSInner = src->unalignSInner;
        dst->tensorAOffset = src->tensorAOffset;
        dst->attentionOutOffset = src->attentionOutOffset;
        dst->batchNOffset = src->batchNOffset;
        dst->sOuterOffset = src->sOuterOffset;
        dst->sInnerLoopOffset = src->sInnerLoopOffset;
        dst->multiSeqOffset = src->multiSeqOffset;
        dst->multiSeqOffsetBSNDOut = src->multiSeqOffsetBSNDOut;
        dst->SoftMaxOffset = src->SoftMaxOffset;
        dst->sInnerOffsetDataSize = src->sInnerOffsetDataSize;
        dst->taskBatch = src->taskBatch;
        dst->preTokensPerBatch = src->preTokensPerBatch;
        dst->nextTokensPerBatch = src->nextTokensPerBatch;
        dst->actualSeqLengthPerBatch = src->actualSeqLengthPerBatch;
        dst->actualSeqLengthKVPerBatch = src->actualSeqLengthKVPerBatch;
        dst->fakeMsg = src->fakeMsg;
        if constexpr (PFAT::enablePrefix) {
            dst->singleProcessSInnerPrefixSizeTail = src->singleProcessSInnerPrefixSizeTail;
            dst->maskInnerPrefixTailAlign = src->maskInnerPrefixTailAlign;
            dst->padPrefixSize = src->padPrefixSize;
            dst->pseShiftInnerPrefixTailAlign = src->pseShiftInnerPrefixTailAlign;
            dst->pseShiftPadPrefixSize = src->pseShiftPadPrefixSize;
            dst->unalignSInnerPrefix = src->unalignSInnerPrefix;
            dst->isPrefixInnerIter = src->isPrefixInnerIter;
        }

        dst->antiqParamOffsetPerToken = src->antiqParamOffsetPerToken;
    }
};

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::AllocGlobalResources() {
    for (int i = 0; i < 2; ++i) {
        this->mmResUb[i] = this->Bmm1Queue.template AllocTensor<computeType>();
    }
    for (int i = 0; i < 2; ++i) {
        this->bmm1ResCopyInEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        this->bmm1ResCopyOutEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        this->bmm2ResCopyInEvent[i] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
    }
    this->attenOutCopyOut = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    this->softmaxMaxUb = this->softmaxOutQueue.template AllocTensor<float>();
    this->softmaxSumUb = this->softmaxMaxUb[this->tilingData->promptAttentionTensorSizeRect.softmaxMaxSize];
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::FreeGlobalResources() {
    this->softmaxOutQueue.FreeTensor(this->softmaxMaxUb);

    for (int i = 0; i < 2; ++i) {
        WaitFlag<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_MTE2>(this->bmm1ResCopyOutEvent[i]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[i]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(this->attenOutCopyOut);
    for (int i = 0; i < 2; ++i) {
        this->Bmm1Queue.FreeTensor(this->mmResUb[i]);
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Process() {

    AllocGlobalResources();

    // Invalidate data cache (used by scalar units, might still be polluted from previous kernels)
    AscendC::DataCacheCleanAndInvalid<uint16_t,
                                      AscendC::CacheLine::ENTIRE_DATA_CACHE,
                                      AscendC::DcciDst::CACHELINE_OUT>(this->sabiBaseGm);

    if constexpr (PFAT::MM_TYPE == MatMulType::MM_IBSHARE_NORM) {
        if (this->tilingData->promptAttentionInitOutputParams.isOneN) {
            ComputeEachCoreSplitSeqOneN(this->tmp_block_idx);
        } else {
            ComputeEachCore(this->tmp_block_idx);
        }
    } else {
        if (this->headNumRatio != 1 || this->tilingData->promptAttentionInitOutputParams.needInit ||
            this->tilingData->promptAttentionBaseParams.batchSize != 1) {
            ComputeEachCore(this->tmp_block_idx);
        }
        else {
            ComputeEachCoreBalance(this->tmp_block_idx);
        }
    }

    // Clear the remaining parameters of the queue.
    while (this->queSize > 0) {
        this->queSize--;
        ComputeEachCoreSInnerLoop();

        this->preHeadParams = this->headParams;

        // Out of queue
        this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
        this->headParams = &this->pfaParamsQueue[this->headId];
    }
    ProcessLastSouterLoopFinalRes();

    FreeGlobalResources();
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::PseOrMaskCopyIn(int64_t offset, uint32_t souterSize,
    bool isInnerTail, uint32_t alignSInner, uint32_t unalignSInner, uint32_t padSize, bool isMask, uint32_t souterTailMaskCopyInSize) {
    // Optimize for small Q_S using the actual required Q_S.
    uint32_t neededSouterSize = souterTailMaskCopyInSize;
    if (this->tilingData->promptAttentionBaseParams.seqSize < neededSouterSize) {
        neededSouterSize = this->tilingData->promptAttentionBaseParams.seqSize;
    }

    uint32_t lenOfType = 1;  // The length of each data.
    uint32_t dataStride = 0; // stride size

    if (isMask) {  // pse and mask reuse this function, reuse ub.
        this->attenMaskUb = this->tempBmm2Queue.template AllocTensor<U>();
        this->attenMaskUb.SetSize(souterSize * alignSInner);
        lenOfType = sizeof(U);
        dataStride = this->attentionMaskStride;
    } else {
        this->pseShiftUb = this->tempBmm2Queue.template AllocTensor<pseShiftType>();
        this->pseShiftUb.SetSize(souterSize * alignSInner);
        lenOfType = sizeof(pseShiftType);
        dataStride = this->pseShiftStride;
    }

    DataCopyExtParams intriParams;
    intriParams.blockCount = neededSouterSize;  // This should be non aligned.
    intriParams.blockLen = alignSInner * lenOfType;
    intriParams.srcStride = (dataStride - alignSInner) * lenOfType;
    if (isInnerTail) {
        intriParams.blockLen = unalignSInner * lenOfType;
        intriParams.srcStride = (dataStride - unalignSInner) * lenOfType;
    }
    intriParams.dstStride = 0;

    if (isMask) {
        DataCopyPadExtParams<U> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 1;
        if (isInnerTail) {
            padParams.rightPadding = padSize;
        } else {
            padParams.rightPadding = 0;
        }
        DataCopyPad(this->attenMaskUb, this->attenMaskGm[offset], intriParams, padParams);
        this->tempBmm2Queue.template EnQue<U>(this->attenMaskUb);
    } else {
        DataCopyPadExtParams<pseShiftType> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        padParams.paddingValue = 1;
        if (isInnerTail) {
            padParams.rightPadding = padSize;
            if constexpr (IsSameType<T, int8_t>::value) {
                if (((intriParams.blockLen / lenOfType + padSize) % 32) != 0) {
                    intriParams.dstStride = 1;  // If qkv is int8，the length of the pad differs by one block and needs to be skipped for storage.
                }
            }
        } else {
            padParams.rightPadding = 0;
        }

        DataCopyPad(this->pseShiftUb, this->pseShiftGm[offset], intriParams, padParams);
        this->tempBmm2Queue.template EnQue<pseShiftType>(this->pseShiftUb);
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1VecInputCopyIn() {
    PFAComputeParam *params = this->headParams;
    if (params->fakeMsg) {
        return;
    }
    this->softmaxSouterStepLen = this->tilingData->promptAttentionBaseParams.softmaxOuterSize;

    // Optimize the tail block. Number of softmax cycles
    if (this->softmaxFlashTilingData.srcK != params->singleProcessSInnerSizeNow) {
        uint32_t minSoftmaxSouterStepLen = this->softmaxFlashTilingData.srcSize / params->singleProcessSInnerSizeNow / 8 * 8; // 8 alignment
        if (params->useMask) {  // When D<=64，maskubsize maybe greater than bmm2ubsize，It will only be divided into 16k by maskubsize. When mask padding size > sinner padding size，will lead to unauthorized access.
            uint32_t maskSouter = this->maskBmm2ShareSize / params->maskCopyInCol / 8 * 8;  // 8对齐
            minSoftmaxSouterStepLen = (minSoftmaxSouterStepLen < maskSouter) ? minSoftmaxSouterStepLen : maskSouter;
        }
        this->softmaxSouterStepLen = minSoftmaxSouterStepLen;
        this->softmaxSouterStepLen = ((this->softmaxSouterStepLen > params->singleProcessSOuterSize) ||
        (this->softmaxSouterStepLen == 0)) ?
        params->singleProcessSOuterSize : this->softmaxSouterStepLen;
    }

    if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
        if (this->tilingData->promptAttentionBaseParams.headSize > 256) { // 256:big d for msd OptimizationMode
            uint32_t msdComputeLinesTiling = this->tilingData->promptAttentionTensorSizeRect.msdComputeLines;
            this->softmaxSouterStepLen = this->softmaxSouterStepLen > msdComputeLinesTiling ? msdComputeLinesTiling : this->softmaxSouterStepLen;
        } else {
            uint32_t maxSouterSizeTmp = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize / params->singleProcessSInnerSizeNow;
            this->softmaxSouterStepLen = this->softmaxSouterStepLen > maxSouterSizeTmp ? maxSouterSizeTmp : this->softmaxSouterStepLen;
        }
        this->softmaxSouterStepLen = (this->softmaxSouterStepLen > params->singleProcessSOuterSize)? params->singleProcessSOuterSize : this->softmaxSouterStepLen;
    }
    uint32_t souterSize = this->softmaxSouterStepLen;
    uint32_t souterTailMaskCopyInSize = ClipSInnerToken(souterSize, 0, params->singleProcessSOuterSize);
    uint32_t padSize = 0;
    if (params->usePseShift) {
        if constexpr (PFAT::enablePrefix) {
            padSize = params->isPrefixInnerIter ? params->pseShiftPadPrefixSize : params->pseShiftPadSize;
        } else {
            padSize = params->pseShiftPadSize;
        }
        this->PseOrMaskCopyIn(params->pseShiftOffset, souterSize, params->isInnerTail, params->pseShiftCopyInCol,
            params->singleProcessSInnerBmmTail, padSize, false, souterTailMaskCopyInSize);
    } else if (params->useMask && params->sparseBandSelect0) {  // In non band OptimizationMode, sparseBandSelect0 is true. Just focus on what's ahead useMask.
        if constexpr (PFAT::enablePrefix) {
            padSize = params->isPrefixInnerIter ? params->padPrefixSize : params->padSize;
        } else {
            padSize = params->padSize;
        }
        this->PseOrMaskCopyIn(params->attenMaskOffset, souterSize, params->isInnerTail, params->maskCopyInCol,
            params->singleProcessSInnerBmmTail, padSize, true, souterTailMaskCopyInSize);
    }

    if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
        Bmm1MsdSqueeze(params, this->mmResUb[0], 0, souterSize, 0, params->gmPingpong);
    } else {
        this->Bmm1GmResCopyInUb(this->mmResUb[0], 0,
            souterSize, params->singleProcessSInnerSizeNow / this->softmaxTypeByteNum,
            (params->mm1SingleCoreN - params->singleProcessSInnerSizeNow) / this->softmaxTypeByteNum,
            params->gmPingpong, 0, souterSize, params->isInnerTail, params->singleProcessSInnerSizeNow,
            params->singleProcessSInnerBmmTail, false);
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::SparseBandElewiseCompute(
    int32_t ubPingpong, uint32_t souterSize, int64_t attenMaskOffsetPre)
{
    PFAComputeParam *params = this->headParams;
    if (params->sparseBandSelect0) {    // Select 0 part
        this->template ElewiseCompute<U>(
            this->mmResUb[ubPingpong], souterSize, params->singleProcessSInnerSizeNow,
            params->maskCopyInCol, params->useMask, this->bmm1ResCopyInEvent[ubPingpong], 0);
    }
    if (params->sparseBandSelect1) {    // Select 1 part
        uint32_t padSize = params->padSize;
        if constexpr (PFAT::enablePrefix) {
            padSize = params->isPrefixInnerIter ? params->padPrefixSize : params->padSize;
        }
        this->PseOrMaskCopyIn(attenMaskOffsetPre, souterSize, params->isInnerTail, params->maskCopyInCol,
            params->singleProcessSInnerBmmTail, padSize, true, souterSize);

        PipeBarrier<PIPE_V>();
        this->template ElewiseCompute<U>(
            this->mmResUb[ubPingpong], souterSize, params->singleProcessSInnerSizeNow,
            params->maskCopyInCol, params->useMask, this->bmm1ResCopyInEvent[ubPingpong], 1);
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdRowSum(LocalTensor<FT>& dstUb, LocalTensor<FT> srcUb,
                                                                         uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t dtype_mask = REPEAT_BLOCK_BYTE_PFA / sizeof(FT);
    uint32_t blockCount = actualColumnCount / dtype_mask;
    uint32_t remain = actualColumnCount % dtype_mask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / (BYTE_BLOCK_PFA / sizeof(FT));
    repeatParamsMax.src1RepStride = columnCount / (BYTE_BLOCK_PFA / sizeof(FT));
    repeatParamsMax.dstRepStride = columnCount / (BYTE_BLOCK_PFA / sizeof(FT));
    if (blockCount > 0 && remain > 0) {
        Add(srcUb, srcUb, srcUb[blockCount * dtype_mask], remain, dealRowCount, repeatParamsMax);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
        blockCount = (blockCount + 1) / 2;
        for (uint32_t j = 0; j < loopCount; j++) {
        Add(srcUb[j * dtype_mask], srcUb[j * dtype_mask], srcUb[(j + blockCount) * dtype_mask], dtype_mask, dealRowCount,
            repeatParamsMax);
        }
        PipeBarrier<PIPE_V>();
    }

    WholeReduceSum(dstUb, srcUb, (actualColumnCount < dtype_mask) ? actualColumnCount : dtype_mask, dealRowCount, 1,
                    1, columnCount / (BYTE_BLOCK_PFA / sizeof(FT)));
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdRowMax(LocalTensor<FT>& aMaxDstUb, LocalTensor<FT>& srcUb,
                                                                         uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t dtype_mask = this->MSD_REPEAT_ELEMENT_NUM;
    uint32_t blockCount = actualColumnCount / dtype_mask;
    uint32_t remain = actualColumnCount % dtype_mask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    repeatParamsMax.dstRepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    if (blockCount > 0 && remain > 0) {
        Max(srcUb, srcUb, srcUb[blockCount * dtype_mask], remain, dealRowCount, repeatParamsMax);
        PipeBarrier<PIPE_V>();
    }

    for (uint32_t loopCount = blockCount / 2; loopCount > 0; loopCount = blockCount / 2) {
        blockCount = (blockCount + 1) / 2;
        for (uint32_t j = 0; j < loopCount; j++) {
            Max(srcUb[j * dtype_mask], srcUb[j * dtype_mask], srcUb[(j + blockCount) * dtype_mask], dtype_mask, dealRowCount,
                repeatParamsMax);
        }
        PipeBarrier<PIPE_V>();
    }

    WholeReduceMax(aMaxDstUb, srcUb, (actualColumnCount < dtype_mask) ? actualColumnCount : dtype_mask, dealRowCount, 1,
                    1, columnCount / this->MSD_BLOCK_ELEMENT_NUM, ReduceOrder::ORDER_ONLY_VALUE);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdAbsRowMax(LocalTensor<FT>& tmpAMaxRes, LocalTensor<FT>& srcUb, LocalTensor<FT> tmpAUb, 
                                                                            uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    Abs(tmpAUb, srcUb, dealRowCount * columnCount);
    PipeBarrier<PIPE_V>();
    LocalTensor<FT> tmpRowMaxUb = this->msdAMaxTmpBuff.template Get<FT>();
    MsdRowMax(tmpRowMaxUb, tmpAUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    Brcb(tmpAMaxRes, tmpRowMaxUb, (dealRowCount + 7) / 8, {1, 8});
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdRowMuls(LocalTensor<FT>& dstUb, LocalTensor<FT>& src0Ub, LocalTensor<FT>& src1Ub, 
                                                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t dtype_mask = this->MSD_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtype_mask;
    uint32_t dRemain = actualColumnCount % dtype_mask;

#if (defined(MSD_MUL_TYPE_BY_ROW) && (MSD_MUL_TYPE_BY_ROW == 1))
    BinaryRepeatParams columnRepeatParams;
    columnRepeatParams.src0BlkStride = 1;
    columnRepeatParams.src1BlkStride = 0;
    columnRepeatParams.dstBlkStride = 1;
    columnRepeatParams.src0RepStride = 8;
    columnRepeatParams.src1RepStride = 0;
    columnRepeatParams.dstRepStride = 8;
    for (uint32_t i = 0; i < dealRowCount; i++) {
        Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * this->MSD_BLOCK_ELEMENT_NUM], 
            dtype_mask, dLoop, columnRepeatParams);
        if (dRemain > 0) {
            Mul(dstUb[dLoop * dtype_mask + i * columnCount], src0Ub[dLoop * dtype_mask + i * columnCount], src1Ub[i * this->MSD_BLOCK_ELEMENT_NUM], 
                dRemain, 1, columnRepeatParams);
        }
    }
#else 
    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 0;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0RepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;

    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            // offset = i * dtype_mask
            Mul(dstUb[offset], src0Ub[offset], src1Ub, 
                dtype_mask, dealRowCount, repeatParams);
            offset += dtype_mask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8;
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * this->MSD_BLOCK_ELEMENT_NUM], 
                dtype_mask, columnRepeatCount, columnRepeatParams);
        }
    }

    if (dRemain > 0) {
        Mul(dstUb[dLoop * dtype_mask], src0Ub[dLoop * dtype_mask], src1Ub, 
            dRemain, dealRowCount, repeatParams);
    }
#endif
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdAntiParamsMulByRow(LocalTensor<FT>& dstUb, LocalTensor<FT>& antiParams, LocalTensor<FT>& src1Ub, 
                                                                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t dtype_mask = this->MSD_REPEAT_ELEMENT_NUM;  
    uint32_t dLoop = actualColumnCount / dtype_mask;    
    uint32_t dRemain = actualColumnCount % dtype_mask;   

    BinaryRepeatParams columnRepeatParams;
#if (defined(MSD_MUL_TYPE_BY_ROW) && (MSD_MUL_TYPE_BY_ROW == 1))
    columnRepeatParams.src0BlkStride = 1;
    columnRepeatParams.src1BlkStride = 0;
    columnRepeatParams.dstBlkStride = 1;
    columnRepeatParams.src0RepStride = 8;
    columnRepeatParams.src1RepStride = 0;
    columnRepeatParams.dstRepStride = 8;
    for (uint32_t i = 0; i < dealRowCount; i++) {
        Mul(dstUb[i * columnCount], antiParams, src1Ub[i * this->MSD_BLOCK_ELEMENT_NUM], 
            dtype_mask, dLoop, columnRepeatParams);
        if (dRemain > 0) {
            Mul(dstUb[dLoop * dtype_mask + i * columnCount], antiParams[dLoop * dtype_mask], src1Ub[i * this->MSD_BLOCK_ELEMENT_NUM], 
                dRemain, 1, columnRepeatParams);
        }
    }
#else
    columnRepeatParams.src0RepStride = 1;
    columnRepeatParams.src0BlkStride = 0;
    columnRepeatParams.src1RepStride = 0;
    columnRepeatParams.src1BlkStride = 1;
    columnRepeatParams.dstRepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    columnRepeatParams.dstBlkStride = 1;
    for (int i = 0; i < dLoop; i++) {
        Mul(dstUb[i * dtype_mask], src1Ub, antiParams[i * dtype_mask], 
            dtype_mask, dealRowCount, columnRepeatParams);
    }
    if (dRemain > 0) {
        Mul(dstUb[dLoop * dtype_mask], src1Ub, antiParams[dLoop * dtype_mask], 
            dRemain, dealRowCount, columnRepeatParams);
    }
#endif
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdFusedMulAddByRow(LocalTensor<FT>& dstLocal, LocalTensor<FT>& src0Local, LocalTensor<FT>& src1Local, 
                                                                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t dtype_mask = this->MSD_REPEAT_ELEMENT_NUM;  
    uint32_t dLoop = actualColumnCount / dtype_mask;     
    uint32_t dRemain = actualColumnCount % dtype_mask;   

    BinaryRepeatParams columnRepeatParams;
#if (defined(MSD_MUL_TYPE_BY_ROW) && (MSD_MUL_TYPE_BY_ROW == 1))   
    columnRepeatParams.src0BlkStride = 0;
    columnRepeatParams.src1BlkStride = 1;
    columnRepeatParams.dstBlkStride = 1;
    columnRepeatParams.src0RepStride = 0;
    columnRepeatParams.src1RepStride = 8;
    columnRepeatParams.dstRepStride = 8;

    for (uint32_t i = 0; i < dealRowCount; i++) {
        FusedMulAdd(dstLocal[i * columnCount], 
                    src0Local[i * this->MSD_BLOCK_ELEMENT_NUM], 
                    src1Local[i * columnCount], 
                    dtype_mask, dLoop, columnRepeatParams);
        if (dRemain > 0) {
            FusedMulAdd(dstLocal[dLoop * dtype_mask + i * columnCount], 
                        src0Local[i * this->MSD_BLOCK_ELEMENT_NUM], 
                        src1Local[dLoop * dtype_mask + i * columnCount], 
                        dRemain, 1, columnRepeatParams);
        }
    }
#else
    columnRepeatParams.src0RepStride = 1;
    columnRepeatParams.src0BlkStride = 0;
    columnRepeatParams.src1RepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    columnRepeatParams.src1BlkStride = 1;
    columnRepeatParams.dstRepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    columnRepeatParams.dstBlkStride = 1;
    for (int j = 0; j < dLoop; j++) {
        FusedMulAdd(dstLocal[j * dtype_mask], src0Local, src1Local[j * dtype_mask], dtype_mask, dealRowCount, columnRepeatParams);
    }
    if (dRemain > 0) {
        FusedMulAdd(dstLocal[dLoop * dtype_mask], src0Local, src1Local[dLoop * dtype_mask], dRemain, dealRowCount, columnRepeatParams);
    }
#endif
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdMulMaxAddSumByRow(LocalTensor<FT>& dstLocal, LocalTensor<FT>& src0RowMax, LocalTensor<FT>& src1RowSum, 
                                                                                    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t dtype_mask = this->MSD_REPEAT_ELEMENT_NUM;  
    uint32_t dLoop = actualColumnCount / dtype_mask;     
    uint32_t dRemain = actualColumnCount % dtype_mask;   

    BinaryRepeatParams columnRepeatParams;
#if (defined(MSD_MUL_TYPE_BY_ROW) && (MSD_MUL_TYPE_BY_ROW == 1))     
    columnRepeatParams.src0BlkStride = 0;
    columnRepeatParams.src1BlkStride = 0;
    columnRepeatParams.dstBlkStride = 1;
    columnRepeatParams.src0RepStride = 0;
    columnRepeatParams.src1RepStride = 0;
    columnRepeatParams.dstRepStride = 8;
    for (uint32_t i = 0; i < dealRowCount; i++) {
        FusedMulAdd(dstLocal[i * columnCount], 
                    src0RowMax[i * this->MSD_BLOCK_ELEMENT_NUM], 
                    src1RowSum[i * this->MSD_BLOCK_ELEMENT_NUM], 
                    dtype_mask, dLoop, columnRepeatParams);
        if (dRemain > 0) {
            FusedMulAdd(dstLocal[dLoop * dtype_mask + i * columnCount], 
                        src0RowMax[i * this->MSD_BLOCK_ELEMENT_NUM], 
                        src1RowSum[i * this->MSD_BLOCK_ELEMENT_NUM], 
                        dRemain, 1, columnRepeatParams);
        }
    }
#else
    columnRepeatParams.src0RepStride = 1;
    columnRepeatParams.src0BlkStride = 0;
    columnRepeatParams.src1RepStride = 1;
    columnRepeatParams.src1BlkStride = 0;
    columnRepeatParams.dstRepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    columnRepeatParams.dstBlkStride = 1;
    PipeBarrier<PIPE_V>();
    for (int j = 0; j < dLoop; j++) {
        FusedMulAdd(dstLocal[j * dtype_mask], src0RowMax, src1RowSum, 
                    dtype_mask, dealRowCount, columnRepeatParams);
    }
    if (dRemain > 0) {
        FusedMulAdd(dstLocal[dLoop * dtype_mask], src0RowMax, src1RowSum, 
                    dRemain, dealRowCount, columnRepeatParams);
    }
#endif
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdVecMulMat(LocalTensor<FT>& dstUb, LocalTensor<FT>& src0Ub, LocalTensor<FT>& src1Ub,
                                                                            uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t dtype_mask = this->MSD_REPEAT_ELEMENT_NUM;  
    uint32_t dLoop = actualColumnCount / dtype_mask;     
    uint32_t dRemain = actualColumnCount % dtype_mask;   
    BinaryRepeatParams columnRepeatParams;

#if (defined(MSD_MUL_TYPE_BY_ROW) && (MSD_MUL_TYPE_BY_ROW == 1))
    columnRepeatParams.src0BlkStride = 1;
    columnRepeatParams.src1BlkStride = 1;
    columnRepeatParams.dstBlkStride = 1;
    columnRepeatParams.src0RepStride = 8;
    columnRepeatParams.src1RepStride = 8;
    columnRepeatParams.dstRepStride = 8;
    for (uint32_t i = 0; i < dealRowCount; i++) {
        Mul(dstUb[i * columnCount], src0Ub, src1Ub[i * columnCount], 
            dtype_mask, dLoop, columnRepeatParams);
        if (dRemain > 0) {
            Mul(dstUb[dLoop * dtype_mask + i * columnCount], src0Ub[dLoop * dtype_mask], src1Ub[dLoop * dtype_mask + i * columnCount], 
                dRemain, 1, columnRepeatParams);
        }
    }
#else
    columnRepeatParams.dstBlkStride = 1;
    columnRepeatParams.src0BlkStride = 1;
    columnRepeatParams.src1BlkStride = 1;
    columnRepeatParams.dstRepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    columnRepeatParams.src0RepStride = 0;
    columnRepeatParams.src1RepStride = columnCount / this->MSD_BLOCK_ELEMENT_NUM;
    int64_t offset = 0;
    for (int i = 0; i < dLoop; i++) {
        // offset = i * dtype_mask
        Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], dtype_mask, dealRowCount, columnRepeatParams);
        offset += dtype_mask;
    }
    if (dRemain > 0) {
        // offset = dLoop * dtype_mask
        Mul(dstUb[offset], src0Ub[offset], src1Ub[offset], dRemain, dealRowCount, columnRepeatParams);
    }
#endif    
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdCopyQueryGm2Ub(LocalTensor<FT> &queryUb, int64_t queryOffset,
                                                                                     uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t queryCntPerBlk = BYTE_BLOCK_PFA / sizeof(T);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = dealRowCount;
    copyInParams.blockLen = actualColumnCount * sizeof(T);
    if constexpr (PFAT::layout == PFALayout::BNSD) {
        copyInParams.srcStride = 0;  // BNSD
    } else {
        copyInParams.srcStride = (this->tilingData->promptAttentionBaseParams.headNumSize - 1) * actualColumnCount * sizeof(T);    // BSH
    }

    copyInParams.dstStride = (columnCount - actualColumnCount) / queryCntPerBlk;
    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % queryCntPerBlk;
    copyInPadParams.paddingValue = 0;

    LocalTensor<T> inputUb = this->msdInQueue.template AllocTensor<T>();
    DataCopyPad(inputUb, this->queryGm[queryOffset], copyInParams, copyInPadParams);
    this->msdInQueue.template EnQue(inputUb);
    inputUb = this->msdInQueue.template DeQue<T>();
    Cast(queryUb, inputUb, RoundMode::CAST_NONE, dealRowCount * columnCount);
    this->msdInQueue.template FreeTensor(inputUb);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdAIterExpand(GlobalTensor<int8_t> dstGm, LocalTensor<FT>& tmpA1, LocalTensor<FT>& tmpA2, 
                                                                              uint32_t calcSize, bool isFirst, int64_t outOffset, int gmPingpong) {
    if (!isFirst) {
        // sub
        Sub(tmpA1, tmpA1, tmpA2, calcSize);
        PipeBarrier<PIPE_V>();
        // muls 128
        Muls(tmpA1, tmpA1, this->msdAntiqExpandCoeff, calcSize);
        PipeBarrier<PIPE_V>();
    }

    // castFloor-fp32
    Cast(tmpA2, tmpA1, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    // cast-fp16
    LocalTensor<half> aResOutUb = this->msdOutQueue.template AllocTensor<half>();
    Cast(aResOutUb, tmpA2, RoundMode::CAST_ROUND, calcSize);
    PipeBarrier<PIPE_V>();

    // cast-int8
    LocalTensor<KV_T> aResOutUbI8 = aResOutUb.template ReinterpretCast<KV_T>();
    aResOutUbI8.SetSize(aResOutUb.GetSize());
    Cast(aResOutUbI8, aResOutUb, RoundMode::CAST_ROUND, calcSize);
    
    // copyOut Ak
    this->msdOutQueue.template EnQue(aResOutUbI8);
    this->msdOutQueue.template DeQue<KV_T>();
    DataCopyExtParams copyParams{1, calcSize, 0, 0, 0};
    DataCopyPad(dstGm[outOffset], aResOutUbI8, copyParams);
    this->msdOutQueue.FreeTensor(aResOutUbI8);
}

// 𝑄2=𝑟𝑜𝑢𝑛𝑑((127/𝑄𝑚𝑎𝑥×𝑄 −𝑄1)×254)
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdMatmulPreProcess(PFAComputeParam *params, GlobalTensor<KV_T> dstGm, LocalTensor<FT>& msdMaxResUb, LocalTensor<FT>& srcUb, LocalTensor<FT>& tmpAFloorUb,
                                                                                   uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    int64_t actualSeqLength = params->actualSeqLengthPerBatch;
    int64_t step = actualSeqLength * columnCount;
    int64_t baseOffset = startRow * columnCount;
    uint32_t calcSize = dealRowCount * columnCount;

    LocalTensor<FT> tmpAMaxRes = msdMaxResUb[startRow * this->MSD_BLOCK_ELEMENT_NUM];
    MsdAbsRowMax(tmpAMaxRes, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // 127/Amax*A
    Duplicate(tmpAFloorUb, this->msdAntiqCoeff1, dealRowCount * this->MSD_BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    Div(tmpAFloorUb, tmpAFloorUb, tmpAMaxRes, dealRowCount * this->MSD_BLOCK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    MsdRowMuls(srcUb, srcUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < this->msdIterNum; i++) {
        MsdAIterExpand(dstGm, srcUb, tmpAFloorUb, calcSize, (i == 0 ? true : false), step * i + baseOffset, gmPingpong);
    }
}

// Calc ∑𝐴[𝑗]
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdCalcQueryRowSum(LocalTensor<FT> &queryUb, 
                                                                                      uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    int64_t baseOffset = startRow * this->MSD_BLOCK_ELEMENT_NUM;
    LocalTensor<FT> rowSumUb = this->msdAMaxTmpBuff.template Get<FT>();

    MsdRowSum(rowSumUb, queryUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    
    Brcb(this->msdRowSumUb[gmPingpong][baseOffset], rowSumUb, (dealRowCount + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
}

// Copy Query 2 Ub, Calc ∑𝐴[𝑗], expend to queryMsdExpandGm
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdExpandPertoken(PFAComputeParam *params) {
    uint32_t startRow = 0;
    uint32_t dealRowCount = params->actualSeqLengthPerBatch;
    uint32_t columnCount = this->tilingData->promptAttentionBaseParams.alignedHeadSize;
    uint32_t actualColumnCount = columnCount;
    
    // 1. Copy Query 2 Ub
    int64_t queryOffset = params->tensorAOffset;
    LocalTensor<FT> queryUb = this->msdTmpMm1Buff.template Get<FT>();
    LocalTensor<FT> tmpQueryUb = this->msdTmpMm2Buff.template Get<FT>(); 
    Bmm1MsdCopyQueryGm2Ub(queryUb, queryOffset, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    // 2. 𝑆𝑐𝑎𝑙𝑒𝐾×(𝑺+𝑂𝑓𝑓𝑠𝑒𝑡𝐾×∑A[j])
    if (this->msdIsKOffsetExist) {
        Adds(tmpQueryUb, queryUb, (FT)0, dealRowCount * columnCount); 
        PipeBarrier<PIPE_V>();
        Bmm1MsdCalcQueryRowSum(tmpQueryUb, startRow, dealRowCount, columnCount, actualColumnCount, params->gmPingpong);
    }

    // 3. expand to queryMsdExpandGm
    MsdMatmulPreProcess(params, this->queryMsdExpandGm, this->msdMaxBmm1Ub[params->gmPingpong], queryUb, tmpQueryUb, 
                        startRow, dealRowCount, columnCount, actualColumnCount, params->gmPingpong);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdExpand(PFAComputeParam *params) {
    if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 0) {
        Bmm1MsdExpandPerchannel(params);
    } else if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 1) {
        Bmm1MsdExpandPertoken(params);
    }
}

// 𝑆′=(𝑆1+𝑆2×1/254 +S3×1/254^2)×1/127
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdMatmulResCombine(LocalTensor<FT> &bmmResUb, GlobalTensor<mmOutputType> srcGm, 
                                            uint32_t expandLineCnt, uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    uint32_t expandOffset = expandLineCnt * columnCount;              
    int64_t baseOffset = startRow * columnCount;                     
    uint32_t copyElementCnt = dealRowCount * columnCount;           

    FT scale = 1;
    int64_t offset = baseOffset;

    event_t copyEvt1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t copyEvt2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::V_MTE2>(copyEvt1);
    for (uint32_t i = 0; i < this->msdIterNum; i++) {
        LocalTensor<mmOutputType> tmpCInt = this->msdTmpMm2Buff.template Get<mmOutputType>();
        WaitFlag<HardEvent::V_MTE2>(copyEvt1);
        DataCopy(tmpCInt, srcGm[offset], copyElementCnt);  // offset = i * copyElementCnt + baseOffset
        SetFlag<HardEvent::MTE2_V>(copyEvt2);
        WaitFlag<HardEvent::MTE2_V>(copyEvt2);

        if (i == 0) {
            Cast(bmmResUb, tmpCInt, AscendC::RoundMode::CAST_NONE, copyElementCnt);
        } else {
            LocalTensor<FT> tmpCFp = tmpCInt.template ReinterpretCast<FT>();
            tmpCFp.SetSize(tmpCInt.GetSize());
            Cast(tmpCFp, tmpCInt, AscendC::RoundMode::CAST_NONE, copyElementCnt);
            PipeBarrier<PIPE_V>();
            Muls(tmpCFp, tmpCFp, scale, copyElementCnt);
            PipeBarrier<PIPE_V>();
            Add(bmmResUb, bmmResUb, tmpCFp, copyElementCnt);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE2>(copyEvt1);

        offset += expandOffset;
        scale = scale / this->msdAntiqExpandCoeff;
    }
    WaitFlag<HardEvent::V_MTE2>(copyEvt1);

    // muls 1/antiqCoeff1
    Muls(bmmResUb, bmmResUb, this->msdAntiqCoeff2, copyElementCnt);
    PipeBarrier<PIPE_V>();
}

// Copy antiquant params from GM->Ub
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdCopyAntiquantParamsPerToken(GlobalTensor<FT> srcGm, int64_t offset, 
                                                                                              uint32_t columnCount, uint32_t actualColumnCount) {
    uint32_t paramsTypeElementSize = BYTE_BLOCK_PFA / sizeof(FT);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<FT> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = 1;
    copyInParams.blockLen = actualColumnCount * sizeof(FT);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (columnCount - actualColumnCount) % paramsTypeElementSize;
    copyInPadParams.paddingValue = 0;

    LocalTensor<FT> paramsUb = this->msdInQueue.template AllocTensor<FT>();
    DataCopyPad(paramsUb, srcGm[offset], copyInParams, copyInPadParams);
    this->msdInQueue.template EnQue(paramsUb);
}

// 𝑺=𝑺'×𝑄𝑚𝑎𝑥 + 𝑂𝑓𝑓𝑠𝑒𝑡𝐾×∑Q[𝑗]
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdSqueezeResMulOffsetKPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, 
                                                                                                   uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    LocalTensor<FT> aMax = this->msdMaxBmm1Ub[gmPingpong][startRow * this->MSD_BLOCK_ELEMENT_NUM];
    int64_t baseOffset = startRow * this->MSD_BLOCK_ELEMENT_NUM;

    if (this->msdIsKOffsetExist) {
        MsdCopyAntiquantParamsPerToken(this->keyAntiquantOffsetGmPerToken, params->antiqParamOffsetPerToken, columnCount, actualColumnCount);
        LocalTensor<FT> antiqOffsetPerTokenUb = this->msdInQueue.template DeQue<FT>();

        LocalTensor<FT> tmpOffset = this->msdTmpMm2Buff.template Get<FT>();
        LocalTensor<FT> aRowSum = this->msdRowSumUb[gmPingpong][baseOffset];

        // offset * rowsum(A)
        MsdAntiParamsMulByRow(tmpOffset, antiqOffsetPerTokenUb, aRowSum, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        this->msdInQueue.template FreeTensor(antiqOffsetPerTokenUb);

        // Amax * C + rowsum(A) * offset
        MsdFusedMulAddByRow(mmResUb, aMax, tmpOffset, dealRowCount, columnCount, actualColumnCount);
    } else {
        // Amax * C
        MsdRowMuls(mmResUb, mmResUb, aMax, dealRowCount, columnCount, actualColumnCount);
    }
    PipeBarrier<PIPE_V>();
}

// 𝑚𝑚1𝑅𝑒𝑠=𝑆𝑐𝑎𝑙𝑒𝐾×𝑺
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdSqueezeResMulScaleKPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, 
                                                                                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    MsdCopyAntiquantParamsPerToken(this->keyAntiquantScaleGmPerToken, params->antiqParamOffsetPerToken, columnCount, actualColumnCount);
    LocalTensor<FT> antiqScalePerTokenUb = this->msdInQueue.template DeQue<FT>();
    // (Amax * C + rowsum(A) * offset) * scale
    MsdVecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    this->msdInQueue.template FreeTensor(antiqScalePerTokenUb);
}

// 𝑚𝑚1𝑅𝑒𝑠 = 𝑆𝑐𝑎𝑙𝑒𝐾×(𝑺 + 𝑂𝑓𝑓𝑠𝑒𝑡𝐾×∑𝐴[𝑗])
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdSqueezePerToken(PFAComputeParam *params, LocalTensor<FT>& bmm1ResUb, uint32_t startRow, uint32_t dealRowCount, int gmPingpong) {
    uint32_t expandLineCnt = params->singleProcessSOuterSize;        
    uint32_t columnCount = params->singleProcessSInnerSizeNow;      
    uint32_t actualColumnCount = (params->isInnerTail ? params->singleProcessSInnerBmmTail : params->singleProcessSInnerSizeNow); //params->unalignSInner; //params->singleProcessSInnerBmmTail;

    // 1. 𝑆′=(𝑆1+𝑆2×1/254 +S3×1/254^2)×1/127
    MsdMatmulResCombine(bmm1ResUb, this->bmm1ResGmDb[params->gmPingpong], expandLineCnt, startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);

    // 2. 𝑺 = 𝑺'×𝑄𝑚𝑎𝑥 + 𝑂𝑓𝑓𝑠𝑒𝑡𝐾×∑Q[𝑗]
    Bmm1MsdSqueezeResMulOffsetKPerToken(params, bmm1ResUb, startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);

    // 3. 𝑚𝑚1𝑅𝑒𝑠 = 𝑆𝑐𝑎𝑙𝑒𝐾×𝑺
    Bmm1MsdSqueezeResMulScaleKPerToken(params, bmm1ResUb, dealRowCount, columnCount, actualColumnCount, gmPingpong);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdSqueeze(PFAComputeParam *params, LocalTensor<FT>& bmm1ResUb, uint32_t startRow, uint32_t dealRowCount, int ubPingpong, int gmPingpong) {
    if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 0) {
        Bmm1MsdSqueezePerchannel(params, bmm1ResUb, startRow, dealRowCount, gmPingpong);
    } else if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 1){
        Bmm1MsdSqueezePerToken(params, bmm1ResUb, startRow, dealRowCount, gmPingpong);
    }
    SetFlag<HardEvent::MTE2_V>(this->bmm1ResCopyInEvent[ubPingpong]);
}

// 𝐴′=𝐴×𝑆𝑐𝑎𝑙𝑒𝑉 
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdExpandMm1ResMulScaleVPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, 
                                                                                                    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    MsdCopyAntiquantParamsPerToken(this->valueAntiquantScaleGmPerToken, params->antiqParamOffsetPerToken, columnCount, actualColumnCount);
    LocalTensor<FT> antiqScalePerTokenUb = this->msdInQueue.template DeQue<FT>();
    
    MsdVecMulMat(mmResUb, antiqScalePerTokenUb, mmResUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    this->msdInQueue.template FreeTensor(antiqScalePerTokenUb);
}

// Expend 𝐴′ to vec1ResGm
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdExpandResPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, GlobalTensor<KV_T> bmm1ExpandGm,
                                                                                        uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    LocalTensor<FT> tmpAFloorUb = this->msdTmpMm2Buff.template Get<FT>();
    Adds(tmpAFloorUb, mmResUb, (FT)0, dealRowCount * columnCount); 
    PipeBarrier<PIPE_V>();

    MsdMatmulPreProcess(params, bmm1ExpandGm, this->msdMaxBmm2Ub[gmPingpong], mmResUb, tmpAFloorUb, 
                        startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);
}

// 𝑃=𝐴′×𝑂𝑓𝑓𝑠𝑒𝑡𝑉; ∑P[𝑗]
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdExpandResMulOffsetVPerToken(PFAComputeParam *params, LocalTensor<FT> &mmResUb,
                                                                                                  uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    if (!this->msdIsVOffsetExist) {
        return;
    }
    
    int64_t baseOffset = startRow * this->MSD_BLOCK_ELEMENT_NUM;
    LocalTensor<FT> tmpAFloorUb = this->msdTmpMm2Buff.template Get<FT>();
    Adds(tmpAFloorUb, mmResUb, (FT)0, dealRowCount * columnCount); 
    PipeBarrier<PIPE_V>();

    // get 𝑂𝑓𝑓𝑠𝑒𝑡𝑉
    MsdCopyAntiquantParamsPerToken(this->valueAntiquantOffsetGmPerToken, params->antiqParamOffsetPerToken, columnCount, actualColumnCount);

    // tmpAFloorUb = 𝑃 = 𝐴′×𝑂𝑓𝑓𝑠𝑒𝑡𝑉
    LocalTensor<FT> antiqScalePerTokenUb = this->msdInQueue.template DeQue<FT>();

    MsdVecMulMat(tmpAFloorUb, antiqScalePerTokenUb, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    this->msdInQueue.template FreeTensor(antiqScalePerTokenUb);
    PipeBarrier<PIPE_V>();

    // tmpASum = ∑P[𝑗]
    LocalTensor<FT> tmpASum = this->msdAMaxTmpBuff.template Get<FT>();
    MsdRowSum(tmpASum, tmpAFloorUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
    // brcb 
    Brcb(this->msdSoftmaxScaleResRowSumUb[gmPingpong][baseOffset], tmpASum, (dealRowCount + 7) / 8, {1, 8});
    PipeBarrier<PIPE_V>();
}

// 𝐴′=𝐴×𝑆𝑐𝑎𝑙𝑒𝑉; Expand 𝐴′; Calc 𝑃=𝐴′×𝑂𝑓𝑓𝑠𝑒𝑡𝑉
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdExpandPertoken(PFAComputeParam *params, LocalTensor<FT> &mmResUb, GlobalTensor<KV_T> bmm1ExpandGm, 
                                                                                     uint32_t startRow, uint32_t dealRowCount, int gmPingpong) {
    uint32_t columnCount = params->singleProcessSInnerSizeNow;       
    uint32_t actualColumnCount = (params->isInnerTail ? params->singleProcessSInnerBmmTail : params->singleProcessSInnerSizeNow); 
    // 1. Calc 𝐴′=𝐴×𝑆𝑐𝑎𝑙𝑒𝑉
    Bmm2MsdExpandMm1ResMulScaleVPerToken(params, mmResUb, dealRowCount, columnCount, actualColumnCount, gmPingpong);

    // 2. Calc 𝑃=𝐴′×𝑂𝑓𝑓𝑠𝑒𝑡𝑉; ∑P[𝑗]
    Bmm2MsdExpandResMulOffsetVPerToken(params, mmResUb, startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);

    // 3. Expand 𝐴′
    Bmm2MsdExpandResPerToken(params, mmResUb, bmm1ExpandGm, startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdExpand(PFAComputeParam *params, LocalTensor<FT> &mmResUb, GlobalTensor<KV_T> bmm1ExpandGm, 
                                                                             uint32_t startRow, uint32_t dealRowCount, int gmPingpong) {
    if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 0) {
        Bmm2MsdExpandPerchannel(params, mmResUb, bmm1ExpandGm, startRow, dealRowCount, gmPingpong);
    } else if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 1) {
        Bmm2MsdExpandPertoken(params, mmResUb, bmm1ExpandGm, startRow, dealRowCount, gmPingpong);
    }
}

// 𝑏𝑚𝑚2𝑅𝑒𝑠𝑈𝑏 = 𝑏𝑚𝑚2𝑅𝑒𝑠𝑈𝑏∗𝐴′𝑚𝑎𝑥 + 𝑃
template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdSqueezeResMulMax(LocalTensor<FT> &bmm2ResUb,
                                                                                       uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    int64_t baseOffset = startRow * this->MSD_BLOCK_ELEMENT_NUM;
    LocalTensor<FT> aRowMax = this->msdMaxBmm2Ub[gmPingpong][baseOffset];

    if (this->msdIsVOffsetExist) {
        LocalTensor<FT> aRowSum = this->msdSoftmaxScaleResRowSumUb[gmPingpong][baseOffset];
        MsdMulMaxAddSumByRow(bmm2ResUb, aRowMax, aRowSum, dealRowCount, columnCount, actualColumnCount);
    } else {
        MsdRowMuls(bmm2ResUb, bmm2ResUb, aRowMax, dealRowCount, columnCount, actualColumnCount);
    }
    PipeBarrier<PIPE_V>();
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdSqueezePertoken(PFAComputeParam *params, LocalTensor<FT>& bmm2ResUb, GlobalTensor<mmOutputType> bmm2ResGmDb, int gmPingpong) {
    uint32_t startRow = 0;
    uint32_t dealRowCount = params->actualSeqLengthPerBatch;
    uint32_t columnCount = this->tilingData->promptAttentionBaseParams.alignedHeadSize;
    uint32_t actualColumnCount = columnCount;
    uint32_t copySize =  dealRowCount * columnCount;

    // 1. combine 𝑏𝑚𝑚2𝑅𝑒𝑠𝑈𝑏   
    MsdMatmulResCombine(bmm2ResUb, bmm2ResGmDb, params->singleProcessSOuterSize, startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);

    // 2. Calc  𝑏𝑚𝑚2𝑅𝑒𝑠𝑈𝑏 = 𝑏𝑚𝑚2𝑅𝑒𝑠𝑈𝑏∗𝐴′𝑚𝑎𝑥 + 𝑃
    Bmm2MsdSqueezeResMulMax(bmm2ResUb, startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdSqueeze(PFAComputeParam *params, LocalTensor<computeType>& bmm2ResUb, GlobalTensor<mmOutputType> bmm2ResGmDb, int gmPingpong) {
    if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 0) {
        Bmm2MsdSqueezePerchannel(params, bmm2ResUb, bmm2ResGmDb, gmPingpong);
    } else if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 1) {
        Bmm2MsdSqueezePertoken(params, bmm2ResUb, bmm2ResGmDb, gmPingpong);
    }
}

template <typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::CopyAntiquantScale(LocalTensor<FT>& castUb,
                                                                                  GlobalTensor<T> srcGm,
                                                                                  int64_t offset) {
    int headDimAlign = this->tilingData->promptAttentionBaseParams.alignedHeadSize;
    int headDim = this->tilingData->promptAttentionBaseParams.headSize;
    uint32_t qTypeElementSize = BYTE_BLOCK_PFA / sizeof(T);
    DataCopyExtParams copyInParams;
    DataCopyPadExtParams<T> copyInPadParams;
    // antiq scale copy in
    copyInParams.blockCount = 1;
    copyInParams.blockLen = headDim * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (headDimAlign - headDim) / qTypeElementSize;

    copyInPadParams.isPad = true;
    copyInPadParams.leftPadding = 0;
    copyInPadParams.rightPadding = (headDimAlign - headDim) % qTypeElementSize;
    copyInPadParams.paddingValue = 0;

    LocalTensor<T> inputUb = this->msdInQueue.template AllocTensor<T>();  // using msdInQueue
    DataCopyPad(inputUb, srcGm[offset], copyInParams, copyInPadParams);
    this->msdInQueue.template EnQue(inputUb);

    inputUb = this->msdInQueue.template DeQue<T>();
    Cast(castUb, inputUb, RoundMode::CAST_NONE, headDim);
    this->msdInQueue.template FreeTensor(inputUb);
}

template <typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::VecAddMat(
    LocalTensor<FT> dstUb, LocalTensor<FT> src0Ub, LocalTensor<FT> src1Ub, uint32_t dealRowCount, uint32_t columnCount,
    uint32_t actualColumnCount) {
    constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK_PFA / sizeof(FT);
    constexpr uint32_t REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE_PFA / sizeof(FT);

    BinaryRepeatParams repeatParams;
    repeatParams.dstBlkStride = 1;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = columnCount / BLOCK_ELEMENT_NUM;
    repeatParams.src0RepStride = 0;
    repeatParams.src1RepStride = columnCount / BLOCK_ELEMENT_NUM;
    uint32_t mask = REPEAT_ELEMENT_NUM;
    uint32_t loopCount = actualColumnCount / mask;
    uint32_t remainCount = actualColumnCount % mask;

    uint32_t offset = 0;
    for (int i = 0; i < loopCount; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], mask, dealRowCount, repeatParams);
        offset += mask;
    }
    if (remainCount > 0) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], remainCount, dealRowCount, repeatParams);
    }
}

template <typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::MsdSoftmaxResPreProcess(
    PFAComputeParam *params, GlobalTensor<KV_T> dstGm, LocalTensor<FT> srcUb, LocalTensor<FT> tmpAFloorUb, uint32_t startRow,
    uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount, int gmPingpong) {
    int64_t actualSeqLength = params->actualSeqLengthPerBatch;
    int64_t step = actualSeqLength * columnCount;
    int64_t baseOffset = startRow * columnCount; 
    uint32_t calcSize = dealRowCount * columnCount;
    /**
     * Amax= 1
     * A1=round(127/Amax * A)
     * A2=round(254*(127/Amax * A - A1))
     * A3=round(254*254*(127/Amax * A - A1 - A2 / 254))
     */
    Muls(srcUb, srcUb, this->msdAntiqCoeff1, calcSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < this->msdIterNum; i++) {
        MsdAIterExpand(dstGm, srcUb, tmpAFloorUb, calcSize, (i == 0 ? true : false), step * i + baseOffset, gmPingpong);
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdExpandPerchannel(PFAComputeParam *params) {
    // copy scale
    int headSize = this->tilingData->promptAttentionBaseParams.alignedHeadSize;
    int64_t offset = params->batchNOffset / this->headNumRatio * headSize;
    LocalTensor<FT> scaleUbFloat = this->msdScaleBuff.template Get<FT>();
    CopyAntiquantScale(scaleUbFloat, this->keyAntiquantScaleGm, offset);
    PipeBarrier<PIPE_V>();

    uint32_t startRow = 0;
    int64_t actualSeqLength = params->actualSeqLengthPerBatch;
    int64_t qOffset = params->tensorAOffset;
    uint32_t dealRowCount = actualSeqLength;
    uint32_t actualColumnCount = this->tilingData->promptAttentionBaseParams.headSize;
    uint32_t columnCount = headSize;

    LocalTensor<FT> queryCastUb = this->msdTmpMm1Buff.template Get<FT>();
    Bmm1MsdCopyQueryGm2Ub(queryCastUb, qOffset, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    MsdVecMulMat(queryCastUb, scaleUbFloat, queryCastUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();

    if (this->tilingData->promptAttentionBaseParams.isSoftMaxLseEnable && this->msdIsKOffsetExist) {
        LocalTensor<FT> tmpRowSumUb = this->msdTmpMm2Buff.template Get<FT>();
        LocalTensor<FT> offsetUbFloat = this->msdOffsetBuff.template Get<FT>();
        CopyAntiquantScale(offsetUbFloat, this->keyAntiquantOffsetGm, offset);
        PipeBarrier<PIPE_V>();

        MsdVecMulMat(tmpRowSumUb, offsetUbFloat, queryCastUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
        Bmm1MsdCalcQueryRowSum(tmpRowSumUb, startRow, dealRowCount, columnCount, actualColumnCount, params->gmPingpong);
        PipeBarrier<PIPE_V>();
    }

    LocalTensor<FT> aMaxBmm1Ub = this->msdAMaxResBuff[params->gmPingpong].template Get<FT>();

    LocalTensor<FT> aFloorUb = this->msdTmpMm2Buff.template Get<FT>();

    /**
     * Amax = rowmax(|A|)
     * A1 = round(127/Amax * A)
     * A2 = round(254*(127/Amax * A - A1))
     * A3 = round(254*254*(127/Amax * A - A1 - A2 / 254))
     */
    MsdMatmulPreProcess(params, this->queryMsdExpandGm, aMaxBmm1Ub, queryCastUb, aFloorUb,
                        startRow, dealRowCount, columnCount, actualColumnCount, params->gmPingpong);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1MsdSqueezePerchannel(
        PFAComputeParam *params, LocalTensor<FT> &bmm1ResUb, uint32_t startRow, uint32_t dealRowCount, int gmPingpong) {
    uint32_t expandLineCnt = params->singleProcessSOuterSize;
    uint32_t columnCount = params->singleProcessSInnerSizeNow;
    uint32_t actualColumnCount = params->isInnerTail ? params->singleProcessSInnerBmmTail : params->singleProcessSInnerSizeNow;

    constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK_PFA / sizeof(FT);
    LocalTensor<FT> aMaxBmm1Ub = this->msdMaxBmm1Ub[gmPingpong][startRow * this->MSD_BLOCK_ELEMENT_NUM];

    MsdMatmulResCombine(bmm1ResUb, this->bmm1ResGmDb[params->gmPingpong], expandLineCnt, startRow, dealRowCount,
                        columnCount, actualColumnCount, gmPingpong);
    PipeBarrier<PIPE_V>();
    MsdRowMuls(bmm1ResUb, bmm1ResUb, aMaxBmm1Ub, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdExpandPerchannel(
        PFAComputeParam *params, LocalTensor<FT> &bmm1ResUb, GlobalTensor<KV_T> &bmm1ExpandGm,
        uint32_t startRow, uint32_t dealRowCount, int gmPingpong) {
    LocalTensor<FT> aFloorUb = this->msdTmpMm2Buff.template Get<FT>();

    uint32_t columnCount = params->singleProcessSInnerSizeNow;
    uint32_t actualColumnCount = params->isInnerTail ? params->singleProcessSInnerBmmTail : params->singleProcessSInnerSizeNow;

    MsdSoftmaxResPreProcess(params, bmm1ExpandGm, bmm1ResUb, aFloorUb,
                            startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2MsdSqueezePerchannel(
        PFAComputeParam *params, LocalTensor<FT> &bmm2ResUb, GlobalTensor<mmOutputType> &srcGM, int gmPingpong) {
    // S1 * D
    uint32_t startRow = 0;
    uint32_t columnCount = this->tilingData->promptAttentionBaseParams.alignedHeadSize;
    uint32_t actualColumnCount = columnCount;
    uint32_t dealRowCount = params->actualSeqLengthPerBatch;
    MsdMatmulResCombine(bmm2ResUb, srcGM, params->singleProcessSOuterSize, startRow, dealRowCount, columnCount, actualColumnCount, gmPingpong);
    PipeBarrier<PIPE_V>();
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm2UpdateDivPerchannel(LocalTensor<FT> &bmm2ResLastUb) {
    PFAComputeParam *params = this->preHeadParams;
    uint32_t dealRowCount = params->singleProcessSOuterSize;
    uint32_t columnCount = this->tilingData->promptAttentionBaseParams.headSize;
    uint32_t actualColumnCount = columnCount;
    int headSize = columnCount;
    int64_t offset = params->batchNOffset / this->headNumRatio * headSize;

    if (this->msdIsVOffsetExist) {
        // copy value offset
        // offset shape [N, D], [H], [N, 1, D]
        LocalTensor<FT> offsetUbFloat = this->msdOffsetBuff.template Get<FT>();
        CopyAntiquantScale(offsetUbFloat, this->valueAntiquantOffsetGm, offset);
        PipeBarrier<PIPE_V>();

        // update + offset
        VecAddMat(bmm2ResLastUb, offsetUbFloat, bmm2ResLastUb, dealRowCount, columnCount, actualColumnCount);
        PipeBarrier<PIPE_V>();
    }

    // copy value scale
    LocalTensor<FT> scaleUbFloat = this->msdScaleBuff.template Get<FT>();
    CopyAntiquantScale(scaleUbFloat, this->valueAntiquantScaleGm, offset);
    PipeBarrier<PIPE_V>();

    // scale * update
    MsdVecMulMat(bmm2ResLastUb, scaleUbFloat, bmm2ResLastUb, dealRowCount, columnCount, actualColumnCount);
    PipeBarrier<PIPE_V>();
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::Bmm1ResDoVecBmm2Compute() {
    PFAComputeParam *params = this->headParams;
    LocalTensor<computeType> bmm2ResUb;
    uint32_t resShapeSize;

    // Select parameters for block-sparse
    bool firstIter = params->isBlockSparse ? params->isFirstComputedIter : params->isFirstInnerIter;
    bool secondIter = params->isBlockSparse ? params->isSecondComputedIter : params->isSecondInnerIter;
    bool lastIter = params->isBlockSparse ? params->isLastComputedIter : params->isLastInnerIter;

    // Handling the current loop softmax，using headParams.
    this->Res1VecCompute(params);
    if (firstIter) {
        ProcessLastSouterLoopFinalRes();  // Process the output of the last task souter loop. All internal calls require the use of preHeadParams.
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            if constexpr (PFAT::msdMode == MsdMode::MSD_OFF) {
                this->Bmm2Antiquant(params);
            }
        }
        this->Bmm2ComputeIterate(params->taskBatch, params->batchNOffset, params->sInnerOffsetDataSize);
    } else if (secondIter) {    
        if (this->preHeadParams->fakeMsg) {
            this->bmm2.WaitIterateAll();
            this->Bmm2ComputeIterate(params->taskBatch, params->batchNOffset, params->sInnerOffsetDataSize);
        }else{                         
            bmm2ResUb = AllocBmm2UbRes(this->headParams, true, resShapeSize);  // The second time doesn't require addition, use Tbuf.
            this->bmm2.WaitIterateAll();
            if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
                Bmm2MsdSqueeze(params, bmm2ResUb, this->bmm2ResGmDb[this->headParams->gmPingpong ^ 1], this->headParams->gmPingpong ^ 1);
            } else {
                DataCopy(bmm2ResUb, this->bmm2ResGmDb[this->headParams->gmPingpong ^ 1], resShapeSize);  // Process the bmm2 result of the last cycle，so it's ^1.
                SetFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[this->headParams->gmPingpong ^ 1]);
                WaitFlag<HardEvent::MTE2_V>(this->bmm2ResCopyInEvent[this->headParams->gmPingpong ^ 1]);
            }

            if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
                if constexpr (PFAT::msdMode == MsdMode::MSD_OFF) {
                    this->Bmm2Antiquant(params);
                }
            }
            this->Bmm2ComputeIterate(params->taskBatch, params->batchNOffset, params->sInnerOffsetDataSize);    // Triggering the current loop bmm2 computing，using headParams.
            this->UpdateVmul(this->softmaxExpUb);
        } 
    } else {
        if (this->preHeadParams->fakeMsg) {
            this->bmm2.WaitIterateAll();
            this->Bmm2ComputeIterate(params->taskBatch, params->batchNOffset, params->sInnerOffsetDataSize);
        }else {
            bmm2ResUb = AllocBmm2UbRes(this->headParams, false, resShapeSize);
            this->bmm2.WaitIterateAll();
            if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
                Bmm2MsdSqueeze(params, bmm2ResUb, this->bmm2ResGmDb[this->headParams->gmPingpong ^ 1], this->headParams->gmPingpong ^ 1);
            } else {
                DataCopy(bmm2ResUb, this->bmm2ResGmDb[this->headParams->gmPingpong ^ 1], resShapeSize);  // Process the bmm2 result of the last cycle，so it's ^1.
                this->tempBmm2Queue.template EnQue<computeType>(bmm2ResUb);
                bmm2ResUb = this->tempBmm2Queue.template DeQue<computeType>();
            }

            this->Bmm2UpdateAdd(bmm2ResUb);
            this->tempBmm2Queue.FreeTensor(bmm2ResUb);
            PipeBarrier<PIPE_V>();
            if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
                if constexpr (PFAT::msdMode == MsdMode::MSD_OFF) {
                    this->Bmm2Antiquant(params);
                }
            }
            this->Bmm2ComputeIterate(params->taskBatch, params->batchNOffset, params->sInnerOffsetDataSize); // Triggering the current loop bmm2 computing，using headParams.
            this->UpdateVmul(this->softmaxExpUb);
        } 
    }

    if (lastIter) {
        // copy sle
        if (this->tilingData->promptAttentionBaseParams.isSoftMaxLseEnable) {
            this->SoftmaxLseCopyOut(this->softmaxSumUb, this->softmaxMaxUb);
        }
        // Reuse softmaxExp Ub to copy sum.
        LocalTensor<float> softmaxSumTmp = this->softmaxExpUb_.template Get<float>(this->softmaxSumSize);
        DataCopy(softmaxSumTmp, this->softmaxSumUb, this->softmaxSumSize);
        PipeBarrier<PIPE_V>();
        this->copyOutPrevIter = true;
        this->needAdd = !firstIter;    // When the first loop is the last loop, no add is required.
    }
}

template <typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::CheckRowInvalid(int64_t preTokens, int64_t nextTokens,
                                                                             PFAComputeParam* params) {
    // 1. nextToken cross souter   2. preToken cross souter   3. preToken cross souter 2
    //  |____            |          |____  \        |           |___\_   \      |
    //  | \  |           |          |   |   \       |           |    \|   \     |
    //  |_ \ |           |          | \ |    \      |           |     |\   \    |
    //  |   \            |          |  \|     \     |           |____ | \   \   |
    //  |\   \           |          |___|\     \    |           |     |  \   \  |
    //  | \              |          |               |           |     |         |

    bool nextokenCrossSouter = nextTokens < 0 && abs(nextTokens) > params->sOuterOffset &&
                             abs(nextTokens) < (params->sOuterOffset + params->singleProcessSOuterSize);
    int32_t sinnerSize = params->isInnerTail ? params->singleProcessSInnerBmmTail : params->singleProcessSInnerSize;
    bool pretokenCrossSouter = preTokens > 0 && preTokens > params->sOuterOffset &&
                             preTokens < (params->sOuterOffset + params->singleProcessSOuterSize) &&
                             (params->sOuterOffset + params->singleProcessSOuterSize - preTokens) > sinnerSize;
    bool pretokenCrossSouter2 =
         preTokens < 0 && preTokens + params->actualSeqLengthKVPerBatch > params->sOuterOffset &&
         preTokens + params->actualSeqLengthKVPerBatch < (params->sOuterOffset + params->singleProcessSOuterSize);
    
    bool firstIter = params->isBlockSparse ? params->isFirstComputedIter : params->isFirstInnerIter;
    if (firstIter && (nextokenCrossSouter || pretokenCrossSouter || pretokenCrossSouter2)) {
        params->kernelInvalidRow = 1;
    } else {
        params->kernelInvalidRow = 0;
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::SInnerLoopFunc(int64_t sInnerFirstToken, int64_t sInnerLastToken, int curBatch,
                                                                            int64_t preTokens, int64_t nextTokens, U16VecView rowSabi) {
    // params passing on references. When tailParams, params also update accordingly.
    PFAComputeParam *&params = this->tailParams;                // Configure new tasks, which will be placed at the end of the queue. Use tailParams.
    int32_t basicSInnerSize = (int32_t)(params->singleProcessSInnerSize);
    // Per-sabi-entry token stride (block_shape[1]): 128 for sabi-at-128 mode, 512 for sabi-at-512.
    // Direct member access (mirrors Bmm1/Bmm2 ComputeIterate).
    const int32_t kvBlockSize =
        (int32_t)this->tilingData->promptAttentionBaseParams.kvBlockSize;
    int32_t startIndex = sInnerFirstToken / basicSInnerSize;
    int32_t endIndex = (sInnerLastToken + basicSInnerSize - 1) / basicSInnerSize;
    bool isS2Load = (this->maxInnerLoopTimes == 1);
    if constexpr (PFAT::enablePrefix) {
        if (this->actualKVPrefixLen < sInnerLastToken) {
            endIndex = (this->actualKVPrefixLen + basicSInnerSize - 1) / basicSInnerSize + (sInnerLastToken - this->actualKVPrefixLen + basicSInnerSize - 1) / basicSInnerSize;
        }
    }
    if (endIndex > this->maxInnerLoopTimes) {
        endIndex = this->maxInnerLoopTimes;
    }

    constexpr int32_t softmaxInnerBasicSize = 64;
    // Upper Triangle Mask Scene，dynamic last sinnersize. According to the upper triangle mask，calculate the most compact sinnersize of the current last inner iter.（The minimum sinnersize containing all mask 0 values)
    int64_t firstInnerMargin = (sInnerFirstToken - startIndex * basicSInnerSize) / softmaxInnerBasicSize * softmaxInnerBasicSize;
    int64_t lastInnerMargin = (endIndex * basicSInnerSize - sInnerLastToken) / softmaxInnerBasicSize * softmaxInnerBasicSize;
    if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {
        firstInnerMargin = 0; // To ensure that the mm is not transferred across blocks in PA OptimizationMode, firstInnerMargin must be set to 0.
    }
    params->tensorAOffset = this->tensorACoreOffset;
    params->mm1SingleCoreN = params->singleProcessSInnerSize;
    params->isFirstInnerIter = true;
    params->isSecondInnerIter = true;
    params->taskBatch = curBatch;
    this->isSoftmaxLseNeedUpdate = false;

    params->isBlockSparse = rowSabi.IsValid();
    // TODO (mmarz): remove this if condition once sabi is passed properly
    if (params->isBlockSparse && endIndex > kPadU16) {
        // u16 encoding can't represent block indices >=kPadU16
        params->isBlockSparse = false;   // fall back to dense path
    }
    // Note: a "dense-row detection" scan that pre-checks whether rowSabi covers
    // [startIndex*sabiPerTile, endIndex*sabiPerTile) contiguously (and short-
    // circuits to the dense path if so) was evaluated and removed. With the
    // lookahead-fusion in the sparse loop the sparse path is efficient enough
    // that the scan cost exceeded its savings — the sparse path beats
    // "scan + dense" even at 0% sparsity. An O(1) last-element-only detection
    // (rowSabi.Get(len-1) != kPadU16 => route to dense) was also tested and
    // consistently regressed low-sparsity cases by 5–10% (likely an I-cache
    // layout shift in the hot sparse loop), so it was not kept.
    if (!params->isBlockSparse) {    // old loop
        for (int32_t sInnerLoopIdx = startIndex; sInnerLoopIdx < endIndex; sInnerLoopIdx++) {
            params->sInnerLoopOffset = sInnerLoopIdx;  // S2 Align offset
            params->isFirstInnerIter = (sInnerLoopIdx == startIndex);
            params->isSecondInnerIter = (sInnerLoopIdx == (startIndex + 1));
            params->isLastInnerIter = (sInnerLoopIdx == endIndex - 1);
            if constexpr (PFAT::enablePrefix) {
                params->isPrefixInnerIter = sInnerLoopIdx * basicSInnerSize < this->actualKVPrefixLen;
            } else {
                params->isPrefixInnerIter = 0;
            }
            if (unlikely(isS2Load)) {
                params->isInnerTail = true;
            } else {
                params->isInnerTail = (sInnerLoopIdx == (int32_t)this->maxInnerLoopTimes - 1) || (sInnerLoopIdx == (int32_t)this->maxInnerLoopPrefixTimes - 1);
            }

            if (unlikely(params->isInnerTail)) {
                if (!params->isPrefixInnerIter) {
                    lastInnerMargin = (sInnerLoopIdx * basicSInnerSize + params->unalignSInner - sInnerLastToken)
                        / softmaxInnerBasicSize * softmaxInnerBasicSize;
                    lastInnerMargin = (lastInnerMargin > 0) ? lastInnerMargin : 0;
                    if constexpr (PFAT::enablePrefix) {
                        lastInnerMargin = 0;
                    }
                    params->mm1SingleCoreN = params->singleProcessSInnerSizeTail - lastInnerMargin;
                    params->singleProcessSInnerSizeNow = params->singleProcessSInnerSizeTail - lastInnerMargin;
                    params->singleProcessSInnerBmmTail = params->unalignSInner - lastInnerMargin;
                    params->maskCopyInCol = params->maskInnerTailAlign - lastInnerMargin;
                    params->pseShiftCopyInCol = params->pseShiftInnerTailAlign - lastInnerMargin;
                } else {
                    if constexpr (PFAT::enablePrefix) {
                        params->mm1SingleCoreN = params->singleProcessSInnerPrefixSizeTail;
                        params->singleProcessSInnerSizeNow = params->singleProcessSInnerPrefixSizeTail;
                        params->singleProcessSInnerBmmTail = params->unalignSInnerPrefix;
                        params->maskCopyInCol = params->maskInnerPrefixTailAlign;
                        params->pseShiftCopyInCol = params->pseShiftInnerPrefixTailAlign;
                    }
                }
            } else {
                params->mm1SingleCoreN = params->singleProcessSInnerSize;
                params->singleProcessSInnerSizeNow = params->singleProcessSInnerSize;
                params->singleProcessSInnerBmmTail = params->singleProcessSInnerSize;
                params->maskCopyInCol = params->singleProcessSInnerSize;
                params->pseShiftCopyInCol = params->singleProcessSInnerSize;
                if (params->isLastInnerIter) {
                    if constexpr (PFAT::enablePrefix) {
                        lastInnerMargin = 0;
                    }
                    params->mm1SingleCoreN -= lastInnerMargin;
                    params->singleProcessSInnerSizeNow -= lastInnerMargin;
                    params->singleProcessSInnerBmmTail -= lastInnerMargin;
                    params->maskCopyInCol -= lastInnerMargin;
                    params->pseShiftCopyInCol -= lastInnerMargin;
                }
            }
            params->mm2SingleKAlign = (params->mm1SingleCoreN + MM2_SINGLE_K_ALIGN_SIZE - 1) / MM2_SINGLE_K_ALIGN_SIZE * MM2_SINGLE_K_ALIGN_SIZE;
            if (params->isFirstInnerIter) {
                if constexpr (PFAT::enablePrefix) {
                    firstInnerMargin = 0;
                }
                params->mm1SingleCoreN -= firstInnerMargin;
                params->singleProcessSInnerSizeNow -= firstInnerMargin;
                params->singleProcessSInnerBmmTail -= firstInnerMargin;
                params->maskCopyInCol -= firstInnerMargin;
                params->pseShiftCopyInCol -= firstInnerMargin;
                params->tensorBOffset = this->GetBmm1TensorBOffset(params, sInnerLoopIdx, firstInnerMargin);
                this->ComputeOffset(params, sInnerLoopIdx, firstInnerMargin);
            } else {
                params->tensorBOffset = this->GetBmm1TensorBOffset(params, sInnerLoopIdx, 0);
                this->ComputeOffset(params, sInnerLoopIdx, 0);
            }

            // T2.A: attentionMaskType ∈ {0, 1} only — leftUp/rightDown/band branches removed.
            // Determine whether the row invalidation OptimizationMode is enabled in the core.
            CheckRowInvalid(preTokens, nextTokens, params);
            params->sparseBandSelect0 = true;
            params->sparseBandSelect1 = true;

            if (this->queSize >= this->queSizeLimit) {
                // When the queue is full, task is triggered. The task specified by headParams starts to send instructions.
                ComputeEachCoreSInnerLoop();

                // prehead update
                this->preHeadParams = this->headParams;

                // head out of queue
                this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                this->headParams = &this->pfaParamsQueue[this->headId];

                // tail join the queue
                this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                if ((sInnerLoopIdx - startIndex) < PFA_PARAMS_QUEUE_CAPBABILITY - 1) {
                    // Overwrite the old head parameter. The next tail is not assigned a value outside the Inner loop and has no parameters. We need to copy the parameters that will be recorded outside the loop.
                    this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                }
                nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                this->tailParams = nextTailParams;
            }
            else {// tail join the queue
                this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                // Overwrite the old head parameter. The next tail is not assigned a value outside the Inner loop and has no parameters. We need to copy the parameters that will be recorded outside the loop.
                this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                this->tailParams = nextTailParams;
                this->queSize++;
            }
        }
    } else if (kvBlockSize > basicSInnerSize) {
        // ===========================================================================
        // Wide-KV path (BLOCK_SIZE_KV > basicSInnerSize, currently only KV=1024).
        // One sabi entry covers `kvBlockSize` tokens but the cube tile is only
        // `basicSInnerSize` (=512 for D=128 BF16). Each sabi entry is therefore
        // split into `subTilesPerSabi = kvBlockSize / basicSInnerSize` cube ops
        // sharing the same flash-softmax accumulator. Per cube op:
        //   packedCount = 1, packedSabi[0] = real sabi value,
        //   params->tensorBOffset / valueOffset are pre-baked with the per-
        //     sub-tile token offset  (sub * basicSInnerSize * kvTokenStride)
        //     so that Bmm1/Bmm2ComputeIterate's existing
        //       kvOffset = kvBase + packedSabi[0]*kvBlockSize*kvTokenStride
        //     formula reaches the correct K/V slice WITHOUT any extra term —
        //     keeping the hot cube path byte-identical to the legacy
        //     KV <= basicSInnerSize case (preserves I-cache layout).
        // The contiguous-K fast path in Bmm1/Bmm2ComputeIterate fires because
        // firstSabi == lastSabi (packedCount==1), so one wide matmul of width
        // mm1SingleCoreN = basicSInnerSize (or smaller for tail) is issued.
        // ===========================================================================
        const int32_t subTilesPerSabi = basicSInnerSize == 0 ? 1 : (int32_t)(kvBlockSize / basicSInnerSize);
        const int64_t kvTokenStride_packer =
            (PFAT::layout == PFALayout::BNSD)
            ? (int64_t)this->tilingData->promptAttentionBaseParams.headSize
            : (int64_t)this->MultiHeadKV;
        // startIndex is in basicSInnerSize-units; convert to sabi-index units.
        // The first sabi entry whose token range overlaps sInnerFirstToken is at
        // sabi index = (startIndex * basicSInnerSize) / kvBlockSize.
        const int32_t startSabiIdx = (int32_t)((int64_t)startIndex * basicSInnerSize / kvBlockSize);
        int32_t firstSabiIdx = FirstGreaterEqual(rowSabi, (uint16_t)startSabiIdx);
        if (firstSabiIdx < 0) return;

        const int64_t sInnerLastTokenAligned =
            ((int64_t)sInnerLastToken + (int64_t)softmaxInnerBasicSize - 1) /
            (int64_t)softmaxInnerBasicSize * (int64_t)softmaxInnerBasicSize;

        int32_t computedBlocks = 0;
        int32_t chunkIdx = firstSabiIdx;
        while (chunkIdx < (int32_t)rowSabi.len) {
            uint16_t s = rowSabi.Get(static_cast<uint32_t>(chunkIdx));
            if (s == kPadU16) break;
            const int64_t sabiTokenStart = (int64_t)s * kvBlockSize;
            if (sabiTokenStart >= sInnerLastTokenAligned) break;

            // Peek ahead to detect last sabi for this row, for isLastComputedIter.
            bool isLastSabi;
            if (chunkIdx + 1 >= (int32_t)rowSabi.len) {
                isLastSabi = true;
            } else {
                uint16_t nxt = rowSabi.Get(static_cast<uint32_t>(chunkIdx + 1));
                isLastSabi = (nxt == kPadU16) ||
                             ((int64_t)nxt * kvBlockSize >= sInnerLastTokenAligned);
            }

            for (int sub = 0; sub < subTilesPerSabi; ++sub) {
                const int64_t tileTokenStart = sabiTokenStart + (int64_t)sub * basicSInnerSize;
                if (tileTokenStart >= sInnerLastTokenAligned) break;

                const int64_t lastCols64 = sInnerLastTokenAligned - tileTokenStart;
                const int32_t lastCols =
                    (lastCols64 >= basicSInnerSize) ? basicSInnerSize : (int32_t)lastCols64;
                const int64_t lastColsActual64 = (int64_t)sInnerLastToken - tileTokenStart;
                const int32_t lastColsActual =
                    (lastColsActual64 >= basicSInnerSize) ? basicSInnerSize
                    : (lastColsActual64 < 0) ? 0
                    : (int32_t)lastColsActual64;

                params->packedSabi[0]            = s;
                params->packedCount              = 1;
                // Pre-bake the sub-tile offset into the K/V base so bmm1/bmm2
                // can use the legacy `kvBase + packedSabi[0]*kvBlockSize*stride`
                // formula unchanged — no per-cube-call extra add, no I-cache shift.
                const int64_t subOff             = (int64_t)sub * basicSInnerSize * kvTokenStride_packer;

                params->mm1SingleCoreN           = (uint32_t)lastCols;
                params->singleProcessSInnerSizeNow = (uint32_t)lastCols;
                params->singleProcessSInnerBmmTail = (uint32_t)lastCols;
                params->bmm2K                    = (uint32_t)lastColsActual;
                params->mm2SingleKAlign = (params->mm1SingleCoreN + MM2_SINGLE_K_ALIGN_SIZE - 1)
                                        / MM2_SINGLE_K_ALIGN_SIZE * MM2_SINGLE_K_ALIGN_SIZE;
                params->maskCopyInCol            = (uint32_t)lastColsActual;
                params->pseShiftCopyInCol        = (uint32_t)lastColsActual;

                const bool isLastSubTile =
                    (sub == subTilesPerSabi - 1) ||
                    (tileTokenStart + basicSInnerSize >= sInnerLastTokenAligned);
                const bool doneNow = isLastSabi && isLastSubTile;

                params->isFirstComputedIter   = (computedBlocks == 0);
                params->isSecondComputedIter  = (computedBlocks == 1);
                params->isLastComputedIter    = doneNow;
                params->isFirstInnerIter      = params->isFirstComputedIter;
                params->isSecondInnerIter     = params->isSecondComputedIter;
                params->isLastInnerIter       = params->isLastComputedIter;
                params->isPrefixInnerIter     = 0;
                params->isInnerTail           = (lastCols < basicSInnerSize);

                params->sInnerLoopOffset = (uint32_t)(tileTokenStart / basicSInnerSize);

                params->tensorBOffset            = this->tensorBCoreOffset + subOff;
                params->valueOffset              = this->valueCoreOffset   + subOff;
                params->tensorAOffset            = this->tensorACoreOffset;
                params->sInnerOffsetDataSize     = tileTokenStart;
                params->antiqParamOffsetPerToken = this->antiqParamBatchOffsetPerToken + tileTokenStart;
                params->attenMaskOffset          = 0;
                params->attenMaskOffsetPre       = 0;
                params->pseShiftOffset           = 0;

                CheckRowInvalid(preTokens, nextTokens, params);
                params->sparseBandSelect0 = true;
                params->sparseBandSelect1 = true;

                if (this->queSize >= this->queSizeLimit) {
                    ComputeEachCoreSInnerLoop();
                    this->preHeadParams = this->headParams;
                    this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                    this->headParams = &this->pfaParamsQueue[this->headId];
                    this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                    PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                    if (computedBlocks < PFA_PARAMS_QUEUE_CAPBABILITY - 1) {
                        this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                    }
                    nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                    this->tailParams = nextTailParams;
                } else {
                    this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                    PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                    this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                    nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                    this->tailParams = nextTailParams;
                    this->queSize++;
                }
                ++computedBlocks;
                if (doneNow) { chunkIdx = (int32_t)rowSabi.len; break; }
            }
            ++chunkIdx;
        }
    } else {
        // ===========================================================================
        // K-contiguous-only packing: each packed batch contains sabi entries
        // forming a consecutive run [s, s+1, ..., s+packedCount-1]. The slow
        // gather path in Bmm1/Bmm2ComputeIterate is therefore unreachable in
        // this packer — every packed batch is K-contiguous by construction
        // and takes the fast path (one wide matmul). The slow gather code is
        // kept compiled because a future packer relaxation may reach it.
        // ===========================================================================
        // kPackSize is the max number of sabi entries the cube can absorb in one call.
        // With basicSInnerSize-wide cube tiles and kvBlockSize-token sabi entries,
        // kPackSize = basicSInnerSize / kvBlockSize (4 for 128×128, 1 for 128×512).
        const int32_t kPackSize = basicSInnerSize / kvBlockSize;
        const int32_t sabiPerTile = basicSInnerSize / kvBlockSize;  // 4 for kvBlockSize=128, 1 for kvBlockSize=512
        int32_t firstSabiIdx = FirstGreaterEqual(rowSabi, (uint16_t)(startIndex * sabiPerTile));
        if (firstSabiIdx < 0) return;

        // Effective last K column read by the dense path = sInnerLastToken aligned UP
        // to softmaxInnerBasicSize (64). Matching this preserves arithmetic equivalence
        // with the original dense + sub-block-mask path under causal/preTokens clipping.
        const int64_t sInnerLastTokenAligned =
            ((int64_t)sInnerLastToken + (int64_t)softmaxInnerBasicSize - 1) /
            (int64_t)softmaxInnerBasicSize * (int64_t)softmaxInnerBasicSize;

        const int64_t kvTokenStride =
            (PFAT::layout == PFALayout::BNSD)
            ? (int64_t)this->tilingData->promptAttentionBaseParams.headSize
            : (int64_t)this->MultiHeadKV;

        int32_t computedBlocks = 0;
        int32_t chunkIdx = firstSabiIdx;

        while (chunkIdx < (int32_t)rowSabi.len) {
            uint16_t batch[4] = {0, 0, 0, 0};
            int32_t batchCount = 0;
            bool exhausted = false;
            while (batchCount < kPackSize && chunkIdx < (int32_t)rowSabi.len) {
                uint16_t s = rowSabi.Get(static_cast<uint32_t>(chunkIdx));
                if (s == kPadU16) { exhausted = true; break; }
                if ((int64_t)s * kvBlockSize >= sInnerLastTokenAligned) { exhausted = true; break; }
                // D.1 K-contiguous-only guard: pack iff next sabi == last + 1.
                // On break we do NOT advance chunkIdx, so the next outer iter
                // starts a fresh batch with this entry as its first element.
                if (batchCount > 0 &&
                    (uint32_t)s != (uint32_t)batch[batchCount - 1] + 1u) {
                    break;
                }
                batch[batchCount++] = s;
                ++chunkIdx;
            }
            if (batchCount == 0) break;

            bool done;
            if (exhausted || chunkIdx >= (int32_t)rowSabi.len) {
                done = true;
            } else {
                uint16_t nxt = rowSabi.Get(static_cast<uint32_t>(chunkIdx));
                done = (nxt == kPadU16) ||
                       ((int64_t)nxt * kvBlockSize >= sInnerLastTokenAligned);
            }

            const uint16_t lastSabi = batch[batchCount - 1];
            const int64_t lastSabiTokenStart = (int64_t)lastSabi * kvBlockSize;
            const int64_t lastCols64 = sInnerLastTokenAligned - lastSabiTokenStart;
            const int32_t lastCols = (lastCols64 >= kvBlockSize) ? kvBlockSize : (int32_t)lastCols64;
            const uint32_t totalCols = (uint32_t)((batchCount - 1) * kvBlockSize + lastCols);

            // Keep softmax's fast-path firing on the global last batch
            // (BmmTail = 64-aligned `totalCols`) by pre-masking the cube-OOB
            // workspace cols [bmm2K, mm1SingleCoreN) to -10000 in
            // Res1VecCompute. bmm2's V load narrows to
            // bmm2K = totalColsActual (unaligned), so V reads stay strictly
            // inside actualSeqLengthKVPerBatch — no allocator-slack dependency.
            //
            // Why bmm2K = totalColsActual (not ceil(actual/8)*8): PyTorch's NPU
            // caching-allocator slack at large S can contain NaN bit-patterns.
            // Even a 2-token V OOB read poisons every row of the last head via
            // the atomic-add in bmm2. Keeping the V load tight and paying for
            // an unaligned pre-mask region (handled with a 2-stage Duplicate:
            // bitmask form for the partial leading BLOCK + count form for the
            // full aligned BLOCKs) is the only way to keep softmax fast AND
            // not depend on allocator behaviour.
            const int64_t lastColsActual64 =
                (int64_t)sInnerLastToken - lastSabiTokenStart;
            const int32_t lastColsActual =
                (lastColsActual64 >= kvBlockSize) ? kvBlockSize
                : (lastColsActual64 < 0) ? 0
                : (int32_t)lastColsActual64;
            const uint32_t totalColsActual =
                (uint32_t)((batchCount - 1) * kvBlockSize + lastColsActual);

            for (int i = 0; i < batchCount; ++i) {
                params->packedSabi[i] = batch[i];
            }
            params->packedCount = static_cast<uint8_t>(batchCount);

            // mm1SingleCoreN = singleProcessSInnerSizeNow = totalCols
            // (64-aligned) — the bmm1→UB DataCopy stride math (blockLen,
            // srcStride divided by softmaxTypeByteNum=8) uses these. totalCols
            // is divisible by 8 because lastCols ∈ {64, 128}.
            // singleProcessSInnerBmmTail = totalCols keeps softmax fast-path firing
            // (gate is BmmTail % 64 == 0).  bmm2K = totalColsActual (unaligned)
            // bounds bmm2's V load to actualSeqLengthKVPerBatch.
            params->mm1SingleCoreN              = totalCols;
            params->singleProcessSInnerSizeNow  = totalCols;
            params->singleProcessSInnerBmmTail  = totalCols;
            params->bmm2K                       = totalColsActual;
            params->mm2SingleKAlign = (params->mm1SingleCoreN + MM2_SINGLE_K_ALIGN_SIZE - 1)
                                    / MM2_SINGLE_K_ALIGN_SIZE * MM2_SINGLE_K_ALIGN_SIZE;
            params->maskCopyInCol      = totalColsActual;
            params->pseShiftCopyInCol  = totalColsActual;

            params->isFirstComputedIter  = (computedBlocks == 0);
            params->isSecondComputedIter = (computedBlocks == 1);
            params->isLastComputedIter   = done;

            // Legacy *InnerIter flags now mirror Computed-iter flags. Downstream code
            // selects between them via params->isBlockSparse, but keeping them
            // consistent prevents stale values leaking through any unreviewed path.
            params->isFirstInnerIter   = params->isFirstComputedIter;
            params->isSecondInnerIter  = params->isSecondComputedIter;
            params->isLastInnerIter    = params->isLastComputedIter;
            params->isPrefixInnerIter  = 0;       // prefix path not implemented in this packer
            params->isInnerTail        = (lastCols < kvBlockSize);

            params->sInnerLoopOffset = (uint32_t)(lastSabi / sabiPerTile);

            // KV offsets — sparse path stores per-batch-head BASE only. Bmm1/Bmm2
            // ComputeIterate add the per-sabi offset (packedSabi[i] * kvBlockSize *
            // stride) at compute time. Reading this->tensorBCoreOffset directly
            // inside the cube would race against multi-head LoopSOuterOffsetInit
            // updates (the queue can outlive a batch transition).
            const int64_t firstSabiTokenOff = (int64_t)batch[0] * kvBlockSize;
            params->tensorBOffset            = this->tensorBCoreOffset;
            params->valueOffset              = this->valueCoreOffset;
            params->tensorAOffset            = this->tensorACoreOffset;
            params->sInnerOffsetDataSize     = firstSabiTokenOff;
            params->antiqParamOffsetPerToken = this->antiqParamBatchOffsetPerToken + firstSabiTokenOff;
            params->attenMaskOffset          = 0;
            params->attenMaskOffsetPre       = 0;
            params->pseShiftOffset           = 0;

            CheckRowInvalid(preTokens, nextTokens, params);
            params->sparseBandSelect0 = true;
            params->sparseBandSelect1 = true;

            if (this->queSize >= this->queSizeLimit) {
                ComputeEachCoreSInnerLoop();
                this->preHeadParams = this->headParams;
                this->headId = (this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                this->headParams = &this->pfaParamsQueue[this->headId];
                this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                if (computedBlocks < PFA_PARAMS_QUEUE_CAPBABILITY - 1) {
                    this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                }
                nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                this->tailParams = nextTailParams;
            } else {
                this->tailId = (this->tailId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY;
                PFAComputeParam *nextTailParams = &this->pfaParamsQueue[this->tailId];
                this->CopyParamsAttrOutOfInnerLoop(nextTailParams, this->tailParams);
                nextTailParams->gmPingpong = this->tailParams->gmPingpong ^ 1;
                this->tailParams = nextTailParams;
                this->queSize++;
            }
            ++computedBlocks;

            if (done) break;
        }
    }
}

template<typename PFAT>
__aicore__ inline int64_t BlitzSparseAttentionS1s2Bns1X910<PFAT>::ClipSInnerToken(int64_t sInnerToken, int64_t minValue, int64_t maxValue) {
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::ComputeEachCoreSInnerLoop() {
    PFAComputeParam *params = this->headParams;
    PFAComputeParam *nextParams = &(this->pfaParamsQueue[(this->headId + 1) % PFA_PARAMS_QUEUE_CAPBABILITY]);

    // mm1 compute
    if (this->isGlobalFirstCompute) {
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
                this->Bmm1MsdExpand(params);
            } else {
                this->Bmm1Antiquant(params);
            }
        }
        this->Bmm1ComputeIterate(params);
    }

    this->mm.WaitIterateAll();

    Bmm1VecInputCopyIn();

    if (this->queSize > 0) {
        // Pre fetch the next mm1 calculation.
        if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value) {
            if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
                this->Bmm1MsdExpand(nextParams);
            } else {
                this->Bmm1Antiquant(nextParams);
            }
        }
        this->Bmm1ComputeIterate(nextParams);
    }

    Bmm1ResDoVecBmm2Compute();
    this->isGlobalFirstCompute = false;
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::ComputeEachCore(uint32_t coreIdx) {
    int64_t blockNum = GetBlockNum() * GetTaskRation();
    InitEachCoreWorkspace(coreIdx, blockNum);
    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (g_coreType == AIV && coreIdx >= actualCoreNums) {
        return;
    }
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;
    uint32_t sOuterCoreIdx = coreIdx;
    uint32_t sOuterSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    if constexpr (PFAT::MM_TYPE == MatMulType::MM_IBSHARE_NORM) {
        // If enable cube split, the coreIdx should be cube index and sOuterSize should be the sum of two vecters.
        sOuterCoreIdx = coreIdx / 2;  // C : V = 1 : 2, so need to divide 2 to get cube index.
        sOuterSize = sOuterSize * 2;  // 2 : Multiply by 2 to get the cube sOuterSize.
    }
    // Temporary reuse
    // CoreHeadNumTail to coreNStart
    // actualS1 to coreNEnd
    // actualCoreNums to coreSidStart
    // singleCoreHeadNumSize to coreSidEnd
    int sIdStart = this->tilingData->promptAttentionSeqParams.actualCoreNums[sOuterCoreIdx];
    int sIdEnd = this->tilingData->promptAttentionSeqParams.singleCoreHeadNumSize[sOuterCoreIdx];
    int outerLoopStart = this->tilingData->promptAttentionSeqParams.coreSeqPosStart[sOuterCoreIdx];
    int outerLoopEnd = this->tilingData->promptAttentionSeqParams.coreSeqPosEnd[sOuterCoreIdx];
    int nLoopStart = this->tilingData->promptAttentionSeqParams.CoreHeadNumTail[sOuterCoreIdx];
    int nLoopEnd = this->tilingData->promptAttentionSeqParams.actualS1[sOuterCoreIdx];
    int64_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int64_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    int tmpOuterLoopEnd;
    int tmpSLoopEnd;
    bool isLast = false;
    int64_t actualSeqLengthsIdx = 0;
    // You must pass the reference assignment params because the head address is updated internally.
    PFAComputeParam *&params = this->tailParams;
    if constexpr (PFAT::enablePrefix) {
        this->actualKVPrefixLen = this->tilingData->promptAttentionBaseParams.isActualSharedPrefixLenNull ?
            this->tilingData->promptAttentionBaseParams.prefixSeqInnerSize : this->actualSharedPrefixLenGm.GetValue(0);
    } else {
        this->actualKVPrefixLen = 0;
    }
    for (uint32_t loopNIdx = nLoopStart; loopNIdx < nLoopEnd; loopNIdx++) {
        params->batchNOffset = loopNIdx;
        this->CalPrefixCoreOffset(params);
        if (loopNIdx != nLoopEnd - 1) {
            tmpSLoopEnd = sNum;
        } else {
            tmpSLoopEnd = sIdEnd;
            isLast = true;
        }
        for (int sIdx = sIdStart; sIdx < tmpSLoopEnd; sIdx++) {
            if (this->isKvContinuous == 0) {
                ListTensorDesc keyListTensorDesc((__gm__ void*)this->key_ptr);

                uint64_t dimInfo[4];
                this->kvTensorDesc.SetShapeAddr(&dimInfo[0]);
                keyListTensorDesc.GetDesc(this->kvTensorDesc, sIdx);
                if (PFAT::layout == PFALayout::BNSD) {
                    this->s2InCurrentBatch = this->kvTensorDesc.GetShape(2);
                } else {
                    this->s2InCurrentBatch = this->kvTensorDesc.GetShape(1);
                }
            }
            this->GetSingleCoreParam(sIdx);
            this->GetSparseParam(&preTokens, &nextTokens, sIdx, params);

            int sOuterBlockNum = (params->actualSeqLengthPerBatch + sOuterSize - 1) / sOuterSize;

            if (this->tilingData->promptAttentionBaseParams.isLayoutSH) {    // SH format offset
                params->multiSeqOffset = 0;
                for (int i = 0; i < sIdx; i++) {
                    params->multiSeqOffset += this->actualSeqLengthsGm.GetValue(i);
                }
                params->multiSeqOffset *= this->MultiHeadQ;
            } else {    // no SH format offset
                params->multiSeqOffset = this->CalMultiSeqOffset(sIdx);
            }

            if (isLast && sIdx == tmpSLoopEnd - 1) {
                tmpOuterLoopEnd = outerLoopEnd;
            } else {
                tmpOuterLoopEnd = sOuterBlockNum;
            }
            for (uint32_t sOuterLoopIdx = outerLoopStart; sOuterLoopIdx < tmpOuterLoopEnd; sOuterLoopIdx++) {
                int64_t sInnerFirstToken;
                int64_t sInnerLastToken;
                if constexpr (PFAT::MM_TYPE == MatMulType::MM_IBSHARE_NORM) {  // Processing from the perspective of cube to enable L1 reuse.
                    int64_t sOuterOffsetByCube = (int64_t)sOuterLoopIdx * (int64_t)sOuterSize;
                    if (unlikely(sOuterLoopIdx == sOuterBlockNum - 1)) {
                        uint32_t singleProcessSOuterSizeTailByCube = (params->actualSeqLengthPerBatch % sOuterSize != 0) ?
                                                                    params->actualSeqLengthPerBatch % sOuterSize : sOuterSize;
                        uint32_t singleProcessSOuterSizeTailV0 = (singleProcessSOuterSizeTailByCube + 1) / 2;  // Tail is divided into two vectors.
                        uint32_t singleProcessSOuterSizeTailV1 = singleProcessSOuterSizeTailByCube - singleProcessSOuterSizeTailV0;
                        params->singleProcessSOuterSize = (coreIdx % 2 == 0) ? singleProcessSOuterSizeTailV0 : singleProcessSOuterSizeTailV1;  // 2 : Get whether the vector is odd or even.
                        params->sOuterOffset = sOuterOffsetByCube + ((coreIdx % 2 == 0) ? 0 : (int64_t)singleProcessSOuterSizeTailV0);  // 2 : Get whether the vector is odd or even.
                    } else {
                        params->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
                        params->sOuterOffset = sOuterOffsetByCube + ((coreIdx % 2 == 0) ? 0 : (int64_t)this->singleProcessSOuterSizeWhole);  // 2 : Get whether the vector is odd or even.
                    }
                    params->fakeMsg = (params->singleProcessSOuterSize == 0);
                    sInnerFirstToken = ClipSInnerToken(sOuterOffsetByCube - preTokens, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
                    sInnerLastToken = ClipSInnerToken(sOuterOffsetByCube + nextTokens + (int64_t)sOuterSize, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
                } else {
                    if (sOuterLoopIdx == sOuterBlockNum - 1) {
                        params->singleProcessSOuterSize = this->singleProcessSOuterSizeTail;
                    } else {
                        params->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
                    }
                    params->sOuterOffset = sOuterLoopIdx * this->singleProcessSOuterSizeWhole;
                    if (nextTokens < 0 && params->sOuterOffset < ((nextTokens * (-1)) /
                        this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
                            continue;
                    }
                    sInnerFirstToken = ClipSInnerToken(params->sOuterOffset - (int64_t)preTokens, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
                    sInnerLastToken = ClipSInnerToken(params->sOuterOffset + (int64_t)nextTokens + params->singleProcessSOuterSize, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
                }
                if (sInnerLastToken <= sInnerFirstToken) {
                    continue;
                }

                this->LoopSOuterOffsetInit(params->multiSeqOffset, sIdx);
                SInnerLoopFunc(sInnerFirstToken, sInnerLastToken, sIdx, preTokens, nextTokens);
            }
            outerLoopStart = 0;
        }
        sIdStart = 0;
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::ComputeEachCoreBalance(uint32_t coreIdx) {
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;
    if (sNum == 0) {
	    return;
    }
    int64_t blockNum = GetBlockNum() * GetTaskRation();
    if (coreIdx % 2 == 1) {
        coreIdx = blockNum - coreIdx;
    }

    InitEachCoreWorkspace(coreIdx, blockNum);

    int64_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int64_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    int32_t sIdx = 0;
    // All seq within batch have the same length.
    int64_t actualSeqLengthsIdx = this->isActualLenDimsNull ? this->tilingData->promptAttentionBaseParams.seqSize : this->actualSeqLengthsGm.GetValue(sIdx);

    PFAComputeParam *&params = this->tailParams;
    if constexpr (PFAT::enablePrefix) {
        this->actualKVPrefixLen = this->tilingData->promptAttentionBaseParams.isActualSharedPrefixLenNull ?
            this->tilingData->promptAttentionBaseParams.prefixSeqInnerSize : this->actualSharedPrefixLenGm.GetValue(0);
    } else {
        this->actualKVPrefixLen = 0;
    }
    // T2.A: attentionMaskType ∈ {0, 1} only — band (==4) branch removed.
    actualSeqLengthsIdx = (this->attentionMaskType == 0 && (int64_t)actualSeqLengthsIdx >
                        (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize + this->actualKVPrefixLen +
                        (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                        (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize + this->actualKVPrefixLen +
                        (int64_t)this->tilingData->promptAttentionBaseParams.preTokens :
                        actualSeqLengthsIdx;

    int64_t sOuterBlockNum = (actualSeqLengthsIdx +
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize - 1) /
                              this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize;
    int64_t sNumMulHeadNum = this->tilingData->promptAttentionBaseParams.headNumSize * sNum;    // num_heads * batch_size
    int64_t totalTilingN = sNumMulHeadNum * sOuterBlockNum; //total number of Qblocks * number of heads; Qblocks = L/128

    int64_t sInnerFirstToken;
    int64_t sInnerLastToken;

    // For SABI blocks
    U16VecView sabiRow{};  // per-iteration, reset later
    const uint32_t sabiBatchSize  = this->tilingData->promptAttentionBaseParams.sabiBatchSize;
    const uint32_t sabiheadNum  = this->tilingData->promptAttentionBaseParams.sabiHeadNum;
    const uint32_t sabiQBlocks  = this->tilingData->promptAttentionBaseParams.sabiQblocks;
    const uint32_t sabiKVBlocks  = this->tilingData->promptAttentionBaseParams.sabiKVBlocks;
    const uint32_t sabiLen = this->tilingData->promptAttentionBaseParams.sabiLen;
    // physRow runs in 128-token Q-stride units (sOuterSize=128). The sabi tensor
    // has `ceil(S / qBlockSize)` rows, so multiple consecutive physRow values
    // share the same sabi row when qBlockSize > 128. sabiRow = physRow * 128 / qBlockSize.
    // Direct member access (mirrors kvBlockSize read in SInnerLoopFunc).
    const uint32_t qBlockSize = this->tilingData->promptAttentionBaseParams.qBlockSize;

    // Validate SABI metadata once
    // Superfluous len check, it is already computed in the same way
    uint64_t want64 = 0;
    bool sabiMetaOk = false;
    if (sabiLen != 0 && sabiBatchSize != 0 && sabiheadNum != 0 && sabiQBlocks != 0 && sabiKVBlocks != 0) {
        want64 = uint64_t(sabiBatchSize) * uint64_t(sabiheadNum) * uint64_t(sabiQBlocks) * uint64_t(sabiKVBlocks);
        // strict match for debug payload
        sabiMetaOk = (want64 == uint64_t(sabiLen));
    }

    const bool isSabi = sabiMetaOk;
    // Read from gm
    U16T4View t4{};
    U16T3View t3{};
    if (isSabi) {
        const __gm__ uint16_t* base = this->sabiBaseGm.GetPhyAddr();
        t4 = U16T4View{ base, sabiBatchSize, sabiheadNum, sabiQBlocks, sabiKVBlocks, sabiLen };
        // In this method is should always be bs == 1
        t3 = t4.AtBatch(0);
    }

    for (int64_t tilingIdx = coreIdx; tilingIdx < totalTilingN; tilingIdx += (blockNum - (tilingIdx % blockNum)) * 2 - 1) {
        int64_t sIdxMulbatchNOffset = tilingIdx % sNumMulHeadNum;
        sIdx = sIdxMulbatchNOffset % sNum;  // batch id
        params->batchNOffset = sIdxMulbatchNOffset / sNum;  // head
        this->CalPrefixCoreOffset(params);
        int64_t sOuterLoopIdx = sOuterBlockNum - 1 - (tilingIdx / sNumMulHeadNum);
        this->GetSingleCoreParam(sIdx);
        this->GetSparseParam(&preTokens, &nextTokens, sIdx, params);

        sabiRow = {};
        if (this->tilingData->promptAttentionBaseParams.isLayoutSH) {    // SH format offset
            params->multiSeqOffset = 0;
            for (int i = 0; i < sIdx; i++) {
                params->multiSeqOffset += this->actualSeqLengthsGm.GetValue(i);
            }
            params->multiSeqOffset *= this->MultiHeadQ;
        } else {    // no SH format offset
            params->multiSeqOffset = this->CalMultiSeqOffset(sIdx);
        }

        int64_t physRow;
        if (isSabi) {   // iterate top down instead of bottom up, so that tail is in the end as expected
            physRow = sOuterBlockNum - 1 - sOuterLoopIdx;   // convert reversed traversal -> forward row id

            bool isTailRow = (physRow == sOuterBlockNum - 1);
            params->singleProcessSOuterSize = isTailRow ? this->singleProcessSOuterSizeTail
                                                        : this->singleProcessSOuterSizeWhole;
            params->sOuterOffset = physRow * this->singleProcessSOuterSizeWhole;
        } else {
            if (sOuterLoopIdx == 0) {
                params->singleProcessSOuterSize = this->singleProcessSOuterSizeTail;
                params->sOuterOffset = 0;
            } else {
                params->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
                params->sOuterOffset = this->singleProcessSOuterSizeTail + (sOuterLoopIdx-1) * this->singleProcessSOuterSizeWhole;
            }
        }
        if (nextTokens < 0 && params->sOuterOffset < ((nextTokens * (-1)) /
            this->singleProcessSOuterSizeWhole * this->singleProcessSOuterSizeWhole)) {
                continue;
        }
        // Default behavior
        sInnerFirstToken = ClipSInnerToken(params->sOuterOffset - preTokens, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
        sInnerLastToken = ClipSInnerToken(params->sOuterOffset + nextTokens + params->singleProcessSOuterSize, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
        
        // Block sparsity
        if (isSabi) {
            const uint32_t H = static_cast<uint32_t>(this->tilingData->promptAttentionBaseParams.headNumSize);
            const uint32_t bh = static_cast<uint32_t>(sIdx * H + params->batchNOffset);

            // physRow is in 128-token Q-stride units; sabi rows are at qBlockSize
            // stride. Map physRow → sabiRow by physRow*128/qBlockSize. Equals physRow
            // when qBlockSize=128 (no change to legacy behavior).
            const uint32_t sabiRowIdx = (qBlockSize == 128U)
                ? static_cast<uint32_t>(physRow)
                : static_cast<uint32_t>(((int64_t)physRow * 128) / (int64_t)qBlockSize);
            sabiRow = t3.At(bh, sabiRowIdx);
        }
        if (sInnerLastToken <= sInnerFirstToken) {
            continue;
        }

        this->LoopSOuterOffsetInit(params->multiSeqOffset, sIdx);
        this->SInnerLoopFunc(sInnerFirstToken, sInnerLastToken, sIdx, preTokens, nextTokens, sabiRow);
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::InitEachCoreWorkspace(uint32_t coreIdx, int32_t blockNum) {
    this->spmTmpSize = this->tilingData->promptAttentionTensorSizeRect.spmTmpSize;
    this->mmResUbSize = this->tilingData->promptAttentionTensorSizeRect.mmResUbSize;
    this->bmm2ResUbSize = this->tilingData->promptAttentionTensorSizeRect.bmm2ResUbSize;

    constexpr int reuseWorkspaceRatio = 2;
    int64_t msdExpandsize = 1;
    if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
        msdExpandsize = this->msdIterNum;
    }

    int64_t mm1ResSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * \
        this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize;
    mm1ResSize = mm1ResSize * msdExpandsize;
    int64_t mm2ResSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * \
        this->tilingData->promptAttentionBaseParams.headSize  * msdExpandsize;
    this->bmm1ResGmDb[0].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize  * reuseWorkspaceRatio].GetPhyAddr());
    this->bmm1ResGmDb[1].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[blockNum * this->spmTmpSize +
        coreIdx * mm1ResSize * reuseWorkspaceRatio + mm1ResSize ].GetPhyAddr());
    if constexpr (IsSameType<T, int8_t>::value) {
        this->quant1ResGmDb[0].SetGlobalBuffer((__gm__ int8_t*)this->workspaceGm[blockNum * this->spmTmpSize +
            coreIdx * mm1ResSize * reuseWorkspaceRatio].GetPhyAddr());
        this->quant1ResGmDb[1].SetGlobalBuffer((__gm__ int8_t*)this->workspaceGm[blockNum * this->spmTmpSize +
            coreIdx * mm1ResSize * reuseWorkspaceRatio + mm1ResSize].GetPhyAddr());
    }

    if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
        this->bmm1ExpandGm[0].SetGlobalBuffer((__gm__ KV_T*)this->workspaceGm[blockNum * this->spmTmpSize +
            coreIdx * mm1ResSize * reuseWorkspaceRatio].GetPhyAddr());
        this->bmm1ExpandGm[1].SetGlobalBuffer((__gm__ KV_T*)this->workspaceGm[blockNum * this->spmTmpSize +
            coreIdx * mm1ResSize * reuseWorkspaceRatio + mm1ResSize].GetPhyAddr());
    }

    int64_t buff_offset = blockNum * (this->spmTmpSize + mm1ResSize * reuseWorkspaceRatio);
    this->bmm2ResGmDb[0].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[buff_offset +
        coreIdx * mm2ResSize * reuseWorkspaceRatio].GetPhyAddr());
    this->bmm2ResGmDb[1].SetGlobalBuffer((__gm__ mmOutputType*)this->workspaceGm[buff_offset +
        coreIdx * mm2ResSize * reuseWorkspaceRatio + mm2ResSize].GetPhyAddr());

    buff_offset += blockNum * mm2ResSize * reuseWorkspaceRatio;

    if constexpr (!IsSameType<T, KV_T>::value && IsSameType<KV_T, int8_t>::value && PFAT::msdMode != MsdMode::MSD_ON) {  // The offset is too large. workspace can be optimized. For details, see the IFA.
        GlobalTensor<T> workspaceGmAntiquant;
        // High precision OptimizationMode, workspace is fp32，but antiquant result is fp16.
        workspaceGmAntiquant.SetGlobalBuffer((__gm__ T*)this->workspaceGm[buff_offset].GetPhyAddr());
        int64_t kvAntiquantSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSInnerSize * \
            this->tilingData->promptAttentionBaseParams.headSize;
        this->keyGmAntiquant.SetGlobalBuffer((__gm__ T*)workspaceGmAntiquant[
            coreIdx * kvAntiquantSize * reuseWorkspaceRatio].GetPhyAddr());
        this->valueGmAntiquant.SetGlobalBuffer((__gm__ T*)workspaceGmAntiquant[
            coreIdx * kvAntiquantSize * reuseWorkspaceRatio + kvAntiquantSize].GetPhyAddr());
        buff_offset += blockNum * kvAntiquantSize * reuseWorkspaceRatio;
    }

    // After placing the four structures of the first kernel, place the four structures of the second kernel，When IFA, After placing the first structure of all kernel, place the second one.
    if constexpr (PFAT::MM_TYPE == MatMulType::MM_PA) {  // If compute type is different，the offset size here is different.
        GlobalTensor<uint32_t> workspaceGmPA;  //  storage PA callback structure data
        workspaceGmPA.SetGlobalBuffer((__gm__ uint32_t*)this->workspaceGm[buff_offset].GetPhyAddr());
        int32_t paStructSize = 64 / sizeof(uint32_t);  //  dcci cacheline 64B alignment  16 * 4B = 64B
        int32_t NumOfBmm = 2;
        int64_t baseCBDataOffset = coreIdx * paStructSize * NumOfBmm * reuseWorkspaceRatio;
        this->bmm1CBDataGm[0].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset].GetPhyAddr());
        this->bmm1CBDataPtr[0] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset].GetPhyAddr();
        this->bmm1CBDataGm[1].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize].GetPhyAddr());
        this->bmm1CBDataPtr[1] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize].GetPhyAddr();

        this->bmm2CBDataGm[0].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize * reuseWorkspaceRatio].GetPhyAddr());
        this->bmm2CBDataPtr[0] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize * reuseWorkspaceRatio].GetPhyAddr();
        this->bmm2CBDataGm[1].SetGlobalBuffer((__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize * reuseWorkspaceRatio + paStructSize].GetPhyAddr());
        this->bmm2CBDataPtr[1] = (__gm__ uint32_t*)workspaceGmPA[baseCBDataOffset + paStructSize * reuseWorkspaceRatio + paStructSize].GetPhyAddr();
        buff_offset += blockNum * baseCBDataOffset + paStructSize * reuseWorkspaceRatio + paStructSize * reuseWorkspaceRatio;
    }
    
    if constexpr ( PFAT::msdMode == MsdMode::MSD_ON) {  
        GlobalTensor<KV_T> workspaceGmMsd; 
        workspaceGmMsd.SetGlobalBuffer((__gm__ KV_T*)this->workspaceGm[buff_offset].GetPhyAddr());
        int32_t MsdtructSize = msdExpandsize * this->tilingData->promptAttentionBaseParams.seqSize * this->tilingData->promptAttentionBaseParams.headSize;  //  dcci cacheline 64B 对齐  16 * 4B = 64B
        int64_t baseCBDataOffset = coreIdx * MsdtructSize;
        this->queryMsdExpandGm.SetGlobalBuffer((__gm__ KV_T*)workspaceGmMsd[baseCBDataOffset].GetPhyAddr());
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::ComputeEachCoreSplitSeqOneN(uint32_t coreIdx) {
    int32_t blockNum = GetBlockNum() * GetTaskRation();
    InitEachCoreWorkspace(coreIdx, blockNum);

    int actualCoreNums = this->tilingData->promptAttentionSingleCoreParams.actualCoreNums;
    if (g_coreType == AIV && coreIdx >= actualCoreNums) {
        return;
    }
    int sNum = this->tilingData->promptAttentionBaseParams.dimNumOfseq;
    int headNum = this->tilingData->promptAttentionBaseParams.headNumSize;
    int64_t preTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.preTokens);
    int64_t nextTokens = (int32_t)(this->tilingData->promptAttentionBaseParams.nextTokens);

    // This solution does not support scenarios with actualSeq.
    uint32_t actualSeqLengths = this->tilingData->promptAttentionBaseParams.seqSize;

    // You must pass the reference assignment params，because the head address is updated internally.
    PFAComputeParam *&params = this->tailParams;

    int64_t bnIdx = 0;
    for (int sIdx = 0; sIdx < sNum; sIdx++) {
        this->GetSingleCoreParam(sIdx);
        this->GetSparseParam(&preTokens, &nextTokens, sIdx, params);
        actualSeqLengths = (this->attentionMaskType == 0 && (int64_t)actualSeqLengths >
                           (int64_t)this->tilingData->promptAttentionBaseParams.seqInnerSize +
                           (int64_t)this->tilingData->promptAttentionBaseParams.preTokens) ?
                           (this->tilingData->promptAttentionBaseParams.seqInnerSize +
                           this->tilingData->promptAttentionBaseParams.preTokens) :
                           actualSeqLengths;

        uint32_t cubeSOuterSize = this->tilingData->promptAttentionSingleCoreParams.singleProcessSOuterSize * 2;
        int sOuterBlockNum = (params->actualSeqLengthPerBatch + cubeSOuterSize - 1) / cubeSOuterSize;
        if (this->tilingData->promptAttentionBaseParams.isLayoutSH) {    // SH format offset
            params->multiSeqOffset = 0;
            for (int i = 0; i < sIdx; i++) {
                params->multiSeqOffset += this->actualSeqLengthsGm.GetValue(i);
            }
            params->multiSeqOffset *= this->MultiHeadQ;
        } else {    // no SH format offset
            params->multiSeqOffset = this->CalMultiSeqOffset(sIdx);
        }
        for (uint32_t loopNIdx = 0; loopNIdx < headNum; loopNIdx++) {
            params->batchNOffset = loopNIdx;
            // In order to make the amount of computation as uniform as possible on each kernel.
            uint32_t coreIdxCube = coreIdx / 2;
            uint32_t sOutPolicyIdx = (coreIdxCube + bnIdx) % (actualCoreNums / 2);      // actualCoreNums is the number of vector kernel. When cube split kernel, need to compute the number of cube kernel.
            int outerLoopStart = this->tilingData->promptAttentionSeqParams.coreSeqPosStart[sOutPolicyIdx];
            int outerLoopEnd = this->tilingData->promptAttentionSeqParams.coreSeqPosEnd[sOutPolicyIdx];
            for (uint32_t sOuterLoopIdxByCube = outerLoopStart; sOuterLoopIdxByCube < outerLoopEnd; sOuterLoopIdxByCube++) {
                uint32_t singleProcessSOuterSizeByCube = this->singleProcessSOuterSizeWhole * 2;  // 2 : cube sOuterSize
                int64_t sOuterOffsetByCube = (int64_t)sOuterLoopIdxByCube * (int64_t)singleProcessSOuterSizeByCube;
                if (unlikely(sOuterLoopIdxByCube == sOuterBlockNum - 1)) {
                    uint32_t singleProcessSOuterSizeTailByCube = (params->actualSeqLengthPerBatch % singleProcessSOuterSizeByCube != 0) ?
                                                                  params->actualSeqLengthPerBatch % singleProcessSOuterSizeByCube : singleProcessSOuterSizeByCube;
                    uint32_t singleProcessSOuterSizeTailV0 = (singleProcessSOuterSizeTailByCube + 1) / 2;
                    uint32_t singleProcessSOuterSizeTailV1 = singleProcessSOuterSizeTailByCube - singleProcessSOuterSizeTailV0;
                    params->singleProcessSOuterSize = (coreIdx % 2 == 0) ? singleProcessSOuterSizeTailV0 : singleProcessSOuterSizeTailV1;
                    params->sOuterOffset = sOuterOffsetByCube + ((coreIdx % 2 == 0) ? 0 : (int64_t)singleProcessSOuterSizeTailV0);
                } else {
                    params->singleProcessSOuterSize = this->singleProcessSOuterSizeWhole;
                    params->sOuterOffset = sOuterOffsetByCube + ((coreIdx % 2 == 0) ? 0 : (int64_t)this->singleProcessSOuterSizeWhole);
                }
                params->fakeMsg = (params->singleProcessSOuterSize == 0);
                int64_t sInnerFirstToken = ClipSInnerToken(sOuterOffsetByCube - preTokens, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
                int64_t sInnerLastToken = ClipSInnerToken(sOuterOffsetByCube + nextTokens + (int64_t)singleProcessSOuterSizeByCube, 0, params->actualSeqLengthKVPerBatch + this->actualKVPrefixLen);
                if (sInnerLastToken <= sInnerFirstToken) {
                    continue;
                }
                this->LoopSOuterOffsetInit(params->multiSeqOffset, sIdx);
                SInnerLoopFunc(sInnerFirstToken, sInnerLastToken, sIdx, preTokens, nextTokens);
            }
            bnIdx++;
        }
    }
}

template<typename PFAT>
__aicore__ inline void BlitzSparseAttentionS1s2Bns1X910<PFAT>::ProcessLastSouterLoopFinalRes() {
    if (this->copyOutPrevIter) {
        if (this->preHeadParams->fakeMsg) {
            this->bmm2.WaitIterateAll();
            this->copyOutPrevIter = false;
            return;
        }
        LocalTensor<float> softmaxSumTmp = this->softmaxExpUb_.template Get<float>(this->softmaxSumSize);
        LocalTensor<computeType> bmm2ResPreUb = this->tempBmm2Ub.template Get<computeType>(this->bmm2ResUbSize);
        LocalTensor<computeType>& FinalResUb = bmm2ResPreUb;
        uint32_t resShapeSize;

        LocalTensor<computeType> bmm2ResUb = AllocBmm2UbRes(this->preHeadParams, !this->needAdd, resShapeSize);    // When not adding, use Tbuf directly.
        this->bmm2.WaitIterateAll();
        if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
            Bmm2MsdSqueeze(this->preHeadParams, bmm2ResUb, this->bmm2ResGmDb[this->preHeadParams->gmPingpong], this->preHeadParams->gmPingpong);
        } else {
            DataCopy(bmm2ResUb, this->bmm2ResGmDb[this->preHeadParams->gmPingpong], resShapeSize);
        }

        if (this->needAdd) {
            this->tempBmm2Queue.template EnQue<computeType>(bmm2ResUb);
            bmm2ResUb = this->tempBmm2Queue.template DeQue<computeType>();
            this->Bmm2UpdateAdd(bmm2ResUb);
            this->tempBmm2Queue.FreeTensor(bmm2ResUb);
            PipeBarrier<PIPE_V>();
        } else {
            event_t tmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));  // If set wait is used continuously，can use Fetch. Otherwise, use Alloc.
            SetFlag<HardEvent::MTE2_V>(tmp);
            WaitFlag<HardEvent::MTE2_V>(tmp);
        }

        this->Bmm2UpdateDivNoTail(bmm2ResPreUb, softmaxSumTmp);
        if constexpr (PFAT::msdMode == MsdMode::MSD_ON) {
            if (this->tilingData->promptAttentionBaseParams.keyAntiquantMode == 0) {
                this->Bmm2UpdateDivPerchannel(bmm2ResPreUb);
            }
        }
        if ((PFAT::layout == PFALayout::BSH) ||
            (PFAT::layout == PFALayout::BNSD && this->tilingData->promptAttentionBaseParams.isBSNDOut == 1)) {
            this->DataCopyTransposeOutBSH(FinalResUb);
        } else {
            this->DataCopyTransposeOutBNSD(FinalResUb);
        }
        this->copyOutPrevIter = false;
    }
}

#endif  // BLITZ_SPARSE_ATTENTION_S1S2_BNS1_X910_H
