/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_compress_attention_infer.h
 * \brief
 */

#ifndef NSA_COMPRESS_ATTENTION_INFER_H
#define NSA_COMPRESS_ATTENTION_INFER_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "ncai_public_define.h"
#include "common_func.h"
#include "hardware.h"
#include "common.h"
#include "iterator.h"

using namespace AscendC;
using namespace matmul;

namespace NSA_COMPRESS_ATTENTION_INFER {

#ifdef __DAV_C220_CUBE__

template <typename NCAIType>
class NsaCompressAttentionInferAic {
public:
    using Q_T = typename NCAIType::queryType;
    using KV_T = typename NCAIType::kvType;
    using OUT_T = typename NCAIType::outputType;
    static constexpr LAYOUT LAYOUT_T = NCAIType::layout;

    __aicore__ inline NsaCompressAttentionInferAic(){};

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
        GM_ADDR blockTable, GM_ADDR actualQSeqLen, GM_ADDR actualKvSeqLen, GM_ADDR output,
        GM_ADDR topkIndicesOut, GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData);

    __aicore__ inline void Process();

protected:

    /* tiling data */
    NsaCompressAttentionInferBaseParams baseTilingData;
    NsaCompressAttentionInferSplitBNParams splitBNTilingData;

    /* variable */
    AscendC::GlobalTensor<Q_T> queryGm;
    AscendC::GlobalTensor<KV_T> keyGm;
    AscendC::GlobalTensor<KV_T> valueGm;
    AscendC::GlobalTensor<OUT_T> outGm;

    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int64_t> actualQSeqLenGm;
    GlobalTensor<int64_t> actualKvSeqLenGm;
    GlobalTensor<float> mm1ResGm;
    GlobalTensor<float> scoreInGm;
    GlobalTensor<Q_T> mm2InGm;
    AscendC::GlobalTensor<float> impScoreResultGm_;
    AscendC::GlobalTensor<float> impScoreParamGm_;
    AscendC::LocalTensor<float> impScoreL1aTensor;
    AscendC::LocalTensor<float> impScoreL0aTensor;
    AscendC::LocalTensor<float> impScoreL1bTensor;
    AscendC::LocalTensor<float> impScoreL0bTensor;
    AscendC::LocalTensor<float> impScoreL0cTensor;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    LocalTensor<Q_T> l1qBufAddrTensor;
    LocalTensor<KV_T> l1kBufAddrTensor;
    LocalTensor<KV_T> l1vBufAddrTensor;
    LocalTensor<Q_T> l1pBufAddrTensor;
    LocalTensor<Q_T> l0aBufTensor;
    LocalTensor<KV_T> l0bBufTensor;
    LocalTensor<float> l0cBufTensor;

    __gm__ uint8_t *blocktablePtr = nullptr;

    uint32_t l1qBufAddrOffset;
    uint32_t l1kBufAddrOffset;
    uint32_t l1vBufAddrOffset;
    uint32_t l1pBufAddrOffset;

    uint32_t vecBlockIdx;
    uint32_t cubeBlockIdx;
    uint32_t bn2LoopTimes;
    uint32_t beforeBlockSplitBn2Nums;

    uint32_t bIdx;

    // 基础参数
    uint32_t batchSize;
    uint32_t qSeqSize;
    uint32_t seqSize;
    uint32_t qHeadNum;
    uint32_t kvHeadNum;
    uint32_t headSizeQk;
    uint32_t headSizeVo;
    uint32_t blockSize;
    uint32_t groupSize;

    // 新增
    uint32_t maxBlockNumPerBatch;
    uint32_t workSpaceElemNum;
    uint32_t mm1ResWorkSpaceSize;
    uint32_t mm2InWorkSpaceSize;
    uint32_t scoreInWorkSpaceSize;
    uint32_t topKInWorkSpaceSize;
    uint32_t curSeqlen_;
    uint32_t curSeqlenRound_;
    uint32_t impScoreWorkSpaceSize_;
    uint32_t impScoreResultEleNum_;
    uint32_t impScoreResultOffset_;
    uint32_t impScoreResultRelativeOffset_;
    uint32_t impScoreResultCol_;
    uint32_t impScoreResultColPad_;
    uint16_t impScoreParamValidCol_;
    uint16_t impScoreParamTotalCol_;
    bool impScoreParamHasPrefix_;

    uint32_t processPerBatch;
    uint32_t kvHeadSplitSize;
    uint32_t kvHeadSplitNum;
    uint32_t qSeqLenSplitSize;
    uint32_t qSeqLenOffset;
    uint32_t qSeqLenCurProcess;
    uint32_t qSeqLenCumSum;
    uint32_t kvHeadOffset;
    uint32_t curKvHeadNum;
    uint32_t processNum;
    uint32_t aicNum;

    uint32_t coreNumUsed;
    uint32_t bnPerHeadCore;
    uint32_t bnPerTailCore;

    __aicore__ inline void InitTilingData(const NsaCompressAttentionInferTilingData *__restrict tilingData);
    __aicore__ inline void CopyQToL1(const uint32_t qRowNum, const uint32_t qRowNumRound,
                                     const uint32_t headSizeQkRound, const uint32_t actualKvHeadIdx,
                                     const uint32_t l1qPingPongFlag);
    __aicore__ inline void CopyKToL1(const uint32_t kvSeqTile, const uint32_t kvSeqTileRound,
                                     const uint32_t strideK, const uint32_t kCoreOffset,
                                     const uint32_t l1kPingPongFlag);
    __aicore__ inline void CopyPVToL1(const uint32_t pRowNum, const uint32_t pRowNumRound,
                                      const uint32_t kvSeqTile, const uint32_t kvSeqTileRound, const uint32_t strideV,
                                      const uint64_t pCoreOffset, const uint32_t vCoreOffset,
                                      const uint32_t l0abPingPongFlag, const uint32_t l1pvPingPongFlag);
    __aicore__ inline void LoadQKToL0(const uint32_t qRowNum, const uint32_t qRowNumRound,
                                      const uint32_t headSizeQkRound, const uint32_t kvSeqTileRound,
                                      const uint32_t l1qPingPongFlag, const uint32_t l1kPingPongFlag,
                                      const uint32_t sIdx, const uint32_t sLoop);
    __aicore__ inline void LoadPVToL0(const uint32_t pRowNum, const uint32_t pRowNumRound,
                                      const uint32_t headSizeVoRound, const uint32_t kvSeqTileRound,
                                      const uint32_t l1pvPingPongFlag, const uint32_t l0abPingPongFlag);
    __aicore__ inline void PerformQKMmad(const uint32_t qRowNum, const uint32_t kvSeqTile,
                                         const uint32_t l0cPingPongFlag);
    __aicore__ inline void PerformPVMmad(const uint32_t pRowNum, const uint32_t kvSeqTile,
                                         const uint32_t l0abPingPongFlag, const uint32_t l0cPingPongFlag,
                                         const uint32_t sIdx);
    __aicore__ inline void CopySToWorkSpace(const uint32_t qRowNum, const uint32_t qRowNumRound,
                                            const uint32_t kvSeqTileRound, const uint32_t curKvHeadIdx,
                                            const uint32_t sIdx, const uint32_t l0cPingPongFlag,
                                            const uint32_t mm1ResPingPongFlag);
    __aicore__ inline void CopyOToGm(const uint32_t pRowNum, const uint32_t pRowNumRound,
                                     const uint32_t oCoreOffset, const uint32_t l0cPingPongFlag);
    __aicore__ inline void ProcessMm1(const uint32_t processIdx);
    __aicore__ inline void ProcessMm2(const uint32_t processIdx);
    __aicore__ inline void PreProcess(const uint32_t processIdx);
    __aicore__ inline uint8_t CeilCubeBlock(uint32_t len, uint32_t block_size)
    {
        return (len + block_size - 1) / block_size;
    }
    __aicore__ inline void ProcessImportanceScore(const uint32_t processIdx, const uint32_t kvHeadOffset,
                                                  const uint32_t curKvHeadNum);
    __aicore__ inline void SplitImportanceScoreL0A(const uint16_t totalRow, const uint16_t l1LoopIdx,
                                                   const uint16_t l1LoopCol);
    __aicore__ inline void ProcessImportanceScoreMmad(uint16_t totalRow, bool l0PingPongFlag);
    __aicore__ inline void CopyImportanceScoreOut(bool useAtomicAdd, uint16_t totalRow, bool l0PingPongFlag);
    __aicore__ inline void LoadImportanceScoreParam();
};

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR blockTable, GM_ADDR actualQSeqLen, GM_ADDR actualKvSeqLen, GM_ADDR output, GM_ADDR topkIndicesOut,
    GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    this->aicNum = GetBlockNum();
    this->splitBNTilingData = tilingData->splitBNParams;

    InitTilingData(tilingData);
    this->vecBlockIdx = GetBlockIdx();
    this->cubeBlockIdx = GetBlockIdx();

    // gm数据
    this->queryGm.SetGlobalBuffer((__gm__ Q_T *)query);
    this->keyGm.SetGlobalBuffer((__gm__ KV_T *)key);
    this->valueGm.SetGlobalBuffer((__gm__ KV_T *)value);
    this->blockTableGm.SetGlobalBuffer((__gm__ int32_t *)blockTable);
    this->actualQSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualQSeqLen);
    this->actualKvSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualKvSeqLen);
    this->outGm.SetGlobalBuffer((__gm__ OUT_T *)output);
    this->mm1ResGm.SetGlobalBuffer((__gm__ float *)workspace);
    uint64_t offset = mm1ResWorkSpaceSize;
    this->scoreInGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += scoreInWorkSpaceSize;
    this->mm2InGm.SetGlobalBuffer((__gm__ Q_T *)(workspace + offset));
    offset += mm2InWorkSpaceSize + topKInWorkSpaceSize;
    impScoreResultGm_.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += impScoreWorkSpaceSize_;
    impScoreParamGm_.SetGlobalBuffer((__gm__ float *)(workspace + offset));

    l1qBufAddrOffset = 0;
    l1kBufAddrOffset = BASE_L1_BUF_ADDR_OFFSET;
    l1vBufAddrOffset = l1kBufAddrOffset + BASE_L1_BUF_ADDR_OFFSET;
    l1pBufAddrOffset = l1vBufAddrOffset + BASE_L1_BUF_ADDR_OFFSET;

    l1qBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1qBufAddrOffset);
    l1kBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, KV_T>(l1kBufAddrOffset);
    l1vBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, KV_T>(l1vBufAddrOffset);
    l1pBufAddrTensor = buf.GetBuffer<BufferType::ASCEND_CB, Q_T>(l1pBufAddrOffset);

    l0aBufTensor = buf.GetBuffer<BufferType::ASCEND_L0A, Q_T>(0);
    l0bBufTensor = buf.GetBuffer<BufferType::ASCEND_L0B, KV_T>(0);
    l0cBufTensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);

    impScoreL1aTensor = buf.GetBuffer<BufferType::ASCEND_CB, float>(0);
    impScoreL0aTensor = buf.GetBuffer<BufferType::ASCEND_L0A, float>(0);
    impScoreL1bTensor = buf.GetBuffer<BufferType::ASCEND_CB, float>(0);
    impScoreL0bTensor = buf.GetBuffer<BufferType::ASCEND_L0B, float>(0);
    impScoreL0cTensor = buf.GetBuffer<BufferType::ASCEND_L0C, float>(0);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::InitTilingData(
    const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    batchSize = tilingData->baseParams.batchSize;
    qSeqSize = tilingData->baseParams.qSeqSize;
    seqSize = tilingData->baseParams.seqSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    headSizeQk = tilingData->baseParams.headSizeQk;
    headSizeVo = tilingData->baseParams.headSizeVo;
    blockSize = tilingData->baseParams.blockSize;
    groupSize = tilingData->baseParams.groupSize;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    workSpaceElemNum = tilingData->baseParams.workSpaceElemNum;
    mm1ResWorkSpaceSize = tilingData->baseParams.mm1ResWorkSpaceSize;
    mm2InWorkSpaceSize = tilingData->baseParams.mm2InWorkSpaceSize;
    scoreInWorkSpaceSize = tilingData->baseParams.scoreInWorkSpaceSize;

    topKInWorkSpaceSize = tilingData->baseParams.topKInWorkSpaceSize;
    impScoreWorkSpaceSize_ = tilingData->baseParams.impScoreWorkSpaceSize;
    impScoreResultEleNum_ = tilingData->baseParams.impScoreResultEleNum;
    impScoreResultCol_ = tilingData->baseParams.impScoreResultCol;
    impScoreResultColPad_ = tilingData->baseParams.impScoreResultColPad;
    impScoreParamValidCol_ = tilingData->baseParams.impScoreParamValidCol;
    impScoreParamHasPrefix_ = tilingData->baseParams.impScoreParamHasPrefix;
    impScoreParamTotalCol_ = impScoreParamValidCol_ + impScoreParamHasPrefix_;

    coreNumUsed = tilingData->splitBNParams.coreNumUsed;
    processNum = tilingData->splitBNParams.processNum;
    processPerBatch = tilingData->splitBNParams.processPerBatch;
    kvHeadSplitSize = tilingData->splitBNParams.kvHeadSplitSize;
    kvHeadSplitNum = tilingData->splitBNParams.kvHeadSplitNum;
    qSeqLenSplitSize = tilingData->splitBNParams.qSeqLenSplitSize;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyQToL1(const uint32_t qRowNum,
                                                                         const uint32_t qRowNumRound,
                                                                         const uint32_t headSizeQkRound,
                                                                         const uint32_t actualKvHeadIdx,
                                                                         const uint32_t l1qPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1qPingPongFlag + 6);
    for (uint32_t qSeqLenCurProcessIdx = 0; qSeqLenCurProcessIdx < qSeqLenCurProcess; qSeqLenCurProcessIdx++) {
        uint64_t qCoreOffset = (qSeqLenCumSum + qSeqLenOffset + qSeqLenCurProcessIdx) * qHeadNum * headSizeQk
                             + actualKvHeadIdx * groupSize * headSizeQk;
        if (qRowNum != 1) {
            AscendC::DataCopy(
                l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + qSeqLenCurProcessIdx * groupSize * 16],
                            queryGm[qCoreOffset],
                            AscendC::Nd2NzParams(1,           // ndNum
                                                groupSize, // nValue
                                                headSizeQk, // dValue
                                                0,           // srcNdMatrixStride, unused
                                                headSizeQk,        // srcDValue
                                                qRowNumRound,   // dstNzC0Stride
                                                1,           // dstNzNStride
                                                0));         // dstNzMatrixStride, unused
        } else {
            AscendC::DataCopy(l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                            queryGm[qCoreOffset],
                            AscendC::DataCopyParams(1,
                                                    CeilDiv(headSizeQkRound, BLOCK_SIZE), // nValue
                                                    0,           // dstNzNStride
                                                    0));         // dstNzMatrixStride, unused
        }
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(0);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyKToL1(const uint32_t kvSeqTile,
                                                                         const uint32_t kvSeqTileRound,
                                                                         const uint32_t strideK,
                                                                         const uint32_t kCoreOffset,
                                                                         const uint32_t l1kPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1kPingPongFlag);
    AscendC::DataCopy(l1kBufAddrTensor[l1kPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                      keyGm[kCoreOffset],
                      AscendC::Nd2NzParams(1,           // ndNum
                                           kvSeqTile, // nValue
                                           headSizeQk, // dValue
                                           0,           // srcNdMatrixStride, unused
                                           strideK,        // srcDValue
                                           kvSeqTileRound,   // dstNzC0Stride
                                           1,           // dstNzNStride
                                           0));         // dstNzMatrixStride, unused
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(1);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyPVToL1(const uint32_t pRowNum,
                                                                          const uint32_t pRowNumRound,
                                                                          const uint32_t kvSeqTile,
                                                                          const uint32_t kvSeqTileRound,
                                                                          const uint32_t strideV,
                                                                          const uint64_t pCoreOffset,
                                                                          const uint32_t vCoreOffset,
                                                                          const uint32_t l0abPingPongFlag,
                                                                          const uint32_t l1pvPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 2);
    if (pRowNum != 1) {
        AscendC::DataCopy(
                    l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                    mm2InGm[pCoreOffset],
                    AscendC::Nd2NzParams(1,           // ndNum
                                         pRowNum, // nValue
                                         kvSeqTile, // dValue
                                         0,           // srcNdMatrixStride, unused
                                         curSeqlen_,        // srcDValue
                                         pRowNumRound,   // dstNzC0Stride
                                         1,           // dstNzNStride
                                         0));         // dstNzMatrixStride, unused
    } else {
                AscendC::DataCopy(
                    l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
                    mm2InGm[pCoreOffset],
                    AscendC::DataCopyParams(1,
                                            CeilDiv(kvSeqTileRound, BLOCK_SIZE),
                                            0,
                                            0));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 2);

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 4); //
    AscendC::DataCopy(
        l1vBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
        valueGm[vCoreOffset],
        AscendC::Nd2NzParams(1,           // ndNum
                             kvSeqTile, // nValue
                             headSizeVo, // dValue
                             0,           // srcNdMatrixStride, unused
                             strideV,        // srcDValue
                             kvSeqTileRound,   // dstNzC0Stride
                             1,           // dstNzNStride
                             0));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 4);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::LoadQKToL0(const uint32_t qRowNum,
                                                                          const uint32_t qRowNumRound,
                                                                          const uint32_t headSizeQkRound,
                                                                          const uint32_t kvSeqTileRound,
                                                                          const uint32_t l1qPingPongFlag,
                                                                          const uint32_t l1kPingPongFlag,
                                                                          const uint32_t sIdx,
                                                                          const uint32_t sLoop)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    if (qRowNum != 1) {
        for (uint64_t l0aLoadIdx = 0; l0aLoadIdx < qRowNumRound / BLOCK_SIZE; ++l0aLoadIdx) {
            AscendC::LoadData(
                l0aBufTensor[l0aLoadIdx * headSizeQkRound * BLOCK_SIZE],
                l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + l0aLoadIdx * 256],
                AscendC::LoadData2dParams(0, headSizeQkRound / BLOCK_SIZE, qRowNumRound / BLOCK_SIZE,
                                          0, 0, false, 0));
        }
    } else {
        AscendC::LoadData(
            l0aBufTensor,
            l1qBufAddrTensor[l1qPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
            AscendC::LoadData2dParams(0, CeilDiv(headSizeQkRound, 256), 1,
                                      0, 0, false, 0));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(0);
    if (sIdx == sLoop - 1) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1qPingPongFlag + 6);
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::LoadData(
        l0bBufTensor,
        l1kBufAddrTensor[l1kPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
        AscendC::LoadData2dParams(0, headSizeQkRound * kvSeqTileRound / 256, 1,
                                  0, 0, false, 0));
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1kPingPongFlag);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(1);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::LoadPVToL0(const uint32_t pRowNum,
                                                                          const uint32_t pRowNumRound,
                                                                          const uint32_t headSizeVoRound,
                                                                          const uint32_t kvSeqTileRound,
                                                                          const uint32_t l1pvPingPongFlag,
                                                                          const uint32_t l0abPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag);
    if (pRowNum != 1) {
        for (uint64_t l0aLoadIdx = 0; l0aLoadIdx < pRowNumRound / BLOCK_SIZE; ++l0aLoadIdx) {
            AscendC::LoadData(
                l0aBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM + l0aLoadIdx * kvSeqTileRound * BLOCK_SIZE],
                l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + l0aLoadIdx * 256],
                AscendC::LoadData2dParams(0, kvSeqTileRound / BLOCK_SIZE, pRowNumRound / BLOCK_SIZE,
                                          0, 0, false, 0));
        }
    } else {
        AscendC::LoadData(
            l0aBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM],
            l1pBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV],
            AscendC::LoadData2dParams(0, CeilDiv(kvSeqTileRound, 256), 1,
                                      0, 0, false, 0));
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 2);

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l0abPingPongFlag + 4);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag + 2);
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.startIndex = 0;
    loadDataParams.dstFracGap = 0;
    loadDataParams.repeatTimes = headSizeVoRound / BLOCK_SIZE;
    loadDataParams.srcStride = kvSeqTileRound / BLOCK_SIZE;
    loadDataParams.dstGap = 0;
    for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < kvSeqTileRound / BLOCK_SIZE; ++l0bLoadIdx) {
        AscendC::LoadDataWithTranspose(
            l0bBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM + l0bLoadIdx * headSizeVoRound * BLOCK_SIZE],
            l1vBufAddrTensor[l1pvPingPongFlag * BASE_L1_BLOCK_ELEM_NUM_QKV + l0bLoadIdx * 256],
            loadDataParams);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 4);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1pvPingPongFlag + 4);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::PerformQKMmad(const uint32_t qRowNum,
                                                                             const uint32_t kvSeqTile,
                                                                             const uint32_t l0cPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag);
    AscendC::Mmad(
        l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM],
        l0aBufTensor,
        l0bBufTensor,
        AscendC::MmadParams(qRowNum, kvSeqTile, headSizeQk, 0, false, 1));
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::PerformPVMmad(const uint32_t pRowNum,
                                                                             const uint32_t kvSeqTile,
                                                                             const uint32_t l0abPingPongFlag,
                                                                             const uint32_t l0cPingPongFlag,
                                                                             const uint32_t sIdx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0abPingPongFlag + 4);
    if (sIdx == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag + 2);
    }
    AscendC::Mmad(
        l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM],
        l0aBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM],
        l0bBufTensor[l0abPingPongFlag * BASE_L0AB_BLOCK_ELEM_NUM],
        AscendC::MmadParams(pRowNum, headSizeVo, kvSeqTile, 0, false, sIdx == 0));
    AscendC::PipeBarrier<PIPE_M>();
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0abPingPongFlag + 2);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopySToWorkSpace(const uint32_t qRowNum,
                                                                                const uint32_t qRowNumRound,
                                                                                const uint32_t kvSeqTileRound,
                                                                                const uint32_t curKvHeadIdx,
                                                                                const uint32_t sIdx,
                                                                                const uint32_t l0cPingPongFlag,
                                                                                const uint32_t mm1ResPingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
    auto intriParams = AscendC::FixpipeParamsV220(
        kvSeqTileRound, // nSize
        qRowNum, // mSize
        qRowNumRound,   // srcStride
        curSeqlenRound_,   // dstStride
        false);      // enRelu
    intriParams.quantPre = QuantMode_t::NoQuant;
    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
        mm1ResGm[mm1ResPingPongFlag * (mm1ResWorkSpaceSize / DOUBLE_BUFFER / sizeof(float)) + // db offset
                 cubeBlockIdx * workSpaceElemNum +                    // core offset
                 curKvHeadIdx * qSeqLenCurProcess * groupSize * curSeqlenRound_ +          // row offset
                 sIdx * blockSize],                                   // column offset
        l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM],
        intriParams);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyOToGm(const uint32_t pRowNum,
                                                                         const uint32_t pRowNumRound,
                                                                         const uint32_t oCoreOffset,
                                                                         const uint32_t l0cPingPongFlag)
{
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0cPingPongFlag);
    auto intriParams = AscendC::FixpipeParamsV220(
        headSizeVo, // nSize
        pRowNum, // mSize
        pRowNumRound,   // srcStride
        headSizeVo,   // dstStride
        false);      // enRelu
    if (std::is_same<OUT_T, __bf16>::value) {
        intriParams.quantPre = QuantMode_t::F322BF16;
    } else {
        intriParams.quantPre = QuantMode_t::F322F16;
    }
    AscendC::Fixpipe<OUT_T, float, AscendC::CFG_ROW_MAJOR>(
        outGm[oCoreOffset], l0cBufTensor[l0cPingPongFlag * BASE_L0C_BLOCK_ELEM_NUM], intriParams);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0cPingPongFlag + 2);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::ProcessMm1(const uint32_t processIdx)
{
    uint64_t strideQ = qHeadNum * headSizeQk;
    uint64_t strideK = kvHeadNum * headSizeQk;
    uint32_t headSizeQkRound = AlignUp(headSizeQk, 16U);
    uint32_t mm1ResPingPongFlag = (processIdx / aicNum) % 2;
    for (uint32_t curKvHeadIdx = 0; curKvHeadIdx < curKvHeadNum; curKvHeadIdx++) {
        uint32_t actualKvHeadIdx = kvHeadOffset + curKvHeadIdx;
        uint32_t l1qPingPongFlag = curKvHeadIdx % 2;
        uint32_t qRowNum = groupSize * qSeqLenCurProcess;
        uint32_t qRowNumRound = (qRowNum + 16U - 1) / 16U * 16U;

        CopyQToL1(qRowNum, qRowNumRound, headSizeQkRound, actualKvHeadIdx, l1qPingPongFlag);
        uint32_t sLoop = (static_cast<uint32_t>(curSeqlen_) + blockSize - 1) / blockSize;
        uint32_t kvSeqTile = blockSize;

        for (uint32_t sIdx = 0; sIdx < sLoop; sIdx++) {
            uint32_t l1kPingPongFlag = (curKvHeadIdx * sLoop + sIdx) % 2;
            uint32_t l0cPingPongFlag = (curKvHeadIdx * sLoop + sIdx) % 2;
            if (sIdx == sLoop - 1) {
                kvSeqTile = curSeqlen_ - sIdx * blockSize;
            }
            uint32_t kvSeqTileRound = AlignUp(kvSeqTile, 16U);
            uint32_t blockTableOffset = maxBlockNumPerBatch * bIdx + sIdx;
            uint32_t blockIdx = blockTableGm.GetValue(blockTableOffset);
            uint32_t kBlockOffset = blockIdx * blockSize * strideK;
            uint32_t kHiddenOffset = actualKvHeadIdx * headSizeQk;
            uint32_t kCoreOffset = kBlockOffset + kHiddenOffset;

            CopyKToL1(kvSeqTile, kvSeqTileRound, strideK, kCoreOffset, l1kPingPongFlag);

            LoadQKToL0(qRowNum, qRowNumRound, headSizeQkRound, kvSeqTileRound,
                       l1qPingPongFlag, l1kPingPongFlag, sIdx, sLoop);

            PerformQKMmad(qRowNum, kvSeqTile, l0cPingPongFlag);

            CopySToWorkSpace(qRowNum, qRowNumRound, kvSeqTileRound,
                             curKvHeadIdx, sIdx, l0cPingPongFlag, mm1ResPingPongFlag);
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::ProcessMm2(const uint32_t processIdx)
{
    uint64_t strideO = qHeadNum * headSizeVo;
    uint64_t strideV = kvHeadNum * headSizeVo;
    uint32_t mm2InPingPongFlag = (processIdx / aicNum) % 2;
    for (uint32_t curKvHeadIdx = 0; curKvHeadIdx < curKvHeadNum; curKvHeadIdx++) {
        uint32_t actualKvHeadIdx = kvHeadOffset + curKvHeadIdx;

        for (uint32_t qSeqLenCurProcessIndex = 0; qSeqLenCurProcessIndex < qSeqLenCurProcess;
             qSeqLenCurProcessIndex++) {
            uint64_t oCoreOffset = (qSeqLenCumSum + qSeqLenOffset + qSeqLenCurProcessIndex) * strideO
                                 + actualKvHeadIdx * groupSize * headSizeVo;

            uint32_t pRowNum = groupSize;
            uint32_t pRowNumRound = AlignUp(pRowNum, 16U);
            uint32_t headSizeVoRound = AlignUp(headSizeVo, 16U);
            uint32_t sLoop = (static_cast<uint32_t>(curSeqlen_) + blockSize - 1) / blockSize;
            uint32_t kvSeqTile = blockSize;
            uint32_t l0cPingPongFlag = (curKvHeadIdx * qSeqLenCurProcess + qSeqLenCurProcessIndex) % 2;
            for (uint32_t sIdx = 0; sIdx < sLoop; sIdx++) {
                uint32_t l1pvPingPongFlag =
                    (curKvHeadIdx * qSeqLenCurProcess * sLoop + qSeqLenCurProcessIndex * sLoop + sIdx) % 2;
                uint32_t l0abPingPongFlag =
                    (curKvHeadIdx * qSeqLenCurProcess * sLoop + qSeqLenCurProcessIndex * sLoop + sIdx) % 2;
                if (sIdx == sLoop - 1) {
                    kvSeqTile = curSeqlen_ - sIdx * blockSize;
                }
                uint32_t kvSeqTileRound = AlignUp(kvSeqTile, 16U);
                uint64_t pCoreOffset = mm2InPingPongFlag * (mm2InWorkSpaceSize / 2 / 2)
                    + cubeBlockIdx * workSpaceElemNum
                    + (curKvHeadIdx * qSeqLenCurProcess + qSeqLenCurProcessIndex) * groupSize * curSeqlen_
                    + sIdx * blockSize;
                uint32_t blockTableOffset = maxBlockNumPerBatch * bIdx + sIdx;
                uint32_t blockIdx = blockTableGm.GetValue(blockTableOffset);
                uint32_t vBlockOffset = blockIdx * blockSize * strideV;
                uint32_t vHiddenOffset = actualKvHeadIdx * headSizeVo;
                uint32_t vCoreOffset = vBlockOffset + vHiddenOffset;

                CopyPVToL1(pRowNum, pRowNumRound, kvSeqTile, kvSeqTileRound, strideV,
                        pCoreOffset, vCoreOffset, l0abPingPongFlag, l1pvPingPongFlag);

                LoadPVToL0(pRowNum, pRowNumRound, headSizeVoRound, kvSeqTileRound, l1pvPingPongFlag, l0abPingPongFlag);

                PerformPVMmad(pRowNum, kvSeqTile, l0abPingPongFlag, l0cPingPongFlag, sIdx);
            }
            CopyOToGm(pRowNum, pRowNumRound, oCoreOffset, l0cPingPongFlag);
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::PreProcess(const uint32_t processIdx)
{
    uint32_t processCumSum = 0;
    int64_t curQSeqlen = 0;
    uint32_t qSeqLenSplitNum = 0;
    qSeqLenCumSum = 0;
    for (uint32_t i = 0; i < batchSize; i++) {
        curQSeqlen = (qSeqSize == 1) ? 1: actualQSeqLenGm.GetValue(i);
        qSeqLenSplitNum = (static_cast<uint32_t>(curQSeqlen) + qSeqLenSplitSize - 1) / qSeqLenSplitSize;
        if ((processCumSum + qSeqLenSplitNum * kvHeadSplitNum - 1) >= processIdx) {
            bIdx = i;
            break;
        }
        qSeqLenCumSum +=  (LAYOUT_T == LAYOUT::TND) ? curQSeqlen : qSeqSize;
        processCumSum += qSeqLenSplitNum * kvHeadSplitNum;
    }

    uint32_t processOffsetInCurBatch = processIdx - processCumSum;
    uint32_t qSeqLenSplitIdx = processOffsetInCurBatch / kvHeadSplitNum;
    qSeqLenOffset = qSeqLenSplitIdx * qSeqLenSplitSize;
    qSeqLenCurProcess = (qSeqLenSplitIdx == qSeqLenSplitNum - 1) ? (curQSeqlen - qSeqLenOffset) : qSeqLenSplitSize;

    uint32_t kvHeadSplitIdx = processOffsetInCurBatch % kvHeadSplitNum;
    kvHeadOffset = kvHeadSplitIdx * kvHeadSplitSize;
    curKvHeadNum = (kvHeadSplitIdx == kvHeadSplitNum - 1) ? (kvHeadNum - kvHeadOffset) : kvHeadSplitSize;
    curSeqlen_ = actualKvSeqLenGm.GetValue(bIdx);
    curSeqlenRound_ = AlignUp(curSeqlen_, 16L);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::ProcessImportanceScore(
    const uint32_t processIdx, const uint32_t kvHeadOffset, const uint32_t curKvHeadNum)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(IMPORTANCE_SCORE_EVENT_ID);
    LoadImportanceScoreParam();
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(IMPORTANCE_SCORE_EVENT_ID);

    uint16_t totalRow = curKvHeadNum * groupSize;

    uint16_t l1LoopMaxCol = L1_SIZE / totalRow / IMPORTANCE_SCORE_PARAM_ROW * IMPORTANCE_SCORE_PARAM_ROW;
    uint16_t l1LoopCount = (curSeqlen_ + l1LoopMaxCol - 1) / l1LoopMaxCol;
    uint16_t l1LoopTailCol = curSeqlen_ % l1LoopMaxCol == 0 ? l1LoopMaxCol : curSeqlen_ % l1LoopMaxCol;
    uint32_t scoreOffset = cubeBlockIdx * workSpaceElemNum;

    Nd2NzParams l1CopyParam {
        1,                               // ndNum
        totalRow,                        // nValue
        l1LoopMaxCol,                    // dValue
        0,                               // srcNdMatrixStride
        (uint16_t)(curSeqlen_),          // srcDValue
        (uint16_t)AlignUp(totalRow, 8),  // dstNzC0Stride
        1,                               // dstNzNStride
        0                                // dstNzMatrixStride
    };

    impScoreResultOffset_ = cubeBlockIdx * impScoreResultEleNum_;
    impScoreResultRelativeOffset_ = 0;
    for (uint16_t l1LoopIdx = 0, l1LoopCol = l1LoopMaxCol; l1LoopIdx < l1LoopCount; l1LoopIdx++) {
        if (l1LoopIdx + 1 == l1LoopCount) {
            l1LoopCol = l1LoopTailCol;
            l1CopyParam.dValue = l1LoopCol;
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(IMPORTANCE_SCORE_EVENT_ID);
        AscendC::DataCopy(impScoreL1aTensor, scoreInGm[scoreOffset], l1CopyParam);

        AscendC::SetFlag<HardEvent::MTE2_MTE1>(IMPORTANCE_SCORE_EVENT_ID);
        AscendC::WaitFlag<HardEvent::MTE2_MTE1>(IMPORTANCE_SCORE_EVENT_ID);

        SplitImportanceScoreL0A(totalRow, l1LoopIdx, l1LoopCol);

        scoreOffset += l1LoopMaxCol;
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::SplitImportanceScoreL0A(
    const uint16_t totalRow, const uint16_t l1LoopIdx, const uint16_t l1LoopCol)
{
    uint8_t blockColCount = CeilCubeBlock(IMPORTANCE_SCORE_PARAM_ROW, 8);
    uint8_t blockRowCount = CeilCubeBlock(totalRow, 16);
    LoadData2DParams loadParams { 0,               // startIndex
                                  blockColCount,   // repeatTimes
                                  blockRowCount,   // srcStride
                                  0,               // sid
                                  0,               // dstGap
                                  false,           // ifTranspose
                                  0 };             // addrMode
    bool useAtomicAdd = false;
    uint32_t l1ASrcOffset = 0;
    uint32_t l1ASrcOffsetGap = FP32_CUBE_BLOCK_SIZE;
    uint32_t l1ASrcOffsetRoundGap = blockRowCount * (blockColCount - 1) * FP32_CUBE_BLOCK_SIZE;
    uint32_t l0ADstOffset = 0;
    uint32_t l0ADstOffsetGap = blockColCount * FP32_CUBE_BLOCK_SIZE;
    uint16_t l0LoopCount = (l1LoopCol + IMPORTANCE_SCORE_PARAM_ROW - 1) / IMPORTANCE_SCORE_PARAM_ROW;
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(IMPORTANCE_SCORE_EVENT_ID + 1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(IMPORTANCE_SCORE_EVENT_ID + 1);
    bool l0PingPongFlag = false;
    for (uint16_t l0LoopIdx = 0; l0LoopIdx < l0LoopCount; l0LoopIdx++) {
        l0ADstOffset = l0PingPongFlag * L0A_PING_PONG_SIZE;
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);
        for (uint16_t row = 0; row < blockRowCount; row++) {
            AscendC::LoadData(impScoreL0aTensor[l0ADstOffset], impScoreL1aTensor[l1ASrcOffset], loadParams);
            l1ASrcOffset += l1ASrcOffsetGap;
            l0ADstOffset += l0ADstOffsetGap;
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);
        if (l0LoopIdx + 1 == l0LoopCount) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(IMPORTANCE_SCORE_EVENT_ID);
        }
        ProcessImportanceScoreMmad(totalRow, l0PingPongFlag);

        useAtomicAdd = (l1LoopIdx > 0 || l0LoopIdx > 0) && impScoreParamHasPrefix_;

        CopyImportanceScoreOut(useAtomicAdd, totalRow, l0PingPongFlag);

        impScoreResultOffset_ += impScoreParamValidCol_;
        impScoreResultRelativeOffset_ += impScoreParamValidCol_;
        l1ASrcOffset += l1ASrcOffsetRoundGap;
        l0PingPongFlag = !l0PingPongFlag;
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(IMPORTANCE_SCORE_EVENT_ID + 1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(IMPORTANCE_SCORE_EVENT_ID + 1);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::ProcessImportanceScoreMmad(
    uint16_t totalRow, bool l0PingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);

    AscendC::Mmad(
           impScoreL0cTensor[l0PingPongFlag * L0C_PING_PONG_SIZE],
           impScoreL0aTensor[l0PingPongFlag * L0A_PING_PONG_SIZE],
           impScoreL0bTensor,
           AscendC::MmadParams(totalRow, impScoreParamTotalCol_, IMPORTANCE_SCORE_PARAM_ROW, 0, false, true));
    AscendC::PipeBarrier<PIPE_M>();
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::CopyImportanceScoreOut(
    bool useAtomicAdd, uint16_t totalRow, bool l0PingPongFlag)
{
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);
    if (likely(useAtomicAdd)) {
        AscendC::SetAtomicAdd<float>();
    }
    auto copyCol = impScoreParamTotalCol_;
    if (impScoreResultRelativeOffset_ + copyCol >= impScoreResultColPad_) {
        copyCol = impScoreResultColPad_ - impScoreResultRelativeOffset_ - 1;
    }
    AscendC::FixpipeParamsV220 intriParams {
        copyCol,                // nSize
        totalRow,                              // mSize
        totalRow,                              // srcStride
        impScoreResultColPad_,                 // dstStride
        false                                  // enRelu
    };

    AscendC::Fixpipe<float, float, AscendC::CFG_ROW_MAJOR>(
        impScoreResultGm_[impScoreResultOffset_], impScoreL0cTensor[l0PingPongFlag * L0C_PING_PONG_SIZE], intriParams);

    if (likely(useAtomicAdd)) {
        AscendC::SetAtomicNone();
    }
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(IMPORTANCE_SCORE_EVENT_ID + l0PingPongFlag);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::LoadImportanceScoreParam()
{
    uint8_t colPad = CeilCubeBlock(impScoreParamTotalCol_, 16) * 16;
    Nd2NzParams l1CopyParam {
        1,                           // ndNum
        impScoreParamTotalCol_,      // nValue
        IMPORTANCE_SCORE_PARAM_ROW,  // dValue
        0,                           // srcNdMatrixStride
        IMPORTANCE_SCORE_PARAM_ROW,  // srcDValue
        colPad,               // dstNzC0Stride
        1,                           // dstNzNStride
        0                            // dstNzMatrixStride
    };
    AscendC::DataCopy(impScoreL1bTensor, impScoreParamGm_, l1CopyParam);

    AscendC::SetFlag<HardEvent::MTE2_MTE1>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<HardEvent::MTE2_MTE1>(IMPORTANCE_SCORE_EVENT_ID);

    AscendC::LoadData(
        impScoreL0bTensor,
        impScoreL1bTensor,
        AscendC::LoadData2dParams(0, colPad * IMPORTANCE_SCORE_PARAM_ROW * sizeof(float) / 512, 1,
                                  0, 0, false, 0));
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAic<NCAIType>::Process()
{
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(4);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(5);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(6);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(7);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(3);
    for (uint32_t processIdx = cubeBlockIdx; processIdx < processNum; processIdx+=aicNum) {
        PreProcess(processIdx);
        ProcessMm1(processIdx);
        CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(MM1_READY);
        CrossCoreWaitFlag(SOFTMAX_READY);
        if constexpr(NCAIType::IMP_SCORE_OPT) {
            ProcessImportanceScore(processIdx, kvHeadOffset, curKvHeadNum);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(IMPORTANCE_SCORE_READY);
        }
        ProcessMm2(processIdx);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(4);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(5);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(7);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(0);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(1);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(2);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(3);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(1);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(2);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(3);
}

#elif __DAV_C220_VEC__

template <typename NCAIType>
class NsaCompressAttentionInferAiv {
public:
    using Q_T = typename NCAIType::queryType;
    using KV_T = typename NCAIType::kvType;
    using OUT_T = typename NCAIType::outputType;
    static constexpr LAYOUT LAYOUT_T = NCAIType::layout;

    __aicore__ inline NsaCompressAttentionInferAiv(){};

    __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockTable, GM_ADDR actualQSeqLen,
        GM_ADDR actualKvSeqLen, GM_ADDR actualSelKvSeqLen, GM_ADDR output, GM_ADDR topkIndicesOut,
        GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData);

    __aicore__ inline void Process();

protected:

    /* tiling data */
    NsaCompressAttentionInferBaseParams baseTilingData;
    NsaCompressAttentionInferSplitBNParams splitBNTilingData;

    /* variable */
    AscendC::GlobalTensor<int32_t> topkOutGm;

    GlobalTensor<int64_t> actualQSeqLenGm;
    GlobalTensor<int64_t> actualSelKvSeqLenGm;
    GlobalTensor<int64_t> actualKvSeqLenGm;
    GlobalTensor<float> mm1ResGm;
    GlobalTensor<float> scoreInGm;
    GlobalTensor<float> topKInGm;
    GlobalTensor<Q_T> mm2InGm;
    AscendC::GlobalTensor<float> impScoreResultGm_;
    AscendC::GlobalTensor<float> impScoreParamGm_;
    AscendC::LocalTensor<float> impScoreReduceResult_;
    AscendC::LocalTensor<uint8_t> impScoreReduceTmpBuffer_;
    AscendC::LocalTensor<int32_t> impScoreReduceResultInt_;

    AsdopsBuffer<ArchType::ASCEND_V220> buf;
    __gm__ uint8_t *blocktablePtr = nullptr;

    SoftMaxTiling softmaxTilingData;
    TopkTiling topkTilingData;

    uint32_t vecBlockIdx;
    uint32_t cubeBlockIdx;
    uint32_t bIdx;

    // 基础参数
    uint32_t batchSize;
    uint32_t qSeqSize;
    uint32_t seqSize;
    uint32_t blockSize;
    uint32_t groupSize;
    uint32_t qHeadNum;
    uint32_t kvHeadNum;
    uint32_t headSizeQk;
    uint32_t headSizeVo;
    float scaleValue;
    uint32_t selectNum;

    // 新增
    uint32_t maxBlockNumPerBatch;
    uint32_t workSpaceElemNum;
    uint32_t mm1ResWorkSpaceSize;
    uint32_t mm2InWorkSpaceSize;
    uint32_t scoreInWorkSpaceSize;
    uint32_t topKInWorkSpaceSize;
    uint32_t impScoreWorkSpaceSize_;
    uint32_t impScoreResultEleNum_;
    uint32_t impScoreResultCol_;
    uint32_t impScoreResultColPad_;
    bool impScoreParamHasPrefix_;
    uint32_t impScoreReduceRound_;
    uint32_t impScoreReduceCol_;
    uint32_t impScoreReduceColPad_;
    uint32_t impScoreReduceTailCol_;
    uint32_t impScoreReduceTailColPad_;

    uint32_t processPerBatch;
    uint32_t kvHeadSplitSize;
    uint32_t kvHeadSplitNum;
    uint32_t qSeqLenSplitSize;
    uint32_t qSeqLenOffset;
    uint32_t qSeqLenCurProcess;
    uint32_t qSeqLenCumSum;
    uint32_t kvHeadOffset;
    uint32_t curKvHeadNum;
    uint32_t attenMaskFlag;
    uint32_t processNum;
    uint32_t aicNum;

    uint32_t coreNumUsed;
    uint32_t bnPerHeadCore;
    uint32_t bnPerTailCore;
    uint32_t rowNumVec0;      // 第1个核处理的行数
    uint32_t rowNumVec1;      // 第2个核处理的行数
    uint32_t basicRowNumSMVec0;   // 第1个核每次计算的行数
    uint32_t basicRowNumSMVec1;   // 第2个每次计算的行数
    uint32_t softmaxRowLenPerCore;
    uint32_t softmaxBasicRowLen;
    uint32_t softmaxRowLoop;
    uint32_t basicRowLenCal;
    int64_t curSeqlen;
    uint8_t softmaxRightPadding;
    uint64_t softmaxTmpVecGmOffset;
    uint64_t mm2InWorkSpaceOffset;
    uint64_t scoreInWorkSpaceOffset;
    uint32_t alignedColLen;
    uint32_t softmaxTileLength;
    uint32_t halfTileLength;
    uint32_t groupNumPerSubCore; // 每个vector核分到的N2数量
    uint32_t currentQSeqLenOffset; // 当前softmaxloop属于哪一个token

    AscendC::LocalTensor<float> softmaxInputbufTensor;
    AscendC::LocalTensor<float> softmaxOut32bufTensor;
    AscendC::LocalTensor<Q_T> softmaxOut16bufTensor;
    AscendC::LocalTensor<uint8_t> sharedTmpBuffer;

    // importance score params
    uint32_t ubSize = 0;
    uint32_t baseBlockSize = 0;
    uint32_t gHeadNums = 0;
    uint32_t gHeadNumsAlign;
    uint32_t outS2 = 0;
    uint32_t maxOutS2 = 0;
    uint32_t maxBaseS2 = 0;
    uint32_t maxN = 0;
    uint32_t maxM = 0;
    uint32_t maxBaseJ = 0;
    uint32_t strideOut = 0;
    uint32_t compSizeL = 0;
    uint32_t compStrideD = 0;
    uint32_t selectSize = 0;
    uint64_t impScoreInputGmOffset;
    uint64_t impScoreOutputGmOffset;
    uint32_t impSocreCoreRowCount = 0;
    uint32_t maxRowCountPerLoop = 0;
    int32_t FLOAT_TYPE_MASK = 0x7f800000; // 0111 1111 1000 0000 0000 0000 0000 0000
    float FLOAT_MIN_MASK = -3e38;
    AscendC::LocalTensor<float> pslcTensor;
    AscendC::LocalTensor<float> pslcCalcTensor;
    AscendC::LocalTensor<float> pslcTmpTensor;
    AscendC::LocalTensor<uint8_t> pslcSharedBuffer;

    // topk params
    uint8_t topkRightPadding;
    uint32_t alignedoutS2;
    uint32_t alignedTok32;
    uint32_t alignedTokloop;
    AscendC::LocalTensor<float> topkInputbufTensor;
    AscendC::LocalTensor<float> topkoutvalueLocal;
    AscendC::LocalTensor<int32_t> topkoutindexLocal;
    AscendC::LocalTensor<int32_t> topksrcindexLocal;
    AscendC::LocalTensor<bool> topksrcLocalFinish;
    uint64_t topkInputGmOffset;
    uint64_t topkOutputGmOffset;
    AscendC::LocalTensor<uint8_t> topksharedTmpBuffer;
    AscendC::LocalTensor<int32_t> arithbuffer;

    __aicore__ inline void InitTilingData(const NsaCompressAttentionInferTilingData *__restrict tilingData);
    __aicore__ inline void PreProcessOffset(const uint32_t processIdx);
    __aicore__ inline void PreProcess(const uint32_t processIdx);
    __aicore__ inline void ProcessSoftmax(uint32_t processIdx);
    __aicore__ inline void SoftmaxComputeVecInGmOffset(uint32_t bn2Idx, uint32_t ridx);
    __aicore__ inline void SoftmaxCopyIn(DataCopyParams &splitCopyinParams, uint32_t ridx);
    __aicore__ inline void SoftmaxCompute(uint32_t ridx);
    __aicore__ inline void SoftmaxCopyOut(DataCopyParams &splitCopyoutParams,
        DataCopyParams &splitCopyout32Params, uint32_t ridx);
    __aicore__ inline void ComputeCurrentTokenOffset(uint32_t startIdx);
    __aicore__ inline void AddMask(uint32_t ridx, uint32_t startIdx, uint32_t endIdx);
    __aicore__ inline void MaskOperation(uint32_t ridx, uint32_t startIdx, uint32_t endIdx);
    __aicore__ inline void MaskApply(uint32_t ridx);

    // importance score
    __aicore__ inline void InitImportanceScoreParams();
    __aicore__ inline void ProcessImportanceScore(uint32_t processIdx);
    __aicore__ inline void ClearImpScoreWorspace();
    __aicore__ inline void GenerateImpScoreParam();
    __aicore__ inline void ProcessImportanceScoreForCube(uint32_t processIdx);
    __aicore__ inline void ComputeImportanceScoreReduce(uint32_t colLoopIdx, uint32_t reduceCol, uint32_t reduceColPad,
                                                        uint32_t srcOffset);
    __aicore__ inline void ProcessImpScoreS2Loop(uint32_t loopCnt, uint32_t processIdx, uint32_t taskId,
        uint32_t startRowIdx, uint32_t endRowIdx);
    __aicore__ inline void ComputeImpScoreValue(uint32_t loopCnt, uint32_t loopOutS2, uint32_t taskId,
        uint32_t baseS2Id, uint32_t startRowIdx, uint32_t endRowIdx, uint32_t startJ);
    __aicore__ inline void ImpScoreDataCopyIn(int64_t offset, uint16_t blockCount, uint32_t blockLength);
    __aicore__ inline void ImpScoreDataCopyOut(uint32_t taskId, uint32_t baseS2Id, int64_t offset,
        uint16_t blockCount, uint32_t blockLength, uint32_t srcStride);
    __aicore__ inline void ImpScoreTransposeTensor(uint32_t height, uint32_t width);
    __aicore__ inline void ImpScoreAccumulation(uint32_t startJ, uint32_t rowCount, uint32_t offsetS2Split);
    __aicore__ inline void ImpScoreReduceSum(uint32_t startJ, uint32_t rowCount, uint32_t outNewS2);

    // topk
    __aicore__ inline void ProcessTopK();
    __aicore__ inline void TopkComputeVecInGmOffset();
    __aicore__ inline void TopkCopyIn(DataCopyParams &splitCopyintopkParams);
    __aicore__ inline void TopkCompute();
    __aicore__ inline void TopkCopyOut(DataCopyParams &splitCopyouttopkParams);
};

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::Init(GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR blockTable, GM_ADDR actualQSeqLen, GM_ADDR actualKvSeqLen, GM_ADDR actualSelKvSeqLen, GM_ADDR output,
    GM_ADDR topkIndicesOut, GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    this->aicNum = GetBlockNum();

    this->splitBNTilingData = tilingData->splitBNParams;

    InitTilingData(tilingData);
    vecBlockIdx = GetBlockIdx();
    this->cubeBlockIdx = vecBlockIdx / GetSubBlockNum();

    this->actualQSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualQSeqLen);
    this->actualSelKvSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualSelKvSeqLen);
    this->actualKvSeqLenGm.SetGlobalBuffer((__gm__ int64_t *)actualKvSeqLen);
    this->topkOutGm.SetGlobalBuffer((__gm__ int32_t *)topkIndicesOut);

    mm1ResGm.SetGlobalBuffer((__gm__ float *)workspace);
    uint64_t offset = mm1ResWorkSpaceSize;
    scoreInGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += scoreInWorkSpaceSize;
    mm2InGm.SetGlobalBuffer((__gm__ Q_T *)(workspace + offset));
    offset += mm2InWorkSpaceSize;
    topKInGm.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += topKInWorkSpaceSize;
    impScoreResultGm_.SetGlobalBuffer((__gm__ float *)(workspace + offset));
    offset += impScoreWorkSpaceSize_;
    impScoreParamGm_.SetGlobalBuffer((__gm__ float *)(workspace + offset));
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::InitTilingData(
    const NsaCompressAttentionInferTilingData *__restrict tilingData)
{
    batchSize = tilingData->baseParams.batchSize;
    qSeqSize = tilingData->baseParams.qSeqSize;
    seqSize = tilingData->baseParams.seqSize;
    kvHeadNum = tilingData->baseParams.kvHeadNum;
    qHeadNum = tilingData->baseParams.qHeadNum;
    headSizeQk = tilingData->baseParams.headSizeQk;
    headSizeVo = tilingData->baseParams.headSizeVo;
    blockSize = tilingData->baseParams.blockSize;
    scaleValue = tilingData->baseParams.scaleValue;
    selectNum = tilingData->baseParams.selectNum;
    groupSize = tilingData->baseParams.groupSize;
    attenMaskFlag = tilingData->baseParams.attenMaskFlag;

    maxBlockNumPerBatch = tilingData->baseParams.maxBlockNumPerBatch;
    workSpaceElemNum = tilingData->baseParams.workSpaceElemNum;
    mm1ResWorkSpaceSize = tilingData->baseParams.mm1ResWorkSpaceSize;
    mm2InWorkSpaceSize = tilingData->baseParams.mm2InWorkSpaceSize;
    scoreInWorkSpaceSize = tilingData->baseParams.scoreInWorkSpaceSize;
    topKInWorkSpaceSize = tilingData->baseParams.topKInWorkSpaceSize;
    impScoreWorkSpaceSize_ = tilingData->baseParams.impScoreWorkSpaceSize;
    impScoreResultEleNum_ = tilingData->baseParams.impScoreResultEleNum;
    impScoreResultCol_ = tilingData->baseParams.impScoreResultCol;
    impScoreResultColPad_ = tilingData->baseParams.impScoreResultColPad;
    impScoreParamHasPrefix_ = tilingData->baseParams.impScoreParamHasPrefix;
    impScoreReduceRound_ = tilingData->baseParams.impScoreReduceRound;
    impScoreReduceCol_ = tilingData->baseParams.impScoreReduceCol;
    impScoreReduceColPad_ = tilingData->baseParams.impScoreReduceColPad;
    impScoreReduceTailCol_ = tilingData->baseParams.impScoreReduceTailCol;
    impScoreReduceTailColPad_ = tilingData->baseParams.impScoreReduceTailColPad;
    impScoreReduceResult_ = buf.GetBuffer<BufferType::ASCEND_UB, float>(
        groupSize * impScoreReduceColPad_ * sizeof(float));
    impScoreReduceResultInt_ = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(
        groupSize * impScoreReduceColPad_ * sizeof(float));
    // since reduce sum isReuseSource flag is true, should not use this buffer
    impScoreReduceTmpBuffer_ = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(
        (groupSize + 1) * impScoreReduceColPad_ * sizeof(float));

    coreNumUsed = tilingData->splitBNParams.coreNumUsed;

    softmaxTilingData = tilingData->softmaxTilingData;
    topkTilingData = tilingData->topkTilingData;

    processNum = tilingData->splitBNParams.processNum;
    processPerBatch = tilingData->splitBNParams.processPerBatch;
    kvHeadSplitSize = tilingData->splitBNParams.kvHeadSplitSize;
    kvHeadSplitNum = tilingData->splitBNParams.kvHeadSplitNum;
    qSeqLenSplitSize = tilingData->splitBNParams.qSeqLenSplitSize;

    // importance score
    ubSize = tilingData->baseParams.ubSize;
    selectSize = tilingData->baseParams.selectSize;
    compSizeL = tilingData->baseParams.compSizeL;
    compStrideD = tilingData->baseParams.compStrideD;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::PreProcess(const uint32_t processIdx)
{
    curSeqlen = actualKvSeqLenGm.GetValue(bIdx);
    // 对齐列数
    alignedColLen = AlignUp(curSeqlen, 16);

    uint32_t groupNumVec0 = CeilDiv(curKvHeadNum * qSeqLenCurProcess, 2);
    uint32_t groupNumVec1 = (curKvHeadNum * qSeqLenCurProcess) / 2;

    rowNumVec0 = groupNumVec0 * groupSize;
    rowNumVec1 = groupNumVec1 * groupSize;

    softmaxRowLenPerCore = (vecBlockIdx % 2 == 0) ? rowNumVec0 : rowNumVec1;
    uint32_t maxRowNumSM = 12 * 1024 / alignedColLen;
    softmaxBasicRowLen = (maxRowNumSM < softmaxRowLenPerCore) ? maxRowNumSM : softmaxRowLenPerCore;
    softmaxRowLoop = softmaxBasicRowLen == 0 ? 0 : (softmaxRowLenPerCore + softmaxBasicRowLen - 1) / softmaxBasicRowLen;

    // 每次处理的元素数量， 行数*对齐后的列数
    softmaxTileLength = softmaxBasicRowLen * alignedColLen;
    // 列需要填充几个数？
    softmaxRightPadding = alignedColLen - curSeqlen;

    // 设置softmax buffer
    const uint32_t softmaxInputbufOffset = 0;
    const uint32_t softmaxOut32bufOffset = softmaxTileLength * sizeof(float);
    const uint32_t softmaxOut16bufOffset = 2 * softmaxTileLength * sizeof(float);
    const uint32_t softmaxbufOffset = 3 * softmaxTileLength * sizeof(float);
    softmaxInputbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(softmaxInputbufOffset);
    softmaxOut32bufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(softmaxOut32bufOffset);
    softmaxOut16bufTensor = buf.GetBuffer<BufferType::ASCEND_UB, Q_T>(softmaxOut16bufOffset);
    sharedTmpBuffer = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(softmaxbufOffset);

    InitImportanceScoreParams();

    // 设置topk buffer
    const uint32_t topkInputbufOffset = 0;
    const uint32_t topkOutvaluebufOffset = BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    const uint32_t topkOutindexbufOffset = 2 * BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    const uint32_t arithbufferoffset = 3 * BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    const uint32_t topktempbufOffset = 4 * BASE_TOPK_ELEM_NUM_OFFSET * sizeof(float);
    topkInputbufTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(topkInputbufOffset);
    topkoutvalueLocal = buf.GetBuffer<BufferType::ASCEND_UB, float>(topkOutvaluebufOffset);
    topkoutindexLocal = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(topkOutindexbufOffset);
    arithbuffer = buf.GetBuffer<BufferType::ASCEND_UB, int32_t>(arithbufferoffset);
    topksharedTmpBuffer = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(topktempbufOffset);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::InitImportanceScoreParams()
{
    baseBlockSize = (ubSize - 18*1024 - 32*1024) / (2 + 0) / sizeof(float);
    uint32_t t = (curSeqlen-1) * compStrideD + compSizeL;
    gHeadNums = qHeadNum / kvHeadNum;
    gHeadNumsAlign = AlignUp(gHeadNums, 16);

    outS2 = t % selectSize == 0 ? t / selectSize : t / selectSize + 1;
    strideOut = selectSize / compStrideD;
    maxM = selectSize /compStrideD - 1;
    maxN = compSizeL /compStrideD - 1;

    impSocreCoreRowCount = (vecBlockIdx % 2 == 0) ? rowNumVec0 : rowNumVec1;
    uint32_t impSocreCoreRowCountAlign = AlignUp(impSocreCoreRowCount, 16);
    maxRowCountPerLoop = impSocreCoreRowCountAlign > 128 ? 128 : impSocreCoreRowCountAlign;
    maxBaseS2 = baseBlockSize / maxRowCountPerLoop;
    maxBaseJ = (maxBaseS2 - maxM - maxN) / strideOut + 1;
    maxBaseS2 = (maxBaseJ - 1) * strideOut + maxM + maxN + 1;
    baseBlockSize = AlignUp(maxBaseS2, 16) * maxRowCountPerLoop;
    const uint32_t impScoreInputbufOffset = 0;
    const uint32_t impScoreCalcbufOffset = baseBlockSize * sizeof(float);
    const uint32_t impScoreTmpbufOffset = 2 * baseBlockSize * sizeof(float);
    const uint32_t impScoreOffset = 2 * baseBlockSize * sizeof(float) + 1024;

    pslcTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(impScoreInputbufOffset);
    pslcCalcTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(impScoreCalcbufOffset);
    pslcTmpTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(impScoreTmpbufOffset);
    pslcSharedBuffer = buf.GetBuffer<BufferType::ASCEND_UB, uint8_t>(impScoreOffset);

    uint32_t maxT = (seqSize-1) * compStrideD + compSizeL;
    maxOutS2 = maxT % selectSize == 0 ? maxT / selectSize : maxT / selectSize + 1;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::PreProcessOffset(const uint32_t processIdx)
{
    qSeqLenCumSum = 0;
    uint32_t processCumSum = 0;
    int64_t curQSeqlen = 0;
    uint32_t qSeqLenSplitNum = 0;
    for (uint32_t i = 0; i < batchSize; i++) {
        curQSeqlen = (qSeqSize == 1) ? 1: actualQSeqLenGm.GetValue(i);
        qSeqLenSplitNum = (static_cast<uint32_t>(curQSeqlen) + qSeqLenSplitSize - 1) / qSeqLenSplitSize;
        if ((processCumSum + qSeqLenSplitNum * kvHeadSplitNum - 1) >= processIdx) {
            bIdx = i;
            break;
        }
        qSeqLenCumSum += (LAYOUT_T == LAYOUT::TND) ? curQSeqlen : qSeqSize;
        processCumSum += qSeqLenSplitNum * kvHeadSplitNum;
    }

    uint32_t processOffsetInCurBatch = processIdx - processCumSum;

    uint32_t kvHeadSplitIdx = processOffsetInCurBatch % kvHeadSplitNum;
    kvHeadOffset = kvHeadSplitIdx * kvHeadSplitSize;
    curKvHeadNum = (kvHeadSplitIdx == kvHeadSplitNum - 1) ? (kvHeadNum - kvHeadOffset) : kvHeadSplitSize;

    uint32_t qSeqLenSplitIdx = processOffsetInCurBatch / kvHeadSplitNum;
    qSeqLenOffset = qSeqLenSplitIdx * qSeqLenSplitSize;
    qSeqLenCurProcess = (qSeqLenSplitIdx == qSeqLenSplitNum - 1) ? (curQSeqlen - qSeqLenOffset) : qSeqLenSplitSize;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::Process()
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
    if constexpr(!NCAIType::IMP_SCORE_OPT) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
    if (vecBlockIdx == 0) {
        GenerateImpScoreParam();
    }
    for (uint32_t processIdx = cubeBlockIdx; processIdx < processNum; processIdx+=aicNum) {
        PreProcessOffset(processIdx);
        PreProcess(processIdx);
        CrossCoreWaitFlag(MM1_READY);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
        // Softmax
        ProcessSoftmax(processIdx);
        ClearImpScoreWorspace();
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SOFTMAX_READY);
        if constexpr(NCAIType::IMP_SCORE_OPT) {
        CrossCoreWaitFlag(IMPORTANCE_SCORE_READY);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(0); // score搬入等softmax搬出结束
        // ImportanceScore
        if constexpr(NCAIType::IMP_SCORE_OPT) {
            ProcessImportanceScoreForCube(processIdx);
        } else {
            ProcessImportanceScore(processIdx);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(2); // topk搬入等score搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(2); // topk搬入等score搬出结束
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0); // topk Duplicate等score搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0); // topk Duplicate等score搬出结束
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0); // topk SetValue等score搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0); // topk SetValue等score搬出结束
        // TopK
        ProcessTopK();
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
    if constexpr(!NCAIType::IMP_SCORE_OPT) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(3); // softmax搬入等待topk搬出结束
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessSoftmax(uint32_t processIdx)
{
    for (uint32_t ridx = 0; ridx < softmaxRowLoop; ridx++) {
        basicRowLenCal =
                static_cast<uint32_t>((ridx == softmaxRowLoop - 1) ?
                                      (softmaxRowLenPerCore - (softmaxRowLoop - 1) * softmaxBasicRowLen)
                                      : softmaxBasicRowLen);  // 每核处理的最后一个行循环单独处理
        DataCopyParams splitCopyinParams;
        DataCopyParams splitCopyoutParams;
        DataCopyParams splitCopyout32Params;

        splitCopyinParams = {basicRowLenCal, (uint16_t)(alignedColLen * sizeof(float)), 0, 0};
        splitCopyoutParams = {basicRowLenCal, (uint16_t)(curSeqlen * sizeof(half)), 0, 0};

        uint16_t copyoutSrcStride = alignedColLen - curSeqlen >= 8 ? 1 : 0;
        splitCopyout32Params = {basicRowLenCal, (uint16_t)(curSeqlen * sizeof(float)), copyoutSrcStride, 0};
        // 计算从哪个数据开始
        SoftmaxComputeVecInGmOffset(processIdx, ridx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
        SoftmaxCopyIn(splitCopyinParams, ridx);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0); // softmax计算等待softmax搬入结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0); // softmax计算等待softmax搬入结束
        SoftmaxCompute(ridx);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0); // softmax搬出等待softmax计算结束
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0); // softmax搬出等待softmax计算结束

        SoftmaxCopyOut(splitCopyoutParams, splitCopyout32Params, ridx);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(4); // softmax搬入等待softmax搬出结束
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxComputeVecInGmOffset(
    uint32_t processIdx, uint32_t ridx)
{
    uint32_t mm1ResPingPongFlag = (processIdx / aicNum) % 2;
    uint32_t mm2InPingPongFlag = (processIdx / aicNum) % 2;
    uint32_t vecInSplitOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 * alignedColLen;  // 输入对齐
    uint32_t vecOutSplitOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 * curSeqlen;  // 输出不对齐
    uint32_t vecOutSplitOffsetOut = (vecBlockIdx % 2 == 0) ? 0 : workSpaceElemNum / 2;
    // 计算从哪个数据开始处理
    softmaxTmpVecGmOffset = vecBlockIdx / 2 * workSpaceElemNum +
                            mm1ResPingPongFlag * (mm1ResWorkSpaceSize / DOUBLE_BUFFER / sizeof(float)) +
                            vecInSplitOffset + ridx * softmaxBasicRowLen * alignedColLen;
    mm2InWorkSpaceOffset = vecBlockIdx / 2 * workSpaceElemNum +
                           mm2InPingPongFlag * (mm2InWorkSpaceSize / DOUBLE_BUFFER / sizeof(half)) +
                           vecOutSplitOffset + ridx * softmaxBasicRowLen * curSeqlen;
    scoreInWorkSpaceOffset = vecBlockIdx / 2 * workSpaceElemNum + vecOutSplitOffsetOut +
                             ridx * softmaxBasicRowLen * curSeqlen;
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxCopyIn(
    DataCopyParams &splitCopyinParams, uint32_t ridx)
{
    // 非对齐拷贝，不填充
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyPad(softmaxInputbufTensor, mm1ResGm[softmaxTmpVecGmOffset], splitCopyinParams, padParams);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxCopyOut(DataCopyParams &splitCopyoutParams,
    DataCopyParams &splitCopyout32Params,
    uint32_t ridx)
{
    DataCopyPad(mm2InGm[mm2InWorkSpaceOffset], softmaxOut16bufTensor, splitCopyoutParams);

    if constexpr(!NCAIType::IMP_SCORE_OPT) {
    if (ridx == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
        }
    }
    DataCopyPad(scoreInGm[scoreInWorkSpaceOffset], softmaxOut32bufTensor, splitCopyout32Params);
}


template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ComputeCurrentTokenOffset(uint32_t startIdx)
{
    // 已知起点属于哪一个token（qSeqLenOffset），属于哪一个kvhead（kvHeadOffset），当前块的起始地址startIdx
    // 注意：groupIdx是在当前process的索引
    uint32_t groupIdx = startIdx / groupSize;
    // 计算的当前的qseqlen索引
    currentQSeqLenOffset = (groupIdx % qSeqLenCurProcess + qSeqLenOffset);
}


template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::AddMask(
    uint32_t ridx, uint32_t startIdx, uint32_t endIdx)
{
    const int32_t targetCol = curSeqlen - 1; // 目标列索引
    const float minValue = FLOAT_MIN_MASK; // 需设置的常量值

    for (int32_t row = startIdx; row <= endIdx; ++row) {
        // 计算内存偏移：行索引*列总数 + 列索引。这里算出来的是整个vector的偏移，需要计算每个softmaxloop之内的偏移
        uint32_t offset = row * alignedColLen + targetCol - ((vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 * alignedColLen);
        offset -= ridx * softmaxBasicRowLen * alignedColLen;
        softmaxInputbufTensor.SetValue(offset, minValue);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::MaskOperation(
    uint32_t ridx, uint32_t startIdx, uint32_t endIdx)
{
    // 根据当前qseqidx、kvseq的长度（alignedColLen）、压缩的l(压缩窗口的大小)判断当前seq是否需要掩蔽
    // 计算压缩的时候，原始序列多余的token
    // 计算最后一个有效窗户的起始位置, curSeqlen - compSizeL表示了窗口起点可能的最大坐标
    uint32_t curSelKvSeq = actualSelKvSeqLenGm.GetValue(bIdx);
    uint32_t curSelKvSeqAlignStrideD = (actualKvSeqLenGm.GetValue(bIdx) - 1) * compStrideD + compSizeL;
    uint32_t tailKvTokens = curSelKvSeq - curSelKvSeqAlignStrideD;

    ComputeCurrentTokenOffset(startIdx);
    // 判断是否需要mask
    int64_t currentQSeqlen = actualQSeqLenGm.GetValue(bIdx);
    if (currentQSeqlen - currentQSeqLenOffset - 1 > tailKvTokens) {
        AddMask(ridx, startIdx, endIdx);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::MaskApply(uint32_t ridx)
{
    AscendC::SetFlag<AscendC::HardEvent::V_S>(1);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(1);
    uint32_t startRowIdx = (vecBlockIdx % 2 == 0) ? ridx * softmaxBasicRowLen : ridx * softmaxBasicRowLen + rowNumVec0;
    uint32_t endRowIdx = startRowIdx + basicRowLenCal - 1;
    uint32_t curRowIdx = startRowIdx;
    uint32_t nextGroupsizeIdx = startRowIdx;

    // 等号可取，相当于当前只需要计算一行数据
    while (curRowIdx <= endRowIdx) {
        // 下一个groupsize整数倍的点
        nextGroupsizeIdx = (curRowIdx / groupSize + 1) * groupSize;

        if (nextGroupsizeIdx > endRowIdx) {
            // 只计算curRowIdx到endRowIdx之间的
            MaskOperation(ridx, curRowIdx, endRowIdx);
            break; // 处理完剩余部分后退出循环
        } else {
            // 处理从curRowIdx到nextGroupsizeIdx - 1之间的部分
            MaskOperation(ridx, curRowIdx, nextGroupsizeIdx - 1);
        }

        curRowIdx = nextGroupsizeIdx;
    }

    AscendC::SetFlag<AscendC::HardEvent::S_V>(1);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(1);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::SoftmaxCompute(uint32_t ridx)
{
    // 设置softmax的shape
    SoftMaxShapeInfo srcShape = { basicRowLenCal, alignedColLen, basicRowLenCal, static_cast<uint32_t>(curSeqlen)};
    // 调用softmax接口
    AscendC::Muls<float>(
        softmaxInputbufTensor,
        softmaxInputbufTensor,
        scaleValue,
        softmaxTileLength
    );
    PipeBarrier<PIPE_V>();

    // Mask处理
    if (attenMaskFlag == 1) {
        MaskApply(ridx);
    }

    SoftMax<float>(softmaxOut32bufTensor, softmaxInputbufTensor, sharedTmpBuffer, softmaxTilingData, srcShape);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0); // softmax搬入等待softmax计算结束
    PipeBarrier<PIPE_V>();
    if (std::is_same<OUT_T, __bf16>::value) {
        AscendC::Cast(softmaxOut16bufTensor, softmaxOut32bufTensor, AscendC::RoundMode::CAST_RINT,
                      softmaxOut32bufTensor.GetSize());
    } else {
        AscendC::Cast(softmaxOut16bufTensor, softmaxOut32bufTensor, AscendC::RoundMode::CAST_NONE,
                      softmaxOut32bufTensor.GetSize());
    }
    PipeBarrier<PIPE_V>();
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessImportanceScore(uint32_t processIdx)
{
    uint32_t perBaseGroup = maxRowCountPerLoop / gHeadNums;
    uint32_t perCoreGroup = impSocreCoreRowCount / gHeadNums;
    uint32_t loopCnt = perBaseGroup > 0 ? CeilDiv(perCoreGroup, perBaseGroup) : 0;
    uint32_t tailGroupCnt = perCoreGroup % perBaseGroup;

    // S1方向切分
    for (uint32_t taskId = 0; taskId < loopCnt; taskId++) {
        uint32_t startRowIdx = taskId * perBaseGroup * gHeadNums;
        uint32_t curGroupCnt = (taskId == loopCnt - 1) ? (perCoreGroup - taskId * perBaseGroup) : perBaseGroup;
        uint32_t endRowIdx = startRowIdx + curGroupCnt * gHeadNums;
        ProcessImpScoreS2Loop(loopCnt, processIdx, taskId, startRowIdx, endRowIdx);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ClearImpScoreWorspace()
{
    if constexpr(NCAIType::IMP_SCORE_OPT) {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::Duplicate(pslcTensor, 0.0f, impScoreResultEleNum_);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::DataCopy(impScoreResultGm_[cubeBlockIdx * impScoreResultEleNum_], pslcTensor,
                          impScoreResultEleNum_);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::GenerateImpScoreParam()
{
    uint32_t slcNum = selectSize / compStrideD;
    uint32_t cmpNum = compSizeL / compStrideD;
    uint32_t cmpNumPad = AlignUp(cmpNum, ALIGNED_8);
    uint32_t windowSize = slcNum + cmpNum - 1;
    uint32_t paramCol = IMPORTANCE_SCORE_PARAM_ROW;
    bool hasPrefix = compSizeL != compStrideD;
    uint32_t paramRow = IMPORTANCE_SCORE_PARAM_ROW / slcNum + hasPrefix;
    auto slideWindowTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>(0);
    auto patternTensor = buf.GetBuffer<BufferType::ASCEND_UB, uint32_t>(NUM64 * sizeof(float));
    auto impScoreParamTensor = buf.GetBuffer<BufferType::ASCEND_UB, float>((NUM64 + ALIGNED_8) * sizeof(float));
    AscendC::Duplicate(slideWindowTensor, 0.0f, NUM64);
    AscendC::Duplicate(impScoreParamTensor, 0.0f, paramRow * paramCol);
    AscendC::PipeBarrier<PIPE_V>();
    uint32_t row = 0;
    uint64_t mask[] = {(1UL << (cmpNum)) - 1};
    uint64_t rsvdCnt = 0;
    for (uint32_t i = 0; i < slcNum; ++i) {
        AscendC::Adds(slideWindowTensor[ALIGNED_8], slideWindowTensor[ALIGNED_8], 1.0f, mask, 1,
                      {1, 1, ALIGNED_8, ALIGNED_8});
        AscendC::PipeBarrier<PIPE_V>();
        mask[0] <<= 1;
    }
    if (hasPrefix) {
        patternTensor.SetValue(0, mask[0]);
        AscendC::GatherMask(impScoreParamTensor, slideWindowTensor[ALIGNED_8], patternTensor, true, NUM64,
                             {1, 1, ALIGNED_8, ALIGNED_8}, rsvdCnt);
        AscendC::PipeBarrier<PIPE_V>();
        ++row;
    }
    for (uint32_t dstBaseOffset = paramCol, dstRelativeOffset = 0, dstActualOffset = 0, dstActualOffsetPad = 0;
         row < paramRow; ++row) {
        dstActualOffset = dstBaseOffset + dstRelativeOffset;
        uint32_t remain = dstActualOffset % ALIGNED_8;
        if (remain == 0) {
            AscendC::Copy(impScoreParamTensor[dstActualOffset], slideWindowTensor[ALIGNED_8], windowSize, 1,
                          {1, 1, ALIGNED_8, ALIGNED_8});
        } else {
            dstActualOffsetPad = dstActualOffset - remain;
            patternTensor.SetValue(0, ((1UL << (windowSize + remain)) - 1) << (ALIGNED_8 - remain));
            AscendC::GatherMask(impScoreParamTensor[dstActualOffsetPad], slideWindowTensor, patternTensor, true, NUM64,
                                {1, 1, ALIGNED_8, ALIGNED_8}, rsvdCnt);
        }
        AscendC::PipeBarrier<PIPE_V>();
        dstBaseOffset += paramCol;
        dstRelativeOffset += slcNum;
        dstActualOffset += slcNum;
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    AscendC::DataCopy(impScoreParamGm_, impScoreParamTensor, paramCol * paramRow);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessImportanceScoreForCube(uint32_t processIdx)
{
    uint32_t srcBaseOffset = vecBlockIdx / 2 * impScoreResultEleNum_ + impScoreParamHasPrefix_;
    uint32_t srcOffset;
    uint32_t dstBaseOffset = vecBlockIdx / 2 * workSpaceElemNum / groupSize;
    uint32_t dstOffset;
    if (vecBlockIdx % 2 != 0) {
        srcBaseOffset += rowNumVec0 * impScoreResultColPad_;
        dstBaseOffset += rowNumVec0 / groupSize * maxOutS2;
    }
    uint16_t reduceRowLoopCount = impSocreCoreRowCount / groupSize;
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(IMPORTANCE_SCORE_EVENT_ID);
    for (uint32_t rowLoopIdx = 0; rowLoopIdx < reduceRowLoopCount; rowLoopIdx++) {
        srcOffset = srcBaseOffset;
        dstOffset = dstBaseOffset;
        for (uint32_t colLoopIdx = 0, reduceCol = impScoreReduceCol_, reduceColPad = impScoreReduceColPad_;
             colLoopIdx < impScoreReduceRound_; colLoopIdx++) {
            if (colLoopIdx + 1 == impScoreReduceRound_) {
                reduceCol = impScoreReduceTailCol_;
                reduceColPad = impScoreReduceTailColPad_;
            }

            ComputeImportanceScoreReduce(colLoopIdx, reduceCol, reduceColPad, srcOffset);

            DataCopyParams copyOutParam {
                1,                                      // blockCount
                (uint16_t)(reduceCol * sizeof(float)),  // blockLen
                0,                                      // srcStride
                0                                       // dstStride
            };
            AscendC::DataCopyPad(topKInGm[dstOffset], impScoreReduceResult_, copyOutParam);

            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(IMPORTANCE_SCORE_EVENT_ID);

            srcOffset += impScoreReduceColPad_;
            dstOffset += impScoreReduceColPad_;
        }
        srcBaseOffset += groupSize * impScoreResultColPad_;
        dstBaseOffset += maxOutS2;
    }
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(IMPORTANCE_SCORE_EVENT_ID);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ComputeImportanceScoreReduce(
    uint32_t colLoopIdx, uint32_t reduceCol, uint32_t reduceColPad, uint32_t srcOffset)
{
    DataCopyParams copyInParam {
        (uint16_t)groupSize,
        (uint16_t)(reduceColPad / ALIGNED_8),
        (uint16_t)((impScoreResultColPad_ - reduceColPad) / ALIGNED_8),
        0
    };
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::DataCopy(pslcTensor, impScoreResultGm_[srcOffset], copyInParam);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(IMPORTANCE_SCORE_EVENT_ID);
    uint32_t shape[] = { groupSize, reduceColPad };
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
        impScoreReduceResult_, pslcTensor, impScoreReduceTmpBuffer_, shape, true);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(IMPORTANCE_SCORE_EVENT_ID);

    AscendC::SetFlag<AscendC::HardEvent::V_S>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(IMPORTANCE_SCORE_EVENT_ID);
    if (colLoopIdx == 0) {
        impScoreReduceResultInt_.SetValue(0, FLOAT_TYPE_MASK);
    }
    if (colLoopIdx + 1 == impScoreReduceRound_) {
        if (reduceCol > 1) {
            impScoreReduceResultInt_.SetValue(reduceCol - 2, FLOAT_TYPE_MASK);
        }
        impScoreReduceResultInt_.SetValue(reduceCol - 1, FLOAT_TYPE_MASK);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(IMPORTANCE_SCORE_EVENT_ID);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(IMPORTANCE_SCORE_EVENT_ID);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessImpScoreS2Loop(
    uint32_t loopCnt, uint32_t processIdx, uint32_t taskId, uint32_t startRowIdx, uint32_t endRowIdx)
{
    // S2方向上的切分
    uint32_t loopOutS2 =  CeilDiv(outS2, maxBaseJ);
    uint32_t vecOutSplitOffset = (vecBlockIdx % 2 == 0) ? 0 : workSpaceElemNum / 2;
    uint32_t impScoreOutputSubCoreOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 / gHeadNums * maxOutS2;
    impScoreInputGmOffset = vecBlockIdx / 2 * workSpaceElemNum + vecOutSplitOffset + startRowIdx * curSeqlen;
    impScoreOutputGmOffset = vecBlockIdx / 2 * workSpaceElemNum / gHeadNums + impScoreOutputSubCoreOffset +
                             startRowIdx * outS2 / gHeadNums;

    for (uint32_t baseS2Id = 0; baseS2Id < loopOutS2; baseS2Id++) {
        uint32_t startJ = baseS2Id * maxBaseJ;
        ComputeImpScoreValue(loopCnt, loopOutS2, taskId, baseS2Id, startRowIdx, endRowIdx, startJ);
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ComputeImpScoreValue(
    uint32_t loopCnt, uint32_t loopOutS2, uint32_t taskId, uint32_t baseS2Id, uint32_t startRowIdx, uint32_t endRowIdx,
    uint32_t startJ)
{
    int32_t offsetS2Split = (startJ + 1) * strideOut - maxM - maxN;
    offsetS2Split = offsetS2Split < 0 ? 0 : offsetS2Split;
    int64_t inputOffsetGm = impScoreInputGmOffset + offsetS2Split;
    uint16_t rowCount = endRowIdx - startRowIdx;
    uint32_t groupNum = rowCount / gHeadNums;
    uint32_t blockLength = maxBaseS2 * 4;

    if (startJ == outS2 - 1) {
        if constexpr(!NCAIType::IMP_SCORE_OPT) {
        if (taskId == loopCnt - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);
        LocalTensor<int32_t> tmpInfLocal = pslcTensor.template ReinterpretCast<int32_t>();
        Duplicate(tmpInfLocal[0], FLOAT_TYPE_MASK, 128);
        PipeBarrier<PIPE_V>();
        int64_t outputOffsetTail = impScoreOutputGmOffset + startJ;
        blockLength = 1 * 4;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
        ImpScoreDataCopyOut(taskId, baseS2Id, outputOffsetTail, groupNum, blockLength, 0);
        return;
    }
    if ((offsetS2Split + maxBaseS2) > curSeqlen) {
        blockLength = (curSeqlen - offsetS2Split) * 4;
    }

    ImpScoreDataCopyIn(inputOffsetGm, rowCount, blockLength);

    if constexpr(!NCAIType::IMP_SCORE_OPT) {
    if ((taskId == loopCnt - 1) && (baseS2Id == loopOutS2 - 1)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0); // softmax搬出等待score搬入结束
        }
    }
    uint32_t baseS2Align = ((blockLength + 63) / 64 * 64) / 4;
    uint32_t rowAlign = (rowCount + 15) / 16 * 16;
    ImpScoreTransposeTensor(rowAlign, baseS2Align);

    uint32_t actualStartJ = startJ + 1;
    uint32_t actualEndJ = (actualStartJ + maxBaseJ) > outS2 ? outS2 + 1 : actualStartJ + maxBaseJ;
    uint32_t outNewS2 = actualEndJ - actualStartJ;
    uint32_t outNewS2Align = AlignUp(outNewS2, 16);
    ImpScoreAccumulation(startJ, rowCount, offsetS2Split);
    ImpScoreReduceSum(startJ, rowCount, outNewS2);

    int64_t outputOffsetGm = impScoreOutputGmOffset + startJ;
    blockLength = outNewS2 * 4;
    uint32_t srcStride = outNewS2Align - outNewS2 >= 8 ? 1 : 0;

    ImpScoreDataCopyOut(taskId, baseS2Id, outputOffsetGm, groupNum, blockLength, srcStride);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreTransposeTensor(
    uint32_t height, uint32_t width)
{
    ConfusionTransposeTiling tiling;
    uint32_t blockSizeT = 8;
    uint32_t highBlock = height / 16;
    uint32_t stride = height * blockSizeT * 4 / 32;
    uint32_t repeat = width / blockSizeT;

    tiling.param0 = blockSizeT;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;

    ConfusionTranspose(pslcCalcTensor, pslcTensor, pslcSharedBuffer, TransposeType::TRANSPOSE_ND2ND_ONLY, tiling);
    PipeBarrier<PIPE_V>();
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreAccumulation(
    uint32_t startJ, uint32_t rowCount, uint32_t offsetS2Split)
{
    uint32_t actualStartJ = startJ + 1;
    uint32_t addLoop = maxM + maxN + 1;
    uint32_t times = 1;
    uint32_t rowAlign = (rowCount + 15) / 16 * 16;
    uint32_t actualEndJ = (actualStartJ + maxBaseJ) > outS2 ? outS2 + 1 : actualStartJ + maxBaseJ;
    uint32_t outNewS2 = actualEndJ - actualStartJ;

    Duplicate(pslcTensor, (float)0, baseBlockSize);
    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < addLoop; i++) {
        if (i < maxN) {
            times = i + 1;
        }
        if (i >= maxN && i <= maxM) {
            times = maxN + 1;
        }
        if (i > maxM) {
            times = addLoop - i;
        }
        int64_t pIdxOffset = i + offsetS2Split;
        for (uint32_t j=actualStartJ; j<actualEndJ; j++) {
            uint32_t pslcOffset = (j - actualStartJ) * rowAlign;

            int64_t pIdx = strideOut * j - pIdxOffset;

            uint32_t pcmpOffsetLocal = pIdx * rowAlign;
            if (pIdx < 0) {
                continue;
            }
            if ((pIdx + offsetS2Split) >= curSeqlen) {
                break;
            }
            if (times > 1) {
                float mulTimes = times * (float)1.0;
                Muls(pslcTmpTensor, pslcCalcTensor[pcmpOffsetLocal], mulTimes, rowCount);
                PipeBarrier<PIPE_V>();
                Add(pslcTensor[pslcOffset], pslcTensor[pslcOffset], pslcTmpTensor, rowCount);
                PipeBarrier<PIPE_V>();
            } else {
                Add(pslcTensor[pslcOffset], pslcTensor[pslcOffset], pslcCalcTensor[pcmpOffsetLocal], rowCount);
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreReduceSum(
    uint32_t startJ, uint32_t rowCount, uint32_t outNewS2)
{
    uint32_t rowAlign = (rowCount + 15) / 16 * 16;
    uint32_t groupNum = rowCount / gHeadNums;
    LocalTensor<int32_t> tmpLocal = pslcTensor.template ReinterpretCast<int32_t>();
    if (startJ == 0 || outNewS2 == 1) {
        Duplicate(tmpLocal[0], FLOAT_TYPE_MASK, rowCount);
    }
    if ((outNewS2+startJ) == (outS2 - 1)) {
        Duplicate(tmpLocal[(outNewS2-1)*rowAlign], FLOAT_TYPE_MASK, rowAlign);
    }
    if (outNewS2 > 1 && (outNewS2+startJ) >= outS2) {
        Duplicate(tmpLocal[ (outNewS2-2)*rowAlign], FLOAT_TYPE_MASK, rowAlign*2);
    }
    PipeBarrier<PIPE_V>();

    uint32_t outNewS2Align = AlignUp(outNewS2, 16);
    ImpScoreTransposeTensor(outNewS2Align, rowAlign);

    Duplicate(pslcTensor, (float)0.0, baseBlockSize);
    PipeBarrier<PIPE_V>();
    for (uint32_t i = 0; i < groupNum; i++) {
        for (uint32_t j = 0; j < gHeadNums; j++) {
            Add(pslcTensor[i*outNewS2Align], pslcTensor[i*outNewS2Align],
                pslcCalcTensor[(i * gHeadNums + j) *outNewS2Align], outNewS2Align);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreDataCopyIn(
    int64_t offset, uint16_t blockCount, uint32_t blockLength)
{
    uint32_t srcStride = curSeqlen * 4 - blockLength;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    uint32_t dstStride = 0;
    if ((blockLength % 64) != 0) {
        padParams.isPad = true;
        uint32_t paddingNum = (64 - (blockLength % 64)) / 4;
        if (paddingNum * 4 > 32) { // padding最大不能超过32字节
            paddingNum = paddingNum - 8;
            dstStride = 1;
        }
        padParams.rightPadding = paddingNum;
    }
    DataCopyExtParams copyParams{blockCount, blockLength, srcStride, dstStride, 0};

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址

    DataCopyPad(pslcTensor, scoreInGm[offset], copyParams, padParams);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1); // score计算等score搬入结束
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1); // score计算等score搬入结束
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ImpScoreDataCopyOut(
    uint32_t taskId, uint32_t baseS2Id, int64_t offset, uint16_t blockCount, uint32_t blockLength, uint32_t srcStride)
{
    if (taskId == 0 && baseS2Id == 0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(1); // score搬出等待score计算结束
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(1); // score搬出等待score计算结束

    uint32_t dstStride = outS2 * 4 - blockLength;

    DataCopyExtParams copyParams{blockCount, blockLength, srcStride, dstStride, 0};
    DataCopyPad(topKInGm[offset], pslcTensor, copyParams);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(1); // score搬入等score搬出结束，由于ub上使用了同一块地址
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::ProcessTopK()
{
    DataCopyParams splitCopyintopkParams;
    DataCopyParams splitCopyouttopkParams;

    alignedoutS2 = AlignUp(outS2, 8);
    alignedTok32 = AlignUp(outS2, 32);
    alignedTokloop = AlignUp(outS2, 32) - alignedoutS2;
    topkRightPadding = alignedoutS2 - outS2;
    uint32_t impScoreOutputSubCoreOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 / gHeadNums * maxOutS2;
    uint32_t coreRowOffset = (vecBlockIdx % 2 == 0) ? 0 : rowNumVec0 / gHeadNums;
    uint32_t perCoreGroup = impSocreCoreRowCount / gHeadNums;
    splitCopyintopkParams = {1, (uint16_t)(outS2 * sizeof(float)), 0, 0};

    splitCopyouttopkParams = {1, (uint16_t)(selectNum * sizeof(int32_t)), 0, 0};
    // S1方向切分
    for (uint32_t taskId = 0; taskId < perCoreGroup; taskId++) {
        topkInputGmOffset = vecBlockIdx / 2 * workSpaceElemNum / gHeadNums + impScoreOutputSubCoreOffset +
        taskId * outS2;
        // 推算, taskId，qSeqLenCurProcess中第几个qSeqLen，第几个kvHeadSplit, 第几个kvHead
        uint32_t actualRowIdx = taskId + coreRowOffset;
        uint32_t kvHeadIdxInSplit = actualRowIdx / qSeqLenCurProcess;
        uint32_t qIdxInSplit = actualRowIdx % qSeqLenCurProcess;

        topkOutputGmOffset = qSeqLenCumSum * kvHeadNum * selectNum +
                                qSeqLenOffset * kvHeadNum * selectNum + qIdxInSplit * kvHeadNum * selectNum +
                                kvHeadOffset * selectNum + kvHeadIdxInSplit * selectNum;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
        TopkCopyIn(splitCopyintopkParams);
        if constexpr(!NCAIType::IMP_SCORE_OPT) {
        if (taskId == perCoreGroup - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(1); // score搬出等待topk搬入结束
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(2); // topk计算等待topk搬入结束
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(2); // topk计算等待topk搬入结束
        AscendC::SetFlag<AscendC::HardEvent::V_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(0);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(0);
        TopkCompute();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(2); // topk搬入等待topk计算结束
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(2);
        TopkCopyOut(splitCopyouttopkParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(5); // topk搬入等待topk搬出结束
    }
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::TopkCopyIn(DataCopyParams &splitCopyintopkParams)
{
    PipeBarrier<PIPE_V>();
    AscendC::Duplicate<float>(topkInputbufTensor, static_cast<float>(0), BASE_TOPK_ELEM_NUM_OFFSET);
    PipeBarrier<PIPE_V>();
    DataCopyPadParams padParams{true, 0, topkRightPadding, 0};
    DataCopyPad(topkInputbufTensor, topKInGm[topkInputGmOffset], splitCopyintopkParams, padParams);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::TopkCopyOut(DataCopyParams &splitCopyouttopkParams)
{
    DataCopyPad(topkOutGm[topkOutputGmOffset], topkoutindexLocal[0], splitCopyouttopkParams);
}

template <typename NCAIType>
__aicore__ inline void NsaCompressAttentionInferAiv<NCAIType>::TopkCompute()
{
    TopKInfo topkinfo;
    topkinfo = {(int32_t)(1), (int32_t)(alignedTok32), (int32_t)(outS2)};
    for (int i = 0; i < 8; i++) {
        arithbuffer.SetValue(i, i);
    }
    AscendC::SetFlag<AscendC::HardEvent::S_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(0);
    uint32_t topindexloop = alignedTok32 / 8;
    for (int i = 1; i < topindexloop; i++) {
        Adds(arithbuffer[i * 8], arithbuffer, (int32_t)i * 8, 8);
        PipeBarrier<PIPE_V>();
    }
    AscendC::TopK<float, true, false, false, AscendC::TopKMode::TOPK_NORMAL>(topkoutvalueLocal,
        topkoutindexLocal, topkInputbufTensor, arithbuffer, topksrcLocalFinish, topksharedTmpBuffer, selectNum,
        this->topkTilingData, topkinfo, true);
    PipeBarrier<PIPE_V>();
}

#endif

template <typename NCAIType>
class NsaCompressAttentionInfer {
public:

    __aicore__ inline NsaCompressAttentionInfer(){};

    __aicore__ inline void Run(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockTable, GM_ADDR actualQSeqLen,
        GM_ADDR actualKvSeqLen, GM_ADDR actualSelKvSeqLen, GM_ADDR output,
        GM_ADDR topkIndicesOut, GM_ADDR workspace, const NsaCompressAttentionInferTilingData *__restrict tilingData)
    {
#ifdef __DAV_C220_CUBE__
            NsaCompressAttentionInferAic<NCAIType> opAic;
            opAic.Init(query, key, value, blockTable, actualQSeqLen, actualKvSeqLen, output,
                topkIndicesOut, workspace, tilingData);
            opAic.Process();

#elif __DAV_C220_VEC__
            NsaCompressAttentionInferAiv<NCAIType> opAiv;
            opAiv.Init(query, key, value, blockTable, actualQSeqLen, actualKvSeqLen, actualSelKvSeqLen, output,
                topkIndicesOut, workspace, tilingData);
            opAiv.Process();
#endif
    }
};
} // namespace NSA_COMPRESS_ATTENTION_INFER
#endif  // NSA_COMPRESS_ATTENTION_INFER_H
