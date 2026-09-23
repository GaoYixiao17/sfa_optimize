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
 * \file sparse_flash_attention_antiquant_service_cube_gqa.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_CUBE_GQA_MSD_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_CUBE_GQA_MSD_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_antiquant_common.h"

#define SALSC_S2BASEIZE 1024
#define SALSC_S2MERGESIZE 384
#define SALSC_S2BASEIZE_1 1023

template <typename SFAAT> class SFAAMatmulServiceGqaMsd {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename SFAAT::queryType;
    using KV_T = typename SFAAT::kvType;
    using KV_ORIGIN_T = typename SFAAT::queryType;
    using OUT_T = typename SFAAT::outputType;
    using MM_OUT_T = half;
    using L0C_T = int32_t;

    __aicore__ inline SFAAMatmulServiceGqaMsd(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> keyGm,
                                               GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                               GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(const GlobalTensor<KV_T>& keyMergeGm, const GlobalTensor<KV_T>& valueMergeGm,
                                                 const GlobalTensor<KV_T>& queryPreProcessResGm, GlobalTensor<int32_t> blockTableGm,
                                                 GlobalTensor<int32_t> topKGm, uint32_t blockSize, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CalcTopKBlockInfo(const RunInfo &info, uint32_t &curTopKIdx,
                                             uint64_t &curOffsetInSparseBlock, uint32_t curSeqIdx,
                                             uint32_t &copyRowCnt, uint64_t &idInTopK);
    __aicore__ inline void ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = SFAAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAAT::templateMode;
    static constexpr bool FLASH_DECODE = SFAAT::flashDecode;
    static constexpr SFAA_LAYOUT LAYOUT_T = SFAAT::layout;
    static constexpr SFAA_LAYOUT KV_LAYOUT_T = SFAAT::kvLayout;

    static constexpr float quantScaleC1S1 = 1.0 / (1024);
    static constexpr float quantScaleC1S2 = 1.0 / (1024 * 254);
    static constexpr float quantScaleC2O1 = 1.0 / (1024 * 127);
    static constexpr float quantScaleC2O2 = 1.0 / (1024 * 254 * 127);
    static constexpr uint32_t msdIterNum = 2;
    static constexpr uint32_t P_LOAD_TO_L1_ROW_NUM = 128 / sizeof(KV_T);
    static constexpr uint32_t KV_LOAD_TO_L1_ROW_NUM = 512 / sizeof(KV_T);
    static constexpr uint64_t DATABLOCK_BYTES = 32UL;
    // L1轴切分大小
    static constexpr uint32_t S2_SPLIT_SIZE = 512;     // S2方向切分，对于mm1是N轴，对于mm2是K轴

    // L0轴切分大小
    static constexpr uint32_t M_BASE_SIZE = 128;     // m方向基本块大小
    static constexpr uint32_t K_BASE_SIZE = 128;     // k方向基本块大小
    static constexpr uint32_t N_BASE_SIZE = 256;     // n方向基本块大小
    static constexpr uint32_t L1_NZ_BLOCK_SIZE_MM1 = 256;
    static constexpr uint32_t L1_NZ_BLOCK_SIZE_MM1_SHIFT = 8;
    static constexpr uint32_t L1_NZ_BLOCK_SIZE_MM1_MASK = L1_NZ_BLOCK_SIZE_MM1 - 1;
    static constexpr uint32_t L1_NZ_BLOCK_SIZE_MM2 = 32;
    static constexpr uint32_t L1_NZ_BLOCK_SIZE_MM2_SHIFT = 5;
    static constexpr uint32_t L1_NZ_BLOCK_SIZE_MM2_MASK = L1_NZ_BLOCK_SIZE_MM2 - 1;
    static constexpr uint32_t L1_NZ_KV_BLOCK_ELEMENT = 32;

    static constexpr uint32_t L1Q_BLOCK_SIZE= 64 * 1024; // 64K
    static constexpr uint32_t L1KP_BLOCK_SIZE = 64 * 1024; // 64k
    static constexpr uint32_t L1V_BLOCK_SIZE = 64 * 1024; // 64k

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);
    static constexpr uint32_t L0AB_BLOCK_OFFSET = L0A_PP_SIZE / sizeof(KV_T);
    static constexpr uint32_t L0C_BLOCK_OFFSET = L0C_PP_SIZE / sizeof(MM_OUT_T);

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
    static constexpr uint32_t L1Q_EVENT0 = EVENT_ID0;
    static constexpr uint32_t L1KP_EVENT0 = EVENT_ID1;
    static constexpr uint32_t L1KP_EVENT1 = EVENT_ID2;
    static constexpr uint32_t L1KP_EVENT2 = EVENT_ID3;
    static constexpr uint32_t L1V_EVENT0 = EVENT_ID4;
    static constexpr uint32_t L1V_EVENT1 = EVENT_ID5;
    static constexpr uint32_t L1V_EVENT2 = EVENT_ID6;
    static constexpr uint32_t L1V_EVENT3 = EVENT_ID7;
    static constexpr uint32_t L1V_BUFFER_NUM = 4;
    static constexpr uint32_t L1KP_BUFFER_NUM = 3;

    // m <> mte1 EventID
    static constexpr uint32_t L0A_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0A_EVENT1 = EVENT_ID4;
    static constexpr uint32_t L0B_EVENT0 = EVENT_ID5;
    static constexpr uint32_t L0B_EVENT1 = EVENT_ID6;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32B / sizeof(KV_T);

    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t headDimAlign = 0;
    ConstInfo constInfo{};

    // L1分成3块buf, 用于记录
    uint32_t qL1BufIter = 0;
    uint32_t kpL1BufIter = -1;
    uint32_t vL1BufIter = -1;
    uint32_t aL0BufIter = 0;
    uint32_t bL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_T> keyMergeGm_;
    GlobalTensor<KV_T> queryPreProcessResGm_;

    // mm2
    GlobalTensor<KV_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<KV_T> valueMergeGm_;

    // block_table
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    TBuf<TPosition::A1> tmpBufL1Q;
    LocalTensor<KV_T> qL1Buffers;
    TBuf<TPosition::A1> tmpBufL1KP;
    LocalTensor<KV_T> kpL1Buffers;
    TBuf<TPosition::A1> tmpBufL1V;
    LocalTensor<KV_T> vL1Buffers;

    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_T> aL0TensorPingPong;
    // L0B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_T> bL0TensorPingPong;
    // L0C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<L0C_T> cL0TensorPingPong;

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_T> &l1Tensor, GlobalTensor<KV_T> &gmSrcTensor,
                                      uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const RunInfo &info,
                                          uint32_t mSizeAct, uint32_t headSize);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor, const RunInfo &info, uint32_t mCopyIdx,
                                          uint32_t mCopyRowCount, uint32_t mActCopyRowCount,
                                          uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount);
    __aicore__ inline void CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor, const RunInfo &info,
        uint32_t kCopyIdx, uint32_t kActCopyRowCountAlign, uint32_t kActCopyRowCount);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor, LocalTensor<KV_T> &aL1Tensor,
                                        uint32_t idx, uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_T> &bL0Tensor, LocalTensor<KV_T> &bL1Tensor,
                                        uint32_t idx, uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
    __aicore__ inline void CopyInMmBToL1(LocalTensor<KV_T> &bl1Tensor, GlobalTensor<KV_T> &gmSrcTensor,
                                          uint32_t subNSizeAct, uint32_t nOffset, const RunInfo &info);
    __aicore__ inline void CopyInMm1BToL1VecMerge(LocalTensor<KV_T>& bL1Tensor, const RunInfo &info,
        uint32_t aicNZBlockBum, uint32_t aicNzBlockTail, uint32_t VecAccessSize, uint32_t alreadyVecAccessSize,
        uint32_t nActCopyRowCountAlign, uint32_t CubeAccessSize);
    __aicore__ inline void CopyInMm2BToL1VecMerge(LocalTensor<KV_T>& bL1Tensor, const RunInfo &info,
        uint32_t aicNZBlockBum, uint32_t aicNzBlockTail, uint32_t VecAccessSize, uint32_t alreadyVecAccessSize,
        uint32_t kActCopyRowCountAlign, uint32_t CubeAccessSize);
    __aicore__ inline void CopyInMm1BToL1CubeDisepr(LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t nCopyIdx,
        uint32_t nCopyRowCount, uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign);
    __aicore__ inline void CopyInMm2BToL1CubeDisepr(LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t nCopyIdx,
        uint32_t nCopyRowCount, uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign);
    __aicore__ inline void CopyInMm1BToL1(LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t nCopyIdx,
        uint32_t nCopyRowCount, uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign);
    __aicore__ inline void CopyInMm1BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset,
        uint32_t copyTotalRowCnt, uint32_t copyStartRowCnt, uint32_t nActCopyRowCount, const RunInfo &info);
    __aicore__ inline void CopyInMm2BToL1ForPA(LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset,
        uint32_t copyStartRowCnt, uint32_t kActCopyRowCount);
    __aicore__ inline void LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize);
};

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
    headDimAlign = SFAAAlign(constInfo.headDim, ConstInfo::BUFFER_SIZE_BYTE_32B);
}

template <typename SFAAT>
__aicore__ inline void
SFAAMatmulServiceGqaMsd<SFAAT>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> keyGm,
                                              GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->keyGm = keyGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename SFAAT>
__aicore__ inline void
SFAAMatmulServiceGqaMsd<SFAAT>::InitMm2GlobalTensor(GlobalTensor<KV_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                              GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->valueGm = valueGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SFAAT>
__aicore__ inline void
SFAAMatmulServiceGqaMsd<SFAAT>::InitPageAttentionInfo(const GlobalTensor<KV_T>& keyMergeGm, const GlobalTensor<KV_T>& valueMergeGm,
                                                      const GlobalTensor<KV_T>& queryPreProcessResGm, GlobalTensor<int32_t> blockTableGm,
                                                      GlobalTensor<int32_t> topKGm, uint32_t blockSize, uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->topKGm = topKGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
    this->keyMergeGm_ = keyMergeGm;
    this->valueMergeGm_ = valueMergeGm;
    this->queryPreProcessResGm_ = queryPreProcessResGm;
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(tmpBufL1Q, L1Q_BLOCK_SIZE); // (64K) * 1
    qL1Buffers = tmpBufL1Q.Get<KV_T>();
    pipe->InitBuffer(tmpBufL1KP, L1KP_BLOCK_SIZE * L1KP_BUFFER_NUM); // 64K * 3
    kpL1Buffers = tmpBufL1KP.Get<KV_T>();
    pipe->InitBuffer(tmpBufL1V, L1V_BLOCK_SIZE * L1V_BUFFER_NUM); // 64K * 4
    vL1Buffers = tmpBufL1V.Get<KV_T>();

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<L0C_T>();
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT3);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0B_EVENT1);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT3);
    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT1);
    WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0B_EVENT1);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyGmToL1(LocalTensor<KV_T> &l1Tensor,
                                                            GlobalTensor<KV_T> &gmSrcTensor, uint32_t srcN,
                                                            uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = SFAAAlign(srcN, 16U); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm1AToL1(LocalTensor<KV_T> &l1Tensor, const RunInfo &info,
                                                                uint32_t mSizeAct, uint32_t headSize)
{
    auto srcGm = queryPreProcessResGm_[(info.bN2Idx % constInfo.preLoadNum) * constInfo.bmm2ResUbSize * 2];
    // 不切G, mSizeAct必然是constInfo.gSize的整数倍
    uint32_t s1Size = mSizeAct / constInfo.gSize;
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = s1Size;
    nd2nzPara.nValue = constInfo.gSize; // 行数
    nd2nzPara.dValue = headSize;
    nd2nzPara.srcDValue = headSize;
    nd2nzPara.dstNzC0Stride = (mSizeAct + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = constInfo.qHeadNum * constInfo.headDim; // 这里小于65536
    nd2nzPara.dstNzMatrixStride = constInfo.gSize * BLOCK_ELEMENT_NUM;
    DataCopy(l1Tensor, srcGm, nd2nzPara);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::LoadDataMm1A(LocalTensor<KV_T> &aL0Tensor,
                                                              LocalTensor<KV_T> &aL1Tensor, uint32_t idx,
                                                              uint32_t kSplitSize, uint32_t mSize, uint32_t kSize)
{
    LocalTensor<KV_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16;         // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = mSize; // M
    loadData3DParams.kExtension = kSize; // K
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    loadData3DParams.channelSize = kSize; // Cin=K
    LoadData<KV_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::LoadDataMm1B(LocalTensor<KV_T> &l0Tensor,
                                                              LocalTensor<KV_T> &l1Tensor, uint32_t idx,
                                                              uint32_t kSplitSize, uint32_t kSize, uint32_t nSize)
{
    // N 方向全载
    LocalTensor<KV_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm2AToL1(LocalTensor<KV_T> &aL1Tensor, const RunInfo &info,
                                                                uint32_t mSeqIdx, uint32_t subMSizeAct,
                                                                uint32_t nSize, uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMmBToL1(LocalTensor<KV_T> &bl1Tensor,
                                                                   GlobalTensor<KV_T> &gmSrcTensor,
                                                                   uint32_t subNSizeAct, uint32_t nOffset,
                                                                   const RunInfo &info)
{
    auto srcGm = gmSrcTensor[(info.loop % 4) * SALSC_S2BASEIZE * constInfo.headDim + nOffset];

    CopyGmToL1(bl1Tensor, srcGm, subNSizeAct, constInfo.headDim, constInfo.headDim);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm2AToL1(LocalTensor<KV_T>& aL1Tensor,
    const RunInfo &info, uint32_t mCopyIdx, uint32_t mCopyRowCount, uint32_t mActCopyRowCount,
    uint32_t kCopyIdx, uint32_t kCopyRowCount, uint32_t kActCopyRowCount)
{
    uint32_t mmRowCount = mActCopyRowCount;
    uint32_t copyStrideL1 = 16 * kActCopyRowCount;
    uint32_t copyStrideGm = 16 * info.actualSingleProcessSInnerSizeAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++){
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = mmRowCount - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数  16
        }
        mm1Nd2NzParamsForA.dValue = kActCopyRowCount; // 单个ND矩阵的列数, 单位为元素个数
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.srcDValue = info.actualSingleProcessSInnerSizeAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移
        DataCopy(aL1Tensor[i * copyStrideL1],
                 vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize * 2 +
                 (mCopyIdx * mCopyRowCount) * info.actualSingleProcessSInnerSizeAlign +
                 kCopyIdx * kCopyRowCount + i * copyStrideGm],
                 mm1Nd2NzParamsForA);
    }
}

// nCopyRowCount需要32元素对齐
// 先不化简吧，最后我们统一化简降低scaler
template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm1BToL1CubeDisepr(
    LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t nCopyIdx, uint32_t nCopyRowCount,
    uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign)
{
    if (nActCopyRowCount == 0) {
        return;
    }
    uint32_t copySparseTotalBlockCount = (nActCopyRowCount + constInfo.sparseBlockSize - 1) / constInfo.sparseBlockSize;
    uint32_t copySparseBlockSizeTail = nActCopyRowCount - ((copySparseTotalBlockCount - 1) * constInfo.sparseBlockSize);
    uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
    uint32_t copyFinishRowCnt = 0;
    for (uint32_t sparseBlockCountIdx = 0; sparseBlockCountIdx < copySparseTotalBlockCount; sparseBlockCountIdx++) {
        // 计算当前block count需要搬入的block size
        int32_t copySparseBlockSize = constInfo.sparseBlockSize;
        if (sparseBlockCountIdx + 1 == copySparseTotalBlockCount) {  //尾块
            copySparseBlockSize = copySparseBlockSizeTail;
        }
        uint32_t curLogicSeqIdx = info.s2BatchOffset + nCopyIdx * nCopyRowCount + sparseBlockCountIdx * constInfo.sparseBlockSize;
        uint32_t topkGmIdx = curLogicSeqIdx / constInfo.sparseBlockSize;
        uint32_t realS2Idx = topKGm.GetValue(info.topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
            (curLogicSeqIdx % constInfo.sparseBlockSize);
        PipeBarrier<PIPE_V>();
        uint64_t blockIdOffset = realS2Idx / kvCacheBlockSize;
        uint64_t reaminRowCnt = realS2Idx % kvCacheBlockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
        uint32_t copyRowCnt = ((kvCacheBlockSize - reaminRowCnt) < (copySparseBlockSize)) ?
            (kvCacheBlockSize - reaminRowCnt) : copySparseBlockSize;
        if (copyFinishRowCnt + copyRowCnt > nActCopyRowCount) {
            copyRowCnt = nActCopyRowCount - copyFinishRowCnt;
        }
        uint64_t keyOffset = idInBlockTable * kvCacheBlockSize * constInfo.headDim * constInfo.kvHeadNum;
        keyOffset += (uint64_t)(info.n2Idx * constInfo.headDim * kvCacheBlockSize) + reaminRowCnt * L1_NZ_KV_BLOCK_ELEMENT;
        CopyInMm1BToL1ForPA(bL1Tensor, keyOffset, nActCopyRowCountAlign, copyFinishRowCnt, copyRowCnt, info);
        copyFinishRowCnt += copyRowCnt;
        curLogicSeqIdx += copyRowCnt;
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm2BToL1CubeDisepr(
    LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t kCopyIdx, uint32_t kCopyRowCount,
    uint32_t kActCopyRowCount, uint32_t kActCopyRowCountAlign)
{   
    if (info.aicS2AccessSize == 0) {
        return;
    }
    uint32_t copySparseTotalBlockCount = (kActCopyRowCount + constInfo.sparseBlockSize - 1) / constInfo.sparseBlockSize;
    uint32_t copySparseBlockSizeTail = kActCopyRowCount - ((copySparseTotalBlockCount - 1) * constInfo.sparseBlockSize);
    uint64_t blockTableBaseOffset = info.bIdx * maxBlockNumPerBatch;
    uint32_t copyFinishRowCnt = 0;
    for (uint32_t sparseBlockCountIdx = 0; sparseBlockCountIdx < copySparseTotalBlockCount; sparseBlockCountIdx++) {
        int32_t copySparseBlockSize = constInfo.sparseBlockSize;
        if (sparseBlockCountIdx + 1 == copySparseTotalBlockCount) {  //尾块
            copySparseBlockSize = copySparseBlockSizeTail;
        }
        uint32_t curLogicSeqIdx = info.s2BatchOffset + kCopyIdx * kCopyRowCount + sparseBlockCountIdx * constInfo.sparseBlockSize;
        uint32_t topkGmIdx = curLogicSeqIdx / constInfo.sparseBlockSize;
        uint32_t realS2Idx = topKGm.GetValue(info.topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
            (curLogicSeqIdx % constInfo.sparseBlockSize);
        PipeBarrier<PIPE_V>();
        uint64_t blockIdOffset = realS2Idx / kvCacheBlockSize;
        uint64_t reaminRowCnt = realS2Idx % kvCacheBlockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // 从block table上的获取编号
        uint32_t copyRowCnt = ((kvCacheBlockSize - reaminRowCnt) < (copySparseBlockSize)) ?
            (kvCacheBlockSize - reaminRowCnt) : copySparseBlockSize;
        if (copyFinishRowCnt + copyRowCnt > kActCopyRowCount) {
            copyRowCnt = kActCopyRowCount - copyFinishRowCnt;
        }
        uint64_t valueOffset = idInBlockTable * kvCacheBlockSize * constInfo.headDim * constInfo.kvHeadNum;
        valueOffset += (uint64_t)(info.n2Idx * constInfo.headDim * kvCacheBlockSize) + reaminRowCnt * L1_NZ_KV_BLOCK_ELEMENT;
        CopyInMm2BToL1ForPA(bL1Tensor, valueOffset, copyFinishRowCnt, copyRowCnt);
        copyFinishRowCnt += copyRowCnt;
        curLogicSeqIdx += copyRowCnt;
    }
}


template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm2BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t valueGmBaseOffset, uint32_t copyStartRowCnt, uint32_t kActCopyRowCount)
{
    uint32_t blockK = 32 / sizeof(KV_T); // 单个ND矩阵的行数
    uint32_t blockElementCnt = 32 / sizeof(KV_T);

    uint64_t step = blockElementCnt;

    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInABlockK = copyStartRowCnt % blockK;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInABlockK + kActCopyRowCount <= blockK) {
        nHead = 0;
        midNdNum = 0;
        nTail = kActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInABlockK;
    } else {
        if (copyStartRowCntInABlockK == 0) {
            nHead = 0;
        } else {
            nHead = blockK - copyStartRowCntInABlockK;
        }
        midNdNum = (kActCopyRowCount - nHead) / blockK;
        nTail = (kActCopyRowCount - nHead) % blockK;
        copyStartRowCntInBaseNTail = 0;
    }

    if (nHead != 0) {
        DataCopyParams intriParams;
        intriParams.blockLen = nHead;
        intriParams.blockCount = constInfo.headDim / blockElementCnt;
        intriParams.dstStride = blockK - nHead;
        intriParams.srcStride = kvCacheBlockSize - nHead;

        uint32_t ndNumFinish = copyStartRowCnt / blockK;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInABlockK * (32 / sizeof(KV_T))],
            valueGm[valueGmBaseOffset], intriParams);
    }

    if (midNdNum != 0) {
        DataCopyParams intriParams;
        intriParams.blockLen = blockK;
        intriParams.blockCount = constInfo.headDim / blockElementCnt;
        intriParams.dstStride = 0;
        intriParams.srcStride = kvCacheBlockSize - blockK;

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK;
        for (uint32_t i = 0; i < midNdNum; i++) {
            DataCopy(bL1Tensor[(ndNumFinish + i) * blockK * headDimAlign],
                valueGm[valueGmBaseOffset + nHead * step + i * blockK * step], intriParams);
        }
    }

    if (nTail != 0) {
        DataCopyParams intriParams;
        intriParams.blockLen = nTail;
        intriParams.blockCount = constInfo.headDim / blockElementCnt;
        intriParams.dstStride = blockK - nTail;
        intriParams.srcStride = kvCacheBlockSize - nTail;

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / blockK + midNdNum;
        DataCopy(bL1Tensor[ndNumFinish * blockK * headDimAlign + copyStartRowCntInBaseNTail * (32 / sizeof(KV_T))],
            valueGm[valueGmBaseOffset + (nHead + midNdNum * blockK) * step], intriParams);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm1BToL1ForPA(
    LocalTensor<KV_T>& bL1Tensor, uint64_t keyGmBaseOffset, uint32_t copyTotalRowCnt,
    uint32_t copyStartRowCnt, uint32_t nActCopyRowCount, const RunInfo &info)
{
    uint32_t baseN = 256 / sizeof(KV_T);   // 256

    uint32_t nHead = 0;
    uint32_t nTail = 0;
    uint32_t midNdNum = 0;
    uint32_t copyStartRowCntInBaseN = copyStartRowCnt % L1_NZ_BLOCK_SIZE_MM1;
    uint32_t copyStartRowCntInBaseNTail = 0;
    if (copyStartRowCntInBaseN + nActCopyRowCount <= L1_NZ_BLOCK_SIZE_MM1) {
        nHead = 0;
        midNdNum = 0;
        nTail = nActCopyRowCount;
        copyStartRowCntInBaseNTail = copyStartRowCntInBaseN;
    } else {
        if (copyStartRowCntInBaseN == 0) {
            nHead = 0;
        } else {
            nHead = L1_NZ_BLOCK_SIZE_MM1 - copyStartRowCntInBaseN;
        }
        midNdNum = (nActCopyRowCount - nHead) / L1_NZ_BLOCK_SIZE_MM1;
        nTail = (nActCopyRowCount - nHead) % L1_NZ_BLOCK_SIZE_MM1;
        copyStartRowCntInBaseNTail = 0;
    }

    uint32_t dstNzC0StrideTail = L1_NZ_BLOCK_SIZE_MM1;
    if (copyTotalRowCnt % L1_NZ_BLOCK_SIZE_MM1 != 0) {
        if ((copyStartRowCnt + nActCopyRowCount) > (copyTotalRowCnt / L1_NZ_BLOCK_SIZE_MM1 * L1_NZ_BLOCK_SIZE_MM1)) {
            dstNzC0StrideTail = copyTotalRowCnt % L1_NZ_BLOCK_SIZE_MM1;
        }
    }
    Nd2NzParams mm1Nd2NzParamsForB;
    if (nHead != 0) {
        DataCopyParams intriParams;
        intriParams.blockLen = nHead;
        intriParams.blockCount = constInfo.headDim / L1_NZ_KV_BLOCK_ELEMENT;
        intriParams.dstStride = L1_NZ_BLOCK_SIZE_MM1 - nHead;
        intriParams.srcStride = kvCacheBlockSize - nHead;

        uint32_t ndNumFinish = copyStartRowCnt / L1_NZ_BLOCK_SIZE_MM1;
        DataCopy(bL1Tensor[ndNumFinish * L1_NZ_BLOCK_SIZE_MM1 * headDimAlign + copyStartRowCntInBaseN * (32 / sizeof(KV_T))],
            keyGm[keyGmBaseOffset], intriParams);
        copyTotalRowCnt -= nHead;
    }

    if (midNdNum != 0) {
        DataCopyParams intriParams;
        intriParams.blockLen = L1_NZ_BLOCK_SIZE_MM1;
        intriParams.blockCount = constInfo.headDim / L1_NZ_KV_BLOCK_ELEMENT;
        intriParams.dstStride = 0;
        intriParams.srcStride = kvCacheBlockSize - L1_NZ_BLOCK_SIZE_MM1;

        int32_t ndNumFinish = (copyStartRowCnt + nHead) / L1_NZ_BLOCK_SIZE_MM1;
        for (uint32_t i = 0; i < midNdNum; i++) {
            DataCopy(bL1Tensor[(ndNumFinish + i) * L1_NZ_BLOCK_SIZE_MM1 * headDimAlign],
                keyGm[keyGmBaseOffset + nHead * L1_NZ_KV_BLOCK_ELEMENT + i * L1_NZ_BLOCK_SIZE_MM1 * L1_NZ_KV_BLOCK_ELEMENT], intriParams);
        }
        copyTotalRowCnt -= midNdNum * L1_NZ_BLOCK_SIZE_MM1;
    }

    if (nTail != 0) {
        DataCopyParams intriParams;
        intriParams.blockLen = nTail;  // 16
        intriParams.blockCount = constInfo.headDim / L1_NZ_KV_BLOCK_ELEMENT;  // 4
        intriParams.dstStride = dstNzC0StrideTail - nTail;
        intriParams.srcStride = kvCacheBlockSize - nTail;

        uint32_t ndNumFinish = (copyStartRowCnt + nHead) / L1_NZ_BLOCK_SIZE_MM1 + midNdNum;
        DataCopy(bL1Tensor[ndNumFinish * L1_NZ_BLOCK_SIZE_MM1 * headDimAlign + copyStartRowCntInBaseNTail * (L1_NZ_KV_BLOCK_ELEMENT / sizeof(KV_T))],
            keyGm[keyGmBaseOffset + (nHead + midNdNum * L1_NZ_BLOCK_SIZE_MM1) * L1_NZ_KV_BLOCK_ELEMENT], intriParams);
    }
}


// nCopyRowCount需要32元素对齐
template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm1BToL1VecMerge(
    LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t aicNZBlockBum, uint32_t aicNzBlockTail,
    uint32_t VecAccessSize, uint32_t alreadyVecAccesssSize, uint32_t nActCopyRowCountAlign, uint32_t CubeAccessSize)
{
    if (VecAccessSize == 0) {
        return;
    }
    uint64_t keyMergeGmBaseOffset = (info.loop % 4) * SALSC_S2MERGESIZE * constInfo.headDim + alreadyVecAccesssSize * L1_NZ_KV_BLOCK_ELEMENT;
    int32_t currentAivSize = 0;
    if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ) {
        // head mid tail
        int32_t aivNzBlockHeadSize = aicNzBlockTail == 0 ? 0 : L1_NZ_BLOCK_SIZE_MM1 - aicNzBlockTail;
        int32_t remaining;
        int32_t aivNzBlockNum;
        int32_t aivNzBlockTail;
        int32_t aivNzBlockTailAlign;
        int32_t HeadBlockSize = L1_NZ_BLOCK_SIZE_MM1;
        if (aivNzBlockHeadSize > VecAccessSize) {
            aivNzBlockHeadSize = nActCopyRowCountAlign - CubeAccessSize;
            aivNzBlockNum = 0;
            aivNzBlockTail = 0;
            aivNzBlockTailAlign = 0;
            HeadBlockSize = nActCopyRowCountAlign - aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM1;
        } else {
            remaining =  VecAccessSize - aivNzBlockHeadSize;
            aivNzBlockNum = remaining >> L1_NZ_BLOCK_SIZE_MM1_SHIFT;
            aivNzBlockTail = remaining & L1_NZ_BLOCK_SIZE_MM1_MASK;
            aivNzBlockTailAlign = nActCopyRowCountAlign - CubeAccessSize - aivNzBlockHeadSize - (aivNzBlockNum) * L1_NZ_BLOCK_SIZE_MM1;
        }
        DataCopyParams mm1CopyParamsForB;
        mm1CopyParamsForB.blockCount = constInfo.headDim / L1_NZ_KV_BLOCK_ELEMENT;
        if (aivNzBlockHeadSize != 0) { 
            mm1CopyParamsForB.blockLen = aivNzBlockHeadSize;
            mm1CopyParamsForB.dstStride = HeadBlockSize - aivNzBlockHeadSize;
            mm1CopyParamsForB.srcStride =  SALSC_S2MERGESIZE - aivNzBlockHeadSize;
            DataCopy(bL1Tensor[aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM1 * 128 + aicNzBlockTail * L1_NZ_KV_BLOCK_ELEMENT],
                keyMergeGm_[keyMergeGmBaseOffset], mm1CopyParamsForB);
            aicNZBlockBum = aicNZBlockBum + 1;
            currentAivSize = currentAivSize + aivNzBlockHeadSize;
        }
        if (aivNzBlockNum != 0) {
            for (uint32_t i = 0; i < aivNzBlockNum; ++i) {
                mm1CopyParamsForB.blockLen = L1_NZ_BLOCK_SIZE_MM1;
                mm1CopyParamsForB.dstStride = 0;
                mm1CopyParamsForB.srcStride =  SALSC_S2MERGESIZE - L1_NZ_BLOCK_SIZE_MM1;
                DataCopy(bL1Tensor[aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM1 * 128],
                    keyMergeGm_[keyMergeGmBaseOffset + currentAivSize * L1_NZ_KV_BLOCK_ELEMENT], mm1CopyParamsForB);
                aicNZBlockBum = aicNZBlockBum + 1;
                currentAivSize = currentAivSize + L1_NZ_BLOCK_SIZE_MM1;
            }
        }
        if(aivNzBlockTail != 0) {
            mm1CopyParamsForB.blockLen = SfaaCeilDiv(aivNzBlockTailAlign * L1_NZ_KV_BLOCK_ELEMENT, DATABLOCK_BYTES);
            mm1CopyParamsForB.srcStride = SALSC_S2MERGESIZE - aivNzBlockTailAlign;
            mm1CopyParamsForB.dstStride = 0;
            DataCopy(bL1Tensor[aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM1 * constInfo.headDim],
                keyMergeGm_[keyMergeGmBaseOffset + currentAivSize * L1_NZ_KV_BLOCK_ELEMENT], mm1CopyParamsForB);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm2BToL1VecMerge(
    LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t aicNZBlockBum,
    uint32_t aicNzBlockTail, uint32_t VecAccessSize, uint32_t alreadyVecAccessSize,
    uint32_t kActCopyRowCountAlign, uint32_t CubeAccessSize)
{
    if (VecAccessSize == 0) {
        return;
    }
    uint64_t valueMergeGmBaseOffset = (info.loop % 4) * SALSC_S2MERGESIZE * constInfo.headDim + alreadyVecAccessSize * L1_NZ_KV_BLOCK_ELEMENT;
    if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ) {
        // head mid tail
        int32_t aivNzBlockHeadSize = aicNzBlockTail == 0 ? 0 : L1_NZ_BLOCK_SIZE_MM2 - aicNzBlockTail;
        int32_t currentAivSize = 0;
        int32_t remaining;
        int32_t aivNzBlockNum;
        int32_t aivNzBlockTail;
        int32_t aivNzBlockTailAlign;
        int32_t HeadBlockSize = L1_NZ_BLOCK_SIZE_MM2;
        if (aivNzBlockHeadSize > VecAccessSize) {
            aivNzBlockHeadSize = kActCopyRowCountAlign - CubeAccessSize;
            aivNzBlockNum = 0;
            aivNzBlockTail = 0;
            aivNzBlockTailAlign = 0;
            HeadBlockSize = kActCopyRowCountAlign - aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM2;
        } else {
            remaining = VecAccessSize - aivNzBlockHeadSize;
            aivNzBlockNum = remaining >> L1_NZ_BLOCK_SIZE_MM2_SHIFT;
            aivNzBlockTail = remaining & L1_NZ_BLOCK_SIZE_MM2_MASK;
            aivNzBlockTailAlign = kActCopyRowCountAlign - CubeAccessSize - aivNzBlockHeadSize - (aivNzBlockNum) * L1_NZ_BLOCK_SIZE_MM2;
        }
        DataCopyParams mm1CopyParamsForB;
        mm1CopyParamsForB.blockCount = constInfo.headDim / L1_NZ_KV_BLOCK_ELEMENT;
        if (aivNzBlockHeadSize != 0) { 
            mm1CopyParamsForB.blockLen = aivNzBlockHeadSize;
            mm1CopyParamsForB.dstStride = HeadBlockSize - aivNzBlockHeadSize;
            mm1CopyParamsForB.srcStride =  SALSC_S2MERGESIZE - aivNzBlockHeadSize;
            DataCopy(bL1Tensor[aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM2 * 128 + aicNzBlockTail * L1_NZ_KV_BLOCK_ELEMENT],
                valueMergeGm_[valueMergeGmBaseOffset], mm1CopyParamsForB);
            aicNZBlockBum = aicNZBlockBum + 1;
            currentAivSize = currentAivSize + aivNzBlockHeadSize;
        }
        if (aivNzBlockNum != 0) {
            for (uint32_t i = 0; i < aivNzBlockNum; ++i) {
                mm1CopyParamsForB.blockLen = L1_NZ_BLOCK_SIZE_MM2;
                mm1CopyParamsForB.dstStride = 0;
                mm1CopyParamsForB.srcStride =  SALSC_S2MERGESIZE - L1_NZ_BLOCK_SIZE_MM2;
                DataCopy(bL1Tensor[aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM2 * 128],
                    valueMergeGm_[valueMergeGmBaseOffset + currentAivSize * L1_NZ_KV_BLOCK_ELEMENT], mm1CopyParamsForB);
                aicNZBlockBum = aicNZBlockBum + 1;
                currentAivSize = currentAivSize + L1_NZ_BLOCK_SIZE_MM2;
            }
        }
        if(aivNzBlockTail != 0) {
            mm1CopyParamsForB.blockLen = SfaaCeilDiv(aivNzBlockTailAlign * L1_NZ_KV_BLOCK_ELEMENT, DATABLOCK_BYTES);
            mm1CopyParamsForB.srcStride = SALSC_S2MERGESIZE - aivNzBlockTailAlign;
            mm1CopyParamsForB.dstStride = 0;
            DataCopy(bL1Tensor[aicNZBlockBum * L1_NZ_BLOCK_SIZE_MM2 * constInfo.headDim],
                valueMergeGm_[valueMergeGmBaseOffset + currentAivSize * 32], mm1CopyParamsForB);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm1BToL1(
    LocalTensor<KV_T>& bL1Tensor, const RunInfo &info, uint32_t nCopyIdx, uint32_t nCopyRowCount,
    uint32_t nActCopyRowCount, uint32_t nActCopyRowCountAlign)
{
    uint32_t baseN = 256 / sizeof(KV_T);
    uint64_t keyMergeGmBaseOffset = (info.loop % 4) * SALSC_S2BASEIZE * constInfo.headDim;
    uint32_t nTail = nActCopyRowCount % baseN;
    if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ) {
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        uint32_t L1nLoopTimes = SfaaCeilDiv(nActCopyRowCount, baseN);
        uint32_t nTailAlign = nActCopyRowCountAlign - (L1nLoopTimes - 1) * baseN;

        DataCopyParams mm1CopyParamsForB;
        mm1CopyParamsForB.blockCount = constInfo.headDim / blockElementCnt;
        mm1CopyParamsForB.dstStride = 0;
        for (uint32_t i = 0; i < L1nLoopTimes; ++i) {
            uint32_t n0RealSizeAlign = (i == L1nLoopTimes - 1) ? nTailAlign : baseN;
            mm1CopyParamsForB.blockLen = SfaaCeilDiv(n0RealSizeAlign * blockElementCnt * sizeof(KV_T), DATABLOCK_BYTES);
            mm1CopyParamsForB.srcStride = nActCopyRowCountAlign - n0RealSizeAlign;
            DataCopy(bL1Tensor[i * baseN * constInfo.headDim], keyMergeGm_[keyMergeGmBaseOffset + i * baseN * blockElementCnt],
                mm1CopyParamsForB);
        }
    } else {
        uint64_t step = constInfo.headDim;
        uint32_t ndNum_tmp = nActCopyRowCount / baseN;
        uint32_t nTailAlign = nActCopyRowCountAlign - ndNum_tmp * baseN;

        Nd2NzParams mm1Nd2NzParamsForB;
        if (ndNum_tmp != 0) {
            mm1Nd2NzParamsForB.nValue = baseN; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.dValue = constInfo.headDim; // 单个ND矩阵的列数, 单位为元素个数
            mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzC0Stride = baseN;       // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数
            mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            mm1Nd2NzParamsForB.ndNum = ndNum_tmp;
            mm1Nd2NzParamsForB.srcNdMatrixStride = baseN * step; // 相邻ND矩阵起始地址之间的偏移, 单位为元素
            mm1Nd2NzParamsForB.dstNzMatrixStride = baseN * headDimAlign; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数
            DataCopy(bL1Tensor, keyMergeGm_[keyMergeGmBaseOffset], mm1Nd2NzParamsForB);
        }

        if (nTail != 0){
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = nTail; // 单个ND矩阵的行数, 单位为元素个数
            mm1Nd2NzParamsForB.dValue = constInfo.headDim; // 单个ND矩阵的列数, 单位为元素个数
            mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
            mm1Nd2NzParamsForB.dstNzC0Stride = nTailAlign;      // 转换为NZ矩阵后，相邻Block起始地址之间的偏移, 单位为Block个数 
            mm1Nd2NzParamsForB.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移, 单位为Block个数
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移, 单位为元素个数
            DataCopy(bL1Tensor[ndNum_tmp * baseN * headDimAlign], keyMergeGm_[keyMergeGmBaseOffset + ndNum_tmp * baseN * step],
                    mm1Nd2NzParamsForB);  //需要调整偏移地址，bL1Tensor的偏移， keyGm的偏移
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm1AToL1(LocalTensor<KV_T>& aL1Tensor, const RunInfo &info)
{
    uint32_t mmRowCount = msdIterNum * info.mSize;
    uint32_t copyStride = 16 * headDimAlign;
    uint32_t copyIterNum = (mmRowCount + 15) / 16;
    for(int i = 0; i < copyIterNum; i++) {
        Nd2NzParams mm1Nd2NzParamsForA;
        mm1Nd2NzParamsForA.ndNum = 1; // ND矩阵的个数
        if(i == copyIterNum - 1) {
            mm1Nd2NzParamsForA.nValue = msdIterNum * info.mSize - i * 16;
        }
        else {
            mm1Nd2NzParamsForA.nValue = 16; // 单个ND矩阵的行数, 单位为元素个数
        }
        mm1Nd2NzParamsForA.dValue = headDimAlign; // 单个ND矩阵的列数, 单位为元素个数
        mm1Nd2NzParamsForA.srcDValue = headDimAlign; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForA.dstNzC0Stride = 16; // 转换为NZ矩阵后，同一行相邻Block起始地址之间的偏移, 单位为Block个数
        mm1Nd2NzParamsForA.dstNzNStride = 1; // 转换为NZ矩阵后，ND中之前相邻两行在NZ矩阵中起始地址之间的偏移
        mm1Nd2NzParamsForA.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移   
        DataCopy(aL1Tensor[i * copyStride], queryPreProcessResGm_[(info.bN2Idx % constInfo.preLoadNum) * constInfo.bmm2ResUbSize * 2 + i * copyStride], mm1Nd2NzParamsForA);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    LocalTensor<KV_T> aL1Tensor = qL1Buffers[qL1BufIter * L1Q_BLOCK_SIZE / sizeof(KV_T)];
    if (info.isFirstSInnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0 + qL1BufIter);
        CopyInMm1AToL1(aL1Tensor, info);
        SetFlag<HardEvent::MTE2_MTE1>(L1Q_EVENT0 + qL1BufIter);
        WaitFlag<HardEvent::MTE2_MTE1>(L1Q_EVENT0 + qL1BufIter);
    }
    constexpr uint32_t nCopyRowCount = KV_LOAD_TO_L1_ROW_NUM;
    uint32_t nCopyTimes = (info.actualSingleProcessSInnerSize + nCopyRowCount - 1) / nCopyRowCount;
    uint32_t nTailCopyRowCount = info.actualSingleProcessSInnerSize - (nCopyTimes - 1) * nCopyRowCount;
    uint32_t nTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (nCopyTimes - 1) * nCopyRowCount;
    uint32_t alreadyVecAccesssSize = 0;
    uint32_t intraCoreStart;
    uint32_t intraCoreEnd;
    uint32_t CubeAccessSize;
    uint32_t VecAccessSize;
    for (uint32_t nL1 = 0, nActCopyRowCount = nCopyRowCount, nActCopyRowCountAlign = nCopyRowCount; nL1 < nCopyTimes; nL1++) {
        if (nL1 + 1 == nCopyTimes) { // 尾块处理
            nActCopyRowCount = nTailCopyRowCount;
            nActCopyRowCountAlign = nTailCopyRowCountAlign;
        }
        intraCoreStart = nL1 * KV_LOAD_TO_L1_ROW_NUM;       // 本轮循环的起始行号
        intraCoreEnd = intraCoreStart + nActCopyRowCount;   // 本轮循环的结束行号
        if (info.aicS2AccessSize <= intraCoreStart) {
            CubeAccessSize = 0;
        } else if (info.aicS2AccessSize >= intraCoreEnd) {
            CubeAccessSize = nActCopyRowCount;
        } else {
            CubeAccessSize = info.aicS2AccessSize - intraCoreStart;
        }
        VecAccessSize = nActCopyRowCount - CubeAccessSize;
        kpL1BufIter++;
        LocalTensor<KV_T> bL1Tensor = kpL1Buffers[(kpL1BufIter % 3) * L1KP_BLOCK_SIZE / sizeof(KV_T)];
        WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));

        CopyInMm1BToL1CubeDisepr(bL1Tensor, info, nL1, nCopyRowCount, CubeAccessSize, nActCopyRowCountAlign);
        uint32_t aicNZBlockBum = CubeAccessSize / L1_NZ_BLOCK_SIZE_MM1;
        uint32_t aicNzBlockTail = CubeAccessSize % L1_NZ_BLOCK_SIZE_MM1;
        CopyInMm1BToL1VecMerge(bL1Tensor, info, aicNZBlockBum, aicNzBlockTail, VecAccessSize, alreadyVecAccesssSize,
            nActCopyRowCountAlign, CubeAccessSize);
        alreadyVecAccesssSize = alreadyVecAccesssSize + VecAccessSize;
        SetFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));
        WaitFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));

        constexpr uint32_t baseM = 64 / sizeof(KV_T); // 64
        uint32_t mActCopyRowCount = msdIterNum * info.mSize; // 8
        uint32_t mLoopTimes = (mActCopyRowCount + baseM - 1) / baseM; // 1
        uint32_t mTail = mActCopyRowCount - (mLoopTimes - 1) * baseM; // 8
        bool isHead = false;
        bool isMid = false;
        bool isTail = false;
        bool isOdd = (mLoopTimes % 2) != 0 ? true : false;
        bool hasTail = (mTail == baseM) ? false : true;
        // A:256*128，M方向按照64循环
        for (uint32_t i = 0, actualBaseM = baseM; i < mLoopTimes; i++) {
            if (i + 1 == mLoopTimes) {
                actualBaseM = mTail;
            }
            isHead = (!isOdd && ((!hasTail && (i < (mLoopTimes / 2))) || (i < (mLoopTimes / 2 - 1)))) || (isOdd && (i < (mLoopTimes / 2)));
            isMid = (!isOdd && hasTail && (i == (mLoopTimes / 2 - 1))) || (isOdd && (i == (mLoopTimes / 2)));
            isTail = (!isOdd && (!hasTail || (i > (mLoopTimes / 2 - 1)))) || (isOdd && (i > (mLoopTimes / 2)));
            LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * L0A_PP_SIZE / sizeof(KV_T)];
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
            LoadData2DParams loadData2DParamsForA;
            loadData2DParamsForA.startIndex = 0;
            loadData2DParamsForA.repeatTimes = (actualBaseM + 15) / 16 * headDimAlign / (32 / sizeof(KV_T));
            loadData2DParamsForA.srcStride = 1;
            loadData2DParamsForA.dstGap = 0;
            loadData2DParamsForA.ifTranspose = false;
            LoadData(aL0Tensor, aL1Tensor[i * baseM * headDimAlign], loadData2DParamsForA);
            SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));
            WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));

            constexpr uint32_t baseN = 256 / sizeof(KV_T); // 256
            uint32_t nLoopTimes = (nActCopyRowCountAlign + baseN - 1) / baseN;
            uint32_t nTail = nActCopyRowCountAlign - (nLoopTimes - 1) * baseN;
            for (uint32_t j = 0, actualBaseN = baseN; j < nLoopTimes; j++) {
                if (j + 1 == nLoopTimes) {
                    actualBaseN = nTail;
                }
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * L0B_PP_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));
                uint32_t blockElementCnt = 32 / sizeof(KV_T);
                LoadData2DParams loadData2DParamsForB;
                loadData2DParamsForB.startIndex = 0;
                loadData2DParamsForB.srcStride = 1;
                loadData2DParamsForB.dstGap = 0;
                loadData2DParamsForB.ifTranspose = false;
                loadData2DParamsForB.repeatTimes = (actualBaseN / 16) * (headDimAlign / blockElementCnt);
                LoadData(bL0Tensor, bL1Tensor[baseN * headDimAlign * j], loadData2DParamsForB);
                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));

                MmadParams mmadParams;
                mmadParams.m = actualBaseM;
                if (mmadParams.m == 1) {  // m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = actualBaseN; // 无效数据不参与计算
                mmadParams.k = 128;
                mmadParams.cmatrixInitVal = true;
                mmadParams.cmatrixSource = false;

                LocalTensor<L0C_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * L0C_PP_SIZE / sizeof(L0C_T)];
                WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));

                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
                WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
                if (mLoopTimes == 1) {
                    for (uint32_t mIter = 0; mIter < msdIterNum; mIter++) {
                        float tmp = quantScaleC1S1;
                        if (mIter == 1) {
                            tmp = quantScaleC1S2;
                        }
                        FixpipeParamsV220 fixParams;
                        fixParams.nSize = actualBaseN;
                        fixParams.mSize = actualBaseM / msdIterNum; // 有效数据不足16行，只需要输出部分行即可
                        fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                        fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                        fixParams.ndNum = 1;
                        fixParams.quantPre = QuantMode_t::DEQF16;
                        fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                        if (mIter == 1) {
                            SetAtomicAdd<half>();
                        }
                        Fixpipe(mm1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize + i * baseM * info.actualSingleProcessSInnerSizeAlign + nL1 * nCopyRowCount + j * baseN],
                                cL0Tensor[info.mSize * mIter * 16], fixParams);
                        if (mIter == 1) {
                            SetAtomicNone();
                        }
                        PipeBarrier<PIPE_FIX>();
                    }
                } else if (isHead) {
                    float tmp = quantScaleC1S1;
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = actualBaseN;
                    fixParams.mSize = actualBaseM; // 有效数据不足16行，只需要输出部分行即可
                    fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;
                    fixParams.quantPre = QuantMode_t::DEQF16;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    Fixpipe(mm1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize + i * baseM * info.actualSingleProcessSInnerSizeAlign + nL1 * nCopyRowCount + j * baseN],
                            cL0Tensor, fixParams);
                    PipeBarrier<PIPE_FIX>();
                } else if (isTail) {
                    float tmp = quantScaleC1S2;
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = actualBaseN;
                    fixParams.mSize = actualBaseM; // 有效数据不足16行，只需要输出部分行即可
                    fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;
                    fixParams.quantPre = QuantMode_t::DEQF16;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    SetAtomicAdd<half>();
                    Fixpipe(mm1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize + (i * baseM - info.mSize) * info.actualSingleProcessSInnerSizeAlign + nL1 * nCopyRowCount + j * baseN],
                            cL0Tensor, fixParams);
                    SetAtomicNone();
                    PipeBarrier<PIPE_FIX>();
                } else {
                    float tmp = quantScaleC1S1;
                    FixpipeParamsV220 fixParams;
                    fixParams.nSize = actualBaseN;
                    fixParams.mSize = info.mSize - i * baseM; // 有效数据不足16行，只需要输出部分行即可
                    fixParams.srcStride = ((actualBaseM + 15) / 16) * 16;
                    fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
                    fixParams.ndNum = 1;
                    fixParams.quantPre = QuantMode_t::DEQF16;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    Fixpipe(mm1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize + i * baseM * info.actualSingleProcessSInnerSizeAlign + nL1 * nCopyRowCount + j * baseN],
                                cL0Tensor, fixParams);
                    PipeBarrier<PIPE_FIX>();
                    tmp = quantScaleC1S2;
                    fixParams.mSize = (i + 1) * baseM - info.mSize;
                    fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                    SetAtomicAdd<half>();
                    Fixpipe(mm1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize + nL1 * nCopyRowCount + j * baseN], cL0Tensor[(info.mSize - i * baseM) * 16], fixParams);
                    SetAtomicNone();
                    PipeBarrier<PIPE_FIX>();
                }
                SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
                cL0BufIter++;
                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));
                bL0BufIter++;
            }
            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
            aL0BufIter++;
        }
        SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));
    }
    // will change batch
    if ((info.s2Idx + 1) == info.curSInnerLoopTimes) {
        SetFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0 + qL1BufIter);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::CopyInMm2BToL1(LocalTensor<KV_T>& bL1Tensor, const RunInfo &info,
    uint32_t kCopyIdx, uint32_t kActCopyRowCountAlign, uint32_t kActCopyRowCount)
{
    uint32_t blockK = 32 / sizeof(KV_T); // 单个ND矩阵的行数
    uint64_t valueMergeGmBaseOffset = (info.loop % 4) * SALSC_S2BASEIZE * constInfo.headDim;

    if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_NZ) {
        uint32_t blockElementCnt = 32 / sizeof(KV_T);
        DataCopyParams mm2CopyParamsForB;
        uint32_t L1kLoopTimes = SfaaCeilDiv(kActCopyRowCount, blockK);
        uint32_t kTailAlign = kActCopyRowCountAlign - (L1kLoopTimes - 1) * blockK;

        mm2CopyParamsForB.blockCount = constInfo.headDim / blockElementCnt;
        mm2CopyParamsForB.dstStride = 0;
        for (uint32_t i = 0; i < L1kLoopTimes; ++i) {
            uint32_t k0RealSizeAlign = (i == L1kLoopTimes - 1) ? kTailAlign : blockK;
            mm2CopyParamsForB.blockLen = SfaaCeilDiv(k0RealSizeAlign * blockElementCnt * sizeof(KV_T), DATABLOCK_BYTES);
            mm2CopyParamsForB.srcStride = kActCopyRowCountAlign - k0RealSizeAlign;
            DataCopy(bL1Tensor[i * blockK * constInfo.headDim], valueMergeGm_[valueMergeGmBaseOffset + i * blockK * blockElementCnt],
                mm2CopyParamsForB);
        }
    } else {
        uint64_t step = constInfo.headDim;
        uint32_t kTail = kActCopyRowCount % blockK;
        uint32_t ndNum_tmp = kActCopyRowCount / blockK;

        Nd2NzParams mm1Nd2NzParamsForB;
        mm1Nd2NzParamsForB.dValue = constInfo.headDim;
        mm1Nd2NzParamsForB.srcDValue = step; // 同一个ND矩阵中相邻行起始地址之间的偏移, 单位为元素个数
        mm1Nd2NzParamsForB.dstNzC0Stride = blockK;
        mm1Nd2NzParamsForB.dstNzNStride = 1;
        if (ndNum_tmp != 0){
            mm1Nd2NzParamsForB.ndNum = ndNum_tmp;
            mm1Nd2NzParamsForB.nValue = blockK;
            mm1Nd2NzParamsForB.srcNdMatrixStride = blockK * step;
            mm1Nd2NzParamsForB.dstNzMatrixStride = blockK * headDimAlign;
            DataCopy(bL1Tensor, valueMergeGm_[valueMergeGmBaseOffset], mm1Nd2NzParamsForB);
        }
        if (kTail != 0){
            mm1Nd2NzParamsForB.ndNum = 1;
            mm1Nd2NzParamsForB.nValue = kTail;
            mm1Nd2NzParamsForB.srcNdMatrixStride = 0;
            mm1Nd2NzParamsForB.dstNzMatrixStride = 0;
            DataCopy(bL1Tensor[ndNum_tmp * blockK * headDimAlign], valueMergeGm_[valueMergeGmBaseOffset + ndNum_tmp * blockK * step], mm1Nd2NzParamsForB);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::LoadDataMm2A(LocalTensor<KV_T> aL0Tensor, LocalTensor<KV_T> aL1Tensor, uint32_t kSize)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = kSize / (32 / sizeof(KV_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(aL0Tensor, aL1Tensor, loadData2DParams);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqaMsd<SFAAT>::ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    constexpr uint32_t mCopyRowCount = P_LOAD_TO_L1_ROW_NUM; // 128
    uint32_t mActRowCount = msdIterNum * info.mSize; // 2
    uint32_t mCopyTimes = (mActRowCount + mCopyRowCount - 1) / mCopyRowCount; // 1
    uint32_t mTailCopyRowCount = mActRowCount - (mCopyTimes - 1) * mCopyRowCount; // 2

    constexpr uint32_t kCopyRowCount = KV_LOAD_TO_L1_ROW_NUM; // 512
    uint32_t kCopyTimes = (info.actualSingleProcessSInnerSize + kCopyRowCount - 1) / kCopyRowCount;
    uint32_t kTailCopyRowCount = info.actualSingleProcessSInnerSize - (kCopyTimes - 1) * kCopyRowCount;
    uint32_t kTailCopyRowCountAlign = info.actualSingleProcessSInnerSizeAlign - (kCopyTimes - 1) * kCopyRowCount;

    for (uint32_t mCopyIdx = 0, mActCopyRowCount = mCopyRowCount; mCopyIdx < mCopyTimes; mCopyIdx++) {
        if (mCopyIdx + 1 == mCopyTimes) {
            mActCopyRowCount = mTailCopyRowCount;
        }
        LocalTensor<L0C_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * L0C_PP_SIZE / sizeof(L0C_T)];
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
        uint32_t CubeAccessSize;
        uint32_t VecAccessSize;
        uint32_t alreadyVecAccesssSize = 0;
        uint32_t intraCoreStart;
        uint32_t intraCoreEnd;
        for (uint32_t kCopyIdx = 0, kActCopyRowCount = kCopyRowCount, kActCopyRowCountAlign = kCopyRowCount; kCopyIdx < kCopyTimes; kCopyIdx++) {
            if (kCopyIdx + 1 == kCopyTimes) {
                kActCopyRowCount = kTailCopyRowCount;
                kActCopyRowCountAlign = kTailCopyRowCountAlign;
            }
            kpL1BufIter++;
            LocalTensor<KV_T> aL1Tensor = kpL1Buffers[(kpL1BufIter % 3) * L1KP_BLOCK_SIZE / sizeof(KV_T)];
            WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));
            CopyInMm2AToL1(aL1Tensor, info, mCopyIdx, mCopyRowCount, mActCopyRowCount, kCopyIdx, kCopyRowCount, kActCopyRowCountAlign);
            SetFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));
            WaitFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % 3));
            LocalTensor<KV_T> bL1Tensor = vL1Buffers[(vL1BufIter % 4) * L1V_BLOCK_SIZE / sizeof(KV_T)];
            uint32_t kb = 0;
            if (mCopyIdx == 0) {
                intraCoreStart = kCopyIdx * KV_LOAD_TO_L1_ROW_NUM;
                intraCoreEnd = intraCoreStart + kActCopyRowCount;
                if (info.aicS2AccessSize <= intraCoreStart) {
                    CubeAccessSize = 0;
                } else if (info.aicS2AccessSize >= intraCoreEnd) {
                    CubeAccessSize = kActCopyRowCount;
                } else {
                    CubeAccessSize = info.aicS2AccessSize - intraCoreStart;
                }
                VecAccessSize = kActCopyRowCount - CubeAccessSize;
                vL1BufIter++;
                bL1Tensor = vL1Buffers[(vL1BufIter % 4) * L1V_BLOCK_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0 + (vL1BufIter % 4));
                CopyInMm2BToL1CubeDisepr(bL1Tensor, info, kCopyIdx, kCopyRowCount, CubeAccessSize, kActCopyRowCountAlign);
                uint32_t aicNZBlockBum = CubeAccessSize / L1_NZ_BLOCK_SIZE_MM2;
                uint32_t aicNzBlockTail = CubeAccessSize % L1_NZ_BLOCK_SIZE_MM2;
                CopyInMm2BToL1VecMerge(bL1Tensor, info, aicNZBlockBum, aicNzBlockTail, VecAccessSize, alreadyVecAccesssSize,
                    kActCopyRowCountAlign, CubeAccessSize);
                alreadyVecAccesssSize = alreadyVecAccesssSize + VecAccessSize;
                SetFlag<HardEvent::MTE2_MTE1>(L1V_EVENT0 + (vL1BufIter % 4));
                WaitFlag<HardEvent::MTE2_MTE1>(L1V_EVENT0 + (vL1BufIter % 4));
                kb = vL1BufIter;
            } else {
                kb = vL1BufIter - (kCopyTimes-kCopyIdx-1);
                bL1Tensor = vL1Buffers[(kb % 4) * L1V_BLOCK_SIZE / sizeof(KV_T)];
            }
            constexpr uint32_t baseK = 128 / sizeof(KV_T);
            uint32_t kLoopTimes = (kActCopyRowCountAlign + baseK - 1) / baseK;
            uint32_t kTailAlign = kActCopyRowCountAlign - (kLoopTimes - 1) * baseK;
            uint32_t kTail = kActCopyRowCount - (kLoopTimes - 1) * baseK;
            for (uint32_t i = 0, actualBaseKAlign = baseK, actualBaseK = baseK; i < kLoopTimes; i++) {
                if (i + 1 == kLoopTimes) {
                    actualBaseKAlign = kTailAlign;
                    actualBaseK = kTail;
                }

                LocalTensor<KV_T> aL0Tensor = aL0TensorPingPong[(aL0BufIter % 2) * L0A_PP_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
                LocalTensor<KV_T> bL0Tensor = bL0TensorPingPong[(bL0BufIter % 2) * L0B_PP_SIZE / sizeof(KV_T)];
                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));

                LocalTensor<KV_T> curAL1Tensor = aL1Tensor[16 * baseK * i];

                uint32_t mmRowCount = mActCopyRowCount;
                uint32_t copyStrideL0 = 16 * actualBaseKAlign;
                uint32_t copyStrideL1 = 16 * kActCopyRowCountAlign;
                uint32_t copyIterNum = (mmRowCount + 15) / 16;
                for(int i = 0; i < copyIterNum; i++){
                    LoadDataMm2A(aL0Tensor[i * copyStrideL0], curAL1Tensor[i * copyStrideL1], actualBaseKAlign);
                }
                
                SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));
                WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + (aL0BufIter % 2));

                uint32_t blockElementCnt = 32 / sizeof(KV_T);
                LoadData2dTransposeParams loadData2DTransposeParamsForB;
                loadData2DTransposeParamsForB.startIndex = 0;
                loadData2DTransposeParamsForB.srcStride = 1;
                loadData2DTransposeParamsForB.dstFracGap = 0;
                loadData2DTransposeParamsForB.repeatTimes = (actualBaseKAlign / blockElementCnt) * (headDimAlign / blockElementCnt); // 16
                loadData2DTransposeParamsForB.dstGap = blockElementCnt / 16 - 1;
                uint32_t l1BaseOffset = baseK * headDimAlign * i;
                LoadDataWithTranspose(bL0Tensor, bL1Tensor[l1BaseOffset], loadData2DTransposeParamsForB);
                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + (bL0BufIter % 2));

                MmadParams mmadParams;
                mmadParams.m = mActCopyRowCount;
                if (mmadParams.m == 1) {  // m等于1会默认开GEMV模式，且不可关闭GEMV，所以规避当作矩阵计算
                    mmadParams.m = 16;
                }
                mmadParams.n = 128;
                mmadParams.k = actualBaseK; // 无效数据不参与计算
                mmadParams.cmatrixInitVal = (kCopyIdx == 0) && (i == 0);
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                PipeBarrier<PIPE_M>();

                SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + (aL0BufIter % 2));
                aL0BufIter++;
                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + (bL0BufIter % 2));
                bL0BufIter++;
            }

            SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % 3));
            if ((mCopyIdx + 1) == mCopyTimes) {
                SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0 + (kb % 4));
            }
        }
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
        if (mCopyTimes == 1) {
            for (uint32_t mIter = 0; mIter < msdIterNum; mIter++) {
                float tmp = quantScaleC2O1;
                if (mIter == 1) {
                    tmp = quantScaleC2O2;
                }
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = mActCopyRowCount / msdIterNum; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((msdIterNum * fixParams.mSize + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                if (mIter == 1) {
                    SetAtomicAdd<half>();
                }
                Fixpipe(mm2ResGm[(info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize], cL0Tensor[mIter * fixParams.mSize * 16], fixParams);
                if (mIter == 1) {
                    SetAtomicNone();
                }
                PipeBarrier<PIPE_FIX>();
            }
        } else {
            if (mTailCopyRowCount != mCopyRowCount && mCopyIdx == 0) {
                float tmp = quantScaleC2O1;
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = info.mSize; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((mActCopyRowCount + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                Fixpipe(mm2ResGm[(info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize], cL0Tensor, fixParams);
                PipeBarrier<PIPE_FIX>();
                tmp = quantScaleC2O2;
                fixParams.mSize = mActCopyRowCount - info.mSize;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                SetAtomicAdd<half>();
                Fixpipe(mm2ResGm[(info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize], cL0Tensor[info.mSize * 16], fixParams);
                SetAtomicNone();
                PipeBarrier<PIPE_FIX>();
            } else if (mCopyIdx == 0) {
                float tmp = quantScaleC2O1;
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = mActCopyRowCount; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((mActCopyRowCount + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                Fixpipe(mm2ResGm[(info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize], cL0Tensor, fixParams);
                PipeBarrier<PIPE_FIX>();
            } else {
                float tmp = quantScaleC2O2;
                FixpipeParamsV220 fixParams;
                fixParams.nSize = 128;
                fixParams.mSize = mActCopyRowCount; // 有效数据不足16行，只需要输出部分行即可
                fixParams.srcStride = ((mActCopyRowCount + 15) / 16) * 16;
                fixParams.dstStride = 128;
                fixParams.ndNum = 1;
                fixParams.quantPre = QuantMode_t::DEQF16;
                fixParams.deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&tmp));
                SetAtomicAdd<half>();
                Fixpipe(mm2ResGm[(info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + (mCopyIdx * mCopyRowCount - info.mSize) * headDimAlign], cL0Tensor, fixParams);
                SetAtomicNone();
                PipeBarrier<PIPE_FIX>();
            }
        }
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
        cL0BufIter++;
    }
}

#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_CUBE_GQA_MSD_H
