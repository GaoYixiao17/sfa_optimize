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
 * \file quant_sals_indexer_service_cube.h
 * \brief use 5 buffer for matmul l1, better pipeline
 */
#ifndef QUANT_SALS_INDEXER_SERVICE_CUBE_H
#define QUANT_SALS_INDEXER_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_sals_indexer_common.h"

namespace QSIKernel {
using namespace AscendC;
using namespace QSICommon;

template <typename QSIT>
class QSIMatmulInt4 {
public:
    using Q_T = typename QSIT::queryType;
    using K_T = typename QSIT::keyType;

    __aicore__ inline QSIMatmulInt4(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                               const GlobalTensor<Q_T> &queryGm, const GlobalTensor<int32_t> &mm1ResGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const QSICommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 3;
    static constexpr uint64_t QUERY_BUF_NUM = 2;
    static constexpr uint64_t L0_BUF_NUM = 2;

    static constexpr uint32_t KEY_MTE1_MTE2_EVENT = EVENT_ID2;
    static constexpr uint32_t QUERY_MTE1_MTE2_EVENT = EVENT_ID5;         // KEY_MTE1_MTE2_EVENT + KEY_BUF_NUM;
    static constexpr uint32_t M_MTE1_EVENT = EVENT_ID3;

    static constexpr uint32_t MTE2_MTE1_EVENT = EVENT_ID2;
    static constexpr uint32_t MTE1_M_EVENT = EVENT_ID2;

    static constexpr uint64_t S8_BLOCK_CUBE = 32;
    static constexpr uint64_t S4_BLOCK_CUBE = 64;
    static constexpr uint64_t M_BASIC_BLOCK = 16;
    static constexpr uint64_t D_BASIC_BLOCK = 64;
    static constexpr uint64_t S2_BASIC_BLOCK = 1024;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 16;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 64;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 1024;

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK * D_BASIC_BLOCK;
    static constexpr uint64_t KEY_BUFFER_OFFSET = 2048 * D_BASIC_BLOCK;
    static constexpr uint64_t L0B_BUFFER_OFFSET = D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;
    static constexpr uint64_t L0A_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0;

protected:
    __aicore__ inline void Fixp(uint64_t s1gL0RealSize,
                                uint64_t s2L0RealSize, const QSICommon::RunInfo &runInfo);
    __aicore__ inline void ComputeL0c(uint64_t s2L0Offset, uint64_t s1gL0RealSize,
        uint64_t s2L0RealSize, const QSICommon::RunInfo &runInfo);
    __aicore__ inline void LoadKeyToL0b(uint64_t s2L0RealSize, uint64_t s2L1Offset,
                                        const QSICommon::RunInfo &runInfo);
    __aicore__ inline void LoadQueryToL0a(uint64_t s1gL1RealSize,
                                          uint64_t s1gL0RealSize, const QSICommon::RunInfo &runInfo);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, const QSICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize,
        uint64_t s2L1Offset, uint64_t s2GmOffset, const QSICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize,
        uint64_t s2L1Offset, uint64_t s2GmOffset, const QSICommon::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;
    GlobalTensor<int32_t> mm1ResGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<int32_t> cL0_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t queryL1Mte2BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;

    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = QSIT::pageAttention;
    static constexpr QSI_LAYOUT K_LAYOUT_T = QSIT::keyLayout;
};

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK * D_BASIC_BLOCK * sizeof(Q_T)); // (2) * 16 * 64 * 0.5 = 1K
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * 2048 * D_BASIC_BLOCK * sizeof(K_T)); // (3) * 2048 * 64 * 0.5 = 192K
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0_BUF_NUM * M_BASIC_BLOCK_L0 *
        D_BASIC_BLOCK_L0 * sizeof(Q_T)); // (2) * 16 * 64 * 0.5 = 1K
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0_BUF_NUM * D_BASIC_BLOCK_L0 *
        S2_BASIC_BLOCK_L0 * sizeof(K_T)); // (2) * 64 * 1024 * 0.5 = 64K
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0_BUF_NUM * M_BASIC_BLOCK_L0 *
        S2_BASIC_BLOCK_L0 * sizeof(int32_t)); // 16 * 2048 * 4 = 128K, 不开double buffer
    cL0_ = bufL0C_.Get<int32_t>();
}

template <typename QSIT>
__aicore__ inline void
QSIMatmulInt4<QSIT>::InitMm1GlobalTensor(const GlobalTensor<int32_t> &blkTableGm, const GlobalTensor<K_T> &keyGm,
                                   const GlobalTensor<Q_T> &queryGm, const GlobalTensor<int32_t> &mm1ResGm)
{
    blkTableGm_ = blkTableGm;
    keyGm_ = keyGm;
    queryGm_ = queryGm;
    mm1ResGm_ = mm1ResGm;
}

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::ComputeMm1(const QSICommon::RunInfo &runInfo)
{
    if (runInfo.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
        QueryNd2Nz(1, 0, runInfo);
    }

    uint64_t s2GmBaseOffset = runInfo.s2Idx * constInfo_.s2BaseSize; // 当前基本块在S2方向上的偏移
    for (uint64_t s2GmOffset = 0; s2GmOffset < runInfo.actualSingleProcessSInnerSize; s2GmOffset += S2_BASIC_BLOCK) {
        // 切N轴: 基本块内S2方向上的偏移，每次处理1024个数，循环2次
        WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK > runInfo.actualSingleProcessSInnerSize ?
                runInfo.actualSingleProcessSInnerSize - s2GmOffset : S2_BASIC_BLOCK; // key矩阵mte2的单次搬运量
        if constexpr (PAGE_ATTENTION) {
            KeyNd2NzForPA(s2L1RealSize, s2GmOffset, s2GmBaseOffset + s2GmOffset, runInfo);
        }else {
            KeyNd2Nz(s2L1RealSize, s2GmOffset, s2GmBaseOffset+s2GmOffset, runInfo);
        }

        SetFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);
        WaitFlag<HardEvent::MTE2_MTE1>(MTE2_MTE1_EVENT);

        WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);
        LoadQueryToL0a(1, 1, runInfo);
        LoadKeyToL0b(s2L1RealSize, s2GmOffset, runInfo);

        ComputeL0c(s2GmOffset, 1, s2L1RealSize, runInfo);

        SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

        l0BufIdx_++;

        SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
    CrossCoreWaitFlag(constInfo_.syncV1C1);
    Fixp(1, runInfo.actualSingleProcessSInnerSize, runInfo);
    if (runInfo.isLastS2InnerLoop) {
        SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
        queryL1Mte2BufIdx_++;
    }
}

// bsnd
template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2L1Offset, uint64_t s2GmOffset,
                                               const QSICommon::RunInfo &runInfo)
{
    // DMA BSND
    DataCopyParams copyInParams;
    if (constInfo_.kHeadNum == 1) {
        copyInParams.blockCount = 1; // 待搬运的连续传输数据块个数
        copyInParams.blockLen = s2L1RealSize * constInfo_.headDim / S4_BLOCK_CUBE; // 待搬运的每个连续传输数据块长度，单位为DataBlock(32字节)
        copyInParams.srcStride = 0;
    } else {
        copyInParams.blockCount = s2L1RealSize; // 待搬运的连续传输数据块个数
        copyInParams.blockLen = constInfo_.headDim / S4_BLOCK_CUBE; // 待搬运的每个连续传输数据块长度，单位为DataBlock(32字节)
        copyInParams.srcStride = (constInfo_.kHeadNum - 1) * constInfo_.headDim / S4_BLOCK_CUBE;
    }
    copyInParams.dstStride = 0;
    DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * constInfo_.headDim],
             keyGm_[runInfo.tensorKeyOffset + s2GmOffset * constInfo_.kHeadNum * constInfo_.headDim], copyInParams);
}

// blkNum, blkSize, N2, D
template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2L1BaseOffset, uint64_t s2GmOffset,
                                                    const QSICommon::RunInfo &runInfo)
{
    uint64_t s2L1Offset = 0; // 已经搬运了多少
    while (s2L1Offset < s2L1RealSize) { 
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / constInfo_.kCacheBlockSize; // 当前batch中blockTable中PA block ID
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % constInfo_.kCacheBlockSize; // PA block中的偏移
        uint64_t keyGmOffset = 0;
        uint64_t s2Mte2Size = s2L1RealSize - s2L1Offset; // 本次PA搬运量
        s2Mte2Size = s2BlkOffset + s2Mte2Size >= constInfo_.kCacheBlockSize ? constInfo_.kCacheBlockSize - s2BlkOffset :
                                                                            s2Mte2Size; // 当前搬运量是否跨PA block，跨block时分多次搬运
        DataCopyParams copyInParams;
        if constexpr (K_LAYOUT_T == QSI_LAYOUT::PA_BNSD) { // PA_BNSD(blockNum, n2, blockSize, d)
            keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                               constInfo_.kHeadNum * constInfo_.kCacheBlockSize * constInfo_.headDim +
                                               runInfo.n2Idx * constInfo_.kCacheBlockSize * constInfo_.headDim +
                                               s2BlkOffset * constInfo_.headDim;
            // DMA
            copyInParams.blockCount = 1; // 待搬运的连续传输数据块个数
            copyInParams.blockLen = s2Mte2Size * constInfo_.headDim / S4_BLOCK_CUBE; // 待搬运的每个连续传输数据块长度，单位为DataBlock(32字节)
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
        } else if constexpr (K_LAYOUT_T == QSI_LAYOUT::PA_NZ) {
            uint32_t blockElementCntNZ = 32 / sizeof(K_T);
            if constexpr (IsSameType<K_T, int4b_t>::value) {
                blockElementCntNZ = 64;
            }
            keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                               constInfo_.kHeadNum * constInfo_.kCacheBlockSize * constInfo_.headDim +
                                               runInfo.n2Idx * constInfo_.kCacheBlockSize * constInfo_.headDim +
                                               s2BlkOffset * blockElementCntNZ;
            copyInParams.blockLen = s2Mte2Size; // 待搬运的每个连续传输数据块长度，单位为DataBlock(32字节)
            copyInParams.blockCount = constInfo_.headDim / blockElementCntNZ; // 待搬运的连续传输数据块个数
            copyInParams.dstStride = 0;
            copyInParams.srcStride = constInfo_.kCacheBlockSize - s2Mte2Size;
        } else { // PA_BSND(blockNum, blockSize, n2, d)
            keyGmOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo_.maxBlockNumPerBatch + s2BlkId) *
                                               constInfo_.kCacheBlockSize * constInfo_.kHeadNum * constInfo_.headDim +
                                               s2BlkOffset * constInfo_.kHeadNum * constInfo_.headDim + runInfo.n2Idx * constInfo_.headDim;
            // DMA
            copyInParams.blockCount = s2Mte2Size; // 待搬运的连续传输数据块个数
            copyInParams.blockLen = constInfo_.headDim / S4_BLOCK_CUBE; // 待搬运的每个连续传输数据块长度，单位为DataBlock(32字节)
            copyInParams.srcStride = (constInfo_.kHeadNum - 1) * constInfo_.headDim / S4_BLOCK_CUBE;
            copyInParams.dstStride = 0;
        }
        DataCopy(keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1BaseOffset * constInfo_.headDim + s2L1Offset * constInfo_.headDim],
                    keyGm_[keyGmOffset], copyInParams);
        s2L1Offset += s2Mte2Size;
    }
}

// batch, s1, n2, g, d
template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset,
                                                 const QSICommon::RunInfo &runInfo)
{
    // m=1，按照gemv方式优化
    DataCopyParams copyInParams;
    copyInParams.blockCount = 1; // 待搬运的连续传输数据块个数
    copyInParams.blockLen = constInfo_.headDim / S4_BLOCK_CUBE; // 待搬运的每个连续传输数据块长度，单位为DataBlock(32字节)
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * constInfo_.headDim], copyInParams);
}

// 1, d
template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::LoadQueryToL0a(uint64_t s1gL1RealSize,
                                                     uint64_t s1gL0RealSize, const QSICommon::RunInfo &runInfo)
{
    // m=1，按照gemv方案优化
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = 1;
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(queryL0_[(l0BufIdx_ % L0_BUF_NUM) * L0A_BUFFER_OFFSET],
             queryL1_[(queryL1Mte2BufIdx_ % QUERY_BUF_NUM) * QUERY_BUFFER_OFFSET], loadData2DParams);
}

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::LoadKeyToL0b(uint64_t s2L0RealSize, uint64_t s2L1Offset,
                                                   const QSICommon::RunInfo &runInfo)
{
    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = QSiCeilDiv(s2L0RealSize, static_cast<uint64_t>(BLOCK_CUBE)) * QSiCeilDiv(constInfo_.headDim, static_cast<uint64_t>(S4_BLOCK_CUBE));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(keyL0_[(l0BufIdx_ % L0_BUF_NUM) * L0B_BUFFER_OFFSET],
             keyL1_[(keyL1BufIdx_ % KEY_BUF_NUM) * KEY_BUFFER_OFFSET + s2L1Offset * constInfo_.headDim], loadData2DParams);
}

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::ComputeL0c(uint64_t s2L0Offset, uint64_t s1gL0RealSize, uint64_t s2L0RealSize,
                                                const QSICommon::RunInfo &runInfo)
{
    SetFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    WaitFlag<HardEvent::MTE1_M>(MTE1_M_EVENT);
    MmadParams mmadParams;
    mmadParams.m = 1;
    mmadParams.n = s2L0RealSize;
    mmadParams.k = constInfo_.headDim;
    mmadParams.cmatrixInitVal = true;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = 0b11;
    Mmad(cL0_.template ReinterpretCast<int32_t>()[BLOCK_CUBE * s2L0Offset],
         queryL0_.template ReinterpretCast<int4b_t>()[(l0BufIdx_ % L0_BUF_NUM) * L0A_BUFFER_OFFSET],
         keyL0_.template ReinterpretCast<int4b_t>()[(l0BufIdx_ % L0_BUF_NUM) * L0B_BUFFER_OFFSET], mmadParams);
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::Fixp( uint64_t s1gL0RealSize,
                                           uint64_t s2L0RealSize, const QSICommon::RunInfo &runInfo)
{
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = 1;
    intriParams.nSize = s2L0RealSize;
    intriParams.dstStride = runInfo.actualSingleProcessSInnerSizeAlign; // 使能NZ2ND功能，dst同一ND矩阵的相邻行的偏移(头与头)，取值不为0，单位为元素
    intriParams.srcStride = QSiCeilAlign(s1gL0RealSize, static_cast<uint64_t>(BLOCK_CUBE)); // 使能NZ2ND功能，src同一NZ矩阵的相邻Z排布的偏移(头与头)，必须为16的倍数，取值范围：srcStride∈[0, 65535], 单位C0_size
    // set mode according to dtype
    intriParams.quantPre = QuantMode_t::NoQuant;
    intriParams.nz2ndEn = true;
    intriParams.unitFlag = 0b11; // 3 unitflag
    intriParams.reluPre = 0;
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * constInfo_.mBaseSize * constInfo_.s2BaseSize],
                      cL0_, intriParams);
}

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    SetFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    SetFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}

template <typename QSIT>
__aicore__ inline void QSIMatmulInt4<QSIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + 2);

    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 0);
    WaitFlag<HardEvent::MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + 1);

    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 0);
    WaitFlag<HardEvent::M_MTE1>(M_MTE1_EVENT + 1);
}
} // namespace QSIKernel
#endif