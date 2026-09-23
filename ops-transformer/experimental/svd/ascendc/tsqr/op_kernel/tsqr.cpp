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
 * \file tsqr.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
class QrHouseholderTilingData {
public:
    int batchSize, mDim, kDim, ubSize;
};
#include "qr_householder_single_vec.h"

using namespace AscendC;

template <typename T> class TsqrKernel {
public:

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace,
        const TsqrTilingData tiling, TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessBatch(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
                                        const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm);
    __aicore__ inline void CallQR(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
                                  const GlobalTensor<T>& rGm, int32_t localM, int32_t localN);
    __aicore__ inline void CopyVec(const GlobalTensor<T>& dst, const GlobalTensor<T>& src,
                                   int32_t rows, int32_t columns);
    __aicore__ inline void CopyCube(const GlobalTensor<T>& dst, const GlobalTensor<T>& src,
                                    int32_t rows, int32_t columns);
    __aicore__ inline void ForwardZeroStep(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
        int rId, int32_t localM, int32_t blockSize);
    __aicore__ inline void ForwardStep(int aId, const GlobalTensor<T>& qGm,
                                       int rId, int32_t localM, int32_t blockSize);
    __aicore__ inline void Forward(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
        const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm, int32_t blockSize);
    __aicore__ inline void Reorder(const GlobalTensor<T>& qGm, bool onCube);
    __aicore__ inline void RunMatmul(int M, int N, bool isTransposeB,
        const GlobalTensor<T>& aGm, const GlobalTensor<T>& bGm, const GlobalTensor<T>& cGm);
    __aicore__ inline void BackwardStep(const GlobalTensor<T>& resultQGm, const GlobalTensor<T>& leftQGm,
        const GlobalTensor<T>& rightQGm, int32_t bsLeft, int32_t numBlocksLeft, bool hasTail);
    __aicore__ inline void Backward(const GlobalTensor<T>& qGm, const GlobalTensor<T>& tmpQGm,
        int32_t blockSize);
    __aicore__ inline GlobalTensor<T> getRBlock(int id);

    GlobalTensor<T> aGlobal_; // Input
    GlobalTensor<T> qGlobal_, rGlobal_; // Output
    GlobalTensor<T> tmpQGlobal_, tmpRGlobal_, bufferQGlobal_; // temporary storage in workspace

    QRHouseholderSingleVec qrSubkernel;
    QrHouseholderTilingData qrTilingData;

    TBuf<TPosition::VECOUT> vecBuf;
    TBuf<TPosition::B1> cubeBuf;
    LocalTensor<T> localTensor;

    int32_t M_, N_, batchSize_;
    int32_t blockSize_, numLevels_;
    int64_t tmpQSize_, tmpRSize_, bufferQSize_, maxQrWorkspace_;
    __gm__ uint8_t* qrWorkspace;
    TsqrTilingData tiling;
    TPipe* pipe;

    bool firstTail = true;

    MatmulImpl <MatmulType<TPosition::GM, CubeFormat::ND, T, true>,
                MatmulType<TPosition::GM, CubeFormat::ND, T, true>,
                MatmulType<TPosition::GM, CubeFormat::ND, T>> mm;
    MatmulImpl <MatmulType<TPosition::GM, CubeFormat::ND, T, true>,
                MatmulType<TPosition::GM, CubeFormat::ND, T, false>,
                MatmulType<TPosition::GM, CubeFormat::ND, T>> mmF;
};


template <typename T>
__aicore__ inline void TsqrKernel<T>::Init(GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace,
    const TsqrTilingData tiling, TPipe* tpipe)
{
    M_ = tiling.m;
    N_ = tiling.n;
    batchSize_ = tiling.batchSize;
    blockSize_ = tiling.blockSize;
    numLevels_ = tiling.numLevels;
    tmpQSize_ = tiling.tmpQSize;
    tmpRSize_ = tiling.tmpRSize;
    bufferQSize_ = tiling.bufferQSize;
    maxQrWorkspace_ = tiling.maxQrWorkspace;

    this->tiling = tiling;
    this->pipe = tpipe;

    aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(a), batchSize_ * M_ * N_);
    qGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q), batchSize_ * M_ * N_);
    rGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(r), batchSize_ * N_ * N_);

    __gm__ uint8_t* tmpQPtr = workspace;
    __gm__ uint8_t* tmpRPtr = tmpQPtr + tmpQSize_ * (batchSize_ > 1 ? 2 : 1) * sizeof(T);
    __gm__ uint8_t* bufferQPtr = tmpRPtr + tmpRSize_ * sizeof(T);
    qrWorkspace = bufferQPtr + bufferQSize_ * sizeof(T);
    tmpQGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tmpQPtr), (tmpQSize_) * (batchSize_ > 1 ? 2 : 1));
    tmpRGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tmpRPtr), tmpRSize_);
    bufferQGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bufferQPtr), bufferQSize_);
}

template <typename T>
__aicore__ inline GlobalTensor<T> TsqrKernel<T>::getRBlock(int id)
{
    int64_t offset = (id * N_ * N_);
    return tmpRGlobal_[offset];
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Process()
{
    for (int i = 0; i < batchSize_; i++) {
        ProcessBatch(aGlobal_[M_ * N_ * i], qGlobal_[M_ * N_ * i], rGlobal_[N_ * N_ * i],
                     tmpQGlobal_[tmpQSize_ * (i % 2)]);
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::ProcessBatch(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
    const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm)
{
    if ASCEND_IS_AIV {
        Forward(aGm, qGm, rGm, tmpQGm, blockSize_);
    }
    SyncAll<false>();
#if defined(__DAV_C310__)
    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            Reorder(tmpQGm, false);
        }
    }
    SyncAll<false>();
#else
    if ASCEND_IS_AIC {
        if (GetBlockIdx() == 0) {
            Reorder(tmpQGm, true);
        }
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
        CrossCoreWaitFlag(0x8);
    }
#endif
    if ASCEND_IS_AIC {
        Backward(qGm, tmpQGm, blockSize_);
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::CallQR(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
    const GlobalTensor<T>& rGm, int32_t localM, int32_t localN)
{
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();
    qrTilingData.batchSize = 1;
    qrTilingData.mDim = localM;
    qrTilingData.kDim = localN;
    qrTilingData.ubSize = tiling.ubSize;
    __gm__ uint8_t* aGmPtr = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ float*>(aGm.GetPhyAddr()));
    __gm__ uint8_t* qGmPtr = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ float*>(qGm.GetPhyAddr()));
    __gm__ uint8_t* rGmPtr = reinterpret_cast<__gm__ uint8_t*>(const_cast<__gm__ float*>(rGm.GetPhyAddr()));
    __gm__ uint8_t* localWorkspace = qrWorkspace + GetBlockIdx() * maxQrWorkspace_ * sizeof(T);
    qrSubkernel.Init(aGmPtr, qGmPtr, rGmPtr, localWorkspace, &qrTilingData, pipe);
    qrSubkernel.Process();
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::CopyVec(const GlobalTensor<T>& dst, const GlobalTensor<T>& src,
    int32_t rows, int32_t columns)
{
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();

    pipe->InitBuffer(vecBuf, rows * columns * sizeof(T));
    LocalTensor localTensor = vecBuf.Get<T>();
    DataCopyParams params = { static_cast<uint16_t>(columns), static_cast<uint16_t>(rows / 8), 0, 0 };
    DataCopy(localTensor, src, params);
    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    DataCopy(dst, localTensor, params);
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::CopyCube(const GlobalTensor<T>& dst, const GlobalTensor<T>& src,
    int32_t rows, int32_t columns)
{
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();

    pipe->InitBuffer(cubeBuf, rows * columns * sizeof(T));
    LocalTensor localTensor = cubeBuf.Get<T>();
    DataCopyParams params = { static_cast<uint16_t>(columns), static_cast<uint16_t>(rows / 8), 0, 0 };
    DataCopy(localTensor, src, params);
    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    PipeBarrier<PIPE_ALL>();
    DataCopy(dst, localTensor, params);
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::ForwardZeroStep(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
    int rId, int32_t localM, int32_t blockSize)
{
    int numBlocks = localM / blockSize;
    int coreIdx = AscendC::GetBlockIdx();
    int numCores = GetBlockNum() * 2; // vector cores
    if (numBlocks < numCores) {
        numCores = numBlocks; // don't use other cores
    }
    if (coreIdx < numCores) {
        int perCore = numBlocks / numCores;
        int tail = numBlocks % numCores;
        int start, end;
        if (coreIdx < tail) {
            start = (perCore + 1) * coreIdx;
            end = start + perCore + 1;
        } else {
            start = perCore * coreIdx + tail;
            end = start + perCore;
        }
        for (int idx = start; idx < end; idx++) {
            int64_t qaOffset = idx * blockSize * N_;
            CallQR(aGm[qaOffset], qGm[qaOffset], getRBlock(rId + idx), blockSize, N_);
        }
    }
    SyncAll<true>();
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::ForwardStep(int aId, const GlobalTensor<T>& qGm,
    int rId, int32_t localM, int32_t blockSize)
{
    int numBlocks = localM / blockSize;
    int hasTail = numBlocks % 2;
    int processedBlocks = numBlocks / 2 + hasTail;
    blockSize *= 2;
    int coreIdx = AscendC::GetBlockIdx();
    int numCores = GetBlockNum() * 2; // vector cores
    if (processedBlocks < numCores) {
        numCores = processedBlocks; // don't use other cores
    }
    if (coreIdx < numCores) {
        int perCore = processedBlocks / numCores;
        int tail = processedBlocks % numCores;
        int start, end;
        if (coreIdx < tail) {
            start = (perCore + 1) * coreIdx;
            end = start + perCore + 1;
        } else {
            start = perCore * coreIdx + tail;
            end = start + perCore;
        }
        for (int idx = start; idx < end; idx++) {
            int64_t qOffset = idx * blockSize * N_;
            if (hasTail && (idx == processedBlocks - 1)) {
                // unpaired block
                CopyVec(getRBlock(rId + idx), getRBlock(aId + 2 * idx), N_, N_);;
            } else {
                CallQR(getRBlock(aId + 2 * idx), qGm[qOffset], getRBlock(rId + idx), blockSize, N_);
            }
        }
    }
    SyncAll<true>();
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Forward(const GlobalTensor<T>& aGm, const GlobalTensor<T>& qGm,
    const GlobalTensor<T>& rGm, const GlobalTensor<T>& tmpQGm, int32_t blockSize)
{
    // zero step
    // aGm -> qGm, tmpRGlobal_
    ForwardZeroStep(aGm, tmpQGm, 0, M_, blockSize);
    // blocks are stacked in paires
    int numBlocks = M_ / blockSize;
    int numPairs = numBlocks / 2;
    int tail = (numBlocks % 2 > 0);
    int64_t aOffset = 0;
    int64_t qOffset = M_ * N_;
    int64_t rOffset = numBlocks;
    int localM = numBlocks * N_;
    for (int lvl = 0; lvl < numLevels_ - 1; lvl++) {
        // tmpRGlobal_ -> tmpQGlobal_, tmpRGlobal_

        ForwardStep(aOffset, tmpQGm[qOffset], rOffset, localM, N_);

        aOffset += numBlocks;
        qOffset += (numPairs * 2 + tail) * N_ * N_;
        rOffset += (numPairs + tail);
        numBlocks = numPairs + tail;
        numPairs = numBlocks / 2;
        tail = (numBlocks % 2 > 0);
        localM = numBlocks * N_;
    }
    // last iteration
    // tmpRGlobal_ -> tmpQGlobal_, rGm
    if (GetBlockIdx() == 0) {
        CallQR(getRBlock(aOffset), tmpQGm[qOffset], rGm, 2 * N_, N_);
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Reorder(const GlobalTensor<T>& qGm, bool onCube)
{
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();

    int numBlocks = M_ / blockSize_;
    int numPairs = numBlocks / 2;
    int tail = (numBlocks % 2 > 0);
    int64_t qOffset = M_ * N_;
    for (int lvl = 0; lvl < numLevels_ - 1; lvl++) {
        qOffset += (numPairs * 2 + tail) * N_ * N_;
        numBlocks = numPairs + tail;
        numPairs = numBlocks / 2;
        tail = (numBlocks % 2 > 0);
    }
    GlobalTensor<T> global = qGm[qOffset];

    LocalTensor<T> localTensor;
    if (onCube) {
        pipe->InitBuffer(cubeBuf, 2 * N_ * N_ * sizeof(T));
        localTensor = cubeBuf.Get<float>();
    } else {
        pipe->InitBuffer(vecBuf, 2 * N_ * N_ * sizeof(T));
        localTensor = vecBuf.Get<float>();
    }
    uint16_t n = static_cast<uint16_t>(N_);
    uint16_t blocks = static_cast<uint16_t>(N_ / 8);

    DataCopyParams inParams = { static_cast<uint16_t>(N_), blocks, blocks, 0 };
    DataCopyParams outParams = { n, static_cast<uint16_t>(2 * n / 8), 0, 0 };
    DataCopy(localTensor, global, inParams);
    DataCopy(localTensor[N_ * N_], global[N_], inParams);
    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    PipeBarrier<PIPE_ALL>();
    DataCopy(global, localTensor, outParams);
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::RunMatmul(int M, int N, bool isTransposeB,
    const GlobalTensor<T>& aGm, const GlobalTensor<T>& bGm, const GlobalTensor<T>& cGm)
{
    PipeBarrier<PIPE_ALL>();
    pipe->Reset();
    PipeBarrier<PIPE_ALL>();
    if (isTransposeB) {
        mm.SetSubBlockIdx(0);
        mm.Init(&tiling.mmTilingData, pipe);
        mm.SetOrgShape(M, N, N);
        mm.SetSingleShape(M, N, N);
        mm.SetTensorA(aGm, true);
        mm.SetTensorB(bGm, true);
        mm.IterateAll(cGm);
        mm.End();
    } else {
        mmF.SetSubBlockIdx(0);
        mmF.Init(&tiling.mmTilingDataF, pipe);
        mmF.SetOrgShape(M, N, N);
        mmF.SetSingleShape(M, N, N);
        mmF.SetTensorA(aGm, true);
        mmF.SetTensorB(bGm, false);
        mmF.IterateAll(cGm);
        mmF.End();
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::BackwardStep(const GlobalTensor<T>& outQGm, const GlobalTensor<T>& leftQGm,
    const GlobalTensor<T>& rightQGm, int32_t bsLeft, int32_t numBlocksLeft, bool hasTail)
{
    int coreIdx = AscendC::GetBlockIdx();
    int numCores = GetBlockNum(); // cube cores
    if (numBlocksLeft < numCores) {
        numCores = numBlocksLeft; // don't use other cores
    }
    if (coreIdx < numCores) {
        int perCore = numBlocksLeft / numCores;
        int start, end;
        if (coreIdx < numBlocksLeft % numCores) {
            start = (perCore + 1) * coreIdx;
            end = start + perCore + 1;
        } else {
            start = perCore * coreIdx + numBlocksLeft % numCores;
            end = start + perCore;
        }
        int64_t leftOffset = bsLeft * N_;
        int64_t rightOffset = N_ * N_;
        int64_t outQOffset = bsLeft * N_;
        for (int idx = start; idx < end; idx++) {
            PipeBarrier<PIPE_ALL>();
            bool isFirstIter = (numBlocksLeft == 2);
            bool isTransfered = (idx == numBlocksLeft - 1) && (numBlocksLeft % 2 == 1) && (!hasTail) && firstTail;
            if (hasTail && (idx == numBlocksLeft - 1)) {
                // unpaired block
                CopyCube(outQGm[idx * outQOffset], rightQGm[idx * rightOffset], N_, N_);
            } else {
                PipeBarrier<PIPE_ALL>();
                pipe->Reset();
                PipeBarrier<PIPE_ALL>();
                RunMatmul(bsLeft, N_, (isFirstIter || isTransfered),
                    leftQGm[idx * leftOffset], rightQGm[idx * rightOffset], outQGm[idx * outQOffset]);
            }
        }
    }
    PipeBarrier<PIPE_ALL>();
    if ((numBlocksLeft % 2 == 1) && (!hasTail) && firstTail) {
        firstTail = false;
    }
}

template <typename T>
__aicore__ inline void TsqrKernel<T>::Backward(const GlobalTensor<T>& qGm, const GlobalTensor<T>& tmpQGm,
    int32_t blockSize)
{
    int numBlocks, numPairs, tail, lvl;
    int64_t rightOffset, leftOffset;
    bool hasTail;
    for (int lvl = numLevels_ - 1; lvl > 0; lvl--) {
        numBlocks = M_ / blockSize;
        numPairs = numBlocks / 2;
        tail = (numBlocks % 2 > 0);
        rightOffset = M_ * N_;
        leftOffset = 0;
        for (int __lvl = 0; __lvl < lvl; __lvl++) {
            leftOffset = rightOffset;
            rightOffset += (numPairs * 2 + tail) * N_ * N_;
            numBlocks = numPairs + tail;
            numPairs = numBlocks / 2;
            hasTail = tail;
            tail = (numBlocks % 2 > 0);
        }
        int64_t rrightOffset = rightOffset + (numPairs * 2 + tail) * N_ * N_;
        if (lvl == numLevels_ - 1) {
            firstTail = hasTail;
        }
        if ((numLevels_ - 1 - lvl) % 2 == 0) {
            if (lvl == numLevels_ - 1) {
                BackwardStep(bufferQGlobal_, tmpQGm[leftOffset], tmpQGm[rightOffset], 2 * N_, numBlocks, hasTail);
            } else {
                BackwardStep(bufferQGlobal_, tmpQGm[leftOffset], tmpQGm[rrightOffset], 2 * N_, numBlocks, hasTail);
            }
        } else {
            BackwardStep(tmpQGm[rightOffset], tmpQGm[leftOffset], bufferQGlobal_, 2 * N_, numBlocks, hasTail);
        }
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
        CrossCoreWaitFlag(0x8);
    }
    numBlocks = M_ / blockSize_;
    tail = (numBlocks % 2 > 0);
    numPairs = numBlocks / 2;
    rightOffset = M_ * N_;
    for (int __lvl = 0; __lvl < 1; __lvl++) {
        rightOffset += (numPairs * 2 + tail) * N_ * N_;
        numBlocks = numPairs + tail;
        numPairs = numBlocks / 2;
        hasTail = tail;
        tail = (numBlocks % 2 > 0);
    }
    if ((numLevels_ - 1) % 2 == 0) {
        BackwardStep(qGm, tmpQGm, tmpQGm[rightOffset], blockSize_, M_ / blockSize_, false);
    } else {
        BackwardStep(qGm, tmpQGm, bufferQGlobal_, blockSize_, M_ / blockSize_, false);
    }
}

extern "C" __global__ __aicore__ void tsqr(GM_ADDR a, GM_ADDR q, GM_ADDR r,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    TsqrKernel<float> tsqrKernel;
    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GM_ADDR user = GetUserWorkspace(workspace);

    tsqrKernel.Init(a, q, r, user, tilingData, &pipe);
    tsqrKernel.Process();
}
