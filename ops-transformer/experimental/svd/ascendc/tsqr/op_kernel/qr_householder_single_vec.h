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
 * \file qr_householder_single_vec.h
 * \brief
 */

#ifndef QR_HOUSEHOLDER_SINGLE_VEC_H_
#define QR_HOUSEHOLDER_SINGLE_VEC_H_

#include "kernel_operator.h"

using namespace AscendC;

static constexpr int32_t BUFFER_NUM = 2;
static constexpr int32_t FLOAT_MASK = 64;

#define ALIGN_DOWN(x, y) ((x) / (y) * (y))
#define ALIGN_UP(x, y) (((x) + (y) - 1) / (y) * (y))
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

class QRHouseholderSingleVec {
public:
    __aicore__ inline QRHouseholderSingleVec() {}
    __aicore__ inline void Init (GM_ADDR input_x, GM_ADDR output_q, GM_ADDR output_r, GM_ADDR workspace,
                                 const QrHouseholderTilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    TPipe* tpipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueX_;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQue_;
    TQue<TPosition::VECCALC, 1> reducedColQue_;
    TBuf<TPosition::VECCALC> vBuf_, tmpBuf_, wTBuf_, betaBuf_, betaTmpBuf_;
    GlobalTensor<float> inputXGm_, outputQGm_, outputRGm_, betaGm_, vGm_;

    int32_t batchSize_ = 0;
    int32_t mDim_ = 0;
    int32_t kDim_ = 0;
    int32_t ubSize_ = 0;
    int32_t maxNumElements_ = 0;
    float signX_ = 0.f;

    __aicore__ inline void ComputeR();
    __aicore__ inline void ComputeQ();
    __aicore__ inline void InitGm(const GlobalTensor<float> &outGm, int32_t matrixSize);
    __aicore__ inline void CopyInColumn(const GlobalTensor<float> &inputGm, int32_t gmOffset,
                                        const DataCopyParams &dataCopyParams, bool setSign);
    __aicore__ inline void ReducePaddedColumn(LocalTensor<float> &dstTensor, int32_t colSize);
    __aicore__ inline void CalculateTensorV(int32_t tensorSize);
    __aicore__ inline void CopyOutTensorV(int32_t gmOffset, int32_t tensorSize);
    __aicore__ inline void CalculateBeta(int32_t tensorSize, int32_t betaIdx);
    __aicore__ inline void CalculateWtTensor(const GlobalTensor<float> &inputGm,
                                             const DataCopyParams &dataCopyParams,
                                             int32_t tensorSize, int32_t startIdx);
    __aicore__ inline void UpdateR(int32_t rowLen, int32_t startIdx);
    __aicore__ inline void InitAndUpdateR();
    __aicore__ inline void CopyInRow(const GlobalTensor<float> &inputGm, int32_t gmOffset,
                                     const DataCopyParams &dataCopyParams, const DataCopyPadParams &dataCopyPadParams);
    __aicore__ inline void CopyOutRow(const GlobalTensor<float> &outGm, int32_t gmOffset,
                                      const DataCopyParams &dataCopyParams);
    __aicore__ inline void Dot(LocalTensor<float> &dst, const LocalTensor<float> &src1, const LocalTensor<float> &src2,
                               LocalTensor<float> &tmp, int32_t tensorSize);
    __aicore__ inline void MulByBlock(const LocalTensor<float> &dst, const LocalTensor<float> &src,
                                      const LocalTensor<float> &blockSrc, int32_t size);
    __aicore__ inline void ComputeQBlock(const LocalTensor<float> &vTensor, const LocalTensor<float>& q,
                                         int32_t betaIdx);
    __aicore__ inline void CopyUbToUb(const LocalTensor<float> &dstTensor, const LocalTensor<float>& src,
                                      int32_t size);
};

inline __aicore__ void QRHouseholderSingleVec::Init(GM_ADDR input_x, GM_ADDR output_q, GM_ADDR output_r,
                                                    GM_ADDR workspace, const QrHouseholderTilingData* tiling,
                                                    TPipe* pipe)
{
    set_atomic_none();
    tpipe_ = pipe;
    batchSize_ = tiling->batchSize;
    mDim_ = tiling->mDim;
    kDim_ = tiling->kDim;
    ubSize_ = tiling->ubSize;
    
    uint64_t inputSize = static_cast<uint64_t>(batchSize_ * mDim_ * kDim_);
    uint64_t rSize = static_cast<uint64_t>(batchSize_ * kDim_ * kDim_);

    inputXGm_.SetGlobalBuffer((__gm__ float*)input_x, inputSize);
    outputQGm_.SetGlobalBuffer((__gm__ float*)output_q, inputSize);
    outputRGm_.SetGlobalBuffer((__gm__ float*)output_r, rSize);
    __gm__ float* vPtr = (__gm__ float*)workspace;
    vGm_.SetGlobalBuffer(vPtr, mDim_ * kDim_);
    
    int32_t bufSize = ALIGN_UP(mDim_, 8) > FLOAT_MASK ? ALIGN_UP(mDim_, 8) : FLOAT_MASK;
    int32_t betaBufSize = ALIGN_UP(kDim_, 8) > FLOAT_MASK ? ALIGN_UP(kDim_, 8) : FLOAT_MASK;
    int32_t needMemoryforBuffers = 5 * bufSize * sizeof(float) + FLOAT_MASK * 4 * sizeof(float) +
                                   betaBufSize * sizeof(float);
    int32_t bufferCoeff = static_cast<int32_t>(sizeof(float)) * static_cast<int32_t>(BUFFER_NUM);
    maxNumElements_ = ALIGN_DOWN(((ubSize_ - needMemoryforBuffers) / bufferCoeff), 8);
    
    tpipe_->InitBuffer(inQueX_, BUFFER_NUM, maxNumElements_ * sizeof(float));
    tpipe_->InitBuffer(outQue_, BUFFER_NUM, bufSize * sizeof(float));
    tpipe_->InitBuffer(reducedColQue_, 1, bufSize * sizeof(float));
    tpipe_->InitBuffer(tmpBuf_, FLOAT_MASK * 2 * sizeof(float));
    tpipe_->InitBuffer(vBuf_, bufSize * sizeof(float));
    tpipe_->InitBuffer(wTBuf_, bufSize * sizeof(float));
    tpipe_->InitBuffer(betaBuf_, betaBufSize * sizeof(float));
    tpipe_->InitBuffer(betaTmpBuf_, FLOAT_MASK * 2 * sizeof(float));
}

inline __aicore__ void QRHouseholderSingleVec::Process()
{
    InitGm(outputRGm_, batchSize_ * kDim_ * kDim_);
    ComputeR();
    ComputeQ();
}

inline __aicore__ void QRHouseholderSingleVec::CopyInColumn(const GlobalTensor<float> &inputGm,
    int32_t gmOffset, const DataCopyParams &dataCopyParams, bool setSign)
{
    LocalTensor<float> inputTensor = inQueX_.AllocTensor<float>();
    DataCopyPad(inputTensor, inputGm[gmOffset], dataCopyParams, {true, 0, 7, 0});
    if (setSign) {
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        signX_ = inputTensor.GetValue(0) >= 0 ? 1.f : -1.f;
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
    }
    inQueX_.EnQue(inputTensor);
}

inline __aicore__ void QRHouseholderSingleVec::ReducePaddedColumn(LocalTensor<float> &dstTensor, int32_t colSize)
{
    LocalTensor<float> paddedCol = inQueX_.DeQue<float>();
    uint8_t maxRepeatTime = ALIGN_DOWN(255, 8);
    int32_t numIters = colSize / static_cast<int32_t>(maxRepeatTime);
    int32_t tailRepeatTime = colSize % static_cast<int32_t>(maxRepeatTime);
    for (int32_t i = 0; i < numIters; i++) {
        WholeReduceSum(dstTensor[i * maxRepeatTime], paddedCol[i * maxRepeatTime * 8], 8, maxRepeatTime, 1, 1, 1);
    }
    if (tailRepeatTime > 0) {
        WholeReduceSum(dstTensor[numIters * maxRepeatTime], paddedCol[numIters * maxRepeatTime * 8], 8,
                       static_cast<uint8_t>(tailRepeatTime), 1, 1, 1);
    }
    inQueX_.FreeTensor(paddedCol);
    PipeBarrier<PIPE_V>();
}

inline __aicore__ void QRHouseholderSingleVec::CopyUbToUb(const LocalTensor<float> &dstTensor,
    const LocalTensor<float>& src, int32_t size)
{
    uint8_t maxRepeatTime = 255;
    int32_t repeatTime = size / FLOAT_MASK;
    int32_t tailElements = size % FLOAT_MASK;
    int32_t numIters = repeatTime / static_cast<int32_t>(maxRepeatTime);
    uint8_t tailRepeatTime = repeatTime % static_cast<int32_t>(maxRepeatTime);

    for (int32_t i = 0; i < numIters; i++) {
        int32_t offset = i * FLOAT_MASK * static_cast<int32_t>(maxRepeatTime);
        Copy(dstTensor[offset], src[offset], FLOAT_MASK, maxRepeatTime, {1, 1, 8, 8});
    }
    if (tailRepeatTime > 0) {
        int32_t offset = numIters * FLOAT_MASK * static_cast<int32_t>(maxRepeatTime);
        Copy(dstTensor[offset], src[offset], FLOAT_MASK, tailRepeatTime, {1, 1, 8, 8});
    }
    if (tailElements > 0) {
        int32_t offset = repeatTime * FLOAT_MASK;
        Copy(dstTensor[offset], src[offset], tailElements, 1, {1, 1, 8, 8});
    }
    PipeBarrier<PIPE_V>();
}

inline __aicore__ void QRHouseholderSingleVec::CalculateTensorV(int32_t tensorSize)
{
    LocalTensor<float> reducedTensor = reducedColQue_.DeQue<float>();
    LocalTensor<float> vTensor = vBuf_.Get<float>();
    LocalTensor<float> vOutTensor = outQue_.AllocTensor<float>();
    LocalTensor<float> tmpTensor = tmpBuf_.Get<float>();
    Dot(tmpTensor, reducedTensor, reducedTensor, vOutTensor, tensorSize);
    Sqrt(tmpTensor, tmpTensor, 1);
    CopyUbToUb(vTensor, reducedTensor, tensorSize);
    Muls(tmpTensor, tmpTensor, signX_, 1);
    PipeBarrier<PIPE_V>();
    Add(vTensor, vTensor, tmpTensor, 1);
    PipeBarrier<PIPE_V>();
    CopyUbToUb(vOutTensor, vTensor, tensorSize);
    outQue_.EnQue(vOutTensor);
    reducedColQue_.EnQue(reducedTensor);
}

inline __aicore__ void QRHouseholderSingleVec::CopyOutTensorV(int32_t gmOffset, int32_t tensorSize)
{
    LocalTensor<float> vTensor = outQue_.DeQue<float>();
    DataCopyPad(vGm_[gmOffset], vTensor, {1, static_cast<uint16_t>(tensorSize * sizeof(float)), 0, 0});
    outQue_.FreeTensor(vTensor);
}

inline __aicore__ void QRHouseholderSingleVec::CalculateBeta(int32_t tensorSize, int32_t betaIdx) {
    LocalTensor<float> vTensor = vBuf_.Get<float>();
    LocalTensor<float> betaTmpTensor = betaTmpBuf_.GetWithOffset<float>(8, (betaIdx % 8) * 8 * sizeof(float));
    LocalTensor<float> dotRes = tmpBuf_.Get<float>();
    LocalTensor<float> tmpTensor = wTBuf_.Get<float>();

    if (betaIdx % 8 == 0 && betaIdx != 0) {
        LocalTensor<float> betaTensor = betaBuf_.Get<float>();
        WholeReduceMax(betaTensor[betaIdx - 8], betaTmpTensor, 8, 8, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
    }

    Dot(dotRes, vTensor, vTensor, tmpTensor, tensorSize);
    Brcb(dotRes, dotRes, 1, {1, 8});
    Duplicate(tmpTensor, 2.f, 8);
    PipeBarrier<PIPE_V>();
    Div(betaTmpTensor, tmpTensor, dotRes, 8);

    if (betaIdx == (kDim_ - 1)) {
        LocalTensor<float> betaTensor = betaBuf_.Get<float>();
        betaTmpTensor = betaTmpBuf_.Get<float>();
        int32_t offset = ALIGN_DOWN(betaIdx, 8);
        uint8_t numRepeat = static_cast<uint8_t>(kDim_ - offset);
        PipeBarrier<PIPE_V>();
        WholeReduceMax(betaTensor[offset], betaTmpTensor, 8, numRepeat, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
    }
    PipeBarrier<PIPE_V>();
}

inline __aicore__ void QRHouseholderSingleVec::CalculateWtTensor(const GlobalTensor<float> &inputGm,
                                                                 const DataCopyParams &dataCopyParams,
                                                                 int32_t tensorSize, int32_t startIdx)
{
    LocalTensor<float> wtTensor = wTBuf_.Get<float>();
    LocalTensor<float> vTensor = vBuf_.Get<float>();
    LocalTensor<float> reducedTensor = reducedColQue_.DeQue<float>();
    LocalTensor<float> dotResult = tmpBuf_.Get<float>();
    LocalTensor<float> betaTensor = betaTmpBuf_.GetWithOffset<float>(8, (startIdx % 8) * 8 * sizeof(float));
    int32_t step = 0;
    int32_t dotResIdx = 1;

    Dot(dotResult, reducedTensor, vTensor, reducedTensor, tensorSize);
    Brcb(dotResult, dotResult, 1, {1, 8});
    for (int32_t i = startIdx + 1; i < kDim_; i++, dotResIdx++) {
        if (dotResIdx == 8) {
            dotResIdx = 0;
            LocalTensor<float> wtPaddedTensor = tmpBuf_.Get<float>();
            PipeBarrier<PIPE_V>();
            WholeReduceMax(wtTensor[step * 8], wtPaddedTensor, 8, 8, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
            step++;
        }
        int32_t gmOffset = kDim_ * startIdx + i;
        CopyInColumn(inputGm, gmOffset, dataCopyParams, false);
        ReducePaddedColumn(reducedTensor, tensorSize);
        LocalTensor<float> dotResult = tmpBuf_.GetWithOffset<float>(64, dotResIdx * 8 * sizeof(float));
        Dot(dotResult, reducedTensor, vTensor, reducedTensor, tensorSize);
        Brcb(dotResult, dotResult, 1, {1, 8});
    }
    if (dotResIdx > 0) {
        LocalTensor<float> wtPaddedTensor = tmpBuf_.Get<float>();
        PipeBarrier<PIPE_V>();
        WholeReduceMax(wtTensor[step * 8], wtPaddedTensor, 8, dotResIdx, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
    }
    PipeBarrier<PIPE_V>();
    MulByBlock(wtTensor, wtTensor, betaTensor, tensorSize);
    reducedColQue_.FreeTensor(reducedTensor);
}

inline __aicore__ void QRHouseholderSingleVec::CopyInRow(const GlobalTensor<float> &inputGm, int32_t gmOffset,
                                                        const DataCopyParams &dataCopyParams,
                                                        const DataCopyPadParams &dataCopyPadParams)
{
    LocalTensor<float> inRow = inQueX_.AllocTensor<float>();
    DataCopyPad(inRow, inputGm[gmOffset], dataCopyParams, dataCopyPadParams);
    inQueX_.EnQue(inRow);
}

inline __aicore__ void QRHouseholderSingleVec::CopyOutRow(const GlobalTensor<float> &outGm,
    int32_t gmOffset, const DataCopyParams &dataCopyParams)
{
    LocalTensor<float> resultRow = outQue_.DeQue<float>();
    DataCopyPad(outGm[gmOffset], resultRow, dataCopyParams);
    outQue_.FreeTensor(resultRow);
}

inline __aicore__ void QRHouseholderSingleVec::InitAndUpdateR()
{
    LocalTensor<float> wtTensor = wTBuf_.Get<float>();
    LocalTensor<float> vTensor = vBuf_.Get<float>();
    LocalTensor<float> bcastTensor = tmpBuf_.Get<float>();
    uint8_t rightPad = static_cast<uint8_t>(kDim_ % 8);
    DataCopyParams dataCopyParams = {1, static_cast<uint16_t>(kDim_ * sizeof(float)), 0, 0};
    DataCopyPadParams dataCopyPadParams = {rightPad != 0, 0, rightPad, 0};
    CopyInRow(inputXGm_, 0, dataCopyParams, dataCopyPadParams);
    Brcb(bcastTensor, vTensor, 1, {1, 8});
    PipeBarrier<PIPE_V>();
    LocalTensor<float> row = inQueX_.DeQue<float>();
    LocalTensor<float> resultRow = outQue_.AllocTensor<float>();
    MulByBlock(resultRow, wtTensor, bcastTensor, kDim_);
    PipeBarrier<PIPE_V>();
    Sub(resultRow, row, resultRow, kDim_);
    inQueX_.FreeTensor(row);
    outQue_.EnQue(resultRow);
    CopyOutRow(outputRGm_, 0, dataCopyParams);
    int32_t alignedRowLen = ALIGN_UP(kDim_, 8);
    int32_t ubNumRows = maxNumElements_ / alignedRowLen;
    int32_t numRows = (mDim_ - 1);
    int32_t numIters = CEIL_DIV(numRows, ubNumRows);
    dataCopyParams = {static_cast<uint16_t>(ubNumRows), static_cast<uint16_t>(kDim_ * sizeof(float)), 0, 0};
    int32_t processedRowIdx = 1;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    LocalTensor<float> tmp = outQue_.AllocTensor<float>();
    for (int32_t i = 0; i < numIters; i++, numRows -= ubNumRows) {
        ubNumRows = ubNumRows < numRows ? ubNumRows : numRows;
        dataCopyParams = {static_cast<uint16_t>(ubNumRows), static_cast<uint16_t>(kDim_ * sizeof(float)), 0, 0};
        int32_t gmOffset = processedRowIdx * kDim_;
        CopyInRow(inputXGm_, gmOffset, dataCopyParams, dataCopyPadParams);
        LocalTensor<float> rows = inQueX_.DeQue<float>();
        for (int32_t j = 0; j < ubNumRows; j++, processedRowIdx++) {
            int32_t bcastId = processedRowIdx % 8;
            if (bcastId == 0) {
                Brcb(bcastTensor, vTensor[processedRowIdx], 1, {1, 8});
                PipeBarrier<PIPE_V>();
            }
            MulByBlock(tmp, wtTensor, bcastTensor[bcastId * 8], kDim_);
            Sub(rows[j * alignedRowLen], rows[j * alignedRowLen], tmp, kDim_);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(outputQGm_[gmOffset], rows, dataCopyParams);
        inQueX_.FreeTensor(rows);
    }
    outQue_.FreeTensor(tmp);
}

inline __aicore__ void QRHouseholderSingleVec::UpdateR(int32_t rowLen, int32_t startIdx)
{
    LocalTensor<float> wtTensor = wTBuf_.Get<float>();
    LocalTensor<float> vTensor = vBuf_.Get<float>();
    LocalTensor<float> bcastTensor = tmpBuf_.Get<float>();
    uint8_t rightPad = static_cast<uint8_t>(rowLen % 8);
    DataCopyParams dataCopyParams = {1, static_cast<uint16_t>(rowLen * sizeof(float)), 0, 0};
    DataCopyPadParams dataCopyPadParams = {rightPad != 0, 0, rightPad, 0};
    CopyInRow(outputQGm_, startIdx * kDim_ + startIdx, dataCopyParams, dataCopyPadParams);
    Brcb(bcastTensor, vTensor, 1, {1, 8});
    PipeBarrier<PIPE_V>();
    Muls(bcastTensor, bcastTensor, -1.f, FLOAT_MASK);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> row = inQueX_.DeQue<float>();
    LocalTensor<float> resultRow = outQue_.AllocTensor<float>();
    MulByBlock(resultRow, wtTensor, bcastTensor, rowLen);
    PipeBarrier<PIPE_V>();
    Add(resultRow, row, resultRow, rowLen);
    inQueX_.FreeTensor(row);
    outQue_.EnQue(resultRow);
    CopyOutRow(outputRGm_, startIdx * kDim_ + startIdx, dataCopyParams);
    int32_t alignedRowLen = ALIGN_UP(rowLen, 8);
    int32_t ubNumRows = maxNumElements_ / alignedRowLen;
    int32_t numRows = mDim_ - (startIdx + 1);
    int32_t numIters = CEIL_DIV(numRows, ubNumRows);
    int32_t processedRowIdx = 1;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetAtomicAdd<float>();
    for (int32_t i = 0; i < numIters; i++, numRows -= ubNumRows) {
        ubNumRows = ubNumRows < numRows ? ubNumRows : numRows;
        int32_t gmOffset = startIdx * kDim_ + processedRowIdx * kDim_ + startIdx;
        DataCopyParams dataCopyOutParams = {static_cast<uint16_t>(ubNumRows), static_cast<uint16_t>(
                        rowLen * sizeof(float)), 0, static_cast<uint16_t>(startIdx * sizeof(float))};
        LocalTensor<float> rows = inQueX_.AllocTensor<float>();
        for (int32_t j = 0; j < ubNumRows; j++, processedRowIdx++) {
            int32_t bcastId = processedRowIdx % 8;
            if (bcastId == 0) {
                Brcb(bcastTensor, vTensor[processedRowIdx], 1, {1, 8});
                PipeBarrier<PIPE_V>();
                Muls(bcastTensor, bcastTensor, -1.f, FLOAT_MASK);
                PipeBarrier<PIPE_V>();
            }
            int32_t dstOffset = j * alignedRowLen;
            MulByBlock(rows[dstOffset], wtTensor, bcastTensor[bcastId * 8], rowLen);
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        DataCopyPad(outputQGm_[gmOffset], rows, dataCopyOutParams);
        inQueX_.FreeTensor(rows);
    }
    SetAtomicNone();
}
    

inline __aicore__ void QRHouseholderSingleVec::ComputeR()
{
    DataCopyParams dataCopyParams = {static_cast<uint16_t>(mDim_), static_cast<uint16_t>(sizeof(float)),
                                     static_cast<uint16_t>((kDim_ - 1) * sizeof(float)), 0};
    LocalTensor<float> reducedTensor = reducedColQue_.AllocTensor<float>();
    CopyInColumn(inputXGm_, 0, dataCopyParams, true);
    ReducePaddedColumn(reducedTensor, mDim_);
    reducedColQue_.EnQue(reducedTensor);
    CalculateTensorV(mDim_);
    CopyOutTensorV(0, mDim_);
    CalculateBeta(mDim_, 0);
    CalculateWtTensor(inputXGm_, dataCopyParams, mDim_, 0);
    InitAndUpdateR();
    PipeBarrier<PIPE_ALL>();

    for (int32_t col = 1; col < kDim_; col++) {
        int32_t colLen = mDim_ - col;
        DataCopyParams dataCopyParams = {
            static_cast<uint16_t>(colLen),
            static_cast<uint16_t>(sizeof(float)),
            static_cast<uint16_t>((kDim_ - 1) * sizeof(float)),
            0
        };
        LocalTensor<float> reducedTensor = reducedColQue_.AllocTensor<float>();
        CopyInColumn(outputQGm_, col * kDim_ + col, dataCopyParams, true);
        ReducePaddedColumn(reducedTensor, colLen);
        reducedColQue_.EnQue(reducedTensor);
        CalculateTensorV(colLen);
        CopyOutTensorV(col * mDim_ + col, colLen);
        CalculateBeta(colLen, col);
        CalculateWtTensor(outputQGm_, dataCopyParams, colLen, col);
        UpdateR(kDim_ - col, col);
        PipeBarrier<PIPE_ALL>();
    }
}

inline __aicore__ void QRHouseholderSingleVec::ComputeQBlock(const LocalTensor<float> &vTensor,
    const LocalTensor<float>& q, int32_t betaIdx)
{
    LocalTensor<float> beta = betaBuf_.Get<float>();
    LocalTensor<float> dotRes = tmpBuf_.Get<float>();
    LocalTensor<float> tmpTensor = wTBuf_.Get<float>();
    Duplicate(vTensor, 0.f, betaIdx);
    PipeBarrier<PIPE_V>();
    Dot(dotRes, vTensor, q, tmpTensor, mDim_);
    Brcb(dotRes, dotRes, 1, {1, 8});
    PipeBarrier<PIPE_V>();
    Muls(dotRes, dotRes, beta.GetValue(betaIdx), 8);
    PipeBarrier<PIPE_V>();
    MulByBlock(tmpTensor, vTensor, dotRes, mDim_);
    Sub(q, q, tmpTensor, mDim_);
    PipeBarrier<PIPE_V>();
}

inline __aicore__ void QRHouseholderSingleVec::ComputeQ()
{
    uint8_t rightPad = static_cast<uint8_t>(mDim_ % 8);
    int32_t alignedRowLen = ALIGN_UP(mDim_, 8);
    int32_t ubNumRows = maxNumElements_ / alignedRowLen;
    ubNumRows = ubNumRows > kDim_ ? kDim_ : ubNumRows;
    DataCopyParams dataCopyParams = {static_cast<uint16_t>(ubNumRows), static_cast<uint16_t>(mDim_ * sizeof(float)),
                                     0, 0};
    DataCopyParams dataCopyOutParams = {1, static_cast<uint16_t>(mDim_ * sizeof(float)), 0, 0};
    DataCopyPadParams dataCopyPadParams = {rightPad != 0, 0, rightPad, 0};
    CopyInRow(vGm_, 0, dataCopyParams, dataCopyPadParams);
    LocalTensor<float> vTensor = inQueX_.DeQue<float>();
    for (int32_t i = 0; i < kDim_; i++) {
        LocalTensor<float> q = outQue_.AllocTensor<float>();
        uint64_t maskValue = (uint64_t)1 << static_cast<uint64_t>(i % FLOAT_MASK);
        int32_t qOffset = ALIGN_DOWN(i, FLOAT_MASK);
        uint64_t mask[1] = {maskValue};
        Duplicate(q, 0.f, mDim_);
        PipeBarrier<PIPE_V>();
        Duplicate(q[qOffset], 1.f, mask, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        if (ubNumRows <= i) {
            int32_t numRows = i + 1;
            int32_t numIters = CEIL_DIV(numRows, ubNumRows);
            int32_t procRows = ubNumRows;
            int32_t betaIdx = i;
            for (int32_t j = 0; j < numIters; j++, numRows -= procRows) {
                procRows = procRows < numRows ? procRows : numRows;
                int32_t gmOffset = (numRows - procRows) * mDim_;
                dataCopyParams = {static_cast<uint16_t>(procRows), static_cast<uint16_t>(mDim_ * sizeof(float)),
                                  0, 0};
                CopyInRow(vGm_, gmOffset, dataCopyParams, dataCopyPadParams);
                vTensor = inQueX_.DeQue<float>();
                for (int32_t k = procRows; k > 0; k--, betaIdx--) {
                    ComputeQBlock(vTensor[(k - 1) * alignedRowLen], q, betaIdx);
                }
                inQueX_.FreeTensor(vTensor);
            }
        } else {
            for (int32_t k = i; k > -1; k--) {
                ComputeQBlock(vTensor[k * alignedRowLen], q, k);
            }
            if (ubNumRows == (i + 1)) {
                inQueX_.FreeTensor(vTensor);
            }
        }
        outQue_.EnQue(q);
        CopyOutRow(outputQGm_, i * mDim_, dataCopyOutParams);
    }
}

inline __aicore__ void QRHouseholderSingleVec::Dot(LocalTensor<float> &dst,
                                                const LocalTensor<float> &src1,
                                                const LocalTensor<float> &src2,
                                                LocalTensor<float> &tmp,
                                                int32_t tensorSize)
{
    uint8_t maxRepeatTime = 255;
    int32_t repeatTime = tensorSize / FLOAT_MASK;
    int32_t tailElements = tensorSize % FLOAT_MASK;
    int32_t numIters = repeatTime / static_cast<int32_t>(maxRepeatTime);
    uint8_t tailRepeatTime = repeatTime % static_cast<int32_t>(maxRepeatTime);

    Duplicate(dst, 0.f, FLOAT_MASK);
    Mul(tmp, src1, src2, tensorSize);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 0; i < numIters; i++) {
        Add(dst, tmp[i * FLOAT_MASK * static_cast<int32_t>(maxRepeatTime)], dst, FLOAT_MASK, maxRepeatTime,
            {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (tailRepeatTime > 0) {
        Add(dst, tmp[numIters * FLOAT_MASK * static_cast<int32_t>(maxRepeatTime)], dst, FLOAT_MASK, tailRepeatTime,
            {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (tailElements > 0) {
        Add(dst, tmp[repeatTime * FLOAT_MASK], dst, tailElements);
        PipeBarrier<PIPE_V>();
    }
    WholeReduceSum(dst, dst, FLOAT_MASK, 1, 1, 1, 8);
    PipeBarrier<PIPE_V>();
}

inline __aicore__ void QRHouseholderSingleVec::MulByBlock(const LocalTensor<float> &dst,
                                                        const LocalTensor<float> &src,
                                                        const LocalTensor<float> &blockSrc,
                                                        int32_t size)
{
    uint8_t repeat = static_cast<uint8_t>(size / FLOAT_MASK);
    uint8_t tailRepeat = static_cast<uint8_t>((size % FLOAT_MASK) / 8);
    int32_t tail = (size % FLOAT_MASK) % 8;
    if (repeat > 0) {
        Mul(dst, src, blockSrc, FLOAT_MASK, repeat, {1, 1, 0, 8, 8, 0});
    }
    if (tailRepeat > 0) {
        int32_t offset = repeat * FLOAT_MASK;
        Mul(dst[offset], src[offset], blockSrc, 8, tailRepeat, {1, 1, 0, 1, 1, 0});
    }
    if (tail > 0) {
        int32_t offset = repeat * FLOAT_MASK + tailRepeat * 8;
        Mul(dst[offset], src[offset], blockSrc, tail);
    }
    PipeBarrier<PIPE_V>();
}

inline __aicore__ void QRHouseholderSingleVec::InitGm(const GlobalTensor<float> &outGm, int32_t matrixSize)
{
    int32_t maxElements = 16384;
    maxElements = maxElements > maxNumElements_ ? maxNumElements_ : maxElements;
    int32_t numIters = matrixSize / maxElements;
    int32_t tail = matrixSize % maxElements;
    int32_t offset = 0;
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    LocalTensor<float> tensor = inQueX_.AllocTensor<float>();

    Duplicate(tensor, 0.f, maxElements);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    for (int32_t i = 0; i < numIters; i++, offset += maxElements) {
        DataCopy(outGm[offset], tensor, maxElements);
    }
    if (tail > 0) {
        DataCopyPad(outGm[offset], tensor, {1, static_cast<uint16_t>(tail * sizeof(float)), 0, 0});
    }
    inQueX_.FreeTensor(tensor);

    // This sync point is neccessary to prevent sporadic accuracy issues in R matrix
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

#endif // QR_HOUSEHOLDER_SINGLE_VEC_H_
