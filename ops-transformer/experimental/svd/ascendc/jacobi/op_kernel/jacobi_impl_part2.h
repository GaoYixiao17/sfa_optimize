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
 * \file jacobi_impl_part2.h
 * \brief
 */

#ifndef JACOBI_IMPL_PART2_H
#define JACOBI_IMPL_PART2_H

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::CalcNorms(
        const LocalTensor<float>& norms1, const LocalTensor<float>& norms2, const LocalTensor<float>& bucket1,
        const LocalTensor<float>& bucket2, const RotateMaskInfo& maskInfo, const LocalTensor<float>& tmpBucket1,
        const LocalTensor<float>& tmpBucket2, const LocalTensor<float>& dotProductTmp1,
        const LocalTensor<float>& dotProductTmp2)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
    LocalTensor<float>& tmpNorm1 = tmpBlock18;
    LocalTensor<float>& tmpNorm2 = tmpBlock17;
    Duplicate(dotProductTmp1, 0.f, 2 * rowMask * maxBucketSize);
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.leftBucketSize; ++rowIdx) {
        Mul(tmpBucket1[rowIdx * nSizeAligned], bucket1[rowIdx * nSizeAligned],
            bucket1[rowIdx * nSizeAligned], rowMask, rowRepeatsNum,
            { 1, 1, 1, repeatStride, repeatStride, repeatStride });
    }
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
        Mul(tmpBucket2[rowIdx * nSizeAligned], bucket2[rowIdx * nSizeAligned],
            bucket2[rowIdx * nSizeAligned], rowMask, rowRepeatsNum,
            { 1, 1, 1, repeatStride, repeatStride, repeatStride });
    }
    PipeBarrier<PIPE_V>();
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.leftBucketSize; ++rowIdx) {
        Add(dotProductTmp1[rowIdx * rowMask], tmpBucket1[rowIdx * nSizeAligned], dotProductTmp1[rowIdx * rowMask],
                rowMask, rowRepeatsNum, { 1, 1, 1, 0, repeatStride, 0 });
    }
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
        Add(dotProductTmp2[rowIdx * rowMask], tmpBucket2[rowIdx * nSizeAligned], dotProductTmp2[rowIdx * rowMask],
                rowMask, rowRepeatsNum, { 1, 1, 1, 0, repeatStride, 0 });
    }
    PipeBarrier<PIPE_V>();

    WholeReduceSum(tmpNorm1, dotProductTmp1, rowMask, static_cast<uint8_t>(maskInfo.leftBucketSize), 1, 1,
            repeatStride);
    WholeReduceSum(tmpNorm2, dotProductTmp2, rowMask, static_cast<uint8_t>(maskInfo.rightBucketSize), 1, 1,
            repeatStride);
    PipeBarrier<PIPE_V>();
    Sqrt(norms1, tmpNorm1, SMNE);
    Sqrt(norms2, tmpNorm2, SMNE);
    PipeBarrier<PIPE_V>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool BUCKET_1_TARGET, bool NEED_CALCULATE_RIGHT_NORM>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::CalculateDotProducts(
        const LocalTensor<float>& sqNormsBucket1, const LocalTensor<float>& sqNormsBucket2,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& bucket1, const LocalTensor<float>& norms1,
        const LocalTensor<float>& bucket2, const LocalTensor<float>& norms2, const RotateMaskInfo& maskInfo)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    LocalTensor<float>& tmpBucket1 = tmpMemSpace;
    LocalTensor<float> tmpBucket2 = tmpMemSpace[maxBucketSize * nSizeAligned];
    LocalTensor<float>& dotProductTmp1 = tmpMaskSpace;
    LocalTensor<float> dotProductTmp2 = tmpMaskSpace[rowMask * maxBucketSize];
    const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;

    #ifdef RUNTIME_NORM_GENERATION
        CalcNorms(norms1, norms2, bucket1, bucket2, maskInfo, tmpBucket1, tmpBucket2, dotProductTmp1, dotProductTmp2);
    #endif
    if constexpr (NEED_CALCULATE_RIGHT_NORM) {
        CalculateRightNormAndDotProductForDoubleBucketsCase(dotProducts, norms2, bucket1, bucket2, maskInfo,
                tmpBucket1, tmpBucket2, dotProductTmp1, dotProductTmp2);
    } else {
        CalculateDotProductForDoubleBucketCase(dotProducts, bucket1, bucket2, maskInfo, tmpBucket1, dotProductTmp1);
    }

    PipeBarrier<PIPE_V>();
    LocalTensor<float>& broadcastedNorms = dotProductTmp1;
    LocalTensor<float> broadCastedNormsWithZero = dotProductTmp1[rowMask];
    Mul(sqNormsBucket1, norms1, norms1, SMNE);
    Mul(sqNormsBucket2, norms2, norms2, SMNE);

    PipeBarrier<PIPE_V>();
    if constexpr (BUCKET_1_TARGET) {
        GatherNormVector(sqNormsBucket2, sqNormsBucket2, maskInfo.rightMaskU64Val, broadCastedNormsWithZero,
            broadcastedNorms);
    } else {
        GatherNormVector(sqNormsBucket1, sqNormsBucket1, maskInfo.leftMaskU64Val, broadCastedNormsWithZero,
            broadcastedNorms);
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::
        CalculateRightNormAndDotProductForDoubleBucketsCase(const LocalTensor<float>& dstDotProduct,
        const LocalTensor<float>& dstRightNorms, const LocalTensor<float>& bucket1, const LocalTensor<float>& bucket2,
        const RotateMaskInfo& maskInfo, const LocalTensor<float>& tmpBucket1, const LocalTensor<float>& tmpBucket2,
        const LocalTensor<float>& dotProductTmp1, const LocalTensor<float>& dotProductTmp2)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
    Duplicate(dotProductTmp1, 0.f, rowMask * maskInfo.numPairs);
    Duplicate(dotProductTmp2, 0.f, rowMask * maskInfo.rightBucketSize);
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
        uint16_t leftRowIdx = maskInfo.leftRowsIds[rowIdx];
        uint16_t rightRowIdx = maskInfo.rightRowsIds[rowIdx];
        Mul(tmpBucket1[rowIdx * nSizeAligned], bucket1[leftRowIdx * nSizeAligned],
            bucket2[rightRowIdx * nSizeAligned], rowMask, rowRepeatsNum,
            { 1, 1, 1, repeatStride, repeatStride, repeatStride });
    }
    for (uint8_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
        Mul(tmpBucket2[rowIdx * nSizeAligned], bucket2[rowIdx * nSizeAligned],
            bucket2[rowIdx * nSizeAligned], rowMask, rowRepeatsNum,
            { 1, 1, 1, repeatStride, repeatStride, repeatStride });
    }
    PipeBarrier<PIPE_V>();
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
        Add(dotProductTmp1[rowIdx * rowMask], tmpBucket1[rowIdx * nSizeAligned], dotProductTmp1[rowIdx * rowMask],
                rowMask, rowRepeatsNum, { 1, 1, 1, 0, repeatStride, 0 });
    }
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.rightBucketSize; ++rowIdx) {
        Add(dotProductTmp2[rowIdx * rowMask], tmpBucket2[rowIdx * nSizeAligned], dotProductTmp2[rowIdx * rowMask],
                rowMask, rowRepeatsNum, { 1, 1, 1, 0, repeatStride, 0 });
    }
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dstDotProduct, dotProductTmp1, rowMask, static_cast<uint8_t>(maskInfo.numPairs), 1, 1,
            repeatStride);
    WholeReduceSum(dstRightNorms, dotProductTmp2, rowMask, static_cast<uint8_t>(maskInfo.rightBucketSize), 1, 1,
            repeatStride);
    PipeBarrier<PIPE_V>();
    Sqrt(dstRightNorms, dstRightNorms, SMNE);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::
CalculateDotProductForDoubleBucketCase(const LocalTensor<float>& dstDotProduct, const LocalTensor<float>& bucket1,
        const LocalTensor<float>& bucket2, const RotateMaskInfo& maskInfo, const LocalTensor<float>& tmpBucket1,
        const LocalTensor<float>& dotProductTmp1)
{
    const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
    Duplicate(dotProductTmp1, 0.f, rowMask * maskInfo.numPairs);
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
        uint16_t leftRowIdx = maskInfo.leftRowsIds[rowIdx];
        uint16_t rightRowIdx = maskInfo.rightRowsIds[rowIdx];
        Mul(tmpBucket1[rowIdx * nSizeAligned], bucket1[leftRowIdx * nSizeAligned],
            bucket2[rightRowIdx * nSizeAligned], rowMask, rowRepeatsNum,
            { 1, 1, 1, repeatStride, repeatStride, repeatStride });
    }
    PipeBarrier<PIPE_V>();
    for (uint16_t rowIdx = 0; rowIdx < maskInfo.numPairs; ++rowIdx) {
        Add(dotProductTmp1[rowIdx * rowMask], tmpBucket1[rowIdx * nSizeAligned], dotProductTmp1[rowIdx * rowMask],
                rowMask, rowRepeatsNum, { 1, 1, 1, 0, repeatStride, 0 });
    }
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dstDotProduct, dotProductTmp1, rowMask, static_cast<uint8_t>(maskInfo.numPairs), 1, 1,
            repeatStride);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::CalculateNormOfBucket(
        const LocalTensor<float>& dstNormVector, const LocalTensor<float>& bucket, const LocalTensor<float>& tmpBucket,
        const LocalTensor<float>& dotProductTmp, const uint8_t(&rowIds)[sizeof(uint64_t)], const uint32_t bucketSize)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;

    Duplicate(dotProductTmp, 0.f, rowMask * bucketSize);
    for (uint16_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
        uint16_t rowId = rowIds[rowIdx];
        Mul(tmpBucket[rowIdx * nSizeAligned], bucket[rowId * nSizeAligned],
            bucket[rowId * nSizeAligned], rowMask, rowRepeatsNum,
            { 1, 1, 1, repeatStride, repeatStride, repeatStride });
    }
    PipeBarrier<PIPE_V>();
    for (uint16_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
        Add(dotProductTmp[rowIdx * rowMask], tmpBucket[rowIdx * nSizeAligned], dotProductTmp[rowIdx * rowMask],
                rowMask, rowRepeatsNum, { 1, 1, 1, 0, repeatStride, 0 });
    }
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dstNormVector, dotProductTmp, rowMask, static_cast<uint8_t>(bucketSize), 1, 1, repeatStride);
    PipeBarrier<PIPE_V>();
    Sqrt(dstNormVector, dstNormVector, SMNE);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::GatherNormVector(
        const LocalTensor<float>& dstNormVector, const LocalTensor<float>& srcNormVector, const uint64_t mask,
        const LocalTensor<float>& workLocal1, const LocalTensor<float>& workLocal2)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    Duplicate(workLocal1, 0.f, rowMask);

    PipeBarrier<PIPE_V>();

    Copy(workLocal2, srcNormVector, SMNE, SMNE, { 1, 1, 1, 0 });
    PipeBarrier<PIPE_V>();
    uint64_t mask_[2] = { mask, 0 };
    Copy(workLocal1, workLocal2, mask_, 1, { 1, 1, 1, 1 });
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dstNormVector, workLocal1, SMNE, static_cast<uint8_t>(SMNE), 1, 1, 1);
    PipeBarrier<PIPE_V>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool NEED_CALCULATE_NORM>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::
CalculateDotProductsForSelfBucket(const LocalTensor<float>& sqNorms1, const LocalTensor<float>& sqNorms2,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& normVector, const LocalTensor<float>& bucket,
        const uint32_t bucketSize, const RotateMaskInfo& maskInfo)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();

    LocalTensor<float>& tmpBucket1 = tmpMemSpace;
    LocalTensor<float> tmpBucket2 = tmpMemSpace[maxBucketSize * nSizeAligned];
    LocalTensor<float>& dotProductTmp1 = tmpMaskSpace;
    LocalTensor<float> dotProductTmp2 = tmpMaskSpace[maxBucketSize * rowMask];

    const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
    #ifdef RUNTIME_NORM_GENERATION
    {
        Duplicate(dotProductTmp1, 0.f, rowMask * bucketSize);
        const uint8_t repeatStride = (rowMask * sizeof(float)) / 32;
        for (uint16_t rowID = 0; rowID < bucketSize; ++rowID) {
            Mul(tmpBucket1[rowID * nSizeAligned], bucket[rowID * nSizeAligned], bucket[rowID * nSizeAligned],
                    rowMask, rowRepeatsNum, { 1, 1, 1, repeatStride, repeatStride, repeatStride });
        }

        PipeBarrier<PIPE_V>();
        for (uint16_t rowID = 0; rowID < bucketSize; ++rowID) {
            Add(dotProductTmp1[rowID * rowMask], tmpBucket1[rowID * nSizeAligned], dotProductTmp1[rowID * rowMask],
                    rowMask, rowRepeatsNum, { 1, 1, 1, 0, repeatStride, 0 });
        }
        PipeBarrier<PIPE_V>();
        WholeReduceSum(normVector, dotProductTmp1, rowMask, static_cast<uint8_t>(bucketSize), 1, 1, repeatStride);
        PipeBarrier<PIPE_V>();
        Sqrt(normVector, normVector, SMNE);
        PipeBarrier<PIPE_V>();
    }
    #endif

    if constexpr (NEED_CALCULATE_NORM) {
        CalculateRightNormAndDotProductForDoubleBucketsCase(dotProducts, normVector, bucket, bucket, maskInfo,
                tmpBucket1, tmpBucket2, dotProductTmp1, dotProductTmp2);
    } else {
        CalculateDotProductForDoubleBucketCase(dotProducts, bucket, bucket, maskInfo, tmpBucket1, dotProductTmp1);
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<float>& broadcastedNorms = dotProductTmp1;
    LocalTensor<float> sqNorms1BroadCasted = dotProductTmp1[rowMask];

    GatherNormsVectorToLeftRightNorms(sqNorms1, sqNorms2, normVector, maskInfo, broadcastedNorms, sqNorms1BroadCasted);
    PipeBarrier<PIPE_V>();
    Mul(sqNorms1, sqNorms1, sqNorms1, SMNE);
    Mul(sqNorms2, sqNorms2, sqNorms2, SMNE);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::
GatherNormsVectorToLeftRightNorms(const LocalTensor<float>& dstSqNorms1, const LocalTensor<float>& dstSqNorms2,
        const LocalTensor<float>& srcNorms, const RotateMaskInfo& maskInfo, const LocalTensor<float>& workLocal1,
        const LocalTensor<float>& workLocal2)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    Copy(workLocal1, srcNorms, SMNE, SMNE, { 1, 1, 1, 0 });
    Duplicate(workLocal2, 0.f, 2 * rowMask);
    PipeBarrier<PIPE_V>();
    uint64_t mask1[2] = { maskInfo.leftMaskU64Val, 0 };
    uint64_t mask2[2] = { maskInfo.rightMaskU64Val, 0 };
    Copy(workLocal2, workLocal1, mask1, 1, { 1, 1, 1, 1});
    Copy(workLocal2[rowMask], workLocal1, mask2, 1, { 1, 1, 1, 1});
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dstSqNorms1, workLocal2, SMNE, static_cast<uint8_t>(maskInfo.numPairs), 1, 1, 1);
    WholeReduceSum(dstSqNorms2, workLocal2[rowMask], SMNE, static_cast<uint8_t>(maskInfo.numPairs), 1, 1, 1);
}

#endif // JACOBI_IMPL_PART2_H