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
 * \file jacobi_impl_part3.h
 * \brief
 */

#ifndef JACOBI_IMPL_PART3_H
#define JACOBI_IMPL_PART3_H

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool LEFT_TARGET, bool NEED_CALCULATE_RIGHT_NORM>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::RotateBucket(
        const LocalTensor<float>& leftBucketSpace, const LocalTensor<float>& rightBucketSpace,
        const LocalTensor<float>& leftNorms, const LocalTensor<float>& rightNorms, const uint32_t leftBucketSize,
        const uint32_t rightBucketSize, const RotateMaskInfo& maskInfo)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    constexpr uint8_t BMNE = BIG_MASK_NUM_ELEMENTS<float>();

    LocalTensor<float>& leftSquareNorms = tmpBlock9;
    LocalTensor<float>& rightSquareNorms = tmpBlock10;
    LocalTensor<float>& dotProducts = tmpBlock11;
    CalculateDotProducts<LEFT_TARGET, NEED_CALCULATE_RIGHT_NORM>(leftSquareNorms, rightSquareNorms, dotProducts,
        leftBucketSpace, leftNorms, rightBucketSpace, rightNorms, maskInfo);
    PipeBarrier<PIPE_V>();
    LocalTensor<float>& sinTensor = tmpBlock12;
    LocalTensor<float>& cosTensor = tmpBlock13;
    CalculateSinCos(leftSquareNorms, rightSquareNorms, dotProducts, sinTensor, cosTensor);
    PipeBarrier<PIPE_V>();
    RotateNorms<LEFT_TARGET>(leftSquareNorms, rightSquareNorms, dotProducts, leftNorms, rightNorms, sinTensor,
            cosTensor, maskInfo);
    PipeBarrier<PIPE_V>();
    ApplySinCos<LEFT_TARGET>(leftBucketSpace, leftBucketSize, rightBucketSpace, rightBucketSize, sinTensor, cosTensor,
            maskInfo);
    PipeBarrier<PIPE_V>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool LEFT_TARGET>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::RotateNorms(
        const LocalTensor<float>& leftSquareNorms, const LocalTensor<float>& rightSquareNorms,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& leftNorms,
        const LocalTensor<float>& rightNorms, const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor,
        const RotateMaskInfo& maskInfo)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    LocalTensor<float>& tmpVec1 = tmpBlock20;
    LocalTensor<float>& tmpVec2 = tmpBlock21;
    LocalTensor<float>& tmpVec3 = tmpBlock22;
    LocalTensor<float>& tmpVec4 = tmpBlock23;
    LocalTensor<float>& sinSq = tmpBlock24;
    LocalTensor<float>& cosSq = tmpBlock25;
    LocalTensor<float>& sin2XDoub = tmpBlock26;
    LocalTensor<float>& broadcastedNorms = tmpMaskSpace;
    LocalTensor<float> broadcastedNormsWithZero = tmpMaskSpace[rowMask];
    const LocalTensor<float>& targetSquareNorms = (LEFT_TARGET) ? leftSquareNorms : rightSquareNorms;
    const LocalTensor<float>& nonTargetSquareNorms = (LEFT_TARGET) ? rightSquareNorms : leftSquareNorms;
    const LocalTensor<float>& targetNorms = (LEFT_TARGET) ? leftNorms : rightNorms;
    const LocalTensor<float>& nonTargetNorms = (LEFT_TARGET) ? rightNorms : leftNorms;
    uint64_t targetMaskUpdate = (1 << maskInfo.numPairs) - 1;
    uint64_t targetMaskUpdater[2] = { targetMaskUpdate, 0 };
    uint64_t nonTargetScatteringMask[2] = { 0, 0 };
    uint64_t nonTargetMaskUpdater[2] = { maskInfo.normUpdateMask, 0 };
    nonTargetScatteringMask[0] = (LEFT_TARGET)?maskInfo.rightMaskU64Val:maskInfo.leftMaskU64Val;
    Duplicate(broadcastedNormsWithZero, 0.f, rowMask);
    UpdateSquareNorms(leftSquareNorms, rightSquareNorms, dotProducts, sinTensor, cosTensor);
    PipeBarrier<PIPE_V>();
    Sqrt(targetNorms, targetSquareNorms, targetMaskUpdater, 1, { 1, 1, 1, 1 });
    Brcb(broadcastedNorms, nonTargetSquareNorms, 1, { 1, 8 });
    PipeBarrier<PIPE_V>();
    Duplicate(nonTargetSquareNorms, 0.f, SMNE);
    Copy(broadcastedNormsWithZero, broadcastedNorms, nonTargetScatteringMask, 1, { 1, 1, 1, 1 });
    PipeBarrier<PIPE_V>();
    Add(nonTargetSquareNorms, broadcastedNormsWithZero, nonTargetSquareNorms, SMNE, SMNE, { 1, 1, 1, 0, 1, 0 });
    PipeBarrier<PIPE_V>();
    Sqrt(nonTargetNorms, nonTargetSquareNorms, nonTargetMaskUpdater, 1, { 1, 1, 1, 1 });
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool NEED_CALCULATE_NORM>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::RotateSelfBucket(
        const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketSize,
        const RotateMaskInfo& maskInfo)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    constexpr uint8_t BMNE = BIG_MASK_NUM_ELEMENTS<float>();

    LocalTensor<float>& leftSquareNorms = tmpBlock9;
    LocalTensor<float>& rightSquareNorms = tmpBlock10;
    LocalTensor<float>& dotProducts = tmpBlock11;
    LocalTensor<float>& leftNorms = tmpBlock14;
    LocalTensor<float>& rightNorms = tmpBlock15;
    CalculateDotProductsForSelfBucket<NEED_CALCULATE_NORM>(leftSquareNorms, rightSquareNorms, dotProducts, normVector,
            bucketSpace, bucketSize, maskInfo);
    PipeBarrier<PIPE_V>();
    LocalTensor<float>& sinTensor = tmpBlock12;
    LocalTensor<float>& cosTensor = tmpBlock13;
    Duplicate(sinTensor, 0.f, SMNE);
    Duplicate(cosTensor, 0.f, SMNE);
    CalculateSinCos(leftSquareNorms, rightSquareNorms, dotProducts, sinTensor, cosTensor);
    PipeBarrier<PIPE_V>();
    // (Sqrt(leftSquareNorms), Sqrt(rightSquareNorms))-> normVector
    RotateSelfNorms(leftSquareNorms, rightSquareNorms, dotProducts, normVector, sinTensor, cosTensor, maskInfo);
    PipeBarrier<PIPE_V>();
    ApplySinCosForSelfBucket(bucketSpace, bucketSize, sinTensor, cosTensor, maskInfo);
    PipeBarrier<PIPE_V>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::UpdateSquareNorms(
        const LocalTensor<float>& leftSquareNorms, const LocalTensor<float>& rightSquareNorms,
        const LocalTensor<float>& dotProducts,  const LocalTensor<float>& sinTensor,
        const LocalTensor<float>& cosTensor)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    LocalTensor<float>& tmpVec1 = tmpBlock20;
    LocalTensor<float>& tmpVec2 = tmpBlock21;
    LocalTensor<float>& tmpVec3 = tmpBlock22;
    LocalTensor<float>& tmpVec4 = tmpBlock23;
    LocalTensor<float>& sinSq = tmpBlock25;
    LocalTensor<float>& cosSq = tmpBlock26;
    LocalTensor<float>& sin2XDoub = tmpBlock27;
    Mul(sinSq, sinTensor, sinTensor, SMNE);
    Mul(cosSq, cosTensor, cosTensor, SMNE);
    Mul(sin2XDoub, sinTensor, cosTensor, SMNE);
    PipeBarrier<PIPE_V>();
    Muls(sin2XDoub, sin2XDoub, 2.f, SMNE);
    Mul(tmpVec1, leftSquareNorms, cosSq, SMNE);
    Mul(tmpVec2, leftSquareNorms, sinSq, SMNE);
    Mul(tmpVec3, rightSquareNorms, sinSq, SMNE);
    Mul(tmpVec4, rightSquareNorms, cosSq, SMNE);
    PipeBarrier<PIPE_V>();
    Mul(sin2XDoub, sin2XDoub, dotProducts, SMNE);
    Add(tmpVec1, tmpVec1, tmpVec3, SMNE);
    Add(tmpVec2, tmpVec2, tmpVec4, SMNE);
    PipeBarrier<PIPE_V>();
    Sub(leftSquareNorms, tmpVec1, sin2XDoub, SMNE);
    Add(rightSquareNorms, tmpVec2, sin2XDoub, SMNE);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::RotateSelfNorms(
        const LocalTensor<float>& leftSquareNorms, const LocalTensor<float>& rightSquareNorms,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& normVector,
        const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor, const RotateMaskInfo& maskInfo)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    LocalTensor<float>& tmpVec1 = tmpBlock20;
    LocalTensor<float>& tmpVec2 = tmpBlock21;
    LocalTensor<float>& tmpVec3 = tmpBlock22;
    LocalTensor<float>& tmpVec4 = tmpBlock23;
    LocalTensor<float>& newScatteredNorm = tmpBlock24;

    LocalTensor<float>& broadcastedLeftNorms = tmpMaskSpace;
    LocalTensor<float> broadcastedRightNorms = tmpMaskSpace[rowMask];
    LocalTensor<float> broadcastedScatteredNorms = tmpMaskSpace[2 * rowMask];
    Duplicate(broadcastedScatteredNorms, 0.f, 2 * rowMask);
    UpdateSquareNorms(leftSquareNorms, rightSquareNorms, dotProducts, sinTensor, cosTensor);
    PipeBarrier<PIPE_V>();
    Brcb(broadcastedLeftNorms, leftSquareNorms, 1, { 1, 8 });
    Brcb(broadcastedRightNorms, rightSquareNorms, 1, { 1, 8 });

    PipeBarrier<PIPE_V>();
    uint64_t mask1[2] = { maskInfo.leftMaskU64Val, 0 };
    uint64_t mask2[2] = { maskInfo.rightMaskU64Val, 0 };
    Copy(broadcastedScatteredNorms, broadcastedLeftNorms, mask1, 1, { 1, 1, 8, 8 });
    Copy(broadcastedScatteredNorms[rowMask], broadcastedRightNorms, mask2, 1, { 1, 1, 8, 8 });
    Duplicate(newScatteredNorm, 0.f, SMNE);
    PipeBarrier<PIPE_V>();
    Add(newScatteredNorm, broadcastedScatteredNorms, newScatteredNorm, SMNE, 2 * SMNE, { 1, 1, 1, 0, 1, 0 });
    PipeBarrier<PIPE_V>();
    uint64_t updateMask[2] = { maskInfo.normUpdateMask, 0 };
    Sqrt(normVector, newScatteredNorm, updateMask, 1, { 1, 1, 0, 0 });
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ApplySinCosForRowSet(
        const LocalTensor<float>& bucket1, const LocalTensor<float>& bucket2,
        const uint32_t bucketSize, const LocalTensor<float>& sinBroadcastedValues,
        const LocalTensor<float>& cosBroadcastedValues, const LocalTensor<float>& calcTmpTensor1,
        const LocalTensor<float>& calcTmpTensor2, const RotateMaskInfo& maskInfo, const uint32_t rowLength,
        const uint8_t numRepeats)
{
    uint8_t maskRepeatStride = rowMask * sizeof(float) / 32;
    for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
        uint16_t rightRowIdx = maskInfo.rightRowsIds[rowIdx];
        Mul(calcTmpTensor1[rowIdx * rowLength], bucket2[rightRowIdx * rowLength],
            sinBroadcastedValues[rowIdx * rowMask], rowMask, numRepeats,
            { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
        Mul(calcTmpTensor2[rowIdx * rowLength], bucket2[rightRowIdx * rowLength],
            cosBroadcastedValues[rowIdx * rowMask], rowMask, numRepeats,
            { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
    }
    PipeBarrier<PIPE_V>();
    Muls(calcTmpTensor1, calcTmpTensor1, -1.f, bucketSize * rowLength);
    PipeBarrier<PIPE_V>();
    for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
        uint16_t leftRowIdx = maskInfo.leftRowsIds[rowIdx];
        MulAddDst(calcTmpTensor1[rowIdx * rowLength], cosBroadcastedValues[rowIdx * rowMask],
            bucket1[leftRowIdx * rowLength], rowMask, numRepeats, { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
        MulAddDst(calcTmpTensor2[rowIdx * rowLength], sinBroadcastedValues[rowIdx * rowMask],
            bucket1[leftRowIdx * rowLength], rowMask, numRepeats, { 1, 1, 1, maskRepeatStride, 0, maskRepeatStride });
    }
    PipeBarrier<PIPE_V>();
    for (uint32_t rowIdx = 0; rowIdx < bucketSize; ++rowIdx) {
        uint16_t leftRowIdx = maskInfo.leftRowsIds[rowIdx];
        uint16_t rightRowIdx = maskInfo.rightRowsIds[rowIdx];
        Copy(bucket1[leftRowIdx * rowLength], calcTmpTensor1[rowIdx * rowLength], rowMask, numRepeats,
            { 1, 1, maskRepeatStride, maskRepeatStride });
        Copy(bucket2[rightRowIdx * rowLength], calcTmpTensor2[rowIdx * rowLength], rowMask, numRepeats,
            { 1, 1, maskRepeatStride, maskRepeatStride });
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool BUCKET_1_TARGET>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ApplySinCos(
        const LocalTensor<float>& bucket1, const uint32_t bucket1Size, const LocalTensor<float>& bucket2,
        const uint32_t bucket2Size, const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor,
        const RotateMaskInfo& maskInfo)
{
    const uint32_t bucketSize = (BUCKET_1_TARGET) ? bucket1Size : bucket2Size;
    LocalTensor<float> sinBroadcastedValues = tmpMaskSpace;
    LocalTensor<float> cosBroadcastedValues = tmpMaskSpace[rowMask * maxBucketSize];
    LocalTensor<float>& calcTmpTensor1 = tmpMemSpace;
    LocalTensor<float>  calcTmpTensor2 = tmpMemSpace[maxBucketSize * nSizeAligned];
    if (rowMask * sizeof(float) < 256) {
        Brcb(cosBroadcastedValues, cosTensor, 1, { 1, 1 });
        PipeBarrier<PIPE_V>();
        Brcb(sinBroadcastedValues, sinTensor, 1, { 1, 1 });
        PipeBarrier<PIPE_V>();
    } else {
        Brcb(calcTmpTensor1, cosTensor, 1, { 1, 8 });
        Brcb(calcTmpTensor2, sinTensor, 1, { 1, 8 });
        PipeBarrier<PIPE_V>();
        Brcb(cosBroadcastedValues, calcTmpTensor1, static_cast<uint8_t>(bucketSize), { 1, 8 });
        Brcb(sinBroadcastedValues, calcTmpTensor2, static_cast<uint8_t>(bucketSize), { 1, 8 });
    }
    PipeBarrier<PIPE_V>();
    uint32_t uOffsetForBacket1 = bucket1Size*nSizeAligned;
    uint32_t uOffsetForBacket2 = bucket2Size*nSizeAligned;

    ApplySinCosForRowSet(bucket1, bucket2, bucketSize, sinBroadcastedValues, cosBroadcastedValues, calcTmpTensor1,
                    calcTmpTensor2, maskInfo, nSizeAligned, rowRepeatsNum);
    PipeBarrier<PIPE_V>();
    ApplySinCosForRowSet(bucket1[uOffsetForBacket1], bucket2[uOffsetForBacket2], bucketSize, sinBroadcastedValues,
                    cosBroadcastedValues, calcTmpTensor1, calcTmpTensor2, maskInfo, uNSizeAligned, uRepeatsNum);
    PipeBarrier<PIPE_V>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool BUCKET_1_TARGET>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ApplySinCosForSelfBucket(
        const LocalTensor<float>& bucket, const uint32_t bucketSize, const LocalTensor<float>& sinTensor,
        const LocalTensor<float>& cosTensor, const RotateMaskInfo& maskInfo)
{
    LocalTensor<float> sinBroadcastedValues = tmpMaskSpace;
    LocalTensor<float> cosBroadcastedValues = tmpMaskSpace[rowMask * maxBucketSize];
    if (rowMask * sizeof(float) < 256) {
        Brcb(cosBroadcastedValues, cosTensor, 1, { 1, 1 });
        PipeBarrier<PIPE_V>();
        Brcb(sinBroadcastedValues, sinTensor, 1, { 1, 1 });
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<float>& tmpMaskTensor1 = tmpMemSpace;
        LocalTensor<float>  tmpMaskTensor2 = tmpMemSpace[rowMask * bucketSize];
        Brcb(tmpMaskTensor1, cosTensor, 1, { 1, 8 });
        Brcb(tmpMaskTensor2, sinTensor, 1, { 1, 8 });
        PipeBarrier<PIPE_V>();
        Brcb(cosBroadcastedValues, tmpMaskTensor1, static_cast<uint8_t>(maskInfo.numPairs), { 1, 8 });
        Brcb(sinBroadcastedValues, tmpMaskTensor2, static_cast<uint8_t>(maskInfo.numPairs), { 1, 8 });
    }

    PipeBarrier<PIPE_V>();

    LocalTensor<float>& calcTmpTensor1 = tmpMemSpace;
    LocalTensor<float>  calcTmpTensor2 = tmpMemSpace[maxBucketSize * nSizeAligned];
    uint8_t maskRepeatStride = rowMask * sizeof(float) / 32;
    ApplySinCosForRowSet(bucket, bucket,  maskInfo.numPairs, sinBroadcastedValues, cosBroadcastedValues,
                calcTmpTensor1, calcTmpTensor2, maskInfo, nSizeAligned, rowRepeatsNum);
    // TODO: Add for APPLYING SIN AND COS FOR U TENSOR
    // Applaing U matrice
    PipeBarrier<PIPE_V>();
    uint32_t uOffset = bucketSize*nSizeAligned;
    ApplySinCosForRowSet(bucket[uOffset], bucket[uOffset],  maskInfo.numPairs, sinBroadcastedValues,
                cosBroadcastedValues, calcTmpTensor1, calcTmpTensor2, maskInfo, uNSizeAligned, uRepeatsNum);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::InitV(
        const LocalTensor<float>& bucketSpace, const uint32_t bucketBegin, const uint32_t bucketEnd)
{
    uint16_t numRows = bucketEnd - bucketBegin;
    Duplicate(bucketSpace[nSizeAligned*numRows], 0.f, uNSizeAligned*numRows);
    PipeBarrier<PIPE_V>();
    for (uint32_t rowId = bucketBegin; rowId < bucketEnd; ++rowId) {
        uint32_t localBlockOffset = (rowId / rowMask) * rowMask;
        uint32_t onePosition = rowId % rowMask;
        uint64_t localMask = 1;
        localMask <<= onePosition;
        uint64_t mask[2] = { localMask, 0 };
        uint32_t rowOffset = nSizeAligned*numRows+ (rowId - bucketBegin) * uNSizeAligned;
        Duplicate(bucketSpace[rowOffset + localBlockOffset], 1.f, mask, 1, 1, 8);
    }
    PipeBarrier<PIPE_V>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::CleanTailThrash(
        const LocalTensor<float>& bucketSpace, const uint32_t bucketBegin, const uint32_t bucketEnd)
{
    uint32_t nSizeAligned32 = (((nSize*sizeof(float)+31)/32)*32) / sizeof(float);
    if (nSizeAligned32 == nSizeAligned) {
        return;
    }
    uint8_t numRows = static_cast<uint8_t>(bucketEnd-bucketBegin);
    uint8_t tailMask = nSizeAligned - nSizeAligned32;
    for (uint8_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        uint32_t rowOffset = rowIdx*nSizeAligned;
        Duplicate(bucketSpace[rowOffset + nSizeAligned32], 0.f, tailMask, 1, 1, 8);
    }
    PipeBarrier<PIPE_V>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::LoadBucket(
    const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketBegin,
    const uint32_t bucketEnd, const bool isFirstLoadDataLocal)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    uint16_t numRows = bucketEnd - bucketBegin;

    if (!isFirstLoadDataLocal) {
        DataCopy(bucketSpace, aTmpGm[batchOffsetTmp + bucketBegin * recordSizeAligned],
                { numRows, static_cast<uint16_t>(recordSizeAligned * sizeof(AType) / 32), 0, 0 });
        DataCopyPad(normVector, sGm[batchOffsetS + bucketBegin],
                { 1, static_cast<uint16_t>(numRows * sizeof(AType)), 0, 0 },
                { true, 0, static_cast<uint8_t>(SMNE - numRows), 0 });
    } else {
        uint32_t nSizeAligned32 = ((((nSize * sizeof(AType)) + 31) / 32) * 32) / sizeof(AType);
        DataCopyPad(bucketSpace, aGm[batchOffsetI + bucketBegin * nSize],
            { numRows, static_cast<uint16_t>(nSize * sizeof(AType)), 0,
                    static_cast<uint16_t>((nSizeAligned - nSizeAligned32) * sizeof(AType) / 32) },
            { true, 0, static_cast<uint8_t>(nSizeAligned32 - nSize), 0 });
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::SaveBucket(
    const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketBegin,
    const uint32_t bucketEnd)
{
    uint16_t numRows = bucketEnd - bucketBegin;
    DataCopy(aTmpGm[batchOffsetTmp + bucketBegin * recordSizeAligned], bucketSpace,
            { numRows, static_cast<uint16_t>(recordSizeAligned * sizeof(AType) / 32), 0, 0 });
    DataCopyPad(sGm[batchOffsetS + bucketBegin], normVector,
            { 1, static_cast<uint16_t>(numRows * sizeof(AType)), 0, 0 });
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<bool IS_LEFT_BUCKET_SAVING>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::SaveOutputs(
        const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector, const uint32_t bucketBegin,
        const uint32_t bucketEnd)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    uint16_t numRows = bucketEnd - bucketBegin;
    uint32_t uNSizeAligned32 = (((uNSize * sizeof(float) + 31) / 32) * 32) / sizeof(float);
    uint32_t vNSizeAligned32 = (((vNSize * sizeof(float) + 31) / 32) * 32) / sizeof(float);
    const uint8_t maskRepeatStride = rowMask * sizeof(float) / 32;
    LocalTensor<float> singularValues;
    LocalTensor<float> tmpCurrMemSpace, tmpCurrMaskSpace;
    if constexpr(IS_LEFT_BUCKET_SAVING) {
        singularValues = tmpBlock4;
        tmpCurrMemSpace = tmpMemSpace;
        tmpCurrMaskSpace = tmpMaskSpace;
    } else {
        singularValues = tmpBlock5;
        tmpCurrMemSpace = tmpMemSpace[maxBucketSize*nSizeAligned];
        tmpCurrMaskSpace = tmpMaskSpace[maxBucketSize*rowMask];
    }
    Duplicate(tmpCurrMaskSpace, 0.f, numRows * rowMask);
    PipeBarrier<PIPE_V>();
    
    for (uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        MulAddDst(tmpCurrMaskSpace[rowIdx*rowMask], bucketSpace[rowIdx*nSizeAligned], bucketSpace[rowIdx*nSizeAligned],
                rowMask, rowRepeatsNum, {1, 1, 1, 0, maskRepeatStride, maskRepeatStride});
    }
    PipeBarrier<PIPE_V>();
    WholeReduceSum(singularValues, tmpCurrMaskSpace, rowMask, static_cast<uint8_t>(numRows), 1, 1, maskRepeatStride);
    PipeBarrier<PIPE_V>();
    Sqrt(singularValues, singularValues, SMNE);
    PipeBarrier<PIPE_V>();
    Brcb(tmpCurrMemSpace, singularValues, 1, { 1, 8 });
    PipeBarrier<PIPE_V>();
    Brcb(tmpCurrMaskSpace, tmpCurrMemSpace, static_cast<uint8_t>(numRows), { 1, 8 });
    PipeBarrier<PIPE_V>();

    for (uint32_t rowIdx = 0; rowIdx < numRows; ++rowIdx) {
        Div(bucketSpace[rowIdx * nSizeAligned], bucketSpace[rowIdx * nSizeAligned], tmpCurrMaskSpace[rowIdx * rowMask],
            rowMask, rowRepeatsNum, { 1, 1, 1, maskRepeatStride, maskRepeatStride, 0 });
    }

    PipeBarrier<PIPE_V>();
    event_t evtVToMte3_1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(evtVToMte3_1);
    WaitFlag<HardEvent::V_MTE3>(evtVToMte3_1);
    DataCopyPad(uGm[batchOffsetU + bucketBegin * uNSize], bucketSpace[numRows*nSizeAligned],
            {static_cast<uint16_t>(numRows), static_cast<uint16_t>(uNSize * sizeof(AType)),
            static_cast<uint16_t>((uNSizeAligned - uNSizeAligned32) * sizeof(AType) / 32), 0 });
    DataCopyPad(sGm[batchOffsetS + bucketBegin], singularValues,
            { 1, static_cast<uint16_t>(numRows * sizeof(AType)), 0, 0 });
    DataCopyPad(vGm[batchOffsetV + bucketBegin * vNSize], bucketSpace,
            { static_cast<uint16_t>(numRows), static_cast<uint16_t>(vNSize * sizeof(AType)),
            static_cast<uint16_t>((nSizeAligned - vNSizeAligned32) * sizeof(AType) / 32), 0 });
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::CalculateSinCosConvertTaoToArg(
        const LocalTensor<float>& tauTensor, const LocalTensor<float>& tTensor)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    LocalTensor<float>& absTaoTensor = tmpBlock5;
    LocalTensor<float>& tmp1 = tmpBlock6;
    LocalTensor<float>& tmp2 = tmpBlock7;

    Abs(absTaoTensor, tauTensor, SMNE);
    Mul(tmp1, tauTensor, tauTensor, SMNE);
    PipeBarrier<PIPE_V>();
    Adds(tmp1, tmp1, 1.f, SMNE);
    Div(tmp2, tauTensor, absTaoTensor, SMNE);
    PipeBarrier<PIPE_V>();
    Sqrt(tmp1, tmp1, SMNE);
    PipeBarrier<PIPE_V>();
    Add(tmp1, tmp1, absTaoTensor, SMNE);
    PipeBarrier<PIPE_V>();
    Div(tTensor, tmp2, tmp1, SMNE);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::CalculateSinCosValues(
        const LocalTensor<float>& tTensor, const LocalTensor<float>& s1Tensor, const LocalTensor<float>& s2Tensor,
        const LocalTensor<uint8_t>&normSquareDiffIsPositive)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();
    LocalTensor<float>& negativeOneTensor = tmpBlock6;
    Duplicate(negativeOneTensor, -1.f, SMNE);
    Mul(s1Tensor, tTensor, tTensor, SMNE);
    Duplicate(s2Tensor, 1.f, SMNE);
    PipeBarrier<PIPE_V>();
    Add(s1Tensor, s2Tensor, s1Tensor, SMNE);
    PipeBarrier<PIPE_V>();
    Sqrt(s1Tensor, s1Tensor, SMNE);
    PipeBarrier<PIPE_V>();
    Div(s1Tensor, s2Tensor, s1Tensor, SMNE);
    PipeBarrier<PIPE_V>();
    Mul(s2Tensor, s1Tensor, tTensor, SMNE);
    LocalTensor<float>& coeff = tmpBlock4;
    Select(coeff, normSquareDiffIsPositive, negativeOneTensor, 1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SMNE);
    PipeBarrier<PIPE_V>();
    Mul(s2Tensor, s2Tensor, coeff, SMNE);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::CalculateSinCos(
        const LocalTensor<float>& leftSquareNorms, const LocalTensor<float>& rightSquareNorms,
        const LocalTensor<float>& dotProducts, const LocalTensor<float>& sinTensor,
        const LocalTensor<float>& cosTensor)
{
    constexpr uint8_t SMNE = SMALL_MASK_NUM_ELEMENTS<float>();

    LocalTensor<float>& tauTensor = tmpBlock1;
    LocalTensor<uint8_t> normSquareDiffIsPositive = tmpBlock2.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> isFakeRotateCase = tmpBlock3.ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> isNotFakeRotateCase =  tmpBlock3.ReinterpretCast<uint8_t>();
    LocalTensor<float>& normSquareDiff = tmpBlock4;
    LocalTensor<float>& constValues = tmpBlock5;
    LocalTensor<float>& absDotProducts = tmpBlock6;

    Duplicate(constValues, 0.f, SMNE);

    Sub(normSquareDiff, rightSquareNorms, leftSquareNorms, SMNE);
    PipeBarrier<PIPE_V>();
    VcmpvImpl((__ubuf__ uint8_t*)normSquareDiffIsPositive.GetPhyAddr(), (__ubuf__ float*)normSquareDiff.GetPhyAddr(),
            (__ubuf__ float*)constValues.GetPhyAddr(), CMPMODE::GE, SMNE, 1, { 1, 1, 1, 1, 1, 1 });
    Div(tauTensor, normSquareDiff, dotProducts, SMNE);
    PipeBarrier<PIPE_V>();
    Muls(tauTensor, tauTensor, 0.5f, SMNE);
    Abs(normSquareDiff, normSquareDiff, SMNE);
    Abs(absDotProducts, dotProducts, SMNE);
    Duplicate(constValues, NEAR_ZERO_VALUE, SMNE);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint8_t>& normSquareDiffIsZero = isFakeRotateCase;
    LocalTensor<uint8_t> dotProductIsZero = tmpBlock7.ReinterpretCast<uint8_t>();
    VcmpvImpl((__ubuf__ uint8_t*)normSquareDiffIsZero.GetPhyAddr(), (__ubuf__ float*)normSquareDiff.GetPhyAddr(),
            (__ubuf__ float*)constValues.GetPhyAddr(), CMPMODE::LT, SMNE, 1, { 1, 1, 1, 1, 1, 1 });
    VcmpvImpl((__ubuf__ uint8_t*)dotProductIsZero.GetPhyAddr(), (__ubuf__ float*)absDotProducts.GetPhyAddr(),
            (__ubuf__ float*)constValues.GetPhyAddr(), CMPMODE::LT, SMNE, 1, { 1, 1, 1, 1, 1, 1 });
    PipeBarrier<PIPE_V>();
    Or(isFakeRotateCase, normSquareDiffIsZero, dotProductIsZero, SMNE);
    PipeBarrier<PIPE_V>();

    LocalTensor<float>& tTensor = tmpBlock4;

    NotImpl((__ubuf__ uint16_t*)isNotFakeRotateCase.GetPhyAddr(), (__ubuf__ uint16_t*)isFakeRotateCase.GetPhyAddr(),
                SMNE, 1, {1, 1, 1, 1});
    CalculateSinCosConvertTaoToArg(tauTensor, tTensor);
    PipeBarrier<PIPE_V>();

    LocalTensor<float>& s1Tensor = tmpBlock1;
    LocalTensor<float>& s2Tensor = tmpBlock5;
    CalculateSinCosValues(tTensor, s1Tensor, s2Tensor, normSquareDiffIsPositive);
    PipeBarrier<PIPE_V>();

    Select(sinTensor, normSquareDiffIsPositive, s1Tensor, s2Tensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
    Select(cosTensor, normSquareDiffIsPositive, s2Tensor, s1Tensor, SELMODE::VSEL_CMPMASK_SPR, SMNE);
    PipeBarrier<PIPE_V>();
    Select(cosTensor, isNotFakeRotateCase, cosTensor, 1.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SMNE);
    Select(sinTensor, isNotFakeRotateCase, sinTensor, 0.f, SELMODE::VSEL_TENSOR_SCALAR_MODE, SMNE);
    PipeBarrier<PIPE_V>();
}

#endif // JACOBI_IMPL_PART3_H