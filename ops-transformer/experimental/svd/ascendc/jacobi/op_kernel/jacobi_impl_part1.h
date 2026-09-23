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
 * \file jacobi_impl_part1.h
 * \brief
 */

#ifndef JACOBI_IMPL_PART1_H
#define JACOBI_IMPL_PART1_H
template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::JacobiBase(TPipe* pipe)
{
    pipe_ = pipe;
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::Process()
{
    if (g_coreType == AIC) {
        return;
    }
    ProcessImpl();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::Init(
    __gm__ uint8_t* a, __gm__ uint8_t* s, __gm__ uint8_t* u, __gm__ uint8_t* v, __gm__ uint8_t* workspace,
    const JMTilingData* tilingData)
{
    coreIdx = GetBlockIdxImpl();
    needSort = true;
    numIterations = tilingData->nIterations;
    nSize = tilingData->nSize;
    nSizeAligned = tilingData->nSizeAligned;
    mSize = tilingData->mSize;
    batchSize = (int32_t)tilingData->batchSize;
    numStages = tilingData->nStages;
    numGlobalSets = tilingData->numGlobalSets;
    globSetMaxSize = tilingData->globalSetSizeMax;
    globSetMinSize = tilingData->globalSetSizeMin;
    numGlobalSetsWithMaxSize = tilingData->numGlobalSetsWithMaxSize;
    numGlobalSetsWithMinSize = tilingData->numGlobalSetsWithMinSize;
    globSetMinOffset = numGlobalSetsWithMaxSize * globSetMaxSize;

    ubSize = tilingData->ubSize;
    rowSize = tilingData->aRowSize;
    rowSizeAligned = tilingData->aRowSizeAligned;
    rowMask = tilingData->nMaskSize;
    rowRepeatsNum = tilingData->nRepeatsNum;
    rowTail = tilingData->nTailSize;
    uMSize = tilingData->uMSize;
    uNSize = tilingData->uNSize;
    uNSizeAligned = tilingData->uNSizeAligned;

    vMSize = tilingData->vMSize;
    vNSize = tilingData->vNSize;
    vNSizeAligned = tilingData->vNSizeAligned;
    sMNSize = tilingData->sMNSize;
    sMNSizeAligned = ((sMNSize * sizeof(float) + 31) / 32) * 32;
    numAvailableVectorCores = tilingData->numUsedVectorCores;
    maxBucketSize = tilingData->maxBucketSize;
    tmpRowSize = tilingData->tmpMemRowSize;
    tmpRowSizeAligned = tilingData->tmpMemRowSizeAligned;

    uRepeatsNum = tilingData->uNumRepeats;
    uTailSize = tilingData->uTailSize;
    recordSize = tilingData->recordSize;
    recordSizeAligned = tilingData->recordSizeAligned;
    aGm.SetGlobalBuffer((__gm__ AType*)a);
    aTmpGm.SetGlobalBuffer((__gm__ float*)workspace, mSize * recordSizeAligned);
    sGm.SetGlobalBuffer((__gm__ SType*)s);
    uGm.SetGlobalBuffer((__gm__ UType*)u);
    vGm.SetGlobalBuffer((__gm__ VType*)v);
    pipe_->InitBuffer(ubBuf, ubSize);
    ubMemory = ubBuf.Get<uint8_t>();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::AllocMem()
{
#define ALLOC_TMP_BLOCKS1_16 GTB(1) GTB(2) GTB(3) GTB(4) GTB(5) GTB(6) GTB(7) GTB(8) GTB(9) GTB(10) \
                            GTB(11) GTB(12) GTB(13)  GTB(14) GTB(15) GTB(16)
#define ALLOC_TMP_BLOCKS17_32 GTB(17) GTB(18) GTB(19) GTB(20) GTB(21) GTB(22) GTB(23) GTB(24) GTB(25) GTB(26) \
                            GTB(27) GTB(28) GTB(29)  GTB(30) GTB(31) GTB(32)
#define GTB(ID) tmpBlock##ID = ubBuf.GetWithOffset<float>(32 * MIN_BLOCK_ELEMENTS_NUM, offsetPtr); \
                        offsetPtr += MIN_BLOCK_SIZE;

    uint32_t offsetPtr = 0;

    // Small blocks for cos/sin vectorisation
    constexpr uint8_t MIN_BLOCK_ELEMENTS_NUM = SMALL_MASK_NUM_ELEMENTS<float>();
    constexpr uint32_t MIN_BLOCK_SIZE = GET_MIN_BLOCK_SIZE<uint32_t>();
    tmpBlockSpace = ubBuf.GetWithOffset<float>(32 * MIN_BLOCK_ELEMENTS_NUM, offsetPtr);
    ALLOC_TMP_BLOCKS1_16
    ALLOC_TMP_BLOCKS17_32
    uint32_t bucketElementsNum = maxBucketSize * recordSizeAligned;
    bucketSpace1 = ubBuf.GetWithOffset<float>(bucketElementsNum, offsetPtr);
    offsetPtr += bucketElementsNum * sizeof(float);
    bucketSpace2 = ubBuf.GetWithOffset<float>(bucketElementsNum, offsetPtr);
    offsetPtr += bucketElementsNum * sizeof(float);
    bucketSpace3 = ubBuf.GetWithOffset<float>(bucketElementsNum, offsetPtr);
    offsetPtr += bucketElementsNum * sizeof(float);

    // tmpMaskSpace, bucketSpace1, bucketSpace2, bucketSpace3
    uint32_t tmpMemElementsNum = 2 * maxBucketSize * tmpRowSizeAligned;
    tmpMemSpace = ubBuf.GetWithOffset<float>(tmpMemElementsNum, offsetPtr);
    offsetPtr += tmpMemElementsNum * sizeof(float);

    uint32_t tmpMaskElementsNum = 2 * maxBucketSize * rowMask;
    if (maxBucketSize == 1) {
        tmpMaskElementsNum = 2 * 2 * rowMask;
    }

    tmpMaskSpace = ubBuf.GetWithOffset<float>(tmpMaskElementsNum, offsetPtr);
    offsetPtr += tmpMaskElementsNum * sizeof(float);
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ProcessImpl()
{
    AllocMem();
    batchOffsetV = 0;
    batchOffsetU = 0;
    batchOffsetS = 0;
    batchOffsetI = 0;
    batchOffsetTmp = 0;
    JacobiProcess();
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::JacobiProcess()
{
    for (uint16_t iter = 0; iter < numIterations; ++iter) {
        bool isFinalIteration=(iter==numIterations-1);
        bool isZeroIter=(iter==0);
        for (uint32_t stageID = 0; stageID < numStages; ++stageID) {
            bool isFinalStage = (stageID==numStages-1);
            bool isZeroStage=(stageID==0);
            numPhases = 1 << stageID;
            for (uint32_t phaseID = 0; phaseID < numPhases; ++phaseID) {
                bool isFinalPhaseID = (phaseID==numPhases-1);
                uint16_t leftGlobalSetId = 0;
                uint16_t rightGlobalSetId = 0;
                bool isSaveUV=isFinalIteration&&isFinalStage&&isFinalPhaseID;
                GetLeftRightGlobalSetIds(leftGlobalSetId, rightGlobalSetId, phaseID, numPhases, coreIdx,
                        numAvailableVectorCores);
                LocalPhaseProcess(leftGlobalSetId, rightGlobalSetId, isSaveUV, isZeroStage, isZeroIter);
                CrossCoreSetFlag<0x0, PIPE_MTE3>(INTERNAL_SYNC_ALL);
                CrossCoreWaitFlag(INTERNAL_SYNC_ALL);
            }
        }
    }
}


template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::LocalPhaseProcess(
        const uint16_t leftGlobalSetId, const uint16_t rightGlobalSetId, const bool isSaveUV, const bool isZeroStage,
        const bool isZeroIter)
{
    if (leftGlobalSetId >= numGlobalSets || rightGlobalSetId >= numGlobalSets) {
        return;
    }
    bool leftSetHasMaxSize = (leftGlobalSetId < numGlobalSetsWithMaxSize);
    bool rightSetHasMaxSize = (rightGlobalSetId < numGlobalSetsWithMaxSize);
    uint32_t leftBegin = (leftSetHasMaxSize) ? leftGlobalSetId * globSetMaxSize :
        globSetMinOffset + (leftGlobalSetId - numGlobalSetsWithMaxSize) * globSetMinSize;
    uint32_t leftEnd = (leftSetHasMaxSize) ? leftBegin + globSetMaxSize : leftBegin + globSetMinSize;
    uint32_t rightBegin = (rightSetHasMaxSize) ? rightGlobalSetId * globSetMaxSize :
        globSetMinOffset + (rightGlobalSetId - numGlobalSetsWithMaxSize) * globSetMinSize;
    uint32_t rightEnd = (rightSetHasMaxSize) ? rightBegin + globSetMaxSize : rightBegin + globSetMinSize;
    
    for (uint32_t batchIdx = 0;batchIdx < batchSize; ++batchIdx) {
        batchOffsetI = batchIdx*mSize*nSize;
        batchOffsetU = batchIdx*uMSize*uNSize;
        batchOffsetV = batchIdx*vMSize*vNSize;
        batchOffsetS = batchIdx*sMNSize;
        batchOffsetTmp = batchIdx*sMNSize*recordSizeAligned;
        if (!isZeroStage) {
            ApplyGlobalSet(leftBegin, leftEnd, rightBegin, rightEnd, isSaveUV);
        } else {
            ApplyGlobalSet(leftBegin, leftEnd, isZeroIter);
            ApplyGlobalSet(rightBegin, rightEnd, isZeroIter);
            ApplyGlobalSet(leftBegin, leftEnd, rightBegin, rightEnd, isSaveUV);
        }
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::GetLeftRightGlobalSetIds(
        uint16_t& dstLeftGlobalSetId, uint16_t& dstRightGlobalSetId, const uint32_t phaseID, const uint32_t numPhases,
        const uint32_t coreID, const uint32_t numCores)
{
    uint32_t groupSize = numPhases * 2;
    uint32_t coresInGroup = groupSize / 2;
    uint32_t numPhases_ = (numPhases!=0)?numPhases:1;
    if (coresInGroup > numCores) {
        // usually on last stage when number of coresInGroup is not equal real number of cores,
        // we try to reassing tasks from imaginary cores to real cores
        uint32_t realCoreIdx = 0;
        for (uint32_t virtualCoreIdx = 0; virtualCoreIdx < coresInGroup; virtualCoreIdx++) {
            uint32_t groupID = virtualCoreIdx / coresInGroup;
            uint32_t offsetInGroup = virtualCoreIdx % coresInGroup;
            uint16_t leftGlobalSetIdForVirtualCore = groupID * groupSize + offsetInGroup;
            uint32_t rightGlobalSetIdForVirtualCore = groupID * groupSize + coresInGroup +
                    (offsetInGroup + phaseID) % numPhases;
            if (leftGlobalSetIdForVirtualCore >= numGlobalSets ||
                rightGlobalSetIdForVirtualCore >= numGlobalSets) {
                continue;
            }
            if (realCoreIdx == coreID) {
                dstLeftGlobalSetId = leftGlobalSetIdForVirtualCore;
                dstRightGlobalSetId = rightGlobalSetIdForVirtualCore;
                return;
            }
            realCoreIdx++;
        }

        dstLeftGlobalSetId = numGlobalSets;
        dstRightGlobalSetId = numGlobalSets;
    } else {
        uint32_t groupID = coreID / coresInGroup;
        uint32_t offsetInGroup = coreID % coresInGroup;
        dstLeftGlobalSetId = groupID * groupSize + offsetInGroup;
        dstRightGlobalSetId = groupID * groupSize + coresInGroup + (offsetInGroup + phaseID) % numPhases;
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ApplyGlobalSet(
        const uint16_t lBegin, const uint16_t lEnd, const uint16_t rBegin, const uint16_t rEnd,
        const bool isFinalIteration)
{
    if (coreIdx >= numAvailableVectorCores) {
        return;
    }
    uint32_t leftSetSize = lEnd - lBegin;
    uint32_t rightSetSize = rEnd - rBegin;
    uint32_t numBucketsInLeftSet = (leftSetSize + maxBucketSize - 1) / maxBucketSize;
    uint32_t numBucketsInRightSet = (rightSetSize + maxBucketSize - 1) / maxBucketSize;
    LocalTensor<float> leftBucketSpace, rightBucketSpace, leftNorms, rightNorms;
    {
        // Roole of choosing leftBucketSpace
        leftBucketSpace = bucketSpace1;
        rightBucketSpace = bucketSpace2;
        leftNorms = tmpBlock31;
        rightNorms = tmpBlock32;
    }
    for (uint32_t leftBucketId = 0; leftBucketId < numBucketsInLeftSet; ++leftBucketId) {
        uint32_t leftBucketBegin = lBegin + leftBucketId * maxBucketSize;
        uint32_t leftBucketEnd = MIN(leftBucketBegin + maxBucketSize, lEnd);
        LoadBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd, false);
        for (uint32_t rightBucketId = 0; rightBucketId < numBucketsInRightSet; ++rightBucketId) {
            uint32_t rightBucketBegin = rBegin + rightBucketId * maxBucketSize;
            uint32_t rightBucketEnd = MIN(rightBucketBegin + maxBucketSize, rEnd);
            LoadBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd, false);
            FAST_SET_WAIT_FLAG(evtMte2ToV, MTE2_V);
            ApplyBucket(leftBucketSpace, leftNorms, rightBucketSpace, rightNorms, leftBucketEnd - leftBucketBegin,
                rightBucketEnd - rightBucketBegin, false);
            FAST_SET_WAIT_FLAG(evtVToMte3, V_MTE3);
            if (!isFinalIteration || (isFinalIteration && (leftBucketId != (numBucketsInLeftSet-1)))) {
                SaveBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd);
            } else {
                PipeBarrier<PIPE_V>();
                SaveOutputs<false>(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd);
            }
            if (rightBucketId < numBucketsInRightSet - 1) {
                FAST_SET_WAIT_FLAG(evtMte3ToMte2, MTE3_MTE2);
            }
        }
        if (!isFinalIteration) {
            SaveBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd);
        } else {
            PipeBarrier<PIPE_V>();
            SaveOutputs<true>(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd);
        }
        FAST_SET_WAIT_FLAG(evtMte3ToMte2D, MTE3_MTE2);
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ApplyGlobalSet(
        const uint16_t sBegin, const uint16_t sEnd, const bool isFirstLoadData)
{
    if (coreIdx >= numAvailableVectorCores) {
        return;
    }
    uint32_t setSize = sEnd - sBegin;
    uint32_t numBucketsInSet = (setSize + maxBucketSize - 1) / maxBucketSize;
    bool isFirstLoadDataLocal = isFirstLoadData;
    LocalTensor<float> leftBucketSpace, rightBucketSpace, leftNorms, rightNorms;
    {
        // Roole of choosing leftBucketSpace
        leftBucketSpace = bucketSpace1;
        rightBucketSpace = bucketSpace2;
        leftNorms = tmpBlock31;
        rightNorms = tmpBlock32;
    }
    for (uint32_t leftBucketId = 0; leftBucketId < numBucketsInSet; ++leftBucketId) {
        uint32_t leftBucketBegin = sBegin + leftBucketId * maxBucketSize;
        uint32_t leftBucketEnd = MIN(leftBucketBegin + maxBucketSize, sEnd);

        LoadBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd, isFirstLoadDataLocal);
        FAST_SET_WAIT_FLAG(evtMte2ToV, MTE2_V);
        if (isFirstLoadDataLocal)
        {
            InitV(leftBucketSpace, leftBucketBegin, leftBucketEnd);
            CleanTailThrash(leftBucketSpace, leftBucketBegin, leftBucketEnd);
        }
        ApplyBucket(leftBucketSpace, leftNorms, leftBucketEnd - leftBucketBegin, isFirstLoadDataLocal);
        if (leftBucketId + 1 >= numBucketsInSet) {
            FAST_SET_WAIT_FLAG(evtVToMte3, V_MTE3);
        }

        for (uint32_t rightBucketId = leftBucketId + 1; rightBucketId < numBucketsInSet; ++rightBucketId) {
            uint32_t rightBucketBegin = sBegin + rightBucketId * maxBucketSize;
            uint32_t rightBucketEnd = MIN(rightBucketBegin + maxBucketSize, sEnd);
            LoadBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd, isFirstLoadDataLocal);
            FAST_SET_WAIT_FLAG(evtMte2ToV, MTE2_V);
            if (isFirstLoadDataLocal)
            {
                InitV(rightBucketSpace, rightBucketBegin, rightBucketEnd);
                CleanTailThrash(rightBucketSpace, rightBucketBegin, rightBucketEnd);
            }
            ApplyBucket(leftBucketSpace, leftNorms, rightBucketSpace, rightNorms, leftBucketEnd - leftBucketBegin,
                    rightBucketEnd - rightBucketBegin, isFirstLoadDataLocal);
            FAST_SET_WAIT_FLAG(evtVToMte3, V_MTE3);
            SaveBucket(rightBucketSpace, rightNorms, rightBucketBegin, rightBucketEnd);
            if (rightBucketId != numBucketsInSet - 1)
            {
                FAST_SET_WAIT_FLAG(evtMte3ToMte2, MTE3_MTE2);
            }
        }
        isFirstLoadDataLocal = false;
        SaveBucket(leftBucketSpace, leftNorms, leftBucketBegin, leftBucketEnd);
        FAST_SET_WAIT_FLAG(evtMte3ToMte2, MTE3_MTE2);
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ApplyBucket(
        const LocalTensor<float>& leftBucketSpace, const LocalTensor<float>& leftNorms,
        const LocalTensor<float>& rightBucketSpace, const LocalTensor<float>& rightNorms,
        const uint32_t leftBucketSize, const uint32_t rightBucketSize, const bool calculateRightNorms)
{
    if (leftBucketSize==0&&rightBucketSize==0) {
        return;
    }
    bool leftIsLess = (leftBucketSize <= rightBucketSize);
    uint32_t aNum = (leftIsLess) ? rightBucketSize : leftBucketSize;
    RotateMaskInfo maskInfo;
    bool calculateRightNorm = calculateRightNorms;
    for (uint32_t aOffset = 0; aOffset < aNum; ++aOffset) {
        maskInfo.normUpdateMask = 0;
        maskInfo.leftBucketSize = leftBucketSize;
        maskInfo.rightBucketSize = rightBucketSize;
        maskInfo.rightMaskU64Val = 0;
        maskInfo.leftMaskU64Val = 0;
        if (leftIsLess) {
            maskInfo.numPairs = leftBucketSize;
            for (uint32_t idx = 0; idx < leftBucketSize; ++idx) {
                uint8_t rightIdx = (idx + aOffset) % rightBucketSize;
                maskInfo.rightRowsIds[idx] = rightIdx;
                maskInfo.rightMaskU8Val[idx] = 1 << rightIdx;
                maskInfo.normUpdateMask |= 1 << rightIdx;
                maskInfo.leftRowsIds[idx] = idx;
                maskInfo.leftMaskU8Val[idx] = 1 << idx;
            }
            if (!calculateRightNorm) {
                RotateBucket<true, false>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize,
                        rightBucketSize, maskInfo);
            } else {
                RotateBucket<true, true>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize,
                        rightBucketSize, maskInfo);
            }
        } else {
            maskInfo.numPairs = rightBucketSize;
            for (uint32_t idx = 0; idx < rightBucketSize; ++idx) {
                uint8_t leftIdx = (idx + aOffset) % leftBucketSize;
                maskInfo.leftRowsIds[idx] = leftIdx;
                maskInfo.leftMaskU8Val[idx] = 1 << leftIdx;
                maskInfo.normUpdateMask |= 1 << leftIdx;
                maskInfo.rightRowsIds[idx] = idx;
                maskInfo.rightMaskU8Val[idx] = 1 << idx;
            }
            if (!calculateRightNorm) {
                RotateBucket<false, false>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize,
                        rightBucketSize, maskInfo);
            } else {
                RotateBucket<false, true>(leftBucketSpace, rightBucketSpace, leftNorms, rightNorms, leftBucketSize,
                        rightBucketSize, maskInfo);
            }
        }
        calculateRightNorm = false;
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<typename T>
__aicore__ inline T  JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ScalarLog2(const T val)
{
    T tmp = val;
    T res = 0;
    while (tmp > 0) {
        res++;
        tmp = tmp >> 1;
    }

    return res - 1;
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
template<typename T>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ExtendBucketSize(
        T& extendedBucketSize, T& aOffsetNumStages, const T bucketSize)
{
    uint32_t lg2 = ScalarLog2(bucketSize);
    uint32_t possibleEtendedBucketSize = 1 << lg2;
    if (possibleEtendedBucketSize == bucketSize) {
        aOffsetNumStages = lg2;
        extendedBucketSize = bucketSize;
    } else {
        aOffsetNumStages = lg2 + 1;
        extendedBucketSize = 1 << (aOffsetNumStages);
    }
}

template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
__aicore__ inline void JacobiBase<AType, SType, UType, VType, VEC_INSTRUCTION_SET_SIZE>::ApplyBucket(
        const LocalTensor<float>& bucketSpace, const LocalTensor<float>& norms, const uint32_t bucketSize,
        const bool calculateNorms)
{
    if (bucketSize < 2) {
        if (calculateNorms) {
            RotateMaskInfo maskInfo;
            maskInfo.numPairs = bucketSize;
            for (int idx = 0; idx < bucketSize; ++idx) {
                maskInfo.leftRowsIds[idx] = idx;
            }
            CalculateNormOfBucket(norms, bucketSpace, tmpMemSpace, tmpMaskSpace, maskInfo.leftRowsIds, bucketSize);
            PipeBarrier<PIPE_V>();
        }
        return;
    }
    uint32_t extendedBucketSize = 0;
    uint32_t aOffsetNumStages = 0;
    ExtendBucketSize(extendedBucketSize, aOffsetNumStages, bucketSize);
    RotateMaskInfo maskInfo;
    bool isCalculateNorm = calculateNorms;

    for (uint32_t aOffsetStage = 0; aOffsetStage < aOffsetNumStages; ++aOffsetStage) {
        uint32_t aOffsetNumPhases = 1 << aOffsetStage;

        for (uint32_t aOffsetPhase = 0; aOffsetPhase < aOffsetNumPhases; ++aOffsetPhase) {
            maskInfo.numPairs = 0;

            uint32_t groupSize = aOffsetNumPhases * 2;
            uint32_t pairsInGroup = groupSize / 2;
            maskInfo.leftMaskU64Val = 0;
            maskInfo.rightMaskU64Val = 0;
            maskInfo.normUpdateMask = 0;
            for (uint16_t pairID = 0; pairID < extendedBucketSize / 2; ++pairID) {
                uint32_t groupID = pairID / pairsInGroup;
                uint32_t offsetInGroup = pairID % pairsInGroup;

                uint32_t leftRowIdx = groupID * groupSize + offsetInGroup;
                uint32_t rightRowIdx = groupID * groupSize + pairsInGroup +
                        (offsetInGroup + aOffsetPhase) % aOffsetNumPhases;
                if ((leftRowIdx < bucketSize) && (rightRowIdx < bucketSize)) {
                    maskInfo.numPairs++;
                    maskInfo.leftRowsIds[maskInfo.numPairs - 1] = static_cast<uint8_t>(leftRowIdx);
                    maskInfo.leftMaskU8Val[maskInfo.numPairs - 1] = 1 << leftRowIdx;
                    maskInfo.rightRowsIds[maskInfo.numPairs - 1] = static_cast<uint8_t>(rightRowIdx);
                    maskInfo.rightMaskU8Val[maskInfo.numPairs - 1] = 1 << rightRowIdx;
                    maskInfo.normUpdateMask |= 1 << leftRowIdx;
                    maskInfo.normUpdateMask |= 1 << rightRowIdx;
                }
            }

            maskInfo.leftBucketSize = maskInfo.rightBucketSize = bucketSize;
            if (!isCalculateNorm) {
                RotateSelfBucket<false>(bucketSpace, norms, bucketSize, maskInfo);
            } else {
                RotateSelfBucket<true>(bucketSpace, norms, bucketSize, maskInfo);
            }
            isCalculateNorm = false;
        }
    }
}
#endif // JACOBI_IMPL_PART1_H