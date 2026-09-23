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
 * \file jacobi_base.h
 * \brief
 */

#ifndef JACOBI_BASE_H
#define JACOBI_BASE_H

#ifdef DEBUG_MODE
#define PRINT_VAL(X)   printf("Variable %s has value %d\n", #X, (int)X)
#else
#define PRINT_VAL(X)
#endif

#define FAST_SET_WAIT_FLAG(EVENT_NAME, SET_WAIT_UNITS) event_t EVENT_NAME = \
                    static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::SET_WAIT_UNITS)); \
                    SetFlag<HardEvent::SET_WAIT_UNITS>(EVENT_NAME); \
                    WaitFlag<HardEvent::SET_WAIT_UNITS>(EVENT_NAME)

#define MIN(A, B) (((A)<(B))?(A):(B))

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SVD {

    template<typename T>
    constexpr __aicore__ T GET_MIN_BLOCK_SIZE()
    {
        constexpr uint32_t MIN_BLOCK_SIZE = 32;
        return static_cast<T>(MIN_BLOCK_SIZE);
    }

    template<typename SourceType, typename ResultType = uint8_t>
    constexpr __aicore__ ResultType SMALL_MASK_NUM_ELEMENTS()
    {
        return static_cast<ResultType>(GET_MIN_BLOCK_SIZE<size_t>() / sizeof(SourceType));
    }

    template<typename T>
    constexpr __aicore__ T GET_MAX_BLOCK_SIZE()
    {
        constexpr uint32_t MIN_BLOCK_SIZE = 256;
        return static_cast<T>(MIN_BLOCK_SIZE);
    }

    template<typename SourceType, typename ResultType = uint8_t>
    constexpr __aicore__ uint8_t BIG_MASK_NUM_ELEMENTS()
    {
        return static_cast<ResultType>(GET_MAX_BLOCK_SIZE<size_t>() / sizeof(SourceType));
    }

    using namespace AscendC;

    template<typename AType, typename SType, typename UType, typename VType, uint8_t VEC_INSTRUCTION_SET_SIZE>
    class JacobiBase {
    public:
        __aicore__ inline JacobiBase(TPipe* pipe);
        __aicore__ inline void Init(__gm__ uint8_t* a, __gm__ uint8_t* s,
            __gm__ uint8_t* u, __gm__ uint8_t* v, __gm__ uint8_t* workspace,
            const JMTilingData* tilingData);
        __aicore__ inline void Process();

    private:
        struct RotateMaskInfo;
        __aicore__ inline void AllocMem();

        __aicore__ inline void ProcessImpl();

        __aicore__ inline void JacobiProcess();

        __aicore__ inline void LocalPhaseProcess(const uint16_t leftGlobalSetId, const uint16_t rightGlobalSetId,
                const bool isSaveUV, const bool isZeroStage, const bool isZeroIter);

        __aicore__ inline void GetLeftRightGlobalSetIds(uint16_t& dstLeftGlobalSetId, uint16_t& dstRightGlobalSetId,
                const uint32_t phaseID, const uint32_t numPhases, const uint32_t coreID, const uint32_t numCores);

        __aicore__ inline void ApplyGlobalSet(const uint16_t lBegin, const uint16_t lEnd, const uint16_t rBegin,
                const uint16_t rEnd, const bool isFinalIteration = false);

        __aicore__ inline void ApplyGlobalSet(const uint16_t sBegin, const uint16_t sEnd,
                const bool isFirstLoadData = false);

        __aicore__ inline void ApplyBucket(const LocalTensor<float>& leftBucketSpace,
                const LocalTensor<float>& leftNorms, const LocalTensor<float>& rightBucketSpace,
                const LocalTensor<float>& rightNorms, const uint32_t leftBucketSize,
                const uint32_t rightBucketSize, const bool calculateRightNorms);

        template<typename T = uint32_t>
        __aicore__ inline T  ScalarLog2(const T val);

        template<typename T>
        __aicore__ inline void ExtendBucketSize(T& extendedBucketSize, T& aOffsetNumStages, const T bucketSize);

        __aicore__ inline void ApplyBucket(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& norms,
                const uint32_t bucketSize, const bool calculateNorms);

        template<bool BUCKET_1_TARGET = true, bool NEED_CALCULATE_RIGHT_NORM = false>
        __aicore__ inline void CalculateDotProducts(const LocalTensor<float>& sqNormsBucket1,
                const LocalTensor<float>& sqNormsBucket2, const LocalTensor<float>& dotProducts,
                const LocalTensor<float>& bucket1, const LocalTensor<float>& norms1,
                const LocalTensor<float>& bucket2, const LocalTensor<float>& norms2, const RotateMaskInfo& maskInfo);

        __aicore__ inline void CalcNorms(const LocalTensor<float>& norms1, const LocalTensor<float>& norms2,
                const LocalTensor<float>& bucket1, const LocalTensor<float>& bucket2, const RotateMaskInfo& maskInfo,
                const LocalTensor<float>& tmpBucket1, const LocalTensor<float>& tmpBucket2,
                const LocalTensor<float>& dotProductTmp1, const LocalTensor<float>& dotProductTmp2);

        __aicore__ inline void CalculateRightNormAndDotProductForDoubleBucketsCase(
                const LocalTensor<float>& dstDotProduct, const LocalTensor<float>& dstRightNorms,
                const LocalTensor<float>& bucket1, const LocalTensor<float>& bucket2, const RotateMaskInfo& maskInfo,
                const LocalTensor<float>& tmpBucket1, const LocalTensor<float>& tmpBucket2,
                const LocalTensor<float>& dotProductTmp1, const LocalTensor<float>& dotProductTmp2);

        __aicore__ inline void CalculateDotProductForDoubleBucketCase(const LocalTensor<float>& dstDotProduct,
                const LocalTensor<float>& bucket1, const LocalTensor<float>& bucket2, const RotateMaskInfo& maskInfo,
                const LocalTensor<float>& tmpBucket1, const LocalTensor<float>& dotProductTmp1);

        __aicore__ inline void CalculateNormOfBucket(const LocalTensor<float>& dstNormVector,
                const LocalTensor<float>& bucket, const LocalTensor<float>& tmpBucket,
                const LocalTensor<float>& dotProductTmp, const uint8_t(&rowIds)[sizeof(uint64_t)],
                const uint32_t bucketSize);

        template<bool NEED_CALCULATE_NORM>
        __aicore__ inline void CalculateDotProductsForSelfBucket(const LocalTensor<float>& sqNorms1,
                const LocalTensor<float>& sqNorms2, const LocalTensor<float>& dotProducts,
                const LocalTensor<float>& normVector, const LocalTensor<float>& bucket, const uint32_t bucketSize,
                const RotateMaskInfo& maskInfo);

        __aicore__ inline void GatherNormVector(const LocalTensor<float>& dstNormVector,
                const LocalTensor<float>& srcNormVector, const uint64_t mask, const LocalTensor<float>& workLocal1,
                const LocalTensor<float>& workLocal2);

        __aicore__ inline void GatherNormsVectorToLeftRightNorms(const LocalTensor<float>& dstSqNorms1,
                const LocalTensor<float>& dstSqNorms2, const LocalTensor<float>& srcNorms,
                const RotateMaskInfo& maskInfo, const LocalTensor<float>& workLocal1,
                const LocalTensor<float>& workLocal2);

        __aicore__ inline void UpdateSquareNorms(const LocalTensor<float>& leftSquareNorms,
                const LocalTensor<float>& rightSquareNorms, const LocalTensor<float>& dotProducts,
                const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor);

        template<bool LEFT_TARGET = true, bool NEED_CALCULATE_RIGHT_NORM = false>
        __aicore__ inline void RotateBucket(const LocalTensor<float>& leftBucketSpace,
                const LocalTensor<float>& rightBucketSpace, const LocalTensor<float>& leftNorms,
                const LocalTensor<float>& rightNorms, const uint32_t leftBucketSize, const uint32_t rightBucketSize,
                const RotateMaskInfo& maskInfo);

        template<bool LEFT_TARGET>
        __aicore__ inline void RotateNorms(const LocalTensor<float>& leftSquareNorms,
                const LocalTensor<float>& rightSquareNorms, const LocalTensor<float>& dotProducts,
                const LocalTensor<float>& leftNorms, const LocalTensor<float>& rightNorms,
                const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor,
                const RotateMaskInfo& maskInfo);

        // Self bucket rotate
        template<bool NEED_CALCULATE_NORM>
        __aicore__ inline void RotateSelfBucket(const LocalTensor<float>& bucketSpace,
                const LocalTensor<float>& normVector, const uint32_t bucketSize, const RotateMaskInfo& maskInfo);

        __aicore__ inline void RotateSelfNorms(const LocalTensor<float>& leftSquareNorms,
                const LocalTensor<float>& rightSquareNorms, const LocalTensor<float>& dotProducts,
                const LocalTensor<float>& normVector, const LocalTensor<float>& sinTensor,
                const LocalTensor<float>& cosTensor, const RotateMaskInfo& maskInfo);

        __aicore__ inline void ApplySinCosForRowSet(const LocalTensor<float>& bucket1,
                const LocalTensor<float>& bucket2, const uint32_t bucketSize,
                const LocalTensor<float>& sinBroadcastedValues, const LocalTensor<float>& cosBroadcastedValues,
                const LocalTensor<float>& calcTmpTensor1, const LocalTensor<float>& calcTmpTensor2,
                const RotateMaskInfo& maskInfo, const uint32_t rowLength, const uint8_t numRepeats);

        template<bool BUCKET_1_TARGET = true>
        __aicore__ inline void  ApplySinCos(const LocalTensor<float>& bucket1, const uint32_t bucket1Size,
                const LocalTensor<float>& bucket2, const uint32_t bucket2Size, const LocalTensor<float>& sinTensor,
                const LocalTensor<float>& cosTensor, const RotateMaskInfo& maskInfo);

        template<bool BUCKET_1_TARGET = true>
        __aicore__ inline void  ApplySinCosForSelfBucket(const LocalTensor<float>& bucket, const uint32_t bucketSize,
                const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor,
                const RotateMaskInfo& maskInfo);

        __aicore__ inline void CleanTailThrash(const LocalTensor<float>& bucketSpace, const uint32_t bucketBegin,
                const uint32_t bucketEnd);

        __aicore__ inline void InitV(const LocalTensor<float>& bucketSpace, const uint32_t bucketBegin,
                const uint32_t bucketEnd);

        __aicore__ inline void LoadBucket(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector,
                const uint32_t bucketBegin, const uint32_t bucketEnd, const bool isFirstLoadDataLocal);
        
        __aicore__ inline void SaveBucket(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector,
                const uint32_t bucketBegin, const uint32_t bucketEnd);

        template<bool IS_LEFT_BUCKET_SAVING>
        __aicore__ inline void SaveOutputs(const LocalTensor<float>& bucketSpace, const LocalTensor<float>& normVector,
                const uint32_t bucketBegin, const uint32_t bucketEnd);

        __aicore__ inline void CalculateSinCosConvertTaoToArg(const LocalTensor<float>& tauTensor,
                const LocalTensor<float>& tTensor);
        
        __aicore__ inline void CalculateSinCosValues(const LocalTensor<float>& tTensor,
                const LocalTensor<float>& s1Tensor, const LocalTensor<float>& s2Tensor,
                const LocalTensor<uint8_t>& normSquareDiffIsPositive);

        __aicore__ inline void CalculateSinCos(const LocalTensor<float>& leftSquareNorms,
                const LocalTensor<float>& rightSquareNorms, const LocalTensor<float>& dotProducts,
                const LocalTensor<float>& sinTensor, const LocalTensor<float>& cosTensor);
        
    private:
        // constexpr functions
        template<typename DST_TYPE>
        constexpr __aicore__ DST_TYPE VEC_INSTRUCTION_NUM()
        {
            return static_cast<DST_TYPE>(VEC_INSTRUCTION_SET_SIZE / sizeof(float));
        }
    private:
        static constexpr float NEAR_ZERO_VALUE = 0.0000000001f;
    private:
        struct RotateMaskInfo {
            uint8_t leftRowsIds[sizeof(uint64_t)];
            uint8_t rightRowsIds[sizeof(uint64_t)];
            uint64_t normUpdateMask = 0;
            union {
                uint64_t leftMaskU64Val;
                uint8_t leftMaskU8Val[sizeof(uint64_t)];
            };
            union {
                uint64_t rightMaskU64Val;
                uint8_t rightMaskU8Val[sizeof(uint64_t)];
            };
            uint8_t numPairs = 0;
            uint8_t leftBucketSize = 0;
            uint8_t rightBucketSize = 0;
        };
        static constexpr uint64_t INTERNAL_SYNC_ALL = 0x1;
        TPipe* pipe_;
        GlobalTensor<AType> aGm;
        GlobalTensor<AType> aTmpGm;
        GlobalTensor<AType> sGm;
        GlobalTensor<AType> uGm;
        GlobalTensor<AType> vGm;
        uint32_t coreIdx, numAvailableVectorCores;
        uint32_t globSetMaxSize, globSetMinSize, numGlobalSetsWithMaxSize, numGlobalSetsWithMinSize, globSetMinOffset;
        uint32_t numIterations, numStages, numPhases, numGlobalSets, leftGlobalSetBegin, leftGlobalSetEnd,
                rightGlobalSetBegin, rightGlobalSetEnd;
        uint32_t mSize, nSize, nSizeAligned, uMSize, uNSize, uNSizeAligned, vMSize, vNSize, vNSizeAligned, sMNSize,
                sMNSizeAligned, recordSize, recordSizeAligned;
        int32_t batchSize;
        uint64_t batchOffsetV, batchOffsetU, batchOffsetS, batchOffsetI, batchOffsetTmp;
        uint32_t rowSize, rowSizeAligned, rowMask, rowRepeatsNum, rowTail, maxBucketSize, ubSize, uRepeatsNum,
                uTailSize, tmpRowSize, tmpRowSizeAligned;
        TBuf<TPosition::VECCALC> ubBuf;
        LocalTensor<uint8_t> ubMemory;
        LocalTensor<int32_t> arithmProgressive;
        LocalTensor<float>  tmpMemSpace, tmpMaskSpace, bucketSpace1, bucketSpace2, bucketSpace3;
        LocalTensor<float> tmpBlock1, tmpBlock2, tmpBlock3, tmpBlock4, tmpBlock5,
                tmpBlock6, tmpBlock7, tmpBlock8, tmpBlock9, tmpBlock10, tmpBlock11, tmpBlock12, tmpBlock13,
                tmpBlock14, tmpBlock15, tmpBlock16, tmpBlock17, tmpBlock18, tmpBlock19, tmpBlock20, tmpBlock21,
                tmpBlock22, tmpBlock23, tmpBlock24, tmpBlock25, tmpBlock26, tmpBlock27, tmpBlock28, tmpBlock29,
                tmpBlock30, tmpBlock31, tmpBlock32, tmpBlockSpace;
        bool needSort;
    };

    #include "jacobi_impl_part1.h"
    #include "jacobi_impl_part2.h"
    #include "jacobi_impl_part3.h"
}
#endif // JACOBI_BASE_H