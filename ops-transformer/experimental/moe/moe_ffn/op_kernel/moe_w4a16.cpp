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
 * \file moe_w4a16.cpp
 * \brief
 */

#include "moe_w4a16.h"
#include <cmath>

using namespace AscendC;

template <typename T>
__aicore__ inline void MOEW4A16<T>::Init(GM_ADDR x, GM_ADDR quantized_weight_13, GM_ADDR weight_scale_13,
                                         GM_ADDR weight_offset_13, GM_ADDR quantized_weight_2, GM_ADDR weight_scale_2,
                                         GM_ADDR weight_offset_2, GM_ADDR expanded_expert_idx, GM_ADDR sorted_row_idx,
                                         GM_ADDR expanded_row_idx, GM_ADDR topk_weights, GM_ADDR y,
                                         GM_ADDR usrworkspace, const MoeFFNTilingData *__restrict moeTiling)
{
    tilingData = moeTiling;
    block_id = get_block_idx();
    subblock_id = get_subblockid();
    block_id_ = block_id + subblock_id * tilingData->CoreNum;

    InitEventId();

    InitGlobalTensors(x, quantized_weight_13, weight_scale_13, weight_offset_13, quantized_weight_2, weight_scale_2,
                      weight_offset_2, expanded_expert_idx, sorted_row_idx, expanded_row_idx, topk_weights, y);
    InitWorkspaceTensors(usrworkspace);

    InitLocalTensors();
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::InitEventId()
{
    if ASCEND_IS_AIV {
        eventIdMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE3>());
        eventIdMTE2ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_S>());
        eventIdMTE3ToS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>());
        eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        eventIdMTE3ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        eventIdMTE3ToV[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
        eventIdVToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdVToMTE2[2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
        eventIdMTE2ToV[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE2ToV[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE2ToV[2] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        eventIdMTE3ToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
        eventIdMTE3ToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    } else {
        eventIdMTE1ToMTE2[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE1ToMTE2[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        eventIdMTE2ToMTE1 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
        eventIdMTE1ToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
        eventIdMToMTE1[0] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToMTE1[1] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        eventIdMToFIX = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_FIX>());
        eventIdFIXToM = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::FIX_M>());
    }
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::InitWorkspaceTensors(GM_ADDR usrWorkspace)
{
    groupListGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(usrWorkspace), tilingData->originE);
    usrWorkspace += (tilingData->originE * 4 + 512 - 1) / 512 * 512;

    uint32_t workspaceBaseN = tilingData->L0ASize / 2 / sizeof(T);
    workspaceWGm[0][0].SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(usrWorkspace + tilingData->L0ASize / 2 * block_id * 2 * MY_BUFFER_NUM),
        workspaceBaseN);
    workspaceWGm[1][0].SetGlobalBuffer(
        reinterpret_cast<__gm__ T *>(usrWorkspace + tilingData->L0ASize / 2 * (block_id * 2 + 1) * MY_BUFFER_NUM),
        workspaceBaseN);
    for (int32_t i = 1; i < MY_BUFFER_NUM; i++) {
        workspaceWGm[0][i] = workspaceWGm[0][i - 1][workspaceBaseN];
        workspaceWGm[1][i] = workspaceWGm[1][i - 1][workspaceBaseN];
    }
    usrWorkspace += tilingData->L0ASize * tilingData->CoreNum * MY_BUFFER_NUM;

    MM1InputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(usrWorkspace), expandedM * tilingData->originK_13);
    MM2OutputGm = MM1InputGm;
    usrWorkspace += (expandedM * tilingData->originK_13 * 2 + 512 - 1) / 512 * 512;

    MM1OutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(usrWorkspace), expandedM * tilingData->originN_13);
    usrWorkspace += (expandedM * tilingData->originN_13 * 4 + 512 - 1) / 512 * 512;

    MM2InputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(usrWorkspace), expandedM * tilingData->originK_2);
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::InitGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight_13, GM_ADDR weight_scale_13,
                                                      GM_ADDR weight_offset_13, GM_ADDR quantized_weight_2,
                                                      GM_ADDR weight_scale_2, GM_ADDR weight_offset_2,
                                                      GM_ADDR expanded_expert_idx, GM_ADDR sorted_row_idx,
                                                      GM_ADDR expanded_row_idx, GM_ADDR topk_weights, GM_ADDR y)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x), tilingData->originM * tilingData->originK_13);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y), tilingData->originM * tilingData->originN_2);
    wGm_13.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(quantized_weight_13),
                           tilingData->originE * tilingData->originK_13 * tilingData->originN_13 / 8);
    wGm_13.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    wScaleGm_13.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight_scale_13),
                                tilingData->originE * tilingData->scaleK_13 * tilingData->originN_13);
    wOffsetGm_13.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight_offset_13),
                                 tilingData->originE * tilingData->scaleK_13 * tilingData->originN_13);
    wGm_2.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(quantized_weight_2),
                          tilingData->originE * tilingData->originK_2 * tilingData->originN_2 / 8);
    wScaleGm_2.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight_scale_2),
                               tilingData->originE * tilingData->scaleK_2 * tilingData->originN_2);
    wOffsetGm_2.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight_offset_2),
                                tilingData->originE * tilingData->scaleK_2 * tilingData->originN_2);
    expandedM = tilingData->originM * tilingData->topK;
    expertIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expanded_expert_idx), expandedM);
    rowIdxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(sorted_row_idx), expandedM);
    rowIdxGmRev.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(expanded_row_idx), expandedM);
    topKwGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(topk_weights), expandedM);
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::InitLocalTensors()
{
    if ASCEND_IS_AIV {
        pipe.InitBuffer(UbBuf, tilingData->UBSize);
        ubBase = UbBuf.Get<uint8_t>();
    }
    if ASCEND_IS_AIC {
        uint32_t baseNumber = tilingData->L0ASize / 2 / sizeof(T);
        pipe.InitBuffer(L1Buf, tilingData->L1Size);
        pipe.InitBuffer(L0ABuf, tilingData->L0ASize);
        pipe.InitBuffer(L0BBuf, tilingData->L0BSize);
        pipe.InitBuffer(L0CBuf, tilingData->L0CSize);
    }
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::PrepareMMCommon()
{
    if ASCEND_IS_AIV {
        uint32_t offset = 0;
        ubWFloat = ubBase.ReinterpretCast<float>();
        ubWHalf_ = ubBase.ReinterpretCast<half>();

        offset = 64 * 1024;
        ubWFloat_ = ubBase[offset].ReinterpretCast<float>();
        ubWHalf = ubBase[offset].ReinterpretCast<half>();

        offset = 128 * 1024;
        ubWInt4 = ubBase[offset].ReinterpretCast<int4b_t>();
        offset += 8 * 1024;
        ubWT = ubBase[offset].ReinterpretCast<T>();
        offset += 32 * 1024;
        ubWScaleT = ubBase[offset].ReinterpretCast<T>();
        offset += 2 * 1024;
        ubWOffsetHalf = ubBase[offset].ReinterpretCast<half>();
        offset += 2 * 1024;
        offset += 32;
        ubWScaleFloat = ubBase[offset].ReinterpretCast<float>();
    }
    if ASCEND_IS_AIC {
        uint32_t baseNumber = tilingData->L0ASize / 2 / sizeof(T);
        L1MatB[0] = L1Buf.Get<T>();
        L1MatB[1] = L1MatB[0][baseNumber];
        L1MatA[0] = L1MatB[1][baseNumber];
        L1MatA[1] = L1MatA[0][baseNumber];
        L0MatA[0] = L0ABuf.Get<T>();
        L0MatA[1] = L0MatA[0][baseNumber];
        L0MatB[0] = L0BBuf.Get<T>();
        L0MatB[1] = L0MatB[0][baseNumber];
        L0MatC = L0CBuf.Get<float>();
    }
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::ComputeMMTiling()
{
    mmAGm = MM2InputGm;
    mmCGm = MM2OutputGm;
    mmBGm0 = wGm_2;
    mmScaleGm0 = wScaleGm_2;
    mmOffsetGm0 = wOffsetGm_2;
    fracK = tilingData->fracK_2;
    fracN = tilingData->fracN_2;
    scaleK = tilingData->scaleK_2;
    needBias = tilingData->needBias_2;
    NZFormat = false;
    totalTaskNum = 0;
    // 64 * 1024 / (16 * 16 * 2) / 2 = 64
    // 128 * 1024 / (16 * 16 * 4) = 128
    baseNNumA = (fracN + 16 - 1) / 16;
    uint32_t baseNA = (fracN + baseNNumA - 1) / baseNNumA;
    baseMLimitA = 128 / baseNA;
    uint32_t baseKLimit = 64 / (baseNA > baseMLimitA ? baseNA : baseMLimitA);
    baseKNum = (fracK + baseKLimit - 1) / baseKLimit;
    baseNNumB = baseNNumA;
    for (int32_t i = 0; i < tilingData->originE; i++) {
        if (realM[i] == 0) {
            baseMNum[i] = 0;
            continue;
        }
        uint32_t fracM = (realM[i] + 16 - 1) / 16;
        baseMNum[i] = (fracM + baseMLimitA - 1) / baseMLimitA;
        totalTaskNum += (baseMNum[i] * baseNNumA);
    }

    MMtaskNum = totalTaskNum / tilingData->CoreNum + (block_id < (totalTaskNum % tilingData->CoreNum) ? 1 : 0);
    MMtaskOffset = totalTaskNum / tilingData->CoreNum * block_id +
                   (block_id < (totalTaskNum % tilingData->CoreNum) ? block_id : (totalTaskNum % tilingData->CoreNum));
    MMtaskNum *= baseKNum;
    MMtaskOffset *= baseKNum;
}

template <>
__aicore__ inline void MOEW4A16<half>::AntiQuantWeight(uint32_t task_id, uint32_t ping_pong, bool endKFlag,
                                                       uint32_t baseN, uint32_t baseK, uint32_t offsetN,
                                                       uint32_t offsetK)
{
    uint32_t scaleBegin = offsetK * 16 / tilingData->scaleGroupSize;
    uint32_t scaleEnd = ((offsetK + baseK) * 16 - 1) / tilingData->scaleGroupSize;
    UnaryRepeatParams unaryParams16To32, unaryParams32To16, unaryParams4To16;
    unaryParams16To32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams32To16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams4To16.srcRepStride = ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
    DataCopyParams repeatParamsInt4Weight;
    repeatParamsInt4Weight.blockCount = baseK;
    repeatParamsInt4Weight.blockLen = baseN * 4;
    repeatParamsInt4Weight.srcStride = (fracN - baseN) * 4;
    repeatParamsInt4Weight.dstStride = 0;
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    DataCopy<int32_t>(ubWInt4.ReinterpretCast<int32_t>(), mmBGm[((offsetK * fracN) + offsetN) * 16 * 2],
                      repeatParamsInt4Weight);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    DataCopyParams repeatParamsScale;
    repeatParamsScale.blockCount = (scaleEnd - scaleBegin + 1);
    repeatParamsScale.blockLen = baseN;
    repeatParamsScale.srcStride = (fracN - baseN);
    repeatParamsScale.dstStride = 0;
    if (needBias) {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        DataCopy<half>(ubWOffsetHalf, mmOffsetGm[scaleBegin * fracN * 16 + offsetN * 16], repeatParamsScale);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
    }

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
    DataCopy<half>(ubWScaleT, mmScaleGm[scaleBegin * fracN * 16 + offsetN * 16], repeatParamsScale);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    Cast<half, int4b_t, false>(ubWHalf, ubWInt4, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams4To16);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetMaskNorm();
    ResetMask();

    BinaryRepeatParams binaryParams;
    binaryParams.dstBlkStride = 1;
    binaryParams.src0BlkStride = 1;
    binaryParams.src1BlkStride = 0;
    binaryParams.dstRepStride = 16;
    binaryParams.src0RepStride = 16;
    binaryParams.src1RepStride = 1;
    uint32_t mulOffsetA = 0;
    uint32_t mulOffsetB = 0;

    if (needBias) {
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < baseK; i++) {
            Add<half, false>(ubWHalf[mulOffsetB], ubWHalf[mulOffsetB], ubWOffsetHalf[mulOffsetA], MASK_PLACEHOLDER,
                             baseN, binaryParams);
            Add<half, false>(ubWHalf[mulOffsetB + 16 * 8], ubWHalf[mulOffsetB + 16 * 8], ubWOffsetHalf[mulOffsetA],
                             MASK_PLACEHOLDER, baseN, binaryParams);
            bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
            mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
            mulOffsetB += 16 * 16 * baseN;
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    }

    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);
    PipeBarrier<PIPE_V>();
    mulOffsetA = 0;
    mulOffsetB = 0;
    for (int32_t i = 0; i < baseK; i++) {
        Mul<half, false>(ubWT[mulOffsetB], ubWHalf[mulOffsetB], ubWScaleT[mulOffsetA], MASK_PLACEHOLDER, baseN,
                         binaryParams);
        Mul<half, false>(ubWT[mulOffsetB + 16 * 8], ubWHalf[mulOffsetB + 16 * 8], ubWScaleT[mulOffsetA],
                         MASK_PLACEHOLDER, baseN, binaryParams);
        bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
        mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
        mulOffsetB += 16 * 16 * baseN;
    }
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    if ((task_id / 2) == 0)
        wait_flag_dev(ping_pong ? SYNC_PING_AIC_AIV_FLAG : SYNC_PONG_AIC_AIV_FLAG);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    DataCopyParams repeatParamsBF16Weight;
    repeatParamsBF16Weight.blockLen = baseK * baseN * 16;
    DataCopy<half>(workspaceWGm[ping_pong][task_id], ubWT, repeatParamsBF16Weight);

    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    if (((task_id / 2 + 1) == MY_BUFFER_NUM / 2) || endKFlag)
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIV_AIC_FLAG : SYNC_PONG_AIV_AIC_FLAG));
}

template <>
__aicore__ inline void MOEW4A16<bfloat16_t>::AntiQuantWeight(uint32_t task_id, uint32_t ping_pong, bool endKFlag,
                                                             uint32_t baseN, uint32_t baseK, uint32_t offsetN,
                                                             uint32_t offsetK)
{
    uint32_t scaleBegin = offsetK * 16 / tilingData->scaleGroupSize;
    uint32_t scaleEnd = ((offsetK + baseK) * 16 - 1) / tilingData->scaleGroupSize;
    UnaryRepeatParams unaryParams16To32, unaryParams32To16, unaryParams4To16;
    unaryParams16To32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams32To16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams4To16.srcRepStride = ONE_FOURTH_DEFAULT_REPEAT_STRIDE;
    DataCopyParams repeatParamsInt4Weight;
    repeatParamsInt4Weight.blockCount = baseK;
    repeatParamsInt4Weight.blockLen = baseN * 4;
    repeatParamsInt4Weight.srcStride = (fracN - baseN) * 4;
    repeatParamsInt4Weight.dstStride = 0;
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    DataCopy<int32_t>(ubWInt4.ReinterpretCast<int32_t>(), mmBGm[((offsetK * fracN) + offsetN) * 16 * 2],
                      repeatParamsInt4Weight);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

    DataCopyParams repeatParamsScale;
    repeatParamsScale.blockCount = (scaleEnd - scaleBegin + 1);
    repeatParamsScale.blockLen = baseN;
    repeatParamsScale.srcStride = (fracN - baseN);
    repeatParamsScale.dstStride = 0;
    if (needBias) {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        DataCopy<half>(ubWOffsetHalf, mmOffsetGm[scaleBegin * fracN * 16 + offsetN * 16], repeatParamsScale);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
    }

    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
    DataCopy<bfloat16_t>(ubWScaleT, mmScaleGm[scaleBegin * fracN * 16 + offsetN * 16], repeatParamsScale);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    if (needBias) {
        Cast<half, int4b_t, false>(ubWHalf_, ubWInt4, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams4To16);
    } else {
        Cast<half, int4b_t, false>(ubWHalf, ubWInt4, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams4To16);
    }
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);


    BinaryRepeatParams binaryParams;
    uint32_t mulOffsetA = 0;
    uint32_t mulOffsetB = 0;

    if (needBias) {
        SetMaskNorm();
        ResetMask();

        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[1]);
        PipeBarrier<PIPE_V>();
        binaryParams.dstBlkStride = 1;
        binaryParams.src0BlkStride = 1;
        binaryParams.src1BlkStride = 0;
        binaryParams.dstRepStride = 16;
        binaryParams.src0RepStride = 16;
        binaryParams.src1RepStride = 1;
        for (int32_t i = 0; i < baseK; i++) {
            Add<half, false>(ubWHalf[mulOffsetB], ubWHalf_[mulOffsetB], ubWOffsetHalf[mulOffsetA], MASK_PLACEHOLDER,
                             baseN, binaryParams);
            Add<half, false>(ubWHalf[mulOffsetB + 16 * 8], ubWHalf_[mulOffsetB + 16 * 8], ubWOffsetHalf[mulOffsetA],
                             MASK_PLACEHOLDER, baseN, binaryParams);
            bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
            mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
            mulOffsetB += 16 * 16 * baseN;
        }
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        SetMaskCount();
        SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    }
    PipeBarrier<PIPE_V>();
    Cast<float, half, false>(ubWFloat, ubWHalf, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1, unaryParams16To32);

    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[2]);
    SetVectorMask<half, MaskMode::COUNTER>((scaleEnd - scaleBegin + 1) * baseN * 16);
    Cast<float, bfloat16_t, false>(ubWScaleFloat, ubWScaleT, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                                   unaryParams16To32);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);

    PipeBarrier<PIPE_V>();
    SetMaskNorm();
    ResetMask();
    binaryParams.dstBlkStride = 2;
    binaryParams.src0BlkStride = 2;
    binaryParams.src1BlkStride = 0;
    binaryParams.dstRepStride = 32;
    binaryParams.src0RepStride = 32;
    binaryParams.src1RepStride = 2;
    mulOffsetA = 0;
    mulOffsetB = 0;
    for (int32_t i = 0; i < baseK; i++) {
        Mul<float, false>(ubWFloat_[mulOffsetB], ubWFloat[mulOffsetB], ubWScaleFloat[mulOffsetA], MASK_PLACEHOLDER,
                          baseN, binaryParams);
        Mul<float, false>(ubWFloat_[mulOffsetB + 16 * 8], ubWFloat[mulOffsetB + 16 * 8], ubWScaleFloat[mulOffsetA],
                          MASK_PLACEHOLDER, baseN, binaryParams);
        Mul<float, false>(ubWFloat_[mulOffsetB + 8], ubWFloat[mulOffsetB + 8], ubWScaleFloat[mulOffsetA + 8],
                          MASK_PLACEHOLDER, baseN, binaryParams);
        Mul<float, false>(ubWFloat_[mulOffsetB + 16 * 8 + 8], ubWFloat[mulOffsetB + 16 * 8 + 8],
                          ubWScaleFloat[mulOffsetA + 8], MASK_PLACEHOLDER, baseN, binaryParams);
        bool tmp = ((offsetK + i + 1) * 16) % tilingData->scaleGroupSize == 0;
        mulOffsetA += baseN * 16 * (tmp ? 1 : 0);
        mulOffsetB += 16 * 16 * baseN;
    }

    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    PipeBarrier<PIPE_V>();
    SetMaskCount();
    SetVectorMask<half, MaskMode::COUNTER>(baseN * baseK * 16 * 16);
    Cast<bfloat16_t, float, false>(ubWT, ubWFloat_, RoundMode::CAST_RINT, MASK_PLACEHOLDER, 1, unaryParams32To16);
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    SetMaskNorm();
    ResetMask();

    if ((task_id / 2) == 0)
        wait_flag_dev(ping_pong ? SYNC_PING_AIC_AIV_FLAG : SYNC_PONG_AIC_AIV_FLAG);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    DataCopyParams repeatParamsBF16Weight;
    repeatParamsBF16Weight.blockLen = baseK * baseN * 16;
    DataCopy<bfloat16_t>(workspaceWGm[ping_pong][task_id], ubWT, repeatParamsBF16Weight);

    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    if (((task_id / 2 + 1) == MY_BUFFER_NUM / 2) || endKFlag)
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIV_AIC_FLAG : SYNC_PONG_AIV_AIC_FLAG));
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::ComputeMatMul(uint32_t task_id, uint32_t ping_pong, bool startKFlag, bool endKFlag,
                                                  bool needSetFlag, uint32_t realM, uint32_t baseM, uint32_t baseN,
                                                  uint32_t baseK, uint32_t offsetM, uint32_t offsetN, uint32_t offsetK)
{
    MmadParams mmadParams;
    LoadData2dParams loadDataA, loadDataB;
    DataCopyParams repeatParamsB, repeatParamsA_;
    Nd2NzParams repeatParamsA;
    uint32_t pingpong = task_id % 2;
    WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[pingpong]);
    if (NZFormat) {
        repeatParamsA_.blockCount = baseM;
        if ((baseM + offsetM) * 16 > realM) {
            repeatParamsA_.blockCount -= 1;
        }
        repeatParamsA_.blockLen = baseK * 16;
        repeatParamsA_.srcStride = (fracK - baseK) * 16;
        repeatParamsA_.dstStride = 0;
        DataCopy<T>(L1MatA[pingpong], mmAGm[(offsetM * fracK + offsetK) * 256], repeatParamsA_);
        if ((baseM + offsetM) * 16 > realM) {
            repeatParamsA.ndNum = 1;
            repeatParamsA.nValue = realM % 16;
            repeatParamsA.dValue = baseK * 16;
            repeatParamsA.srcNdMatrixStride = 0;
            repeatParamsA.srcDValue = fracK * 16;
            repeatParamsA.dstNzC0Stride = 16;
            repeatParamsA.dstNzNStride = 1;
            repeatParamsA.dstNzMatrixStride = 1;
            DataCopy<T>(L1MatA[pingpong][(baseM - 1) * baseK * 256],
                        mmAGm[(offsetM + baseM - 1) * fracK * 256 + offsetK * 16], repeatParamsA);
        }
    } else {
        repeatParamsA.ndNum = 1;
        repeatParamsA.nValue = ((baseM + offsetM) * 16) < realM ? (baseM * 16) : (realM - offsetM * 16);
        repeatParamsA.dValue = baseK * 16;
        repeatParamsA.srcNdMatrixStride = 0;
        repeatParamsA.srcDValue = fracK * 16;
        repeatParamsA.dstNzC0Stride = baseM * 16;
        repeatParamsA.dstNzNStride = 1;
        repeatParamsA.dstNzMatrixStride = 1;
        DataCopy<T>(L1MatA[pingpong], mmAGm[offsetM * fracK * 256 + offsetK * 16], repeatParamsA);
    }

    if (task_id == 0)
        wait_flag_dev(ping_pong ? SYNC_PING_AIV_AIC_FLAG : SYNC_PONG_AIV_AIC_FLAG);
    repeatParamsB.blockLen = baseK * baseN * 16;
    DataCopy<T>(L1MatB[pingpong], workspaceWGm[ping_pong][task_id], repeatParamsB);
    SetFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1);
    if ((task_id + 1) == MY_BUFFER_NUM && needSetFlag)
        ffts_cross_core_sync(PIPE_MTE2, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIC_AIV_FLAG : SYNC_PONG_AIC_AIV_FLAG));

    WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[pingpong]);
    WaitFlag<HardEvent::MTE2_MTE1>(eventIdMTE2ToMTE1);
    if (NZFormat) {
        loadDataA.srcStride = 1;
        loadDataA.ifTranspose = true;
        loadDataA.repeatTimes = baseK * baseM;
        if ((baseM + offsetM) * 16 > realM) {
            loadDataA.repeatTimes -= baseK;
        }
        LoadData<T>(L0MatA[pingpong], L1MatA[pingpong], loadDataA);

        if ((baseM + offsetM) * 16 > realM) {
            loadDataA.ifTranspose = false;
            loadDataA.repeatTimes = baseK;
            LoadData<T>(L0MatA[pingpong][(baseM - 1) * baseK * 256], L1MatA[pingpong][(baseM - 1) * baseK * 256],
                        loadDataA);
        }
    } else {
        loadDataA.srcStride = baseM;
        loadDataA.ifTranspose = false;
        loadDataA.repeatTimes = baseK;
        for (int32_t i = 0; i < baseM; i++) {
            LoadData<T>(L0MatA[pingpong][i * baseK * 256], L1MatA[pingpong][i * 256], loadDataA);
        }
    }

    loadDataB.srcStride = 1;
    loadDataB.ifTranspose = true;
    loadDataB.repeatTimes = baseK * baseN;
    LoadData<T>(L0MatB[pingpong], L1MatB[pingpong], loadDataB);
    SetFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);
    SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[pingpong]);

    WaitFlag<HardEvent::MTE1_M>(eventIdMTE1ToM);
    mmadParams.m = baseM * 16;
    mmadParams.n = baseN * 16;
    mmadParams.k = baseK * 16;
    mmadParams.cmatrixInitVal = startKFlag ? true : false;
    Mmad<float, T, T>(L0MatC, L0MatA[pingpong], L0MatB[pingpong], mmadParams);
    SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[pingpong]);

    if (endKFlag) {
        SetFlag<HardEvent::M_FIX>(eventIdMToFIX);
        WaitFlag<HardEvent::M_FIX>(eventIdMToFIX);
        DataCopyCO12DstParams repeatParamsC;
        repeatParamsC.nSize = baseN * 16;
        repeatParamsC.mSize = ((baseM + offsetM) * 16) < realM ? (baseM * 16) : (realM - offsetM * 16);
        repeatParamsC.dstStride = fracN * 16;
        repeatParamsC.srcStride = baseM * 16;
        if (tilingData->dataType == 1) {
            repeatParamsC.quantPre = QuantMode_t::F322F16;
        } else if (tilingData->dataType == 27) {
            repeatParamsC.quantPre = QuantMode_t::F322BF16;
        }
        repeatParamsC.nz2ndEn = true;
        SetFixpipePreQuantFlag(static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&deqScalar)));
        SetFixpipeNz2ndFlag(1, 0, 0);
        DataCopy<T, float>(mmCGm[offsetM * fracN * 256 + offsetN * 16], L0MatC, repeatParamsC);

        SetFlag<HardEvent::FIX_M>(eventIdFIXToM);
        WaitFlag<HardEvent::FIX_M>(eventIdFIXToM);
    }
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::ProcessMM()
{
    uint32_t total_pingpong_loop = (MMtaskNum + MY_BUFFER_NUM - 1) / MY_BUFFER_NUM;
    if ASCEND_IS_AIV {
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_CLEAR_AIV_FLAG));
        wait_flag_dev(SYNC_CLEAR_AIV_FLAG);
        ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_CLEAR_AIV_AIC_FLAG));

        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
    }
    if ASCEND_IS_AIC {
        wait_flag_dev(SYNC_CLEAR_AIV_AIC_FLAG);

        if (total_pingpong_loop >= 1)
            ffts_cross_core_sync(PIPE_MTE2, GetffstMsg(0x02, SYNC_PONG_AIC_AIV_FLAG));
        if (total_pingpong_loop >= 2)
            ffts_cross_core_sync(PIPE_MTE2, GetffstMsg(0x02, SYNC_PING_AIC_AIV_FLAG));
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        SetFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);
        SetFlag<HardEvent::M_MTE1>(eventIdMToMTE1[1]);
    }
    uint32_t ping_pong = 0;

    uint32_t baseK = fracK / baseKNum;
    uint32_t tailK = fracK % baseKNum;

    uint32_t task_id = 0;
    for (int32_t i = 0; i < tilingData->originE; i++) {
        if (realM[i] == 0)
            continue;
        uint32_t fracM = (realM[i] + 15) / 16;
        uint32_t baseNNum = fracM <= baseMLimitA ? baseNNumA : baseNNumB;
        if ((task_id + (baseMNum[i] * baseKNum * baseNNum)) <= MMtaskOffset || task_id >= (MMtaskOffset + MMtaskNum)) {
            mmAGm = mmAGm[realM[i] * fracK * 16];
            mmCGm = mmCGm[realM[i] * fracN * 16];
            task_id += (baseMNum[i] * baseKNum * baseNNum);
            continue;
        }
        mmBGm = mmBGm0[i * fracK * fracN * 256 / 8];
        mmScaleGm = mmScaleGm0[i * scaleK * fracN * 16];
        mmOffsetGm = mmOffsetGm0[i * scaleK * fracN * 16];

        uint32_t baseM = fracM / baseMNum[i];
        uint32_t tailM = fracM % baseMNum[i];
        uint32_t baseN = fracN / baseNNum;
        uint32_t tailN = fracN % baseNNum;

        uint32_t offsetM = 0;
        for (int32_t j = 0; j < baseMNum[i]; j++) {
            uint32_t realbaseM = baseM + (j < tailM ? 1 : 0);
            uint32_t offsetN = 0;
            for (int32_t k = 0; k < baseNNum; k++) {
                uint32_t realbaseN = baseN + (k < tailN ? 1 : 0);
                uint32_t offsetK = 0;
                for (int32_t l = 0; l < baseKNum; l++) {
                    uint32_t realbaseK = baseK + (l < tailK ? 1 : 0);
                    if (task_id < MMtaskOffset || task_id >= (MMtaskOffset + MMtaskNum)) {
                        task_id++;
                        offsetK += realbaseK;
                        continue;
                    }
                    uint32_t inner_task_id = (task_id - MMtaskOffset) % MY_BUFFER_NUM;
                    if ASCEND_IS_AIV {
                        if (subblock_id == (inner_task_id % 2)) {
                            bool endK = (task_id == (MMtaskOffset + MMtaskNum - 1)) ||
                                        (task_id == (MMtaskOffset + MMtaskNum - 2));
                            AntiQuantWeight(inner_task_id, ping_pong, endK, realbaseN, realbaseK, offsetN, offsetK);
                        } else if (inner_task_id == 0 && (task_id == (MMtaskOffset + MMtaskNum - 1))) {
                            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, ping_pong ? SYNC_PING_AIV_AIC_FLAG :
                                                                                         SYNC_PONG_AIV_AIC_FLAG));
                        }
                    }
                    if ASCEND_IS_AIC {
                        bool startK = (task_id == MMtaskOffset) || (l == 0);
                        bool endK = (task_id == (MMtaskOffset + MMtaskNum - 1)) || (l == (baseKNum - 1));
                        bool needSetFlag = (total_pingpong_loop - (task_id - MMtaskOffset) / MY_BUFFER_NUM) > 2;
                        ComputeMatMul(inner_task_id, ping_pong, startK, endK, needSetFlag, realM[i], realbaseM,
                                      realbaseN, realbaseK, offsetM, offsetN, offsetK);
                    }
                    task_id++;
                    offsetK += realbaseK;
                    if ((inner_task_id + 1) % MY_BUFFER_NUM == 0)
                        ping_pong = !ping_pong;
                }
                offsetN += realbaseN;
            }
            offsetM += realbaseM;
        }
        mmAGm = mmAGm[realM[i] * fracK * 16];
        mmCGm = mmCGm[realM[i] * fracN * 16];
    }
    if ASCEND_IS_AIV {
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[2]);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV[0]);
        wait_flag_dev(SYNC_CLEAR_AIC_AIV_FLAG);
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[0]);
        WaitFlag<HardEvent::MTE1_MTE2>(eventIdMTE1ToMTE2[1]);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[0]);
        WaitFlag<HardEvent::M_MTE1>(eventIdMToMTE1[1]);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x0, SYNC_CLEAR_AIC_FLAG));
        wait_flag_dev(SYNC_CLEAR_AIC_FLAG);
        ffts_cross_core_sync(PIPE_FIX, GetffstMsg(0x02, SYNC_CLEAR_AIC_AIV_FLAG));
    }
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::ProcessCountGroup()
{
    if ((expandedM * 2) <= (tilingData->originE * 3)) {
        if ASCEND_IS_AIV {
            LocalTensor<int32_t> expertIdxUb = ubBase.ReinterpretCast<int32_t>();
            DataCopyParams repeatParams;
            repeatParams.blockLen = (expandedM + 8 - 1) / 8;
            DataCopy<int32_t>(expertIdxUb, expertIdxGm, repeatParams);
            SetFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            WaitFlag<HardEvent::MTE2_S>(eventIdMTE2ToS);
            for (int i = 0; i < expandedM; i++) {
                realM[expertIdxUb.GetValue(i)]++;
            }
        }
        if ASCEND_IS_AIC {
            for (int i = 0; i < expandedM; i++) {
                realM[expertIdxGm.GetValue(i)]++;
            }
        }
    } else {
        if ASCEND_IS_AIV {
            LocalTensor<int32_t> groupListUb = ubBase.ReinterpretCast<int32_t>();
            SetMaskCount();
            SetVectorMask<int32_t, MaskMode::COUNTER>(0, tilingData->originE);
            Duplicate<int32_t, false>(groupListUb, static_cast<int32_t>(0), MASK_PLACEHOLDER, 1, DEFAULT_BLK_STRIDE,
                                      DEFAULT_REPEAT_STRIDE);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
            SetMaskNorm();
            ResetMask();

            DataCopyParams repeatParams;
            repeatParams.blockLen = (tilingData->originE + 8 - 1) / 8;
            DataCopy<int32_t>(groupListGm, groupListUb, repeatParams);
            SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);

            uint32_t loopN = expandedM / (2 * tilingData->CoreNum);
            uint32_t tailN = expandedM % (2 * tilingData->CoreNum);
            uint32_t offset = loopN * block_id_ + (block_id_ < tailN ? block_id_ : tailN);
            loopN += (block_id_ < tailN ? 1 : 0);
            for (int i = offset; i < (offset + loopN); i++) {
                realM[expertIdxGm.GetValue(i)]++;
            }

            WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
            for (int i = 0; i < tilingData->originE; i++) {
                if (realM[i] != 0) {
                    groupListUb.SetValue(i, realM[i]);
                }
            }

            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_CLEAR_AIV_FLAG));
            wait_flag_dev(SYNC_CLEAR_AIV_FLAG);
            SetAtomicAdd<int32_t>();
            DataCopy<int32_t>(groupListGm, groupListUb, repeatParams);
            AscendC::SetAtomicNone();

            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x0, SYNC_CLEAR_AIV_FLAG));
            wait_flag_dev(SYNC_CLEAR_AIV_FLAG);

            ffts_cross_core_sync(PIPE_MTE3, GetffstMsg(0x02, SYNC_CLEAR_AIV_AIC_FLAG));
        }
        if ASCEND_IS_AIC {
            wait_flag_dev(SYNC_CLEAR_AIV_AIC_FLAG);
        }
        for (int i = 0; i < tilingData->originE; i++) {
            realM[i] = groupListGm.GetValue(i);
        }
    }
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::PrepareRouteCommon()
{
    if ASCEND_IS_AIV {
        ubRouteND[0] = ubBase.ReinterpretCast<T>();
        ubRouteZN[0] = ubRouteND[0][16 * TRANS_BASEN];
        ubRouteND[1] = ubRouteZN[0][16 * TRANS_BASEN];
        ubRouteZN[1] = ubRouteND[1][16 * TRANS_BASEN];

        NDLocalList[0][0] = reinterpret_cast<uint64_t>(ubRouteND[0].GetPhyAddr());
        ZNLocalList[0][0] = reinterpret_cast<uint64_t>(ubRouteZN[0].GetPhyAddr());
        NDLocalList[1][0] = reinterpret_cast<uint64_t>(ubRouteND[1].GetPhyAddr());
        ZNLocalList[1][0] = reinterpret_cast<uint64_t>(ubRouteZN[1].GetPhyAddr());
        for (int32_t i = 1; i < 16; i++) {
            NDLocalList[0][i] = NDLocalList[0][i - 1] + TRANS_BASEN * sizeof(half);
            NDLocalList[1][i] = NDLocalList[1][i - 1] + TRANS_BASEN * sizeof(half);
            ZNLocalList[0][i] = ZNLocalList[0][i - 1] + 32;
            ZNLocalList[1][i] = ZNLocalList[1][i - 1] + 32;
        }
    }
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::ProcessRoute()
{
    DataCopyParams repeatParamsZN, repeatParamsND;
    TransDataTo5HDParams transDataParams;
    repeatParamsND.blockLen = tilingData->fracK_13;
    transDataParams.dstRepStride = 16;
    transDataParams.srcRepStride = 1;

    uint32_t trans_inner_loop = (tilingData->originK_13 + TRANS_BASEN - 1) / TRANS_BASEN;
    uint32_t trans_tail = tilingData->originK_13 % TRANS_BASEN;

    uint32_t taskN = totalFracM / (2 * tilingData->CoreNum);
    uint32_t tailN = totalFracM % (2 * tilingData->CoreNum);
    uint32_t taskOffset = taskN * block_id_ + (block_id_ < tailN ? block_id_ : tailN);
    taskN += (block_id_ < tailN ? 1 : 0);
    uint32_t task_id = 0;
    uint32_t pingpong = 0;
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
    for (int32_t i = 0; i < tilingData->originE; i++) {
        if (realM[i] == 0)
            continue;
        uint32_t fracM = (realM[i] + 16 - 1) / 16;
        if ((task_id + fracM) <= taskOffset || task_id >= (taskOffset + taskN)) {
            task_id += fracM;
            rowIdxGm = rowIdxGm[realM[i]];
            routedGm = routedGm[realM[i] * tilingData->originK_13];
            continue;
        }
        for (int32_t j = 0; j < fracM; j++) {
            if (task_id < taskOffset || task_id >= (taskOffset + taskN)) {
                task_id++;
                continue;
            }
            if (j == (fracM - 1) && realM[i] % 16 != 0) {
                for (int32_t k = 0; k < (realM[i] % 16); k++) {
                    pingpong = !pingpong;
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong]);
                    DataCopy<T>(ubRouteND[pingpong], xGm[rowIdxGm.GetValue(j * 16 + k) * tilingData->originK_13],
                                repeatParamsND);
                    SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
                    WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2ToMTE3);
                    DataCopy<T>(routedGm[(j * 16 + k) * tilingData->originK_13], ubRouteND[pingpong], repeatParamsND);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong]);
                }
            } else {
                uint32_t offsetN = 0;
                for (int32_t k = 0; k < trans_inner_loop; k++) {
                    pingpong = !pingpong;
                    uint32_t realBaseN = TRANS_BASEN;
                    if (k == (trans_inner_loop - 1) && trans_tail != 0) {
                        realBaseN = trans_tail;
                    }

                    repeatParamsZN.blockLen = realBaseN / 16;
                    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong]);
                    for (int32_t l = 0; l < 16; l++) {
                        DataCopy<T>(ubRouteND[pingpong][l * TRANS_BASEN],
                                    xGm[rowIdxGm.GetValue(j * 16 + l) * tilingData->originK_13 + offsetN],
                                    repeatParamsZN);
                    }
                    SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

                    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
                    transDataParams.repeatTimes = realBaseN / 16;
                    TransDataTo5HD<half>(ZNLocalList[pingpong], NDLocalList[pingpong], transDataParams);
                    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

                    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
                    repeatParamsZN.blockLen = realBaseN;
                    DataCopy<T>(routedGm[j * 16 * tilingData->originK_13 + offsetN * 16], ubRouteZN[pingpong],
                                repeatParamsZN);
                    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong]);

                    offsetN += realBaseN;
                }
            }
            task_id++;
        }
        rowIdxGm = rowIdxGm[realM[i]];
        routedGm = routedGm[realM[i] * tilingData->originK_13];
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::ComputeRouteMMTiling()
{
    MM1InputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    routedGm = MM1InputGm;
    MM1InputGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_NORMAL);
    mmAGm = MM1InputGm;
    mmCGm = MM1OutputGm;
    mmBGm0 = wGm_13;
    mmScaleGm0 = wScaleGm_13;
    mmOffsetGm0 = wOffsetGm_13;
    fracK = tilingData->fracK_13;
    fracN = tilingData->fracN_13;
    scaleK = tilingData->scaleK_13;
    needBias = tilingData->needBias_13;
    NZFormat = true;
    totalFracM = 0;
    totalTaskNum = 0;
    baseNNumA = (fracN + 16 - 1) / 16;
    uint32_t baseNA = (fracN + baseNNumA - 1) / baseNNumA;
    baseMLimitA = 128 / baseNA;
    uint32_t baseKLimitA = 64 / (baseNA > baseMLimitA ? baseNA : baseMLimitA);

    baseNNumB = (fracN + 8 - 1) / 8;
    uint32_t baseNB = (fracN + baseNNumB - 1) / baseNNumB;
    uint32_t baseMLimitB = 128 / baseNB;
    uint32_t baseKLimitB = 64 / (baseNB > baseMLimitB ? baseNB : baseMLimitB);

    uint32_t baseKLimit = baseKLimitA < baseKLimitB ? baseKLimitA : baseKLimitB;
    baseKNum = (fracK + baseKLimit - 1) / baseKLimit;

    for (int i = 0; i < tilingData->originE; i++) {
        if (realM[i] == 0) {
            baseMNum[i] = 0;
            continue;
        }
        uint32_t fracM = (realM[i] + 16 - 1) / 16;
        totalFracM += fracM;

        if (fracM <= baseMLimitA) {
            baseMNum[i] = (fracM + baseMLimitA - 1) / baseMLimitA;
            totalTaskNum += (baseMNum[i] * baseNNumA);
        } else {
            baseMNum[i] = (fracM + baseMLimitB - 1) / baseMLimitB;
            totalTaskNum += (baseMNum[i] * baseNNumB);
        }
    }
    MMtaskNum = totalTaskNum / tilingData->CoreNum + (block_id < (totalTaskNum % tilingData->CoreNum) ? 1 : 0);
    MMtaskOffset = totalTaskNum / tilingData->CoreNum * block_id +
                   (block_id < (totalTaskNum % tilingData->CoreNum) ? block_id : (totalTaskNum % tilingData->CoreNum));
    MMtaskNum *= baseKNum;
    MMtaskOffset *= baseKNum;
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::PrepareSwigluCommon()
{
    if ASCEND_IS_AIV {
        ubSwigluA[0] = ubBase.ReinterpretCast<T>();
        ubSwigluB[0] = ubSwigluA[0][SWIGLU_BASEN];
        ubSwigluA[1] = ubSwigluB[0][SWIGLU_BASEN];
        ubSwigluB[1] = ubSwigluA[1][SWIGLU_BASEN];
        ubSwigluSigmoid[0] = ubSwigluB[1][SWIGLU_BASEN];
        ubSwigluSigmoid[1] = ubSwigluSigmoid[0][SWIGLU_BASEN];
        ubSwigluSigmoidFloat[0] = ubBase.ReinterpretCast<float>();
        ubSwigluSigmoidFloat[1] = ubSwigluSigmoidFloat[0][SWIGLU_BASEN];
        ubSwigluAFloat[0] = ubSwigluSigmoidFloat[1][SWIGLU_BASEN];
        ubSwigluAFloat[1] = ubSwigluAFloat[0][SWIGLU_BASEN];
        ubSwigluBFloat[0] = ubSwigluAFloat[1][SWIGLU_BASEN];
        ubSwigluBFloat[1] = ubSwigluBFloat[0][SWIGLU_BASEN];
    }
}

template <typename T>
template <typename U>
__aicore__ inline void MOEW4A16<T>::SwishFunc(const LocalTensor<U> &dst, const LocalTensor<U> &src)
{
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;
    Muls<U, false>(dst, src, static_cast<U>(-1.0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Exp<U, false>(dst, dst, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Adds<U, false>(dst, dst, static_cast<U>(1.0), MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();
    Div<U, false>(dst, src, dst, MASK_PLACEHOLDER, 1, binaryParams);
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::ProcessSwiglu()
{
    DataCopyParams repeatParamsIn, repeatParamsOut;
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;
    repeatParamsIn.blockLen = tilingData->originK_2 / 16;
    repeatParamsIn.srcStride = tilingData->originK_2 / 16;
    UnaryRepeatParams unaryParams16To32, unaryParams32To16;
    unaryParams16To32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams32To16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;

    uint32_t taskN = expandedM / (2 * tilingData->CoreNum);
    uint32_t tailN = expandedM % (2 * tilingData->CoreNum);
    uint32_t offsetM = taskN * block_id_ + (block_id_ < tailN ? block_id_ : tailN);
    taskN += (block_id_ < tailN ? 1 : 0);
    uint32_t baseM = SWIGLU_BASEN / tilingData->originK_2;
    uint32_t loopM = (taskN + baseM - 1) / baseM;
    uint32_t tailM = taskN % baseM;
    uint32_t pingpong = 0;
    SetMaskCount();
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
    for (int i = 0; i < loopM; i++) {
        pingpong = !pingpong;
        uint32_t realBaseM = baseM;
        if (i == (loopM - 1) && tailM != 0) {
            realBaseM = tailM;
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong]);
        repeatParamsIn.blockCount = realBaseM;
        DataCopy<T>(ubSwigluA[pingpong], MM1OutputGm[offsetM * tilingData->originN_13], repeatParamsIn);
        DataCopy<T>(ubSwigluB[pingpong], MM1OutputGm[offsetM * tilingData->originN_13 + tilingData->originK_2],
                    repeatParamsIn);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        SetVectorMask<half, MaskMode::COUNTER>(realBaseM * tilingData->originK_2);
        Cast<float, T, false>(ubSwigluAFloat[pingpong], ubSwigluA[pingpong], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                              unaryParams16To32);
        Cast<float, T, false>(ubSwigluBFloat[pingpong], ubSwigluB[pingpong], RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
                              unaryParams16To32);

        PipeBarrier<PIPE_V>();
        SwishFunc<float>(ubSwigluSigmoidFloat[pingpong], ubSwigluAFloat[pingpong]);

        PipeBarrier<PIPE_V>();
        Mul<float, false>(ubSwigluSigmoidFloat[pingpong], ubSwigluSigmoidFloat[pingpong], ubSwigluBFloat[pingpong],
                          MASK_PLACEHOLDER, 1, binaryParams);

        PipeBarrier<PIPE_V>();
        Cast<T, float, false>(ubSwigluA[pingpong], ubSwigluSigmoidFloat[pingpong], RoundMode::CAST_RINT,
                              MASK_PLACEHOLDER, 1, unaryParams32To16);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        repeatParamsOut.blockLen = realBaseM * tilingData->originK_2 / 16;
        DataCopy<T>(MM2InputGm[offsetM * tilingData->originK_2], ubSwigluA[pingpong], repeatParamsOut);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong]);
        offsetM += realBaseM;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::PrepareReRouteCommon()
{
    if ASCEND_IS_AIV {
        ubReRouteResFloat[0] = ubBase.ReinterpretCast<float>();
        ubReRouteResFloat[1] = ubReRouteResFloat[0][REROUTE_BASEN];
        ubReRouteBufFloat[0] = ubReRouteResFloat[1][REROUTE_BASEN];
        ubReRouteBufFloat[1] = ubReRouteBufFloat[0][REROUTE_BASEN];
        ubReRouteRes[0] = ubReRouteBufFloat[1][REROUTE_BASEN].ReinterpretCast<T>();
        ubReRouteRes[1] = ubReRouteRes[0][REROUTE_BASEN];
        ubReRouteBuf[0] = ubReRouteRes[1][REROUTE_BASEN];
        ubReRouteBuf[1] = ubReRouteBuf[0][REROUTE_BASEN];
    }
}

template <>
__aicore__ inline void MOEW4A16<bfloat16_t>::ProcessReRoute()
{
    DataCopyParams repeatParams;
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;
    UnaryRepeatParams unaryParams16To32, unaryParams32To16;
    unaryParams16To32.srcRepStride = HALF_DEFAULT_REPEAT_STRIDE;
    unaryParams32To16.dstRepStride = HALF_DEFAULT_REPEAT_STRIDE;

    uint32_t inner_taskN_max = (tilingData->originN_2 + REROUTE_BASEN_MIN - 1) / REROUTE_BASEN_MIN;
    uint32_t inner_taskN = (tilingData->CoreNum + tilingData->originM - 1) / tilingData->originM;
    inner_taskN = inner_taskN < inner_taskN_max ? inner_taskN : inner_taskN_max;

    uint32_t taskN = (tilingData->originM * inner_taskN) / (2 * tilingData->CoreNum);
    uint32_t tailN = (tilingData->originM * inner_taskN) % (2 * tilingData->CoreNum);
    uint32_t taskOffset = taskN * block_id_ + (block_id_ < tailN ? block_id_ : tailN);
    taskN += (block_id_ < tailN ? 1 : 0);
    uint32_t baseN = inner_taskN_max / inner_taskN;
    tailN = inner_taskN_max % inner_taskN;
    uint32_t tailtailN = tilingData->originN_2 % REROUTE_BASEN_MIN;

    uint32_t pingpong1 = 0;
    uint32_t pingpong2 = 0;
    SetMaskCount();
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    for (int32_t i = 0; i < taskN; i++) {
        pingpong1 = !pingpong1;
        uint32_t idxM = (i + taskOffset) / inner_taskN;
        uint32_t idxN = (i + taskOffset) % inner_taskN;
        uint32_t offsetN = (idxN * baseN + (idxN < tailN ? idxN : tailN)) * REROUTE_BASEN_MIN;
        uint32_t realBaseN = (baseN + (idxN < tailN ? 1 : 0)) * REROUTE_BASEN_MIN;
        if (idxN == (inner_taskN - 1) && tailtailN != 0) {
            realBaseN -= (REROUTE_BASEN_MIN - tailtailN);
        }
        repeatParams.blockLen = realBaseN / 16;
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong1]);
        DataCopy<bfloat16_t>(
            ubReRouteRes[pingpong1],
            MM2OutputGm[rowIdxGmRev.GetValue(idxM * tilingData->topK) * tilingData->originN_2 + offsetN], repeatParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        SetVectorMask<float, MaskMode::COUNTER>(realBaseN);
        Cast<float, bfloat16_t, false>(ubReRouteResFloat[pingpong1], ubReRouteRes[pingpong1], RoundMode::CAST_NONE,
                                       MASK_PLACEHOLDER, 1, unaryParams16To32);

        PipeBarrier<PIPE_V>();
        Muls<float, false>(ubReRouteResFloat[pingpong1], ubReRouteResFloat[pingpong1],
                           ToFloat(topKwGm.GetValue(idxM * tilingData->topK)), MASK_PLACEHOLDER, 1, unaryParams);

        for (int32_t j = 1; j < tilingData->topK; j++) {
            pingpong2 = !pingpong2;
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[pingpong2]);
            DataCopy<bfloat16_t>(
                ubReRouteBuf[pingpong2],
                MM2OutputGm[rowIdxGmRev.GetValue(idxM * tilingData->topK + j) * tilingData->originN_2 + offsetN],
                repeatParams);
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            Cast<float, bfloat16_t, false>(ubReRouteBufFloat[pingpong2], ubReRouteBuf[pingpong2], RoundMode::CAST_NONE,
                                           MASK_PLACEHOLDER, 1, unaryParams16To32);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[pingpong2]);

            PipeBarrier<PIPE_V>();
            Muls<float, false>(ubReRouteBufFloat[pingpong2], ubReRouteBufFloat[pingpong2],
                               ToFloat(topKwGm.GetValue(idxM * tilingData->topK + j)), MASK_PLACEHOLDER, 1,
                               unaryParams);

            PipeBarrier<PIPE_V>();
            Add<float, false>(ubReRouteResFloat[pingpong1], ubReRouteResFloat[pingpong1], ubReRouteBufFloat[pingpong2],
                              MASK_PLACEHOLDER, 1, binaryParams);
        }

        PipeBarrier<PIPE_V>();
        Cast<bfloat16_t, float, false>(ubReRouteRes[pingpong1], ubReRouteResFloat[pingpong1], RoundMode::CAST_RINT,
                                       MASK_PLACEHOLDER, 1, unaryParams32To16);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy<bfloat16_t>(yGm[idxM * tilingData->originN_2 + offsetN], ubReRouteRes[pingpong1], repeatParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong1]);
    }
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);

    SetMaskNorm();
    ResetMask();
}

template <>
__aicore__ inline void MOEW4A16<half>::ProcessReRoute()
{
    DataCopyParams repeatParams;
    BinaryRepeatParams binaryParams;
    UnaryRepeatParams unaryParams;

    uint32_t inner_taskN_max = (tilingData->originN_2 + REROUTE_BASEN_MIN - 1) / REROUTE_BASEN_MIN;
    uint32_t inner_taskN = (tilingData->CoreNum + tilingData->originM - 1) / tilingData->originM;
    inner_taskN = inner_taskN < inner_taskN_max ? inner_taskN : inner_taskN_max;

    uint32_t taskN = (tilingData->originM * inner_taskN) / (2 * tilingData->CoreNum);
    uint32_t tailN = (tilingData->originM * inner_taskN) % (2 * tilingData->CoreNum);
    uint32_t taskOffset = taskN * block_id_ + (block_id_ < tailN ? block_id_ : tailN);
    taskN += (block_id_ < tailN ? 1 : 0);
    uint32_t baseN = inner_taskN_max / inner_taskN;
    tailN = inner_taskN_max % inner_taskN;
    uint32_t tailtailN = tilingData->originN_2 % REROUTE_BASEN_MIN;

    uint32_t pingpong1 = 0;
    uint32_t pingpong2 = 0;
    SetMaskCount();
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    for (int32_t i = 0; i < taskN; i++) {
        pingpong1 = !pingpong1;
        uint32_t idxM = (i + taskOffset) / inner_taskN;
        uint32_t idxN = (i + taskOffset) % inner_taskN;
        uint32_t offsetN = (idxN * baseN + (idxN < tailN ? idxN : tailN)) * REROUTE_BASEN_MIN;
        uint32_t realBaseN = (baseN + (idxN < tailN ? 1 : 0)) * REROUTE_BASEN_MIN;
        if (idxN == (inner_taskN - 1) && tailtailN != 0) {
            realBaseN -= (REROUTE_BASEN_MIN - tailtailN);
        }
        repeatParams.blockLen = realBaseN / 16;
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong1]);
        DataCopy<half>(ubReRouteRes[pingpong1],
                       MM2OutputGm[rowIdxGmRev.GetValue(idxM * tilingData->topK) * tilingData->originN_2 + offsetN],
                       repeatParams);
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
        SetVectorMask<half, MaskMode::COUNTER>(realBaseN);
        Muls<half, false>(ubReRouteRes[pingpong1], ubReRouteRes[pingpong1], topKwGm.GetValue(idxM * tilingData->topK),
                          MASK_PLACEHOLDER, 1, unaryParams);

        for (int32_t j = 1; j < tilingData->topK; j++) {
            pingpong2 = !pingpong2;
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[pingpong2]);
            DataCopy<half>(
                ubReRouteBuf[pingpong2],
                MM2OutputGm[rowIdxGmRev.GetValue(idxM * tilingData->topK + j) * tilingData->originN_2 + offsetN],
                repeatParams);
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);

            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV[0]);
            Muls<half, false>(ubReRouteBuf[pingpong2], ubReRouteBuf[pingpong2],
                              topKwGm.GetValue(idxM * tilingData->topK + j), MASK_PLACEHOLDER, 1, unaryParams);

            PipeBarrier<PIPE_V>();
            Add<half, false>(ubReRouteRes[pingpong1], ubReRouteRes[pingpong1], ubReRouteBuf[pingpong2],
                             MASK_PLACEHOLDER, 1, binaryParams);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2[pingpong2]);
        }
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy<half>(yGm[idxM * tilingData->originN_2 + offsetN], ubReRouteRes[pingpong1], repeatParams);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[pingpong1]);
    }
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[0]);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2[1]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[0]);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2[1]);

    SetMaskNorm();
    ResetMask();
}

template <typename T>
__aicore__ inline void MOEW4A16<T>::Process()
{
    // step 1
    ProcessCountGroup();
    PrepareMMCommon();
    // step 2
    PrepareRouteCommon();
    ComputeRouteMMTiling();
    if ASCEND_IS_AIV {
        ProcessRoute();
    }
    ProcessMM();

    // step 3
    PrepareSwigluCommon();
    if ASCEND_IS_AIV {
        ProcessSwiglu();
    }

    // step 4
    ComputeMMTiling();
    ProcessMM();

    // step 5
    PrepareReRouteCommon();
    if ASCEND_IS_AIV {
        ProcessReRoute();
    }
}
