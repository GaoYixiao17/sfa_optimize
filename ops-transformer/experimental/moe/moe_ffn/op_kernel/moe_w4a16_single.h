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
 * \file moe_w4a16.h
 * \brief
 */
#ifndef _ASCENDC_MOE_W4A16SINGLE_H_
#define _ASCENDC_MOE_W4A16SINGLE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "common_utils.h"
using namespace AscendC;

template <typename T>
class MOEW4A16Single {
public:
    TPipe pipe;

    __aicore__ inline MOEW4A16Single()
    {
    }
    __aicore__ inline void Process();
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR quantized_weight_13, GM_ADDR weight_scale_13,
                                GM_ADDR weight_offset_13, GM_ADDR quantized_weight_2, GM_ADDR weight_scale_2,
                                GM_ADDR weight_offset_2, GM_ADDR expanded_expert_idx, GM_ADDR sorted_row_idx,
                                GM_ADDR expanded_row_idx, GM_ADDR topk_weights, GM_ADDR y, GM_ADDR usrworkspace,
                                const MoeFFNTilingData *__restrict moeTiling);

protected:
    __aicore__ inline void InitWorkspaceTensors(GM_ADDR usrWorkspace);
    __aicore__ inline void InitGlobalTensors(GM_ADDR x, GM_ADDR quantized_weight_13, GM_ADDR weight_scale_13,
                                             GM_ADDR weight_offset_13, GM_ADDR quantized_weight_2,
                                             GM_ADDR weight_scale_2, GM_ADDR weight_offset_2,
                                             GM_ADDR expanded_expert_idx, GM_ADDR sorted_row_idx,
                                             GM_ADDR expanded_row_idx, GM_ADDR topk_weights, GM_ADDR y);
    __aicore__ inline void InitLocalTensors();
    __aicore__ inline void InitEventId();

    __aicore__ inline void PrepareMMCommon();
    __aicore__ inline void ComputeMMTiling();
    __aicore__ inline void ComputeRouteMMTiling();
    __aicore__ inline void AntiQuantWeight(uint32_t task_id, uint32_t ping_pong, bool endKFlag, uint32_t baseN,
                                           uint32_t baseK, uint32_t offsetN, uint32_t offsetK);
    __aicore__ inline void ComputeMatMul(uint32_t task_id, uint32_t ping_pong, bool startKFlag, bool endKFlag,
                                         bool needSetFlag, uint32_t realM, uint32_t baseM, uint32_t baseN,
                                         uint32_t baseK, uint32_t offsetM, uint32_t offsetN, uint32_t offsetK);
    __aicore__ inline void ProcessMM();

    __aicore__ inline void PrepareSwigluCommon();
    template <typename U>
    __aicore__ inline void SwishFunc(const LocalTensor<U> &dst, const LocalTensor<U> &src);
    __aicore__ inline void ProcessSwiglu();

    __aicore__ inline void PrepareReRouteCommon();
    __aicore__ inline void ProcessReRoute();

    __aicore__ inline void ProcessCountGroup();
    // inoutTensors
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<int32_t> wGm_13, wGm_2;
    GlobalTensor<T> wScaleGm_13, wScaleGm_2;
    GlobalTensor<half> wOffsetGm_13, wOffsetGm_2;
    GlobalTensor<int32_t> expertIdxGm, rowIdxGm, rowIdxGmRev;
    GlobalTensor<T> topKwGm;
    // workspace
    GlobalTensor<int32_t> groupListGm;
    GlobalTensor<T> workspaceWGm[2][MY_BUFFER_NUM];
    GlobalTensor<T> MM1InputGm;
    GlobalTensor<T> MM1OutputGm;
    GlobalTensor<T> MM2InputGm;
    GlobalTensor<T> MM2OutputGm;
    // Buffer
    TBuf<TPosition::VECCALC> UbBuf;
    TBuf<TPosition::TSCM> L1Buf;
    TBuf<TPosition::A2> L0ABuf;
    TBuf<TPosition::B2> L0BBuf;
    TBuf<TPosition::CO1> L0CBuf;
    LocalTensor<uint8_t> ubBase;

    // Buffer
    event_t eventIdMTE3ToS;
    event_t eventIdVToMTE3;
    event_t eventIdMTE3ToV[2];
    event_t eventIdVToMTE2[3];
    event_t eventIdMTE2ToV[3];
    event_t eventIdMTE3ToMTE2[2];
    event_t eventIdMTE1ToMTE2[2];
    event_t eventIdMTE2ToMTE1;
    event_t eventIdMTE2ToMTE3;
    event_t eventIdMTE1ToM;
    event_t eventIdMToMTE1[2];
    event_t eventIdMToFIX;
    event_t eventIdFIXToM;

    const MoeFFNTilingData *__restrict tilingData;

    // common
    int32_t block_id;
    int32_t block_id_;
    int32_t subblock_id;
    int32_t expandedM;
    float deqScalar = 1.0;
    uint32_t realM[GROUP_MAX] = {0};
    uint32_t rowIdxRev[16] = {0};

    // matmul
    bool firstMM = false;
    bool splitK = false;
    bool NZFormat = false;
    bool needTiling = false;
    bool needBias = false;
    uint32_t fracK;
    uint32_t fracN;
    uint32_t scaleK;
    uint32_t baseKNum;
    uint32_t baseNNum;

    int32_t totalTaskNum = 0;
    uint32_t MMtaskNum = 0;
    uint32_t MMtaskOffset = 0;

    GlobalTensor<T> mmAGm;
    GlobalTensor<T> mmCGm;
    GlobalTensor<int32_t> mmBGm, mmBGm0;
    GlobalTensor<T> mmScaleGm, mmScaleGm0;
    GlobalTensor<half> mmOffsetGm, mmOffsetGm0;

    LocalTensor<T> L1MatA[2];
    LocalTensor<T> L1MatB[2];
    LocalTensor<T> L0MatA[2];
    LocalTensor<T> L0MatB[2];
    LocalTensor<float> L0MatC;

    // for antiquant
    LocalTensor<int4b_t> ubWInt4;
    LocalTensor<half> ubWHalf;
    LocalTensor<float> ubWFloat;
    LocalTensor<T> ubWT;
    LocalTensor<T> ubWScaleT;
    LocalTensor<half> ubWOffsetHalf;
    LocalTensor<float> ubWScaleFloat;
    // swiglu
    LocalTensor<T> ubSwigluA[2];
    LocalTensor<T> ubSwigluB[2];
    LocalTensor<T> ubSwigluSigmoid[2];
    LocalTensor<float> ubSwigluAFloat[2];
    LocalTensor<float> ubSwigluBFloat[2];
    LocalTensor<float> ubSwigluSigmoidFloat[2];
    // reroute
    LocalTensor<T> ubReRouteRes[2];
    LocalTensor<T> ubReRouteBuf[2];
    LocalTensor<float> ubReRouteResFloat[2];
    LocalTensor<float> ubReRouteBufFloat[2];
};
#endif
