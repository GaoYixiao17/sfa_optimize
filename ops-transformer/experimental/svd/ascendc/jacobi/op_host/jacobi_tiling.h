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
 * \file jacobi_tiling.h
 * \brief
 */
#ifndef JACOBI_TILING_H_
#define JACOBI_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "inc/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
// ----------- TilingData ---------------
BEGIN_TILING_DATA_DEF(JMTilingData)
TILING_DATA_FIELD_DEF(uint32_t, nIterations)
TILING_DATA_FIELD_DEF(uint32_t, nStages)
TILING_DATA_FIELD_DEF(uint32_t, batchSize)
TILING_DATA_FIELD_DEF(uint32_t, mSize)
TILING_DATA_FIELD_DEF(uint32_t, nSize)
TILING_DATA_FIELD_DEF(uint32_t, nSizeAligned)
TILING_DATA_FIELD_DEF(uint32_t, numUsedVectorCores)
TILING_DATA_FIELD_DEF(uint32_t, numAICores)
TILING_DATA_FIELD_DEF(uint32_t, globalSetSizeMax)
TILING_DATA_FIELD_DEF(uint32_t, globalSetSizeMin)
TILING_DATA_FIELD_DEF(uint32_t, numGlobalSets)
TILING_DATA_FIELD_DEF(uint32_t, numGlobalSetsWithMaxSize)
TILING_DATA_FIELD_DEF(uint32_t, numGlobalSetsWithMinSize)

TILING_DATA_FIELD_DEF(uint32_t, ubSize)
TILING_DATA_FIELD_DEF(uint32_t, vecInstructionSize)
TILING_DATA_FIELD_DEF(uint32_t, vecInstructionNum)
TILING_DATA_FIELD_DEF(uint32_t, maxBucketSize)
TILING_DATA_FIELD_DEF(uint32_t, aRowSize)
TILING_DATA_FIELD_DEF(uint32_t, aRowSizeAligned)
TILING_DATA_FIELD_DEF(uint32_t, uRowSize)
TILING_DATA_FIELD_DEF(uint32_t, uRowSizeAligned)

TILING_DATA_FIELD_DEF(uint32_t, nMaskSize)
TILING_DATA_FIELD_DEF(uint32_t, nRepeatsNum)
TILING_DATA_FIELD_DEF(uint32_t, nTailSize)
TILING_DATA_FIELD_DEF(uint32_t, uMSize)
TILING_DATA_FIELD_DEF(uint32_t, uNSize)
TILING_DATA_FIELD_DEF(uint32_t, uNSizeAligned)
TILING_DATA_FIELD_DEF(uint32_t, vMSize)
TILING_DATA_FIELD_DEF(uint32_t, vNSize)
TILING_DATA_FIELD_DEF(uint32_t, vNSizeAligned)
TILING_DATA_FIELD_DEF(uint32_t, sMNSize)

TILING_DATA_FIELD_DEF(uint32_t, uNumRepeats)
TILING_DATA_FIELD_DEF(uint32_t, uTailSize)
TILING_DATA_FIELD_DEF(uint32_t, recordSize)
TILING_DATA_FIELD_DEF(uint32_t, recordSizeAligned)
TILING_DATA_FIELD_DEF(uint32_t, tmpMemRowSize)
TILING_DATA_FIELD_DEF(uint32_t, tmpMemRowSizeAligned)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(Jacobi, JMTilingData)

// ----------- CompileInfo -------------------
struct JMCompileInfo {};

// --------------- Tiling ---------------
struct GlobalSetsData {
    uint32_t numGlobSets;
    uint32_t globSetSizeMax;
    uint32_t globSetSizeMin;
    uint32_t numGlobalSetsMax;
    uint32_t numGlobalSetsMin;
    uint32_t numVectorCores;
};

struct ShapesInfo {
    uint32_t mSize;
    uint32_t nSize;
    uint32_t nSizeAlignedUp;
    uint32_t batchSize;
    uint32_t sSize;
    uint32_t uDim1;
    uint32_t uDim2;
    uint32_t uDim2Aligned;
    uint32_t vDim1;
    uint32_t vDim2;
    uint32_t vDim2Aligned;
    bool mIsSingularRank;
};

struct IteratesInfo {
    uint32_t nIterations;
    uint32_t nStages;
};

struct MemoryInfo {
    uint32_t ubSize;
    uint32_t vecReservedMemory;

    uint8_t vecInstructionNum;
    uint8_t vecInstructionSetSize;
    uint32_t maxBucketSize;
    uint32_t rowSize;
    uint32_t rowSizeAligned;
    uint32_t nSizeAligned;
    uint8_t nMaskSize;
    uint8_t nRepeatsNum;
    uint8_t nTailSize;

    uint32_t uRowSize;
    uint32_t uNSizeAligned;
    uint8_t uNumRepeats;
    uint32_t uRowSizeAligned;
    uint8_t uTailSize;

    uint32_t recordSize;
    uint32_t recordSizeAligned;
    uint32_t recordMemSize;

    uint32_t tmpRowSize;
    uint32_t tmpRowSizeAligned;
};

struct JMData {
    GlobalSetsData globSetsInfo;
    ShapesInfo shapeInfo;
    IteratesInfo iterInfo;
    MemoryInfo memInfo;
};

struct JMTiling {
public:
    explicit JMTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus DoTiling();

private:
    gert::TilingContext *context_ = nullptr;
    JMTilingData tilingData_;
    JMData jmData_;
private:
    ge::graphStatus InitAInput();
    ge::graphStatus InitSOutput();
    ge::graphStatus InitVOutput();
    ge::graphStatus InitUOutput();
    ge::graphStatus InitAttrs();
    ge::graphStatus FillTiling();
    ge::graphStatus ReadAscendPlatform();
    ge::graphStatus InitGlobalSetsInfo();
    ge::graphStatus InitMemInfo();
};

} // namespace optiling
#endif // JACOBI_TILING_H_