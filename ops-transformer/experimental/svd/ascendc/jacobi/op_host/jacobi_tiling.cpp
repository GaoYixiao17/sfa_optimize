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
 * \file jacobi_tiling.cpp
 * \brief
 */
#include "jacobi_tiling.h"

using namespace ge;
using namespace AscendC;

#define TRUE_IF_NULLPTR(X) (X)==nullptr
#define CHECK_ON_NULLPTR(X, MESSAGE) if (TRUE_IF_NULLPTR(X)){std::cout<<MESSAGE<<std::endl; return ge::GRAPH_FAILED;}
#define CHECK_ON_ERROR(X, MESSAGE) if (X==ge::GRAPH_FAILED){std::cout<<MESSAGE<<std::endl; return ge::GRAPH_FAILED;}
#define OPS_ERR_IF_CONDITION(context, condition, message, ...) OPS_ERR_IF((condition), \
     OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), message, __VA_ARGS__), return ge::GRAPH_FAILED)
#define RETURN_IF_FAILED(X) if (X==ge::GRAPH_FAILED) {return ge::GRAPH_FAILED;}
#define TO_CHAR(X) #X
#define MIN(A, B) ((A)>(B))?(B):(A)
#define MAX(A, B) ((A)<=(B))?(B):(A)

#ifdef DEBUG_MODE
#define PRINT_VAL(x) std::cout << #x << " = " << x << std::endl
#else
#define PRINT_VAL(x)
#endif
using std::map;
using std::string;

constexpr uint64_t INPUT_A_IDX = 0;
constexpr uint64_t OUTPUT_S_IDX = 1;
constexpr uint64_t OUTPUT_U_IDX = 0;
constexpr uint64_t OUTPUT_V_IDX = 2;
constexpr uint32_t ATTR_NUM_ITERATIONS = 0;

namespace optiling {

// --------------------------TilingPrepare -------------------------------------
static ge::graphStatus TilingPrepareForJacobi(gert::TilingParseContext* /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------JMTiling-----------------------
ge::graphStatus JMTiling::InitAInput()
{
    auto inputDescPtr = context_->GetInputDesc(INPUT_A_IDX);
    OPS_LOG_E_IF_NULL(context_, inputDescPtr, return ge::GRAPH_FAILED);
    auto dtype = inputDescPtr->GetDataType();
    if (dtype != ge::DT_FLOAT) {
        OPS_ERR_IF_CONDITION(context_, true, "Input type %s does not support this type!!!\n", TO_CHAR(dtype));
    }
    auto inputShapePtr = context_->GetInputShape(INPUT_A_IDX);
    OPS_LOG_E_IF_NULL(context_, inputShapePtr, return ge::GRAPH_FAILED);
    const gert::Shape& inputShape = inputShapePtr->GetStorageShape();
    int64_t inputShapeSize = inputShape.GetShapeSize();
    int64_t inputShapeRank = inputShape.GetDimNum();
    OPS_ERR_IF_CONDITION(context_, (inputShapeRank < 2),
        "Input rank must be more than 1. Current rank is %i!!!\n", inputShapeRank);
    inputShapeRank--;
    jmData_.shapeInfo.nSize = static_cast<uint32_t>(inputShape.GetDim(inputShapeRank--));
    jmData_.shapeInfo.mSize = static_cast<uint32_t>(inputShape.GetDim(inputShapeRank--));
    uint32_t batchSize = 1;
    while (inputShapeRank >= 0) {
        int32_t dim = static_cast<int32_t>(inputShape.GetDim(inputShapeRank--));
        batchSize *= dim > 0 ? dim : 1;
    }
    OPS_ERR_IF_CONDITION(context_, (jmData_.shapeInfo.nSize < jmData_.shapeInfo.mSize),
        "Now Jacobi kernel does not support cases, when tensor with size (..., M, N) where M(%i)>N(%i)!!!",
        jmData_.shapeInfo.mSize, jmData_.shapeInfo.nSize);
    jmData_.shapeInfo.batchSize = batchSize;
    jmData_.shapeInfo.mIsSingularRank = jmData_.shapeInfo.mSize <= jmData_.shapeInfo.nSize;
    if (jmData_.shapeInfo.mIsSingularRank) {
        jmData_.shapeInfo.sSize = jmData_.shapeInfo.mSize;
        jmData_.shapeInfo.uDim1 = jmData_.shapeInfo.mSize;
        jmData_.shapeInfo.uDim2 = jmData_.shapeInfo.mSize;
        jmData_.shapeInfo.vDim1 = jmData_.shapeInfo.mSize;
        jmData_.shapeInfo.vDim2 = jmData_.shapeInfo.nSize;
    } else {
        jmData_.shapeInfo.sSize = jmData_.shapeInfo.nSize;
        jmData_.shapeInfo.uDim1 = jmData_.shapeInfo.mSize;
        jmData_.shapeInfo.uDim2 = jmData_.shapeInfo.nSize;
        jmData_.shapeInfo.vDim1 = jmData_.shapeInfo.nSize;
        jmData_.shapeInfo.vDim2 = jmData_.shapeInfo.nSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus JMTiling::InitSOutput()
{
    auto outputSDescPtr = context_->GetOutputDesc(OUTPUT_S_IDX);
    OPS_LOG_E_IF_NULL(context_, outputSDescPtr, return ge::GRAPH_FAILED);
    auto dtype = outputSDescPtr->GetDataType();
    if (dtype != ge::DT_FLOAT) {
        OPS_ERR_IF_CONDITION(context_, true, "Output S type does not support this type!!!", TO_CHAR(dtype));
    }
    auto outputSShapePtr = context_->GetOutputShape(OUTPUT_S_IDX);
    OPS_LOG_E_IF_NULL(context_, outputSShapePtr, return ge::GRAPH_FAILED);
    const gert::Shape& outputShape = outputSShapePtr->GetStorageShape();
    int64_t outputShapeSize = outputShape.GetShapeSize();
    int64_t outputShapeRank = outputShape.GetDimNum();
    OPS_ERR_IF_CONDITION(context_, (outputShapeRank < 1),
        "Output S rank must be more than 0. Current rank is %i!!!\n", outputShapeRank);
    uint32_t sSize = outputShape.GetDim(outputShapeRank - 1);
    OPS_ERR_IF_CONDITION(context_, sSize != jmData_.shapeInfo.sSize,
        "Input  sSize %i and output S sSize %i is not equal!!!", jmData_.shapeInfo.sSize, sSize);

    outputShapeRank -= 2;
    uint32_t batchSize = 1;
    while (outputShapeRank >= 0) {
        int32_t dim = static_cast<int32_t>(outputShape.GetDim(outputShapeRank--));
        batchSize *= dim > 0 ? dim : 1;
    }
    OPS_ERR_IF_CONDITION(context_, batchSize != jmData_.shapeInfo.batchSize,
        "Input  batch %i and output S batch %i is not equal!!!", jmData_.shapeInfo.batchSize, batchSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus JMTiling::InitVOutput()
{
    auto outputVDescPtr = context_->GetOutputDesc(OUTPUT_V_IDX);
    OPS_LOG_E_IF_NULL(context_, outputVDescPtr, return ge::GRAPH_FAILED);
    auto dtype = outputVDescPtr->GetDataType();
    if (dtype != ge::DT_FLOAT) {
        OPS_ERR_IF_CONDITION(context_, true,
            "Output V type does not support this type!!!", TO_CHAR(dtype));
    }

    auto outputVShapePtr = context_->GetOutputShape(OUTPUT_V_IDX);
    OPS_LOG_E_IF_NULL(context_, outputVShapePtr, return ge::GRAPH_FAILED);
    const gert::Shape& outputShape = outputVShapePtr->GetStorageShape();
    int64_t outputShapeSize = outputShape.GetShapeSize();
    int64_t outputShapeRank = outputShape.GetDimNum();
    OPS_ERR_IF_CONDITION(context_, (outputShapeRank < 2),
        "Output V rank must be more than 1. Current rank is %i!!!\n", outputShapeRank);

    outputShapeRank--;
    uint32_t vDim2 = static_cast<uint32_t>(outputShape.GetDim(outputShapeRank--));
    OPS_ERR_IF_CONDITION(context_, vDim2 != jmData_.shapeInfo.vDim2,
        "Input  vDim2 %i and output vDim2 %i is not equal!!!", jmData_.shapeInfo.vDim2, vDim2);
    uint32_t vDim1 = static_cast<uint32_t>(outputShape.GetDim(outputShapeRank--));
    OPS_ERR_IF_CONDITION(context_, vDim1 != jmData_.shapeInfo.vDim1,
        "Input  vDim1 %i and output vDim1 %i is not equal!!!", jmData_.shapeInfo.vDim1, vDim1);
    uint32_t batchSize = 1;
    while (outputShapeRank >= 0) {
        int32_t dim = static_cast<int32_t>(outputShape.GetDim(outputShapeRank--));
        batchSize *= dim > 0 ? dim : 1;
    }
    OPS_ERR_IF_CONDITION(context_, batchSize != jmData_.shapeInfo.batchSize,
        "Input  batch %i and output V batch %i is not equal!!!", jmData_.shapeInfo.batchSize, batchSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus JMTiling::InitUOutput()
{
    auto outputUDescPtr = context_->GetOutputDesc(OUTPUT_U_IDX);
    OPS_LOG_E_IF_NULL(context_, outputUDescPtr, return ge::GRAPH_FAILED);
    auto dtype = outputUDescPtr->GetDataType();
    if (dtype != ge::DT_FLOAT) {
        OPS_ERR_IF_CONDITION(context_, true, "Output U type does not support this type!!!", TO_CHAR(dtype));
    }

    auto outputUShapePtr = context_->GetOutputShape(OUTPUT_U_IDX);
    OPS_LOG_E_IF_NULL(context_, outputUShapePtr, return ge::GRAPH_FAILED);
    const gert::Shape& outputShape = outputUShapePtr->GetStorageShape();
    int64_t outputShapeSize = outputShape.GetShapeSize();
    int64_t outputShapeRank = outputShape.GetDimNum();
    OPS_ERR_IF_CONDITION(context_, (outputShapeRank < 2),
        "Output U rank must be more than 1. Current rank is %i!!!\n", outputShapeRank);

    outputShapeRank--;
    uint32_t uDim2 = static_cast<uint32_t>(outputShape.GetDim(outputShapeRank--));
    OPS_ERR_IF_CONDITION(context_, uDim2 != jmData_.shapeInfo.uDim2,
        "Input  uDim2 %i and output uDim2 %i is not equal!!!", jmData_.shapeInfo.uDim2, uDim2);
    uint32_t uDim1 = static_cast<uint32_t>(outputShape.GetDim(outputShapeRank--));
    OPS_ERR_IF_CONDITION(context_, uDim1 != jmData_.shapeInfo.uDim1,
        "Input  uDim1 %i and output uDim1 %i is not equal!!!", jmData_.shapeInfo.uDim1, uDim1);
    uint32_t batchSize = 1;
    while (outputShapeRank >= 0) {
        int32_t dim = static_cast<int32_t>(outputShape.GetDim(outputShapeRank--));
        batchSize *= dim > 0 ? dim : 1;
    }
    OPS_ERR_IF_CONDITION(context_, batchSize != jmData_.shapeInfo.batchSize,
        "Input  batch %i and output U batch %i is not equal!!!", jmData_.shapeInfo.batchSize, batchSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus JMTiling::InitAttrs()
{
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(context_, attrs, return ge::GRAPH_FAILED);
    auto numIterationsPtr = attrs->GetAttrPointer<int32_t>(ATTR_NUM_ITERATIONS);
    OPS_LOG_E_IF_NULL(context_, numIterationsPtr, return ge::GRAPH_FAILED);
    int numIterations = *numIterationsPtr;
    OPS_ERR_IF_CONDITION(context_, numIterations < 1,
        "Number iterations must be a positive number. Current value is %i", numIterations);
    jmData_.iterInfo.nIterations = numIterations;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus JMTiling::InitGlobalSetsInfo()
{
    uint32_t mSize = jmData_.shapeInfo.mSize;
    uint32_t nVectorCores = MIN(32, jmData_.globSetsInfo.numVectorCores);
    uint32_t usedCores = MIN(mSize / 2, nVectorCores);
    uint32_t numGlobalSets = usedCores * 2;
    int lownNumStages = static_cast<int>(std::log2(static_cast<float>(numGlobalSets)));
    uint32_t numStages = lownNumStages + (((1 << lownNumStages) < numGlobalSets) ? 1 : 0);
    uint32_t globalSetSizeMin = mSize / numGlobalSets;
    uint32_t globalSetSizeMax = globalSetSizeMin + 1;
    uint32_t numMaxGlobalsSets = mSize - numGlobalSets * globalSetSizeMin;
    uint32_t numMinGlobalSets = numGlobalSets - numMaxGlobalsSets;
    jmData_.globSetsInfo.numVectorCores = usedCores;
    PRINT_VAL(jmData_.globSetsInfo.numVectorCores);
    jmData_.globSetsInfo.numGlobSets = numGlobalSets;
    jmData_.iterInfo.nStages = numStages;
    PRINT_VAL(jmData_.globSetsInfo.numGlobSets);
    jmData_.globSetsInfo.numGlobalSetsMax = numMaxGlobalsSets;
    PRINT_VAL(jmData_.globSetsInfo.numGlobalSetsMax);
    jmData_.globSetsInfo.numGlobalSetsMin = numMinGlobalSets;
    PRINT_VAL(jmData_.globSetsInfo.numGlobalSetsMin);
    jmData_.globSetsInfo.globSetSizeMax = globalSetSizeMax;
    PRINT_VAL(jmData_.globSetsInfo.globSetSizeMax);
    jmData_.globSetsInfo.globSetSizeMin = globalSetSizeMin;
    PRINT_VAL(jmData_.globSetsInfo.globSetSizeMin);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus JMTiling::InitMemInfo()
{
    constexpr uint8_t MAX_VEC_INSTRUCTION_SIZE = 32;
    constexpr uint8_t MAX_VEC_INSTRUCTION_NUM = MAX_VEC_INSTRUCTION_SIZE / sizeof(float);
    constexpr uint8_t RESERVED_VECTOR_BLOCKS = 32;
    constexpr uint32_t RESERVED_VECTOR_MEMORY_SIZE = RESERVED_VECTOR_BLOCKS * MAX_VEC_INSTRUCTION_SIZE;

    constexpr uint8_t MASK32_BYTES = 32 / sizeof(float);
    constexpr uint8_t MASK256_BYTES = 256 / sizeof(float);

    jmData_.memInfo.vecInstructionNum = MAX_VEC_INSTRUCTION_NUM;
    jmData_.memInfo.vecInstructionSetSize = MAX_VEC_INSTRUCTION_SIZE;
    jmData_.memInfo.vecReservedMemory = RESERVED_VECTOR_MEMORY_SIZE;
    uint32_t ubSize = jmData_.memInfo.ubSize;
    uint32_t nSize = jmData_.shapeInfo.nSize;
    jmData_.memInfo.rowSize = nSize * sizeof(float);
    jmData_.memInfo.nMaskSize = MASK256_BYTES;
    jmData_.memInfo.nSizeAligned = (((nSize + jmData_.memInfo.nMaskSize - 1) / jmData_.memInfo.nMaskSize) *
                                    jmData_.memInfo.nMaskSize);
    jmData_.memInfo.nTailSize = nSize % jmData_.memInfo.nMaskSize;
    jmData_.memInfo.nRepeatsNum = jmData_.memInfo.nSizeAligned / jmData_.memInfo.nMaskSize;
    jmData_.memInfo.rowSizeAligned = jmData_.memInfo.nSizeAligned * sizeof(float);

    uint32_t uNSize = jmData_.shapeInfo.uDim2;
    jmData_.memInfo.uRowSize = uNSize * sizeof(float);
    jmData_.memInfo.uNSizeAligned = (((uNSize + jmData_.memInfo.nMaskSize - 1) / jmData_.memInfo.nMaskSize) *
                                    jmData_.memInfo.nMaskSize);
    jmData_.memInfo.uNumRepeats = jmData_.memInfo.uNSizeAligned / jmData_.memInfo.nMaskSize;
    uint32_t uRowSizeAligned = jmData_.memInfo.uNSizeAligned * sizeof(float);
    jmData_.memInfo.uTailSize = uNSize % jmData_.memInfo.nMaskSize;

    jmData_.memInfo.recordSize = nSize + uNSize;
    jmData_.memInfo.recordSizeAligned = jmData_.memInfo.nSizeAligned + jmData_.memInfo.uNSizeAligned;
    jmData_.memInfo.tmpRowSize = MAX(nSize, uNSize);
    jmData_.memInfo.tmpRowSizeAligned = MAX(jmData_.memInfo.nSizeAligned, jmData_.memInfo.uNSizeAligned);
    jmData_.memInfo.recordMemSize = jmData_.memInfo.recordSizeAligned * sizeof(float);

    jmData_.shapeInfo.vDim2Aligned = jmData_.memInfo.nSizeAligned;
    jmData_.shapeInfo.uDim2Aligned = (((jmData_.shapeInfo.uDim2 + jmData_.memInfo.nMaskSize - 1) /
        jmData_.memInfo.nMaskSize) * jmData_.memInfo.nMaskSize);

    int bucketSize = (ubSize - RESERVED_VECTOR_MEMORY_SIZE) / (3 * jmData_.memInfo.recordMemSize +
        2 * jmData_.memInfo.tmpRowSizeAligned * sizeof(float) + 2 * jmData_.memInfo.nMaskSize * sizeof(float));
    PRINT_VAL(bucketSize);
    bucketSize = MIN(bucketSize, MAX_VEC_INSTRUCTION_NUM);
    PRINT_VAL(bucketSize);
    OPS_ERR_IF_CONDITION(context_, bucketSize < 1,
        "Bucket size is less than 1 (%i), this configuration is not supported!!!", bucketSize);
    jmData_.memInfo.maxBucketSize = bucketSize;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus JMTiling::ReadAscendPlatform()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();

    jmData_.globSetsInfo.numVectorCores = aivNum;
    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    jmData_.memInfo.ubSize = static_cast<uint32_t>(ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus JMTiling::FillTiling()
{
    tilingData_.set_nIterations(jmData_.iterInfo.nIterations);
    tilingData_.set_nStages(jmData_.iterInfo.nStages);
    tilingData_.set_batchSize(jmData_.shapeInfo.batchSize);
    tilingData_.set_nSize(jmData_.shapeInfo.nSize);
    tilingData_.set_mSize(jmData_.shapeInfo.mSize);
    bool kernelTypeC1V2 = true;
    uint32_t numAiCores = jmData_.globSetsInfo.numVectorCores;
    if (kernelTypeC1V2) {
        numAiCores = (jmData_.globSetsInfo.numVectorCores + 1) / 2;
    }

    tilingData_.set_numUsedVectorCores(jmData_.globSetsInfo.numVectorCores);
    tilingData_.set_numAICores(numAiCores);
    tilingData_.set_numGlobalSets(jmData_.globSetsInfo.numGlobSets);
    tilingData_.set_numGlobalSetsWithMaxSize(jmData_.globSetsInfo.numGlobalSetsMax);
    tilingData_.set_numGlobalSetsWithMinSize(jmData_.globSetsInfo.numGlobalSetsMin);
    tilingData_.set_globalSetSizeMax(jmData_.globSetsInfo.globSetSizeMax);
    tilingData_.set_globalSetSizeMin(jmData_.globSetsInfo.globSetSizeMin);

    // MemInfo Filing
    tilingData_.set_ubSize(jmData_.memInfo.ubSize);
    tilingData_.set_nSizeAligned(jmData_.memInfo.nSizeAligned);
    tilingData_.set_vecInstructionSize(jmData_.memInfo.vecInstructionSetSize);
    tilingData_.set_vecInstructionNum(jmData_.memInfo.vecInstructionNum);
    tilingData_.set_maxBucketSize(jmData_.memInfo.maxBucketSize);
    tilingData_.set_aRowSize(jmData_.memInfo.rowSize);
    tilingData_.set_aRowSizeAligned(jmData_.memInfo.rowSizeAligned);
    tilingData_.set_nMaskSize(jmData_.memInfo.nMaskSize);
    tilingData_.set_nRepeatsNum(jmData_.memInfo.nRepeatsNum);
    tilingData_.set_nTailSize(jmData_.memInfo.nTailSize);

    tilingData_.set_uMSize(jmData_.shapeInfo.uDim1);
    tilingData_.set_uNSize(jmData_.shapeInfo.uDim2);
    tilingData_.set_uNSizeAligned(jmData_.shapeInfo.uDim2Aligned);
    tilingData_.set_vMSize(jmData_.shapeInfo.vDim1);
    tilingData_.set_vNSize(jmData_.shapeInfo.vDim2);
    tilingData_.set_vNSizeAligned(jmData_.shapeInfo.vDim2Aligned);
    tilingData_.set_sMNSize(jmData_.shapeInfo.sSize);

    tilingData_.set_uNumRepeats(jmData_.memInfo.uNumRepeats);
    tilingData_.set_uTailSize(jmData_.memInfo.uTailSize);
    tilingData_.set_recordSize(jmData_.memInfo.recordSize);
    tilingData_.set_recordSizeAligned(jmData_.memInfo.recordSizeAligned);
    tilingData_.set_tmpMemRowSize(jmData_.memInfo.tmpRowSize);
    tilingData_.set_tmpMemRowSizeAligned(jmData_.memInfo.tmpRowSizeAligned);

    return ge::GRAPH_SUCCESS;
}
// --------------------------JMTiling-----------------------
ge::graphStatus JMTiling::DoTiling()
{
    RETURN_IF_FAILED(ReadAscendPlatform());

    RETURN_IF_FAILED(InitAInput());
    RETURN_IF_FAILED(InitSOutput());
    RETURN_IF_FAILED(InitVOutput());
    RETURN_IF_FAILED(InitUOutput());
    RETURN_IF_FAILED(InitAttrs());

    RETURN_IF_FAILED(InitGlobalSetsInfo());
    RETURN_IF_FAILED(InitMemInfo());
    RETURN_IF_FAILED(FillTiling());
    size_t* workSpaces = context_->GetWorkspaceSizes(1);
    uint32_t aTmpSize = jmData_.shapeInfo.batchSize * jmData_.shapeInfo.mSize * jmData_.memInfo.recordSizeAligned *
        sizeof(float) + jmData_.shapeInfo.batchSize * jmData_.shapeInfo.sSize * sizeof(float);
    workSpaces[0] = 16 * 1024 * 1024 + 2 * aTmpSize;

    context_->SetBlockDim(tilingData_.get_numAICores());
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    context_->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForJacobi(gert::TilingContext* context)
{
    JMTiling jmTiling(context);
    return jmTiling.DoTiling();
}

// --------------------------Registering the Tiling and TilingPrepare Functions--------
IMPL_OP_OPTILING(Jacobi)
    .Tiling(TilingForJacobi)
    .TilingParse<JMCompileInfo>(TilingPrepareForJacobi);

} // namespace optiling