/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file swiglu_gated_mlp_tiling.cpp
 * \brief
 */

#include "swiglu_gated_mlp_tiling.h"

#include <algorithm>
#include <cstdint>

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
#define SWIGLU_GATED_MLP_TILING_CHECK(cond, log_func, expr) \
    do {                                                    \
        if (cond) {                                         \
            log_func;                                       \
            expr;                                           \
        }                                                   \
    } while (0)

constexpr uint32_t UB_RESERVED_BUFF = 0;
constexpr uint32_t DEFAULT_BLOCK_DIM = 1;
constexpr uint32_t TILE_ALIGN = 16;
constexpr uint32_t MM_MAX_BASE_M = 128;
constexpr uint32_t MM_MAX_BASE_N = 256;
constexpr uint32_t MM_MAX_BASE_K = 128;
constexpr uint32_t SWI_MAX_BASE_COLS = 2048;
constexpr uint32_t SWI_BUFFER_NUM = 2;
constexpr uint64_t SYS_WORKSPACE_SIZE = 16UL * 1024UL * 1024UL;

constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_GATE_UP_WEIGHT = 1;
constexpr size_t INPUT_DOWN_WEIGHT = 2;
constexpr size_t ATTR_CUBE_MATH_TYPE = 0;

struct TilingInputs {
    const gert::StorageShape *xShape = nullptr;
    const gert::StorageShape *gateUpWeightShape = nullptr;
    const gert::StorageShape *downWeightShape = nullptr;
    ge::DataType dtype = ge::DT_UNDEFINED;
};

struct PlatformResource {
    platform_ascendc::PlatformAscendC ascendcPlatform;
    uint32_t aicCoreNum = 0;
    uint32_t aivCoreNum = 0;
    uint64_t ubSize = 0;

    explicit PlatformResource(fe::PlatFormInfos *platformInfo) : ascendcPlatform(platformInfo)
    {
    }
};

struct StageDims {
    uint64_t totalRows = 0;
    uint64_t hiddenSize = 0;
    uint64_t gateUpSize = 0;
    uint64_t intermediateSize = 0;
    uint64_t outSize = 0;
};

template <typename T>
inline T AlignDown(T value, T align)
{
    if (align == 0 || value < align) {
        return 0;
    }
    return (value / align) * align;
}

template <typename T>
inline T DivCeil(T a, T b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b);
}

template <typename T>
inline T AlignUp(T value, T align)
{
    if (align == 0) {
        return value;
    }
    return ((value + align - 1) / align) * align;
}

inline uint32_t GetDataTypeSize(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT16:
        case ge::DT_BF16:
            return 2U;
        case ge::DT_FLOAT:
            return 4U;
        default:
            return 0U;
    }
}

inline bool HasUnknownDim(const gert::Shape &shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

inline bool GetFlattenRowsAndHidden(const gert::Shape &shape, uint64_t &rows, uint64_t &hidden)
{
    if (shape.GetDimNum() < 2) {
        return false;
    }

    rows = 1;
    for (size_t i = 0; i + 1 < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim <= 0) {
            return false;
        }
        rows *= static_cast<uint64_t>(dim);
    }

    const int64_t lastDim = shape.GetDim(shape.GetDimNum() - 1);
    if (lastDim <= 0) {
        return false;
    }
    hidden = static_cast<uint64_t>(lastDim);
    return true;
}

inline bool GetMatrixShape(const gert::Shape &shape, uint64_t &dim0, uint64_t &dim1)
{
    if (shape.GetDimNum() != 2) {
        return false;
    }
    const int64_t d0 = shape.GetDim(0);
    const int64_t d1 = shape.GetDim(1);
    if (d0 <= 0 || d1 <= 0) {
        return false;
    }
    dim0 = static_cast<uint64_t>(d0);
    dim1 = static_cast<uint64_t>(d1);
    return true;
}

inline uint32_t SelectTileSize(uint64_t dim, uint32_t limit, uint32_t align)
{
    if (dim == 0) {
        return 0;
    }

    uint64_t chosen = std::min<uint64_t>(dim, static_cast<uint64_t>(limit));
    if (align > 1) {
        uint64_t aligned = AlignDown<uint64_t>(chosen, static_cast<uint64_t>(align));
        if (aligned > 0) {
            chosen = aligned;
        }
    }
    return static_cast<uint32_t>(chosen);
}

inline uint64_t SelectTilingKey(ge::DataType dtype, int64_t stage, bool isDynamicShape)
{
    (void)isDynamicShape;

    switch (dtype) {
        case ge::DT_FLOAT16:
            if (stage == WG_MLP_STAGE_MM1) {
                return WG_MLP_KEY_FP16_MM1;
            }
            if (stage == WG_MLP_STAGE_SWIGLU) {
                return WG_MLP_KEY_FP16_SWIGLU;
            }
            return WG_MLP_KEY_FP16_MM2;
        case ge::DT_FLOAT:
            if (stage == WG_MLP_STAGE_MM1) {
                return WG_MLP_KEY_FP32_MM1;
            }
            if (stage == WG_MLP_STAGE_SWIGLU) {
                return WG_MLP_KEY_FP32_SWIGLU;
            }
            if (stage == WG_MLP_STAGE_FUSED) {
                return WG_MLP_KEY_FP32_FUSED;
            }
            return WG_MLP_KEY_FP32_MM2;
        case ge::DT_BF16:
            if (stage == WG_MLP_STAGE_MM1) {
                return WG_MLP_KEY_BF16_MM1;
            }
            if (stage == WG_MLP_STAGE_SWIGLU) {
                return WG_MLP_KEY_BF16_SWIGLU;
            }
            return WG_MLP_KEY_BF16_MM2;
        default:
            return 0;
    }
}

inline int64_t GetStage(const gert::TilingContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return WG_MLP_STAGE_MM1;
    }
    const int64_t *stage = attrs->GetAttrPointer<int64_t>(ATTR_CUBE_MATH_TYPE);
    if (stage == nullptr) {
        return WG_MLP_STAGE_MM1;
    }
    int64_t normalizedStage = *stage;
    if (normalizedStage >= WG_MLP_STAGE_DTYPE_STRIDE) {
        normalizedStage %= WG_MLP_STAGE_DTYPE_STRIDE;
    }
    if (normalizedStage == WG_MLP_STAGE_MM1 ||
        normalizedStage == WG_MLP_STAGE_SWIGLU ||
        normalizedStage == WG_MLP_STAGE_MM2 ||
        normalizedStage == WG_MLP_STAGE_FUSED) {
        return normalizedStage;
    }
    return WG_MLP_STAGE_MM1;
}

inline ge::graphStatus FillMatmulTiling(SwigluGatedMlpTilingData &tilingData,
    const platform_ascendc::PlatformAscendC &ascendcPlatform, uint32_t coreNum, ge::DataType dtype,
    uint64_t m, uint64_t n, uint64_t k)
{
    uint64_t l1Size = 0;
    uint64_t l0cSize = 0;
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    auto mmDataType = static_cast<matmul_tiling::DataType>(dtype);
    matmul_tiling::MultiCoreMatmulTiling mmTiling(ascendcPlatform);
    int32_t ret = mmTiling.SetDim(static_cast<int32_t>(std::max<uint32_t>(1U, coreNum)));
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    ret = mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    ret = mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    ret = mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, mmDataType);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    ret = mmTiling.SetBias(false);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    ret = mmTiling.SetOrgShape(static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k));
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    ret = mmTiling.SetShape(static_cast<int64_t>(m), static_cast<int64_t>(n), static_cast<int64_t>(k));
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    mmTiling.SetBufferSpace(l1Size, l0cSize, ubSize);
    ret = mmTiling.GetTiling(tilingData.matmulTiling);
    if (ret == -1) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline void FillVectorTiling(SwigluGatedMlpTilingData &tilingData, uint64_t rows, uint64_t cols, uint32_t dtypeSize,
    uint64_t ubSize)
{
    const uint32_t alignElems = std::max<uint32_t>(1U, 32U / std::max<uint32_t>(1U, dtypeSize));
    const uint32_t calcBytesPerElem = (dtypeSize * 3U * SWI_BUFFER_NUM) + (sizeof(float) * 3U);
    const uint64_t maxTileElemsRaw = std::max<uint64_t>(alignElems, ubSize / std::max<uint32_t>(1U, calcBytesPerElem));
    const uint64_t maxTileElems = std::max<uint64_t>(
        static_cast<uint64_t>(alignElems),
        AlignDown<uint64_t>(maxTileElemsRaw, static_cast<uint64_t>(alignElems)));

    uint32_t baseCols = SelectTileSize(cols, SWI_MAX_BASE_COLS, alignElems);
    if (baseCols == 0) {
        baseCols = alignElems;
    }
    while (baseCols > alignElems && static_cast<uint64_t>(baseCols) > maxTileElems) {
        baseCols = AlignDown<uint32_t>(baseCols / 2U, alignElems);
    }
    baseCols = std::min<uint32_t>(baseCols, static_cast<uint32_t>(cols));

    const uint32_t alignedBaseCols = static_cast<uint32_t>(
        AlignUp<uint64_t>(static_cast<uint64_t>(baseCols), static_cast<uint64_t>(alignElems)));
    uint32_t baseRows = static_cast<uint32_t>(std::max<uint64_t>(1UL, maxTileElems / alignedBaseCols));
    baseRows = std::min<uint32_t>(baseRows, static_cast<uint32_t>(std::max<uint64_t>(1UL, rows)));
    baseRows = std::min<uint32_t>(baseRows, 4095U);

    tilingData.set_swiBaseRows(baseRows);
    tilingData.set_swiBaseCols(baseCols);
}

inline bool IsMatmulStage(int64_t stage)
{
    return stage == WG_MLP_STAGE_MM1 || stage == WG_MLP_STAGE_MM2;
}

ge::graphStatus GetTilingInputs(gert::TilingContext *context, TilingInputs &inputs)
{
    inputs.xShape = context->GetInputShape(INPUT_X);
    inputs.gateUpWeightShape = context->GetInputShape(INPUT_GATE_UP_WEIGHT);
    inputs.downWeightShape = context->GetInputShape(INPUT_DOWN_WEIGHT);
    auto *xDesc = context->GetInputDesc(INPUT_X);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputs.xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputs.gateUpWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputs.downWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    inputs.dtype = xDesc->GetDataType();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetPlatformResource(gert::TilingContext *context, PlatformResource &resource)
{
    resource.aicCoreNum = resource.ascendcPlatform.GetCoreNumAic();
    resource.aivCoreNum = resource.ascendcPlatform.GetCoreNumAiv();
    resource.ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, resource.ubSize);

    SWIGLU_GATED_MLP_TILING_CHECK((resource.aicCoreNum == 0) || (resource.aivCoreNum == 0) ||
                                      (resource.ubSize <= UB_RESERVED_BUFF),
        OP_LOGE(context->GetNodeName(), "Invalid platform resource, aicCoreNum:%u aivCoreNum:%u ubSize:%lu",
            resource.aicCoreNum, resource.aivCoreNum, resource.ubSize),
        return ge::GRAPH_FAILED);

    resource.ubSize -= UB_RESERVED_BUFF;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitBasicTiling(gert::TilingContext *context, const TilingInputs &inputs, int64_t stage,
    const PlatformResource &resource, StageDims &dims, SwigluGatedMlpTilingData &tilingData)
{
    SWIGLU_GATED_MLP_TILING_CHECK(
        !GetFlattenRowsAndHidden(inputs.xShape->GetStorageShape(), dims.totalRows, dims.hiddenSize),
        OP_LOGE(context->GetNodeName(), "Invalid x shape."),
        return ge::GRAPH_FAILED);

    ge::DataType dtype = inputs.dtype;
    const uint32_t dtypeSize = GetDataTypeSize(dtype);
    SWIGLU_GATED_MLP_TILING_CHECK(dtypeSize == 0,
        OP_LOGE(context->GetNodeName(), "Unsupported dtype:%d", static_cast<int32_t>(dtype)),
        return ge::GRAPH_FAILED);

    const bool isDynamicShape = HasUnknownDim(inputs.xShape->GetOriginShape()) ||
        HasUnknownDim(inputs.gateUpWeightShape->GetOriginShape()) ||
        HasUnknownDim(inputs.downWeightShape->GetOriginShape());
    tilingData.set_dtypeTag(static_cast<uint32_t>(SelectTilingKey(dtype, stage, isDynamicShape)));
    tilingData.set_stage(static_cast<uint32_t>(stage));
    tilingData.set_isDynamicShape(isDynamicShape ? 1U : 0U);
    tilingData.set_dtypeSize(dtypeSize);
    tilingData.set_isDoubleBuffer((resource.ubSize >= (256UL * 1024UL)) ? 1U : 0U);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillSwigluStageDims(gert::TilingContext *context, uint64_t ubSize, StageDims &dims,
    SwigluGatedMlpTilingData &tilingData)
{
    SWIGLU_GATED_MLP_TILING_CHECK(dims.hiddenSize == 0 || (dims.hiddenSize % 2 != 0),
        OP_LOGE(context->GetNodeName(), "stage swiglu input last dim must be even, got %lu", dims.hiddenSize),
        return ge::GRAPH_FAILED);
    dims.gateUpSize = dims.hiddenSize;
    dims.intermediateSize = dims.hiddenSize / 2U;
    dims.outSize = dims.intermediateSize;
    FillVectorTiling(tilingData, dims.totalRows, dims.intermediateSize, tilingData.get_dtypeSize(), ubSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillFusedStageDims(gert::TilingContext *context, const TilingInputs &inputs, StageDims &dims)
{
    uint64_t gateUpIn = 0;
    uint64_t gateUpOut = 0;
    uint64_t downIn = 0;
    SWIGLU_GATED_MLP_TILING_CHECK(!GetMatrixShape(inputs.gateUpWeightShape->GetStorageShape(), gateUpIn, gateUpOut),
        OP_LOGE(context->GetNodeName(), "Invalid fused gate_up_weight shape."),
        return ge::GRAPH_FAILED);
    SWIGLU_GATED_MLP_TILING_CHECK(!GetMatrixShape(inputs.downWeightShape->GetStorageShape(), downIn, dims.outSize),
        OP_LOGE(context->GetNodeName(), "Invalid fused down_weight shape."),
        return ge::GRAPH_FAILED);
    SWIGLU_GATED_MLP_TILING_CHECK(gateUpIn != dims.hiddenSize,
        OP_LOGE(context->GetNodeName(), "fused hidden mismatch, x:%lu gate_up_weight:%lu", dims.hiddenSize, gateUpIn),
        return ge::GRAPH_FAILED);
    SWIGLU_GATED_MLP_TILING_CHECK(gateUpOut == 0 || (gateUpOut % 2 != 0),
        OP_LOGE(context->GetNodeName(), "fused gate_up_weight output dim must be even, got %lu", gateUpOut),
        return ge::GRAPH_FAILED);
    dims.gateUpSize = gateUpOut;
    dims.intermediateSize = gateUpOut / 2U;
    SWIGLU_GATED_MLP_TILING_CHECK(downIn != dims.intermediateSize,
        OP_LOGE(context->GetNodeName(), "fused down_weight input mismatch, down:%lu intermediate:%lu",
            downIn, dims.intermediateSize),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillMatmulStageDims(gert::TilingContext *context, const TilingInputs &inputs, int64_t stage,
    const PlatformResource &resource, StageDims &dims, SwigluGatedMlpTilingData &tilingData)
{
    uint64_t weightIn = 0;
    uint64_t weightOut = 0;
    SWIGLU_GATED_MLP_TILING_CHECK(!GetMatrixShape(inputs.gateUpWeightShape->GetStorageShape(), weightIn, weightOut),
        OP_LOGE(context->GetNodeName(), "Invalid matmul weight shape."),
        return ge::GRAPH_FAILED);
    SWIGLU_GATED_MLP_TILING_CHECK(weightIn != dims.hiddenSize,
        OP_LOGE(context->GetNodeName(), "matmul hidden mismatch, x:%lu weight:%lu", dims.hiddenSize, weightIn),
        return ge::GRAPH_FAILED);

    dims.outSize = weightOut;
    dims.gateUpSize = (stage == WG_MLP_STAGE_MM1) ? weightOut : 0U;
    dims.intermediateSize = (stage == WG_MLP_STAGE_MM1) ? (weightOut / 2U) : dims.hiddenSize;
    SWIGLU_GATED_MLP_TILING_CHECK(stage == WG_MLP_STAGE_MM1 && (weightOut == 0 || (weightOut % 2 != 0)),
        OP_LOGE(context->GetNodeName(), "stage mm1 output dim must be even, got %lu", weightOut),
        return ge::GRAPH_FAILED);

    ge::DataType dtype = inputs.dtype;
    SWIGLU_GATED_MLP_TILING_CHECK(FillMatmulTiling(tilingData, resource.ascendcPlatform, resource.aicCoreNum, dtype,
                                      dims.totalRows, dims.outSize, dims.hiddenSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context->GetNodeName(), "Fill matmul tiling failed."),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FillStageDims(gert::TilingContext *context, const TilingInputs &inputs, int64_t stage,
    const PlatformResource &resource, StageDims &dims, SwigluGatedMlpTilingData &tilingData)
{
    if (stage == WG_MLP_STAGE_SWIGLU) {
        return FillSwigluStageDims(context, resource.ubSize, dims, tilingData);
    }
    if (stage == WG_MLP_STAGE_FUSED) {
        return FillFusedStageDims(context, inputs, dims);
    }
    return FillMatmulStageDims(context, inputs, stage, resource, dims, tilingData);
}

uint32_t SelectUsedCoreNum(int64_t stage, const PlatformResource &resource, const StageDims &dims,
    SwigluGatedMlpTilingData &tilingData)
{
    uint32_t usedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(std::max<uint64_t>(1UL, dims.totalRows), static_cast<uint64_t>(resource.aicCoreNum)));
    if (stage == WG_MLP_STAGE_SWIGLU) {
        return std::min<uint32_t>(resource.aivCoreNum,
            static_cast<uint32_t>(std::max<uint64_t>(
                1UL, dims.totalRows * std::max<uint64_t>(1UL, dims.intermediateSize))));
    }
    if (IsMatmulStage(stage)) {
        return std::max<uint32_t>(DEFAULT_BLOCK_DIM, tilingData.matmulTiling.get_usedCoreNum());
    }
    return (dims.totalRows == 0) ? DEFAULT_BLOCK_DIM : usedCoreNum;
}

void FillCommonTiling(int64_t stage, uint32_t usedCoreNum, const StageDims &dims,
    SwigluGatedMlpTilingData &tilingData)
{
    const uint32_t baseRowsPerCore =
        static_cast<uint32_t>(dims.totalRows / static_cast<uint64_t>(usedCoreNum));
    const uint32_t tailRows =
        static_cast<uint32_t>(dims.totalRows % static_cast<uint64_t>(usedCoreNum));
    const uint32_t tileRows =
        std::max<uint32_t>(1U, baseRowsPerCore + ((tailRows > 0U) ? 1U : 0U));

    tilingData.set_is32BAligned(
        ((dims.hiddenSize * tilingData.get_dtypeSize()) % 32U == 0U &&
         (std::max<uint64_t>(1UL, dims.gateUpSize) * tilingData.get_dtypeSize()) % 32U == 0U &&
         (std::max<uint64_t>(1UL, dims.intermediateSize) * tilingData.get_dtypeSize()) % 32U == 0U &&
         (std::max<uint64_t>(1UL, dims.outSize) * tilingData.get_dtypeSize()) % 32U == 0U) ? 1U : 0U);

    tilingData.set_usedCoreNum(usedCoreNum);
    tilingData.set_totalRows(dims.totalRows);
    tilingData.set_hiddenSize(dims.hiddenSize);
    tilingData.set_gateUpSize(dims.gateUpSize);
    tilingData.set_intermediateSize(dims.intermediateSize);
    tilingData.set_outSize(dims.outSize);
    tilingData.set_baseRowsPerCore(baseRowsPerCore);
    tilingData.set_tailRows(tailRows);
    tilingData.set_tileRows(tileRows);

    tilingData.set_mm1BaseM(SelectTileSize(tileRows, MM_MAX_BASE_M, 1));
    tilingData.set_mm1BaseN(SelectTileSize(dims.gateUpSize, MM_MAX_BASE_N, TILE_ALIGN));
    tilingData.set_mm1BaseK(SelectTileSize(dims.hiddenSize, MM_MAX_BASE_K, TILE_ALIGN));
    if (stage != WG_MLP_STAGE_SWIGLU) {
        tilingData.set_swiBaseRows(SelectTileSize(tileRows, MM_MAX_BASE_M, 1));
        tilingData.set_swiBaseCols(SelectTileSize(dims.intermediateSize, SWI_MAX_BASE_COLS, TILE_ALIGN));
    }
    tilingData.set_mm2BaseM(SelectTileSize(tileRows, MM_MAX_BASE_M, 1));
    tilingData.set_mm2BaseN(SelectTileSize(dims.outSize, MM_MAX_BASE_N, TILE_ALIGN));
    tilingData.set_mm2BaseK(SelectTileSize(dims.intermediateSize, MM_MAX_BASE_K, TILE_ALIGN));
}

ge::graphStatus SaveTilingAndWorkspace(gert::TilingContext *context, int64_t stage, uint32_t usedCoreNum,
    const StageDims &dims, SwigluGatedMlpTilingData &tilingData)
{
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingData.get_dtypeTag());

    size_t *userWorkspaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkspaceSize);
    if (stage == WG_MLP_STAGE_SWIGLU) {
        userWorkspaceSize[0] = 0UL;
    } else if (stage == WG_MLP_STAGE_FUSED) {
        userWorkspaceSize[0] = static_cast<size_t>(
            SYS_WORKSPACE_SIZE +
            std::max<uint64_t>(1UL, static_cast<uint64_t>(usedCoreNum)) *
                dims.intermediateSize * tilingData.get_dtypeSize());
    } else {
        userWorkspaceSize[0] = static_cast<size_t>(SYS_WORKSPACE_SIZE);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForSwigluGatedMlp(gert::TilingParseContext *context)
{
    SWIGLU_GATED_MLP_TILING_CHECK(context == nullptr,
        OP_LOGE("SwigluGatedMlp", "context is null"),
        return ge::GRAPH_FAILED);

    auto *platformInfo = context->GetPlatformInfo();
    SWIGLU_GATED_MLP_TILING_CHECK(platformInfo == nullptr,
        OP_LOGE(context->GetNodeName(), "platformInfo is null"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForSwigluGatedMlp(gert::TilingContext *context)
{
    SWIGLU_GATED_MLP_TILING_CHECK(context == nullptr,
        OP_LOGE("SwigluGatedMlp", "context is null"),
        return ge::GRAPH_FAILED);

    TilingInputs inputs;
    if (GetTilingInputs(context, inputs) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    fe::PlatFormInfos *platformInfo = context->GetPlatformInfo();
    SWIGLU_GATED_MLP_TILING_CHECK(platformInfo == nullptr,
        OP_LOGE(context->GetNodeName(), "platformInfo is null"),
        return ge::GRAPH_FAILED);

    PlatformResource resource(platformInfo);
    if (GetPlatformResource(context, resource) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    const int64_t stage = GetStage(context);
    StageDims dims;
    SwigluGatedMlpTilingData tilingData;
    if (InitBasicTiling(context, inputs, stage, resource, dims, tilingData) != ge::GRAPH_SUCCESS ||
        FillStageDims(context, inputs, stage, resource, dims, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t usedCoreNum = SelectUsedCoreNum(stage, resource, dims, tilingData);
    FillCommonTiling(stage, usedCoreNum, dims, tilingData);
    return SaveTilingAndWorkspace(context, stage, usedCoreNum, dims, tilingData);
}

}  // namespace

IMPL_OP_OPTILING(SwigluGatedMlp)
    .Tiling(TilingForSwigluGatedMlp)
    .TilingParse<SwigluGatedMlpCompileInfo>(TilingPrepareForSwigluGatedMlp);

}  // namespace optiling
