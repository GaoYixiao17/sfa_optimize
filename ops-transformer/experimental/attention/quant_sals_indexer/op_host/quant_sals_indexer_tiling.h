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
 * \file quant_sals_indexer_tiling.h
 * \brief
 */

#ifndef QUANT_SALS_INDEXER_TILING_H_
#define QUANT_SALS_INDEXER_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
const uint32_t NCAI_MAX_AIC_CORE_NUM = 24;      // 样例代码，当前仅支持核数24
// ------------------公共定义--------------------------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};

struct SplitParams {
    uint32_t *bN2End;
    uint32_t *gS1End;
    uint32_t *s2End;
};

enum class DataLayout : uint32_t {
    BSND = 0,
    PA_BNSD = 2,
    PA_BSND = 3,
    PA_NZ = 4
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t QUERY_DEQUANT_SCALE_INDEX = 2;
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t METADATA_INDEX = 6;

// Outputs Index
constexpr uint32_t SPARSE_INDICES_INDEXER = 0;

// Attributes Index
constexpr uint32_t ATTR_MAX_SEQLEN_KEY_INDEX = 0;
constexpr uint32_t ATTR_SPARSE_BLOCK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_RATIO_INDEX = 2;
constexpr uint32_t ATTR_FIXED_TAIL_COUNT_INDEX = 3;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 4;

// Dim Index
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// Dim Num
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;
constexpr uint32_t DIM_NUM_FIVE = 5;

// -----------算子TilingData定义---------------
// 外切分核参数
BEGIN_TILING_DATA_DEF(QuantSalsIndexerSplitParams)
TILING_DATA_FIELD_DEF_ARR(uint32_t, NCAI_MAX_AIC_CORE_NUM, bN2End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, NCAI_MAX_AIC_CORE_NUM, gS1End)
TILING_DATA_FIELD_DEF_ARR(uint32_t, NCAI_MAX_AIC_CORE_NUM, s2End)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantSalsIndexerSplitParamsOp, QuantSalsIndexerSplitParams)

BEGIN_TILING_DATA_DEF(QSITilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, dSize)
TILING_DATA_FIELD_DEF(int32_t, fixedTailCount)
TILING_DATA_FIELD_DEF(int32_t, maxSeqlenKey)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseBlockSize)
TILING_DATA_FIELD_DEF(float, sparseRatio)
TILING_DATA_FIELD_DEF_STRUCT(QuantSalsIndexerSplitParams, splitParams)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantSalsIndexer, QSITilingData)

// -----------算子CompileInfo定义-------------------
struct QSICompileInfo {};

// -----------算子Tiling入参结构体定义---------------
struct QSiParaInfo {
    TilingRequiredParaInfo query = {nullptr, nullptr};
    TilingRequiredParaInfo key = {nullptr, nullptr};
    TilingRequiredParaInfo query_dequant_scale = {nullptr, nullptr};
    TilingRequiredParaInfo key_dequant_scale = {nullptr, nullptr};
    TilingOptionalParaInfo actualSeqLengths = {nullptr, nullptr};
    TilingOptionalParaInfo blockTable = {nullptr, nullptr};
    TilingRequiredParaInfo sparseIndices = {nullptr, nullptr};
    
    const char *layOutKey = nullptr;
    const int32_t *blockSize = nullptr;
    const int32_t *maxSeqlenKey = nullptr;
    const int32_t *sparseBlockSize = nullptr;
    const float *sparseRatio = nullptr;
    const int32_t *fixedTailCount = nullptr;
};
  
// -----------算子Tiling入参信息类---------------
class QSITilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    QSiParaInfo opParamInfo;
    // Base Param
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    uint32_t bSize = 0;
    uint32_t n2Size = 0;
    int64_t s2Size = 0;
    int64_t dSize = 0;
    int32_t sparseCount = 0;
    int32_t sparseBlockSize = 0;
    int32_t fixedTailCount = 0;
    int32_t maxSeqlenKey = 0;
    float sparseRatio = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputKLayout = DataLayout::PA_BNSD;
};

// -----------算子Tiling入参信息解析及Check类---------------
class QSIInfoParser {
public:
    explicit QSIInfoParser(gert::TilingContext *context) : context_(context)
    {
    }
    ~QSIInfoParser() = default;

    ge::graphStatus CheckRequiredInOutExistence() const;
    ge::graphStatus CheckRequiredAttrExistence() const;
    ge::graphStatus CheckRequiredParaExistence() const;
    ge::graphStatus GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                        const std::string &actualSeqLenName);
    ge::graphStatus GetOpName();
    ge::graphStatus GetNpuInfo();
    void GetOptionalInputParaInfo();
    void GetInputParaInfo();
    void GetOutputParaInfo();
    ge::graphStatus GetAndCheckAttrParaInfo();
    ge::graphStatus GetOpParaInfo();
    ge::graphStatus ValidateInputShapesMatch();
    ge::graphStatus GetAndCheckInOutDataType();
    ge::graphStatus GetBatchSize();
    ge::graphStatus GetHeadDim();
    ge::graphStatus GetSparseCount();
    ge::graphStatus GetAndCheckOptionalInput();
    ge::graphStatus CheckShapeDim();
    ge::graphStatus GetAndCheckBlockSize();
    ge::graphStatus CheckBlockCount();
    ge::graphStatus GetS2SizeForPageAttention();
    ge::graphStatus GetS2Size();
    ge::graphStatus GetQueryKeyAndOutLayout();
    ge::graphStatus GetAndCheckN2Size();
    ge::graphStatus GetAttenMaskInfo();
    ge::graphStatus GetActualSeqInfo();
    void GenerateInfo(QSITilingInfo &siInfo);
    ge::graphStatus CheckScaleShape();
    ge::graphStatus ParseAndCheck(QSITilingInfo &siInfo);

public:
    gert::TilingContext *context_ = nullptr;
    const char *opName_;
    fe::PlatFormInfos *platformInfo_;
    QSiParaInfo opParamInfo_;

    // BaseParams
    uint32_t bSize_ = 0;
    uint32_t n2Size_ = 0;
    int64_t s2Size_ = 0;
    uint32_t headDim_ = 0;
    uint32_t sparseCount_ = 0;

    // Layout
    DataLayout kLayout_ = DataLayout::PA_BNSD;
    // PageAttention
    uint32_t maxBlockNumPerBatch_ = 0;
    int32_t blockSize_ = 0;
    platform_ascendc::SocVersion socVersion_ = platform_ascendc::SocVersion::ASCEND910B;
    ge::DataType inputQType_ = ge::DT_FLOAT16;
    ge::DataType inputKType_ = ge::DT_FLOAT16;
    ge::DataType inputQueryScaleType_ = ge::DT_FLOAT;
    ge::DataType inputKeyScaleType_ = ge::DT_FLOAT;
    ge::DataType blockTableType_ = ge::DT_FLOAT16;
    ge::DataType inputKRopeType_ = ge::DT_FLOAT16;
    ge::DataType sparseIndicesType_ = ge::DT_FLOAT16;
};

// ---------------算子Tiling类---------------
class QuantSalsIndexerTiling {
public:
    explicit QuantSalsIndexerTiling(gert::TilingContext *context) : context_(context){};
    ge::graphStatus DoTiling(QSITilingInfo *tilingInfo);
    void SplitCoreBN(uint32_t coreNum, QSITilingInfo *siInfo, SplitParams splitParams);
private:
    gert::TilingContext *context_ = nullptr;
    QSITilingData tilingData_;
};

} // namespace optiling
#endif // QUANT_SALS_INDEXER_TILING_H_