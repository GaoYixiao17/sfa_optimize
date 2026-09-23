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
 * \file quant_sals_indexer_proto.cpp
 * \brief
 */
#include <cmath>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"


using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;

constexpr uint32_t ATTR_MAX_SEQLEN_KEY_INDEX = 0;
constexpr uint32_t ATTR_SPARSE_BLOCK_SIZE_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_RATIO_INDEX = 2;
constexpr uint32_t ATTR_FIXED_TAIL_COUNT_INDEX = 3;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 4;

constexpr uint32_t OUTPUT_IDX_SPARSE_INDICES = 0;
constexpr uint32_t OUTPUT_IDX_SPARSE_SEQ_LENGTHS_KEY = 1;

static ge::graphStatus InferShapeQuantSalsIndexer(gert::InferShapeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("QuantSalsIndexer", "InferShapeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    const gert::Shape *queryShape = context->GetInputShape(QUERY_INDEX);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    const gert::Shape *keyShape = context->GetInputShape(KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, keyShape, return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OPS_LOG_E_IF_NULL(context, attrs, return ge::GRAPH_FAILED);
    const char *inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OPS_LOG_E_IF_NULL(context, inputLayoutKeyPtr, return ge::GRAPH_FAILED);

    const int64_t *maxSeqlenKey = attrs->GetInt(ATTR_MAX_SEQLEN_KEY_INDEX);
    OPS_LOG_E_IF_NULL(context, maxSeqlenKey, return ge::GRAPH_FAILED);
    const int64_t *sparseBlockSize = attrs->GetInt(ATTR_SPARSE_BLOCK_SIZE_INDEX);
    OPS_LOG_E_IF_NULL(context, sparseBlockSize, return ge::GRAPH_FAILED);
    const int64_t *fixedTailCount = attrs->GetInt(ATTR_FIXED_TAIL_COUNT_INDEX);
    OPS_LOG_E_IF_NULL(context, fixedTailCount, return ge::GRAPH_FAILED);
    const float *sparse_ratio = attrs->GetFloat(ATTR_SPARSE_RATIO_INDEX);
    OPS_LOG_E_IF_NULL(context, sparse_ratio, return ge::GRAPH_FAILED);

    int64_t totalNCount = ((*maxSeqlenKey) + ((*sparseBlockSize) - 1)) / (*sparseBlockSize);
    int64_t sparseCount;
    int64_t selectCount;
    if (totalNCount < (*fixedTailCount)) {
        sparseCount = totalNCount;
        selectCount = 0;
    } else {
        selectCount = static_cast<int64_t>(std::ceil((static_cast<double>(totalNCount) - static_cast<double>(*fixedTailCount)) * (std::round((1.0 - *sparse_ratio)*100.0) / 100.0)));
        if (selectCount > 2048) {
            selectCount = 2048;
        }
        sparseCount =  selectCount + (*fixedTailCount);
    }

    gert::Shape *sparseIndicesShape = context->GetOutputShape(OUTPUT_IDX_SPARSE_INDICES);
    OPS_LOG_E_IF_NULL(context, sparseIndicesShape, ge::GRAPH_FAILED);

    sparseIndicesShape->SetDimNum(queryShape->GetDimNum());
    OPS_ERR_IF(
        queryShape->GetDimNum() != 3,
        OPS_LOG_E(context, "queryDims (%zu) must be 3!", queryShape->GetDimNum()),
        return ge::GRAPH_FAILED);
    sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
    sparseIndicesShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim N2
    sparseIndicesShape->SetDim(2, sparseCount);   // 2:Dim K

    OPS_LOG_D(context->GetNodeName(), "QuantSalsIndexer InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQuantSalsIndexer(gert::InferDataTypeContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_LOG_E("QuantSalsIndexer", "InferDataTypeContext is nullptr!"),
               return ge::GRAPH_FAILED);
    OPS_LOG_D(context->GetNodeName(), "Enter QuantSalsIndexer InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = ge::DT_INT32;
    // attention_out, outidx:0
    context->SetOutputDataType(0, outputType);
    OPS_LOG_D(context->GetNodeName(), "QuantSalsIndexer InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantSalsIndexer)
    .InferShape(InferShapeQuantSalsIndexer)
    .InferDataType(InferDataTypeQuantSalsIndexer);
} // namespace ops
