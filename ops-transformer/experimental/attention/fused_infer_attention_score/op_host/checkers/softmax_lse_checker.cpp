/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file softmax_lse_checker.cpp
 * \brief
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fused_infer_attention_score_tiling_constants.h"
#include "softmax_lse_checker.h"

namespace optiling {
using std::map;
using std::string;
using std::pair;
using namespace ge;
using namespace AscendC;
using namespace arch35FIA;

// CheckSinglePara
ge::graphStatus SoftmaxLSEChecker::CheckSingleDtype(const FiaTilingInfo &fiaInfo)
{
    // SoftmaxLse only supports output FP32
    if (fiaInfo.softmaxLseFlag) {
        OP_CHECK_IF(ge::GRAPH_SUCCESS != CheckDtypeSupport(fiaInfo.opParamInfo.lseOut.desc, SOFTMAX_LSE_NAME),
                    OP_LOGE(fiaInfo.opName, "SoftmaxLse only support dtype FP32, but got %s",
                            DataTypeToSerialString(fiaInfo.opParamInfo.lseOut.desc->GetDataType()).c_str()),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckParaExistence
ge::graphStatus SoftmaxLSEChecker::CheckExistenceShapeAndDesc(const FiaTilingInfo &fiaInfo)
{
    // When softmaxLseFlag is true, both the SoftmaxLse shape and the lseOut tensor must not be null.
    if (fiaInfo.softmaxLseFlag) {
        const gert::StorageShape *lseShape = fiaInfo.opParamInfo.lseOut.shape;
        OP_CHECK_IF(lseShape == nullptr,
                    OP_LOGE(fiaInfo.opName, "SoftmaxLse shape is nullptr but softmaxLseFlag is true!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(fiaInfo.opParamInfo.lseOut.desc == nullptr,
                    OP_LOGE(fiaInfo.opName, "Desc of lseOut tensor is nullptr!"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

// CheckCrossFeature
ge::graphStatus SoftmaxLSEChecker::CheckFeatureDimAndShape(const FiaTilingInfo &fiaInfo)
{
    // When softmaxLseFlag is true and emptyTensorFlag is false:
    // -If the LayOut is TND, the shape must have 3 dimensions (matching TN1).
    // -Otherwise, the shape must have 4 dimensions (matching [B,N,S1,1]).
    if (fiaInfo.softmaxLseFlag && (!fiaInfo.emptyTensorFlag)) {
        const gert::StorageShape *lseShape = fiaInfo.opParamInfo.lseOut.shape;
        std::shared_ptr<FiaTilingShapeCompare> softmaxLseShapeCmp_ = nullptr;
        if (fiaInfo.outLayout == FiaLayout::TND || fiaInfo.outLayout == FiaLayout::NTD) {
            OP_CHECK_IF(lseShape->GetStorageShape().GetDimNum() != DIM_NUM_3,
                        OP_LOGE(fiaInfo.opName, "TND/NTD SoftmaxLse shape dim(%zu) should be 3!",
                                lseShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(lseShape->GetStorageShape().GetDim(DIM_NUM_0) != fiaInfo.qTSize ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_1) != fiaInfo.n1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_2) != SHAPE_PARAMS_CONST,
                        OP_LOGE(fiaInfo.opName, "SoftmaxLse shape[%ld, %ld, %ld] does not match TN1!",
                                lseShape->GetStorageShape().GetDim(DIM_NUM_0),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_1),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_2)),
                        return ge::GRAPH_FAILED);
        } else {
            OP_CHECK_IF(lseShape->GetStorageShape().GetDimNum() != DIM_NUM_4,
                        OP_LOGE(fiaInfo.opName, "SoftmaxLse shape dim(%zu) should be 4!",
                                lseShape->GetStorageShape().GetDimNum()),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(lseShape->GetStorageShape().GetDim(DIM_NUM_0) != fiaInfo.bSize ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_1) != fiaInfo.n1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_2) != fiaInfo.s1Size ||
                lseShape->GetStorageShape().GetDim(DIM_NUM_3) != SHAPE_PARAMS_CONST,
                        OP_LOGE(fiaInfo.opName, "SoftmaxLse shape[%ld, %ld, %ld, %ld] does not match [B, N, Q_S, 1]!",
                                lseShape->GetStorageShape().GetDim(DIM_NUM_0),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_1),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_2),
                                lseShape->GetStorageShape().GetDim(DIM_NUM_3)),
                        return ge::GRAPH_FAILED);
        }
    }
    return ge::GRAPH_SUCCESS;
}

// enableAntiQuant 相关校验函数
ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckSingleDtype(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckParaExistence(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckExistenceShapeAndDesc(fiaInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckCrossFeature(const FiaTilingInfo &fiaInfo)
{
    if (ge::GRAPH_SUCCESS != CheckFeatureDimAndShape(fiaInfo)) {
            return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo)
{
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
