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
 * \file quant_sparse_flash_mla_check.cpp
 * \brief
 */

#include "quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

std::string QSMLADataTypeToSerialString(ge::DataType type)
{
    const auto it = DATATYPE_TO_STRING_MAP.find(type);
    if (it != DATATYPE_TO_STRING_MAP.end()) {
        return it->second;
    } else {
        OP_LOGE("QuantSparseFlashMla ", "datatype %d not support", type);
        return "UNDEFINED";
    }
}

std::string QSMLALayoutToSerialString(QSMLALayout layout)
{
    switch (layout) {
        case QSMLALayout::BSND: return "BSND";
        case QSMLALayout::TND: return "TND";
        case QSMLALayout::PA_BBND: return "PA_BBND";
        default: return "UNKNOWN";
    }
}

std::string GetShapeStr(gert::Shape shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

bool QSMLATilingCheck::HasAxis(const QSMLAAxis &axis, const QSMLALayout &layout, const gert::Shape &shape) const
{
    const auto& layoutIt = QSMLA_LAYOUT_AXIS_MAP.find(layout);
    if (layoutIt == QSMLA_LAYOUT_AXIS_MAP.end()) {
        return false;
    }

    const std::vector<QSMLAAxis>& axes = layoutIt->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    if (axisIt == axes.end()) {
        return false;
    }
    const auto& dimIt = QSMLA_LAYOUT_DIM_MAP.find(layout);
    if (dimIt == QSMLA_LAYOUT_DIM_MAP.end() || dimIt->second != shape.GetDimNum()) {
        return false;
    }
    return true;
}

size_t QSMLATilingCheck::GetAxisIdx(const QSMLAAxis &axis, const QSMLALayout &layout) const
{
    const std::vector<QSMLAAxis>& axes = QSMLA_LAYOUT_AXIS_MAP.find(layout)->second;
    const auto& axisIt = std::find(axes.begin(), axes.end(), axis);
    return std::distance(axes.begin(), axisIt);
}

uint32_t QSMLATilingCheck::GetAxisNum(const gert::Shape &shape, const QSMLAAxis &axis, const QSMLALayout &layout) const
{
    return HasAxis(axis, layout, shape) ? shape.GetDim(GetAxisIdx(axis, layout)) : invalidDimValue_;
}

void QSMLATilingCheck::Init()
{
    opName_ = qsmlaInfo_.opName;
    platformInfo_ = qsmlaInfo_.platformInfo;
    opParamInfo_ = qsmlaInfo_.opParamInfo;
    socVersion_ = qsmlaInfo_.socVersion;

    qLayout_ = qsmlaInfo_.qLayout;
    kvLayout_ = qsmlaInfo_.kvLayout;
    outLayout_ = qsmlaInfo_.outLayout;

    if (opParamInfo_.oriBlockTable.tensor != nullptr) {
        bSize_ = opParamInfo_.oriBlockTable.tensor->GetShape().GetStorageShape().GetDim(0);
    } else {
        bSize_ = qsmlaInfo_.bSize;
    }
    n1Size_ = qsmlaInfo_.n1Size;
    n2Size_ = qsmlaInfo_.n2Size;
    s1Size_ = qsmlaInfo_.s1Size;
    s2Size_ = qsmlaInfo_.s2Size;
    gSize_ = qsmlaInfo_.gSize;
    qkHeadDim_ = qsmlaInfo_.qkHeadDim;
    qTSize_ = qsmlaInfo_.qTSize;

    dSize_ = qsmlaInfo_.dSize;
    dSizeV_ = qsmlaInfo_.dSizeV;
    if (opParamInfo_.oriKv.tensor != nullptr) {
        dSizeOriKvInput_ = GetAxisNum(opParamInfo_.oriKv.tensor->GetStorageShape(), QSMLAAxis::D, kvLayout_);
    }
    if (opParamInfo_.cmpKv.tensor != nullptr) {
        dSizeCmpKvInput_ = GetAxisNum(opParamInfo_.cmpKv.tensor->GetStorageShape(), QSMLAAxis::D, kvLayout_);
    }

    maxActualseq_ = qsmlaInfo_.maxActualseq;

    oriMaxBlockNumPerBatch_ = qsmlaInfo_.oriMaxBlockNumPerBatch;
    cmpMaxBlockNumPerBatch_ = qsmlaInfo_.cmpMaxBlockNumPerBatch;

    oriBlockSize_ = qsmlaInfo_.oriBlockSize;
    cmpBlockSize_ = qsmlaInfo_.cmpBlockSize;

    sparseBlockCount_ = qsmlaInfo_.sparseBlockCount;
    sparseBlockSize_ = qsmlaInfo_.sparseBlockSize;

    cmpRatio_ = qsmlaInfo_.cmpRatio;
    oriWinLeft_ = qsmlaInfo_.oriWinLeft;
    oriWinRight_ = qsmlaInfo_.oriWinRight;

    oriMaskMode_ = qsmlaInfo_.oriMaskMode;
    cmpMaskMode_ = qsmlaInfo_.cmpMaskMode;

    qType_ = qsmlaInfo_.qType;
    oriKvType_ = qsmlaInfo_.oriKvType;
    cmpKvType_ = qsmlaInfo_.cmpKvType;
    outputType_ = qsmlaInfo_.outputType;

    if (opParamInfo_.cmpKv.tensor == nullptr) {
        perfMode_ = QSMLATemplateMode::SWA_TEMPLATE_MODE;
    } else if (opParamInfo_.cmpSparseIndices.tensor != nullptr) {
        perfMode_ = QSMLATemplateMode::SCFA_TEMPLATE_MODE;
    } else {
        perfMode_ = QSMLATemplateMode::CFA_TEMPLATE_MODE;
    }
}

ge::graphStatus QSMLATilingCheck::Process()
{
    Init();
    if (CheckSinglePara() != ge::GRAPH_SUCCESS ||
        CheckParaExistence() != ge::GRAPH_SUCCESS ||
        CheckFeature() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

}