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
 * \file quant_sparse_flash_mla_check_feature.cpp
 * \brief
 */

#include "quant_sparse_flash_mla_check.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using std::pair;
namespace optiling {

ge::graphStatus QSMLATilingCheck::CheckFeatureWinKV() const
{
    OP_CHECK_IF(oriWinLeft_ != 127, // 127:当前不泛化
        OP_LOGE(opName_, "oriWinLeft_ only support 127, but got %u", oriWinLeft_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(oriWinRight_ != 0, // 0:当前不泛化
        OP_LOGE(opName_, "oriWinRight_ only support 0, but got %u", oriWinRight_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantShape() const
{
    OP_CHECK_IF(bSize_ <= 0,
        OP_LOGE(opName_, "batch_size should be greater than 0, but got %u", bSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(qTSize_ <= 0 && (qLayout_ == QSMLALayout::TND),
            OP_LOGE(opName_, "T_size of query should be greater than 0, but got %u", qTSize_),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ != 64 && n1Size_ != 128,
            OP_LOGE(opName_, "q_head_num only support 64 and 128, but got %u", n1Size_),
            return ge::GRAPH_FAILED);

    OP_CHECK_IF(n2Size_ != 1,
        OP_LOGE(opName_, "kv_head_num only support 1, but got %u", n2Size_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(n1Size_ % n2Size_ != 0,
        OP_LOGE(opName_, "q_head_num(%u) must be divisible by kv_head_num(%u)", n1Size_, n2Size_),
        return ge::GRAPH_FAILED);

    std::vector<uint32_t> gSizeSupportList = {64, 128};
    OP_CHECK_IF(std::find(gSizeSupportList.begin(), gSizeSupportList.end(), gSize_) == gSizeSupportList.end(),
        OP_LOGE(opName_, "group num only support 64 and 128, but got %u", gSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSize_ != 512, // 512:当前不泛化
        OP_LOGE(opName_, "Head dim of input q only support 512, but got %u", dSize_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSizeV_ != 512, // 512:当前不泛化
        OP_LOGE(opName_, "dSizeV only support 512, but got %u", dSizeV_),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(dSizeVInput_ != 512, // 512:当前不泛化
        OP_LOGE(opName_, "dSizeVInput only support 512, but got %u", dSizeVInput_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantLayout() const
{
    const std::vector<std::string> layoutSupportList = {
        "BSND",
        "TND"
    };
    std::string layoutQuery = opParamInfo_.layoutQ;
    OP_CHECK_IF(std::find(layoutSupportList.begin(), layoutSupportList.end(), layoutQuery) == layoutSupportList.end(),
        OP_LOGE(opName_, "layoutQuery only support BSND/TND, but got %s", layoutQuery.c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantDtype() const
{
    OP_CHECK_IF(qType_ != ge::DT_HIFLOAT8,
        OP_LOGE(opName_, "query dtype only support %s, but got %s",
            QSMLADataTypeToSerialString(ge::DT_HIFLOAT8).c_str(),
            QSMLADataTypeToSerialString(qType_).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureQuantModeAndDtype() const
{
    // qkv_quant_mode=1: q, ori_kv, cmp_kv are all HIFLOAT8
    OP_CHECK_IF(*opParamInfo_.qkvQuantMode != 1,
        OP_LOGE(opName_, "qkv_quant_mode only support 1, but got %ld",
        *opParamInfo_.qkvQuantMode),
        return ge::GRAPH_FAILED);

    // qkv_quant_mode=1 下，Q、ori_kv 和 cmp_kv 数据类型都必须是 HIFLOAT8
    OP_CHECK_IF(qType_ != ge::DT_HIFLOAT8,
        OP_LOGE(opName_, "when qkv_quant_mode=1, query dtype must be DT_HIFLOAT8, but got %s.",
        QSMLADataTypeToSerialString(qType_).c_str()),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(opParamInfo_.oriKv.desc->GetDataType() != ge::DT_HIFLOAT8,
        OP_LOGE(opName_, "when qkv_quant_mode=1, oriKv dtype must be DT_HIFLOAT8, but got %s.",
        QSMLADataTypeToSerialString(opParamInfo_.oriKv.desc->GetDataType()).c_str()),
        return ge::GRAPH_FAILED);

    if (opParamInfo_.cmpKv.desc != nullptr) {
        OP_CHECK_IF(opParamInfo_.cmpKv.desc->GetDataType() != ge::DT_HIFLOAT8,
            OP_LOGE(opName_, "when qkv_quant_mode=1, cmpKv dtype must be DT_HIFLOAT8, but got %s.",
            QSMLADataTypeToSerialString(opParamInfo_.cmpKv.desc->GetDataType()).c_str()),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquantPa() const
{
    if (kvLayout_ != QSMLALayout::PA_BBND) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(oriBlockSize_ <= 0 || oriBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
        OP_LOGE(opName_, "when page attention is enabled, oriBlockSize_(%u) should be in range (0, %u].",
        oriBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);

    if (cmpBlockSize_ != 0) {
        OP_CHECK_IF(cmpBlockSize_ <= 0 || cmpBlockSize_ > static_cast<int32_t>(MAX_BLOCK_SIZE),
            OP_LOGE(opName_, "when page attention is enabled, cmpBlockSize_(%u) should be in range (0, %u].",
            cmpBlockSize_, MAX_BLOCK_SIZE), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeatureAntiquant() const
{
    if (ge::GRAPH_SUCCESS != CheckFeatureQuantModeAndDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantShape() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantLayout() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantDtype() ||
        ge::GRAPH_SUCCESS != CheckFeatureWinKV() ||
        ge::GRAPH_SUCCESS != CheckFeatureAntiquantPa()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSMLATilingCheck::CheckFeature() const
{
    return CheckFeatureAntiquant();
}

}