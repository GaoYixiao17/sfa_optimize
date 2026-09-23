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
 * \file post_quant_checker.h
 * \brief
 */

#ifndef POST_QUANT_CHECKER_H
#define POST_QUANT_CHECKER_H

#include <map>
#include <numeric>
#include "tiling/tiling_api.h"
#include "base_checker.h"

namespace optiling {

class PostQuantChecker : public BaseChecker {
public:
    PostQuantChecker(bool enableNonQuant, bool enableFullQuant, bool enableAntiQuant) :
        BaseChecker(enableNonQuant, enableFullQuant, enableAntiQuant) {}
    ~PostQuantChecker() override = default;

    ge::graphStatus CheckSinglePara(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckParaExistence(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckCrossFeature(const FiaTilingInfo &fiaInfo) override;
    ge::graphStatus CheckMultiParaConsistency(const FiaTilingInfo &fiaInfo) override;

private:
    // 公共校验函数
    ge::graphStatus CheckSingleDtype(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckExistenceQuantScale2(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureAttenOut(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureQueryDType(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureLayout(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureOutputEqual(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeaturePrefix(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckFeatureRowValid(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckAntiquantNotSupport(const FiaTilingInfo &fiaInfo);
    ge::graphStatus CheckMultiParaQuantOffset2(const FiaTilingInfo &fiaInfo);
private:
};

} // namespace optiling
#endif // POST_QUANT_CHECKER_H