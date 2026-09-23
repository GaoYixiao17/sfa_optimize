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
 * \file masked_causal_conv1d_backward_struct.h
 * \brief Tiling data struct for masked_causal_conv1d_backward on arch35
 */

#ifndef MASKED_CAUSAL_CONV1D_BACKWARD_STRUCT_H
#define MASKED_CAUSAL_CONV1D_BACKWARD_STRUCT_H

namespace MaskedCausalConv1dBackwardArch35Tiling {

struct MaskedCausalConv1dBackwardTilingDataV35 {
    // 核间切分参数
    int64_t hMainCoreCnt{0}; // h维度主核核数
    int64_t hTailCoreCnt{0}; // h维度尾核核数
    int64_t hMainSize{0};    // h维度主核处理的大小
    int64_t hTailSize{0};    // h维度尾核处理的大小

    // 主核循环参数
    int64_t hLoopCnt{0}; // 主核UB内h维度循环次数
    int64_t sLoopCnt{0}; // 主核UB内s维度循环次数

    // 主核UB切块参数
    int64_t ubMainFactorH{0};      // 主核UB内h维度主块大小
    int64_t ubTailFactorH{0};      // 主核UB内h维度尾块大小
    int64_t ubMainFactorS{0};      // 主核UB内s维度主块大小（有效输出行数，不包含重叠部分）
    int64_t ubTailFactorS{0};      // 主核UB内s维度尾块大小（有效输出行数，不包含重叠部分）
    int64_t ubMainFactorSCount{0}; // 主块大小出现次数
    int64_t ubTailFactorSCount{0}; // 尾块大小出现次数

    // 尾核循环参数
    int64_t tailHLoopCnt{0}; // 尾核UB内h维度循环次数
    int64_t tailSLoopCnt{0}; // 尾核UB内s维度循环次数

    // 尾核UB切块参数
    int64_t tailCoreUbMainFactorH{0}; // 尾核UB内h维度主块大小
    int64_t tailCoreUbTailFactorH{0}; // 尾核UB内h维度尾块大小
    int64_t tailCoreUbMainFactorS{0}; // 尾核UB内s维度主块大小（有效输出行数，不包含重叠部分）
    int64_t tailCoreUbTailFactorS{0}; // 尾核UB内s维度尾块大小（有效输出行数，不包含重叠部分）

    // 全局参数
    int64_t hasMask{0}; // 1，有mask；0，无mask
    int64_t S{0};       // S维度大小
    int64_t B{0};       // B维度大小
    int64_t H{0};       // H维度大小
    int64_t W{0};       // W维度大小
};

} // namespace MaskedCausalConv1dBackwardArch35Tiling

#endif // MASKED_CAUSAL_CONV1D_BACKWARD_STRUCT_H
