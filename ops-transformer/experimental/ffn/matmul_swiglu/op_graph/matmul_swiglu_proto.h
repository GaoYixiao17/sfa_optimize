/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_swiglu_proto.h
 * \brief MatmulSwiglu 算子原型定义: 融合线性投影 + SwiGLU 门控激活
 *        ([gate|up] = x @ weight (+bias); y = SiLU(gate) * up)。
 */

#ifndef OPS_OP_PROTO_INC_MATMUL_SWIGLU_H_
#define OPS_OP_PROTO_INC_MATMUL_SWIGLU_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief MatmulSwiglu: 融合 gate_proj + up_proj + SiLU + Mul 的 SwiGLU 门控 MLP 前向。
 *        计算公式:  tmp = x @ weight (+ bias)            // [M, 2N]
 *                  gate = tmp[..., 0 : N]                // 前半
 *                  up   = tmp[..., N : 2N]               // 后半
 *                  y    = SiLU(gate) * up               // [M, N], SiLU(a)=a*sigmoid(a)
 *
 * @par Inputs:
 * Two required inputs and one optional input:
 * @li x: 输入激活, 形状 [M, K]。类型为 bfloat16/float16/float32。
 * @li weight: 打包权重 [K, 2N]，前 N 列为 gate_proj，后 N 列为 up_proj。
 *             (transpose_weight=true 时为 [2N, K])。类型同 x。
 * @li bias: 可选偏置, 形状 [2N]。类型固定为 float32(在 cube 内以 fp32 累加)。
 *
 * @par Outputs:
 * @li y: 输出张量, 形状 [M, N]。类型同 x。
 *
 * @par Attributes:
 * @li transpose_weight: 可选 bool, weight 是否转置, 默认 false。
 *
 * @par Third-party framework compatibility:
 * 等价于 down 之前的 SwiGLU MLP: SiLU(gate_proj(x)) * up_proj(x)。
 */
REG_OP(MatmulSwiglu)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .ATTR(transpose_weight, Bool, false)
    .OP_END_FACTORY_REG(MatmulSwiglu)
}  // namespace ge

#endif  // OPS_OP_PROTO_INC_MATMUL_SWIGLU_H_
