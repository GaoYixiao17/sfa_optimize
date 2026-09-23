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
 * \file swiglu_gated_mlp_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GATED_MLP_PROTO_H_
#define OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GATED_MLP_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
    /**
     * @brief Compute the fused SwiGLU gated MLP:
     *        y = MatMul(SwiGlu(MatMul(x, gate_up_weight)), down_weight)
     *
     * @par Inputs:
     * Three inputs, including:
     * @x: A Tensor. Shape [..., hidden_size].
     *     Must be one of the following types: bfloat16, float16, float32.
     *
     * @gate_up_weight: A Tensor. Shape [hidden_size, 2 * intermediate_size].
     *     Must be one of the following types: bfloat16, float16, float32.
     *
     * @down_weight: A Tensor. Shape [intermediate_size, out_size].
     *     Must be one of the following types: bfloat16, float16, float32.
     *
     * @par Outputs:
     * One output, including:
     * @y: A Tensor. Shape [..., out_size].
     *     Must be one of the following types: bfloat16, float16, float32.
     *
     * @par Attributes:
     * One optional attribute, including:
     * @li cube_math_type: An optional int. MatMul compute mode, default is 1.
     *
     * @par Third-party framework compatibility:
     * New operator SwigluGatedMlp.
     */
    REG_OP(SwigluGatedMlp)
        .INPUT(x, "T")
        .INPUT(gate_up_weight, "T")
        .INPUT(down_weight, "T")
        .OUTPUT(y, "T")
        .DATATYPE(T, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT}))
        .ATTR(cube_math_type, Int, 1)
        .OP_END_FACTORY_REG(SwigluGatedMlp)
}  // namespace ge

#endif  // OPS_BUILT_IN_OP_PROTO_INC_SWIGLU_GATED_MLP_PROTO_H_
