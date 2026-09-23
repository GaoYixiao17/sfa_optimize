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
 * \file masked_causal_conv1d_proto.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_PROTO_INC_MASKED_CAUSAL_CONV1D_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MASKED_CAUSAL_CONV1D_H_

#include "graph/operator_reg.h"

namespace ge {

/**
* @brief Applies masked causal 1D convolution on input sequences. \n

* @par Inputs:
* @li x: Input sequence tensor of shape [S, B, H]. Supports float16, bfloat16.
* @li weight: Convolution kernel of shape [W, H], W fixed to 3. Same dtype as x.
* @li mask: Optional. Boolean mask tensor of shape [B, S]. Positions marked as False in the mask
*     will be zeroed out in the output. Tensor data type: bool.

* @par Outputs:
* @li y: Output sequence tensor. Same shape and dtype as x.
*/
REG_OP(MaskedCausalConv1d)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(mask, TensorType({DT_BOOL}))
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(MaskedCausalConv1d)

} // namespace ge

#endif // OPS_BUILT_IN_OP_PROTO_INC_MASKED_CAUSAL_CONV1D_H_
