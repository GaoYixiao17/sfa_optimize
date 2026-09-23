/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_d_proto.h
 * \brief 图模式原型定义
 */
#ifndef TRANSPOSE_D_STUB_PROTO_H
#define TRANSPOSE_D_STUB_PROTO_H

#include <graph/operator_reg.h>

namespace ge {
/**
* @brief Permutes the dimensions according to perm.
         The returned tensor's dimension i will correspond to the input dimension perm[i]. \n

* @par Inputs:
* x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32,
uint64,
* the maximum dimension should not exceed 8 dimensions, and the shape should be consistent with output. \n

* @par Attributes:
* perm: A permutation of the dimensions of "x", the value
* should be within the range of [number of dimensions for self, number of dimensions for self -1]. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x".
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Transpose instead.
*/
REG_OP(TransposeD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16,
                          DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16,
                           DT_FLOAT}))
    .REQUIRED_ATTR(perm, ListInt)
    .OP_END_FACTORY_REG(TransposeD)
} // namespace ge

#endif // TRANSPOSE_D_STUB_PROTO_H
