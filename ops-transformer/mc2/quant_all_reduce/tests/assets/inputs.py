#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__input__ = {"kernel": {"quant_all_reduce": "quant_all_reduce_inputs"}}

import numpy as np


def quant_all_reduce_inputs(
    x,
    scales,
    group: str = "",
    reduce_op: str = "sum",
    output_dtype: int = 1,
    world_size: int = 1,
    **kwargs,
):
    """
    Parameter validation and adjustment for QuantAllReduce

    Args:
        x: Input tensor (quantized)
        scales: Scale tensor for dequantization
        group: HCCL group name (communication domain)
        reduce_op: Reduce operation type ("sum", "max", "min")
        output_dtype: Output data type enum
        world_size: Number of ranks in the communication group
        **kwargs: Additional parameters

    Returns:
        Tuple of validated parameters
    """
    is_mxfp = kwargs.get("is_mxfp", False)

    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(scales, np.ndarray):
        scales = np.array(scales)

    scale_shape = kwargs.get("scale_shape", None)
    if scale_shape is not None:
        if len(scale_shape) == 3:
            scales = scales.reshape(scale_shape[0], -1)
        elif len(scale_shape) == 4:
            scales = scales.reshape(scale_shape[0], scale_shape[1], -1)

    valid_reduce_ops = ["sum", "max", "min"]
    if reduce_op not in valid_reduce_ops:
        reduce_op = "sum"

    if world_size < 1:
        world_size = 1

    return x, scales, group, reduce_op, output_dtype, world_size
