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
__input__ = {
    "kernel": {
        "allto_all_all_gather_batch_mat_mul": "allto_all_all_gather_batch_mat_mul_inputs"
    }
}


def allto_all_all_gather_batch_mat_mul_inputs(
    x,
    weight,
    bias=None,
    group_ep="",
    group_tp="",
    ep_world_size: int = 1,
    tp_world_size: int = 1,
    x_shard_type: int = 0,
    act_type: int = 0,
    transpose_weight: bool = False,
    output_y2_flag: bool = False,
    output_y3_flag: bool = False,
    **kwargs,
):
    """
    AllToAllAllGatherBatchMatMul inputs validation and adjustment

    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Bias tensor (optional)
        group_ep: Communication group for EP
        group_tp: Communication group for TP
        ep_world_size: World size for EP
        tp_world_size: World size for TP
        x_shard_type: Shard type (0 or 1)
        act_type: Activation type
        transpose_weight: Whether to transpose weight
        output_y2_flag: Whether to output y2
        output_y3_flag: Whether to output y3
        **kwargs: Additional arguments

    Returns:
        Processed inputs
    """
    if transpose_weight:
        weight = (
            weight.transpose(0, 2, 1) if len(weight.shape) == 3 else weight.transpose()
        )

    return (
        x,
        weight,
        bias,
        group_ep,
        group_tp,
        ep_world_size,
        tp_world_size,
        x_shard_type,
        act_type,
        transpose_weight,
        output_y2_flag,
        output_y3_flag,
    )
