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
__input__ = {"kernel": {"matmul_reduce_scatter_v2": "matmul_reduce_scatter_v2_inputs"}}


def matmul_reduce_scatter_v2_inputs(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    quant_scale=None,
    group="",
    reduce_op="sum",
    is_trans_a=False,
    is_trans_b=False,
    comm_turn=0,
    rank_size=0,
    block_size=0,
    group_size=0,
    is_amax_out=False,
    y_dtype=0,
    comm_mode="ai_cpu",
    **kwargs,
):
    if is_trans_b:
        x2 = x2.transpose()
        if x2_scale is not None:
            per_block_flag = kwargs.get("per_block_flag", False)
            if per_block_flag:
                x2_scale = x2_scale.transpose()

    if rank_size <= 0:
        rank_size = kwargs.get("world_size", 1)

    if x1.shape[0] % rank_size != 0:
        raise ValueError(
            f"x1.shape[0] ({x1.shape[0]}) must be divisible by rank_size ({rank_size})"
        )

    return (
        x1,
        x2,
        bias,
        x1_scale,
        x2_scale,
        quant_scale,
        group,
        reduce_op,
        is_trans_a,
        is_trans_b,
        comm_turn,
        rank_size,
        block_size,
        group_size,
        is_amax_out,
        y_dtype,
        comm_mode,
    )
