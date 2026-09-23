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
__input__ = {"kernel": {"all_gather_matmul_v2": "all_gather_matmul_v2_inputs"}}


def all_gather_matmul_v2_inputs(
    x1,
    x2,
    bias=None,
    x1_scale=None,
    x2_scale=None,
    quant_scale=None,
    group="",
    is_trans_a=False,
    is_trans_b=False,
    gather_index: int = 0,
    comm_turn: int = 0,
    rank_size: int = 0,
    block_size: int = 0,
    group_size: int = 0,
    is_gather_out: bool = True,
    is_amax_out: bool = False,
    y_dtype: int = -1,
    comm_mode: str = "ai_cpu",
    **kwargs,
):
    """
    AllGatherMatmulV2 inputs validation and adjustment

    Args:
        x1: Input tensor 1
        x2: Input tensor 2
        bias: Bias tensor (optional)
        x1_scale: Scale for x1 (optional)
        x2_scale: Scale for x2 (optional)
        quant_scale: Quantization scale (optional)
        group: Communication group name
        is_trans_a: Whether to transpose x1
        is_trans_b: Whether to transpose x2
        gather_index: Gather index
        comm_turn: Communication turn
        rank_size: Number of ranks
        block_size: Block size for quantization
        group_size: Group size for quantization
        is_gather_out: Whether to output gather result
        is_amax_out: Whether to output amax
        y_dtype: Output data type
        comm_mode: Communication mode
        **kwargs: Additional arguments

    Returns:
        Processed inputs
    """
    if is_trans_b:
        x2 = x2.transpose()

    return (
        x1,
        x2,
        bias,
        x1_scale,
        x2_scale,
        quant_scale,
        group,
        is_trans_a,
        is_trans_b,
        gather_index,
        comm_turn,
        rank_size,
        block_size,
        group_size,
        is_gather_out,
        is_amax_out,
        y_dtype,
        comm_mode,
    )
