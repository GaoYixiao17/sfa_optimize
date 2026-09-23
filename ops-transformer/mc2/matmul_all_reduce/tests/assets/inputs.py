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
__input__ = {"kernel": {"matmul_all_reduce": "matmul_all_reduce_inputs"}}


def matmul_all_reduce_inputs(
    x1,
    x2,
    bias=None,
    x3=None,
    antiquant_scale=None,
    antiquant_offset=None,
    dequant_scale=None,
    pertoken_scale=None,
    comm_quant_scale_1=None,
    comm_quant_scale_2=None,
    group=None,
    reduce_op: str = "sum",
    is_trans_a: bool = False,
    is_trans_b: bool = False,
    comm_turn: int = 0,
    antiquant_group_size: int = 0,
    group_size: int = 0,
    y_dtype: int = 0,
    comm_quant_mode: int = 0,
    **kwargs,
):
    if is_trans_b and "graph" not in kwargs.get("scene", ""):
        x2 = x2.t()
    return (
        x1,
        x2,
        bias,
        x3,
        antiquant_scale,
        antiquant_offset,
        dequant_scale,
        pertoken_scale,
        comm_quant_scale_1,
        comm_quant_scale_2,
        group,
        reduce_op,
        is_trans_a,
        is_trans_b,
        comm_turn,
        antiquant_group_size,
        group_size,
        y_dtype,
        comm_quant_mode,
    )
