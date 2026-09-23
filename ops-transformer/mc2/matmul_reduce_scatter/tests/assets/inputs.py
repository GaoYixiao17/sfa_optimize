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
__input__ = {"kernel": {"matmul_reduce_scatter": "matmul_reduce_scatter_inputs"}}


def matmul_reduce_scatter_inputs(
    x1,
    x2,
    bias=None,
    group: str = "",
    reduce_op: str = "sum",
    is_trans_a: bool = False,
    is_trans_b: bool = False,
    comm_turn: int = 0,
    rank_size: int = 0,
    **kwargs,
):
    """
    参数接收、校验和调整

    参数名称与 matmul_reduce_scatter_def.cpp 完全一致
    """
    if is_trans_b:
        x2 = x2.transpose()

    if is_trans_a:
        x1 = x1.transpose()

    x1_m = x1.shape[0]
    if rank_size > 0 and x1_m % rank_size != 0:
        raise Exception(
            f"x1.shape[0] ({x1_m}) must be divisible by rank_size ({rank_size})"
        )

    return x1, x2, bias, group, reduce_op, is_trans_a, is_trans_b, comm_turn, rank_size
