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
    "kernel": {"grouped_mat_mul_all_reduce": "grouped_mat_mul_all_reduce_inputs"}
}

import numpy as np


def grouped_mat_mul_all_reduce_inputs(
    x,
    weight,
    bias=None,
    group_list=None,
    splitItem: int = 0,
    group: str = "",
    reduceOp: str = "sum",
    commTurn: int = 0,
    **kwargs,
):
    """
    GroupedMatMulAllReduce inputs validation and adjustment

    Args:
        x: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor
        group_list: Optional group list for grouping
        splitItem: Split item flag (default: 0)
        group: HCCL group name (required)
        reduceOp: Reduce operation type (default: "sum")
        commTurn: Communication turn parameter (default: 0)
        **kwargs: Additional arguments
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(weight, np.ndarray):
        weight = np.array(weight)

    if group_list is not None:
        if not isinstance(group_list, np.ndarray):
            group_list = np.array(group_list, dtype=np.int64)

        group_list_cumsum = np.cumsum(group_list) if splitItem == 0 else group_list

        if group_list_cumsum[-1] > x.shape[0]:
            raise Exception(
                "sum of grouplist: ({}) can not be greater than x[0]: ({})".format(
                    group_list_cumsum[-1], x.shape[0]
                )
            )

    if bias is not None:
        if not isinstance(bias, np.ndarray):
            bias = np.array(bias)

    return x, weight, bias, group_list, splitItem, group, reduceOp, commTurn
