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
    "kernel": {"allto_allv_grouped_mat_mul": "allto_allv_grouped_mat_mul_inputs"}
}


def allto_allv_grouped_mat_mul_inputs(
    gmm_x,
    gmm_weight,
    send_counts_tensor=None,
    recv_counts_tensor=None,
    mm_x=None,
    mm_weight=None,
    group="",
    ep_world_size: int = 1,
    send_counts=None,
    recv_counts=None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    permute_out_flag: bool = False,
    **kwargs,
):
    """
    AlltoAllvGroupedMatMul inputs validation and adjustment

    Args:
        gmm_x: Input tensor for grouped matmul
        gmm_weight: Weight tensor for grouped matmul
        send_counts_tensor: Send counts tensor (optional)
        recv_counts_tensor: Receive counts tensor (optional)
        mm_x: Input tensor for matmul (optional)
        mm_weight: Weight tensor for matmul (optional)
        group: Communication group name
        ep_world_size: World size for EP
        send_counts: Send counts list
        recv_counts: Receive counts list
        trans_gmm_weight: Whether to transpose gmm_weight
        trans_mm_weight: Whether to transpose mm_weight
        permute_out_flag: Whether to output permute result
        **kwargs: Additional arguments

    Returns:
        Processed inputs
    """
    if trans_gmm_weight:
        gmm_weight = (
            gmm_weight.transpose(0, 2, 1)
            if len(gmm_weight.shape) == 3
            else gmm_weight.transpose()
        )

    if trans_mm_weight and mm_weight is not None:
        mm_weight = mm_weight.transpose()

    return (
        gmm_x,
        gmm_weight,
        send_counts_tensor,
        recv_counts_tensor,
        mm_x,
        mm_weight,
        group,
        ep_world_size,
        send_counts,
        recv_counts,
        trans_gmm_weight,
        trans_mm_weight,
        permute_out_flag,
    )
