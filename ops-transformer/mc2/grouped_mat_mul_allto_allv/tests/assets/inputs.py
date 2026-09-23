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
    "kernel": {"grouped_mat_mul_allto_allv": "grouped_mat_mul_allto_allv_inputs"}
}

from typing import List


def grouped_mat_mul_allto_allv_inputs(
    gmm_x,
    gmm_weight,
    send_counts_tensor=None,
    recv_counts_tensor=None,
    mm_x=None,
    mm_weight=None,
    group: str = "",
    ep_world_size: int = 1,
    send_counts: List[int] = None,
    recv_counts: List[int] = None,
    trans_gmm_weight: bool = False,
    trans_mm_weight: bool = False,
    **kwargs,
):
    """
    Parameter validation and adjustment for GroupedMatMulAlltoAllv operator

    Parameters must match _def.cpp exactly:
    - Inputs: gmm_x, gmm_weight, send_counts_tensor, recv_counts_tensor, mm_x, mm_weight
    - Attrs: group, ep_world_size, send_counts, recv_counts, trans_gmm_weight, trans_mm_weight
    """
    if send_counts is None:
        send_counts = []
    if recv_counts is None:
        recv_counts = []

    if gmm_x is None:
        raise Exception("gmm_x is required but got None")
    if gmm_weight is None:
        raise Exception("gmm_weight is required but got None")

    if send_counts_tensor is not None and recv_counts_tensor is not None:
        if len(send_counts_tensor.shape) != 1:
            raise Exception(
                f"send_counts_tensor should be 1D tensor, got shape {send_counts_tensor.shape}"
            )
        if len(recv_counts_tensor.shape) != 1:
            raise Exception(
                f"recv_counts_tensor should be 1D tensor, got shape {recv_counts_tensor.shape}"
            )

    if mm_x is not None and mm_weight is None:
        raise Exception("mm_weight is required when mm_x is provided")
    if mm_weight is not None and mm_x is None:
        raise Exception("mm_x is required when mm_weight is provided")

    if len(send_counts) != len(recv_counts):
        raise Exception(
            f"send_counts length ({len(send_counts)}) must match recv_counts length ({len(recv_counts)})"
        )

    if ep_world_size <= 0:
        raise Exception(f"ep_world_size must be positive, got {ep_world_size}")

    total_send = sum(send_counts) if send_counts else 0
    total_recv = sum(recv_counts) if recv_counts else 0

    if total_send > gmm_x.shape[0]:
        raise Exception(
            f"Total send_counts ({total_send}) exceeds gmm_x first dimension ({gmm_x.shape[0]})"
        )

    if trans_gmm_weight:
        if len(gmm_weight.shape) != 3:
            raise Exception(
                f"gmm_weight should be 3D for trans_gmm_weight=True, got shape {gmm_weight.shape}"
            )

    if trans_mm_weight and mm_weight is not None:
        if len(mm_weight.shape) != 2:
            raise Exception(
                f"mm_weight should be 2D for trans_mm_weight=True, got shape {mm_weight.shape}"
            )

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
    )
