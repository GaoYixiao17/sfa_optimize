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
__input__ = {"kernel": {"all_gather_matmul": "all_gather_matmul_inputs"}}


def all_gather_matmul_inputs(
    x1,
    x2,
    bias=None,
    is_trans_a=False,
    is_trans_b=False,
    gather_index: int = 0,
    comm_turn: int = 0,
    rank_size: int = 1,
    is_gather_out: bool = True,
    **kwargs,
):
    """
    参数接收、校验和调整

    参数名称必须与 all_gather_matmul_def.cpp 完全一致

    Args:
        x1: 输入张量 1 (REQUIRED) - 来自 Input("x1")
        x2: 输入张量 2 (REQUIRED) - 来自 Input("x2")
        bias: 偏置 (OPTIONAL) - 来自 Input("bias")
        is_trans_a: 是否转置 x1 - 来自 Attr("is_trans_a").Bool(false)
        is_trans_b: 是否转置 x2 - 来自 Attr("is_trans_b").Bool(false)
        gather_index: Gather 索引 - 来自 Attr("gather_index").Int(0)
        comm_turn: 通信轮次 - 来自 Attr("comm_turn").Int(0)
        rank_size: Rank 数量 - 来自 Attr("rank_size").Int(0)
        is_gather_out: 是否输出 gather 结果 - 来自 Attr("is_gather_out").Bool(true)
        **kwargs: 扩展参数
    """
    # 参数校验
    # Shape 校验：确保 x1 和 x2 的维度匹配
    if is_trans_b:
        # 转置后 x2 的 shape: [K, N] -> [N, K]
        expected_k = x2.shape[1] if len(x2.shape) > 1 else x2.shape[0]
        actual_k = x1.shape[1] if len(x1.shape) > 1 else x1.shape[0]
        if expected_k != actual_k:
            raise ValueError(
                f"x1 and x2 shape mismatch after transpose: "
                f"x1 K={actual_k}, x2 transposed K={expected_k}"
            )
    else:
        # 正常情况：x2 的 shape [K, N]
        expected_k = x2.shape[0]
        actual_k = x1.shape[1] if len(x1.shape) > 1 else x1.shape[0]
        if expected_k != actual_k:
            raise ValueError(
                f"x1 and x2 shape mismatch: x1 K={actual_k}, x2 K={expected_k}"
            )

    # Bias 校验
    if bias is not None:
        output_dim = x2.shape[1] if not is_trans_b else x2.shape[0]
        if bias.shape[-1] != output_dim:
            raise ValueError(
                f"bias shape mismatch: expected {output_dim}, got {bias.shape[-1]}"
            )

    # rank_size 校验
    if rank_size <= 0:
        raise ValueError(f"rank_size must be positive, got {rank_size}")

    # 返回处理后的参数（保持参数顺序与定义一致）
    return (
        x1,
        x2,
        bias,
        is_trans_a,
        is_trans_b,
        gather_index,
        comm_turn,
        rank_size,
        is_gather_out,
    )
