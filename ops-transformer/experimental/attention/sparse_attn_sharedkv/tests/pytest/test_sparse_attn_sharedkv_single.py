#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import result_compare_method
import utils
import itertools
import torch
import torch_npu
from sparse_attn_sharedkv_paramset import ENABLED_PARAMS
import sparse_attn_sharedkv_golden
import os
import pytest
import time
import random
import math
from batch import sparse_attn_sharedkv_process

save_pt = False
pt_save_path = "data"

def fill_random_cu_len(T, SMax, B, qk_equal_len=False):
    cu_seqlens = [0]
    S = min(SMax, T // B)
    S_eq = S
    for i in range(B):
        length = S_eq if qk_equal_len else random.randint(0, S)
        cu_seqlens.append(cu_seqlens[-1] + length)
    cu_seqlens[-1] = T
    return cu_seqlens

def fill_random_used_len(SMax, B, qk_equal_len=False):
    used_lens = []
    S_eq = SMax
    for i in range(B):
        length = S_eq if qk_equal_len else random.randint(0, SMax)
        used_lens.append(length)
    return used_lens

# 处理所有参数组合
param_combinations = []
result_path = os.getenv("SAS_RESULT_SAVE_PATH", './result/sas_result.xlsx')

for params in ENABLED_PARAMS:
    # 确保所有参数都存在，缺失的用默认值填充
    param_values = {
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type", [None]),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2", [None]),
        "T1": params.get("T1", [None]),
        "T2": params.get("T2", [None]),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K", [None]),
        "block_num1": params.get("block_num1"),
        "block_num2": params.get("block_num2", [None]),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2", [None]),
        "seqused_q": params.get("seqused_q", [None]),
        "cu_seqlens_q": params.get("cu_seqlens_q", [None]),
        "seqused_kv": params.get("seqused_kv", [None]),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio", [None]),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode", [None]),
        "ori_win_left": params.get("ori_win_left"),
        "ori_win_right": params.get("ori_win_right"),
        "cu_seqlens_ori_kv": params.get("cu_seqlens_ori_kv", [None]),
        "cu_seqlens_cmp_kv": params.get("cu_seqlens_cmp_kv", [None]),
    }

    # 生成参数名和值列表
    param_names = list(param_values.keys())
    values_lists = [param_values[name] for name in param_names]

    # 生成所有组合
    for combo in itertools.product(*values_lists):
        combination = dict(zip(param_names, combo))
        param_combinations.append(combination)

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations)
def test_example(param_combinations):
    layout_q = param_combinations['layout_q']
    layout_kv = param_combinations['layout_kv']
    q_type = param_combinations['q_type']
    ori_kv_type = param_combinations['ori_kv_type']
    cmp_kv_type = param_combinations['cmp_kv_type']
    B = param_combinations['B']
    S1 = param_combinations['S1']
    S2 = param_combinations['S2']
    T1 = param_combinations['T1']
    T2 = param_combinations['T2']
    N1 = param_combinations['N1']
    N2 = param_combinations['N2']
    D = param_combinations['D']
    K = param_combinations['K']
    block_num1 = param_combinations['block_num1']
    block_num2 = param_combinations['block_num2']
    block_size1 = param_combinations['block_size1']
    block_size2 = param_combinations['block_size2']
    cu_seqlens_q = param_combinations['cu_seqlens_q']
    seqused_q = param_combinations['seqused_q']
    seqused_kv = param_combinations['seqused_kv']
    softmax_scale = param_combinations['softmax_scale']
    cmp_ratio = param_combinations['cmp_ratio']
    ori_mask_mode = param_combinations['ori_mask_mode']
    cmp_mask_mode = param_combinations['cmp_mask_mode']
    ori_win_left = param_combinations['ori_win_left']
    ori_win_right = param_combinations['ori_win_right']
    cu_seqlens_ori_kv = param_combinations['cu_seqlens_ori_kv']
    cu_seqlens_cmp_kv = param_combinations['cu_seqlens_cmp_kv']
    q_datarange = [-10, 10]
    ori_kv_datarange = [-10, 10]
    cmp_kv_datarange = [-10, 10]
    testcase_name = "case_" + str(int(time.time() * 1000000))

    torch_npu.npu.set_device(0)

    if cu_seqlens_q is None and layout_q == "TND":
        cu_seqlens_q = fill_random_cu_len(T1, S1, B, True)
        print("cu_seqlens_q auto set to: ", cu_seqlens_q)

    if layout_kv == "TND" and T2 is None:
        T2 = B * S2
    if cu_seqlens_ori_kv is None and layout_kv == "TND":
        cu_seqlens_ori_kv = fill_random_cu_len(T2, S2, B, True)
        print("cu_seqlens_ori_kv auto set to: ", cu_seqlens_ori_kv)

    if cu_seqlens_cmp_kv is None and layout_kv == "TND" and cmp_ratio is not None:
        cu_seqlens_cmp_kv = [x // cmp_ratio for x in cu_seqlens_ori_kv]
        print("cu_seqlens_cmp_kv auto set to: ", cu_seqlens_cmp_kv)

    if seqused_kv is None and layout_kv != "TND":
        seqused_kv = fill_random_used_len(S2, B, True)
        print("seqused_kv auto set to: ", seqused_kv)

    ori_block_num_sum = 0
    cmp_block_num_sum = 0
    if layout_kv == "PA_ND":
        for cur_ori_act_kv in seqused_kv:
            cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
            ori_block_num_sum += cur_ori_kv_block_num
            if cmp_ratio is not None and block_size2 is not None:
                cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
                cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
                cmp_block_num_sum += cur_cmp_kv_block_num
    block_num1 = ori_block_num_sum
    block_num2 = cmp_block_num_sum
    # 增加参数请在最后增加，保证结果统计
    test_data = layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, \
                block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, \
                ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, testcase_name, S2, q_datarange, \
                ori_kv_datarange, cmp_kv_datarange, seqused_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, T2
    print("test_data:", test_data)
    # 获得cpu结果(真值)和算子结果（测试值）
    input_data = sparse_attn_sharedkv_golden.gen_data(test_data)
    if save_pt:
        sparse_attn_sharedkv_golden.save_test_case(input_data, pt_save_path)
    npu_result, softmax_lse = sparse_attn_sharedkv_process.call_npu(input_data)
    print("npu_result.size():", npu_result.size())

    # 结果精度对比
    result, fulfill_percent = result_compare_method.check_result(input_data['cpu_output'], npu_result)
    # 记录结果
    utils.save_result(result, fulfill_percent, test_data, result_path)