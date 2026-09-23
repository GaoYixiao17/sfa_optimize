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

import itertools
import torch
from quant_sparse_flash_mla_paramset import ENABLED_PARAMS
import result_compare_method
import check_valid_param
import quant_sparse_flash_mla_golden
from batch import quant_sparse_flash_mla_process
import pytest
from pathlib import Path
import multiprocessing as mp
import utils
import logging

pt_save_path = "qsmla_testcase"
device_id = 0
save_pt = False
result_path = Path('result.xlsx')

param_combinations = []
for params in ENABLED_PARAMS:
    param_values = {
        "Testcase_Name": params.get("Testcase_Name", [None]),
        "layout_q": params.get("layout_q"),
        "layout_kv": params.get("layout_kv"),
        "q_type": params.get("q_type"),
        "ori_kv_type": params.get("ori_kv_type"),
        "cmp_kv_type": params.get("cmp_kv_type", [None]),
        "B": params.get("B"),
        "S1": params.get("S1"),
        "S2": params.get("S2", [None]),
        "N1": params.get("N1"),
        "N2": params.get("N2"),
        "D": params.get("D"),
        "K": params.get("K", [None]),
        "block_num1": params.get("block_num1", [None]),
        "block_num2": params.get("block_num2", [None]),
        "block_size1": params.get("block_size1"),
        "block_size2": params.get("block_size2", [None]),
        "seqused_q": params.get("seqused_q", [None]),
        "cu_seqlens_q": params.get("cu_seqlens_q", [None]),
        "seqused_ori_kv": params.get("seqused_ori_kv", [None]),
        "seqused_cmp_kv": params.get("seqused_cmp_kv", [None]),
        "cu_seqlens_ori_kv": params.get("cu_seqlens_ori_kv", [None]),
        "cu_seqlens_cmp_kv": params.get("cu_seqlens_cmp_kv", [None]),
        "cmp_residual_kv": params.get("cmp_residual_kv", [None]),
        "softmax_scale": params.get("softmax_scale"),
        "cmp_ratio": params.get("cmp_ratio", [None]),
        "ori_mask_mode": params.get("ori_mask_mode"),
        "cmp_mask_mode": params.get("cmp_mask_mode", [None]),
        "ori_win_left": params.get("ori_win_left"),
        "ori_win_right": params.get("ori_win_right"),
        "qkv_quant_mode": params.get("qkv_quant_mode"),
        "template_run_mode": params.get("template_run_mode"),
        "actlen_mode": params.get("actlen_mode"),
        "S1EQS2": params.get("S1EQS2", [False]),
        "isSink": params.get("isSink", [True]),
    }

    param_names = list(param_values.keys())
    values_lists = [param_values[name] for name in param_names]

    for combo in itertools.product(*values_lists):
        combination = dict(zip(param_names, combo))
        param_combinations.append(combination)

case_id = 0
def qsmla(param_combinations):
    global case_id

    # 填充None参数的默认值
    params = utils.fill_none_params(param_combinations)

    # 生成测试用例名称
    Testcase_Name = params['Testcase_Name']
    if Testcase_Name is None:
        ops_mode = 'prefill' if params['S1'] > 4 else "decode"
        q_type_str = "BF16" if params['q_type'] == torch.bfloat16 else "FP16"
        kv_type_str = "HIF8" if params['ori_kv_type'] == torch.uint8 else "FP8_E4M3FN"
        Testcase_Name = f"quantSparseFlashMla_{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_{params['B']}_{params['N1']}_{params['N2']}_{params['S1']}_{params['S2']}_{params['D']}_{params['K']}_{case_id:06d}"
        params['Testcase_Name'] = Testcase_Name

    # 输入参数的合法性校验
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as e:
       pytest.skip(f"输入参数校验失败:{e}")

    # 生成测试数据及golden
    test_data = quant_sparse_flash_mla_golden.generate_and_save_testdata(params, save_pt=save_pt, save_path=pt_save_path)

    # 获得cpu结果(真值)和算子结果（测试值）
    try:
        npu_result, cpu_quant_result = quant_sparse_flash_mla_process.test_qsmla_quant_process_ci(
            test_data, device_id=device_id)
        result, fulfill_percent = result_compare_method.check_result(cpu_quant_result, npu_result)
    except Exception as e:
        logging.exception(e)
        result = "NPU ERROR"
        fulfill_percent = 0

    case_id += 1

    utils.save_result(test_data['params'], result, fulfill_percent, result_path)

    if result in ("NPU ERROR", "Failed"):
        pytest.fail(f"用例执行失败:{test_data['Testcase_Name']} 精度:{fulfill_percent:.2f}%")

def _gen_testcase_id(params, idx):
    name = params.get('Testcase_Name')
    if name is not None:
        return name
    ops_mode = 'prefill' if params['S1'] > 4 else "decode"
    q_type_str = "HIF8" if params['q_type'] == torch.bfloat16 else "FP16"
    kv_type_str = "HIF8" if params['ori_kv_type'] == torch.uint8 else "FP8_E4M3FN"
    return f"{params['template_run_mode']}_{ops_mode}_{params['layout_q']}_{q_type_str}_{params['layout_kv']}_{kv_type_str}_B{params['B']}_S1{params['S1']}_S2{params['S2']}_D{params['D']}_K{params['K']}_{idx:06d}"

testcase_ids = [_gen_testcase_id(p, i) for i, p in enumerate(param_combinations)]

@pytest.mark.ci
@pytest.mark.parametrize("param_combinations", param_combinations, ids=testcase_ids)
def test_quant_sparse_flash_mla(param_combinations):
    qsmla(param_combinations)
