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
from pathlib import Path

import pytest

import quant_block_sparse_attn_golden
import check_valid_param
import result_compare_method
import utils
from batch import quant_block_sparse_attn_process
from quant_block_sparse_attn_paramset import ENABLED_PARAMS


PT_SAVE_PATH = "bsa_testcase"
DEVICE_ID = 0
SAVE_PT = False
RESULT_PATH = Path("result.xlsx")


def _build_param_combinations(enabled_params):
    combinations = []
    for params in enabled_params:
        param_names = list(params.keys())
        param_values = [value if isinstance(value, (list, tuple)) else [value] for value in params.values()]
        for combo in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, combo)))
    return combinations


PARAM_COMBINATIONS = _build_param_combinations(ENABLED_PARAMS)


def _fill_case_name(params, case_id):
    if params.get("Testcase_Name") is not None:
        return params
    mode = "prefill" if params["layout_q"] in ("TND", "NTD") else "decode"
    filled = dict(params)
    filled["Testcase_Name"] = (
        f"quantBlockSparseAttn_{mode}_{filled['layout_q']}_{filled['layout_kv']}_"
        f"{filled['B']}_{filled['N1']}_{filled['N2']}_{filled['S1']}_{filled['S2']}_{filled['D']}_{case_id:06d}"
    )
    return filled


def bsa(param_combinations, case_id):
    params = _fill_case_name(param_combinations, case_id)
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as error:
        pytest.skip(f"input param check failed: {error}")

    test_data = quant_block_sparse_attn_golden.generate_and_save_testdata(
        params, save_pt=SAVE_PT, save_path=PT_SAVE_PATH)

    try:
        npu_result, cpu_result = quant_block_sparse_attn_process.test_quant_block_sparse_attn_process_ci(
            test_data, device_id=DEVICE_ID)
        print("PRECISION COMPARE START")
        result, fulfill_percent = result_compare_method.check_result(cpu_result, npu_result)
        print(f"PRECISION COMPARE RESULT: result={result}, fulfill_percent={fulfill_percent:.6f}%")
    except pytest.skip.Exception:
        raise
    except Exception as error:
        print("NPU ERROR:", error)
        print("PRECISION COMPARE SKIPPED: NPU execution failed before result_compare_method.check_result.")
        result = "NPU ERROR"
        fulfill_percent = 0

    utils.save_result(test_data["params"], result, fulfill_percent, RESULT_PATH)

    if result != "Pass":
        pytest.fail(f"case failed: {test_data['Testcase_Name']}, result={result}, fulfill_percent={fulfill_percent}")


@pytest.mark.ci
@pytest.mark.parametrize("case_id,param_combinations", list(enumerate(PARAM_COMBINATIONS)))
def test_quant_block_sparse_attn(case_id, param_combinations):
    bsa(param_combinations, case_id)
