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
import os

import pytest
import torch

import quant_block_sparse_attn_golden
import check_valid_param
from quant_block_sparse_attn_paramset_batch import ENABLED_PARAMS


SAVE_PATH = "bsa_testcase"


def _parse_value(value):
    if value in ("torch.bfloat16", "bfloat16", "bf16"):
        return torch.bfloat16
    if value in ("FALSE", "False", "false"):
        return False
    if value in ("TRUE", "True", "true"):
        return True
    return value


def _build_param_combinations(enabled_params):
    combinations = []
    for params in enabled_params:
        normalized = {key: [_parse_value(item) for item in value] for key, value in params.items()}
        param_names = list(normalized.keys())
        param_values = list(normalized.values())
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


def bsa(param_combinations, case_id, force_regenerate=False):
    params = _fill_case_name(param_combinations, case_id)
    try:
        check_valid_param.check_valid_param(params)
    except ValueError as error:
        pytest.skip(f"input param check failed: {error}")

    case_name = params.get("Testcase_Name") or f"case_{case_id:06d}"
    pt_filepath = os.path.join(SAVE_PATH, f"bsa_case_{case_name}.pt")
    if not force_regenerate and os.path.isfile(pt_filepath):
        pytest.skip(f"test case already exists, skip generation: {pt_filepath}")

    quant_block_sparse_attn_golden.generate_and_save_testdata(params, save_pt=True, save_path=SAVE_PATH)


@pytest.mark.ci
@pytest.mark.parametrize("case_id,param_combinations", list(enumerate(PARAM_COMBINATIONS)))
def test_quant_block_sparse_attn(case_id, param_combinations, force_regenerate):
    bsa(param_combinations, case_id, force_regenerate=force_regenerate)
