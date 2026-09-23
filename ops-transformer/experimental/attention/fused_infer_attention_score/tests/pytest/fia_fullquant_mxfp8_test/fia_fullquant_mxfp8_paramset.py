#!/usr/bin/python3
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

TEST_PARAMS = {
    "PA_NZ_QS4_KVS31_Nq1_Nkv1_D128_SP3_Decode": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[4]],
        "actual_seq_kv": [[31]],
        "enable_pa": [True],
        "kv_cache_layout": ["PA_NZ"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["N2TGD"],
        "p_scale": [1.0],
    },
    "PA_NZ_QS48_KVS48_Nq1_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[48]],
        "actual_seq_kv": [[48]],
        "enable_pa": [True],
        "kv_cache_layout": ["PA_NZ"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["AUTO"],
        "p_scale": [1.0],
    },
    "PA_NZ_QS64_KVS64_Nq1_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[64]],
        "actual_seq_kv": [[64]],
        "enable_pa": [True],
        "kv_cache_layout": ["PA_NZ"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["AUTO"],
        "p_scale": [1.0],
    },
    "PA_NZ_QS66_KVS66_Nq1_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[66]],
        "actual_seq_kv": [[66]],
        "enable_pa": [True],
        "kv_cache_layout": ["PA_NZ"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["AUTO"],
        "p_scale": [1.0],
    },
    "PA_NZ_QS86_KVS86_Nq1_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[86]],
        "actual_seq_kv": [[86]],
        "enable_pa": [True],
        "kv_cache_layout": ["PA_NZ"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["AUTO"],
        "p_scale": [1.0],
    },
    "PA_BNBD_QS48_KVS64_Nq8_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [8],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[48]],
        "actual_seq_kv": [[64]],
        "enable_pa": [False],
        "kv_cache_layout": ["BnNBsD"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["AUTO"],
        "p_scale": [1.0],
    },
    "PA_BNBD_QS66_KVS86_Nq8_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [8],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[66]],
        "actual_seq_kv": [[86]],
        "enable_pa": [False],
        "kv_cache_layout": ["BnNBsD"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["AUTO"],
        "p_scale": [1.0],
    },
    "PA_NZ_QS1023_KVS1024_Nq80_Nkv8_D128_SP5_Prefill": {
        "B": [1],
        "N_q": [80],
        "N_kv": [8],
        "D": [128],
        "actual_seq_q": [[1023]],
        "actual_seq_kv": [[1024]],
        "enable_pa": [True],
        "kv_cache_layout": ["PA_NZ"],
        "block_size": [512],
        "sparse_mode": [5],
        "q_scale_layout": ["TND"],
        "p_scale": [1.0],
    },
    "PA_BNBD_QS128_KVS256_Nq1_Nkv1_D128_SP3": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[128]],
        "actual_seq_kv": [[256]],
        "enable_pa": [True],
        "kv_cache_layout": ["BnNBsD"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["AUTO"],
        "p_scale": [1.0],
    },
}

ENABLED_PARAMS = list(TEST_PARAMS.values())


def _expand_paramset_to_cases():
    import itertools

    PARAM_NAMES = [
        "B", "N_q", "N_kv", "D",
        "actual_seq_q", "actual_seq_kv",
        "enable_pa", "kv_cache_layout", "block_size",
        "sparse_mode", "q_scale_layout", "p_scale",
    ]

    cases = []
    for name, params in TEST_PARAMS.items():
        param_values = [params[n] for n in PARAM_NAMES]
        for combo in itertools.product(*param_values):
            case = {"name": name}
            for key, val in zip(PARAM_NAMES, combo):
                case[key] = val
            cases.append(case)
    return cases


CASES = _expand_paramset_to_cases()
