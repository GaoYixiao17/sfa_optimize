#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

def create_low_rank_matrix(m, n, rank, dtype=torch.float32):

    A = torch.randn(m, rank, dtype=dtype)
    B = torch.randn(rank, n, dtype=dtype)
    W = torch.matmul(A, B)
    return W

def create_low_rank_matrix(b, m, n, rank, dtype=torch.float32):

    A = torch.randn(b, m, rank, dtype=dtype)
    B = torch.randn(b, rank, n, dtype=dtype)

    W = torch.matmul(A, B)
    return W

def generate_input_matrix(test_shapes, rank=None, dtype=torch.float32):
    for b, m, n in test_shapes:
        matrix = create_low_rank_matrix(b, m, n, rank, dtype) if rank else torch.randn(b, m, n, dtype=dtype)
        path = BASE / "data" / f"input_matrix_{b}_{m}_{n}_r_{rank}.npy"
        np.save(path, matrix.detach().cpu().numpy())

def get_input_matrix(m, n, r):
    path = BASE / "data" / f"input_matrix_{m}_{n}_r_{r}.npy"
    return torch.from_numpy(np.load(path))

def get_input_matrix(b, m, n, r):
    path = BASE / "data" / f"input_matrix_{b}_{m}_{n}_r_{r}.npy"
    return torch.from_numpy(np.load(path))

def get_omega(n, r):
    path = BASE / "data" / f"omega_{n}_{r}.npy"
    return torch.from_numpy(np.load(path))
