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
import torch_npu
import npu_linalg
import pytest

from utils.utils import get_input_matrix, get_omega, create_low_rank_matrix
import numpy as np

DEVICE_ID = 6
torch_npu.npu.set_device(int(DEVICE_ID))


JACOBI_SVD_TEST_CASES = [
    ((1, 16, 16), 1e-4, 7),
    ((1, 160, 256), 1e-4, 10),
    ((7, 168, 1024), 1e-4, 5),
    ((1, 16, 16), 1e-5, 8),
    ((2, 1, 3, 168, 512), 1e-4, 10),
    ((48, 160, 512), 1e-4, 10),
]

QR_TEST_CASES = [
#    ((32, 16), 1e-5),
    ((128, 16), 1e-5),
    ((1, 128, 16), 1e-5),
    ((48, 128, 16), 1e-5),
    ((2, 1, 3, 128, 16), 1e-5),
#    ((4, 4*1024, 168), 1e-4),
    ((48, 64*1024, 160), 1e-2),
]

RSVD_TEST_CASES = [
    ((48, 64*1024, 512), 160, 8, 1e-2),
    ((48, 128*1024, 512), 160, 8, 1e-2),
]

@pytest.mark.parametrize("test_shapes,atol,num_iter", JACOBI_SVD_TEST_CASES)
def test_jacobi_svd(test_shapes, atol, num_iter):
    np.random.seed(0)
    A = torch.randn(test_shapes, dtype=torch.float32, device="cpu")
    npu_a = A.to("npu:%s" % DEVICE_ID)

    npu_u, npu_s,  npu_v = torch.ops.npu_linalg.svd(npu_a, num_iter)
    u, s, v = npu_u.cpu(), npu_s.cpu(), npu_v.cpu()
    A = A.reshape(-1, A.shape[-2], A.shape[-1])
    u = torch.transpose(u, -2, -1).reshape(-1, A.shape[-2], A.shape[-2])
    s = torch.unsqueeze(s.reshape(-1, A.shape[-2]), 1)
    v = v.reshape(-1, A.shape[-2], A.shape[-1])
    res = torch.matmul((u * s), v)
    assert torch.allclose(A, res, atol=atol)

@pytest.mark.parametrize("test_shapes,atol", QR_TEST_CASES)
def test_qr_decomposition(test_shapes, atol):
    np.random.seed(0)
    A = torch.randn(test_shapes, dtype=torch.float32, device="cpu")
    npu_a = A.to("npu:%s" % DEVICE_ID)

    npu_q, npu_r = torch.ops.npu_linalg.tsqr(npu_a)
    q, r = npu_q.cpu(), npu_r.cpu()
    A = A.reshape(-1, A.shape[-2], A.shape[-1])
    q = q.reshape(-1, A.shape[-2], A.shape[-1])
    r = r.reshape(-1, A.shape[-1], A.shape[-1])
    assert torch.allclose(A, torch.matmul(q, r), atol=atol)

@pytest.mark.parametrize("test_shapes,rank,gen_mtx_rank,atol", RSVD_TEST_CASES)
def test_svd_lowrank(test_shapes, rank, gen_mtx_rank, atol):
    b, m, n = test_shapes
    A = create_low_rank_matrix(b, m, n, gen_mtx_rank)
    npu_a = A.to("npu:%s" % DEVICE_ID)

    npu_u, npu_s, npu_v = torch.ops.npu_linalg.svd_lowrank(npu_a, q=rank)
    u, s, v = npu_u.cpu(), npu_s.cpu(), npu_v.cpu()
    assert torch.allclose(A, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v), atol=atol)
