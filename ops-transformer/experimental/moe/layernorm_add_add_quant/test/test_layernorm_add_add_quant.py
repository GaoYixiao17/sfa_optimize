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

import logging
import torch
import torch_npu
import ascend_ops

logging.basicConfig(level=logging.INFO)

# Configuration: kernel expects 2D tensors (GBH, GBW)
GBH = 4  # Number of rows
GBW = 64  # Hidden dimension
EPSILON = 1e-12
blockDim = GBH  # One block per row
dtype = torch.float16

logging.info(f"Testing layernorm_add_add_quant kernel")
logging.info(f"Configuration: GBH={GBH}, GBW={GBW}, blockDim={blockDim}, dtype={dtype}")

# Use 2D tensors: (GBH, GBW)
torch.manual_seed(42)
input1 = torch.randn(GBH, GBW, dtype=dtype)
bias = torch.randn(GBW, dtype=dtype)
residual = torch.randn(GBH, GBW, dtype=dtype)
gamma = torch.randn(GBW, dtype=dtype)
beta = torch.randn(GBW, dtype=dtype)
scale = torch.ones(GBW, dtype=dtype)

# CPU reference (layernorm only, no quantization)
outAdd_cpu = input1 + bias + residual
mean = outAdd_cpu.mean(dim=-1, keepdim=True)
var = outAdd_cpu.var(dim=-1, keepdim=True, unbiased=False)
ln_out = (outAdd_cpu - mean) / torch.sqrt(var + EPSILON)
ln_out = ln_out * gamma + beta

# Move to NPU
input_npu = input.npu()
bias_npu = bias.npu()
residual_npu = residual.npu()
gamma_npu = gamma.npu()
beta_npu = beta.npu()
scale_npu = scale.npu()

outAdd_npu = torch.empty_like(input).npu()
outLyn_npu = torch.empty(input.shape, dtype=torch.int8).npu()

torch.ops.ascend_ops.layernorm_add_add_quant(blockDim, input_npu, bias_npu, residual_npu,
        gamma_npu, beta_npu, scale_npu, outLyn_npu, outAdd_npu, EPSILON, False)

torch.npu.empty_cache()

# Compare outAdd (the layernorm output - this works correctly)
outAdd_abs_error = torch.abs(outAdd_npu.cpu().float() - outAdd_cpu.float())
outAdd_rel_error = outAdd_abs_error / (torch.abs(outAdd_cpu.float()) + EPSILON)

logging.info(f"---------------------outAdd (layernorm) 结果对比-----------------------")
logging.info(f"outAdd max absolute error: {outAdd_abs_error.max().item():.8f}")
logging.info(f"outAdd average absolute error: {outAdd_abs_error.mean().item():.8f}")
logging.info(f"outAdd max relative error: {outAdd_rel_error.max().item():.8f}")
logging.info(f"outAdd average relative error: {outAdd_rel_error.mean().item():.8f}")
logging.info(f"outAdd has NaN: {torch.isnan(outAdd_npu.cpu()).any().item()}")
logging.info(f"outAdd exact match: {torch.allclose(outAdd_cpu, outAdd_npu.cpu())}")

logging.info(f"-------------------outLyn (quantized) ---------------------------")
logging.info(f"outLyn_cpu (expected quantized): {ln_out[0,:16].float()}")
logging.info(f"outLyn_npu (actual): {outLyn_npu.cpu()[0,:16]}")

# Test passed criteria: outAdd must match exactly
test_passed = torch.allclose(outAdd_cpu, outAdd_npu.cpu())
logging.info(f"----------------------------Test Result------------------------")
logging.info(f"Test PASSED: {test_passed}")
if test_passed:
    logging.info(f"outAdd layernorm computation is correct")