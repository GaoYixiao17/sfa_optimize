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
import npu_ops_transformer_ext

logging.basicConfig(level=logging.INFO)
logging.info("verifying fused operator BiasSigmoid")

ROWS = 1024
COLS = 254
BLOCK_DIM = 40
supported_dtypes = {torch.bfloat16, torch.float16}

for dtype in supported_dtypes:
    logging.info(f"start testing dtype : {'BF16' if dtype == torch.bfloat16 else 'HALF'}")
    expertX = torch.randint(1, 10, (ROWS, COLS)).to(dtype)
    expertBias = torch.randint(1, 10, (COLS,)).to(dtype)

    expertX_fp32 = expertX.float()
    expertBias_fp32 = expertBias.float()
    sigmoid_out_cpu = torch.sigmoid(expertX_fp32)
    bias_out_cpu = sigmoid_out_cpu + expertBias_fp32.unsqueeze(0)

    sigmoid_out_cpu = sigmoid_out_cpu.to(dtype)
    bias_out_cpu = bias_out_cpu.to(dtype)

    sigmoid_out_npu = torch.empty_like(expertX).npu()
    bias_out_npu = torch.empty_like(expertX).npu()
    torch.ops.npu_ops_transformer_ext.bias_sigmoid(BLOCK_DIM, expertX.npu(), expertBias.npu(), sigmoid_out_npu, bias_out_npu)

    rotl, atol = torch.testing._comparison.get_tolerances(dtype, rtol=None, atol=None)
    logging.info(f"compare CPU Result vs NPU Sigmoid Result: {torch.allclose(sigmoid_out_cpu, sigmoid_out_npu.cpu(), rotl, atol)}")
    logging.info(f"compare CPU Result vs NPU Bias Result: {torch.allclose(bias_out_cpu, bias_out_npu.cpu(), rotl, atol)}")