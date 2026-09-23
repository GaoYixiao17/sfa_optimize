# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import npu_ops_transformer_ext
import logging

logging.basicConfig(level=logging.INFO)
batch_size = 2
sequence_len = 10
head_num = 2
hidden_dim = 8
epsilon = 1e-5
supported_dtypes = {torch.bfloat16, torch.float16}

for dtype in supported_dtypes:
    logging.info(f"start testing dtype : {'BF16' if dtype == torch.bfloat16 else 'HALF'}")
    input = torch.randn(batch_size, sequence_len, head_num, hidden_dim).to(dtype)
    sin = torch.arange(1, sequence_len + 1).unsqueeze(1).expand(sequence_len, hidden_dim).to(dtype)
    cos = torch.arange(0, sequence_len).unsqueeze(1).expand(sequence_len, hidden_dim).to(dtype)
    output = torch.empty_like(input)

    sin_expanded = sin.unsqueeze(0).unsqueeze(2)[..., ::2]
    cos_expanded = cos.unsqueeze(0).unsqueeze(2)[..., ::2]
    input1, input2 = torch.chunk(input, chunks = 2, dim = -1)
    rotated_input1 = input1 * cos_expanded - input2 * sin_expanded
    rotated_input2 = input2 * cos_expanded + input1 * sin_expanded
    output_cpu = torch.cat([rotated_input1, rotated_input2], dim = -1)

    input_npu = input.npu()
    sin_npu = sin.npu()
    cos_npu = cos.npu()
    output_npu = output.npu()
    torch.ops.npu_ops_transformer_ext.rotary_stride(40, input_npu, sin_npu, cos_npu, output_npu, hidden_dim)

    abs_error = torch.abs(output_npu.cpu() - output_cpu)
    rel_error = abs_error / (torch.abs(output_npu.cpu()) + epsilon)

    logging.info(f"input tensor: \n{input}")
    logging.info(f"cpu result tensor: \n{output_cpu}")
    logging.info(f"npu result tensor: \n{output_npu}")
    logging.info(f"max absolute error: {abs_error.max().item():.8f}")
    logging.info(f"average absolute error: {abs_error.mean().item():.8f}")
    logging.info(f"max relative error: {rel_error.max().item():.8f}")
    logging.info(f"average relative error: {rel_error.mean().item():.8f}")

