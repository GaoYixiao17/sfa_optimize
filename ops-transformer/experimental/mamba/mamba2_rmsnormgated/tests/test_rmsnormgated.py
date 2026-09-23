# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np
import os
import torch
import torch.nn.functional as F
import torch_npu
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4, suppress=True)
from rich.text import Text
from rich.style import Style
from rich.console import Console
import time
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))

## npu kernels
import npu_ops_transformer_ext
from utils.utils import check_diff, profiling


def rms_norm_gated_ref(x, weight, bias, z=None, eps=1e-6, group_size=None):
    
    dtype = x.dtype
    N = x.shape[-1]
    x = x.float()
    weight = weight.float()
    if z is not None:
        z = z.float()
        x = x * F.silu(z)
    if group_size is None:
        std = torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        rstd = 1 / std
        out = x * rstd * weight
    else:
        B, S, D = x.shape
        x_group = x.reshape(B, S, -1, group_size)
        std = torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        rstd = 1 / std
        out = (x_group * rstd).reshape(B,S,D) * weight
    if bias is not None:
        bias = bias.float()
        out = out + bias

    return out.to(dtype)

if __name__ == '__main__':
    B = 1
    S = 1024
    D = 8192 
    G = 8
    E = 1e-05
    
    device = torch.device('npu:6')

    tensor_x = torch.randn([B, S, D], dtype=torch.float32, device=device) * 0.2
    tensor_w = torch.randn([D], dtype=torch.float32, device=device) * 0.2
    tensor_z = torch.randn([B, S, D], dtype=torch.float32, device=device) * 0.2
    assert D%G==0

    outmtx = rms_norm_gated_ref(tensor_x, tensor_w, None, tensor_z, eps=E, group_size=D//G)
    kernel_out = torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated(tensor_x, tensor_z, tensor_w, G, E)

    check_diff(outmtx.cpu(), kernel_out.cpu())

    profiling(rms_norm_gated_ref,
              [tensor_x, tensor_w, None, tensor_z, E, D//G],
              'TORCH')
    profiling(torch.ops.npu_ops_transformer_ext.mambav2_rmsnormgated,
              [tensor_x, tensor_z, tensor_w, G, E],
              'NPU_KERNEL')

    