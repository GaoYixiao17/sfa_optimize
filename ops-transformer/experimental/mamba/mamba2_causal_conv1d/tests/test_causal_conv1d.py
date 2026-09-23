# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import os
import numpy as np
np.set_printoptions(linewidth=200)
np.set_printoptions(precision=4, suppress=True)
from rich.text import Text
from rich.style import Style
from rich.console import Console
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))
import torch, torch_npu
import torch.nn.functional as F
import time

from utils.utils import check_diff, profiling
# npu kernel
import npu_ops_transformer_ext

def causal_conv1d_fwd(x, weight, bias, B, D, S, W):
    x = x.to(torch.float32)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)

    x = F.conv1d(
        x,
        weight=weight,
        bias=bias,
        stride=1,
        padding=W-1,
        dilation=1,
        groups=D,
    )
    
    x = x[..., :S] 
    x = F.silu(x)
    return x
    
def mamba_causal_conv1d_npu(
    x,
    weight,
    bias
):
    npu_out = torch.ops.npu_ops_transformer_ext.mambav2_causal_conv1d(
                    x,
                    weight, 
                    bias,
                )
    
    return npu_out

if __name__ == '__main__':
    B = 1
    D = 10240
    S = 1024
    W = 4

    device = torch.device("npu:6")
       
    print('torch compute')
    tensor_xmtx = torch.randn([B, D, S], dtype=torch.float16, device=device) * 0.2
    tensor_wmtx = torch.randn([D, 1, W], dtype=torch.float16, device=device) * 0.2
    tensor_bias = torch.randn([D], dtype=torch.float16, device=device) * 0.2

    outmtx = causal_conv1d_fwd(tensor_xmtx, tensor_wmtx, tensor_bias, B, D, S, W)
    npu_out = mamba_causal_conv1d_npu(tensor_xmtx, tensor_wmtx, tensor_bias)
    check_diff(outmtx, npu_out)

    profiling(causal_conv1d_fwd,
                [tensor_xmtx, tensor_wmtx, tensor_bias, B, D, S, W],
                'TORCH')
    profiling(mamba_causal_conv1d_npu,
                [tensor_xmtx, tensor_wmtx, tensor_bias],
                'NPU_KERNEL')
    



