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
import time

# npu kernel
import npu_ops_transformer_ext
from utils.utils import check_diff, profiling

def mamba2_chunk_state_forward(dtout, dacs, bt, xt, num_repeats):

    B, C, L, H = dacs.shape
    
    da_sub = torch.reshape(dacs[:,:,-1,:], (B, C, 1, H)) - dacs
    da = torch.exp(da_sub) * dtout  # BCLH
    
    bt_repeated_tensor = torch.repeat_interleave(bt, num_repeats, dim=3).float()
    dab = bt_repeated_tensor * torch.reshape(da, (B, C, L, H, 1))   
    dab = dab.permute(0,1,3,4,2).to(torch.float16)
    out = dab.to(torch.float32) @ xt.to(torch.float32).permute(0, 1, 3, 2, 4)
    
    return out
    

if __name__ == '__main__':
    B = 1
    C = 4
    H = 128
    G = 8
    L = 256
    N = 128
    P = 64
    BASEH=8
    CBASEM=64
    
    device = torch.device('npu:6')
       
    print('torch compute')

    tensor_dtout = torch.randn([B, C, L, H], dtype=torch.float32, device=device) * 0.2
    tensor_dacs = torch.randn([B, C, L, H], dtype=torch.float32, device=device) * 0.2
    tensor_bt = torch.randn([B, C, L, G, N], dtype=torch.float16, device=device) * 0.2
    tensor_xt = torch.randn([B, C, L, H, P], dtype=torch.float16, device=device) * 0.2
    assert H%G==0

    outmtx = mamba2_chunk_state_forward(tensor_dtout, tensor_dacs, tensor_bt, tensor_xt, H//G)
    npu_out = torch.ops.npu_ops_transformer_ext.mambav2_chunk_state(tensor_dtout, tensor_dacs, tensor_bt, tensor_xt)

    check_diff(outmtx.cpu(), npu_out.cpu())

    profiling(mamba2_chunk_state_forward,
              [tensor_dtout, tensor_dacs, tensor_bt, tensor_xt, H//G],
              'TORCH')
    profiling(torch.ops.npu_ops_transformer_ext.mambav2_chunk_state,
              [tensor_dtout, tensor_dacs, tensor_bt, tensor_xt],
              'NPU_KERNEL')

