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
import torch
import torch_npu
import time
from rich.text import Text
from rich.style import Style
from rich.console import Console
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))

# npu kernel
import npu_ops_transformer_ext
from utils.utils import check_diff, profiling

def chunk_state_passing_fwd(dacs, initial_states, states, cmtx, P=64, out_dtype=None):
    batch, nchunks, nheads, Z = states.shape
    batch, nchunks, chunk_size, ngroups, N = cmtx.shape
    assert Z == N*P
    
    cmtx = torch.repeat_interleave(cmtx.permute(0,1,3,2,4), nheads//ngroups, dim=2)

    out_dtype = states.dtype if out_dtype is None else out_dtype
    out = torch.zeros((batch, nchunks, nheads, Z), device=states.device, dtype=out_dtype)
    final_state = torch.zeros((batch, nheads, Z), device=states.device, dtype=torch.float32)

    tmp_states = torch.zeros((batch, nheads, Z), device=states.device, dtype=out_dtype)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, Z)
        tmp_states = initial_states

    out[:, 0, :, :] = tmp_states

    for c in range(nchunks):
        new_states = states[:, c, :, :]
        scale = torch.exp(dacs[:, c, :])
        tmp_states = scale[...,None] * tmp_states + new_states
        if c < nchunks - 1:
            out[:, c+1, :, :] = tmp_states
        else:
            final_state = tmp_states

    out = out.reshape(batch, nchunks, nheads, N, P).to(cmtx.dtype)
    final_state = final_state.reshape(batch, nheads, N, P)
    
    out_a = torch.matmul(cmtx.to(torch.float32), out.to(torch.float32))  # B, 4, 80, 256, 64  

    return out, final_state, out_a


if __name__ == '__main__':
    B = 1
    H = 128
    S = 1024
    C = 4
    L = 256
    G = 8
    N = 128
    P = 64
    Z = (N * P)
    
    device = torch.device("npu:6")
    print("torch compute")

    tensor_dacs = torch.randn([B, C, H], dtype=torch.float32, device=device) * 0.2
    tensor_initmtx = torch.randn([B, H, Z], dtype=torch.float32, device=device) * 0.2
    tensor_statemtx = torch.randn([B, C, H, Z], dtype=torch.float32, device=device) * 0.2
    tensor_cmtx = torch.randn([B, C, L, G, N], dtype=torch.float16, device=device) * 0.2

    out_states, final_state, out_bmm = chunk_state_passing_fwd(tensor_dacs, tensor_initmtx, tensor_statemtx, tensor_cmtx)
    npu_out, npu_final_state = torch.ops.npu_ops_transformer_ext.mambav2_chunk_state_passing(tensor_dacs, tensor_initmtx, tensor_statemtx, tensor_cmtx)

    check_diff(final_state.cpu(), npu_final_state.cpu())
    check_diff(out_bmm.cpu(), npu_out.cpu())

    inputs = [tensor_dacs, tensor_initmtx, tensor_statemtx, tensor_cmtx]
    profiling(chunk_state_passing_fwd,
              inputs,
              'TORCH')
    profiling(torch.ops.npu_ops_transformer_ext.mambav2_chunk_state_passing,
              inputs,
              'NPU_KERNEL')
    
