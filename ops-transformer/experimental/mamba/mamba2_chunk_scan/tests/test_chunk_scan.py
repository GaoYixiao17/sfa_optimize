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
import math
import torch.nn.functional as F
from einops import rearrange, repeat
console = Console()
print = Console().print
print(Text('VALIDATING OUTPUT...', style=Style(color='red', underline=True)))

import npu_ops_transformer_ext
from utils.utils import check_diff, profiling

def mamba2_chunkscan_custom_fwd(C, B, input_dacs, input_dttrans, input_mask, x, input_outa, input_dmtx, chunk_size=256, ngroups=8, headdim=64, z=None, seq_idx=None):
    #preprocess
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    x = rearrange(x, "b l (h p) -> b l h p", p=headdim)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = C.shape
    assert C.shape == (batch, seqlen, ngroups, dstate)

    batch, seqlen, ngroups, dstate = B.shape
    nchunks = math.ceil(seqlen / chunk_size)
    padLen = nchunks * chunk_size - seqlen 
    B = F.pad(B, (0,0,0,0,0,padLen), "constant", 0)
    B = B.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    B = torch.permute(B, (0, 1, 3, 4, 2))  

    batch, seqlen, nheads, headdim = x.shape
    x = F.pad(x, (0,0,0,0,0,padLen), "constant", 0)
    x = x.reshape(batch, nchunks, chunk_size, nheads, headdim)
    x = torch.permute(x, (0, 1, 2, 3, 4))

    C = F.pad(C, (0,0,0,0,0,padLen), "constant", 0)
    C = C.reshape(batch, nchunks, chunk_size, ngroups, dstate)
    C = C.permute(0, 1, 3, 2, 4)  
    batch, nchunks, chunk_size, nheads, headdim = x.shape

    cb = torch.matmul(C.to(torch.float32), B.to(torch.float32)).to(torch.float32) 
    out_cb = cb
    cb = torch.repeat_interleave(cb, nheads//ngroups, dim=2)
    dummy = (input_dacs.reshape([batch, nchunks, nheads, chunk_size,1]) - input_dacs.reshape([batch, nchunks, nheads,1, chunk_size]))
    
    scale_b = torch.exp(dummy)
    val = cb * scale_b
    
    val = val * input_dttrans.reshape(batch,nchunks,nheads,1,chunk_size)
    val.masked_fill_(input_mask.bool(), 0.0)
    val = val.to(torch.float16)
    out_m = val.clone()
    
    out_b = torch.matmul(val.to(torch.float32), x.permute(0,1,3,2,4).to(torch.float32))
    
    out_a = input_outa * torch.exp(input_dacs.reshape([batch, nchunks, nheads, chunk_size, 1]))
    
    out = out_a + out_b

    x_res = x.to(torch.float32) * input_dmtx[..., None].to(torch.float32)
    out += x_res.permute(0,1,3,2,4)
    out = out.transpose(2,3)
    y = out.reshape(batch, -1, nheads*headdim)

    return out_cb, out_m, out_b, y


if __name__ == '__main__':
    B = 1
    S = 1024
    C = 4
    H = 128
    L = 256
    G = 8
    N = 128
    P = 64

    device = torch.device("npu:6")

    tensor_cmtx = torch.randn([B, C*L, G*N], dtype=torch.float16, device=device) * 0.2
    tensor_bmtx = torch.randn([B, C*L, G*N], dtype=torch.float16, device=device) * 0.2
    tensor_xmtx = torch.randn([B, C*L, H*P], dtype=torch.float16, device=device) * 0.2
    tensor_dacs = torch.randn([B, C, H, L], dtype=torch.float32, device=device) * 0.2
    tensor_dtout = torch.randn([B, C, H, L], dtype=torch.float32, device=device) * 0.2
    tensor_outa = torch.randn([B, C, H, L, P], dtype=torch.float32, device=device) * 0.2
    tensor_dmtx = torch.randn([H], dtype=torch.float16, device=device) * 0.2
    mask = torch.triu(torch.ones(L, L), diagonal=1).to(device)
    
    out_cb, out_m, out_b, out_y = mamba2_chunkscan_custom_fwd(tensor_cmtx, tensor_bmtx, tensor_dacs, tensor_dtout, mask, tensor_xmtx, tensor_outa, tensor_dmtx, chunk_size=L)

    npu_y = torch.ops.npu_ops_transformer_ext.mambav2_chunk_scan(
                tensor_cmtx.reshape(B,C,L,G,N),
                tensor_bmtx,
                tensor_xmtx,
                tensor_dmtx,
                tensor_outa,
                tensor_dacs,
                tensor_dtout  
            )
    check_diff(out_y.cpu(), npu_y.cpu())

    profiling(mamba2_chunkscan_custom_fwd,
              [tensor_cmtx, tensor_bmtx, tensor_dacs, tensor_dtout, mask, tensor_xmtx, tensor_outa, tensor_dmtx, L],
              'TORCH')
    profiling(torch.ops.npu_ops_transformer_ext.mambav2_chunk_scan,
              [tensor_cmtx.reshape(B,C,L,G,N),
                tensor_bmtx,
                tensor_xmtx,
                tensor_dmtx,
                tensor_outa,
                tensor_dacs,
                tensor_dtout],
              'NPU_KERNEL')
