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
import ascend_ops
import logging

#基础配置
logging.basicConfig(level=logging.INFO)
batch_size = 2
sequence_len = 2
head_num = 2
hidden_dim = 64
epsilon = 1e-12
blockDim=40
clamp=True
clamp_value=1
simplified=0
supported_dtypes = {torch.bfloat16, torch.float16} #匹配NPU算子支持的类型dtype



#CPU侧参考实现
def moe_layernorm_add_add_cpu(input,bias,residual,gamma,beta,eps=1e-12,clamp=True,clamp_val=1):
    #step1:计算outAdd
    outAdd=input+bias+residual
   
    #step2：对outAdd做LayerNorm
    mean=outAdd.mean(dim=-1,keepdim=True)
    var=outAdd.var(dim=-1,keepdim=True,unbiased=False)
    ln_out=(outAdd-mean)/torch.sqrt(var+eps)
    ln_out=ln_out*gamma+beta
    
    #Step3：对layerNorm做clamp
    if clamp:
      outLyn=torch.clamp(ln_out,min=-clamp_val,max=clamp_val)
    else:
      outLyn=ln_out
    return outAdd,outLyn

#遍历测试不同的dtype
for dtype in supported_dtypes:
    logging.info(f"start testing dtype : {'BF16' if dtype == torch.bfloat16 else 'HALF'}")
    
    #1.构造测试数据
    input = torch.randn(batch_size, sequence_len, hidden_dim).to(dtype)
    bias=torch.randn(hidden_dim).to(dtype) #bias
    residual=torch.randn_like(input).to(dtype) #residual
    gamma=torch.randn(hidden_dim).to(dtype)#layernorm weight
    beta=torch.randn(hidden_dim).to(dtype)#layernorm bias

    
    #2.CPU侧计算结果
    outAdd_cpu,outLyn_cpu=moe_layernorm_add_add_cpu(input,bias,residual,gamma,beta,eps=epsilon,clamp=clamp,clamp_val=clamp_value)
  

    #3.NPU侧计算结果
      #3.1数据迁移到NPU
    input_npu=input.npu()
    bias_npu=bias.npu()
    residual_npu=residual.npu()
    gamma_npu=gamma.npu()
    beta_npu=beta.npu()
    outAdd_npu=torch.empty_like(input_npu).npu()
    outLyn_npu=torch.empty_like(input_npu).npu()
      #3.2 调用NPU算子
    dtype_code=1 if dtype==torch.float16 else 27
    torch.ops.ascend_ops.moe_layernorm_add_add(blockDim,batch_size*sequence_len,hidden_dim,epsilon,input_npu,bias_npu,residual_npu,gamma_npu,beta_npu,outLyn_npu,outAdd_npu,clamp,clamp_value,simplified,dtype_code)
    
    torch.npu.synchronize()

    #4.结果对比(分别对比outAdd和outLyn，并且被除数不能为0)

    outAdd_abs_error=torch.abs(outAdd_npu.cpu()-outAdd_cpu)
    outAdd_rel_error=outAdd_abs_error/(torch.abs(outAdd_cpu)+epsilon)
    
    outLyn_abs_error=torch.abs(outLyn_npu.cpu()-outLyn_cpu)
    outLyn_rel_error=outLyn_abs_error/(torch.abs(outLyn_cpu)+ epsilon)

   
    logging.info(f"------------------------------------------outAdd的结果对比-------------------------------------------------------------------------------")
    #outADD
    logging.info(f"input tensor: \n{input}")
    logging.info(f"outAdd cpu result tensor: \n{outAdd_cpu}")
    logging.info(f"outAdd npu result tensor: \n{outAdd_npu.cpu()}")
    logging.info(f"outAdd max absolute error: {outAdd_abs_error.max().item():.8f}")
    logging.info(f"outAdd average absolute error: {outAdd_abs_error.mean().item():.8f}")
    logging.info(f"outAdd max relative error: {outAdd_rel_error.max().item():.8f}")
    logging.info(f"outAdd average relative error: {outAdd_rel_error.mean().item():.8f}")
    

    logging.info(f"------------------------------------------outLyn的结果对比-------------------------------------------------------------------------------")
    #LynAdd
    logging.info(f"outLyn的结果对比")
    logging.info(f"input tensor: \n{input}")
    logging.info(f"outLyn cpu result tensor: \n{outLyn_cpu}")
    logging.info(f"outLyn npu result tensor: \n{outLyn_npu.cpu()}")
    logging.info(f"outLyn max absolute error: {outLyn_abs_error.max().item():.8f}")
    logging.info(f"outLyn average absolute error: {outLyn_abs_error.mean().item():.8f}")
    logging.info(f"outLyn max relative error: {outLyn_rel_error.max().item():.8f}")
    logging.info(f"outLyn average relative error: {outLyn_rel_error.mean().item():.8f}")
    
