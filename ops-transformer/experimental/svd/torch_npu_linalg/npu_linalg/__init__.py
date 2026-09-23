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

import os
import pkgutil
import warnings

import torch
import torch_npu

# 导入so 和 python
from . import npu_linalg_lib

"""
import custom ops as torch_npu ops to support the following usage:
'torch.ops.npu_linalg.npu_selected_flash_attention()'
'torch_npu.npu_selected_flash_attention()'
"""

# get torch.ops.npu_linalg module
custom_ops_module = getattr(torch.ops, 'npu_linalg', None)


if custom_ops_module is not None:
    for op_name in dir(custom_ops_module):
        if op_name.startswith('_'):
            # skip built-in method, such as __name__, __doc__
            continue

        # get custom ops and set to torch_npu
        custom_op_func = getattr(custom_ops_module, op_name)
        setattr(torch_npu, op_name, custom_op_func)

else:
    WARN_MSG = "torch.ops.npu_linalg module is not found, mount custom ops to torch_npu failed." \
               "Calling by torch_npu.xxx for custom ops is unsupported, please use torch.ops.npu_linalg.xxx."
    warnings.warn(WARN_MSG)
    warnings.filterwarnings("ignore", message=WARN_MSG)
