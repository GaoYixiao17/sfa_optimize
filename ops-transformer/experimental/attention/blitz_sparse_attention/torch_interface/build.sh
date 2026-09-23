#!/bin/bash
# -*- coding: utf-8 -*-
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# This script builds the operator and installs a python torch extension package 'torch_bsa'

# Check for exactly one argument
if [ $# -ne 1 ]; then
    echo "Error: Exactly one argument required" >&2
    echo "Usage: $0 <PFA_LIB_value>" >&2
    echo "  where PFA_LIB_value is either 'custom' or 'ops_transformer'" >&2
    exit 1
fi

# build torch extension
rm -rf build torch_bsa.egg-info
pip uninstall -y torch_bsa

# set PFA_LIB option to be "custom" if you compiled ops-transformer kernel using "./build.sh --make_clean -j96 --pkg --soc=ascend910b --ops=blitz_sparse_attention"
# set PFA_LIB option to be "ops_transformer" if you compiled ops-transformer kernel using "./build.sh --make_clean -j96 --pkg --soc=ascend910b"
PFA_LIB=$1 pip install . --no-build-isolation