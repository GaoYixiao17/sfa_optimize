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
import glob
import torch
import torch_npu
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension


def get_base_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_torch_npu_path():
    return os.path.dirname(os.path.abspath(torch_npu.__file__))


def collect_sources(base_dir):
    pattern = os.path.join(base_dir, "npu_linalg", "csrc", "*.cpp")
    return glob.glob(pattern, recursive=True)


def make_extension():
    base_dir = get_base_dir()
    torch_npu_path = get_torch_npu_path()

    include_dir = os.path.join(
        torch_npu_path,
        "include",
        "third_party",
        "acl",
        "inc",
    )

    return NpuExtension(
        name="npu_linalg.npu_linalg_lib",
        sources=collect_sources(base_dir),
        extra_compile_args=[f"-I{include_dir}"],
    )


USE_NINJA = os.getenv("USE_NINJA", "0") == "1"

setup(
    name="npu_linalg",
    version="1.0",
    keywords="npu_linalg",
    packages=find_packages(),
    ext_modules=[make_extension()],
    package_data={
        "npu_linalg": ["*.py", "*.so"],
    },
    cmdclass={
        "build_ext": BuildExtension.with_options(use_ninja=USE_NINJA),
    },
)
