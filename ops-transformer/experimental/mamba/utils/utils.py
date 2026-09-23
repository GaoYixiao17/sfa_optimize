# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------


import logging
import torch
import torch_npu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_diff(x, y):
    diff = torch.abs(x - y)
    rel_diff = diff.max() / torch.abs(x.max())
    logger.info(
        "Max diff: %.04f | Relative max diff: %.05f",
        diff.max(),
        rel_diff,
    )


def profiling(model, inputs, mode):
    repeat = 10
    warmup = 5
    
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU
        ],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(f'./{mode}_profile_results')
    ) as prof:
        
        for _ in range(warmup):
            outputs = model(*inputs)

        t1 = torch.npu.Event(enable_timing=True)
        t2 = torch.npu.Event(enable_timing=True)
        t1.record()
        for _ in range(repeat):
            outputs = model(*inputs)
        t2.record()
        elapsed = t1.elapsed_time(t2)

        logger.info(
            ">>>> %s IMPL TIME ELAPSED: %.1f us",
            mode,
            elapsed / repeat * 1000,
        )

        prof.step()
