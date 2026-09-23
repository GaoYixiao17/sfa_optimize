#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import torch.distributed as dist

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor


@register("function_allgather_matmul")
class AllGatherMatmul(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AllGatherMatmul, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        input_chunk = input_data.kwargs["x1"]
        weight_chunk = input_data.kwargs["x2"]

        m_size = input_chunk.shape[0] // world_size
        if m_size != 0:
            start = self.dist_task_info.rank * m_size
            end = start + m_size
            input_chunk = input_chunk[start:end, :]

        if input_chunk.shape == []:
            if self.name == "cpu" or self.dist_task_info.is_bm:
                return torch.tensor([]), torch.tensor([])

        if self.name == "cpu":
            input_chunk = input_chunk.cpu().to(torch.float32)
            weight_chunk = weight_chunk.cpu().to(torch.float32)
            all_gather_out = [torch.zeros_like(input_chunk) for _ in range(world_size)]
            dist.all_gather(all_gather_out, input_chunk)
            all_gather_out = torch.cat(all_gather_out)
            output = torch.matmul(all_gather_out, weight_chunk)
            gather_output_cpu = all_gather_out
            return output, gather_output_cpu

        if self.dist_task_info.is_bm:
            tensor_allgather_shape = [
                input_chunk.shape[0] * world_size,
                input_chunk.shape[1],
            ]
            all_gather_out = torch.zeros(
                tensor_allgather_shape, dtype=input_chunk.dtype
            ).npu()
            dist._all_gather_base(all_gather_out, input_chunk)
            output = torch.matmul(all_gather_out, weight_chunk)
            gather_output_cpu = all_gather_out
            return output, gather_output_cpu

    def init_by_input_data(self, input_data: InputDataset):
        OpsDataset.seed_everything()
        input_chunk = input_data.kwargs["x1"]
        weight_chunk = input_data.kwargs["x2"]

        if (
            weight_chunk.shape[0] != input_chunk.shape[1]
            and weight_chunk.shape[1] == input_chunk.shape[1]
        ):
            input_data.kwargs["x2"] = input_data.kwargs["x2"].transpose(0, 1)

        if self.device == "pyaclnn" and dist.is_available():
            rank_id = self.dist_task_info.rank
            input_data.kwargs["group"] = AllGatherMatmul.get_hcomm_info(rank_id)

    @staticmethod
    def get_hcomm_info(rank_id):
        from torch.distributed.distributed_c10d import _get_default_group

        default_pg = _get_default_group()
        if torch.__version__ > "2.0.1":
            hcomm_info = default_pg._get_backend(
                torch.device("npu")
            ).get_hccl_comm_name(rank_id)
        else:
            hcomm_info = default_pg.get_hccl_comm_name(rank_id)
        return hcomm_info


@register("aclnn_allgather_matmul")
class AclnnAllGatherMatmul(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(AclnnAllGatherMatmul, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info

    def init_by_input_data(self, input_data: InputDataset):
        world_size = self.dist_task_info.world_size
        input_chunk = input_data.kwargs["x1"]
        m_size = input_chunk.shape[0] // world_size
        if m_size != 0:
            start = self.dist_task_info.rank * m_size
            end = start + m_size
            input_data.kwargs["x1"] = input_data.kwargs["x1"][start:end, :]

        input_args, output_packages = super().init_by_input_data(input_data)

        return input_args, output_packages

    def get_null_tensor_pte(self):
        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)
        return null_tensor_ptr
