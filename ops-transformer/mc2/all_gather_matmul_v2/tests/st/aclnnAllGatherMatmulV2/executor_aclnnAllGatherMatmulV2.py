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
import torch_npu

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat


@register("function_allgather_matmul_v2")
class AllGatherMatmulV2(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AllGatherMatmulV2, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = self.dist_task_info.rank
        world_size = self.dist_task_info.world_size
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        x1_scale = input_data.kwargs["x1Scale"]
        x2_scale = input_data.kwargs["x2Scale"]
        output_dtype = input_data.kwargs["outputDtype"]
        use_int64 = input_data.kwargs["useInt64"]
        is_format_nz = input_data.kwargs["isFormatNz"]
        comm_mode = input_data.kwargs["commMode"]
        use_x1_scale = input_data.kwargs["useX1Scale"]

        if x1.dtype != torch.int8:
            x1_scale = None
            x2_scale = None
            output_dtype = None
            use_int64 = None
        if not use_x1_scale:
            x1_scale = None

        if x1.dtype != torch.int8:
            dequantType = 0
        elif use_x1_scale:
            dequantType = 2
        else:
            dequantType = 1

        if x1.shape == torch.Size([]):
            if self.name == "cpu" or self.dist_task_info.is_bm:
                return torch.tensor([]), torch.tensor([])

        if self.name == "cpu":
            if x1.dtype == torch.int8:
                x1 = x1.cpu().to(torch.int32)
                x2 = x2.cpu().to(torch.int32)
            else:
                x1 = x1.cpu().to(torch.float32)
                x2 = x2.cpu().to(torch.float32)
            all_gather_out = [torch.zeros_like(x1) for _ in range(world_size)]
            dist.all_gather(all_gather_out, x1)
            all_gather_out = torch.cat(all_gather_out)
            output = torch.matmul(all_gather_out, x2)
            if dequantType != 0:
                x2_scale = x2_scale.cpu()
                if dequantType == 1:
                    output = (output * x2_scale).to(torch.float32)
                elif dequantType == 2:
                    x1_scale = x1_scale.cpu()
                    x1_all_gather_out = [
                        torch.zeros_like(x1_scale) for _ in range(world_size)
                    ]
                    dist.all_gather(x1_all_gather_out, x1_scale)
                    output = (output * x2_scale * torch.cat(x1_all_gather_out)).to(
                        torch.float32
                    )
            all_gather_out.zero_()
            return output, all_gather_out

        if self.dist_task_info.is_bm:
            allgather_shape = [x1.shape[0] * world_size, x1.shape[1]]
            allgather_out = torch.zeros(allgather_shape, dtype=x1.dtype).npu()
            dist._all_gather_base(allgather_out, x1)
            if dequantType:
                if use_int64:
                    output_dtype = torch.float16
                    x2_scale = torch_npu.npu_trans_quant_param(x2_scale, round_mode=1)

            if dequantType == 0:
                output = torch.matmul(allgather_out, x2)
            elif dequantType == 2:
                x1_scale_all_gather_out = torch.zeros(
                    [x1_scale.shape[0] * world_size, x1_scale.shape[1]],
                    dtype=x1_scale.dtype,
                ).npu()
                dist._all_gather_base(x1_scale_all_gather_out, x1_scale)
                output = torch_npu.npu_quant_matmul(
                    x1=allgather_out,
                    x2=x2,
                    scale=x2_scale.squeeze(0),
                    output_dtype=output_dtype,
                )
                output = (output.to(torch.float32) * x1_scale_all_gather_out).to(
                    output_dtype
                )
            else:
                output = torch_npu.npu_quant_matmul(
                    x1=allgather_out,
                    x2=x2,
                    scale=x2_scale.squeeze(0),
                    output_dtype=output_dtype,
                )
            allgather_out.zero_()
            return output, allgather_out

    def init_by_input_data(self, input_data: InputDataset):
        OpsDataset.seed_everything()
        rank_id = self.dist_task_info.rank
        device = "npu:" + str(rank_id)

        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        if x2.shape[0] != x1.shape[1] and x2.shape[1] == x1.shape[1]:
            input_data.kwargs["x2"] = input_data.kwargs["x2"].transpose(0, 1)

        if self.device == "pyaclnn" and dist.is_available():
            from torch.distributed.distributed_c10d import _get_default_group

            default_pg = _get_default_group()
            if torch.__version__ > "2.0.1":
                hcomm_info = default_pg._get_backend(
                    torch.device("npu")
                ).get_hccl_comm_name(rank_id)
            else:
                hcomm_info = default_pg.get_hccl_comm_name(rank_id)
            input_data.kwargs["group"] = hcomm_info

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


@register("aclnn_allgather_matmul_v2")
class AclnnAllGatherMatmulV2(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(AclnnAllGatherMatmulV2, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info
        self.is_format_nz = False

    def init_by_input_data(self, input_data: InputDataset):
        if input_data.kwargs["isFormatNz"]:
            input_data.kwargs["x2"] = torch_npu.npu_format_cast(
                input_data.kwargs["x2"], 29
            )
            self.is_format_nz = True

        use_int64 = input_data.kwargs["useInt64"]
        if use_int64:
            input_data.kwargs["x2Scale"] = torch_npu.npu_trans_quant_param(
                input_data.kwargs["x2Scale"], round_mode=1
            )

        use_x1_scale = input_data.kwargs["useX1Scale"]

        rank_id = self.dist_task_info.rank

        input_data.kwargs.pop("outputDtype")
        input_data.kwargs.pop("isFormatNz")
        input_data.kwargs.pop("useX1Scale")
        input_data.kwargs.pop("useInt64")

        input_args, output_packages = super().init_by_input_data(input_data)

        input_args[2] = self.get_null_tensor_pte()
        input_args[5] = self.get_null_tensor_pte()

        if input_data.kwargs["x1"].dtype != torch.int8:
            input_args[3] = self.get_null_tensor_pte()
            input_args[4] = self.get_null_tensor_pte()

        if not use_x1_scale:
            input_args[3] = self.get_null_tensor_pte()

        amax_out = self.get_null_tensor_pte()

        input_args.append(amax_out)
        torch.npu.synchronize()
        return input_args, output_packages

    def after_call(self, output_packages):
        output, allgather_out = super().after_call(output_packages)
        allgather_out.zero_()
        return output, allgather_out

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "x2" and self.is_format_nz:
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def get_null_tensor_pte(self):
        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)
        return null_tensor_ptr
