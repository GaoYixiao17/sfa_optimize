#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import json
import logging as log
import os
import stat
import sys


VALID_PARAM_TYPES = {"dynamic", "optional", "required"}
VALID_OPS_FLAGS = {"OPS_FLAG_OPEN", "OPS_FLAG_CLOSE"}
VALID_SUB_TYPES = {"1", "2", "3", "4"}
IO_PREFIXES = (
    "input",
    "dynamic_input",
    "optional_input",
    "output",
    "dynamic_output",
    "optional_output",
)


def parse_ini_files(ini_files):
    ops_info = {}
    for ini_file in ini_files:
        if not os.path.exists(ini_file):
            log.warning(f"ini file {ini_file} not exists!")
            continue
        parse_ini_to_obj(ini_file, ops_info)
    return ops_info


def parse_ini_to_obj(ini_file, ops_info):
    with open(ini_file, encoding="utf-8") as ini_file_obj:
        lines = ini_file_obj.readlines()
    op = {}
    op_name = ""
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("["):
            op_name = line[1:-1]
            op = {}
            ops_info[op_name] = op
            continue
        key1 = line[: line.index("=")].strip()
        key2 = line[line.index("=") + 1 :].strip()
        key1_0, key1_1 = key1.split(".", 1)
        if key1_0 not in op:
            op[key1_0] = {}
        if key1_1 in op[key1_0]:
            raise RuntimeError(f"Op:{op_name} {key1_0} {key1_1} is repeated!")
        op[key1_0][key1_1] = key2


def is_aicpu_op(op):
    return op.get("opInfo", {}).get("engine") == "DNN_VM_AICPU"


def is_io_key(op_info_key):
    return any(op_info_key.startswith(prefix) for prefix in IO_PREFIXES)


def check_param_type(op_name, op_info_key, op_io_info):
    if "paramType" in op_io_info and op_io_info["paramType"] not in VALID_PARAM_TYPES:
        log.error(
            f"op: {op_name} {op_info_key} paramType not valid, "
            "valid key:[dynamic, optional, required]"
        )
        return False
    return True


def check_io_info(op_name, op_info_key, op_io_info):
    missing_keys = []
    if "name" not in op_io_info:
        missing_keys.append("name")
    if missing_keys:
        log.error(f"op: {op_name} {op_info_key} missing: {','.join(missing_keys)}")
        return False
    return check_param_type(op_name, op_info_key, op_io_info)


def check_op_info_extend_fields(op_name, op_info):
    is_valid = True
    workspace_size = op_info.get("workspaceSize")
    if workspace_size is not None:
        if not workspace_size.isdigit() or not 100 <= int(workspace_size) <= 500:
            log.error(f"op: {op_name} workspaceSize out of range [100, 500]")
            is_valid = False
    ops_flag = op_info.get("opsFlag")
    if ops_flag is not None and ops_flag not in VALID_OPS_FLAGS:
        log.error(
            f"op: {op_name} opsFlag not valid, valid key:[OPS_FLAG_OPEN, OPS_FLAG_CLOSE]"
        )
        is_valid = False
    sub_type = op_info.get("subTypeOfInferShape")
    if sub_type is not None and sub_type not in VALID_SUB_TYPES:
        log.error(
            f"op: {op_name} subTypeOfInferShape not valid, valid key:[1, 2, 3, 4]"
        )
        is_valid = False
    kernel_so = op_info.get("kernelSo")
    if kernel_so is not None and not kernel_so.endswith(".so"):
        log.error(f"op: {op_name} kernelSo not valid, should end with .so")
        is_valid = False
    return is_valid


def check_op_info(ops_info):
    log.info("==============check valid for ops info start==============")
    is_valid = True
    for op_name, op in ops_info.items():
        if not is_aicpu_op(op):
            continue
        for op_info_key, op_info_value in op.items():
            if is_io_key(op_info_key):
                is_valid = (
                    check_io_info(op_name, op_info_key, op_info_value) and is_valid
                )
            if op_info_key == "opInfo":
                is_valid = (
                    check_op_info_extend_fields(op_name, op_info_value) and is_valid
                )
    log.info("==============check valid for ops info end================")
    return is_valid


def write_json_file(ops_info, json_file_path):
    json_file_real_path = os.path.realpath(json_file_path)
    with open(json_file_real_path, "w", encoding="utf-8") as file_handle:
        os.chmod(
            json_file_real_path,
            stat.S_IWGRP + stat.S_IWUSR + stat.S_IRGRP + stat.S_IRUSR,
        )
        json.dump(
            ops_info, file_handle, sort_keys=True, indent=4, separators=(",", ":")
        )
    log.info("Compile op info cfg successfully.")


def parse_ini_to_json(ini_file_paths, outfile_path):
    ops_info = parse_ini_files(ini_file_paths)
    if not check_op_info(ops_info):
        log.error("Compile op info cfg failed.")
        return False
    write_json_file(ops_info, outfile_path)
    return True


if __name__ == "__main__":
    log.basicConfig(
        stream=sys.stdout,
        format=f"{os.path.basename(__file__)}: %(levelname)s: %(message)s",
        level=log.INFO,
    )
    args = sys.argv
    output_file_path = "cust_aicpu_kernel.json"
    ini_file_path_list = []

    for arg in args:
        if arg.endswith("ini"):
            ini_file_path_list.append(arg)
            output_file_path = arg.replace(".ini", ".json")
        if arg.endswith("json"):
            output_file_path = arg

    if not ini_file_path_list:
        ini_file_path_list.append("aicpu_kernel.ini")

    if not parse_ini_to_json(ini_file_path_list, output_file_path):
        sys.exit(1)
    sys.exit(0)
