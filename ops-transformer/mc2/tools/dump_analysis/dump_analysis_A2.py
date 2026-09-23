#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file dump_analysis_A2.py
#  \brief

import os
import ast
import csv
import ctypes
import logging
from dataclasses import dataclass, field
from typing import List
from collections import Counter
import pandas as pd
import numpy as np


logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

UINT32_PER_BLOCK = 8
SERVER_RANK_NUM = 8
PER_CORE = 128
UINT32_BITS = 32
COM_WIN_ENTRY_SIZE = 5

PARENT_PATH = "result_A2/"
DUMP_FILE_PREFIX = "win_dump_input_serverid_"
DUMP_FILE_NAME = DUMP_FILE_PREFIX + "{}.csv"
ERROR_INFO_FILE_NAME = "errorinfo_{}.csv"
CLUSTOR_ERROR_FILE_NAME = "final_errorinfo.csv"


@dataclass
class SendRecvInfo:
    flags: list
    token_nums: list


@dataclass
class UnfinishInfo:
    send_eps: list = field(default_factory=list)
    recv_eps: list = field(default_factory=list)

    def has_unfinish(self):
        return bool(self.send_eps or self.recv_eps)


@dataclass
class FullSendRecvResult:
    send_info: SendRecvInfo
    recv_info: SendRecvInfo
    inner_info: list
    unfinish_info: UnfinishInfo
    unfinish_inner: UnfinishInfo = field(default_factory=UnfinishInfo)


def _check_unfinish(
    flags: list, tag: str, d_c: str, ep_rankid: int, inner: bool = False
):
    cur_server_id = ep_rankid // SERVER_RANK_NUM

    unfinish = UnfinishInfo()
    inner_tag = "inner " if inner else ""
    for i, flag in enumerate(flags):
        if flag == 0:
            if tag == "send" and i == cur_server_id:
                continue
            if tag == "recv" and inner and i == cur_server_id:
                continue
            (unfinish.send_eps if tag == "send" else unfinish.recv_eps).append(i)
    if unfinish.has_unfinish():
        logging.error(
            "[A2] 1.1 %s EP%d %s未完成%s的server:%s",
            d_c,
            ep_rankid,
            inner_tag,
            tag,
            unfinish.send_eps if tag == "send" else unfinish.recv_eps,
        )
    return unfinish


def _extract_field(arr: np.ndarray, core_num: int, offset: int):
    return [arr[offset + i * PER_CORE] for i in range(core_num)]


def get_islayer_ep(arr: np.ndarray, core_num: int, d_c: str):
    islayer = _extract_field(
        arr, core_num, 11
    )  # 魔鬼数字11见moe_distribute_a2_constant.h
    logging.info("[A2] 1.1 %s各核islayer:%s\n", d_c, islayer)
    max_islayer = max(Counter(islayer), key=Counter(islayer).get)
    diff_islayer = [i for i, v in enumerate(islayer) if v != max_islayer]
    if diff_islayer:
        logging.error(
            "[A2] 1.1 %s有如下下标的核的islayer与其他核不相等%s,共%d个核\n",
            d_c,
            diff_islayer,
            len(diff_islayer),
        )
    return islayer, max_islayer


def get_bs_ep(arr: np.ndarray, core_num: int, d_c: str):
    bs = _extract_field(arr, core_num, 10)  # 魔鬼数字10见moe_distribute_a2_constant.h
    logging.info("[A2] 1.1 %s各核bs:%s\n", d_c, bs)
    max_bs = max(Counter(bs), key=Counter(bs).get)
    diff_bs = [i for i, v in enumerate(bs) if v != max_bs]
    if diff_bs:
        logging.error(
            "[A2] 1.1 %s有如下下标的核的bs与其他核不相等%s,共%d个核\n",
            d_c,
            diff_bs,
            len(diff_bs),
        )
    return bs, max_bs


def get_h_ep(arr: np.ndarray, core_num: int, d_c: str):
    h = _extract_field(arr, core_num, 13)  # 魔鬼数字13见moe_distribute_a2_constant.h
    logging.info("[A2] 1.1 %s各核h:%s\n", d_c, h)
    max_h = max(Counter(h), key=Counter(h).get)
    diff_h = [i for i, v in enumerate(h) if v != max_h]
    if diff_h:
        logging.error(
            "[A2] 1.1 %s有如下下标的核的h与其他核不相等%s,共%d个核\n",
            d_c,
            diff_h,
            len(diff_h),
        )
    return h, max_h


def get_send_and_recv_info(
    arr: np.ndarray,
    core_num: int,
    islayer: int,
    ep_rankid: int,
    d_c: str,
    ep_worldsize: int,
) -> FullSendRecvResult:
    raw_send_or_recv, fullmesh_num_tokens = _get_raw_send_or_recv(
        arr, core_num, islayer, ep_worldsize
    )
    if islayer == 0:
        return _get_fullmesh_info(
            raw_send_or_recv,
            core_num,
            ep_rankid,
            d_c,
            ep_worldsize,
            fullmesh_num_tokens,
        )
    elif islayer == 1:
        return _get_hierarchy_info(raw_send_or_recv, ep_rankid, d_c, ep_worldsize)
    else:
        logging.error("[A2] 1.1 %s的islayer为%d,不支持分析\n", d_c, islayer)
        return FullSendRecvResult(
            SendRecvInfo([], []), SendRecvInfo([], []), [], UnfinishInfo()
        )


def _get_raw_send_or_recv(
    arr: np.ndarray, core_num: int, islayer: int, ep_worldsize: int
):
    # 魔鬼数字14见moe_distribute_a2_constant.h
    raw = [arr[14 + i * PER_CORE : (i + 1) * PER_CORE] for i in range(core_num)]
    fullmesh_num_tokens = []
    if islayer == 0:
        start = (core_num + 1) * PER_CORE
        end = start + ep_worldsize * 2
        fullmesh_num_tokens = arr[start:end]
    return raw, fullmesh_num_tokens


def _get_fullmesh_info(
    raw_send_or_recv: list,
    core_num: int,
    ep_rankid: int,
    d_c: str,
    ep_worldsize: int,
    fullmesh_num_tokens: list,
) -> FullSendRecvResult:
    has_send_ep = [0] * ep_worldsize
    has_recv_ep = [0] * ep_worldsize
    for i in range(core_num):
        rec = raw_send_or_recv[i]
        send_start, send_rank_num = int(rec[0]), int(rec[1])
        recv_start, recv_rank_num = int(rec[2]), int(rec[3])
        send_uint32_num = (send_rank_num - 1) // UINT32_BITS + 1
        send_flag = rec[4 : 4 + send_uint32_num]
        recv_flag = rec[4 + send_uint32_num :]

        for local_rank in range(send_rank_num):
            ep = local_rank + send_start
            has_send_ep[ep] = (
                send_flag[local_rank // UINT32_BITS] >> (local_rank % UINT32_BITS)
            ) & 0x1
        for local_rank in range(recv_rank_num):
            ep = local_rank + recv_start
            has_recv_ep[ep] = (
                recv_flag[local_rank // UINT32_BITS] >> (local_rank % UINT32_BITS)
            ) & 0x1

    unfinish_send = [i for i, v in enumerate(has_send_ep) if v == 0]
    unfinish_recv = [i for i, v in enumerate(has_recv_ep) if v == 0]
    if unfinish_send:
        logging.error(
            "[A2] 1.1 %s EP%d 未完成发送的EP:%s\n", d_c, ep_rankid, unfinish_send
        )
    if unfinish_recv:
        logging.error(
            "[A2] 1.1 %s EP%d 未完成接收的EP:%s\n", d_c, ep_rankid, unfinish_recv
        )

    return FullSendRecvResult(
        send_info=SendRecvInfo(
            has_send_ep, fullmesh_num_tokens[:ep_worldsize].tolist()
        ),
        recv_info=SendRecvInfo(
            has_recv_ep, fullmesh_num_tokens[ep_worldsize:].tolist()
        ),
        inner_info=[],
        unfinish_info=UnfinishInfo(unfinish_send, unfinish_recv),
    )


def _get_hierarchy_info(
    raw_send_or_recv: list, ep_rankid: int, d_c: str, ep_worldsize: int
) -> FullSendRecvResult:
    server_num = ep_worldsize // SERVER_RANK_NUM
    cur_server_id = ep_rankid // SERVER_RANK_NUM

    has_send_ep = [0] * server_num
    has_recv_ep = [0] * server_num
    send_num_tokens = [0] * server_num
    recv_num_tokens = [0] * server_num

    for i in range(server_num):
        rec = raw_send_or_recv[i]
        if rec[0] > 0:
            has_send_ep[int(rec[0]) - 1] = 1
        send_num_tokens[i] = int(rec[1])
        if rec[2] > 0:
            has_recv_ep[int(rec[2]) - 1] = 1
        recv_num_tokens[i] = int(rec[3])

    unfinish_send = _check_unfinish(has_send_ep, "send", d_c, ep_rankid)
    unfinish_recv = _check_unfinish(has_recv_ep, "recv", d_c, ep_rankid)

    logging.info("[A2] 1.1 %s EP%d has send rank:%s\n", d_c, ep_rankid, has_send_ep)
    logging.info("[A2] 1.1 %s EP%d has recv rank:%s\n", d_c, ep_rankid, has_recv_ep)
    logging.info(
        "[A2] 1.1 %s EP%d send token num:%s\n", d_c, ep_rankid, send_num_tokens
    )
    logging.info(
        "[A2] 1.1 %s EP%d recv token num:%s\n", d_c, ep_rankid, recv_num_tokens
    )

    if d_c == "dispatch":
        inner_info, unfinish_inner = _get_hierarchy_dis_inner_info(
            raw_send_or_recv, server_num, cur_server_id, d_c, ep_rankid
        )
    else:
        inner_info, unfinish_inner = [], UnfinishInfo()

    return FullSendRecvResult(
        send_info=SendRecvInfo(has_send_ep, send_num_tokens),
        recv_info=SendRecvInfo(has_recv_ep, recv_num_tokens),
        inner_info=inner_info,
        unfinish_info=UnfinishInfo(unfinish_send.send_eps, unfinish_recv.recv_eps),
        unfinish_inner=unfinish_inner,
    )


def _get_hierarchy_dis_inner_info(
    raw_send_or_recv: list,
    server_num: int,
    cur_server_id: int,
    d_c: str,
    ep_rankid: int,
):
    has_send_ep = [0] * server_num
    has_recv_ep = [0] * server_num

    for i in range(server_num):
        rec = raw_send_or_recv[i]
        if rec[4] > 0:
            has_send_ep[int(rec[4]) - 1] = 1
        if rec[5] > 0:
            has_recv_ep[int(rec[5]) - 1] = 1

    unfinish_send = _check_unfinish(has_send_ep, "send", d_c, ep_rankid, inner=True)
    unfinish_recv = _check_unfinish(has_recv_ep, "recv", d_c, ep_rankid, inner=True)

    logging.info("[A2] 1.1 %s EP%d inner has_send_ep:%s\n", d_c, ep_rankid, has_send_ep)
    logging.info("[A2] 1.1 %s EP%d inner has_recv_ep:%s\n", d_c, ep_rankid, has_recv_ep)

    return [has_send_ep, has_recv_ep], UnfinishInfo(
        unfinish_send.send_eps, unfinish_recv.recv_eps
    )


def get_dumpinfo(
    target_path: str, bsk: int, moe_expert_num: int, islayer: int, d_c: str
):
    expandidx_input = (
        get_dump_expandidx(target_path, bsk)
        if (d_c == "combine" and islayer == 0)
        else []
    )
    epsendcnt_input = (
        get_dump_epsendcnt(target_path, moe_expert_num) if d_c == "combine" else None
    )
    xactivemask_input = get_dump_xactivemask(target_path, d_c)
    return expandidx_input, xactivemask_input, epsendcnt_input


def get_dump_expandidx(target_path: str, bsk: int):
    for filename in os.listdir(target_path):
        if "input.2.bin" in filename:
            file_path = os.path.join(target_path, filename)
            return np.fromfile(file_path, dtype=np.int32)[:bsk].tolist()
    return []


def get_dump_xactivemask(target_path: str, d_c: str):
    target_name = "input.3.bin" if d_c == "dispatch" else "input.6.bin"
    for filename in os.listdir(target_path):
        if target_name in filename:
            file_path = os.path.join(target_path, filename)
            mask = np.fromfile(file_path, dtype=np.uint8).tolist()
            return mask if max(mask) != 0 else []
    return []


def get_dump_epsendcnt(target_path: str, moe_expert_num: int):
    for filename in os.listdir(target_path):
        if "input.3.bin" in filename:
            file_path = os.path.join(target_path, filename)
            return np.fromfile(file_path, dtype=np.int32)[:moe_expert_num].tolist()
    return []


def get_win_state(
    win_status: np.ndarray,
    d_c: str,
    moe_expert_num: int,
    ep_worldsize: int,
    islayer: int,
):
    if islayer != 0:
        return None
    if d_c == "dispatch":
        return _get_fullmesh_dis_win_state(win_status, moe_expert_num, ep_worldsize)
    return _get_fullmesh_com_win_state(win_status, ep_worldsize)


def _get_fullmesh_dis_win_state(
    dis_win_status: np.ndarray, moe_expert_num: int, ep_worldsize: int
):
    local_moe_expert_num = moe_expert_num // ep_worldsize
    status_entry_count = (
        local_moe_expert_num // UINT32_PER_BLOCK + 1
    ) * UINT32_PER_BLOCK
    count_per_rank = status_entry_count + 2
    dis_win_status = dis_win_status.view(np.int32)
    win_status = []
    for i in range(ep_worldsize):
        offset = i * count_per_rank
        win_status.append(dis_win_status[offset : offset + count_per_rank].tolist())
    return win_status


def _get_fullmesh_com_win_state(com_win_status: np.ndarray, ep_worldsize: int):
    count_per_rank = COM_WIN_ENTRY_SIZE
    com_win_status = com_win_status.view(np.uint32)
    win_status = []
    for i in range(ep_worldsize):
        offset = i * count_per_rank
        win_status.append(com_win_status[offset : offset + count_per_rank].tolist())
    return win_status


def get_device_count():
    try:
        import torch

        return torch.npu.device_count()
    except Exception:
        logging.info("[A2] No torch, thus try to get device count by another way.")
    arch = os.uname().machine
    if arch == "aarch64":
        return 8
    elif arch == "x86_64":
        return 16
    logging.error("[A2] 未知架构:%s，默认返回8个设备", arch)
    return 8


def _save_csv(filename, row_data, index=None, columns=None):
    if index is not None:
        file_exists = os.path.exists(filename)
        pd.DataFrame(row_data, columns=columns, index=[index]).to_csv(
            filename, index=True, mode="a", header=not file_exists, encoding="gbk"
        )
    else:
        pd.DataFrame(row_data, columns=columns).to_csv(
            filename, mode="w", header=True, encoding="gbk"
        )
    logging.info("[A2] Save %s done.", filename)


def save_dump_data(
    ep_rankid: int,
    ep_worldsize: int,
    moe_expert_num: int,
    k: int,
    islayer: int,
    d_c: str,
    buffer_id: int,
    bs: int,
    h: int,
    *,
    dis_run_cnts=0,
    com_run_cnts=0,
    dis_send_info=None,
    dis_recv_info=None,
    dis_inner_info=None,
    dis_unfinish_info=None,
    com_send_info=None,
    com_recv_info=None,
    dis_win_state=None,
    com_win_state=None,
    com_unfinish_info=None,
    expertids=None,
    expandidx_input=None,
    xactivemask_input=None,
    epsendcnt_input=None,
):
    os.makedirs(PARENT_PATH, exist_ok=True)
    device_count = get_device_count()
    server_id = ep_rankid // device_count
    local_rank_id = ep_rankid % device_count
    file_name = os.path.join(PARENT_PATH, DUMP_FILE_NAME.format(server_id))

    if expertids is None:
        expertids = np.array([], dtype=np.int32)

    row_data = [
        [
            ep_rankid,
            ep_worldsize,
            moe_expert_num,
            bs,
            k,
            islayer,
            d_c,
            buffer_id,
            h,
            dis_run_cnts,
            com_run_cnts,
            dis_send_info,
            dis_recv_info,
            dis_inner_info,
            dis_unfinish_info,
            com_send_info,
            com_recv_info,
            com_unfinish_info,
            dis_win_state,
            com_win_state,
            expertids.tolist(),
            expandidx_input,
            xactivemask_input,
            epsendcnt_input,
        ]
    ]
    columns = [
        "ep_rankid",
        "ep_worldsize",
        "moe_expert_num",
        "bs",
        "k",
        "是否分层",
        "当前报错算子",
        "当前buffer_id",
        "h",
        "dispatch执行次数",
        "combine执行次数",
        "dispatch是否发送及token数",
        "dispatch是否接收及token数",
        "dispatchInner表发送接收情况",
        "dispatch未完成行为",
        "combine是否发送及token数",
        "combine是否接收及token数",
        "combine未完成行为",
        "dispatch接收状态",
        "combine接收状态",
        "expertids",
        "expandidx_input",
        "xactivemask_input",
        "epsendcnt_input",
    ]
    _save_csv(file_name, row_data, local_rank_id, columns)
    logging.info("[A2] EP%d dump info data save to %s", ep_rankid, file_name)


def save_win_data(
    unfinish_info,
    unfinish_info_inner,
    islayer: int,
    ep_rankid: int,
    ep_worldsize: int,
    d_c: str,
    min_run_coreid,
    run_info,
):
    device_count = get_device_count()
    server_id = ep_rankid // device_count
    local_rank_id = ep_rankid % device_count

    os.makedirs(PARENT_PATH, exist_ok=True)
    file_name = os.path.join(PARENT_PATH, ERROR_INFO_FILE_NAME.format(server_id))
    unfinish_send_ep, unfinish_recv_ep = unfinish_info
    unfinish_send_ep_inner, unfinish_recv_ep_inner = unfinish_info_inner

    columns = [
        "ep_rankid",
        "ep_worldsize",
        "是否分层",
        "出错的算子名",
        "最早出问题的核",
        "最早出问题的核执行位置",
        "未发送token到这些rank,hierarchy为serverId",
        "未接收这些rank的token,hierarchy为serverId",
        "未发送inner到这些serverId,hierarchy-dispatch有效",
        "未接收这些serverId的inner,hierarchy-dispatch有效",
    ]
    save_datas = [
        ep_rankid,
        ep_worldsize,
        islayer,
        d_c,
        min_run_coreid,
        run_info[min_run_coreid[0]],
        unfinish_send_ep,
        unfinish_recv_ep,
        unfinish_send_ep_inner,
        unfinish_recv_ep_inner,
    ]
    for coreid, core_run_info in enumerate(run_info):
        columns.append(f"核{coreid}执行位置")
        save_datas.append(core_run_info)
    _save_csv(file_name, [save_datas], local_rank_id, columns)
    logging.info("[A2] EP%d win info data save to %s", ep_rankid, file_name)


def _get_runpos_sort(next_pos):
    n = len(next_pos)
    pos_sort = [0] * n
    cnt = 0
    cur_pos = 0
    while next_pos[cur_pos] < n:
        pos_sort[cur_pos] = cnt
        cnt += 1
        cur_pos = next_pos[cur_pos]
    pos_sort[cur_pos] = cnt
    return pos_sort


def _get_runpos_mapping(islayer: int, d_c: str):
    pos_common = ["UNSTART", "INIT", "FINISH"]

    if islayer == 0:
        pos_diff = (
            [
                "REORDERTOKEN",
                "SEND",
                "WAIT_DISPATCH",
                "GET_STATUS_CUM_SUM",
                "LOCAL_WINDOW_COPY",
            ]
            if d_c == "dispatch"
            else ["ALLTOALLDISPATCH", "WAIT_DISPATCH"]
        )
    else:
        pos_diff = (
            [
                "REORDERTOKEN",
                "SEND_DATA_TO_SERVER",
                "INTER_FINISH",
                "INTRA_FINISH",
                "IPC2OUT",
            ]
            if d_c == "dispatch"
            else ["GM2IPC", "WAIT2IPC", "INTER_SEND", "INTER_RECV"]
        )

    pos_mapping = pos_common + pos_diff
    next_pos = [1, 3, len(pos_mapping)] + list(range(4, len(pos_mapping))) + [2]
    pos_position = _get_runpos_sort(next_pos)
    return pos_mapping, next_pos, pos_position


def analyze_runpos(runpos: List[int], islayer: int, d_c: str, servernum: int):
    pos_mapping, next_pos, pos_position = _get_runpos_mapping(islayer, d_c)
    core_num = len(runpos)
    run_info = [""] * core_num
    min_run_coreid = []
    min_run_pos = len(pos_mapping)

    for core_id in range(core_num):
        cur = runpos[core_id]
        cur_info = pos_mapping[cur]
        next_info = (
            pos_mapping[next_pos[cur]] if next_pos[cur] < len(pos_mapping) else ""
        )
        if islayer == 1 and cur_info == "REORDERTOKEN" and core_id >= servernum:
            next_info = "INTER_FINISH"
        run_info[core_id] = f"{cur_info} -> {next_info}"

        cur_pos = pos_position[cur]
        if cur_pos < min_run_pos:
            min_run_coreid = [core_id]
            min_run_pos = cur_pos
        elif cur_pos == min_run_pos:
            min_run_coreid.append(core_id)

    logging.error(
        "[A2] 最早出问题的核:%s, 对应代码区域:%s",
        min_run_coreid,
        run_info[min_run_coreid[0]],
    )
    return min_run_coreid, run_info


def analyze_single_card(
    arr_func_dis: np.ndarray,
    arr_func_com: np.ndarray,
    core_num: int,
    ep_rankid: int,
    d_c: str,
    ep_worldsize: int,
    runpos: List[int],
    moe_expert_num: int,
    k: int,
    buffer_id: int,
    target_path: str,
    expertids: np.ndarray,
    dis_status: np.ndarray,
    com_status: np.ndarray,
    *,
    dis_run_cnts=0,
    com_run_cnts=0,
):
    arr_func = arr_func_dis if d_c == "dispatch" else arr_func_com
    _, islayer = get_islayer_ep(arr_func, core_num, d_c)
    _, max_h = get_h_ep(arr_func, core_num, d_c)
    _, max_bs = get_bs_ep(arr_func, core_num, d_c)

    result = get_send_and_recv_info(
        arr_func_dis, core_num, islayer, ep_rankid, "dispatch", ep_worldsize
    )
    dis_send_info, dis_recv_info = result.send_info, result.recv_info
    dis_inner_info = result.inner_info
    dis_unfinish_info = [result.unfinish_info.send_eps, result.unfinish_info.recv_eps]

    dis_window_status = get_win_state(
        dis_status, "dispatch", moe_expert_num, ep_worldsize, islayer
    )

    if d_c == "combine":
        com_result = get_send_and_recv_info(
            arr_func_com, core_num, islayer, ep_rankid, d_c, ep_worldsize
        )
        com_send_info, com_recv_info = com_result.send_info, com_result.recv_info
        com_unfinish_info = [
            com_result.unfinish_info.send_eps,
            com_result.unfinish_info.recv_eps,
        ]
        com_window_status = get_win_state(
            com_status, d_c, moe_expert_num, ep_worldsize, islayer
        )
    else:
        com_send_info = com_recv_info = com_unfinish_info = None
        com_window_status = None

    expandidx_input, xactivemask_input, epsendcnt_input = get_dumpinfo(
        target_path, max_bs * k, moe_expert_num, islayer, d_c
    )

    min_run_coreid, run_info = analyze_runpos(
        runpos, islayer, d_c, ep_worldsize // SERVER_RANK_NUM
    )
    save_dump_data(
        ep_rankid,
        ep_worldsize,
        moe_expert_num,
        k,
        islayer,
        d_c,
        buffer_id,
        bs=max_bs,
        h=max_h,
        dis_run_cnts=dis_run_cnts,
        com_run_cnts=com_run_cnts,
        dis_send_info=dis_send_info,
        dis_recv_info=dis_recv_info,
        dis_inner_info=dis_inner_info,
        dis_unfinish_info=dis_unfinish_info,
        com_send_info=com_send_info,
        com_recv_info=com_recv_info,
        dis_win_state=dis_window_status,
        com_win_state=com_window_status,
        com_unfinish_info=com_unfinish_info,
        expertids=expertids,
        expandidx_input=expandidx_input,
        xactivemask_input=xactivemask_input,
        epsendcnt_input=epsendcnt_input,
    )

    unfinish_info = dis_unfinish_info if d_c == "dispatch" else com_unfinish_info
    if d_c == "dispatch" and islayer == 1:
        unfinish_info_inner = [
            result.unfinish_inner.send_eps,
            result.unfinish_inner.recv_eps,
        ]
    else:
        unfinish_info_inner = [[], []]
    save_win_data(
        unfinish_info,
        unfinish_info_inner,
        islayer,
        ep_rankid,
        ep_worldsize,
        d_c,
        min_run_coreid,
        run_info,
    )


# 暂不支持xactivemask
def isinvalid_token(expid, moe_expert_num):
    if expid >= moe_expert_num or expid < 0:
        return True
    return False


def check_single_card_expandidx(
    expandidx_input, expertids, xactivemask, moe_expert_num, bsk
):
    expertids_flatten = np.array(expertids, dtype=np.int32).reshape(-1)
    bsk = min(bsk, expertids_flatten.size)
    expandidx_input = np.array(expandidx_input, dtype=np.int32).reshape(-1)[:bsk]
    expandidx_golden = np.zeros(bsk, dtype=np.int32)
    if xactivemask is not None:
        xactivemask = np.array(xactivemask, dtype=np.int32)
    count_dict = {}
    for i in range(expertids_flatten.size):
        expid = expertids_flatten[i].item()
        if isinvalid_token(expid, moe_expert_num):
            continue
        count_dict[expid] = count_dict.get(expid, -1) + 1
        expandidx_golden[i] = count_dict[expid]
    return np.array_equal(expandidx_golden, expandidx_input)


def check_expandidx(
    expandidx_input_func, expertids_func, xactivemask_func, bsk, moe_expert_num
):
    error_rankid_info = []
    error_rankid = []
    for ep_rankid in sorted(expandidx_input_func.keys(), key=lambda x: int(x)):
        ok = check_single_card_expandidx(
            expandidx_input_func[ep_rankid],
            expertids_func[ep_rankid],
            xactivemask_func[ep_rankid],
            moe_expert_num,
            bsk,
        )
        if not ok:
            error_rankid.append(ep_rankid)
    if error_rankid:
        print_log = f"Combine Input ExpandIdxInput is wrong. Wrong EP: {error_rankid}"
        logging.error("[A2][Fullmesh] %s", print_log)
        error_rankid_info.append(print_log)
    return error_rankid_info


def check_epsendcnt(
    epsendcnt_input_func,
    expertids_func,
    xactivemask_func,
    ep_worldsize,
    bs,
    k,
    moe_expert_num,
    islayer,
):
    error_rankid_info = []
    has_cards = len(epsendcnt_input_func)
    if has_cards != ep_worldsize:
        logging.error(
            "[A2] Combine EP send count check failed: expected %d ranks but get %d ranks, layer: %d",
            ep_worldsize,
            has_cards,
            islayer,
        )
        logging.error("[A2] Only EP%s have data.", list(epsendcnt_input_func.keys()))
        return error_rankid_info

    epsendcnt_golden = np.zeros((ep_worldsize, moe_expert_num), dtype=np.int32)
    for ep_rankid in sorted(epsendcnt_input_func.keys(), key=lambda x: int(x)):
        expert_ids = np.array(expertids_func[ep_rankid], dtype=np.int32).flatten()
        bsk = min(bs * k, expert_ids.size)
        if not islayer and xactivemask_func[ep_rankid] is not None:
            xactivemask = np.array(xactivemask_func[ep_rankid], dtype=np.uint8)
            if xactivemask.ndim == 1:
                xactivemask = np.broadcast_to(xactivemask[:bs, None], (bs, k)).flatten()
            else:
                xactivemask = xactivemask.flatten()[:bsk]
            mask = (expert_ids < moe_expert_num) & (expert_ids >= 0)
            active_indices = np.where(mask)[0]
            expert_ids = expert_ids[active_indices]
        epsendcnt_golden[int(ep_rankid)] = np.bincount(
            expert_ids, minlength=moe_expert_num
        )

    cumulative_shape = (ep_worldsize, moe_expert_num)
    epsendcnt_golden = np.cumsum(
        epsendcnt_golden.T.reshape(cumulative_shape), axis=-1, dtype=np.int32
    )

    error_rankid = []
    for ep_rankid in sorted(epsendcnt_input_func.keys(), key=lambda x: int(x)):
        epsendcnt = np.array(epsendcnt_input_func[ep_rankid], dtype=np.int32)
        if not np.array_equal(epsendcnt, epsendcnt_golden[int(ep_rankid)]):
            error_rankid.append(ep_rankid)
            logging.error(
                "[A2][Fullmesh] Combine EP send count mismatch: golden=%s, dump=%s",
                epsendcnt_golden[int(ep_rankid)].tolist(),
                epsendcnt.tolist(),
            )
    if error_rankid:
        print_log = f"Combine SendCount is wrong. Wrong EP: {error_rankid}"
        logging.error("[A2][Fullmesh] %s", print_log)
        error_rankid_info.append(print_log)
    return error_rankid_info


def check_fullmesh_com_win(com_win_state_func):
    error_rankid_info = []
    for ep_rankid in sorted(com_win_state_func.keys(), key=lambda x: int(x)):
        for target_rankid, win_state in enumerate(com_win_state_func[ep_rankid]):
            if hex(ctypes.c_uint32(win_state[0]).value) != "0xffffffff":
                continue
            print_log = (
                f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Stalled in waitDispatch; "
                "communication not completed."
            )
            error_rankid_info.append(print_log)
            logging.error(print_log)
            target_token_num, recv_token_num = win_state[1], win_state[2]
            target_flag, recv_flag = win_state[3], win_state[4]
            logging.info(
                "[A2][Fullmesh] Combine EP%s win state: "
                "target_token_num=%d, recv_token_num=%d, target_flag=%d, recv_flag=%d",
                ep_rankid,
                target_token_num,
                recv_token_num,
                target_flag,
                recv_flag,
            )
            if target_token_num != recv_token_num:
                print_log = (
                    f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Combine recv token num mismatch: "
                    f"target_token_num={target_token_num}, recv_token_num={recv_token_num}"
                )
                error_rankid_info.append(print_log)
                logging.error(print_log)
                if recv_token_num == 0:
                    print_log = (
                        f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] No tokens received; "
                        f"likely EP{target_rankid} did not send data."
                    )
                    error_rankid_info.append(print_log)
                    logging.error(print_log)
            recv_flag_hex = hex(ctypes.c_uint32(recv_flag).value)
            if recv_flag_hex != "0xffffffff":
                print_log = (
                    f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Combine recv flag mismatch: "
                    f"target_flag=0xffffffff, recv_flag={recv_flag_hex}"
                )
                error_rankid_info.append(print_log)
                logging.error(print_log)
            if target_token_num == recv_token_num and recv_flag_hex != "0xffffffff":
                print_log = (
                    f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] "
                    f"Combined receive token count: {recv_token_num} "
                    "(header correct), but tail flag corrupted. "
                    "Likely HCCL_BUFFSIZE insufficient, causing flag overwrite."
                )
                error_rankid_info.append(print_log)
                logging.error(print_log)
            elif target_token_num != recv_token_num and recv_flag_hex == "0xffffffff":
                print_log = (
                    f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Received token count: {recv_token_num} "
                    "(header mismatch), tail flag valid. Likely sender transmitted incorrect token count."
                )
                error_rankid_info.append(print_log)
                logging.error(print_log)
            elif target_token_num == recv_token_num and recv_flag_hex == "0xffffffff":
                print_log = (
                    f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Received token count: {recv_token_num} "
                    "(header correct), tail flag valid. Why stall?"
                )
                error_rankid_info.append(print_log)
                logging.error(print_log)
    return error_rankid_info


def check_com(
    expandidx_input_func,
    epsendcnt_input_func,
    expertids_func,
    xactivemask_func,
    com_win_state_func,
    ep_worldsize,
    bs,
    k,
    moe_expert_num,
    islayer,
):
    error_rankid_info = []
    rt_info = check_epsendcnt(
        epsendcnt_input_func,
        expertids_func,
        xactivemask_func,
        ep_worldsize,
        bs,
        k,
        moe_expert_num,
        islayer,
    )
    error_rankid_info.extend(rt_info)
    if not islayer:
        rt_info = check_expandidx(
            expandidx_input_func,
            expertids_func,
            xactivemask_func,
            bs * k,
            moe_expert_num,
        )
        error_rankid_info.extend(rt_info)
        rt_info = check_fullmesh_com_win(com_win_state_func)
        error_rankid_info.extend(rt_info)
    return error_rankid_info


def cal_dis_win_status_golden(
    expertids_func, xactivemask_func, ep_worldsize, moe_expert_num, bsk
):
    dis_win_status_golden = [None] * ep_worldsize
    local_moe_expert_num = moe_expert_num // ep_worldsize
    for ep_rankid in sorted(expertids_func.keys(), key=lambda x: int(x)):
        status_golden = np.zeros((ep_worldsize, local_moe_expert_num), dtype=np.int32)
        expertids_flatten = np.array(expertids_func[ep_rankid], dtype=np.int32).reshape(
            -1
        )
        expertids_flatten = expertids_flatten[: min(bsk, expertids_flatten.size)]
        xactivemask = xactivemask_func[ep_rankid]
        if xactivemask is not None:
            xactivemask = np.array(xactivemask, dtype=np.uint8)
        for expid in expertids_flatten:
            if not isinvalid_token(expid, moe_expert_num):
                status_golden[
                    expid // local_moe_expert_num, expid % local_moe_expert_num
                ] += 1
        dis_win_status_golden[int(ep_rankid)] = status_golden
    return dis_win_status_golden


def check_single_card_dis_win(
    dis_win_state, dis_win_status_golden, unfinish_recv_ep, ep_rankid
):
    error_rankid_info = []
    if unfinish_recv_ep:
        logging.info(
            "[A2][Fullmesh] Dispatch has't recv token from EP %s, according to ADump.",
            unfinish_recv_ep,
        )
    for target_rankid, win_state in enumerate(dis_win_state):
        if target_rankid in unfinish_recv_ep:
            logging.error(
                "[A2] According to Dump, EP%s has't recv token from EP %s.",
                ep_rankid,
                target_rankid,
            )
            print_log = (
                f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] "
                "Dispatch According to Dump: unfinish recv token"
            )
            error_rankid_info.append(print_log)
        is_un_recv = hex(ctypes.c_uint32(win_state[0]).value) == "0xffffffff"

        if not is_un_recv:
            continue
        print_log = (
            f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Stalled in waitDispatch; "
            "communication not completed."
        )
        error_rankid_info.append(print_log)
        logging.error(print_log)
        status_flag = hex(ctypes.c_uint32(win_state[-2]).value)
        if status_flag != "0xffffffff":
            print_log = (
                f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Status flag({status_flag}) corrupted, "
                "should be 0xffffffff"
            )
            error_rankid_info.append(print_log)
            logging.error(print_log)
            continue

        if dis_win_status_golden[target_rankid] is None:
            logging.info(
                "[A2][Fullmesh] Dispatch has no EP %d expertids.", target_rankid
            )
            win_state_golden = dis_win_status_golden[target_rankid][int(ep_rankid)]
            local_moe_expert_num = win_state_golden.shape[-1]
            win_state_current = np.array(win_state, dtype=np.int32)[
                1 : local_moe_expert_num + 1
            ]
            if not np.array_equal(win_state_current, win_state_golden):
                print_log = (
                    f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Dispatch recv win status mismatch: "
                    f"golden={win_state_golden.tolist()}, dump={win_state_current.tolist()}"
                )
                error_rankid_info.append(print_log)
                logging.error(
                    "[A2][Fullmesh] Dispatch EP%s recv wrong status from EP%d",
                    ep_rankid,
                    target_rankid,
                )
                logging.error(
                    "[A2][Fullmesh] Win status golden: %s", win_state_golden.tolist()
                )
                logging.error(
                    "[A2][Fullmesh] Win status current: %s", win_state_current.tolist()
                )
        data_flag = hex(ctypes.c_uint32(win_state[-1]).value)
        if data_flag != "0xffffffff":
            print_log = (
                f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] Data flag({data_flag}) corrupted, "
                "should be 0xffffffff"
            )
            error_rankid_info.append(print_log)
            logging.error(print_log)
        else:
            print_log = f"[RecvEP:{ep_rankid}][SendEP:{target_rankid}] All checks passed, but why stall?."
            error_rankid_info.append(print_log)
            logging.error(print_log)
    return error_rankid_info


def check_fullmesh_dis_win(
    dis_win_state_func,
    expertids_func,
    xactivemask_func,
    unfinish_ep_func,
    ep_worldsize,
    moe_expert_num,
    bsk,
):
    dis_win_status_golden = cal_dis_win_status_golden(
        expertids_func, xactivemask_func, ep_worldsize, moe_expert_num, bsk
    )
    error_rankid_infos = []
    for ep_rankid in sorted(dis_win_state_func.keys(), key=lambda x: int(x)):
        error_info = check_single_card_dis_win(
            dis_win_state_func[ep_rankid],
            dis_win_status_golden,
            unfinish_ep_func[ep_rankid][1],
            ep_rankid,
        )
        error_rankid_infos.extend(error_info)
    return error_rankid_infos


def check_dis(
    dis_win_state_func,
    expertids_func,
    xactivemask_func,
    unfinish_ep_func,
    ep_worldsize,
    moe_expert_num,
    bsk,
):
    error_rankid_infos = check_fullmesh_dis_win(
        dis_win_state_func,
        expertids_func,
        xactivemask_func,
        unfinish_ep_func,
        ep_worldsize,
        moe_expert_num,
        bsk,
    )
    if error_rankid_infos:
        logging.error("[A2][Fullmesh] Dispatch Win State status mismatch")
    return error_rankid_infos


def analysis_all_cards(parent_path=PARENT_PATH):
    dis_win_state_func, com_win_state_func = {}, {}
    expertids_func, xactivemask_func = {}, {}
    unfinish_ep_func = {}
    expandidx_input_func, epsendcnt_input_func = {}, {}
    ep_worldsize, moe_expert_num = None, None
    bs, k, islayer, d_c = None, None, None, None
    filename = None

    for filename in os.listdir(parent_path):
        if not (filename.startswith(DUMP_FILE_PREFIX) and filename.endswith(".csv")):
            continue
        with open(os.path.join(parent_path, filename), "r", encoding="gbk") as f:
            for row in csv.DictReader(f):
                ep_rankid = row["ep_rankid"].strip()
                islayer = int(row["是否分层"])
                if islayer == 0:
                    dis_win_state_func[ep_rankid] = ast.literal_eval(
                        row["dispatch接收状态"]
                    )
                d_c = row["当前报错算子"].strip()
                expertids_func[ep_rankid] = ast.literal_eval(row["expertids"])
                xactivemask_func[ep_rankid] = ast.literal_eval(row["xactivemask_input"])
                if d_c == "combine":
                    expandidx_input_func[ep_rankid] = ast.literal_eval(
                        row["expandidx_input"]
                    )
                    epsendcnt_input_func[ep_rankid] = ast.literal_eval(
                        row["epsendcnt_input"]
                    )
                    unfinish_ep_func[ep_rankid] = ast.literal_eval(
                        row["combine未完成行为"]
                    )
                    if islayer == 0:
                        com_win_state_func[ep_rankid] = ast.literal_eval(
                            row["combine接收状态"]
                        )
                else:
                    unfinish_ep_func[ep_rankid] = ast.literal_eval(
                        row["dispatch未完成行为"]
                    )
                ep_worldsize = int(row["ep_worldsize"])
                moe_expert_num = int(row["moe_expert_num"])
                bs = int(row["bs"])
                k = int(row["k"])

    logging.info("[A2] Start to analysis %s", filename)
    logging.info(
        "[A2] ep_worldsize: %s, moe_expert_num: %s, bs: %s, k: %s, islayer: %s, d_c: %s",
        ep_worldsize,
        moe_expert_num,
        bs,
        k,
        islayer,
        d_c,
    )

    if d_c is not None and islayer == 0:
        if d_c == "dispatch":
            error_rankid_info = check_dis(
                dis_win_state_func,
                expertids_func,
                xactivemask_func,
                unfinish_ep_func,
                ep_worldsize,
                moe_expert_num,
                bs * k,
            )
        else:
            error_rankid_info = check_com(
                expandidx_input_func,
                epsendcnt_input_func,
                expertids_func,
                xactivemask_func,
                com_win_state_func,
                ep_worldsize,
                bs,
                k,
                moe_expert_num,
                islayer,
            )
        file_name = os.path.join(PARENT_PATH, CLUSTOR_ERROR_FILE_NAME)
        _save_csv(file_name, error_rankid_info, columns=["error_info"])
    logging.info("[A2] Analysis %s done.", filename)


if __name__ == "__main__":
    analysis_all_cards(parent_path=PARENT_PATH)
