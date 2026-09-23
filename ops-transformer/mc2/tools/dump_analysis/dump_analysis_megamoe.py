#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.

# !
#  \file megamoe_exception_dump_analysis.py
#  \brief MegaMoe ExceptionDump 结构化数据分析脚本。
#         解析 ParseStructuredDump 输出的单一 bin 文件，布局为：
#         [Header][Tiling][BlockStage[]][DumpParams[]][dump data 0][dump data 1]...
#
# 用法:
#   python megamoe_exception_dump_analysis.py <dump_file_path>

import os
import sys
import struct
import logging
from collections import Counter
import numpy as np

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

# ====== 常量（与 mc2_exception_dump.h 保持一致）======
HEADER_MAGIC = 0x5A5A5A5A
MAX_DUMP_ENTRIES = 1024
MAX_DUMP_TILING_SIZE = 2048

# Header 二进制布局（alignas(8)）:
#   int32  headerMagic       offset 0
#   int32  majorVersion      offset 4
#   int32  minorVersion      offset 8
#   int32  patchVersion      offset 12
#   int32  opType            offset 16
#   int32  stageNum          offset 20
#   uint32 tilingDataSize    offset 24
#   uint32 numBlocks         offset 28
#   uint64 dumpCount         offset 32
#   uint64 execTimes[1]      offset 40
#   total: 48 bytes
HEADER_SIZE = 48
HEADER_FORMAT = "<6i 2I 2Q"  # 6 int32 + 2 uint32 + 2 uint64

# DumpParams 二进制布局（alignas(8)）:
#   uintptr_t dumpAddr    8 bytes
#   size_t    blockCount  8 bytes
#   size_t    blockLen    8 bytes
#   size_t    srcStride   8 bytes
#   total: 32 bytes
DUMP_PARAMS_SIZE = 32
DUMP_PARAMS_FORMAT = "<4Q"

# MegaMoe Stage 枚举
STAGE_NAMES = [
    "INIT",  # 0
    "APPLY_XACTIVE_MASK",  # 1
    "MOE_INIT_ROUTING",  # 2
    "ALLGATHER_TOKEN_PER_EXPERT",  # 3
    "CUMSUM_TOKEN_PER_EXPERT",  # 4
    "DISPATCH",  # 5
    "SWIGLU",  # 6
    "COMBINE",  # 7
    "RESET_TOKEN_PER_EXPERT",  # 8
    "CROSS_RANK_SYNC",  # 9
    "UNPERMUTE",  # 10
]

# DumpParams 条目语义（DumpSyncRegions 产生的 5 条固定条目）
DUMP_ENTRY_NAMES = [
    "tokenPerExpert",  # 0: 各 rank 每专家 token 数矩阵
    "sync_counter",  # 1: CrossRankSync sync_counter[rank][0]
    "sync_base",  # 2: CrossRankSync sync_base
    "per_rank_status",  # 3: V2Set/V2Wait per-rank status
    "selfStatus",  # 4: InitStatusTargetSum selfStatus
]


def parse_header(data: bytes):
    """解析 Header，返回字段字典。"""
    if len(data) < HEADER_SIZE:
        logging.error("数据不足 %d 字节，无法解析 Header", HEADER_SIZE)
        return None

    fields = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    hdr = {
        "headerMagic": fields[0],
        "majorVersion": fields[1],
        "minorVersion": fields[2],
        "patchVersion": fields[3],
        "opType": fields[4],
        "stageNum": fields[5],
        "tilingDataSize": fields[6],
        "numBlocks": fields[7],
        "dumpCount": fields[8],
        "execTimes": fields[9],
    }
    return hdr


def parse_block_stages(data: bytes, offset: int, num_blocks: int, stage_num: int):
    """解析 BlockStage[]，返回每个核的 currentStage 和 stageCycle 列表。"""
    block_stage_size = (
        8 + stage_num * 8
    )  # uint64 currentStage + uint64 stageCycle[stageNum]
    results = []
    for i in range(num_blocks):
        bs_offset = offset + i * block_stage_size
        if bs_offset + block_stage_size > len(data):
            break
        current_stage = struct.unpack_from("<Q", data, bs_offset)[0]
        stage_cycles = struct.unpack_from(f"<{stage_num}Q", data, bs_offset + 8)
        results.append(
            {
                "coreId": i,
                "currentStage": current_stage,
                "stageCycle": stage_cycles,
            }
        )
    return results


def parse_dump_params(data: bytes, offset: int, dump_count: int):
    """解析 DumpParams[]，返回列表。"""
    results = []
    for i in range(dump_count):
        dp_offset = offset + i * DUMP_PARAMS_SIZE
        if dp_offset + DUMP_PARAMS_SIZE > len(data):
            break
        fields = struct.unpack(
            DUMP_PARAMS_FORMAT, data[dp_offset : dp_offset + DUMP_PARAMS_SIZE]
        )
        results.append(
            {
                "index": i,
                "dumpAddr": fields[0],
                "blockCount": fields[1],
                "blockLen": fields[2],
                "srcStride": fields[3],
            }
        )
    return results


def parse_dump_entry_data(data: bytes, offset: int, dp: dict):
    """解析单个 DumpParams 指向的数据，返回 numpy 数组。"""
    total_size = dp["blockCount"] * dp["blockLen"]
    if offset + total_size > len(data):
        logging.warning(
            "dump entry %d 数据不完整: 需要 %d 字节, 剩余 %d 字节",
            dp["index"],
            total_size,
            len(data) - offset,
        )
        total_size = min(total_size, len(data) - offset)

    raw = np.frombuffer(data[offset : offset + total_size], dtype=np.uint8)
    return raw, total_size


def get_stage_name(stage_id: int) -> str:
    if stage_id < len(STAGE_NAMES):
        return STAGE_NAMES[stage_id]
    return f"UNKNOWN({stage_id})"


def get_dump_entry_name(index: int) -> str:
    if index < len(DUMP_ENTRY_NAMES):
        return DUMP_ENTRY_NAMES[index]
    return f"dump_{index}"


def analyze_block_stages(block_stages: list):
    """分析各核执行阶段，定位卡死阶段。"""
    if not block_stages:
        return

    stages = [bs["currentStage"] for bs in block_stages]
    stage_counter = Counter(stages)
    majority_stage, majority_count = stage_counter.most_common(1)[0]

    logging.info("--- BlockStage Summary ---")
    logging.info("总核数: %d", len(block_stages))

    for stage_id, count in sorted(stage_counter.items()):
        name = get_stage_name(stage_id)
        logging.info(
            "  stage=%d (%s): %d 核 (%.1f%%)",
            stage_id,
            name,
            count,
            count * 100.0 / len(block_stages),
        )

    # 输出各核详情
    logging.info("各核当前阶段:")
    for bs in block_stages:
        name = get_stage_name(bs["currentStage"])
        # 找到第一个非零的 stageCycle（即最后进入的阶段）
        last_cycle = 0
        for _, c in enumerate(bs["stageCycle"]):
            if c != 0:
                last_cycle = c
        logging.info(
            "  core %2d: stage=%d (%s), lastCycle=0x%x",
            bs["coreId"],
            bs["currentStage"],
            name,
            last_cycle,
        )

    # 定位卡死阶段
    stuck_name = get_stage_name(majority_stage)
    logging.warning(
        ">>> 疑似卡死阶段: %s (%d/%d 核, %.1f%%)",
        stuck_name,
        majority_count,
        len(block_stages),
        majority_count * 100.0 / len(block_stages),
    )

    # 找出不在多数阶段的核
    minority_cores = [
        bs["coreId"] for bs in block_stages if bs["currentStage"] != majority_stage
    ]
    if minority_cores:
        logging.warning(">>> 以下核不在多数阶段: %s", minority_cores)


def analyze_dump_entries(dump_params: list, dump_data_offset: int, data: bytes):
    """分析 DumpParams 条目及对应数据。"""
    if not dump_params:
        logging.info("无 DumpParams 条目")
        return

    logging.info("--- DumpParams List ---")
    logging.info("有效条目数: %d", len(dump_params))

    current_offset = dump_data_offset
    for dp in dump_params:
        name = get_dump_entry_name(dp["index"])
        logging.info(
            "[%d] %s: addr=0x%x, blockCount=%d, blockLen=%d, srcStride=%d",
            dp["index"],
            name,
            dp["dumpAddr"],
            dp["blockCount"],
            dp["blockLen"],
            dp["srcStride"],
        )

        # 解析数据内容
        raw, consumed = parse_dump_entry_data(data, current_offset, dp)
        if consumed == 0:
            continue

        # 根据 dump entry 类型做特定分析
        if dp["index"] == 0:
            analyze_token_per_expert(raw, dp)
        elif dp["index"] == 1:
            analyze_sync_counter(raw, dp)
        elif dp["index"] == 2:
            analyze_sync_base(raw, dp)
        elif dp["index"] == 3:
            analyze_per_rank_status(raw, dp)
        elif dp["index"] == 4:
            analyze_self_status(raw, dp)

        current_offset += consumed


def analyze_token_per_expert(raw: np.ndarray, dp: dict):
    """分析 tokenPerExpert 数据。"""
    if dp["blockLen"] < 4:
        return
    int32_data = raw.view(np.int32)
    logging.info("  tokenPerExpert: %d 个 int32", len(int32_data))
    # 打印前 32 个值
    show = int32_data[: min(32, len(int32_data))]
    logging.info("  前 %d 个值: %s", len(show), show.tolist())
    if len(int32_data) > 32:
        logging.info("  ... (共 %d 个值)", len(int32_data))
    # 检查是否有全 0 或异常值
    non_zero = np.count_nonzero(int32_data)
    logging.info("  非零值数量: %d / %d", non_zero, len(int32_data))


def analyze_sync_counter(raw: np.ndarray, dp: dict):
    """分析 sync_counter 数据。每个 rank 取前 4B (1 个 int32)。"""
    logging.info("  sync_counter: %d 个 rank", dp["blockCount"])
    for i in range(min(dp["blockCount"], 32)):
        offset = i * dp["blockLen"]
        if offset + 4 <= len(raw):
            val = struct.unpack_from("<i", raw, offset)[0]
            logging.info("    rank[%d]: counter=%d (0x%x)", i, val, val & 0xFFFFFFFF)


def analyze_sync_base(raw: np.ndarray, dp: dict):
    """分析 sync_base 数据。单个 int32。"""
    if len(raw) >= 4:
        val = struct.unpack_from("<i", raw, 0)[0]
        logging.info("  sync_base: %d (0x%x)", val, val & 0xFFFFFFFF)


def analyze_per_rank_status(raw: np.ndarray, dp: dict):
    """分析 per-rank status 数据。每个 rank 32B (8 个 int32)。"""
    logging.info(
        "  per_rank_status: %d 个 rank, 每 rank %dB", dp["blockCount"], dp["blockLen"]
    )
    for i in range(min(dp["blockCount"], 32)):
        offset = i * dp["blockLen"]
        if offset + 32 <= len(raw):
            vals = struct.unpack_from("<8i", raw, offset)
            logging.info("    rank[%d]: %s", i, list(vals))


def analyze_self_status(raw: np.ndarray, dp: dict):
    """分析 selfStatus 数据。每个核 4B (1 个 int32)。"""
    logging.info("  selfStatus: %d 个核", dp["blockCount"])
    statuses = []
    for i in range(min(dp["blockCount"], 64)):
        offset = i * dp["blockLen"]
        if offset + 4 <= len(raw):
            val = struct.unpack_from("<i", raw, offset)[0]
            statuses.append(val)
    if statuses:
        active = sum(1 for v in statuses if v == 0x3F800000)
        inactive = sum(1 for v in statuses if v == 0)
        logging.info(
            "    active(1.0f): %d, inactive(0): %d, other: %d",
            active,
            inactive,
            len(statuses) - active - inactive,
        )
        logging.info(
            "    前 %d 个值: %s", len(statuses), [hex(v & 0xFFFFFFFF) for v in statuses]
        )


def main():
    if len(sys.argv) < 2:
        logging.error("用法: python %s <dump_file_path>", sys.argv[0])
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        logging.error("文件不存在: %s", file_path)
        sys.exit(1)

    logging.info("开始解析 MegaMoe ExceptionDump 文件: %s", file_path)

    with open(file_path, "rb") as f:
        data = f.read()

    logging.info("文件大小: %d 字节\n", len(data))

    # 1. 解析 Header
    hdr = parse_header(data)
    if hdr is None:
        sys.exit(1)

    if hdr["headerMagic"] != HEADER_MAGIC:
        logging.error(
            "headerMagic 不匹配: 0x%08X (期望 0x%08X)", hdr["headerMagic"], HEADER_MAGIC
        )
        logging.error("该文件可能不是 ExceptionDump 格式，或未启用 ExceptionDump")
        sys.exit(1)

    logging.info("=== MegaMoe Exception Dump Summary ===")
    logging.info(
        "CANN 版本: %d.%d.%d",
        hdr["majorVersion"],
        hdr["minorVersion"],
        hdr["patchVersion"],
    )
    logging.info("opType: %d, stageNum: %d", hdr["opType"], hdr["stageNum"])
    logging.info("execTimes: %d, dumpCount: %d", hdr["execTimes"], hdr["dumpCount"])
    logging.info(
        "tilingDataSize: %d, numBlocks: %d\n", hdr["tilingDataSize"], hdr["numBlocks"]
    )

    # 校验
    dump_count = min(hdr["dumpCount"], MAX_DUMP_ENTRIES)
    tiling_data_size = min(hdr["tilingDataSize"], MAX_DUMP_TILING_SIZE)
    num_blocks = hdr["numBlocks"]

    if num_blocks == 0:
        logging.error("numBlocks=0, Header 未初始化")
        sys.exit(1)

    # 2. 解析 Tiling 数据
    tiling_offset = HEADER_SIZE
    tiling_end = tiling_offset + tiling_data_size
    if tiling_data_size > 0 and tiling_end <= len(data):
        tiling_raw = data[tiling_offset:tiling_end]
        logging.info("--- Tiling Data ---")
        logging.info(
            "大小: %d 字节 (原始二进制, 需按 TilingDataT 结构体解析)\n",
            tiling_data_size,
        )
    else:
        logging.warning("Tiling 数据不可用\n")

    # 3. 解析 BlockStage[]
    block_stage_offset = tiling_offset + tiling_data_size
    block_stages = parse_block_stages(
        data, block_stage_offset, num_blocks, hdr["stageNum"]
    )
    analyze_block_stages(block_stages)
    logging.info("")

    # 4. 解析 DumpParams[]
    block_stage_size = 8 + hdr["stageNum"] * 8
    dump_params_offset = block_stage_offset + num_blocks * block_stage_size
    dump_params = parse_dump_params(data, dump_params_offset, dump_count)

    # 5. 解析 dump 数据
    dump_data_offset = dump_params_offset + dump_count * DUMP_PARAMS_SIZE
    analyze_dump_entries(dump_params, dump_data_offset, data)

    logging.info("\n=== 分析完成 ===")


if __name__ == "__main__":
    main()
