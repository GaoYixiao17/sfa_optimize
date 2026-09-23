#  Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  This program is free software, you can redistribute it and/or modify it under the terms and conditions of
#  CANN Open Software License Agreement Version 2.0 (the "License").
#  Please refer to the License for details. You may not use this file except in compliance with the License.
#  THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
#  See LICENSE in the root of the software repository for the full text of the License.
#

# !
#  \file dump_analysis.py
#  \brief

import os
import re
import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.NOTSET,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# 获取每张卡对应的csv表的绝对路径
def find_device_and_csv_by_num(profiling_path_func, card_num_func):
    csv_filename = "kernel_details.csv"
    csv_path_list_func = [None] * card_num_func

    # 1. 只在传入路径第一层找包含 ascend_pt 的文件夹（不递归）
    ascend_pt_dirs = []
    try:
        for entry in os.listdir(profiling_path_func):
            dir_full_path = os.path.join(profiling_path_func, entry)
            # 是文件夹 + 名称包含 ascend_pt
            if os.path.isdir(dir_full_path) and "ascend_pt" in entry:
                ascend_pt_dirs.append(dir_full_path)
    except PermissionError:
        logging.error("访问路径{%s}权限不足，无法读取目录", profiling_path_func)
    except Exception as e:
        logging.error("读取目录 {%s} 失败，异常信息:{%s}", profiling_path_func, str(e))

    # 2. 按卡数量取前 N 个
    target_dirs = ascend_pt_dirs[:card_num_func]

    # 3. 在每个 ascend_pt 文件夹内查找csv 文件
    for idx, ascend_dir in enumerate(target_dirs):
        csv_path = None
        # 递归遍历目录找 csv
        for sub_root, _, sub_files in os.walk(ascend_dir):
            if csv_filename in sub_files:
                csv_path = os.path.join(sub_root, csv_filename)
                break  # 找到第一个就停止

        csv_path_list_func[idx] = csv_path
        logging.info("%s卡 profiling数据:%s", idx, csv_path)

    return csv_path_list_func


# 判断取值是否为数字，不是则取NA
def is_numeric(value_func):
    if pd.isna(value_func) or value_func in ["NA", "na", "N/A", ""]:
        return "NA"
    try:
        clean_val = re.sub(r"[^\d.-]", "", str(value_func).strip())
        return float(clean_val) if "." in clean_val else int(clean_val)
    except (ValueError, TypeError):
        return "NA"


# 以0卡为基准获取各个type以及对应的出现次数，并以此基准获取其他卡的对应数据
def get_card_data_by_cprofiling(csv_path_list_func):
    card_num_func = len(csv_path_list_func)
    if csv_path_list_func[0] is None or not os.path.exists(csv_path_list_func[0]):
        logging.error("0 卡文件不存在，无法建立基准")
        return {}
    try:
        df_0 = pd.read_csv(csv_path_list_func[0])
        for col in ["Type", "Duration(us)"]:
            if col not in df_0.columns:
                logging.error("0卡文件缺少列%s", col)
    except Exception as e:
        logging.error("读取0 卡文件失败:%s", e)
        return {}
    base_mapping = {}  # 核心基准 {(type, 次数)：0卡duration}
    type_count = {}  # 统计每个type的出现次数
    for _, row in df_0.iterrows():
        type_val = str(row["Type"]).strip()
        dur_0 = is_numeric(row["Duration(us)"])
        current_count = type_count.get(type_val, 0)
        type_count[type_val] = current_count + 1
        base_mapping[(type_val, current_count)] = dur_0
    result_dict = {}
    for (type_val, count), dur_0 in base_mapping.items():
        result_dict[(type_val, count)] = [dur_0] + ["NA"] * (card_num_func - 1)
    for card_num in range(1, card_num_func):
        card_path = csv_path_list_func[card_num]
        if card_path is None or not os.path.exists(card_path):
            logging.error(
                "%d卡的profiling数据文件不存在,对应数据用NA替代,跳过该卡的分析",
                card_num,
            )
            continue
        try:
            df_card = pd.read_csv(card_path)
        except Exception:
            logging.error(
                "%d卡的profiling数据文件读取失败,对应数据用NA替代,跳过该卡的分析",
                card_num,
            )
            continue
        for col in ["Type", "Duration(us)"]:
            if col not in df_card.columns:
                logging.error(
                    "%d卡的profiling数据文件缺少列%s,对应数据用NA替代,跳过该卡的分析",
                    card_num,
                    col,
                )
        current_type_count = {}
        card_map = {}
        for _, row in df_card.iterrows():
            t = str(row["Type"]).strip()
            dur = is_numeric(row["Duration(us)"])
            idx = current_type_count.get(t, 0)
            current_type_count[t] = idx + 1
            card_map[(t, idx)] = dur
        for key in result_dict:
            if key in card_map:
                result_dict[key][card_num] = card_map[key]

    return result_dict


# 计算最大最小值、抖动并导出
def get_all_data(result_dict):
    script_path = os.path.abspath(__file__)
    csv_filename = "profiling_all_data.csv"
    csv_abs_path = os.path.dirname(os.path.join(script_path, csv_filename))
    rows_func = []
    if not result_dict:
        logging.warning("无数据可导出")
        return
    max_card_num = len(next(iter(result_dict.values())))
    for (type_name, count), dur_list in result_dict.items():
        valid_vals = [v for v in dur_list if v != "NA"]
        if not valid_vals:
            max_val = min_val = del_num = "NA"
        else:
            max_val = max(valid_vals)
            min_val = min(valid_vals)
            del_num = round(max_val - min_val, 6)
        row = {
            "Type": type_name,
            "第n次调用": count,
            "max": max_val,
            "min": min_val,
            "抖动": del_num,
        }
        for i in range(max_card_num):
            row[f"卡{i}(us)"] = dur_list[i] if i < len(dur_list) else "NA"
        rows_func.append(row)
    df_all_data = pd.DataFrame(rows_func)
    df_all_data.to_csv("profiling_all_data.csv", index=False, encoding="gbk")
    logging.info("profiling详细数据已存储至%s", csv_abs_path)


# 计算每种type算子的平均抖动
def calculate_avg(result_dict):
    script_path = os.path.abspath(__file__)
    csv_filename = "profiling_avg_data.csv"
    csv_abs_path = os.path.dirname(os.path.join(script_path, csv_filename))
    logging.info("开始计算所有算子的平均抖动")
    type_total_count = {}
    logging.info("计算算子平均抖动时, 不计算算子第0次调用时的抖动值")
    for type_name, _ in result_dict.keys():
        if type_name not in type_total_count:
            type_total_count[type_name] = 0
        type_total_count[type_name] += 1
    type_del_map = {}
    for (type_name, count), dur_list in result_dict.items():
        if count == 0:
            continue
        valid_vals = [v for v in dur_list if v != "NA"]
        if type_name not in type_del_map:
            type_del_map[type_name] = []
        if len(valid_vals) > 0:
            del_num = max(valid_vals) - min(valid_vals)
            type_del_map[type_name].append(del_num)
    avg_row = []
    for type_name in type_total_count:
        total_cnt = type_total_count[type_name]
        del_num = type_del_map.get(type_name, [])
        if not del_num:
            avg_del_num = "NA"
        else:
            avg_del_num = round(np.mean(del_num), 6)
        avg_row.append(
            {
                "Type": type_name,
                "调用次数": total_cnt,
                "有效调用次数(去除第0次和NA项)": len(del_num),
                "平均抖动(us)": avg_del_num,
            }
        )
    df_avg = pd.DataFrame(avg_row)
    df_avg["排序"] = pd.to_numeric(df_avg["平均抖动(us)"], errors="coerce")
    df_avg = df_avg.sort_values(by="排序", ascending=False, na_position="last")
    df_avg = df_avg.drop(columns="排序")
    df_avg.to_csv("profiling_avg_data.csv", index=False, encoding="gbk")
    logging.info("所有算子的平均抖动已存储至%s", csv_abs_path)


# 检测指定算子的四项数据
def check_aiv_num(csv_path_list_func):
    aiv_cols = [
        "aiv_vec_time(us)",
        "aiv_scalar_time(us)",
        "aiv_mte2_time(us)",
        "aiv_mte3_time(us)",
    ]
    target_types = ["MoeDistributeDispatch", "MoeDistributeCombine"]
    logging.info("默认对 %s 算子进行上述四项数据分析", target_types)
    logging.info("跳过第0层进行检测")
    for card_id, csv_path in enumerate(csv_path_list_func):
        if csv_path_list_func[card_id] is None or not os.path.exists(
            csv_path_list_func[card_id]
        ):
            logging.error("%d 卡profiling数据不存在,跳过分析该卡数据", card_id)
            continue
        df = pd.read_csv(csv_path)
        if "Type" not in df.columns:
            logging.error("%d 卡profiling数据没有Type列,跳过该卡数据", card_id)
            continue
        df = df[df["Type"].isin(target_types)].copy()
        if df.empty:
            logging.error(
                "%d 卡profiling数据中没有%s类型,跳过该卡数据", card_id, target_types
            )
            continue
        df["第几次出现"] = df.groupby("Type").cumcount()
        type_avg = {}
        type_avg = calc_type_avg(df, aiv_cols, card_id, type_avg)
        check_data_range(df, aiv_cols, card_id, type_avg)


# 计算指定type的平均值
def calc_type_avg(df, aiv_cols, card_id, type_avg):
    for typ in df["Type"].unique():
        type_avg[typ] = {}
        type_df = df[(df["Type"] == typ) & (df["第几次出现"] > 0)]
        for col in aiv_cols:
            vaild_series = pd.to_numeric(type_df[col], errors="coerce")
            valid_vals = vaild_series[(vaild_series > 0)]
            type_avg[typ][col] = (
                round(valid_vals.mean(), 6) if not valid_vals.empty else "NA"
            )
            if valid_vals.empty:
                logging.error(
                    "%d 卡| Type=%s| 列名=%s| 无有效数据,平均值计算得到为NA",
                    card_id,
                    typ,
                    col,
                )
    return type_avg


# 校验数据
def check_data_range(df, aiv_cols, card_id, type_avg):
    for _, row in df.iterrows():
        current_type = row["Type"]
        seq_num = row["第几次出现"]
        # 跳过第0层
        if seq_num == 0:
            continue
        for col in aiv_cols:
            if col not in df.columns:
                logging.error("%d 卡数据中未找到%s列,跳过该列分析", card_id, col)
                continue
            value = row[col]
            if pd.isna(value):
                logging.error(
                    "%d 卡| Type=%s| 第%d次出现| 列名=%s| 值=NA",
                    card_id,
                    current_type,
                    seq_num,
                    col,
                )
                continue
            if not isinstance(value, (int, float)) or value <= 0:
                logging.error(
                    "%d 卡| Type=%s| 第%d次出现| 列名=%s| 值=%s | 非正数",
                    card_id,
                    current_type,
                    seq_num,
                    col,
                    str(value),
                )
                continue
            if current_type not in type_avg or col not in type_avg[current_type]:
                continue
            avg_val = type_avg[current_type][col]
            if avg_val is None or not isinstance(avg_val, (int, float)):
                continue
            lower, upper = avg_val * 0.5, avg_val * 1.5
            if not (lower <= value <= upper):
                logging.error(
                    "%d 卡| Type=%s| 第%d次出现| 列名=%s| 值=%.2f| "
                    "平均值=%.2f| 超出范围 %.2f - %.2f(平均值的±50%%)",
                    card_id,
                    current_type,
                    seq_num,
                    col,
                    value,
                    avg_val,
                    lower,
                    upper,
                )


if __name__ == "__main__":
    profiling_path = sys.argv[1]
    card_num = int(sys.argv[2])
    logging.info("开始进行profiling数据分析,指定profiling数据路径%s", profiling_path)
    csv_path_list = find_device_and_csv_by_num(profiling_path, card_num)
    result_profiling = get_card_data_by_cprofiling(csv_path_list)
    get_all_data(result_profiling)
    calculate_avg(result_profiling)
    logging.info(
        "开始对每张卡的profiling数据进行aiv_vec_time, aiv_scalar_time, aiv_mte2_time, aiv_mte3_time数据分析"
    )
    check_aiv_num(csv_path_list)
    logging.info("profiling数据分析完成")
