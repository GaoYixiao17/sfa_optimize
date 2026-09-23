#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import torch
import torch_npu
import check_valid_param
import pytest
import random
import numpy as np
import math
import custom_ops as ops

# 两种运行模式
# 0 不切S2
# 1 切S2

RUN_MODE = 1

DATA_RANGE_LEFT = -2
DATA_RANGE_RIGHT = 2
FP8_DATA_RANGE_LEFT = -1
FP8_DATA_RANGE_RIGHT = 1

FP32_FRACTION_BITS = 23        # fp32尾数位数

HIF8_EXP_ZERO_THRESHOLD = -23  # 边界值
HIF8_EXP_DML_MIN = -22         # DML最小指数
HIF8_EXP_DML_MAX = -15         # DML最大指数
HIF8_EXP_D0 = 0                # D0指数值
HIF8_EXP_D1_BOUNDARY = 1       # D1指数值
HIF8_EXP_D2_MIN, HIF8_EXP_D2_MAX = 2, 3   # D2指数范围
HIF8_EXP_D3_MIN, HIF8_EXP_D3_MAX = 4, 7   # D3指数范围
HIF8_EXP_D4_MIN, HIF8_EXP_D4_MAX = 8, 15  # D4指数范围

HIF8_DOT_DML = 0               # DML: Denormal Low, 指数范围 -22 ~ -16, 0位尾数
HIF8_DOT_D0 = 1                # D0: 指数为0，3位尾数（最高精度）
HIF8_DOT_D1 = 2                # D1: 指数为±1，3位尾数
HIF8_DOT_D2 = 4                # D2: 指数为±2 ~ ±3，3位尾数
HIF8_DOT_D3 = 8                # D3: 指数为±4 ~ ±7，2位尾数
HIF8_DOT_D4 = 12               # D4: 指数为±8 ~ ±15，1位尾数（最低精度）
HIF8_DOT_INVALID = -1          # 无效状态

HIF8_FRAC_BITS_DML = 0         # DML档位尾数位数
HIF8_FRAC_BITS_D0 = 3          # D0档位尾数位数
HIF8_FRAC_BITS_D1 = 3          # D1档位尾数位数
HIF8_FRAC_BITS_D2 = 3          # D2档位尾数位数
HIF8_FRAC_BITS_D3 = 2          # D3档位尾数位数
HIF8_FRAC_BITS_D4 = 1          # D4档位尾数位数

HIF8_EXP_BITS_DML = 3          # DML档位指数位数
HIF8_EXP_BITS_D0 = 0           # D0档位指数位数
HIF8_EXP_BITS_D1 = 1           # D1档位指数位数
HIF8_EXP_BITS_D2 = 2           # D2档位指数位数
HIF8_EXP_BITS_D3 = 3           # D3档位指数位数
HIF8_EXP_BITS_D4 = 4           # D4档位指数位数

HIF8_ZERO = 0
HIF8_NAN = 128                 # 0b10000000, NaN
HIF8_NEG_INF = 239             # 0b11101111, -inf
HIF8_NEG_MAX = 238             # 0b11101110, 负极大值
HIF8_POS_INF = 111             # 0b01101111, +inf
HIF8_POS_MAX = 110             # 0b01101110, 正极大值

HIF8_SIGN_MASK = 128           # 0b10000000, 符号位掩码
HIF8_DOT_MASK = 120            # 0b01110000, dot值掩码
HIF8_FRAC_MASK_3BIT = 7        # 0b00000111, 3位尾数掩码（D0/D1/D2）
HIF8_FRAC_MASK_2BIT = 3        # 0b00000011, 2位尾数掩码（D3）
HIF8_FRAC_MASK_1BIT = 1        # 0b00000001, 1位尾数掩码（D4）
HIF8_EXP_MASK_DML = 7          # 0b00000111, DML指数掩码（bit0-2）
HIF8_EXP_MASK_D4 = 30          # 0b00011110, D4指数掩码（bit1-4）
HIF8_EXP_MASK_D3 = 28          # 0b00011100, D3指数掩码（bit2-4）
HIF8_EXP_MASK_D2 = 24          # 0b00011000, D2指数掩码（bit3-4）
HIF8_EXP_SIGN_MASK_D1 = 8      # 0b00001000, D1指数掩码（bit3）

HIF8_DOT_BIT_SHIFT = 3         # Dot值在HiF8中的起始位置(bit3)
HIF8_DML_EXP_OFFSET = 23       # DML指数偏移值
HIF8_OVERFLOW_SCALE = 1.25     # 溢出阈值缩放因子
HIF8_MAX_FINITE_VALUE = 32768  # 最大有限值（非饱和模式下的边界值, 2^15

SSR_T14_MASK = 16383           # 0b0011 1111 1111 1111, 14位低位掩码
SSR_F14_OFFSET = 8192          # 0b0010 0000 0000 0000, F14偏移值
SSR_DML_SHIFT = 10             # SSR舍入移位值
SSR_RESERVED_BITS = 14         # SSR舍入保留位数
HYBRID_ROUND_EXP_THRESHOLD = 4 # 混合舍入的指数分界点

class GeneralizedSFAQuant:
    def __init__(self, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, block_num2,
                 block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                 ori_win_left, ori_win_right, kv_quant_mode, tile_size, rope_head_dim, template_run_mode):
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.q_type = q_type
        self.ori_kv_type = ori_kv_type
        self.cmp_kv_type = cmp_kv_type
        self.B = B
        self.S1 = S1
        self.T1 = T1
        self.N1 = N1
        self.N2 = N2
        self.D = D
        self.K = K
        self.block_num1 = block_num1
        self.block_num2 = block_num2
        self.block_size1 = block_size1
        self.block_size2 = block_size2
        self.cu_seqlens_q = cu_seqlens_q
        self.seqused_kv = seqused_kv
        self.softmax_scale = softmax_scale
        self.cmp_ratio = cmp_ratio
        self.ori_mask_mode = ori_mask_mode
        self.cmp_mask_mode = cmp_mask_mode
        self.ori_win_left = ori_win_left
        self.ori_win_right = ori_win_right
        self.kv_quant_mode = kv_quant_mode
        self.tile_size = tile_size
        self.rope_head_dim = rope_head_dim
        self.template_run_mode = template_run_mode

    def calculate_by_bnsd(self, q_bnsd, ori_k_bnsd, cmp_k_bnsd, cmp_sparse_indices_bnsd, cu_seqlens_q, seqused_kv, sinks):
        attn_out = torch.zeros(q_bnsd.shape, dtype=q_bnsd.dtype)
        B = q_bnsd.shape[0]
        act_q = prefix_sum_to_original(cu_seqlens_q)
        G = int(self.N1 / self.N2)
        s2_base_size = 128

        for i_B in range(B):
            print(f"i_B = {i_B}/{B}")
            cur_act_q = act_q[i_B]
            cur_ori_act_kv = seqused_kv[i_B]
            for i_N2 in range(self.N2):
                print(f"    i_N2 = {i_N2}/{self.N2}")
                cur_sinks = sinks[i_N2 * G:(i_N2 + 1) * G]
                cur_sinks_expand = cur_sinks.unsqueeze(1)
                for i_S1 in range(cur_act_q):
                    milestones = [int(cur_act_q * pct / 100) for pct in range(10, 101, 10)]
                    milestones = list(dict.fromkeys(milestones))
                    if i_S1 in milestones:
                        current_pct = (i_S1 / cur_act_q) * 100
                        print(f"      进度：{current_pct:.1f}% | 步数：{i_S1:>{len(str(cur_act_q))}}/{cur_act_q}")
                    if i_S1 < cur_act_q - cur_ori_act_kv:  # 根据 win_kv 判断行无效
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = torch.zeros([G, self.D], dtype=torch.float)
                        continue

                    if self.ori_mask_mode == 4:
                        ori_threshold = cur_ori_act_kv - cur_act_q + i_S1 + 1
                        ori_win_end = ori_threshold + self.ori_win_right
                        if self.ori_win_left == -1:
                            ori_win_start = 0
                        else:
                            ori_win_start = max(ori_threshold - self.ori_win_left - 1, 0)

                    cur_ori_k_bnsd = ori_k_bnsd[i_B, i_N2, ori_win_start:ori_win_end, :]

                    if self.template_run_mode == "SCFA" and cmp_sparse_indices_bnsd is not None:
                        topk_id = cmp_sparse_indices_bnsd[i_B, i_N2, i_S1, :]
                        empty_flag, cur_cmp_k = self.gather_cmp_kv(cmp_k_bnsd, topk_id, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q)
                    elif self.template_run_mode == "CFA":
                        empty_flag, cur_cmp_k = self.mask_cmp_kv(cmp_k_bnsd, i_B, i_N2, i_S1, cur_ori_act_kv, cur_act_q)
                    else:
                        empty_flag = True
                        cur_cmp_k = []
                    if cur_cmp_k == []:
                        cmp_s2_loop_time = 0
                        cur_cmp_k_fp32 = []
                    else:
                        cmp_s2_loop_time = math.ceil(cur_cmp_k.size(0) / s2_base_size)
                        cur_cmp_k_fp32 = cur_cmp_k.to(dtype=torch.float32)

                    cur_attn_out = attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :]
                    q_curr = q_bnsd[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :]
                    q_curr_fp32 = q_curr.to(dtype=torch.float32)
                    if RUN_MODE == 0:
                        if empty_flag:
                            k_concat = cur_ori_k_bnsd
                        else:
                            k_concat = torch.concat([cur_ori_k_bnsd, cur_cmp_k], dim=0)
                        k_concat_fp32 = k_concat.to(dtype=torch.float32)
                        v_concat_fp32 = k_concat_fp32.clone()

                        mm1_res = torch.matmul(q_curr_fp32, k_concat_fp32.T)
                        scale_res = mm1_res * self.softmax_scale
                        softmax_res, softmax_sum = self.sinks_softmax(scale_res, cur_sinks_expand)
                        softmax_res = softmax_res.to(q_bnsd.dtype).to(torch.float32)
                        mm2_res = torch.matmul(softmax_res, v_concat_fp32)
                        v2_res = torch.div(mm2_res, softmax_sum)
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = v2_res
                    elif RUN_MODE == 1:
                        ori_s2_loop_time = math.ceil(cur_ori_k_bnsd.size(0) / s2_base_size)
                        total_s2_loop_time = ori_s2_loop_time + cmp_s2_loop_time
                        cur_ori_k_bnsd_fp32 = cur_ori_k_bnsd.to(dtype=torch.float32)
                        row_sum = torch.empty((G), dtype=torch.float32).uniform_(1.0, 1.0)
                        row_max = torch.empty((G, 1), dtype=torch.float32)
                        row_max = cur_sinks

                        for i_S2 in range(total_s2_loop_time):
                            if i_S2 < ori_s2_loop_time: # ori_kv
                                if i_S2 < ori_s2_loop_time - 1:
                                    k_tile = cur_ori_k_bnsd_fp32[i_S2 * s2_base_size:(i_S2 + 1) * s2_base_size, :]
                                else:
                                    k_tile = cur_ori_k_bnsd_fp32[i_S2 * s2_base_size:, :]
                            else: # cmp_kv
                                if i_S2 < total_s2_loop_time - 1:
                                    k_tile = cur_cmp_k_fp32[(i_S2 - ori_s2_loop_time) * s2_base_size:(i_S2 - ori_s2_loop_time + 1) * s2_base_size, :]
                                else:
                                    k_tile = cur_cmp_k_fp32[(i_S2 - ori_s2_loop_time) * s2_base_size:, :]
                            v_tile = k_tile.clone()
                            mm1_res = torch.matmul(q_curr_fp32, k_tile.T)
                            scale_res = mm1_res * self.softmax_scale  # 外层for S1 循环，据实拷入数据，因此不需要mask

                            row_max_old = row_max.clone()
                            row_max_tmp = torch.max(scale_res, dim=1)[0]
                            # row_max_tmp = row_max_tmp.unsqueeze(1)
                            row_max = torch.max(row_max, row_max_tmp)
                            update_mul = torch.exp(row_max_old - row_max)
                            row_max_expand = row_max.unsqueeze(1)
                            update_mul_expand = update_mul.unsqueeze(1)

                            cur_softmax_res = torch.exp(scale_res - row_max_expand)
                            row_sum = update_mul * row_sum + torch.sum(cur_softmax_res, dim=1)
                            cur_softmax_res = cur_softmax_res.to(dtype=q_bnsd.dtype).to(dtype=torch.float)
                            cur_o = torch.matmul(cur_softmax_res, v_tile)
                            cur_attn_out = cur_attn_out * update_mul_expand + cur_o
                        row_sum_expand = row_sum.unsqueeze(1)
                        attn_out[i_B, i_N2 * G: (i_N2 + 1) * G, i_S1, :] = (cur_attn_out / row_sum_expand).to(dtype=q_bnsd.dtype)
        return attn_out

    def gather_cmp_kv(self, k_tensor, topk_id, i_B, i_N2, i_S1, cur_act_kv, cur_act_q, sparse_block_size=1):
        s2_sparse = list()
        cur_cmp_act_kv = math.floor(cur_act_kv / self.cmp_ratio)
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / self.cmp_ratio)
        valid_count = min(self.K, math.ceil(threshold / sparse_block_size))
        for i_valid in range(valid_count):
            cur_topk_id = topk_id[i_valid]

            if cur_topk_id == -1:
                break
            begin_idx = cur_topk_id * sparse_block_size
            end_idx = begin_idx + sparse_block_size if begin_idx + sparse_block_size <= cur_cmp_act_kv else cur_cmp_act_kv
            if begin_idx >= threshold:
                continue
            if end_idx <= threshold:
                s2_sparse.extend(np.arange(begin_idx, end_idx))
            else:
                s2_sparse.extend(np.arange(begin_idx, threshold))

        empty_flag = False
        if len(s2_sparse) == 0:
            k_sparse = []
            empty_flag = True
        else:
            k_sparse = k_tensor[i_B, i_N2, s2_sparse, :]
        return empty_flag, k_sparse

    def mask_cmp_kv(self, k_tensor, i_B, i_N2, i_S1, cur_act_kv, cur_act_q):
        threshold = 0
        if self.cmp_mask_mode == 3:
            threshold = (cur_act_kv - cur_act_q + i_S1 + 1) // self.cmp_ratio
        empty_flag = True
        k_sparse = []
        if threshold > 0:
            empty_flag = False
            k_sparse = k_tensor[i_B, i_N2, :threshold, :]
        return empty_flag, k_sparse

    def sinks_softmax(self, x, sinks):  # [G, S2] [G, 1]
        x = x.to(dtype=torch.float)
        x_concat = torch.cat([x, sinks], dim=1)
        x_max = x_concat.max(dim=-1, keepdims=True)[0]
        x_sub = x - x_max
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True) + torch.exp(sinks - x_max)
        return y, x_sum

    def trans_shape_to_bnsd(self, tensor, shape, layout, act_seq=None):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = shape[3]
            tensor = tensor.permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout in ["TND"]:
            T = shape[0]
            N = shape[1]
            D = shape[2]
            B = len(act_seq) - 1  # TND act_q is cumulative
            max_s1 = get_max_adjacent_diff(act_seq)
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            new_tensor = torch.zeros((B, N, max_s1, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:cur_act_seq, :] = tensor[t_start:t_end, n_index, :]
                t_start += cur_act_seq
            return new_tensor, [B, N, max_s1, D]
        else:
            return tensor, shape

    def trans_bnsd_to_target_layout(self, tensor, layout, act_seq=None):
        if layout in ["BSND"]:
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["TND"]:
            T = act_seq[-1]
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            output = torch.zeros((T, N, D), dtype=torch.float)
            t_start = 0
            act_seq_per_batch = prefix_sum_to_original(act_seq)
            for b_index in range(B):
                cur_act_seq = act_seq_per_batch[b_index]
                t_end = t_start + cur_act_seq
                if cur_act_seq == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :cur_act_seq, :]
                t_start += cur_act_seq
            return output
        else:
            return tensor

    def forward(self, q, ori_k_bnsd, cmp_k_bnsd, cmp_sparse_indices, cu_seqlens_q, seqused_kv, sinks):
        print("cpu执行中...")
        print(f"template_run_mode = {self.template_run_mode}")

        q_bnsd, q_bnsd_shape = self.trans_shape_to_bnsd(q, q.shape, self.layout_q, cu_seqlens_q)
        cmp_sparse_indices_bnsd = None
        cmp_sparse_indices_bnsd_shape = None
        if self.template_run_mode == "SCFA" and cmp_sparse_indices is not None:
            cmp_sparse_indices_bnsd, cmp_sparse_indices_bnsd_shape = self.trans_shape_to_bnsd(cmp_sparse_indices,
                                                                    cmp_sparse_indices.shape, self.layout_q, cu_seqlens_q)

        attn_out = self.calculate_by_bnsd(q_bnsd, ori_k_bnsd, cmp_k_bnsd, cmp_sparse_indices_bnsd, cu_seqlens_q,
                                         seqused_kv, sinks)

        attn_out = self.trans_bnsd_to_target_layout(attn_out, self.layout_q, cu_seqlens_q)
        return attn_out

def prefix_sum_to_original(cu_seqlens_q):
    """
    从前缀和张量反向计算出原始的非前缀和张量（替代原列表逻辑）

    Args:
        cu_seqlens_q (torch.Tensor): 形状为 [B+1] 的一维前缀和张量（元素为数字类型，如int/float）

    Returns:
        torch.Tensor: 原始的非前缀和张量，形状为 [B]（与原列表长度一致）

    Raises:
        TypeError: 输入非tensor/非一维tensor
        ValueError: tensor长度<2（无法计算差值）
    """
    # 1. 基础类型校验：必须是torch.Tensor
    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise TypeError(f"输入必须是torch.Tensor，当前类型：{type(cu_seqlens_q)}")

    # 2. 维度校验：必须是一维tensor（原列表对应一维）
    if cu_seqlens_q.ndim != 1:
        raise TypeError(f"输入必须是一维tensor，当前维度：{cu_seqlens_q.ndim}，形状：{cu_seqlens_q.shape}")

    # 3. 长度校验（前缀和tensor至少需2个元素才能反向计算）
    if len(cu_seqlens_q) < 2:
        raise ValueError(f"前缀和tensor长度需≥2，当前长度：{len(cu_seqlens_q)}")

    # 4. 核心逻辑：计算相邻元素差值（用tensor向量化运算替代循环，效率更高）
    # 原理：original_val[i] = cu_seqlens_q[i+1] - cu_seqlens_q[i]
    # 切片实现：cu_seqlens_q[1:] 取第2个到最后一个元素，cu_seqlens_q[:-1] 取第1个到倒数第2个元素
    original_tensor = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

    return original_tensor

def get_max_adjacent_diff(cu_seqlens_q):
    """
    计算前缀和列表中相邻元素（后-前）的最大差值

    Args:
        cu_seqlens_q (list): 长度为 B+1 的前缀和列表

    Returns:
        float/int: 相邻元素的最大差值；若列表长度<2，返回 None
    """
    # 边界检查：列表长度不足2时无相邻元素
    if len(cu_seqlens_q) < 2:
        return None

    # 初始化最大差值为第一个相邻对的差值
    max_diff = cu_seqlens_q[1] - cu_seqlens_q[0]

    # 遍历所有相邻元素对（从第2对开始）
    for i in range(1, len(cu_seqlens_q)-1):
        current_diff = cu_seqlens_q[i+1] - cu_seqlens_q[i]
        # 更新最大差值
        if current_diff > max_diff:
            max_diff = current_diff

    return max_diff

def gen_cmp_sparse_indices_bsnd(cmp_ratio, B, S1, N2, K, seqused_kv, cmp_mask_mode):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    cmp_sparse_indices = torch.full((B, S1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(S1):
                if cmp_mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - S1 + i_S1 + 1) / cmp_ratio)
                else:
                    raise ValueError(f"cmp_mask_mode only support 3, which is {cmp_mask_mode}")

                valid_blocks_max = max(0, cur_valid_s2_max)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                cmp_sparse_indices[i_B, i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
    return cmp_sparse_indices

def gen_cmp_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, seqused_kv, cmp_mask_mode):
    # 有效索引在叠加了causal后有效tokens中选取，不足sparse_block_count，尾部填充-1
    cmp_sparse_indices = torch.full((T1, N2, K), fill_value=-1, dtype=torch.int32)
    for i_B in range(B):
        cur_act_q = cu_seqlens_q[i_B + 1] - cu_seqlens_q[i_B]
        s1_prefix = cu_seqlens_q[i_B]
        cur_act_kv = seqused_kv[i_B]
        for i_N2 in range(N2):
            for i_S1 in range(cur_act_q):
                if cmp_mask_mode == 3:
                    cur_valid_s2_max = math.floor((cur_act_kv - cur_act_q + i_S1 + 1) / cmp_ratio)

                valid_blocks_max = max(0, cur_valid_s2_max)
                block_indices = torch.randperm(valid_blocks_max).to(torch.int32)
                valid_blocks_topk = min(valid_blocks_max, K)
                cmp_sparse_indices[s1_prefix + i_S1, i_N2, :valid_blocks_topk] = block_indices[0:valid_blocks_topk]
    return cmp_sparse_indices

def fp32_ta_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        return True, 0
    hif8_value_tmp = fraction32_int >> (FP32_FRACTION_BITS - (hif8_bits_num + 1))
    if hif8_value_tmp == pow(2, hif8_bits_num + 1) - 1:
        return True, 0
    elif hif8_value_tmp == 0:
        return False, 0
    elif hif8_value_tmp % 2 == 1:
        hif8_value_tmp += 1
        return False, hif8_value_tmp >> 1
    else:
        return False, hif8_value_tmp >> 1



def fp32_ssr_round_to_hif8(fraction32_int, hif8_bits_num, exponent):
    t14_mask = SSR_T14_MASK
    if exponent == HIF8_EXP_ZERO_THRESHOLD:
        f14_values = (fraction32_int >> SSR_DML_SHIFT) + SSR_F14_OFFSET
        t14_values = fraction32_int & t14_mask
        hif8_value = 0
    else:
        hif8_value = fraction32_int >> (FP32_FRACTION_BITS - hif8_bits_num)
        f14_t14 = fraction32_int - (hif8_value << (FP32_FRACTION_BITS - hif8_bits_num))
        f14_values = f14_t14 >> (FP32_FRACTION_BITS - hif8_bits_num - SSR_RESERVED_BITS)
        t14_values = f14_t14 & t14_mask
    if f14_values >= t14_values:
        if hif8_value == pow(2, hif8_bits_num) - 1:
            return True, 0
        else:
            hif8_value += 1
            return False, hif8_value
    else:
        return False, hif8_value

def get_hif8_fraction_bits_number(exponent):
    if exponent < HIF8_EXP_DML_MIN:
        return HIF8_DOT_INVALID, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if HIF8_EXP_DML_MIN <= exponent < HIF8_EXP_DML_MAX:
        return HIF8_DOT_DML, HIF8_EXP_BITS_DML, HIF8_FRAC_BITS_DML
    if exponent == HIF8_EXP_D0:
        return HIF8_DOT_D0, HIF8_EXP_BITS_D0, HIF8_FRAC_BITS_D0
    if abs(exponent) == HIF8_EXP_D1_BOUNDARY:
        return HIF8_DOT_D1, HIF8_EXP_BITS_D1, HIF8_FRAC_BITS_D1
    if HIF8_EXP_D2_MIN <= abs(exponent) <= HIF8_EXP_D2_MAX:
        return HIF8_DOT_D2, HIF8_EXP_BITS_D2, HIF8_FRAC_BITS_D2
    if HIF8_EXP_D3_MIN <= abs(exponent) <= HIF8_EXP_D3_MAX:
        return HIF8_DOT_D3, HIF8_EXP_BITS_D3, HIF8_FRAC_BITS_D3
    if HIF8_EXP_D4_MIN <= abs(exponent) <= HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_FRAC_BITS_D4
    if exponent > HIF8_EXP_D4_MAX:
        return HIF8_DOT_D4, HIF8_EXP_BITS_D4, HIF8_DOT_INVALID

def cvt_float32_to_hifuint8(x, round_mode = "round", over_mode = True):
    sign = False
    sign_int_value = 0
    x_abs = math.fabs(x)
    ec = 0
    over_value = HIF8_OVERFLOW_SCALE * pow(2.0, HIF8_EXP_D4_MAX + ec)
    if x < 0.0:
        sign = True
        sign_int_value = HIF8_SIGN_MASK
    if torch.isinf(x) or x_abs >= over_value:
        if sign:
            if over_mode:
                return HIF8_NEG_INF
            else:
                return HIF8_NEG_MAX
        else:
            if over_mode:
                return HIF8_POS_INF
            else:
                return HIF8_POS_MAX
    if torch.isnan(x):
        if over_mode:
            return HIF8_NAN
        else:
            return 0
    if x_abs == 0.0:
        return 0
    exponent = math.floor(math.log2(x_abs))
    if round_mode == "hybrid":
        if abs(exponent) < HYBRID_ROUND_EXP_THRESHOLD:
            cut_bit_type = "TA"
        else:
            cut_bit_type = "SSR"
    elif round_mode == "round":
        cut_bit_type = "TA"
    elif round_mode == "storound":
        cut_bit_type = "SSR"
    else:
        cut_bit_type = "TA"
    fraction_int = int(x_abs * pow(2, FP32_FRACTION_BITS) * pow(2, -exponent) - pow(2, FP32_FRACTION_BITS))
    dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits = get_hif8_fraction_bits_number(exponent)
    if cut_bit_type == "TA":
        carry_exp_status, hif8_frac_value = fp32_ta_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    elif cut_bit_type == "SSR":
        carry_exp_status, hif8_frac_value = fp32_ssr_round_to_hif8(fraction_int, fraction_hif8_bits, exponent)
    else:
        print(f"unknown round type")
        return 0

    if carry_exp_status:
        exponent += 1
        dot_hif8_value, exponent_hif8_bits, fraction_hif8_bits_new = get_hif8_fraction_bits_number(exponent)
        fraction_hif8_bits = fraction_hif8_bits_new
    if exponent < HIF8_EXP_ZERO_THRESHOLD:
        return 0
    if exponent < 0:
        sig_exp = 1
    else:
        sig_exp = 0
    if dot_hif8_value <= 0:
        if exponent <= HIF8_EXP_ZERO_THRESHOLD:
            return 0
        else:
            return sign_int_value + exponent + HIF8_DML_EXP_OFFSET
    elif dot_hif8_value == 1:
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + hif8_frac_value
    else:
        abs_exponent = abs(exponent)
        abs_exponent = abs_exponent - pow(2, exponent_hif8_bits - 1)
        exponent_int_value = abs_exponent << fraction_hif8_bits
        sig_exp = sig_exp << (exponent_hif8_bits - 1 + fraction_hif8_bits)
        dot_int_value = dot_hif8_value << HIF8_DOT_BIT_SHIFT
        hif8_int_value = sign_int_value + dot_int_value + sig_exp + exponent_int_value + hif8_frac_value
    return hif8_int_value

def trans_float_tensor_to_hifuint8(in_tensor, round_mode = "round", over_mode = True):
    tensor_shape = in_tensor.shape
    tensor_shape_size = in_tensor.numel()
    if tensor_shape_size == 1.0:
        tensor_shape_size = int(tensor_shape_size)
    
    out_tensor = torch.zeros(tensor_shape_size).to(torch.uint8)
    in_tensor = in_tensor.reshape(tensor_shape_size)
    for i in range(tensor_shape_size):
        out_tensor[i] = cvt_float32_to_hifuint8(in_tensor[i], round_mode, over_mode)
    out_tensor = out_tensor.view(torch.uint8)
    out_tensor = out_tensor.reshape(tensor_shape)
    return out_tensor

def cvt_hifuint8_to_float32(x, over_mode = True):
    x = int(x)
    if x == HIF8_ZERO:
        return float(0)
    elif x == HIF8_NAN:
        if over_mode:
            return float('nan')
        else:
            return float(0)
    elif x == HIF8_NEG_INF:
        if over_mode:
            return -torch.inf
        else:
            return -HIF8_MAX_FINITE_VALUE
    elif x == HIF8_POS_INF:
        if over_mode:
            return torch.inf
        else:
            return HIF8_MAX_FINITE_VALUE
    else:
        if x >= HIF8_NAN:
            sign = -1.0
        else:
            sign = 1.0
        dot_4_bits = x & HIF8_DOT_MASK
        dot_4_value = dot_4_bits >> 3
        if dot_4_value >= HIF8_DOT_D4:
            exponent = x & HIF8_EXP_MASK_D4
            exponent_int = exponent >> 1
            if exponent_int >= 8:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 8

            fra_int = x & HIF8_FRAC_MASK_1BIT
            m_value = 1.0 + fra_int * 0.5
        elif dot_4_value >= HIF8_DOT_D3:
            exponent = x & HIF8_EXP_MASK_D3
            exponent_int = exponent >> 2
            if exponent_int >= 4:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 4

            fra_int = x & HIF8_FRAC_MASK_2BIT
            m_value = 1.0 + fra_int * 0.25
        elif dot_4_value >= HIF8_DOT_D2:
            exponent = x & HIF8_EXP_MASK_D2
            exponent_int = exponent >> 3
            if exponent_int >= 2:
                exponent_value = -exponent_int
            else:
                exponent_value = exponent_int + 2

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value >= HIF8_DOT_D1:
            exponent = x & HIF8_EXP_SIGN_MASK_D1
            exponent_sign = exponent >> 3
            if exponent_sign >= 1:
                exponent_value = -1
            else:
                exponent_value = 1

            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_D0:
            exponent_value = 0
            fra_int = x & HIF8_FRAC_MASK_3BIT
            m_value = 1.0 + fra_int * 0.125
        elif dot_4_value == HIF8_DOT_DML:
            m_value = 1
            exponent_value = (x & HIF8_EXP_MASK_DML) - HIF8_DML_EXP_OFFSET
        else:
            print("error, dot error")
            m_value = 0.0
            exponent_value = 0
        return sign * pow(2.0, exponent_value) * m_value

def trans_hifuint8_tensor_to_float(in_tensor):
    tensor_shape = in_tensor.shape
    tensor_shape_size = in_tensor.numel()
    
    out_tensor = torch.zeros(tensor_shape_size).to(torch.float)
    in_tensor = in_tensor.reshape(tensor_shape_size)
    for i in range(tensor_shape_size):
        out_tensor[i] = cvt_hifuint8_to_float32(in_tensor[i])
    out_tensor = out_tensor.reshape(tensor_shape).to(torch.float)
    return out_tensor

def gen_ori_kv(q_type, ori_kv_type, B, N2, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_aligned_128,
               pad_d, block_num1, block_size1, ori_max_s2, ori_max_block_num_per_batch,
               seqused_kv, quant_param_range_left, quant_param_range_right, kv_quant_mode):
    # ori_kv处理流程
    if kv_quant_mode == 10:
        quant_param = random.uniform(quant_param_range_left, quant_param_range_right)
        quant_range_left = quant_param
        quant_range_right = quant_param
    else:
        quant_range_left = quant_param_range_left
        quant_range_right = quant_param_range_right
    ori_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_range_left, quant_range_right,
        (B, N2, ori_max_s2, quant_scale_head_dim))).to(q_type)
    ori_kv_quant_param_tensor = ori_kv_quant_param_tensor_npu.to(q_type)

    # 分别生成bf16类型的nope合rope矩阵，为了模拟量化的过程，得到准确的量化参数
    if kv_quant_mode == 10:
        ori_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
            (B, N2, ori_max_s2, nope_head_dim))).to(torch.float)
        ori_k_nope_bnsd_npu = trans_float_tensor_to_hifuint8(ori_k_nope_bnsd_npu,
            round_mode = "hybrid", over_mode = True)
        ori_k_nope_bnsd = trans_hifuint8_tensor_to_float(ori_k_nope_bnsd_npu).to(q_type)
    else:
        ori_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
            (B, N2, ori_max_s2, nope_head_dim))).to(torch.float8_e4m3fn)
        ori_k_nope_bnsd = ori_k_nope_bnsd_npu.to(q_type)

    ori_k_rope_bnsd = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, ori_max_s2, rope_head_dim))).to(q_type)

    # nope部分*scale，转成fp8，保存为bin文件，再转回bf16
    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range(tile_size):
            offset = d_loop * tile_size + tile_loop
            ori_k_nope_bnsd[:, :, :, offset:offset+1] = torch.mul(ori_k_nope_bnsd[:, :, :, offset:offset+1], ori_kv_quant_param_tensor[:, :, :, d_loop:d_loop+1])
    # bf16的nope+rope拼接在一起，得到ori_k_bnsd和cmp_k_bnsd
    ori_k_bnsd = torch.concat([ori_k_nope_bnsd, ori_k_rope_bnsd], dim=3)

    ori_pad_tensor = torch.tensor(np.random.uniform(0, 0, (B, N2, ori_max_s2, pad_d))).to(torch.float8_e8m0fnu)
    ori_k_bnsd_npu = torch.concat([ori_k_rope_bnsd.view(ori_kv_type), ori_k_nope_bnsd_npu, ori_kv_quant_param_tensor_npu.view(ori_kv_type), ori_pad_tensor.view(ori_kv_type)], dim=3)

    ori_block_num_per_batch = []
    ori_block_num_sum = 0

    for cur_ori_act_kv in seqused_kv:
        cur_ori_kv_block_num = math.ceil(cur_ori_act_kv / block_size1)
        ori_block_num_per_batch.append(cur_ori_kv_block_num)
        ori_block_num_sum += cur_ori_kv_block_num

    if block_num1 < ori_block_num_sum:
        raise ValueError(f"ori_kv actual_block_num < needed_block_num, which is {block_num1 < ori_block_num_sum}")

    ori_block_id_list = np.arange(block_num1)
    ori_block_id_list = np.random.permutation(ori_block_id_list).astype(np.int32)
    cur_block_id = 0
    ori_block_table = np.full((B, ori_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
    batch_idx = 0
    for cur_block_id_threshold in ori_block_num_per_batch:
        for i_block_id in range(cur_block_id_threshold):
            ori_block_table[batch_idx][i_block_id] = ori_block_id_list[cur_block_id]
            cur_block_id += 1
        batch_idx += 1

    # [B, S, N, D] expand to [B, ori_max_block_num_per_batch * block_size1, N, D]
    ori_k_expand = torch.zeros((B, N2, ori_max_block_num_per_batch * block_size1, d_aligned_128), dtype=ori_kv_type)
    ori_k_expand[:, :, :ori_max_s2, :] = ori_k_bnsd_npu
    ori_k_in_pa_shape = torch.zeros((block_num1, block_size1, N2, d_aligned_128), dtype=ori_kv_type)

    for i_B in range(B):
        for i_block, cur_block_id in enumerate(ori_block_table[i_B]):
            block_start_pos = i_block * block_size1
            if cur_block_id == -1:
                continue
            else:
                for i_N2 in range(N2):
                    ori_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                        ori_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size1, :]

    ori_block_table = torch.tensor(ori_block_table).to(torch.int32)

    ori_v_bnsd = ori_k_bnsd.clone()
    ori_v_in_pa_shape = ori_k_in_pa_shape.clone()

    return ori_k_bnsd, ori_k_in_pa_shape, ori_block_table

def gen_cmp_kv(q_type, layout_q, cmp_kv_type, B, S1, T1, N2, D, K, rope_head_dim, nope_head_dim, tile_size,
               quant_scale_head_dim, d_aligned_128, pad_d, block_num2, block_size2, cmp_max_s2,
               cmp_max_block_num_per_batch, cu_seqlens_q, seqused_kv, cmp_ratio, cmp_mask_mode, template_run_mode,
               quant_param_range_left, quant_param_range_right, kv_quant_mode):
    if cmp_max_s2 == 0:
        return None, None, None, None
    # cmp kv处理
    if kv_quant_mode == 10:
        quant_param = random.uniform(quant_param_range_left, quant_param_range_right)
        quant_range_left = quant_param
        quant_range_right = quant_param
    else:
        quant_range_left = quant_param_range_left
        quant_range_right = quant_param_range_right
    cmp_kv_quant_param_tensor_npu = torch.tensor(np.random.uniform(quant_range_left, quant_range_right,
        (B, N2, cmp_max_s2, quant_scale_head_dim))).to(q_type)
    cmp_kv_quant_param_tensor = cmp_kv_quant_param_tensor_npu.to(q_type)

    if kv_quant_mode == 10:
        cmp_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
            (B, N2, cmp_max_s2, nope_head_dim))).to(torch.float)
        cmp_k_nope_bnsd_npu = trans_float_tensor_to_hifuint8(cmp_k_nope_bnsd_npu,
            round_mode = "hybrid", over_mode = True)
        cmp_k_nope_bnsd = trans_hifuint8_tensor_to_float(cmp_k_nope_bnsd_npu).to(q_type)
    else:
        cmp_k_nope_bnsd_npu = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
            (B, N2, cmp_max_s2, nope_head_dim))).to(torch.float8_e4m3fn)
        cmp_k_nope_bnsd = cmp_k_nope_bnsd_npu.to(q_type)

    cmp_k_rope_bnsd = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT,
        (B, N2, cmp_max_s2, rope_head_dim))).to(q_type)

    for d_loop in range(quant_scale_head_dim):
        for tile_loop in range (tile_size):
            offset = d_loop * tile_size + tile_loop
            cmp_k_nope_bnsd[:, :, :, offset:offset+1] = torch.mul(cmp_k_nope_bnsd[:, :, :, offset:offset+1], cmp_kv_quant_param_tensor[:, :, :, d_loop:d_loop+1])
    cmp_k_bnsd = torch.concat([cmp_k_nope_bnsd, cmp_k_rope_bnsd], dim=3)

    cmp_pad_tensor = torch.tensor(np.random.uniform(0, 0, (B, N2, cmp_max_s2, pad_d))).to(torch.float8_e8m0fnu)
    cmp_k_bnsd_npu = torch.concat([cmp_k_rope_bnsd.view(cmp_kv_type), cmp_k_nope_bnsd_npu, cmp_kv_quant_param_tensor_npu.view(cmp_kv_type), cmp_pad_tensor.view(cmp_kv_type)], dim=3)

    cmp_block_num_per_batch = []
    cmp_block_num_sum = 0

    for cur_ori_act_kv in seqused_kv:
        cur_cmp_act_kv = math.floor(cur_ori_act_kv / cmp_ratio)
        cur_cmp_kv_block_num = math.ceil(cur_cmp_act_kv / block_size2)
        cmp_block_num_per_batch.append(cur_cmp_kv_block_num)
        cmp_block_num_sum += cur_cmp_kv_block_num

    if block_num2 < cmp_block_num_sum:
        raise ValueError(f"cmp_kv actual_block_num < needed_block_num, which is {block_num2 < cmp_block_num_sum}")

    cmp_block_id_list = np.arange(block_num2)
    cmp_block_id_list = np.random.permutation(cmp_block_id_list).astype(np.int32)
    cur_block_id = 0
    cmp_block_table = np.full((B, cmp_max_block_num_per_batch), fill_value=-1, dtype=np.int32)
    batch_idx = 0
    for cur_block_id_threshold in cmp_block_num_per_batch:
        for i_block_id in range(cur_block_id_threshold):
            cmp_block_table[batch_idx][i_block_id] = cmp_block_id_list[cur_block_id]
            cur_block_id += 1
        batch_idx += 1

    # [B, S, N, D] expand to [B, cmp_max_block_num_per_batch * block_size1, N, D]
    cmp_k_expand = torch.zeros((B, N2, cmp_max_block_num_per_batch * block_size2, d_aligned_128), dtype=cmp_kv_type)
    cmp_k_expand[:, :, :cmp_max_s2, :] = cmp_k_bnsd_npu
    cmp_k_in_pa_shape = torch.zeros((block_num2, block_size2, N2, d_aligned_128), dtype=cmp_kv_type)

    for i_B in range(B):
        for i_block, cur_block_id in enumerate(cmp_block_table[i_B]):
            block_start_pos = i_block * block_size2
            if cur_block_id == -1:
                continue
            else:
                for i_N2 in range(N2):
                    cmp_k_in_pa_shape[cur_block_id, :, i_N2, :] = \
                        cmp_k_expand[i_B, i_N2, block_start_pos:block_start_pos + block_size2, :]

    # generate cmp_sparse_indices
    cmp_sparse_indices = None  # CFA
    if template_run_mode == "SCFA" and cmp_max_s2 != 0:
        if layout_q == "BSND":
            cmp_sparse_indices = gen_cmp_sparse_indices_bsnd(cmp_ratio, B, S1, N2, K, seqused_kv, cmp_mask_mode)
        elif layout_q == "TND":
            cmp_sparse_indices = gen_cmp_sparse_indices_tnd(cmp_ratio, B, T1, N2, K, cu_seqlens_q, seqused_kv, cmp_mask_mode)

    cmp_block_table = torch.tensor(cmp_block_table).to(torch.int32)
    cmp_v_bnsd = cmp_k_bnsd.clone()
    cmp_v_in_pa_shape = cmp_k_in_pa_shape.clone()

    return cmp_k_bnsd, cmp_k_in_pa_shape, cmp_block_table, cmp_sparse_indices

def save_test_case(input_data, output_dir):
    """
    保存单条测试用例到文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    case_name = input_data['Testcase_Name']
    
    # 生成文件名
    input_filename = f"sas_case_{case_name}.pt"
    input_filepath = os.path.join(output_dir, input_filename)
    
    # 保存数据
    torch.save(input_data, input_filepath)
    print(f"测试用例已保存到: {input_filepath}")
    
    return input_filepath

def generate_and_save_testdata(params, save_pt=False, save_path=""):
    '''
    生成input param及cpuout
    runNpu: 生成完毕后执行npu计算
    return test_data
    '''

    Testcase_Name, layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K, block_num1, block_num2, \
    block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, \
    ori_win_right, kv_quant_mode, tile_size, rope_head_dim, template_run_mode = params

    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    seqused_kv = torch.tensor(seqused_kv).to(torch.int32)

    # generate q
    if layout_q == "BSND":
        q = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT, (B, S1, N1, D))).to(q_type)
        act_q = B * [S1]
    elif layout_q == "TND":
        q = torch.tensor(np.random.uniform(DATA_RANGE_LEFT, DATA_RANGE_RIGHT, (T1, N1, D))).to(q_type)
        if len(cu_seqlens_q) != (B + 1):
            raise ValueError(f"len(cu_seqlens_q) != B + 1, which is {len(cu_seqlens_q)} != {B + 1}")
        else:
            act_q = prefix_sum_to_original(cu_seqlens_q)
            max_s1 = get_max_adjacent_diff(cu_seqlens_q)
    else:
        raise ValueError(f"layout_q is not support {layout_q}")

    # generate ori_kv/cmp_kv (only support PA_ND)
    if layout_kv == "PA_ND":
        pass
    else:
        raise ValueError(f"layout_kv is not support {layout_kv}")

    if len(seqused_kv) != B:
        raise ValueError(f"len(seqused_kv) != B, which is {len(seqused_kv)} != {B}")
    else:
        ori_max_s2 = max(seqused_kv)
        ori_max_block_num_per_batch = math.ceil(ori_max_s2 / block_size1)

        cmp_max_s2 = math.floor(ori_max_s2 / cmp_ratio)
        cmp_max_block_num_per_batch = math.ceil(cmp_max_s2 / block_size2)

    if kv_quant_mode != 1 and kv_quant_mode != 10:
        raise ValueError(f"input kv_quant_mode = {kv_quant_mode}, only support 1 and 10")

    # 计算kv每个区域D轴长度
    nope_head_dim = D - rope_head_dim
    quant_scale_head_dim = (nope_head_dim + tile_size - 1) // tile_size
    d_aligned_128 = (nope_head_dim + rope_head_dim * 2 + quant_scale_head_dim * 2 + 127) // 128 * 128
    d_combined = nope_head_dim + rope_head_dim * 2 + quant_scale_head_dim
    print(f"d_aligned_128={d_aligned_128}, nope_head_dim={nope_head_dim}, rope_head_dim={rope_head_dim}, quant_scale_head_dim={quant_scale_head_dim}")
    pad_d = d_aligned_128 - nope_head_dim - rope_head_dim * 2 - quant_scale_head_dim * 2
    block_num = block_num1 if block_num1 >= block_num2 else block_num2
    # 根据输入的data range，计算scale范围，生成scale tensor，取倒数保存为bin
    quant_param_range_left = DATA_RANGE_LEFT / FP8_DATA_RANGE_LEFT
    quant_param_range_right = DATA_RANGE_RIGHT / FP8_DATA_RANGE_RIGHT

    # generate sinks tensor
    sinks = torch.tensor(np.random.uniform(DATA_RANGE_LEFT/10, DATA_RANGE_RIGHT/10, (N1))).to(torch.float)

    # generate ori_kv tensor
    ori_k_bnsd, ori_k_in_pa_shape, ori_block_table = gen_ori_kv(q_type, ori_kv_type, B, N2, rope_head_dim,
        nope_head_dim, tile_size, quant_scale_head_dim, d_aligned_128, pad_d, block_num, block_size1, ori_max_s2,
        ori_max_block_num_per_batch, seqused_kv, quant_param_range_left, quant_param_range_right, kv_quant_mode)

    # generate cmp_kv and sparse_indices
    if template_run_mode == "CFA" or template_run_mode == "SCFA":
        cmp_k_bnsd, cmp_k_in_pa_shape, cmp_block_table, cmp_sparse_indices = gen_cmp_kv(q_type, layout_q, cmp_kv_type,
            B, S1, T1, N2, D, K, rope_head_dim, nope_head_dim, tile_size, quant_scale_head_dim, d_aligned_128, pad_d,
            block_num, block_size2, cmp_max_s2, cmp_max_block_num_per_batch, cu_seqlens_q, seqused_kv, cmp_ratio,
            cmp_mask_mode, template_run_mode, quant_param_range_left, quant_param_range_right, kv_quant_mode)
    else:
        cmp_k_in_pa_shape = None
        cmp_sparse_indices = None
        cmp_block_table = None
        cmp_k_bnsd = None

    if layout_kv == "PA_BNBD" and (template_run_mode == "CFA" or template_run_mode == "SCFA"):
        fusionBlock = torch.zeros((block_num, block_size1 + block_size2, N2, d_combined + pad_d), dtype=ori_kv_type)
        fusionBlock.view(block_num, -1)[:, : block_size1 * N2 * (d_combined + pad_d)] = ori_k_in_pa_shape.view(block_num, -1)
        fusionBlock.view(block_num, -1)[:, block_size1 * N2 * (d_combined + pad_d) :] = cmp_k_in_pa_shape.view(block_num, -1)
        fusionBlock.npu()
        ori_k_in_pa_shape = fusionBlock.view(block_num, -1)[:, : block_size1 * N2 * (d_combined + pad_d)].view(block_num, block_size1, N2, d_combined + pad_d)
        cmp_k_in_pa_shape = fusionBlock.view(block_num, -1)[:, block_size1 * N2 * (d_combined + pad_d) :].view(block_num, block_size2, N2, d_combined + pad_d)

    test_sas = GeneralizedSFAQuant(layout_q, layout_kv, q_type, ori_kv_type, cmp_kv_type, B, S1, T1, N1, N2, D, K,
        block_num1, block_num2, block_size1, block_size2, cu_seqlens_q, seqused_kv, softmax_scale, cmp_ratio,
        ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right, kv_quant_mode, tile_size, rope_head_dim, template_run_mode)
    cpu_result = test_sas.forward(q, ori_k_bnsd, cmp_k_bnsd, cmp_sparse_indices, cu_seqlens_q, seqused_kv, sinks)
    
    print("mode:%s\n",template_run_mode)

    cu_seqlens_q = torch.tensor(cu_seqlens_q).to(torch.int32)
    seqused_kv = torch.tensor(seqused_kv).to(torch.int32)
    max_seqlen_q = S1
    if layout_q == "TND":
        max_seqlen_q = cu_seqlens_q.max().item()
    else:
        cu_seqlens_q = None
        max_seqlen_q = S1
    max_seqlen_kv = seqused_kv.max().item()

    # 保存pt
    input_data = {
        # 命名用变量
        'Testcase_Name':Testcase_Name,

        # 固定参数
        'params': params,

        # 输入张量
        'metadata_input': {
            'num_heads_q': N1,
            'num_heads_kv': N2,
            'head_dim': D,
            'cu_seqlens_q': cu_seqlens_q, 
            'seqused_kv': seqused_kv,    
            'batch_size': B,
            'max_seqlen_q': max_seqlen_q,
            'max_seqlen_kv': max_seqlen_kv,
            'topk': K if template_run_mode == "SCFA" else 0, # 仅SCFA需要
            'cmp_ratio': cmp_ratio if template_run_mode != "SWA" else 1,  # 仅SWA 不需要
            'ori_mask_mode': ori_mask_mode,
            'cmp_mask_mode': cmp_mask_mode,
            'ori_win_left': ori_win_left,
            'ori_win_right': ori_win_right,
            'layout_q': layout_q,
            'layout_kv': layout_kv,
            'has_ori_kv': True,
            'has_cmp_kv': False if template_run_mode == "SWA" else True,  # SWA 为false
            'kv_quant_mode': kv_quant_mode,
        },

        'input': {
            'q': q,
            'ori_kv': ori_k_in_pa_shape,
            'cmp_kv': cmp_k_in_pa_shape,
            'cmp_sparse_indices': cmp_sparse_indices,
            'ori_block_table': ori_block_table,
            'cmp_block_table': cmp_block_table,
            'cu_seqlens_q': cu_seqlens_q,
            'seqused_kv': seqused_kv,
            'sinks': sinks,
            'kv_quant_mode': kv_quant_mode,
            'tile_size': 64,
            'rope_head_dim': 64,
            'softmax_scale': softmax_scale,
            'cmp_ratio': cmp_ratio if template_run_mode != "SWA" else 1,
            'ori_mask_mode': ori_mask_mode,
            'cmp_mask_mode': cmp_mask_mode,
            'ori_win_left': ori_win_left,
            'ori_win_right': ori_win_right,
            'layout_q': layout_q,
            'layout_kv': layout_kv
        },

        'cpu_output': cpu_result
    }

    if save_pt:  
        save_test_case(input_data, save_path)
        
    return input_data
