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

import test
import torch
import torch_npu
import pytest
import random
import numpy as np
import math
import ctypes
import copy
import custom_ops as ops

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

class GeneralizedQLI:
    def __init__(self, batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, qk_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode, key_quant_mode, layout_query, layout_key, sparse_count, sparse_mode, cmp_ratio):
        self.batch_size = batch_size
        self.q_seq = q_seq
        self.k_seq = k_seq
        self.q_t_size = q_t_size
        self.k_t_size = k_t_size
        self.q_head_num = q_head_num
        self.k_head_num = k_head_num
        self.group_size = q_head_num // k_head_num
        self.head_dim = head_dim
        self.block_size = block_size
        self.block_num = block_num
        self.qk_dtype = qk_dtype
        self.dequant_dtype = dequant_dtype
        self.actual_seq_dtype = actual_seq_dtype
        self.act_seq_q = act_seq_q
        self.act_seq_k = act_seq_k
        self.query_quant_mode = query_quant_mode
        self.key_quant_mode = key_quant_mode
        self.layout_query = layout_query
        self.layout_key = layout_key
        self.sparse_count = sparse_count
        self.sparse_mode = sparse_mode
        self.cmp_ratio = cmp_ratio
        self.w_dtype = dequant_dtype

        if layout_query == "BSND":
            self.q_shape = [batch_size, q_seq, q_head_num, head_dim]
            self.w_shape = [batch_size, q_seq, q_head_num]
            self.q_tnd_flag = 0
        elif layout_query == "TND":
            self.q_shape = [q_t_size, q_head_num, head_dim]
            self.w_shape = [q_t_size, q_head_num]
            self.q_tnd_flag = 1

        if layout_key == "BSND":
            self.k_shape = [batch_size, k_seq, k_head_num, head_dim]
        elif layout_key == "TND":
            self.k_shape = [k_t_size, k_head_num, head_dim]

        if layout_query == "BSND":
            self.out_shape = [batch_size, q_seq, k_head_num, sparse_count]
        elif layout_query == "TND":
            self.out_shape = [q_t_size, k_head_num, sparse_count]

    def cal_atten_bnsd(self):
        batch_size = self.batch_size
        qs = self.q_seq
        ks = self.k_seq
        n1 = self.q_head_num
        n2 = self.k_head_num
        actualSeqLengths_q = self.actual_seq_lengths_query
        actualSeqLengths_k = self.actual_seq_lengths_key
        q_bnsd_tensor = self.q_bnsd_tensor
        k_bnsd_tensor = self.k_bnsd_tensor
        wt_bnsd_tensor = self.wt_bnsd_tensor
        mask_tensor = self.m_tensor
        q_scale_bnsd_tensor = self.q_scale_bnsd_tensor
        k_scale_bnsd_tensor = self.k_scale_bnsd_tensor
        cmp_ratio = self.cmp_ratio

        out_shape_bnsd = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnsd[1] = n2
        out_shape_bnsd[-1] = self.sparse_count

        out_shape_bnss = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnss[1] = n2
        out_shape_bnss[-1] = math.floor(max(actualSeqLengths_k)/cmp_ratio)

        y = torch.full(out_shape_bnsd,-1,dtype = torch.int32)
        y_value = torch.full(out_shape_bnss,-float('inf'), dtype=torch.float32)

        for b_idx in range(batch_size):
            curr_actualSeq_q = actualSeqLengths_q[b_idx]
            curr_actualSeq_k = math.floor(actualSeqLengths_k[b_idx] /cmp_ratio)
            self.cur_actseq_q = curr_actualSeq_q
            self.cur_actseq_k = curr_actualSeq_k
            self.cur_q = q_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_k = k_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_k, :]
            self.cur_wt = wt_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_q_scale = q_scale_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :]
            self.cur_k_scale = k_scale_bnsd_tensor[b_idx:(b_idx + 1), :, :curr_actualSeq_k]
            if self.sparse_mode != 0:
                self.cur_m = mask_tensor[b_idx:(b_idx + 1), :curr_actualSeq_q, :curr_actualSeq_k]
            self.cur_b_idx = b_idx

            if curr_actualSeq_q != 0:
                actual_selected_count = min(curr_actualSeq_k, self.sparse_count)
                if self.qk_dtype == torch.int8:
                    y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_int8(b_idx)
                elif self.qk_dtype == torch.float8_e4m3fn:
                    y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_fp8(b_idx)
                elif self.qk_dtype == torch.uint8:
                    y[b_idx:(b_idx + 1), :, :curr_actualSeq_q, :actual_selected_count], y_value[b_idx:(b_idx + 1), :,
                                                                                        :curr_actualSeq_q,
                                                                                        :curr_actualSeq_k] = self.cal_atten_per_batch_hifp8(b_idx)
            else:
                pass
        return y, y_value

    def trans_shape_to_bnsd(self, tensor, shape, layout, headnums=None, act_seq=None, is_weights=False, tensor_name=None):
        if layout in ["BSND"]:
            B = shape[0]
            S = shape[1]
            N = shape[2]
            D = 1
            if is_weights:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = shape[3]
            tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
            return tensor, [B, N, S, D]
        elif layout == "BSN":
            print("shape", shape)
            B = shape[0]
            S = shape[1]
            N = shape[2]
            if is_weights:
                D = 1
                tensor = torch.unsqueeze(tensor, dim=-1)  # 补D轴
                tensor = tensor.reshape(B, S, N, D).permute(0, 2, 1, 3)
                return tensor, [B, N, S, D]
            else:
                tensor = tensor.reshape(B, S, N).permute(0, 2, 1)
                return tensor, [B, N, S]
        elif layout in ["TND"]:
            T = shape[0]
            N = shape[1]
            D = 1
            if is_weights:
                tensor = torch.unsqueeze(tensor, dim=-1)
            else:
                D = shape[2]
            B = len(act_seq)
            S = max(act_seq)
            new_tensor = torch.zeros((B, N, S, D), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
                t_start += act_s
            return new_tensor, [B, N, S, D]
        elif layout == "TN":
            T = shape[0]
            N = shape[1]
            D = 1
            B = len(act_seq)
            S = max(act_seq)
            new_tensor = torch.zeros((B, N, S), dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_seq[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    new_tensor[b_index, n_index, 0:act_s] = tensor[t_start:t_end, n_index]
                t_start += act_s
            return new_tensor, [B, N, S]
        else:
            return tensor, shape

    def trans_tnd_actseq(self,list):
        list_len = len(list)
        if list_len == 0:
            raise ValueError(f'TND情况下 act_seq需要必传')
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(f'TND情况下 act_seq_len 为非递减数列 act_seq_len={list}')
        return list_new

    def cal_atten_per_batch_hifp8(self,b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_wt = self.cur_wt.to(dtype=torch.float32)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float32)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float32)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        cur_q = trans_hifuint8_tensor_to_float(cur_q)
        cur_k = trans_hifuint8_tensor_to_float(cur_k)
        qk_bmm_res = torch.bmm(
            cur_q.squeeze(0),
            cur_k.permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
        cur_w = cur_wt * cur_q_scale
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)
        brc_vmul = torch.bmm(
            cur_w.permute(0,2,3,1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0,2,1,3).to(dtype = torch.float32).squeeze(0)
        ).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.bfloat16)
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype = torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device = row_ele.device)

                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele),device = row_ele.device)
                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, :] = sorted_idx[:,0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def cal_atten_per_batch_fp8(self,b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_wt = self.cur_wt.to(dtype=torch.float32)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float32)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float32)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype = torch.float32).squeeze(0),
            cur_k.to(dtype = torch.float32).permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
        cur_w = cur_wt * cur_q_scale
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32)).clamp_min(0.0)
        brc_vmul = torch.bmm(
            cur_w.permute(0,2,3,1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0,2,1,3).to(dtype = torch.float32).squeeze(0)
        ).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf
        to_be_sort_ele = reduce_sum.clone()
        to_be_sort_ele = to_be_sort_ele.to(torch.bfloat16)
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype = torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device = row_ele.device)

                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele),device = row_ele.device)
                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, :] = sorted_idx[:,0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def cal_atten_per_batch_int8(self,b_idx):
        cur_q = self.cur_q
        cur_k = self.cur_k
        cur_wt = self.cur_wt.to(dtype=torch.float16)
        cur_q_scale = self.cur_q_scale.to(dtype=torch.float16)
        cur_k_scale = self.cur_k_scale.to(dtype=torch.float16)
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        cmp_ratio = self.cmp_ratio
        qk_bmm_res = torch.bmm(
            cur_q.to(dtype = torch.int32).squeeze(0),
            cur_k.to(dtype = torch.int32).permute(0, 1, 3, 2).squeeze(0)
        ).unsqueeze(0)
        cur_w = cur_wt * cur_q_scale
        qk_relu_out = (qk_bmm_res.to(dtype=torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
        brc_vmul = torch.bmm(
            cur_w.permute(0,2,3,1).to(dtype=torch.float32).squeeze(0),
            qk_relu_out.permute(0,2,1,3).to(dtype = torch.float32).squeeze(0)
        ).unsqueeze(0)
        temp_b, temp_s1, temp_n1, temp_s2 = brc_vmul.shape
        temp_g = self.group_size
        temp_n2 = self.k_head_num
        temp_b_idx = self.cur_b_idx
        actual_selected_count = min(temp_s2, sparse_count)
        reduce_sum = brc_vmul.reshape(temp_b, temp_n2, temp_s1, temp_s2)
        reduce_sum[0, :, :, :] = reduce_sum[0, :, :, :] * cur_k_scale

        if sparse_mode == 3:
            cur_m = self.cur_m
            cur_m_broadcasted = cur_m.reshape(1, 1, temp_s1, temp_s2)
            cur_m_broadcasted = torch.broadcast_to(cur_m_broadcasted, (1, temp_n2, temp_s1, temp_s2))
            # 根据布尔矩阵置-inf
            reduce_sum[cur_m_broadcasted.to(dtype = torch.bool)] = -torch.inf


        to_be_sort_ele = reduce_sum.clone()
        # 稳定排序
        b_sorted_indices = torch.full(to_be_sort_ele.shape, -1, dtype=torch.int32)
        if sparse_mode == 3:
            for i in range(temp_s1):
                row_mask = cur_m_broadcasted[0, 0, i, :].to(dtype = torch.bool)
                true_indices = torch.where(~row_mask)[0]
                row_ele = to_be_sort_ele[0, 0, i, true_indices]
                indices = torch.arange(len(row_ele), device = row_ele.device)

                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, true_indices] = true_indices[sorted_idx[:, 0]].to(torch.int32)
        else:
            for i in range(temp_s1):
                row_ele = to_be_sort_ele[0, 0, i, :]
                indices = torch.arange(len(row_ele),device = row_ele.device)
                sorted_vals, sorted_idx = torch.sort(
                    torch.stack([-row_ele, indices],dim=1),
                    dim=0,
                    stable=True
                )
                b_sorted_indices[0, 0, i, :] = sorted_idx[:,0]
        topk_indices = b_sorted_indices[..., :actual_selected_count]
        return topk_indices, to_be_sort_ele

    def trans_bnsd_to_layout(self,tensor, shape, layout, act_q=None):
        # 此时的输出D轴是K轴
        if layout == "BSH":
            output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
            return output
        elif layout == "BSND":
            output = tensor.permute(0, 2, 1, 3).contiguous()
            return output
        elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
            output = tensor.permute(1, 0, 2, 3).contiguous()
            return output
        elif layout in ["TND", "TND_NTD"]:
            T = sum(act_q)
            B = tensor.shape[0]
            N = tensor.shape[1]
            D = tensor.shape[3]
            output = torch.full(size=(T, N, D), fill_value=-1, dtype=tensor.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = act_q[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
                t_start += act_s
            if layout == "TND_NTD":
                output = output.permute(1, 0, 2).contiguous()
            return output
        else:
            return tensor

    def broadcast_n_axis(self,n1, n2, temp_tensor, input_dtype):
        g = n1 // n2
        temp_shape = temp_tensor.shape
        B = temp_shape[0]
        S = temp_shape[2]
        D = temp_shape[3]
        modify_tensor = torch.zeros([B, n1, S, D], dtype=temp_tensor.dtype)
        for i in range(n1):
            j = i // g
            modify_tensor[:, i:i + 1, :, :] = temp_tensor[:, j:j + 1, :, :]
        return modify_tensor, modify_tensor.shape

    def create_mask(self, m_shape, act_k, S1):
        atten_masks = torch.zeros(tuple(m_shape), dtype=torch.uint8)
        cmp_ratio = self.cmp_ratio
        tmp_pos_orig = act_k - S1

        for i in range(S1):
            if(((tmp_pos_orig+i+1)/cmp_ratio) < 0):
               atten_masks[i,:] = 1
            else:
               atten_masks[i, math.floor((tmp_pos_orig+i+1)/cmp_ratio):] = 1
        return atten_masks

    def create_mask_right_down(self, m_shape, actualSeqLengthsQ, actualSeqLengthsK, batch):
        mask_s_q = m_shape[0]
        mask_s_kv = m_shape[1]
        cmp_ratio = self.cmp_ratio
        next_tokens_list = []
        re_mask_batch = []
        pre_tokens = 214748647
        for i in range(batch):
            if len(actualSeqLengthsQ) == 0:
                S1 = mask_s_q
            else:
                S1 = actualSeqLengthsQ[i]

            if len(actualSeqLengthsK) == 0:
                S2 = mask_s_kv
            else:
                S2 = math.floor(actualSeqLengthsK[i]/cmp_ratio)
            next_tokens = S2-S1
            next_tokens_list.append(next_tokens)
            act_k = actualSeqLengthsK[i]
            atten_masks = self.create_mask(m_shape, act_k, S1)
            re_mask_batch.append(atten_masks)
        re_mask_np = np.array(re_mask_batch, dtype=np.bool_)
        cpu_mask = torch.from_numpy(re_mask_np)
        return cpu_mask, next_tokens_list


    def forward(self, query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key, block_table):
        print("cpu执行中...")

        # 参数的初始化
        batch_size = self.batch_size
        q_seq = self.q_seq
        k_seq = self.k_seq
        layout_query = self.layout_query
        layout_key = self.layout_key
        sparse_count = self.sparse_count
        sparse_mode = self.sparse_mode
        out_shape = self.out_shape
        q_shape = self.q_shape
        head_dim = self.head_dim
        q_head_num = self.q_head_num
        k_head_num = self.k_head_num
        q_t_size = self.q_t_size
        k_t_size = self.k_t_size
        block_size = self.block_size
        block_num = self.block_num
        q_dtype = self.qk_dtype
        k_dtype = self.qk_dtype
        w_shape = self.w_shape
        w_dtype = self.w_dtype
        actual_seq_dtype = self.actual_seq_dtype
        cmp_ratio = self.cmp_ratio

        if layout_query == "TND":
            if len(actual_seq_lengths_query) == batch_size + 1:
                actual_seq_lengths_query = actual_seq_lengths_query[1:]
            actual_seq_lengths_query = self.trans_tnd_actseq(actual_seq_lengths_query)
            self.actual_seq_lengths_query = torch.tensor(actual_seq_lengths_query)
            actualSeqLengths_q = self.actual_seq_lengths_query
            q_scale_shape = [q_t_size, q_head_num]

        elif layout_query == "BSND":
            self.actual_seq_lengths_query = actual_seq_lengths_query
            actualSeqLengths_q = self.actual_seq_lengths_query
            q_scale_shape = [batch_size, q_seq, q_head_num]

        if layout_key == "TND":
            if len(actual_seq_lengths_key) == batch_size + 1:
                actual_seq_lengths_key = actual_seq_lengths_key[1:]
            actual_seq_lengths_key = self.trans_tnd_actseq(actual_seq_lengths_key)
            self.actual_seq_lengths_key = torch.tensor(actual_seq_lengths_key)
            actualSeqLengths_k = self.actual_seq_lengths_key
            layout_key_scale = "TN"
            k_shape = self.k_shape
            k_scale_shape = [k_t_size, k_head_num]

        elif layout_key == "BSND":
            self.actual_seq_lengths_key = actual_seq_lengths_key
            actualSeqLengths_k = self.actual_seq_lengths_key
            layout_key_scale = "BSN"
            k_shape = self.k_shape
            k_scale_shape = [batch_size, k_seq, k_head_num]

        elif layout_key == "PA_BSND":
            self.actual_seq_lengths_key = actual_seq_lengths_key
            actualSeqLengths_k = self.actual_seq_lengths_key
            layout_key_scale = layout_key
            k_max_s2 = math.floor(max(actualSeqLengths_k)/self.cmp_ratio)
            k_shape = [batch_size, k_head_num, k_max_s2, head_dim]
            k_scale_shape = [batch_size, k_head_num, k_max_s2]
        query = query.cpu()
        key = key.cpu()
        weights = weights.cpu()
        query_dequant_scale = query_dequant_scale.cpu()
        key_dequant_scale = key_dequant_scale.cpu()

        # 将输入转化为BNSD
        ## BSND / TND -> BNSD
        q_bnsd_tensor, q_bnsd_shape = self.trans_shape_to_bnsd(query, q_shape, layout_query,
                                                        q_head_num, actualSeqLengths_q)

        ## BSND/TND/ -> BNSD
        k_bnsd_tensor, k_bnsd_shape = self.trans_shape_to_bnsd(key, k_shape, layout_key,
                                                        k_head_num, torch.floor(actualSeqLengths_k/cmp_ratio).to(actual_seq_dtype))



        k_scale_bnsd_tensor, k_scale_bnsd_shape = self.trans_shape_to_bnsd(key_dequant_scale, k_scale_shape,
                                                                    layout_key_scale,
                                                                    k_head_num, torch.floor(actualSeqLengths_k/cmp_ratio).to(actual_seq_dtype))

        ## BSN1 -> BNS1   TN1 -> BNS1
        is_weights = True
        wt_bnsd_tensor, wt_bnsd_shape = self.trans_shape_to_bnsd(weights, w_shape, layout_query,
                                                            q_head_num, actualSeqLengths_q, is_weights)

        # BSN1 -> BNS1
        q_scale_bnsd_tensor, q_scale_bnsd_shape = self.trans_shape_to_bnsd(query_dequant_scale, q_scale_shape,
                                                                    layout_query,
                                                                    q_head_num, actualSeqLengths_q, is_weights)
        # 将 k n2轴 广播为 n1
        if q_head_num != k_head_num:
            k_bnsd_tensor, k_bnsd_shape = self.broadcast_n_axis(q_head_num, k_head_num, k_bnsd_tensor, k_dtype)
        self.q_bnsd_tensor = q_bnsd_tensor
        self.q_bnsd_shape = q_bnsd_shape
        self.k_bnsd_tensor = k_bnsd_tensor
        self.k_bnsd_shape = k_bnsd_shape
        self.wt_bnsd_tensor = wt_bnsd_tensor
        self.wt_bnsd_shape = wt_bnsd_shape
        self.q_scale_bnsd_tensor = q_scale_bnsd_tensor
        self.q_scale_bnsd_shape = q_scale_bnsd_shape
        self.k_scale_bnsd_tensor = k_scale_bnsd_tensor
        self.k_scale_bnsd_shape = k_scale_bnsd_shape
        # 生成mask, sparse_mode=3时使能
        m_shape_std = [q_bnsd_shape[2], k_bnsd_shape[2]] #m_shape应该是[s1,s2]
        batch = q_bnsd_shape[0]
        m_tensor = []
        if sparse_mode == 3:
            m_tensor, next_tokens_list = self.create_mask_right_down(m_shape_std, actualSeqLengths_q, actualSeqLengths_k, batch)
        elif sparse_mode == 0:
            pass
        else:
            raise ValueError("unsupported sparse_mode!")
        self.m_tensor = m_tensor
        y, y_value = self.cal_atten_bnsd()
        # TND & PA 需要传入out_shape为BNSD
        out_shape_bnsd = copy.deepcopy(self.q_bnsd_shape)
        out_shape_bnsd[1] = k_head_num
        out_shape_bnsd[-1] = sparse_count
        y = self.trans_bnsd_to_layout(y, out_shape_bnsd, layout_query, actualSeqLengths_q)
        return y, y_value

def trans_prefix_actseq(self,list):
        list_len = len(list)
        if list_len == 0:
            raise ValueError(f'PA场景下 act_seq需要必传')
        list_new = []
        list_new.append(list[0])
        for i in range(list_len - 1):
            new_item = list[i + 1] - list[i]
            if new_item >= 0:
                list_new.append(new_item)
            else:
                raise ValueError(f'PA场景下act seq 为非递减数列 act_seq ={list}')
        return list_new

def qli_output_single(params):
    batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num, \
    qk_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode,key_quant_mode, \
    layout_query, layout_key, sparse_count, sparse_mode, query_datarange, key_datarange, weights_datarange,\
    q_scale_datarange, k_scale_datarange, cmp_ratio = params
    
    if qk_dtype == torch.uint8:
        hifp8mode = 1
        query_dtype = torch_npu.hifloat8 # ACL_HIFLOAT8
        key_dtype = torch_npu.hifloat8   # ACL_HIFLOAT8
    else:
        hifp8mode = 0

    test_qli = GeneralizedQLI(batch_size, q_seq, k_seq, q_t_size, k_t_size, q_head_num, k_head_num, head_dim, block_size, block_num,
                              qk_dtype, dequant_dtype, actual_seq_dtype, act_seq_q, act_seq_k, query_quant_mode,
                              key_quant_mode, layout_query, layout_key, sparse_count, sparse_mode, cmp_ratio)

    actual_seq_lengths_query = torch.tensor(np.random.uniform(q_seq, q_seq, batch_size)).to(actual_seq_dtype).npu() \
                            if act_seq_q is None else torch.tensor(act_seq_q).to(actual_seq_dtype).npu()
    actual_seq_lengths_key = torch.tensor(np.random.uniform(k_seq*cmp_ratio, k_seq*cmp_ratio, batch_size)).to(actual_seq_dtype).npu() \
                            if act_seq_k is None else torch.tensor(act_seq_k).to(actual_seq_dtype).npu()

    if layout_query == "BSND":
        if hifp8mode ==1:
            query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1],(batch_size, q_seq, q_head_num, head_dim))).to(torch.float)
            query = trans_float_tensor_to_hifuint8(query, round_mode = "hybrid", over_mode = True).npu()
            q_scale = random.uniform(q_scale_datarange[0], q_scale_datarange[1])
            query_dequant_scale = torch.tensor(np.random.uniform(q_scale, q_scale, (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()
        else:
            query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1],(batch_size, q_seq, q_head_num, head_dim))).to(qk_dtype).npu()
            query_dequant_scale = torch.tensor(np.random.uniform(q_scale_datarange[0], q_scale_datarange[1], (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (batch_size, q_seq, q_head_num))).to(dequant_dtype).npu()

    elif layout_query == "TND":
        if hifp8mode ==1:
            query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_t_size, q_head_num, head_dim))).to(torch.float)
            query = trans_float_tensor_to_hifuint8(query, round_mode = "hybrid", over_mode = True).npu()
            q_scale = random.uniform(q_scale_datarange[0], q_scale_datarange[1])
            query_dequant_scale = torch.tensor(np.random.uniform(q_scale, q_scale, (q_t_size, q_head_num))).to(dequant_dtype).npu()
        else:
            query = torch.tensor(np.random.uniform(query_datarange[0], query_datarange[1], (q_t_size, q_head_num, head_dim))).to(qk_dtype).npu()
            query_dequant_scale = torch.tensor(np.random.uniform(q_scale_datarange[0], q_scale_datarange[1], (q_t_size, q_head_num))).to(dequant_dtype).npu()
        weights = torch.tensor(np.random.uniform(weights_datarange[0], weights_datarange[1], (q_t_size, q_head_num))).to(dequant_dtype).npu()

    if layout_key == "BSND":
        if hifp8mode ==1:
            key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (batch_size, k_seq, k_head_num, head_dim))).to(torch.float)
            key = trans_float_tensor_to_hifuint8(key, round_mode = "hybrid", over_mode = True).npu()
            k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
            key_dequant_scale = torch.tensor(np.random.uniform(k_scale, k_scale, (batch_size, k_seq, k_head_num))).to(dequant_dtype).npu()
        else:
            key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (batch_size, k_seq, k_head_num, head_dim))).to(qk_dtype).npu()
            key_dequant_scale = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (batch_size, k_seq, k_head_num))).to(dequant_dtype).npu()
        block_table = None
        cpu_result, topk_value = test_qli.forward(query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key, block_table)

    elif layout_key == "TND":
        if hifp8mode ==1:
            key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (k_t_size, k_head_num, head_dim))).to(torch.float)
            key = trans_float_tensor_to_hifuint8(key, round_mode = "hybrid", over_mode = True).npu()
            k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
            key_dequant_scale = torch.tensor(np.random.uniform(k_scale, k_scale, (k_t_size, k_head_num))).to(dequant_dtype).npu()
        else:
            key = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1], (k_t_size, k_head_num, head_dim))).to(qk_dtype).npu()
            key_dequant_scale = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (k_t_size, k_head_num))).to(dequant_dtype).npu()
        block_table = None
        cpu_result, topk_value = test_qli.forward(query, key, weights, query_dequant_scale, key_dequant_scale, actual_seq_lengths_query, actual_seq_lengths_key, block_table)

    elif layout_key == "PA_BSND":
        # 以不同batch中最大seq为标准初始化key(bnsd)和key_dequant_scale(bns)
        k_max_s2 = math.floor(max(act_seq_k)/cmp_ratio)
        k_max_block_num_per_batch = math.ceil(k_max_s2 / block_size) #遍历batch得到的最大的block num
        if hifp8mode ==1:
            key_bnsd = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1],(batch_size, k_head_num, k_max_s2, head_dim))).to(torch.float)
            key_bnsd = trans_float_tensor_to_hifuint8(key_bnsd, round_mode = "hybrid", over_mode = True)
            k_scale = random.uniform(k_scale_datarange[0], k_scale_datarange[1])
            key_dequant_scale_bns = torch.tensor(np.random.uniform(k_scale, k_scale, (batch_size, k_head_num, k_max_s2))).to(dequant_dtype)
        else:
            key_bnsd = torch.tensor(np.random.uniform(key_datarange[0], key_datarange[1],(batch_size, k_head_num, k_max_s2, head_dim))).to(qk_dtype)
            key_dequant_scale_bns = torch.tensor(np.random.uniform(k_scale_datarange[0], k_scale_datarange[1], (batch_size, k_head_num, k_max_s2))).to(dequant_dtype)
        key_block_num_per_batch = []
        key_block_num_sum = 0
        for cur_act_k in act_seq_k:
            cur_cmp_act_k = math.floor(cur_act_k / cmp_ratio)
            cur_key_block_num = math.ceil(cur_cmp_act_k / block_size)
            key_block_num_per_batch.append(cur_key_block_num)
            key_block_num_sum += cur_key_block_num
        if block_num < key_block_num_sum:
            raise ValueError(f"key actual block num < needed block num")
        # 构建block table
        block_id_list = np.arange(block_num)
        block_id_list = np.random.permutation(block_id_list).astype(np.int32)
        cur_block_id = 0
        block_table = np.full((batch_size, k_max_block_num_per_batch), fill_value = -1, dtype=np.int32)
        batch_idx = 0
        for cur_block_id_threshold in key_block_num_per_batch:
            for i_block_id in range(cur_block_id_threshold):
                block_table[batch_idx][i_block_id] = block_id_list[cur_block_id]
                cur_block_id += 1
            batch_idx += 1
        # 构建PA场景的key
        # [batch_size, s2, k_head_num, head_dim] expand to [batch_size, k_max_block_num_per_batch * block_size, k_head_num, head_dim]
        key_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size, head_dim), dtype = qk_dtype)
        key_expand[:,:,:k_max_s2,:] = key_bnsd
        key = torch.zeros((block_num, block_size, k_head_num, head_dim), dtype = qk_dtype) # hifp8时qk_dtype为uint8
        for i_batch in range(batch_size):
            for  i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key[cur_block_id, :, i_n, :] = key_expand[i_batch, i_n, block_start_pos:block_start_pos+block_size,:]
        key = key.npu()
        # 构建PA场景的key_dequant_scale
        key_dequant_scale_expand = torch.zeros((batch_size, k_head_num, k_max_block_num_per_batch * block_size), dtype= dequant_dtype)
        key_dequant_scale_expand[:,:,:k_max_s2] = key_dequant_scale_bns
        key_dequant_scale = torch.zeros((block_num, block_size, k_head_num), dtype = dequant_dtype)
        for i_batch in range(batch_size):
            for i_block, cur_block_id in enumerate(block_table[i_batch]):
                block_start_pos = i_block * block_size
                if cur_block_id == -1:
                    continue
                else:
                    for i_n in range(k_head_num):
                        key_dequant_scale[cur_block_id, :, i_n] = key_dequant_scale_expand[i_batch, i_n,block_start_pos:block_start_pos+block_size]
        key_dequant_scale = key_dequant_scale.npu()
        cpu_result, topk_value = test_qli.forward(query, key_bnsd, weights, query_dequant_scale, key_dequant_scale_bns, actual_seq_lengths_query, actual_seq_lengths_key, block_table)
        block_table = torch.from_numpy(block_table).to(dtype=torch.int32).npu()
    max_seqlen_q = actual_seq_lengths_query.max().item()
    max_seqlen_k = actual_seq_lengths_key.max().item()
    metadata = torch.ops.custom.npu_quant_lightning_indexer_metadata(
                                    num_heads_q = q_head_num,
                                    num_heads_k = k_head_num,
                                    head_dim = head_dim,
                                    query_quant_mode = query_quant_mode,
                                    key_quant_mode = key_quant_mode,
                                    actual_seq_lengths_query = actual_seq_lengths_query,
                                    actual_seq_lengths_key = actual_seq_lengths_key,
                                    batch_size = batch_size,
                                    max_seqlen_q = max_seqlen_q,
                                    max_seqlen_k = max_seqlen_k,
                                    layout_query = layout_query,
                                    layout_key = layout_key,
                                    sparse_count = sparse_count,
                                    sparse_mode = sparse_mode,
                                    pre_tokens = (1<<63)-1,
                                    next_tokens = (1<<63)-1,
                                    cmp_ratio = cmp_ratio,
                                    device = 'npu:0')

    metadata = metadata.npu()
    if hifp8mode == 1:
        npu_result,_ = torch.ops.custom.npu_quant_lightning_indexer(query, key, weights,
                                                    query_dequant_scale,
                                                    key_dequant_scale,
                                                    actual_seq_lengths_query = actual_seq_lengths_query,
                                                    actual_seq_lengths_key = actual_seq_lengths_key,
                                                    block_table = block_table,
                                                    metadata = metadata,
                                                    query_quant_mode = query_quant_mode,
                                                    key_quant_mode = key_quant_mode,
                                                    layout_query = layout_query,
                                                    layout_key = layout_key,
                                                    sparse_count = sparse_count,
                                                    sparse_mode = sparse_mode,
                                                    pre_tokens = (1<<63)-1,
                                                    next_tokens = (1<<63)-1,
                                                    cmp_ratio = cmp_ratio,
                                                    return_value = False,
                                                    query_dtype = query_dtype,
                                                    key_dtype = key_dtype)
    else:
        npu_result,_ = torch.ops.custom.npu_quant_lightning_indexer(query, key, weights,
                                                        query_dequant_scale,
                                                        key_dequant_scale,
                                                        actual_seq_lengths_query = actual_seq_lengths_query,
                                                        actual_seq_lengths_key = actual_seq_lengths_key,
                                                        block_table = block_table,
                                                        metadata = metadata,
                                                        query_quant_mode = query_quant_mode,
                                                        key_quant_mode = key_quant_mode,
                                                        layout_query = layout_query,
                                                        layout_key = layout_key,
                                                        sparse_count = sparse_count,
                                                        sparse_mode = sparse_mode,
                                                        pre_tokens = (1<<63)-1,
                                                        next_tokens = (1<<63)-1,
                                                        cmp_ratio = cmp_ratio,
                                                        return_value = False)
    torch.npu.synchronize()
    return cpu_result, npu_result, topk_value

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