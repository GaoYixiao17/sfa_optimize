# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn
import math

from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

def convert_pa_nz_to_pa_bnsd(key_pa_nz, d=None):
    """
    将PA_NZ格式转换为PA_BNSD格式
    
    Args:
        key_pa_nz: PA_NZ格式的key，形状为 (B, N, d//64, S, 8)
        d: 可选参数，原始特征维度，用于验证
    
    Returns:
        PA_BNSD格式的key，形状为 (B, N, S, d//8)
    """
    B, N, z_blocks, S, eight = key_pa_nz.shape
    
    # 可选验证
    if d is not None:
        expected_z_blocks = d // 64
        assert z_blocks == expected_z_blocks, \
            f"z_blocks({z_blocks}) != d//64({expected_z_blocks})"
    
    # 转换
    result = key_pa_nz.permute(0, 1, 3, 2, 4).reshape(B, N, S, -1)
    
    return result


def _get_data_from_pa_cache(key, block_table, act_s2, layout_key):
    if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
        block_num, n2, block_size, d = key.shape
    else:
        block_num, block_size, n2, d = key.shape
    need_blcok_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_blcok_num * block_size
    if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
        out = torch.zeros((n2, act_s2_align, d), dtype=key.dtype, device=key.device)
    else:
        out = torch.zeros((act_s2_align, n2, d), dtype=key.dtype, device=key.device)
    for i in range(need_blcok_num):
        if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
            out[:, i * block_size:(i + 1) * block_size, :] = key[block_table[i], ...].reshape(n2, block_size, d)
        else:
            out[i * block_size:(i + 1) * block_size, :, :] = key[block_table[i], ...].reshape(block_size, n2, d)
    if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
        return out[:, :act_s2, :]
    else:
        return out[:act_s2, :, :]


def _get_k_scale(key_dequant_scale, block_table, act_s2, layout_key):
    if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
        block_num, n2, block_size = key_dequant_scale.shape
    else:
        block_num, block_size, n2 = key_dequant_scale.shape
    need_blcok_num = (act_s2 + block_size - 1) // block_size
    act_s2_align = need_blcok_num * block_size
    if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
        out = torch.zeros((n2, act_s2_align), dtype=key_dequant_scale.dtype, device=key_dequant_scale.device)
        key_dequant_scale = key_dequant_scale.reshape(block_num, n2, block_size)
    else:
        out = torch.zeros((act_s2_align, n2), dtype=key_dequant_scale.dtype, device=key_dequant_scale.device)
        key_dequant_scale = key_dequant_scale.reshape(block_num, block_size, n2)
    for i in range(need_blcok_num):
        if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
            out[:, i * block_size:(i + 1) * block_size] = key_dequant_scale[block_table[i], ...].reshape(n2, block_size)
        else:
            out[i * block_size:(i + 1) * block_size, :] = key_dequant_scale[block_table[i], ...].reshape(block_size, n2)
    if layout_key == 'PA_BNSD' or layout_key == 'PA_NZ':
        return out[:, :act_s2]
    else:
        return out[:act_s2, :]

def trans_int32_2_int4_torch(key):
    """
    PyTorch版本的int32转int4转换
    """
    # 确保key是torch张量
    if not isinstance(key, torch.Tensor):
        key = torch.tensor(key)
    
    # 将key转换为uint32类型进行位操作
    key_uint32 = key.to(torch.int32)  # 使用int32而不是uint32，因为PyTorch对uint32支持有限
    
    # 创建移位张量
    shifts = torch.arange(0, 32, 4, device=key.device, dtype=torch.int32)
    
    # 使用广播进行向量化移位
    # 扩展维度以便广播 [..., 1] >> [8] -> [..., 8]
    expanded_key = key_uint32.unsqueeze(-1)  # 在最后添加一个维度
    shifted = expanded_key >> shifts  # 广播移位
    
    # 提取低4位
    parts = shifted & 0xF
    
    # 处理符号位和数值
    sign = (parts >> 3) & 1
    value = parts & 0x7
    
    # 使用torch.where进行条件赋值
    result = torch.where(sign == 1, -((value ^ 0x7) + 1), value)
    
    return result.to(torch.int8)

def optimize_conversion_torch(key):
    """
    PyTorch优化的主函数
    """
    # 获取原始形状
    original_shape = key.shape
    
    # 如果key是numpy数组，转换为torch张量
    if isinstance(key, np.ndarray):
        key = torch.from_numpy(key)
    
    # 重塑为2D (batch, features)
    key_2d = key.reshape(-1, original_shape[-1])
    
    # 向量化转换
    int4_data = trans_int32_2_int4_torch(key_2d)
    
    # 重塑为最终形状
    final_shape = original_shape[:-1] + (original_shape[-1] * 8,)
    key_int4 = int4_data.reshape(final_shape)
    
    return key_int4

#将Int32类型的数据，拆成int4，存进int8
def trans_int32_2_int4(input_int32):
    # 将Int32类型的数据按bit位平均拆成8份，每份长度为4bit
    parts = [(input_int32 >> i) & 0xf for i in range(0, 32, 4)]
    output_int4 = []
    for part in parts:
        # 将每份数据构造成一个Int8类型的数据
        # 符号位为第一个bit的值
        sign = (part >> 3) & 0x1
        # 剩余的值作为int8最后3个bit的值
        value = part & 0x7
        if sign == 1:
            # 如果符号位为1，则需要将value转换成负数
            value = -((value ^ 0x7) + 1)
        output_int4.append(value)
    return output_int4

def _quant_sals_indexer(query, key, query_dequant_scale, key_dequant_scale, actual_seq_lengths_key, block_table,
                        sparse_block_size, sparse_ratio, fixed_tail_count, layout_key, max_seqlen_key):
    if layout_key == 'PA_NZ':
        key = convert_pa_nz_to_pa_bnsd(key)
    sparse_ratio = round(1.0 - sparse_ratio, 2)
    batch_size = query.shape[0]
    n2 = query.shape[1]
    # query转换
    q_shape_int4 = [query.shape[0], query.shape[1], query.shape[2] * 8]
    query_int4 = torch.zeros(q_shape_int4, dtype=torch.int8, device=query.device)
    query_int4 = optimize_conversion_torch(query)
            
    # key转换
    k_shape_int4 = [key.shape[0], key.shape[1], key.shape[2], key.shape[3] * 8]
    key_int4 = torch.zeros(k_shape_int4, dtype=torch.int8, device=key.device)
    key_int4 = optimize_conversion_torch(key)
    d = query_int4.shape[-1]    
    max_count = (max_seqlen_key + sparse_block_size - 1) // sparse_block_size
    if max_count - fixed_tail_count >= 0:
        sparse_count = min(int(math.ceil((max_count - fixed_tail_count) * sparse_ratio)), 2048) + fixed_tail_count
    else:
        sparse_count = max_count
    sparse_indices_shape = [batch_size, n2, sparse_count]
    # 初始化为全-1
    sparse_indices = torch.zeros(sparse_indices_shape, dtype=torch.int32, device=query.device) - 1

    for batch_id in range(batch_size):
        act_s2 = actual_seq_lengths_key[batch_id]
        fixed_tail_count_tmp = fixed_tail_count

        act_n_count = (act_s2 + sparse_block_size - 1) // sparse_block_size
        sort_n_count = act_n_count - fixed_tail_count_tmp if act_n_count - fixed_tail_count_tmp > 0 else 0
        topk_n_count = int(math.ceil(sort_n_count * sparse_ratio))
        fixed_tail_count_tmp = act_n_count if act_n_count - fixed_tail_count_tmp <= 0 else fixed_tail_count_tmp
        topk_n_count = min(topk_n_count, 2048)
        if topk_n_count > 0:
            # b, n2, d
            now_q = query_int4[batch_id, :, :].reshape(n2, 1, d).to(torch.int32)
            now_block_table = block_table[batch_id, :]
            # s2, n2, d -> n2, d, s2
            if layout_key == "PA_BNSD" or layout_key == "PA_NZ":
                now_k = _get_data_from_pa_cache(key_int4, now_block_table, act_s2, layout_key).permute(0, 2, 1).to(torch.int32)
                now_k_scale = _get_k_scale(key_dequant_scale, now_block_table, act_s2, layout_key).permute(1, 0)
            elif layout_key == "PA_BSND":
                now_k = _get_data_from_pa_cache(key_int4, now_block_table, act_s2, layout_key).permute(1, 2, 0).to(torch.int32)
                now_k_scale = _get_k_scale(key_dequant_scale, now_block_table, act_s2, layout_key)
            else:
                now_k = key_int4[batch_id, :act_s2, :, :].permute(1, 2, 0).to(torch.int32)
                now_k_scale = key_dequant_scale[batch_id, :act_s2, :].to(torch.float)
            now_q_scale = query_dequant_scale[batch_id, :].to(torch.float)
            now_qk_scale = now_q_scale * now_k_scale
            # n2,1,d @ d,s2 -> n2,1,s2
            s_out = torch.matmul(now_q, now_k)
            now_qk_scale = now_qk_scale.permute(1, 0).view(n2, 1, act_s2) # 广播到(n2, 1, s2)
            s_out = s_out * now_qk_scale
            s_max = torch.zeros(n2, 1, sort_n_count, dtype=torch.float32, device=key.device)
            s_lse = torch.zeros(n2, 1, sort_n_count, dtype=torch.float32, device=key.device)
            for sort_n_idx in range(sort_n_count):
                max_value, _ = torch.max(s_out[:,:, sort_n_idx * sparse_block_size : (sort_n_idx + 1)* sparse_block_size], axis = 2)
                s_max[:, :, sort_n_idx] = max_value
                s_lse[:, :, sort_n_idx] = s_max[:, :, sort_n_idx] + torch.log(torch.sum(torch.exp(s_out[:,:, sort_n_idx * sparse_block_size : (sort_n_idx + 1)* sparse_block_size] - s_max[:, :, sort_n_idx: sort_n_idx + 1]), axis=2))


            sorted_value, sorted_indices = torch.sort(s_lse, dim=2, descending=True, stable=True)
            sparse_indices[batch_id, :, :topk_n_count] = sorted_indices.to(torch.int32).permute(1,0,2)[:, :, :topk_n_count]
        for fixed_n_count_idx in range(fixed_tail_count_tmp):
            sparse_indices[batch_id, :, topk_n_count + fixed_n_count_idx] = sort_n_count + fixed_n_count_idx
    return sparse_indices


def _compare_res(golden_res, npu_res, ratio_thress_hold=0.999):
    npu_res = npu_res.reshape(-1)
    golden_res = golden_res.reshape(-1)
    total_res_num = golden_res.numel()
    diff_res = npu_res - golden_res
    match_ratio = (diff_res == 0).sum().float() / total_res_num
    if match_ratio >= ratio_thress_hold:
        print(f"Compare npu cpu res success! Match ratio is {match_ratio:.4%}")
        return True
    else:
        print(f"Match ratio {match_ratio:.4%} is under thress_hold {ratio_thress_hold} ",
              "Please Check!")
        non_zero_index = torch.nonzero(diff_res, as_tuple=False).squeeze(1)
        npu_index = npu_res[non_zero_index]
        golden_index = golden_res[non_zero_index]
        for i in non_zero_index:
            print(f"mismatch idx: {i}, golden and npu res is {golden_res[i]} || {npu_res[i]}")
        return False

class QuantSINetwork(nn.Module):
    def __init__(self):
        super(QuantSINetwork, self).__init__()

    def forward(self, b, s2, n2, d, query, key, query_dequant_scale, key_dequant_scale, actual_seq_lengths_key, block_table, max_seqlen_key, sparse_block_size, sparse_ratio, 
                fixed_tail_count, layout_key):
        # super kernel test
        with torchair.scope.super_kernel("sp_QsQSI", ""):
            metadata = torch.ops.custom.npu_quant_sals_indexer_metadata(
                                            b,
                                            s2,
                                            n2,
                                            d,
                                            sparse_block_size=sparse_block_size,
                                            sparse_ratio=sparse_ratio,
                                            fixed_tail_count=fixed_tail_count,
                                            layout_key=layout_key,
                                            actual_seq_lengths_kv=actual_seq_lengths_key)

            output0 = torch.ops.custom.npu_quant_sals_indexer(query, key, query_dequant_scale = query_dequant_scale,
                                            key_dequant_scale = key_dequant_scale, metadata=metadata,
                                            actual_seq_lengths_key=actual_seq_lengths_key,
                                            block_table=block_table, max_seqlen_key=max_seqlen_key,
                                            sparse_block_size=sparse_block_size, 
                                            sparse_ratio=sparse_ratio, fixed_tail_count=fixed_tail_count,
                                            layout_key=layout_key)
            return output0


class TestCustomLightningIndexerQuant(TestCase):
    def cpu_op_exec(self, query, key, query_dequant_scale, key_dequant_scale, actual_seq_lengths_key, block_table,
                    sparse_block_size, sparse_ratio, fixed_tail_count, layout_key, max_seqlen_key):
        output0 = _quant_sals_indexer(query, key, query_dequant_scale, key_dequant_scale, actual_seq_lengths_key, block_table,
                                               sparse_block_size, sparse_ratio, fixed_tail_count, layout_key, max_seqlen_key)
        output0 = output0.cpu()

        return output0

    def npu_op_exec_graph(self, b, s2, n2, d, query, key, query_dequant_scale, key_dequant_scale, actual_seq_lengths_key,
                          block_table, max_seqlen_key, sparse_block_size, sparse_ratio, fixed_tail_count, layout_key):
        npu_mode = QuantSINetwork().to("npu:%s" % DEVICE_ID)
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        torch._dynamo.reset()
        npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
        npu_out0 = npu_mode(b, s2, n2, d, query, key,
                            query_dequant_scale = query_dequant_scale,
                            key_dequant_scale = key_dequant_scale,
                            actual_seq_lengths_key=actual_seq_lengths_key,
                            block_table=block_table,
                            max_seqlen_key = max_seqlen_key,
                            sparse_block_size=sparse_block_size,
                            sparse_ratio=sparse_ratio,
                            fixed_tail_count=fixed_tail_count,
                            layout_key=layout_key)
        npu_out0 = npu_out0.cpu()
        return npu_out0

    def npu_op_exec_eager(self, b, s2, n2, d, query, key, query_dequant_scale, key_dequant_scale, actual_seq_lengths_key,
                          block_table, max_seqlen_key, sparse_block_size, sparse_ratio, fixed_tail_count, layout_key):
        
        metadata = torch.ops.custom.npu_quant_sals_indexer_metadata(
                                            b,
                                            s2,
                                            n2,
                                            d,
                                            sparse_block_size=sparse_block_size,
                                            sparse_ratio=sparse_ratio,
                                            fixed_tail_count=fixed_tail_count,
                                            layout_key=layout_key,
                                            actual_seq_lengths_kv=actual_seq_lengths_key)
        npu_out0 = torch.ops.custom.npu_quant_sals_indexer(query, key,
                                                    metadata=metadata,
                                                    query_dequant_scale = query_dequant_scale,
                                                    key_dequant_scale = key_dequant_scale,
                                                    actual_seq_lengths_key=actual_seq_lengths_key,
                                                    block_table=block_table,
                                                    max_seqlen_key = max_seqlen_key,
                                                    sparse_block_size=sparse_block_size,
                                                    sparse_ratio=sparse_ratio,
                                                    fixed_tail_count=fixed_tail_count,
                                                    layout_key=layout_key)
        npu_out0 = npu_out0.cpu()
        return npu_out0

    def quant_sals_indexer_result(self, b, s2, n2, d, act_seq_k, sparse_block_size, sparse_ratio, fixed_tail_count, max_seqlen_key):
        # -----固定参数--------
        block_size = 512
        layout_key = 'PA_BNSD'
        np.random.seed(0)
        # -------------
        max_block_table_num = (s2 + block_size - 1) // block_size
        block_table = torch.tensor([range(b * max_block_table_num)], dtype = torch.int32).reshape(b, -1)
        if layout_key == 'PA_BNSD':
            key = torch.tensor(np.random.uniform(-2147483648, 2147483647, (b * max_block_table_num, n2, block_size, d // 8))).to(torch.int32) # d轴8个数合并
            key_dequant_scale = torch.tensor(np.random.uniform(-1, 1, (b * max_block_table_num, n2, block_size))).to(torch.float)
        elif layout_key == 'PA_BSND':
            key = torch.tensor(np.random.uniform(-2147483648, 2147483647, (b * max_block_table_num, block_size, n2, d // 8))).to(torch.int32) # d轴8个数合并
            key_dequant_scale = torch.tensor(np.random.uniform(-1, 1, (b * max_block_table_num, block_size, n2))).to(torch.float)
        elif layout_key == 'PA_NZ':
            key = torch.tensor(np.random.uniform(-2147483648, 2147483647, (b * max_block_table_num, n2, ((d // 8) // 8), block_size, 8))).to(torch.int32)
            key_dequant_scale = torch.tensor(np.random.uniform(-1, 1, (b * max_block_table_num, n2, block_size))).to(torch.float)
        else:
            key = torch.tensor(np.random.uniform(-2147483648, 2147483647, (b, s2, n2, d // 8))).to(torch.int32) # d轴8个数合并
            key_dequant_scale = torch.tensor(np.random.uniform(-1, 1, (b, s2, n2))).to(torch.float)

        query = torch.tensor(np.random.uniform(-2147483648, 2147483647, (b, n2, d // 8))).to(torch.int32) # d轴8个数合并
        actual_seq_lengths_key = torch.tensor(act_seq_k).to(torch.int32)
        query_dequant_scale = torch.tensor(np.random.uniform(-1, 1, (b, n2))).to(torch.float)
        print(f"------- test QuantSI BSND case b:{b} n2:{n2} s2:{s2} ----------")

        cpu_out0 = self.cpu_op_exec(query, key, query_dequant_scale, key_dequant_scale, actual_seq_lengths_key,
                                              block_table, sparse_block_size, sparse_ratio, fixed_tail_count, layout_key, max_seqlen_key)

        torch_npu.npu.set_device(int(DEVICE_ID))
        query = query.to("npu:%s" % DEVICE_ID)
        key = key.to("npu:%s" % DEVICE_ID)
        query_dequant_scale = query_dequant_scale.to("npu:%s" % DEVICE_ID)
        key_dequant_scale = key_dequant_scale.to("npu:%s" % DEVICE_ID)
        actual_seq_lengths_key = actual_seq_lengths_key.to("npu:%s" % DEVICE_ID)
        if layout_key == 'PA_BNSD' or layout_key == 'PA_BSND' or layout_key == 'PA_NZ':
            block_table = block_table.to("npu:%s" % DEVICE_ID)
        else:
            block_table = None
        print("run eager mode")
        sparse_indices = self.npu_op_exec_eager(b, s2, n2, d, query, key, query_dequant_scale, key_dequant_scale,
                                               actual_seq_lengths_key, block_table, max_seqlen_key,
                                               sparse_block_size, sparse_ratio, fixed_tail_count, layout_key)
        print("sparse_indices = ", sparse_indices)
        assert(_compare_res(cpu_out0, sparse_indices))

        print("run graph mode")
        sparse_indices = self.npu_op_exec_graph(b, s2, n2, d, query, key, query_dequant_scale, key_dequant_scale,
                                               actual_seq_lengths_key, block_table, max_seqlen_key,
                                               sparse_block_size, sparse_ratio, fixed_tail_count, layout_key)

        assert(_compare_res(cpu_out0, sparse_indices))

    def test_quant_sals_indexer(self):
        # b, s2, n2, d, act_seq_k, sparse_block_size, sparsity, fixed_tail_count, max_seqlen_key
        import random
        b = 21 * 4
        n2 = 8 // 4
        base_s2 = 1024 * 10
        random_seed = 42
        unbalance_cache = 0
        random.seed(random_seed)
        
        act_seq_k = []
        sumseqlength = 0
        for _ in range(b-1):
            offset_percent = random.randint(-unbalance_cache, unbalance_cache)
            offset = int(base_s2 * (offset_percent / 100))
            s2curent = base_s2 + offset
            sumseqlength = sumseqlength + s2curent
            act_seq_k.append(s2curent)
        totalseq = b * base_s2
        s2_last = totalseq - sumseqlength
        act_seq_k.append(s2_last)
        max_seqlen_key = 256*1024
        print('QSI test case, act_seq_k:', act_seq_k)
        print('QSI test case, max_seqlen_key:', max_seqlen_key)
        test_case_list = [
            #基础性能/功能用例
            (b, max_seqlen_key, n2, 64, act_seq_k, 16, 0.75, 32, max_seqlen_key),
            # (21, 10240, 8, 64, [10240] * 21, 16, 0.75, 16, 10240),
            # 泛化shape
            # (16, 512*64, 1, 64, [19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 16, 0.75, 1, 19),
            # (16, 512*64, 1, 64, [33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], 16, 0.75, 1, 33),
            # (16, 512*64, 1, 64, [6892,1458,3356,2001,0,0,0,0,0,0,0,0,0,0,0,0], 16, 0.75, 1, 6892),
            # (1, 512*64, 1, 64, [100], 16, 0.75, 1, 100),
            # (1, 2289, 1, 64, [2289], 16, 0.75, 1, 2289),
            # (1, 4352, 1, 64, [4352], 16, 0.75, 1, 4352),
            # (35, 4727, 1, 64, [0, 0, 3013, 334, 3194, 4029, 116, 4419, 41, 1495, 4295, 4471, 666, 582, 2863, 847, 4147, 146, 591, 2206, 4318, 3770, 1412, 1782, 853, 4721, 561, 1198, 4043, 2166, 2605, 209, 7, 3864, 2289],
            #     16, 0.75, 16, 4727),
            # (4, 2358, 4, 64, [2358] * 4, 16, 0.75, 16, 2358),
            # (4, 1111, 4, 64, [1111] * 4, 16, 0.75, 16, 1111),
            # (4, 2003, 4, 64, [2003] * 4, 16, 0.75, 16, 2003),
            # (4, 7891, 4, 64, [7891] * 4, 16, 0.75, 16, 7891),
            # (4, 6100, 4, 64, [6100] * 4, 16, 0.75, 16, 6100),
            # (3, 10240, 4, 64, [10240] * 3, 16, 0.75, 16, 10240),
            # (11, 10240, 4, 64, [10240] * 11, 16, 0.75, 16, 10240),
            # (29, 10240, 4, 64, [10240] * 29, 16, 0.75, 16, 10240),
            # (29, 1234, 4, 64, [1234] * 29, 16, 0.78, 16, 1234),
            # (33, 887, 4, 64, [887] * 33, 16, 0.85, 16, 887),
            # (19, 9813, 4, 64, [9813] * 19, 16, 0.65, 16, 9813),
        ]
        for case in test_case_list:
            self.quant_sals_indexer_result(*case)


if __name__ == "__main__":
    run_tests()
