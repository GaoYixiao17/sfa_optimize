/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// custom kernel code

#pragma once
#include "tensorutils.h"
#include "paramutils.h"

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkCumsum {

constexpr int BASEL = 64;
constexpr int BASEN = 64;
constexpr int BASEH = 128;
constexpr int SUB_BASEH = BASEH / 2;
constexpr int TILE_BLK_SIZE = BASEL * SUB_BASEH;
constexpr float COMPARE_VALUE = 20.0;
constexpr float CLAMP_MAX = 10000000.0f;

struct CustVecShapeInfo {
    int nstepsH;
    int BCH;
    int BCH_PER_CORE;
    int BCH1;
    int BCH2;
    int C;
    int L;
    int H;
    int effective_H;      // NEW: Actual H to process per sub-block
    int tile_size;        // NEW: BASEL * effective_H
};


__aicore__ inline void tilingShapeCustVec(int B, int C, int H, int L, CustVecShapeInfo &shape){
    // Determine effective H based on hardware constraints
    if (H <= 32) {
        shape.effective_H = 32;
        shape.nstepsH = 1;
    } else if (H <= 64) {
        shape.effective_H = 64;
        shape.nstepsH = 1;
    } else if (H <= 128) {
        shape.effective_H = 64;  // Process in two sub-blocks of 64
        shape.nstepsH = 2;
    } else {
        shape.effective_H = 64;
        shape.nstepsH = CeilDiv(H, 128);
    }
    shape.tile_size = BASEL * shape.effective_H;
    shape.BCH = ((B * C) * shape.nstepsH);
    shape.BCH_PER_CORE = CeilDiv(shape.BCH, GetBlockNum());
    shape.BCH1 = shape.BCH_PER_CORE * get_block_idx();
    shape.BCH2 = (((shape.BCH_PER_CORE + shape.BCH1))<(shape.BCH)) ? ((shape.BCH_PER_CORE + shape.BCH1)) : (shape.BCH);
    shape.C = C;
    shape.L = L;
    shape.H = H;
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR at_mtx_, GM_ADDR dt_mtx_, GM_ADDR dtbias_mtx_, GM_ADDR dtmask_mtx_, GM_ADDR out_mtx_, GM_ADDR out1_mtx_, GM_ADDR out2_mtx_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        at_mtx.SetGlobalBuffer((__gm__ float*) at_mtx_);
        dt_mtx.SetGlobalBuffer((__gm__ half*) dt_mtx_);
        dtbias_mtx.SetGlobalBuffer((__gm__ half*) dtbias_mtx_);
        dtmask_mtx.SetGlobalBuffer((__gm__ half*) dtmask_mtx_);
        out_mtx.SetGlobalBuffer((__gm__ float*) out_mtx_);
        out1_mtx.SetGlobalBuffer((__gm__ float*) out1_mtx_);
        out2_mtx.SetGlobalBuffer((__gm__ float*) out2_mtx_);
        
        // Local buffers
        at_buf.Init(shape.effective_H);
        dt_buf.Init(shape.tile_size);
        dtbias_buf.Init(shape.effective_H);
        dtmask_buf.Init(shape.tile_size);
        out1buf.Init(shape.tile_size);
        out2buf.Init(shape.tile_size);
        
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp0, shape.tile_size);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp1, shape.tile_size);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp2, shape.effective_H);
        AllocateLocalTensor<TPosition::VECCALC>(cc_tmp3, shape.tile_size);
        AllocateLocalTensor<TPosition::VECCALC>(cmp_mask, shape.tile_size);
        AllocateLocalTensor<TPosition::VECCALC>(max_buf, EIGHT);
        cumsum_tensor.Init(shape.effective_H);
        // Initialize events
        in_ready.Init();
        in_empty.Init();
        out_ready.Init();
        out_empty.Init();
    }
    
    __aicore__ inline void PreCompute(){
        // User-defined pre-computation
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        
        cc_cnt = 0;
        cast_params_h2f = CastHalf2FloatRepeatParams();
        cast_params_f2h = CastFloat2HalfRepeatParams();
        unary_params = MakeDefaultUnaryRepeatParams();
        binary_params = MakeDefaultBinaryRepeatParams();

        for (int bch = shape.BCH1; bch < shape.BCH2; ++bch) {
            int b = bch / (shape.C * shape.nstepsH);
            int c = (bch % (shape.C * shape.nstepsH)) / shape.nstepsH;
            int h_step = (bch % (shape.C * shape.nstepsH)) % shape.nstepsH;
            
            // Calculate actual h based on effective_H
            int h = h_step * shape.effective_H;
            if (shape.H > BASEH) {
                h = h_step * BASEH;  // For large H, use BASEH spacing
            }
            
            // Skip if h is beyond valid range
            if (h >= shape.H) continue;
            
            for (int l = 0; l < shape.L; l += BASEL) {
                in_empty.wait();
                Move_gm2ub(b, c, h, l);
                in_ready.set();
                
                out_empty.wait();
                in_ready.wait();
                Process_dacs(l);
                in_empty.set();
                out_ready.set();

                out_ready.wait();
                Move_ub2gm(b, c, h, l);
                out_empty.set();
                
                cc_cnt = (cc_cnt + 1);
            }
        }
        
        in_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Move_gm2ub(int b, int c, int h, int l){
        int h_offset = h;
        
        // Only use get_subblockid() if H > effective_H
        if (shape.H > shape.effective_H) {
            h_offset += get_subblockid() * shape.effective_H;
        }
        
        // Bounds check
        if (h_offset >= shape.H) {
            return;  // Skip invalid sub-block
        }
        
        int actual_H_to_load = min(shape.effective_H, shape.H - h_offset);
        
        // Load at_mtx
        GM2UB(at_buf.get(cc_cnt), at_mtx[h_offset], 
            1, actual_H_to_load/MTE_FLOAT, 0, 0);
        
        // Load dt_mtx with proper stride calculation
        int stride_1d = 0;
        if (shape.H > shape.effective_H) {
            stride_1d = (shape.H - shape.effective_H) / MTE_HALF;
        }
        
        GM2UB(dt_buf.get(cc_cnt), 
            dt_mtx[((b * shape.C + c) * shape.L * shape.H) + (l * shape.H) + h_offset], 
            BASEL, actual_H_to_load/MTE_HALF, stride_1d, 0);
        
        // Load dtbias_mtx
        GM2UB(dtbias_buf.get(cc_cnt), dtbias_mtx[h_offset], 
            1, actual_H_to_load/MTE_HALF, 0, 0);
        
        // Load dtmask_mtx
        GM2UB(dtmask_buf.get(cc_cnt), 
            dtmask_mtx[((b * shape.C + c) * shape.L * shape.H) + (l * shape.H) + h_offset], 
            BASEL, actual_H_to_load/MTE_HALF, stride_1d, 0);
    }

    __aicore__ inline void Process_dacs(int l){
        int vec_count = shape.tile_size / VEC_FLOAT;
        int head_vec_count = shape.effective_H / VEC_FLOAT;
        
        if (cc_cnt == 0) {
            Duplicate<float, false>(max_buf, 0.0f, MASK_PLACEHOLDER, 1, 1, NUM_DBLK_FLOAT);
            PipeBarrier<PIPE_V>();
            Adds<float, false>(max_buf, max_buf, CLAMP_MAX, MASK_PLACEHOLDER, 1, {0, 0, 0, 0});
            PipeBarrier<PIPE_V>();
        }
        
        // Cast operations - use vec_count
        Cast<float, half, false>(cc_tmp1, dt_buf.get(cc_cnt), 
                                RoundMode::CAST_NONE, MASK_PLACEHOLDER, 
                                vec_count, cast_params_h2f);
        Cast<float, half, false>(cc_tmp2, dtbias_buf.get(cc_cnt), 
                                RoundMode::CAST_NONE, MASK_PLACEHOLDER, 
                                head_vec_count, cast_params_h2f);
        Cast<float, half, false>(cc_tmp3, dtmask_buf.get(cc_cnt), 
                                RoundMode::CAST_NONE, MASK_PLACEHOLDER, 
                                vec_count, cast_params_h2f);
        PipeBarrier<PIPE_V>();
        
        // Add bias with proper stride
        auto custparam = MakeDefaultBinaryRepeatParams();
        custparam.src1RepStride = 0;
        Add<float, false>(cc_tmp1, cc_tmp1, cc_tmp2, MASK_PLACEHOLDER, 
                        vec_count, custparam);
        PipeBarrier<PIPE_V>();
        
        // Softplus computation
        CompareScalar<float, uint8_t>(cmp_mask, cc_tmp1, COMPARE_VALUE, 
                                    CMPMODE::LT, shape.tile_size);
        PipeBarrier<PIPE_V>();
        
        UB2UB(cc_tmp0, cc_tmp1, BASEL, shape.effective_H/MTE_FLOAT, 0, 0);
        PipeBarrier<PIPE_V>();
        
        Exp<float, false>(cc_tmp1, cc_tmp1, MASK_PLACEHOLDER, vec_count, unary_params);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(cc_tmp1, cc_tmp1, 1.0f, MASK_PLACEHOLDER, vec_count, unary_params);
        PipeBarrier<PIPE_V>();
        Ln<float, false>(cc_tmp1, cc_tmp1, MASK_PLACEHOLDER, vec_count, unary_params);
        PipeBarrier<PIPE_V>();
        
        Select<float, uint8_t, false>(cc_tmp1, cmp_mask, cc_tmp1, cc_tmp0, 
                                    SELMODE::VSEL_TENSOR_TENSOR_MODE, VEC_FLOAT, 
                                    vec_count, binary_params);
        PipeBarrier<PIPE_V>();
        
        // Clamp
        CompareScalar<float, uint8_t>(cmp_mask, cc_tmp1, CLAMP_MAX, 
                                    CMPMODE::LT, shape.tile_size);
        PipeBarrier<PIPE_V>();
        Select<float, uint8_t>(cc_tmp1, cmp_mask, cc_tmp1, 
                            static_cast<float>(CLAMP_MAX), 
                            SELMODE::VSEL_TENSOR_SCALAR_MODE, shape.tile_size);
        PipeBarrier<PIPE_V>();
        
        // Multiply by mask and attention
        Mul<float, false>(out1buf.get(cc_cnt), cc_tmp1, cc_tmp3, 
                        MASK_PLACEHOLDER, vec_count, binary_params);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(cc_tmp0, out1buf.get(cc_cnt), at_buf.get(cc_cnt), 
                        MASK_PLACEHOLDER, vec_count, custparam);
        PipeBarrier<PIPE_V>();
        
        // Cumulative sum
        if (l > 0) {
            Add<float, false>(cc_tmp0, cc_tmp0, cumsum_tensor.get(cc_cnt), 
                            MASK_PLACEHOLDER, head_vec_count, {1, 1, 1, 0, 0, 0});
            PipeBarrier<PIPE_V>();
        }
        
        // Sequential accumulation - use effective_H for stride
        for (int l1 = 0; l1 < BASEL - 1; ++l1) {
            Add<float, false>(cc_tmp0[(l1 + 1) * shape.effective_H], 
                            cc_tmp0[l1 * shape.effective_H], 
                            cc_tmp0[(l1 + 1) * shape.effective_H], 
                            MASK_PLACEHOLDER, head_vec_count, {1, 1, 1, 0, 0, 0});
            PipeBarrier<PIPE_V>();
        }
        
        // Save cumsum state
        if (l < shape.L) {
            UB2UB(cumsum_tensor.get(cc_cnt + 1), 
                cc_tmp0[(BASEL - 1) * shape.effective_H], 
                1, shape.effective_H/MTE_FLOAT, 0, 0);
            PipeBarrier<PIPE_V>();
        }
        
        UB2UB(out2buf.get(cc_cnt), cc_tmp0, BASEL, 
            shape.effective_H/MTE_FLOAT, 0, 0);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Move_ub2gm(int b, int c, int h, int l){
        int h_offset = h;
        if (shape.H > shape.effective_H) {
            h_offset += get_subblockid() * shape.effective_H;
        }
        
        if (h_offset >= shape.H) {
            return;  // Skip invalid sub-block
        }
        
        int actual_H_to_store = min(shape.effective_H, shape.H - h_offset);
        int stride_1d = 0;
        if (shape.H > shape.effective_H) {
            stride_1d = (shape.H - shape.effective_H) / MTE_FLOAT;
        }
        
        // Write out_mtx and out1_mtx
        UB2GM(out_mtx[((b * shape.C + c) * shape.L * shape.H) + (l * shape.H) + h_offset], 
            out1buf.get(cc_cnt), BASEL, actual_H_to_store/MTE_FLOAT, 0, stride_1d);
        UB2GM(out1_mtx[((b * shape.C + c) * shape.L * shape.H) + (l * shape.H) + h_offset], 
            out2buf.get(cc_cnt), BASEL, actual_H_to_store/MTE_FLOAT, 0, stride_1d);
        
        // Write final cumsum state
        if ((l + BASEL) >= shape.L) {
            UB2GM(out2_mtx[(b * shape.C + c) * shape.H + h_offset], 
                out2buf.get(cc_cnt)[(BASEL - 1) * shape.effective_H], 
                1, actual_H_to_store/MTE_FLOAT, 0, stride_1d);
        }
    }
    
private:
    CustVecShapeInfo shape;
    // Global Params
    int cc_cnt; 
    UnaryRepeatParams cast_params_h2f;
    UnaryRepeatParams cast_params_f2h;
    UnaryRepeatParams unary_params;
    BinaryRepeatParams binary_params;

    // Global Tensors
    GlobalTensor<float> at_mtx;
    GlobalTensor<half> dt_mtx;
    GlobalTensor<half> dtbias_mtx;
    GlobalTensor<half> dtmask_mtx;
    GlobalTensor<float> out_mtx;
    GlobalTensor<float> out1_mtx;
    GlobalTensor<float> out2_mtx;
    // Local buffers
    DBuff<float, TPosition::VECCALC> at_buf;
    DBuff<half, TPosition::VECCALC> dt_buf;
    DBuff<half, TPosition::VECCALC> dtbias_buf;
    DBuff<half, TPosition::VECCALC> dtmask_buf;
    DBuff<float, TPosition::VECCALC> out1buf;
    DBuff<float, TPosition::VECCALC> out2buf;
    LocalTensor<float> cc_tmp0;
    LocalTensor<float> cc_tmp1;
    LocalTensor<float> cc_tmp2;
    LocalTensor<float> cc_tmp3;
    LocalTensor<uint8_t> cmp_mask;
    LocalTensor<float> max_buf;
    DBuff<float, TPosition::VECCALC> cumsum_tensor;
    
    // Double events
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    // User-defined events
};
} // namespace Mambav2ChunkCumsum
} // namespace npu_ops_transformer_ext
