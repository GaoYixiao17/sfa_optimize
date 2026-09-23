/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#pragma once
#include "tensorutils.h"
#include "paramutils.h"

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkStatePassing {

struct CustVecShapeInfo{
    int B;
    int H;
    int S;
    int C;
    int L;
    int G;
    int N;
    int P;
    int Z;
    int stride_B;
    int stride_C;
    int stride_H;
    int headPerCore;
    int z_pervec;
};


__aicore__ inline void tilingShapeCustVec(int B, int H, int S, int C, int L, int G, int N, int P, int Z, CustVecShapeInfo &shape){
    shape.B = B;
    shape.H = H;
    shape.S = S;
    shape.C = C;
    shape.L = L;
    shape.G = G;
    shape.N = N;
    shape.P = P;
    shape.Z = Z;
    shape.stride_B = ((shape.C * shape.H) * shape.Z);
    shape.stride_C = (shape.H * shape.Z);
    shape.stride_H = shape.Z;
    shape.headPerCore = CeilDiv((B * H),GetBlockNum());
    shape.z_pervec = ((int)shape.Z / (int)FOUR);
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR dacs_, GM_ADDR initmtx_, GM_ADDR statemtx_, GM_ADDR tmpmtx_, GM_ADDR state_fp16_, GM_ADDR final_state_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        dacs.SetGlobalBuffer((__gm__ float*) dacs_);
        initmtx.SetGlobalBuffer((__gm__ float*) initmtx_);
        statemtx.SetGlobalBuffer((__gm__ float*) statemtx_);
        tmpmtx.SetGlobalBuffer((__gm__ float*) tmpmtx_);
        state_fp16.SetGlobalBuffer((__gm__ half*) state_fp16_);
        final_state.SetGlobalBuffer((__gm__ float*) final_state_);
        
        // Local buffers
        dacsbuf.Init(MTE_FLOAT);
        dacsbrcbbuf1.Init(NUM_DBLK_FLOAT);
        dacsbrcbbuf2.Init(VEC_FLOAT);
        tmpbuf.Init(shape.z_pervec);
        outbuf.Init(shape.z_pervec);
        fp32_statebuf.Init(shape.z_pervec);
        fp16_statebuf_c0.Init(shape.z_pervec);
        fp16_statebuf_ci.Init(shape.z_pervec);
        // Initialize events
        in_empty.Init();
        in_ready.Init();
        out_empty.Init();
        out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        len_h = ((shape.headPerCore)<(((shape.B * shape.H) - (get_block_idx() * shape.headPerCore)))) ? (shape.headPerCore) : (((shape.B * shape.H) - (get_block_idx() * shape.headPerCore)));
        cnt = 0;
        ws_cnt = 0;
        cast_params_h2f = CastHalf2FloatRepeatParams();
        cast_params_f2h = CastFloat2HalfRepeatParams();
        unary_params = MakeDefaultUnaryRepeatParams();
        binary_params = MakeDefaultBinaryRepeatParams();

        for (int chunk_id=0; chunk_id<shape.C; chunk_id+=1){
            if ((chunk_id == 0)){
                Process_chunk0(chunk_id);
            }
            if ((chunk_id < (shape.C - 1))){
                // handshake with cube
                Process_chunks(chunk_id);
            }
        }

        Process_final_state(shape.C-1);
        
        WAIT_CUBE(0);
        WAIT_CUBE(0);
        WAIT_CUBE(0);
        in_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Process_chunk0(int chunk_id){
        int bh_index = (get_block_idx() * shape.headPerCore);
        int subBlockid = get_subblockid();
        int base_b = 0;
        int base_h = 0;
        for (int head_id=0; head_id<len_h; head_id+=1){
            base_b = ((int)bh_index / (int)shape.H);
            base_h = (bh_index % shape.H);
            WAIT_CUBE(0);
            for (int base_z=0; base_z<shape.Z; base_z+=(shape.z_pervec * TWO)){
                in_empty.wait();
                GM2UB(tmpbuf.get(cnt), initmtx[((((base_b * shape.stride_C) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                in_ready.set();
                out_empty.wait();
                in_ready.wait();
                Cast<half, float, false>(fp16_statebuf_c0.get(cnt), tmpbuf.get(cnt), RoundMode::CAST_RINT, MASK_PLACEHOLDER, ((int)shape.z_pervec / VEC_FLOAT), cast_params_f2h);
                in_empty.set();
                out_ready.set();
                out_ready.wait();
                UB2GM(state_fp16[((((((ws_cnt % THREE) * GetBlockNum()) * shape.Z) + (get_block_idx() * shape.Z)) + base_z) + (subBlockid * shape.z_pervec))], fp16_statebuf_c0.get(cnt), 1, ((int)shape.z_pervec / MTE_HALF), 0, 0);
                out_empty.set();
                cnt = (cnt + 1);
            }
            VEC_READY(0, PIPE_MTE3);
            ws_cnt = (ws_cnt + 1);
            bh_index = (bh_index + 1);
        }
    }

    __aicore__ inline void Process_chunks(int chunk_id){
        int bh_index = (get_block_idx() * shape.headPerCore);
        int subBlockid = get_subblockid();
        int base_b = 0;
        int base_h = 0;
        for (int head_id=0; head_id<len_h; head_id+=1){
            base_b = ((int)bh_index / (int)shape.H);
            base_h = (bh_index % shape.H);
            WAIT_CUBE(0);
            for (int base_z=0; base_z<shape.Z; base_z+=(shape.z_pervec * TWO)){
                in_empty.wait();
                GM2UB(fp32_statebuf.get(cnt), statemtx[(((((base_b * shape.stride_B) + (chunk_id * shape.stride_C)) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                GM2UB(dacsbuf.get(cnt), dacs[((((base_b * shape.C) * shape.H) + (chunk_id * shape.H)) + base_h)], 1, 1, 0, 0);
                if ((chunk_id == 0)){
                    GM2UB(tmpbuf.get(cnt), initmtx[((((base_b * shape.stride_C) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                }
                if ((chunk_id > 0)){
                    GM2UB(tmpbuf.get(cnt), tmpmtx[((((base_b * shape.stride_C) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                }
                in_ready.set();
                
                out_empty.wait();
                in_ready.wait();
                Brcb(dacsbrcbbuf1.get(cnt), dacsbuf.get(cnt), 1, {1, NUM_DBLK_FLOAT});
                PipeBarrier<PIPE_V>();
                Brcb(dacsbrcbbuf2.get(cnt), dacsbrcbbuf1.get(cnt), 1, {1, NUM_DBLK_FLOAT});
                PipeBarrier<PIPE_V>();
                Exp<float, false>(dacsbrcbbuf2.get(cnt), dacsbrcbbuf2.get(cnt), MASK_PLACEHOLDER, 1, unary_params);
                PipeBarrier<PIPE_V>();
                auto custparam = MakeDefaultBinaryRepeatParams();
                custparam.src1RepStride = 0;
                Mul<float, false>(tmpbuf.get(cnt), tmpbuf.get(cnt), dacsbrcbbuf2.get(cnt), MASK_PLACEHOLDER, ((int)shape.z_pervec / VEC_FLOAT), custparam);
                PipeBarrier<PIPE_V>();
                Add<float, false>(tmpbuf.get(cnt), tmpbuf.get(cnt), fp32_statebuf.get(cnt), MASK_PLACEHOLDER, ((int)shape.z_pervec / VEC_FLOAT), binary_params);
                PipeBarrier<PIPE_V>();
                Cast<half, float, false>(fp16_statebuf_ci.get(cnt), tmpbuf.get(cnt), RoundMode::CAST_RINT, MASK_PLACEHOLDER, ((int)shape.z_pervec / VEC_FLOAT), cast_params_f2h);
                PipeBarrier<PIPE_V>();
                UB2UB(outbuf.get(cnt), tmpbuf.get(cnt), 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                PipeBarrier<PIPE_V>();
                in_empty.set();
                out_ready.set();

                out_ready.wait();
                UB2GM(state_fp16[((((((ws_cnt % THREE) * GetBlockNum()) * shape.Z) + (get_block_idx() * shape.Z)) + base_z) + (subBlockid * shape.z_pervec))], fp16_statebuf_ci.get(cnt), 1, ((int)shape.z_pervec / MTE_HALF), 0, 0);
                UB2GM(tmpmtx[((((base_b * shape.stride_B) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], outbuf.get(cnt), 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                out_empty.set();
                cnt = (cnt + 1);
            }
            VEC_READY(0, PIPE_MTE3);
            ws_cnt = (ws_cnt + 1);
            bh_index = (bh_index + 1);
        }
    }

    __aicore__ inline void Process_final_state(int chunk_id){
        int bh_index = (get_block_idx() * shape.headPerCore);
        int subBlockid = get_subblockid();
        int base_b = 0;
        int base_h = 0;
        for (int head_id=0; head_id<len_h; head_id+=1){
            base_b = ((int)bh_index / (int)shape.H);
            base_h = (bh_index % shape.H);
            for (int base_z=0; base_z<shape.Z; base_z+=(shape.z_pervec * TWO)){
                in_empty.wait();
                GM2UB(fp32_statebuf.get(cnt), statemtx[(((((base_b * shape.stride_B) + ((shape.C - 1) * shape.stride_C)) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                GM2UB(dacsbuf.get(cnt), dacs[((((base_b * shape.C) * shape.H) + ((shape.C - 1) * shape.H)) + base_h)], 1, 1, 0, 0);
                if (((shape.C - 1) == 0)){
                    GM2UB(tmpbuf.get(cnt), initmtx[((((base_b * shape.stride_C) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                }
                if (((shape.C - 1) > 0)){
                    GM2UB(tmpbuf.get(cnt), tmpmtx[((((base_b * shape.stride_C) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                }
                in_ready.set();

                out_empty.wait();
                in_ready.wait();
                Brcb(dacsbrcbbuf1.get(cnt), dacsbuf.get(cnt), 1, {1, NUM_DBLK_FLOAT});
                PipeBarrier<PIPE_V>();
                Brcb(dacsbrcbbuf2.get(cnt), dacsbrcbbuf1.get(cnt), 1, {1, NUM_DBLK_FLOAT});
                PipeBarrier<PIPE_V>();
                Exp<float, false>(dacsbrcbbuf2.get(cnt), dacsbrcbbuf2.get(cnt), MASK_PLACEHOLDER, 1, unary_params);
                PipeBarrier<PIPE_V>();
                auto custparam1 = MakeDefaultBinaryRepeatParams();
                custparam1.src1RepStride = 0;
                Mul<float, false>(tmpbuf.get(cnt), tmpbuf.get(cnt), dacsbrcbbuf2.get(cnt), MASK_PLACEHOLDER, ((int)shape.z_pervec / VEC_FLOAT), custparam1);
                PipeBarrier<PIPE_V>();
                Add<float, false>(tmpbuf.get(cnt), tmpbuf.get(cnt), fp32_statebuf.get(cnt), MASK_PLACEHOLDER, ((int)shape.z_pervec / VEC_FLOAT), binary_params);
                PipeBarrier<PIPE_V>();
                Cast<half, float, false>(fp16_statebuf_ci.get(cnt), tmpbuf.get(cnt), RoundMode::CAST_RINT, MASK_PLACEHOLDER, ((int)shape.z_pervec / VEC_FLOAT), cast_params_f2h);
                PipeBarrier<PIPE_V>();
                UB2UB(outbuf.get(cnt), tmpbuf.get(cnt), 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                PipeBarrier<PIPE_V>();
                in_empty.set();
                out_ready.set();

                out_ready.wait();
                UB2GM(final_state[((((base_b * shape.stride_B) + (base_h * shape.stride_H)) + base_z) + (subBlockid * shape.z_pervec))], outbuf.get(cnt), 1, ((int)shape.z_pervec / MTE_FLOAT), 0, 0);
                out_empty.set();
                cnt = (cnt + 1);
            }
            bh_index = (bh_index + 1);
        }
    }
    
private:
    CustVecShapeInfo shape;
    // Global Params
    int len_h;
    int cnt;
    int ws_cnt;
    UnaryRepeatParams cast_params_h2f;
    UnaryRepeatParams cast_params_f2h;
    UnaryRepeatParams unary_params;
    BinaryRepeatParams binary_params;
    // Global Tensors
    GlobalTensor<float> dacs;
    GlobalTensor<float> initmtx;
    GlobalTensor<float> statemtx;
    GlobalTensor<float> tmpmtx;
    GlobalTensor<half> state_fp16;
    GlobalTensor<float> final_state;
    // Local buffers
    DBuff<float, TPosition::VECCALC> dacsbuf;
    DBuff<float, TPosition::VECCALC> dacsbrcbbuf1;
    DBuff<float, TPosition::VECCALC> dacsbrcbbuf2;
    DBuff<float, TPosition::VECCALC> tmpbuf;
    DBuff<float, TPosition::VECCALC> outbuf;
    DBuff<float, TPosition::VECCALC> fp32_statebuf;
    DBuff<half, TPosition::VECCALC> fp16_statebuf_c0;
    DBuff<half, TPosition::VECCALC> fp16_statebuf_ci;
    
    // Events
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
};

} // namespace Mambav2ChunkStatePassing
} // namespace npu_ops_transformer_ext