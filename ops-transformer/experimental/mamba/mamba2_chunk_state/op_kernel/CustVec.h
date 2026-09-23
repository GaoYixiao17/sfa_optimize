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
namespace Mambav2ChunkState {

constexpr int BASEH = 8;
constexpr int BASEL = 128;
constexpr int CBASEM = 64;
constexpr int CBASEN = 64;
constexpr int CBASEK = 256;

struct CustVecShapeInfo{
    int BCH;
    int BCH_PER_CORE;
    int BCH1;
    int BCH2;
    int C;
    int H;
    int G;
    int L;
    int N;
    int BASEH;
    int BASEN;
    int group_size;
    int BASEL;
    int repeatL;
    int subvec_L;
    int BASEK;
};


__aicore__ inline void tilingShapeCustVec(int B, int C, int H, int G, int L, int N, CustVecShapeInfo &shape){
    shape.BCH = ((int)(B * C * H) / (int)BASEH);
    shape.BCH_PER_CORE = CeilDiv(shape.BCH,GetBlockNum());
    shape.BCH1 = (shape.BCH_PER_CORE * get_block_idx());
    shape.BCH2 = (((shape.BCH_PER_CORE + shape.BCH1))<(shape.BCH)) ? ((shape.BCH_PER_CORE + shape.BCH1)) : (shape.BCH);
    shape.C = C;
    shape.H = H;
    shape.G = G;
    shape.L = L;
    shape.H = H;
    shape.N = N;
    shape.BASEH = BASEH;
    shape.BASEN = CBASEM;
    shape.group_size = ((int)H / (int)G);
    shape.BASEL = BASEL;
    shape.repeatL = ((int)shape.L / (int)shape.BASEL);
    shape.subvec_L = ((int)shape.BASEL / (int)TWO);
    shape.BASEK = CBASEK;
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR da_out_, GM_ADDR dtout_mtx_, GM_ADDR dacs_mtx_, GM_ADDR bt_mtx_, GM_ADDR vec_out_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        da_out.SetGlobalBuffer((__gm__ float*) da_out_);
        dtout_mtx.SetGlobalBuffer((__gm__ float*) dtout_mtx_);
        dacs_mtx.SetGlobalBuffer((__gm__ float*) dacs_mtx_);
        bt_mtx.SetGlobalBuffer((__gm__ half*) bt_mtx_);
        vec_out.SetGlobalBuffer((__gm__ half*) vec_out_);
        
        // Local buffers
        dacs.Init((shape.subvec_L * shape.BASEH));
        dacs_t.Init(shape.BASEH);
        dacs_brcb.Init((shape.subvec_L * shape.BASEH * NUM_DBLK_FLOAT));
        dacs_t_brcb.Init((shape.BASEH * NUM_DBLK_FLOAT));
        dtout.Init((shape.subvec_L * shape.BASEH));
        da_out_fp32.Init((shape.subvec_L * shape.BASEH));
        da.Init((shape.subvec_L * shape.BASEH));
        da_brcb.Init(((shape.subvec_L * shape.BASEH) * NUM_DBLK_FLOAT));
        bt_half.Init((shape.subvec_L * shape.BASEN));
        bt_fp32.Init((shape.subvec_L * shape.BASEN));
        out_fp32.Init((shape.subvec_L * shape.BASEN));
        out_half.Init((shape.subvec_L * shape.BASEN));
        // Initialize events
        in_empty.Init();
        in_ready.Init();
        out_empty.Init();
        out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        da_cnt = 0;
        cs_cnt = 0;
        cs_cnt2 = 0;
        cs_cnt3 = 0;
        ws_cnt = 0;
        cast_params_h2f = CastHalf2FloatRepeatParams();
        cast_params_f2h = CastFloat2HalfRepeatParams();
        unary_params = MakeDefaultUnaryRepeatParams();
        binary_params = MakeDefaultBinaryRepeatParams();
        
        for (int bch=shape.BCH1; bch<shape.BCH2; bch+=1){
            int b = ((int)bch / (int)((int)(shape.C * shape.H) / (int)shape.BASEH));
            int c = ((int)(bch % ((int)(shape.C * shape.H) / (int)shape.BASEH)) / (int)((int)shape.H / (int)shape.BASEH));
            int h = ((bch % ((int)(shape.C * shape.H) / (int)shape.BASEH)) % ((int)shape.H / (int)shape.BASEH));
            for (int r=0; r<shape.repeatL; r+=1){
                Process_part1(b, c, h, r);
            }
            PipeBarrier<PIPE_ALL>();
            for (int madn=0; madn<shape.N; madn+=shape.BASEN){
                WAIT_CUBE(0);
                for (int rp=0; rp<shape.repeatL; rp+=1){
                    for (int madh=0; madh<shape.BASEH; madh+=1){
                        Process_part2(b, c, h, madn, rp, madh, ws_cnt);
                    }
                }
                VEC_READY(0, PIPE_MTE3);
                ws_cnt = (ws_cnt + 1);
            }
        }
        WAIT_CUBE(0);
        WAIT_CUBE(0);
        WAIT_CUBE(0);
        in_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Process_part1(int b, int c, int h, int r){
        in_empty.wait();
        GM2UB(dacs.get(da_cnt), dacs_mtx[((((b * ((shape.C * shape.L) * shape.H)) + (c * (shape.L * shape.H))) + (((r * shape.BASEL) + (get_subblockid() * shape.subvec_L)) * shape.H)) + (h * shape.BASEH))], shape.subvec_L, ((int)shape.BASEH / MTE_FLOAT), ((int)(shape.H - shape.BASEH) / MTE_FLOAT), 0);
        GM2UB(dacs_t.get(da_cnt), dacs_mtx[((((b * ((shape.C * shape.L) * shape.H)) + (c * (shape.L * shape.H))) + ((shape.L - 1) * shape.H)) + (h * shape.BASEH))], 1, 1, 0, 0);
        GM2UB(dtout.get(da_cnt), dtout_mtx[((((b * ((shape.C * shape.L) * shape.H)) + (c * (shape.L * shape.H))) + (((r * shape.BASEL) + (get_subblockid() * shape.subvec_L)) * shape.H)) + (h * shape.BASEH))], shape.subvec_L, 1, ((int)(shape.H - shape.BASEH) / (int)MTE_FLOAT), 0);
        in_ready.set();

        out_empty.wait();
        in_ready.wait();
        Brcb(dacs_brcb.get(da_cnt), dacs.get(da_cnt), shape.subvec_L, {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();
        Brcb(dacs_t_brcb.get(da_cnt), dacs_t.get(da_cnt), 1, {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();
        auto custparam = MakeDefaultBinaryRepeatParams();
        custparam.src0RepStride = 0; // {1,1,1,8,0,8}
        Sub<float, false>(dacs_brcb.get(da_cnt), dacs_t_brcb.get(da_cnt), dacs_brcb.get(da_cnt), MASK_PLACEHOLDER, shape.subvec_L, custparam);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(dacs.get(da_cnt), dacs_brcb.get(da_cnt), shape.subvec_L, MASK_PLACEHOLDER, 1, 1, NUM_DBLK_FLOAT);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(dacs.get(da_cnt), dacs.get(da_cnt), MASK_PLACEHOLDER, ((int)(shape.subvec_L * shape.BASEH) / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(out_fp32.get(da_cnt), dtout.get(da_cnt), dacs.get(da_cnt), MASK_PLACEHOLDER, ((int)(shape.subvec_L * shape.BASEH) / (int)VEC_FLOAT), binary_params);
        in_empty.set();
        out_ready.set();

        out_ready.wait();
        UB2GM(da_out[(((get_block_idx() * shape.L) * shape.BASEH) + (((r * shape.BASEL) + (get_subblockid() * shape.subvec_L)) * shape.BASEH))], out_fp32.get(da_cnt), shape.subvec_L, 1, 0, 0);
        out_empty.set();
        da_cnt = (da_cnt + 1);
    }

    __aicore__ inline void Process_part2(int b, int c, int h, int madn, int rp, int madh, int ws_cnt){
        in_empty.wait();
        if ((madh == 0)){
            GM2UB(da.get(cs_cnt), da_out[(((get_block_idx() * shape.L) * shape.BASEH) + (((rp * shape.BASEL) + (get_subblockid() * shape.subvec_L)) * shape.BASEH))], shape.subvec_L, ((int)shape.BASEH / (int)MTE_FLOAT), 0, 0);
        }
        if (((madh % shape.group_size) == 0)){
            GM2UB(bt_half.get(cs_cnt3), bt_mtx[(((((b * (((shape.C * shape.L) * shape.G) * shape.N)) + (c * ((shape.L * shape.G) * shape.N))) + (((rp * shape.BASEL) + (get_subblockid() * shape.subvec_L)) * (shape.G * shape.N))) + (((int)((h * shape.BASEH) + madh) / (int)shape.group_size) * shape.N)) + madn)], shape.subvec_L, ((int)shape.BASEN / (int)MTE_HALF), ((int)((shape.G * shape.N) - shape.BASEN) / (int)MTE_HALF), 0);
        }
        in_ready.set();
        out_empty.wait();
        in_ready.wait();
        if ((madh == 0)){
            Brcb(da_brcb.get(cs_cnt), da.get(cs_cnt), shape.subvec_L, {1, NUM_DBLK_FLOAT});
            PipeBarrier<PIPE_V>();
        }
        if (((madh % shape.group_size) == 0)){
            Cast<float, half, false>(bt_fp32.get(cs_cnt3), bt_half.get(cs_cnt3), RoundMode::CAST_NONE, MASK_PLACEHOLDER, ((int)(shape.subvec_L * shape.BASEN) / (int)VEC_FLOAT), cast_params_h2f);
            PipeBarrier<PIPE_V>();
        }
        auto custparam1 = MakeDefaultBinaryRepeatParams();
        custparam1.dstBlkStride = (uint8_t)((int)shape.BASEN / NUM_ELE_PERBLK_FLOAT); 
        custparam1.src0BlkStride = (uint8_t)((int)shape.BASEN / NUM_ELE_PERBLK_FLOAT);
        custparam1.src1BlkStride = (uint8_t)((int)(NUM_DBLK_FLOAT * shape.BASEH) / NUM_ELE_PERBLK_FLOAT);
        custparam1.dstRepStride = (uint8_t)((int)(NUM_DBLK_FLOAT * shape.BASEN) / NUM_ELE_PERBLK_FLOAT);
        custparam1.src0RepStride = (uint8_t)((int)(NUM_DBLK_FLOAT * shape.BASEN) / NUM_ELE_PERBLK_FLOAT);
        custparam1.src1RepStride = (uint8_t)((int)(NUM_DBLK_FLOAT * (shape.BASEH * NUM_DBLK_FLOAT)) / NUM_ELE_PERBLK_FLOAT);

        for (int rb=0; rb<shape.BASEN; rb+=NUM_DBLK_FLOAT){
            Mul<float, false>(out_fp32.get(cs_cnt2)[rb], bt_fp32.get(cs_cnt3)[rb], da_brcb.get(cs_cnt)[(madh * NUM_DBLK_FLOAT)], MASK_PLACEHOLDER, ((int)(shape.subvec_L * shape.BASEH) / (int)VEC_FLOAT), custparam1);
            PipeBarrier<PIPE_V>();
        }
        Cast<half, float, false>(out_half.get(cs_cnt2), out_fp32.get(cs_cnt2), RoundMode::CAST_RINT, MASK_PLACEHOLDER, ((int)(shape.subvec_L * shape.BASEN) / (int)VEC_FLOAT), cast_params_f2h);
        in_empty.set();
        out_ready.set();
        out_ready.wait();
        UB2GM(vec_out[(((((ws_cnt % THREE) * (((GetBlockNum() * shape.BASEH) * shape.L) * shape.BASEN)) + (((get_block_idx() * shape.BASEH) * shape.L) * shape.BASEN)) + ((madh * shape.L) * shape.BASEN)) + (((rp * shape.BASEL) + (get_subblockid() * shape.subvec_L)) * shape.BASEN))], out_half.get(cs_cnt2), shape.subvec_L, ((int)shape.BASEN / (int)MTE_HALF), 0, 0);
        out_empty.set();
        cs_cnt2 = (cs_cnt2 + 1);
        if ((((madh + 1) % shape.group_size) == 0)){
            cs_cnt3 = (cs_cnt3 + 1);
        }
    }
    
private:
    CustVecShapeInfo shape;
    // Global Params
    int da_cnt;
    int cs_cnt;
    int cs_cnt2;
    int cs_cnt3;
    int ws_cnt;
    UnaryRepeatParams cast_params_h2f;
    UnaryRepeatParams cast_params_f2h;
    UnaryRepeatParams unary_params;
    BinaryRepeatParams binary_params;

    // Global Tensors
    GlobalTensor<float> da_out;
    GlobalTensor<float> dtout_mtx;
    GlobalTensor<float> dacs_mtx;
    GlobalTensor<half> bt_mtx;
    GlobalTensor<half> vec_out;
    // Local buffers
    DBuff<float, TPosition::VECCALC> dacs;
    DBuff<float, TPosition::VECCALC> dacs_t;
    DBuff<float, TPosition::VECCALC> dacs_brcb;
    DBuff<float, TPosition::VECCALC> dacs_t_brcb;
    DBuff<float, TPosition::VECCALC> dtout;
    DBuff<float, TPosition::VECCALC> da_out_fp32;
    DBuff<float, TPosition::VECCALC> da;
    DBuff<float, TPosition::VECCALC> da_brcb;
    DBuff<half, TPosition::VECCALC> bt_half;
    DBuff<float, TPosition::VECCALC> bt_fp32;
    DBuff<float, TPosition::VECCALC> out_fp32;
    DBuff<half, TPosition::VECCALC> out_half;
    
    // Events
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
};

} // Mambav2ChunkState
} // npu_ops_transformer_ext