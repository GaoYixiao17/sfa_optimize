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
namespace Mambav2ChunkScan {

constexpr int VBASEL = 64;
constexpr int VBASEH = 8;
constexpr int BRCB_BLK = 64;
constexpr int CBASEM = 64;
constexpr int CBASEN = 64;
constexpr int BLK_K = 64;
constexpr int MASKV = 4294967295;

struct CustVecShapeInfo{
    int BCH;
    int H;
    int G;
    int L;
    int N;
    int P;
    int HPERG;
    int BCH_PERCORE;
    int BCH1;
    int BCH2;
    int BASEL;
};


__aicore__ inline void tilingShapeCustVec(int B, int C, int H, int G, int L, int N, int P, CustVecShapeInfo &shape){
    shape.BCH = ((B * C) * H);
    shape.H = H;
    shape.G = G;
    shape.L = L;
    shape.N = N;
    shape.P = P;
    shape.HPERG = ((int)H / (int)G);
    shape.BCH_PERCORE = CeilDiv(shape.BCH,GetBlockNum());
    shape.BCH1 = (shape.BCH_PERCORE * get_block_idx());
    shape.BCH2 = (((shape.BCH1 + shape.BCH_PERCORE))<(shape.BCH)) ? ((shape.BCH1 + shape.BCH_PERCORE)) : (shape.BCH);
    shape.BASEL = VBASEL;
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR cb_ws_, GM_ADDR dacsmtx_, GM_ADDR dtoutmtx_, GM_ADDR mmtx_, GM_ADDR outmtx_, GM_ADDR statesmtx_, GM_ADDR xmtx_, GM_ADDR dmtx_, GM_ADDR sumoutmtx_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        cb_ws.SetGlobalBuffer((__gm__ float*) cb_ws_);
        dacsmtx.SetGlobalBuffer((__gm__ float*) dacsmtx_);
        dtoutmtx.SetGlobalBuffer((__gm__ float*) dtoutmtx_);
        mmtx.SetGlobalBuffer((__gm__ half*) mmtx_);
        outmtx.SetGlobalBuffer((__gm__ float*) outmtx_);
        statesmtx.SetGlobalBuffer((__gm__ float*) statesmtx_);
        xmtx.SetGlobalBuffer((__gm__ half*) xmtx_);
        dmtx.SetGlobalBuffer((__gm__ half*) dmtx_);
        sumoutmtx.SetGlobalBuffer((__gm__ float*) sumoutmtx_);
        
        // Local buffers
        dacs_buf1.Init(shape.BASEL);
        AllocateLocalTensor<TPosition::VECCALC>(dacs_brcb, (shape.BASEL * VBASEH));
        AllocateLocalTensor<TPosition::VECCALC>(dacs_brcb2, (shape.BASEL * BRCB_BLK));
        AllocateLocalTensor<TPosition::VECCALC>(cb_buf, (shape.BASEL * shape.BASEL));
        AllocateLocalTensor<TPosition::VECCALC>(da_out, (shape.BASEL * shape.BASEL));
        AllocateLocalTensor<TPosition::VECCALC>(da_out_half, (shape.BASEL * shape.BASEL));
        dacs_buf2.Init(shape.BASEL);
        dtout_buf.Init(shape.BASEL);
        AllocateLocalTensor<TPosition::VECCALC>(maskdiagbuf, (shape.BASEL * shape.BASEL));
        scalea.Init(shape.BASEL);
        AllocateLocalTensor<TPosition::VECCALC>(scalea_brcb, (shape.BASEL * VBASEH));
        AllocateLocalTensor<TPosition::VECCALC>(scalea_brcb2, (shape.BASEL * BRCB_BLK));
        AllocateLocalTensor<TPosition::VECCALC>(states, (shape.BASEL * shape.P));
        out_b.Init((shape.BASEL * shape.P));
        x_half.Init((shape.BASEL * shape.P));
        sumout.Init((shape.BASEL * shape.P));
        // Initialize events
        in_empty.Init();
        in_ready.Init();
        out_empty.Init();
        out_ready.Init();
        tensor_in_empty.Init();
        tensor_in_ready.Init();
        tensor_out_empty.Init();
        tensor_out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        cnt1 = 0;
        cnt2 = 0;
        v1_cnt1 = 0;
        v1_cnt2 = 0;
        bc_now = v1_bc_now = 0;
        h_now = v1_h_now = 0;
        
        cast_params_h2f = CastHalf2FloatRepeatParams();
        cast_params_f2h = CastFloat2HalfRepeatParams();
        unary_params = MakeDefaultUnaryRepeatParams();
        binary_params = MakeDefaultBinaryRepeatParams();
        
        Init_mask();

        tensor_in_empty.set();
        tensor_out_empty.set();
        VEC_READY(THREE, PIPE_MTE3);
        VEC_READY(THREE, PIPE_MTE3);
        for (int bch=shape.BCH1; bch<(shape.BCH2 + 1); bch+=1){
            bc_now = ((int)bch / (int)shape.H);
            h_now = (bch % shape.H);
            int g = ((int)h_now / (int)shape.HPERG);
            if ((bch < shape.BCH2)){
                if ((((h_now % shape.HPERG) == 0) || (bch == shape.BCH1))){
                    WAIT_CUBE(0);
                }
                WAIT_CUBE(THREE);
                Process_dt();
                VEC_READY(0, PIPE_MTE3);
            }
            if ((bch > shape.BCH1)){
                v1_bc_now = ((int)(bch - 1) / (int)shape.H);
                v1_h_now = ((bch - 1) % shape.H);
                WAIT_CUBE(1);
                Process_out();
                VEC_READY(THREE, PIPE_MTE3);
            }
        }
        WAIT_CUBE(THREE);
        WAIT_CUBE(THREE);
        tensor_in_empty.wait();
        tensor_out_empty.wait();
        in_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Init_mask(){
        LocalTensor<uint32_t> tmptsr_0 = maskdiagbuf.ReinterpretCast<uint32_t>();
        Duplicate<uint32_t, false>(tmptsr_0, (uint32_t)0, MASK_PLACEHOLDER, CBASEM, 1, MTE_FLOAT);
        PipeBarrier<PIPE_V>();
        mask = 1;
        for (int i=0; i<CBASEM; i+=1){
            SetVectorMask<half, MaskMode::NORMAL>(0, mask);
            LocalTensor<uint32_t> tmptsr_1 = maskdiagbuf[(i * CBASEM)].ReinterpretCast<uint32_t>();
            Duplicate<uint32_t, false>(tmptsr_1, (uint32_t)MASKV, MASK_PLACEHOLDER, 1, 1, MTE_FLOAT);
            PipeBarrier<PIPE_V>();
            mask = ((mask * TWO) + 1);
        }
        SetVectorMask<half, MaskMode::NORMAL>(-1, -1);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process_dt(){
        int m1start = 0;
        int m1end = 0;
        int m1 = 0;
        if ((get_subblockid() == 0)){
            m1start = THREE;
            m1end = FIVE;
        }
        else {
            m1start = 1;
            m1end = THREE;
        }
        for (int m1_raw=m1start; m1_raw<m1end; m1_raw+=1){
            m1 = (m1_raw % FOUR);
            for (int m2=0; m2<FOUR; m2+=1){
                in_empty.wait();
                tensor_in_empty.wait();
                if ((m2 == 0)){
                    GM2UB(dacs_buf1.get(cnt1), dacsmtx[(((bc_now * (shape.H * shape.L)) + (h_now * shape.L)) + ((m1 * CBASEM) * 1))], 1, ((int)shape.BASEL / (int)MTE_FLOAT), 0, 0);
                }
                if ((m2 <= m1)){
                    GM2UB(dacs_buf2.get(cnt2), dacsmtx[(((bc_now * (shape.H * shape.L)) + (h_now * shape.L)) + ((m2 * BLK_K) * 1))], 1, ((int)shape.BASEL / (int)MTE_FLOAT), 0, 0);
                    GM2UB(dtout_buf.get(cnt2), dtoutmtx[(((bc_now * (shape.H * shape.L)) + (h_now * shape.L)) + ((m2 * BLK_K) * 1))], 1, ((int)shape.BASEL / (int)MTE_FLOAT), 0, 0);
                    GM2UB(cb_buf, cb_ws[(((get_block_idx() * (shape.L * shape.L)) + ((m1 * CBASEM) * shape.L)) + ((m2 * BLK_K) * 1))], shape.BASEL, ((int)shape.BASEL / (int)MTE_FLOAT), ((int)(shape.L - shape.BASEL) / (int)MTE_FLOAT), 0);
                }
                in_ready.set();
                tensor_in_ready.set();

                out_empty.wait();
                in_ready.wait();
                tensor_out_empty.wait();
                tensor_in_ready.wait();
                Process_dt_calc(m1, m2);
                out_ready.set();
                in_empty.set();
                tensor_out_ready.set();
                tensor_in_empty.set();

                out_ready.wait();
                tensor_out_ready.wait();
                UB2GM(mmtx[((((bc_now * ((shape.H * shape.L) * shape.L)) + (h_now * (shape.L * shape.L))) + ((m1 * CBASEM) * shape.L)) + ((m2 * BLK_K) * 1))], da_out_half, shape.BASEL, ((int)shape.BASEL / (int)MTE_HALF), 0, ((int)(shape.L - shape.BASEL) / (int)MTE_HALF));
                out_empty.set();
                tensor_out_empty.set();
                cnt2 = (cnt2 + 1);
            }
            cnt1 = (cnt1 + 1);
        }
    }

    __aicore__ inline void Process_dt_calc(int m1, int m2){
        if ((m2 == 0)){
            Brcb(dacs_brcb, dacs_buf1.get(cnt1), ((int)shape.BASEL / NUM_ELE_PERBLK_FLOAT), {1, NUM_DBLK_FLOAT});
            PipeBarrier<PIPE_V>();
            Brcb(dacs_brcb2, dacs_brcb, shape.BASEL, {1, NUM_DBLK_FLOAT});
            PipeBarrier<PIPE_V>();
        }
        if ((m2 <= m1)){
            auto custparam = MakeDefaultBinaryRepeatParams();
            custparam.src1RepStride = 0;
            Sub<float, false>(da_out, dacs_brcb2, dacs_buf2.get(cnt2), MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)VEC_FLOAT), custparam);
            PipeBarrier<PIPE_V>();
            Exp<float, false>(da_out, da_out, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)VEC_FLOAT), unary_params);
            PipeBarrier<PIPE_V>();
            Mul<float, false>(da_out, da_out, cb_buf, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)VEC_FLOAT), binary_params);
            PipeBarrier<PIPE_V>();
            Mul<float, false>(da_out, da_out, dtout_buf.get(cnt2), MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)VEC_FLOAT), custparam);
            PipeBarrier<PIPE_V>();
            if ((m1 == m2)){
                LocalTensor<uint16_t> tmptsr_2 = da_out.ReinterpretCast<uint16_t>();
                LocalTensor<uint16_t> tmptsr_3 = da_out.ReinterpretCast<uint16_t>();
                LocalTensor<uint16_t> tmptsr_4 = maskdiagbuf.ReinterpretCast<uint16_t>();
                And<uint16_t, false>(tmptsr_2, tmptsr_3, tmptsr_4, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)VEC_FLOAT), binary_params);
                PipeBarrier<PIPE_V>();
            }
            Cast<half, float, false>(da_out_half, da_out, RoundMode::CAST_RINT, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)VEC_FLOAT), cast_params_f2h);
            PipeBarrier<PIPE_V>();
        }
        else {
            Duplicate<half, false>(da_out_half, (half)0.0, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.BASEL) / (int)VEC_HALF), 1, NUM_DBLK_FLOAT);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Process_out(){
        half d_scale_half = 0;
        d_scale_half = (half) dmtx[v1_h_now].GetValue(0);
        float d_scale = 0;
        d_scale = ((float)d_scale_half);
        int mstart = 0;
        int mend = 0;
        int m = 0;
        if ((get_subblockid() == 0)){
            mstart = THREE;
            mend = FIVE;
        }
        else {
            mstart = 1;
            mend = THREE;
        }
        for (int m2_raw=mstart; m2_raw<mend; m2_raw+=1){
            m = (m2_raw % FOUR);
            in_empty.wait();
            tensor_in_empty.wait();
            GM2UB(scalea.get(v1_cnt1), dacsmtx[(((v1_bc_now * (shape.H * shape.L)) + (v1_h_now * shape.L)) + ((m * CBASEM) * 1))], 1, ((int)shape.BASEL / (int)MTE_FLOAT), 0, 0);
            GM2UB(states, statesmtx[(((v1_bc_now * ((shape.H * shape.L) * shape.P)) + (v1_h_now * (shape.L * shape.P))) + ((m * CBASEM) * shape.P))], 1, ((int)(shape.BASEL * shape.P) / (int)MTE_FLOAT), 0, 0);
            GM2UB(out_b.get(v1_cnt2), outmtx[(((v1_bc_now * ((shape.H * shape.L) * shape.P)) + (v1_h_now * (shape.L * shape.P))) + ((m * CBASEM) * shape.P))], 1, ((int)(shape.BASEL * shape.P) / (int)MTE_FLOAT), 0, 0);
            GM2UB(x_half.get(v1_cnt2), xmtx[(((v1_bc_now * ((shape.L * shape.H) * shape.P)) + ((m * CBASEM) * (shape.H * shape.P))) + (v1_h_now * shape.P))], shape.BASEL, ((int)shape.P / (int)MTE_HALF), ((int)((shape.H - 1) * shape.P) / (int)MTE_HALF), 0);
            in_ready.set();
            tensor_in_ready.set();
            
            out_empty.wait();
            in_ready.wait();
            tensor_out_empty.wait();
            tensor_in_ready.wait();
            Process_out_calc(d_scale);
            out_ready.set();
            in_empty.set();
            tensor_out_ready.set();
            tensor_in_empty.set();

            out_ready.wait();
            tensor_out_ready.wait();
            UB2GM(sumoutmtx[(((v1_bc_now * ((shape.L * shape.H) * shape.P)) + ((m * CBASEM) * (shape.H * shape.P))) + (v1_h_now * shape.P))], sumout.get(v1_cnt2), shape.BASEL, ((int)shape.P / MTE_FLOAT), 0, ((int)((shape.H - 1) * shape.P) / MTE_FLOAT));
            out_empty.set();
            tensor_out_empty.set();
            v1_cnt1 = (v1_cnt1 + 1);
            v1_cnt2 = (v1_cnt2 + 1);
        }
    }

    __aicore__ inline void Process_out_calc(float d_scale){
        Exp<float, false>(scalea.get(v1_cnt1), scalea.get(v1_cnt1), MASK_PLACEHOLDER, ((int)shape.BASEL / (int)VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Brcb(scalea_brcb, scalea.get(v1_cnt1), ((int)shape.BASEL / NUM_ELE_PERBLK_FLOAT), {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();
        Brcb(scalea_brcb2, scalea_brcb, shape.BASEL, {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();
        Mul<float, false>(states, states, scalea_brcb2, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
        Add<float, false>(states, states, out_b.get(v1_cnt2), MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
        Cast<float, half, false>(sumout.get(v1_cnt2), x_half.get(v1_cnt2), RoundMode::CAST_NONE, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)VEC_FLOAT), cast_params_h2f);
        PipeBarrier<PIPE_V>();
        Muls<float, false>(sumout.get(v1_cnt2), sumout.get(v1_cnt2), (float)d_scale, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Add<float, false>(sumout.get(v1_cnt2), sumout.get(v1_cnt2), states, MASK_PLACEHOLDER, ((int)(shape.BASEL * shape.P) / (int)VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
    }
    
private:
    CustVecShapeInfo shape;
    // Global Params
    int cnt1;
    int cnt2;
    int v1_cnt1;
    int v1_cnt2;
    int bc_now;
    int h_now;
    int v1_bc_now;
    int v1_h_now;
    uint64_t mask;
    UnaryRepeatParams cast_params_h2f;
    UnaryRepeatParams cast_params_f2h;
    UnaryRepeatParams unary_params;
    BinaryRepeatParams binary_params;
    // Global Tensors
    GlobalTensor<float> cb_ws;
    GlobalTensor<float> dacsmtx;
    GlobalTensor<float> dtoutmtx;
    GlobalTensor<half> mmtx;
    GlobalTensor<float> outmtx;
    GlobalTensor<float> statesmtx;
    GlobalTensor<half> xmtx;
    GlobalTensor<half> dmtx;
    GlobalTensor<float> sumoutmtx;
    // Local buffers
    DBuff<float, TPosition::VECCALC> dacs_buf1;
    LocalTensor<float> dacs_brcb;
    LocalTensor<float> dacs_brcb2;
    LocalTensor<float> cb_buf;
    LocalTensor<float> da_out;
    LocalTensor<half> da_out_half;
    DBuff<float, TPosition::VECCALC> dacs_buf2;
    DBuff<float, TPosition::VECCALC> dtout_buf;
    LocalTensor<float> maskdiagbuf;
    DBuff<float, TPosition::VECCALC> scalea;
    LocalTensor<float> scalea_brcb;
    LocalTensor<float> scalea_brcb2;
    LocalTensor<float> states;
    DBuff<float, TPosition::VECCALC> out_b;
    DBuff<half, TPosition::VECCALC> x_half;
    DBuff<float, TPosition::VECCALC> sumout;
    
    // Events
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
    SEvent<PIPE_V, PIPE_MTE2> tensor_in_empty;
    SEvent<PIPE_MTE2, PIPE_V> tensor_in_ready;
    SEvent<PIPE_MTE3, PIPE_V> tensor_out_empty;
    SEvent<PIPE_V, PIPE_MTE3> tensor_out_ready;
};

} // namespace Mambav2ChunkScan
} // namespace npu_ops_transformer_ext