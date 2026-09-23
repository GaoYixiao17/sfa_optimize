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

namespace npu_ops_transformer_ext{
namespace Mambav2CausalConv1d{

constexpr int BASES = 2048;

struct CustVecShapeInfo{
    int D;
    int S;
    int W;
    int BD_PER_CORE;
    int BD1;
    int BD2;
    int baseS;
};


__aicore__ inline void tilingShapeCustVec(int B, int D, int S, int W, CustVecShapeInfo &shape){
    shape.D = D;
    shape.S = S;
    shape.W = W;
    shape.BD_PER_CORE = CeilDiv((B * D),(GetBlockNum() * TWO));
    shape.BD1 = (shape.BD_PER_CORE * GetBlockIdx());
    shape.BD2 = (((shape.BD1 + shape.BD_PER_CORE))<((B * D))) ? ((shape.BD1 + shape.BD_PER_CORE)) : ((B * D));
    shape.baseS = Align64B(((shape.S)<(BASES)) ? (shape.S) : (BASES));
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR xmtx_, GM_ADDR wmtx_, GM_ADDR bias_, GM_ADDR outmtx_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        xmtx.SetGlobalBuffer((__gm__ float*) xmtx_);
        wmtx.SetGlobalBuffer((__gm__ float*) wmtx_);
        bias.SetGlobalBuffer((__gm__ half*) bias_);
        outmtx.SetGlobalBuffer((__gm__ float*) outmtx_);
        
        // Local buffers
        xbuf.Init((Align64B(shape.W) + shape.baseS));
        wbuf.Init(Align64B(shape.W));
        tmpbuf.Init(shape.baseS);
        sumbuf.Init(shape.baseS);
        AllocateLocalTensor<TPosition::VECCALC>(offsetbuf, shape.baseS);
        outbuf.Init(shape.baseS);
        // Initialize events
        in_empty.Init();
        in_ready.Init();
        out_empty.Init();
        out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        Duplicate<int32_t, false>(offsetbuf, (int32_t)0.0, MASK_PLACEHOLDER, ((int)Align64B(shape.S) / VEC_FLOAT), 1, NUM_DBLK_FLOAT);
        PipeBarrier<PIPE_V>();
        cnt = 0;
        val = 0;
        cast_params_h2f = CastHalf2FloatRepeatParams();
        cast_params_f2h = CastFloat2HalfRepeatParams();
        unary_params = MakeDefaultUnaryRepeatParams();
        binary_params = MakeDefaultBinaryRepeatParams();

        for (int idx=0; idx<shape.baseS; idx+=1){
            val = (((idx + VEC_FLOAT + 1) - shape.W) * FOUR);
            offsetbuf[idx].SetValue(0, (int32_t)(val));
        }
        for (int bd=shape.BD1; bd<shape.BD2; bd+=1){
            Duplicate<float, false>(xbuf.get(cnt), (float)0.0, MASK_PLACEHOLDER, ((int)Align64B(shape.W) / VEC_FLOAT), 1, NUM_DBLK_FLOAT);
            PipeBarrier<PIPE_V>();
            half b_val_f16 = 0;
            b_val_f16 = (half) bias[(bd % shape.D)].GetValue(0);
            float b_val = 0;
            b_val = ((float)b_val_f16);
            float w_scale = 0;
            for (int baseS=0; baseS<shape.S; baseS+=shape.baseS){
                in_empty.wait();
                Process_movein(bd, baseS);
                in_ready.set();
                
                out_empty.wait();
                in_ready.wait();
                Process_calc(bd, b_val, w_scale);
                out_ready.set();
                in_empty.set();

                out_ready.wait();
                Process_moveout(bd, baseS);
                out_empty.set();
                cnt = (cnt + 1);
            }
        }
        in_empty.release();
        out_empty.release();
    }
    
    __aicore__ inline void Process_movein(int bd, int baseS){
        if (((baseS + shape.baseS) >= shape.S)){
            if ((baseS == 0)){
                GM2UBPad(xbuf.get(cnt)[Align64B(shape.W)], xmtx[(bd * shape.S)], 1, (shape.S * FOUR), 0, 0, 0, 0, 0, 0); // FOUR: 4bytes
            }
            else {
                GM2UBPad(xbuf.get(cnt), xmtx[(((bd * shape.S) + baseS) - Align64B(shape.W))], 1, (((shape.S - baseS) + Align64B(shape.W)) * FOUR), 0, 0, 0, 0, 0, 0);
            }
        }
        else {
            if ((baseS == 0)){
                GM2UB(xbuf.get(cnt)[Align64B(shape.W)], xmtx[(bd * shape.S)], 1, ((int)shape.baseS / MTE_FLOAT), 0, 0);
            }
            else {
                GM2UB(xbuf.get(cnt), xmtx[(((bd * shape.S) + baseS) - Align64B(shape.W))], 1, ((int)(shape.baseS + Align64B(shape.W)) / MTE_FLOAT), 0, 0);
            }
        }
    }

    __aicore__ inline void Process_calc(int bd, float b_val, float w_scale){
        Duplicate<float, false>(sumbuf.get(cnt), (float)0.0, MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), 1, NUM_DBLK_FLOAT);
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> tmptsr_0 = offsetbuf.ReinterpretCast<uint32_t>();
        for (int widx=0; widx<shape.W; widx+=1){
            w_scale = (float) wmtx[(((bd % shape.D) * shape.W) + widx)].GetValue(0);
            Gather(tmpbuf.get(cnt), xbuf.get(cnt), tmptsr_0, 0, VECTORFULLMASK, ((int)shape.baseS / VEC_FLOAT), NUM_DBLK_FLOAT);
            PipeBarrier<PIPE_V>();
            Muls<float, false>(tmpbuf.get(cnt), tmpbuf.get(cnt), (float)w_scale, MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), unary_params);
            Add<float, false>(sumbuf.get(cnt), sumbuf.get(cnt), tmpbuf.get(cnt), MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), binary_params);
            PipeBarrier<PIPE_V>();
            Adds<int32_t, false>(offsetbuf, offsetbuf, (int32_t)FOUR, MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), unary_params);
            PipeBarrier<PIPE_V>();
        }
        Adds<int32_t, false>(offsetbuf, offsetbuf, (int32_t)(-1 * (shape.W * FOUR)), MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(sumbuf.get(cnt), sumbuf.get(cnt), (float)b_val, MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Muls<float, false>(outbuf.get(cnt), sumbuf.get(cnt), (float)-1.0, MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(outbuf.get(cnt), outbuf.get(cnt), MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(outbuf.get(cnt), outbuf.get(cnt), (float)1.0, MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Div<float, false>(outbuf.get(cnt), sumbuf.get(cnt), outbuf.get(cnt), MASK_PLACEHOLDER, ((int)shape.baseS / VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process_moveout(int bd, int baseS){
        if (((baseS + shape.baseS) >= shape.S)){
            UB2GMPad(outmtx[((bd * shape.S) + baseS)], outbuf.get(cnt), 1, ((shape.S - baseS) * FOUR), 0, 0);
        }
        else {
            UB2GM(outmtx[((bd * shape.S) + baseS)], outbuf.get(cnt), 1, ((int)shape.baseS / MTE_FLOAT), 0, 0);
        }
    }

private:
    CustVecShapeInfo shape;
    // Global Params
    int cnt;
    int32_t val;
    UnaryRepeatParams cast_params_h2f;
    UnaryRepeatParams cast_params_f2h;
    UnaryRepeatParams unary_params;
    BinaryRepeatParams binary_params;
    // Global Tensors
    GlobalTensor<float> xmtx;
    GlobalTensor<float> wmtx;
    GlobalTensor<half> bias;
    GlobalTensor<float> outmtx;
    // Local buffers
    DBuff<float, TPosition::VECCALC> xbuf;
    DBuff<float, TPosition::VECCALC> wbuf;
    DBuff<float, TPosition::VECCALC> tmpbuf;
    DBuff<float, TPosition::VECCALC> sumbuf;
    LocalTensor<int32_t> offsetbuf;
    DBuff<float, TPosition::VECCALC> outbuf;
    
    // Events
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
};

}
}