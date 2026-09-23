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
namespace Mambav2Rmsnormgated {

constexpr int BASED = 2048;

struct CustVecShapeInfo{
    int group_size;
    float eps;
    int B;
    int S;
    int D;
    int G;
    float E;
    int ngroups;
    int BASED;
    int BASEG;
    int loopnum;
    int vec_d;
};


__aicore__ inline void tilingShapeCustVec(int B, int S, int D, int G, float E, CustVecShapeInfo &shape){
    shape.group_size = (D / G);
    shape.eps = E;
    shape.B = B;
    shape.S = S;
    shape.D = D;
    shape.G = G;
    shape.E = E;
    shape.ngroups = G;
    shape.BASED = BASED;
    shape.BASEG = (shape.BASED / shape.group_size);
    shape.loopnum = CeilDiv((B * S),(GetBlockNum() * TWO));
    shape.vec_d = (shape.loopnum * shape.D);
}


class CustVec{
public:
    __aicore__ inline CustVec(){}
    __aicore__ inline void Init(GM_ADDR xmtx_, GM_ADDR zmtx_, GM_ADDR wmtx_, GM_ADDR outmtx_, CustVecShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        
        // Global Tensors
        xmtx.SetGlobalBuffer((__gm__ float*) xmtx_);
        zmtx.SetGlobalBuffer((__gm__ float*) zmtx_);
        wmtx.SetGlobalBuffer((__gm__ float*) wmtx_);
        outmtx.SetGlobalBuffer((__gm__ float*) outmtx_);
        
        // Local buffers
        outbuf.Init(shape.BASED);
        fp32_xbuf.Init(shape.BASED);
        fp32_zbuf.Init(shape.BASED);
        fp32_wbuf.Init(shape.BASED);
        fp32_squarebuf.Init(shape.BASED);
        fp32_meanbuf.Init(BLK_SIZE);
        fp32_meantempbuf.Init(BLK_SIZE);
        AllocateLocalTensor<TPosition::VECCALC>(fp32_dupbuf, BLK_SIZE);
        fp32_scaletempbuf.Init((shape.BASEG * NUM_DBLK_FLOAT));
        fp32_normscalebuf.Init((shape.BASEG * BLK_SIZE));
        fp32_sigmoidbuf.Init(shape.BASED);
        // Initialize events
        in_ready.Init();
        in_empty.Init();
        out_ready.Init();
        out_empty.Init();
    }
    
    __aicore__ inline void Compute(){
        in_empty.setall();
        out_empty.setall();
        
        float scale = shape.group_size;
        scale = ((float)((float)1.0 / scale));
        int ub_d = ((shape.vec_d)<((((shape.B * shape.S) - (GetBlockIdx() * shape.loopnum)) * shape.D))) ? (shape.vec_d) : ((((shape.B * shape.S) - (GetBlockIdx() * shape.loopnum)) * shape.D));
        cnt = 0;
        mte2_base_d = 0;
        mte2_kernel_d = 0;
        cast_params_h2f = CastHalf2FloatRepeatParams();
        cast_params_f2h = CastFloat2HalfRepeatParams();
        unary_params = MakeDefaultUnaryRepeatParams();
        binary_params = MakeDefaultBinaryRepeatParams();
        
        for (int kernel_d=0; kernel_d<ub_d; kernel_d+=shape.D){
            for (int base_d=0; base_d<shape.D; base_d+=shape.BASED){
                in_empty.wait();
                if(kernel_d==0 && base_d==0){
                    GM2UB(fp32_xbuf.get(cnt), xmtx[(((GetBlockIdx() * shape.vec_d) + kernel_d) + base_d)], 1, (shape.BASED / MTE_FLOAT), 0, 0);
                    GM2UB(fp32_zbuf.get(cnt), zmtx[(((GetBlockIdx() * shape.vec_d) + kernel_d) + base_d)], 1, (shape.BASED / MTE_FLOAT), 0, 0);
                    GM2UB(fp32_wbuf.get(cnt), wmtx[base_d], 1, (shape.BASED / MTE_FLOAT), 0, 0);
                }
                if (kernel_d + base_d < ub_d - shape.BASED){
                    mte2_base_d = (base_d + shape.BASED) % shape.D;
                    mte2_kernel_d = kernel_d + (int)((base_d + shape.BASED) / shape.D)*shape.D;
                    GM2UB(fp32_xbuf.get(cnt+1), xmtx[(((GetBlockIdx() * shape.vec_d) + mte2_kernel_d) + mte2_base_d)], 1, (shape.BASED / MTE_FLOAT), 0, 0);
                    GM2UB(fp32_zbuf.get(cnt+1), zmtx[(((GetBlockIdx() * shape.vec_d) + mte2_kernel_d) + mte2_base_d)], 1, (shape.BASED / MTE_FLOAT), 0, 0);
                    GM2UB(fp32_wbuf.get(cnt+1), wmtx[mte2_base_d], 1, (shape.BASED / MTE_FLOAT), 0, 0);
                }
                in_ready.set();
                
                out_empty.wait();
                in_ready.wait();
                Process_calc(scale);
                out_ready.set();
                in_empty.set();
                
                out_ready.wait();
                UB2GM(outmtx[(((GetBlockIdx() * shape.vec_d) + kernel_d) + base_d)], outbuf.get(cnt), 1, (shape.BASED / MTE_FLOAT), 0, 0);
                out_empty.set();
                
                cnt = (cnt + 1);
            }
        }
        
        in_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Process_calc(float scale){
        // calc silu
        Duplicate<float, false>(fp32_dupbuf, 1.000000f, MASK_PLACEHOLDER, 1, 1, NUM_DBLK_FLOAT);
        PipeBarrier<PIPE_V>();
        Muls<float, false>(fp32_sigmoidbuf.get(cnt), fp32_zbuf.get(cnt), -1.0f, MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Exp<float, false>(fp32_sigmoidbuf.get(cnt), fp32_sigmoidbuf.get(cnt), MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        Adds<float, false>(fp32_sigmoidbuf.get(cnt), fp32_sigmoidbuf.get(cnt), 1.0f, MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), unary_params);
        PipeBarrier<PIPE_V>();
        auto custparam = MakeDefaultBinaryRepeatParams();
        custparam.src0RepStride = 0; // 1,1,1,8,0,8
        Div<float, false>(fp32_sigmoidbuf.get(cnt), fp32_dupbuf, fp32_sigmoidbuf.get(cnt), MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), custparam);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(fp32_sigmoidbuf.get(cnt), fp32_sigmoidbuf.get(cnt), fp32_zbuf.get(cnt), MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
        Mul<float, false>(fp32_xbuf.get(cnt), fp32_sigmoidbuf.get(cnt), fp32_xbuf.get(cnt), MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
        // calc norm
        Mul<float, false>(fp32_squarebuf.get(cnt), fp32_xbuf.get(cnt), fp32_xbuf.get(cnt), MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
        Duplicate<float, false>(fp32_meanbuf.get(cnt), 0.000000f, MASK_PLACEHOLDER, 1, 1, NUM_DBLK_FLOAT);
        PipeBarrier<PIPE_V>();
        for (int base_g=0; base_g<shape.BASEG; ++base_g){
            ReduceSum<float>(fp32_meanbuf.get(cnt)[base_g], fp32_squarebuf.get(cnt)[(base_g * shape.group_size)], fp32_meantempbuf.get(cnt), shape.group_size);
            PipeBarrier<PIPE_V>();
        }
        Muls<float>(fp32_meanbuf.get(cnt), fp32_meanbuf.get(cnt), scale, shape.BASEG);
        PipeBarrier<PIPE_V>();
        Adds<float>(fp32_meanbuf.get(cnt), fp32_meanbuf.get(cnt), shape.eps, shape.BASEG);
        PipeBarrier<PIPE_V>();
        Sqrt<float>(fp32_meanbuf.get(cnt), fp32_meanbuf.get(cnt), shape.BASEG);
        PipeBarrier<PIPE_V>();
        Div<float, false>(fp32_meanbuf.get(cnt), fp32_dupbuf, fp32_meanbuf.get(cnt), MASK_PLACEHOLDER, 1, custparam);
        PipeBarrier<PIPE_V>();
        // okay
        Brcb(fp32_scaletempbuf.get(cnt), fp32_meanbuf.get(cnt), 1, {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();
        Brcb(fp32_normscalebuf.get(cnt), fp32_scaletempbuf.get(cnt), shape.BASEG, {1, NUM_DBLK_FLOAT});
        PipeBarrier<PIPE_V>();
        
        for (int base_g=0; base_g<shape.BASEG; ++base_g){
            int index = base_g * shape.group_size;
            Mul<float, false>(fp32_xbuf.get(cnt)[index], fp32_normscalebuf.get(cnt)[(base_g * BLK_SIZE)], fp32_xbuf.get(cnt)[index], MASK_PLACEHOLDER, (shape.group_size / VEC_FLOAT), custparam);
            PipeBarrier<PIPE_V>();
        }
        Mul<float, false>(outbuf.get(cnt), fp32_xbuf.get(cnt), fp32_wbuf.get(cnt), MASK_PLACEHOLDER, (shape.BASED / VEC_FLOAT), binary_params);
        PipeBarrier<PIPE_V>();
    }
    
private:
    CustVecShapeInfo shape;
    // Global Params
    int cnt;
    int mte2_base_d;
    int mte2_kernel_d;
    UnaryRepeatParams cast_params_h2f;
    UnaryRepeatParams cast_params_f2h;
    UnaryRepeatParams unary_params;
    BinaryRepeatParams binary_params;
    // Global Tensors
    GlobalTensor<float> xmtx;
    GlobalTensor<float> zmtx;
    GlobalTensor<float> wmtx;
    GlobalTensor<float> outmtx;
    // Local buffers
    DBuff<float, TPosition::VECCALC> outbuf;
    DBuff<float, TPosition::VECCALC> fp32_xbuf;
    DBuff<float, TPosition::VECCALC> fp32_zbuf;
    DBuff<float, TPosition::VECCALC> fp32_wbuf;
    DBuff<float, TPosition::VECCALC> fp32_squarebuf;
    DBuff<float, TPosition::VECCALC> fp32_meanbuf;
    DBuff<float, TPosition::VECCALC> fp32_meantempbuf;
    LocalTensor<float> fp32_dupbuf;
    DBuff<float, TPosition::VECCALC> fp32_scaletempbuf;
    DBuff<float, TPosition::VECCALC> fp32_normscalebuf;
    DBuff<float, TPosition::VECCALC> fp32_sigmoidbuf;
    
    // Double events
    DEvent<PIPE_MTE2, PIPE_V> in_ready;
    DEvent<PIPE_V, PIPE_MTE2> in_empty;
    DEvent<PIPE_V, PIPE_MTE3> out_ready;
    DEvent<PIPE_MTE3, PIPE_V> out_empty;
    // User-defined events
};
// Auto-generated code. Readability is not guaranteed

}
}