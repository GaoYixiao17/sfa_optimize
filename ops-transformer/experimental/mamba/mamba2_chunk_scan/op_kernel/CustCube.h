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

namespace npu_ops_transformer_ext {
namespace Mambav2ChunkScan {

struct CustCubeShapeInfo{
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
    int BASEK1;
    int BASEK2;
    int BASEK;
};


__aicore__ inline void tilingShapeCustCube(int B, int C, int H, int G, int L, int N, int P, CustCubeShapeInfo &shape){
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
    shape.BASEK1 = N;
    shape.BASEK2 = L;
    shape.BASEK = ((shape.BASEK1)>(shape.BASEK2)) ? (shape.BASEK1) : (shape.BASEK2);
}


class CustCube{
public:
    __aicore__ inline CustCube(){}
    __aicore__ inline void Init(GM_ADDR cmtx_, GM_ADDR bmtx_, GM_ADDR cb_ws_, GM_ADDR xmtx_, GM_ADDR mmtx_, GM_ADDR outmtx_, GM_ADDR cbmtx_, CustCubeShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        OccupyMMTE1Events();
        // Global Tensors
        cmtx.SetGlobalBuffer((__gm__ half*) cmtx_);
        bmtx.SetGlobalBuffer((__gm__ half*) bmtx_);
        cb_ws.SetGlobalBuffer((__gm__ float*) cb_ws_);
        xmtx.SetGlobalBuffer((__gm__ half*) xmtx_);
        mmtx.SetGlobalBuffer((__gm__ half*) mmtx_);
        outmtx.SetGlobalBuffer((__gm__ float*) outmtx_);
        cbmtx.SetGlobalBuffer((__gm__ float*) cbmtx_);
        // L1 Buffers
        l1a.Init((CBASEM * shape.BASEK));
        l1b.Init((CBASEN * shape.BASEK));
        // L0A Buffers
        l0a.Init((CBASEM * shape.BASEK));
        // L0B Buffers
        l0b.Init((CBASEN * shape.BASEK));
        // L0C Buffers
        l0c.Init(CBASEM * CBASEN);
        // UB Buffers
        // Initialize events
        l1_empty.Init();
        l1_ready.Init();
        l0_empty.Init();
        l0_ready.Init();
        out_empty.Init();
        out_ready.Init();
    }
    
    __aicore__ inline void Compute(){
        l1_empty.setall();
        l0_empty.setall();
        out_empty.setall();
        input_cnt = 0;
        output_cnt = 0;
        cube1_in_cnt = 0;
        cube1_out_cnt = 0;

        CUBE_READY(THREE, PIPE_FIX);
        CUBE_READY(THREE, PIPE_FIX);

        for (int bch=shape.BCH1; bch<shape.BCH2; bch+=1){
            int bc = ((int)bch / (int)shape.H);
            int h = (bch % shape.H);
            int g = ((int)h / (int)shape.HPERG);
            if ((((h % shape.HPERG) == 0) || (bch == shape.BCH1))){
                Process_cube_cb(bc, g);
                CUBE_READY(0, PIPE_FIX);
            }
            WAIT_VEC(0);
            WAIT_VEC(THREE);
            Process_cube_scan(bc, h);
            CUBE_READY(THREE, PIPE_FIX);
            CUBE_READY(1, PIPE_FIX);
        }
        WAIT_VEC(THREE);
        WAIT_VEC(THREE);
        l1_empty.release();
        l0_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Process_cube_cb(int bc, int g){
        for (int m=0; m<shape.L; m+=CBASEM){
            for (int n=0; n<shape.L; n+=CBASEN){
                for (int k=0; k<shape.N; k+=shape.BASEK1){
                    l1_empty.wait();
                    L1ND2NZ(l1a.get(input_cnt), cmtx[((((bc * ((shape.L * shape.G) * shape.N)) + (m * (shape.G * shape.N))) + (g * shape.N)) + (k * 1))], CBASEM, shape.BASEK1, (shape.N * shape.G), CBASEM);
                    L1ND2NZ(l1b.get(input_cnt), bmtx[((((bc * ((shape.L * shape.G) * shape.N)) + (n * (shape.G * shape.N))) + (g * shape.N)) + (k * 1))], CBASEN, shape.BASEK1, (shape.N * shape.G), CBASEN);
                    l1_ready.set();

                    l0_empty.wait();
                    l1_ready.wait();
                    L0NZ2ZZ(l0a.get(input_cnt), l1a.get(input_cnt), CBASEM, shape.BASEK1, CBASEM, shape.BASEK1);
                    LOADL0(l0b.get(input_cnt), l1b.get(input_cnt), CBASEN, shape.BASEK1);
                    l1_empty.set();
                    l0_ready.set();

                    out_empty.wait();
                    l0_ready.wait();
                    MMAD(l0c.get(output_cnt), l0a.get(input_cnt), l0b.get(input_cnt), CBASEM, shape.BASEK1, CBASEN, (k == 0), 0);
                    out_ready.set();
                    l0_empty.set();

                    out_ready.wait();
                    if (((k + shape.BASEK1) >= shape.N)){
                        L0C2GM_NZ2ND(cbmtx[((((bc * ((shape.G * shape.L) * shape.L)) + (g * (shape.L * shape.L))) + (m * shape.L)) + (n * 1))], l0c.get(output_cnt), CBASEM, CBASEN, shape.L, CBASEM, 0);
                        L0C2GM_NZ2ND(cb_ws[(((get_block_idx() * (shape.L * shape.L)) + (m * shape.L)) + (n * 1))], l0c.get(output_cnt), CBASEM, CBASEN, shape.L, CBASEM, 0);
                        output_cnt = (output_cnt + 1);
                    }
                    out_empty.set();
                    input_cnt = (input_cnt + 1);
                }
            }
        }
    }

    __aicore__ inline void Process_cube_scan(int bc, int h){
        for (int m=0; m<shape.L; m+=CBASEM){
            for (int n=0; n<shape.P; n+=CBASEN){
                for (int k=0; k<shape.L; k+=shape.BASEK2){
                    l1_empty.wait();
                    L1ND2NZ(l1a.get(cube1_in_cnt), mmtx[((((bc * ((shape.H * shape.L) * shape.L)) + (h * (shape.L * shape.L))) + (m * shape.L)) + (k * 1))], CBASEM, shape.BASEK2, shape.L, CBASEM);
                    L1ND2NZ(l1b.get(cube1_in_cnt), xmtx[((((bc * ((shape.L * shape.H) * shape.P)) + (k * (shape.H * shape.P))) + (h * shape.P)) + (n * 1))], shape.BASEK2, CBASEN, (shape.H * shape.P), shape.BASEK2);
                    l1_ready.set();

                    l0_empty.wait();
                    l1_ready.wait();
                    L0NZ2ZZ(l0a.get(cube1_in_cnt), l1a.get(cube1_in_cnt), CBASEM, shape.BASEK2, CBASEM, shape.BASEK2);
                    L0NZ2ZN(l0b.get(cube1_in_cnt), l1b.get(cube1_in_cnt), shape.BASEK2, CBASEN, shape.BASEK2, CBASEN);
                    l1_empty.set();
                    l0_ready.set();

                    out_empty.wait();
                    l0_ready.wait();
                    MMAD(l0c.get(cube1_out_cnt), l0a.get(cube1_in_cnt), l0b.get(cube1_in_cnt), CBASEM, shape.BASEK2, CBASEN, (k == 0), 0);
                    out_ready.set();
                    l0_empty.set();

                    out_ready.wait();
                    if (((k + shape.BASEK2) >= shape.L)){
                        L0C2GM_NZ2ND(outmtx[((((bc * ((shape.H * shape.L) * shape.P)) + (h * (shape.L * shape.P))) + (m * shape.P)) + (n * 1))], l0c.get(cube1_out_cnt), CBASEM, CBASEN, shape.P, CBASEM, 0);
                        cube1_out_cnt = (cube1_out_cnt + 1);
                    }
                    out_empty.set();
                    cube1_in_cnt = (cube1_in_cnt + 1);
                }
            }
        }
    }
    
private:
    CustCubeShapeInfo shape;
    // Global Params
    int input_cnt;
    int output_cnt;
    int cube1_in_cnt;
    int cube1_out_cnt;
    // Global Tensors
    GlobalTensor<half> cmtx;
    GlobalTensor<half> bmtx;
    GlobalTensor<float> cb_ws;
    GlobalTensor<half> xmtx;
    GlobalTensor<half> mmtx;
    GlobalTensor<float> outmtx;
    GlobalTensor<float> cbmtx;
    // Local buffers
    DBuff<half, TPosition::A1> l1a;
    DBuff<half, TPosition::A1> l1b;
    DBuff<half, TPosition::A2> l0a;
    DBuff<half, TPosition::B2> l0b;
    DBuff<float, TPosition::CO1> l0c;
    
    // Events
    DEvent<PIPE_MTE1, PIPE_MTE2> l1_empty;
    DEvent<PIPE_MTE2, PIPE_MTE1> l1_ready;
    DEvent<PIPE_M, PIPE_MTE1> l0_empty;
    DEvent<PIPE_MTE1, PIPE_M> l0_ready;
    DEvent<PIPE_FIX, PIPE_M> out_empty;
    DEvent<PIPE_M, PIPE_FIX> out_ready;
};

} // namespace Mambav2ChunkScan
} // namespace npu_ops_transformer_ext