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
namespace Mambav2ChunkStatePassing {

constexpr int CBASEM = 64;
constexpr int CBASEN = 64;

struct CustCubeShapeInfo{
    int BASEM;
    int BASEK;
    int BASEN;
    int CUBEM;
    int CUBEN;
    int CUBEK;
    int B;
    int H;
    int C;
    int G;
    int Z;
    int groupsize;
    int headPerCore;
    int cmtx_strideB;
    int cmtx_strideC;
    int cmtx_strideL;
    int state_strideB;
    int state_strideC;
    int state_strideH;
    int out_strideB;
    int out_strideC;
    int out_strideH;
};


__aicore__ inline void tilingShapeCustCube(int B, int H, int S, int C, int L, int G, int N, int P, int Z, CustCubeShapeInfo &shape){
    shape.BASEM = CBASEM;
    shape.BASEK = N;
    shape.BASEN = CBASEN;
    shape.CUBEM = L;
    shape.CUBEN = P;
    shape.CUBEK = N;
    shape.B = B;
    shape.H = H;
    shape.C = C;
    shape.G = G;
    shape.Z = Z;
    shape.groupsize = ((int)H / (int)G);
    shape.headPerCore = CeilDiv((B * H),GetBlockNum());
    shape.cmtx_strideB = (((C * L) * G) * N);
    shape.cmtx_strideC = ((L * G) * N);
    shape.cmtx_strideL = (G * N);
    shape.state_strideB = (((C * H) * N) * P);
    shape.state_strideC = ((H * N) * P);
    shape.state_strideH = (N * P);
    shape.out_strideB = (((C * H) * L) * P);
    shape.out_strideC = ((H * L) * P);
    shape.out_strideH = (L * P);
}


class CustCube{
public:
    __aicore__ inline CustCube(){}
    __aicore__ inline void Init(GM_ADDR cmtx_, GM_ADDR state_fp16_, GM_ADDR out_bmm_, CustCubeShapeInfo shape_){
        shape = shape_;
        // Reset tpipe. Start resource distribution
        TPipe* pipe_ptr = GetTPipePtr();
        pipe_ptr->Reset();
        OccupyMMTE1Events();
        // Global Tensors
        cmtx.SetGlobalBuffer((__gm__ half*) cmtx_);
        state_fp16.SetGlobalBuffer((__gm__ half*) state_fp16_);
        out_bmm.SetGlobalBuffer((__gm__ float*) out_bmm_);
        // L1 Buffers
        l1a.Init((shape.BASEM * shape.BASEK));
        l1b.Init((shape.BASEN * shape.BASEK));
        // L0A Buffers
        l0a.Init((shape.BASEM * shape.BASEK));
        // L0B Buffers
        l0b.Init((shape.BASEN * shape.BASEK));
        // L0C Buffers
        l0c.Init((shape.BASEM * shape.BASEN));
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
        ws_cnt = 0;
        len_h = ((shape.headPerCore)<(((shape.B * shape.H) - (get_block_idx() * shape.headPerCore)))) ? (shape.headPerCore) : (((shape.B * shape.H) - (get_block_idx() * shape.headPerCore)));

        CUBE_READY(0, PIPE_FIX);
        CUBE_READY(0, PIPE_FIX);
        CUBE_READY(0, PIPE_FIX);

        for (int chunk_id=0; chunk_id<shape.C; chunk_id+=1){
            Process_cube(chunk_id);
        }

        l1_empty.release();
        l0_empty.release();
        out_empty.release();
    }

    __aicore__ inline void Process_cube(int chunk_id){
        int bh_index = (get_block_idx() * shape.headPerCore);
        int base_b = 0;
        int base_h = 0;
        int base_g = 0;
        for (int head_id=0; head_id<len_h; head_id+=1){
            base_b = ((int)bh_index / (int)shape.H);
            base_h = (bh_index % shape.H);
            base_g = ((int)base_h / (int)shape.groupsize);
            WAIT_VEC(0);
            Process_calc(chunk_id, base_b, base_h, base_g);
            CUBE_READY(0, PIPE_FIX);
            ws_cnt = (ws_cnt + 1);
            bh_index = (bh_index + 1);
        }
    }

    __aicore__ inline void Process_calc(int chunk_id, int base_b, int base_h, int base_g){
        for (int m=0; m<shape.CUBEM; m+=shape.BASEM){
            for (int n=0; n<shape.CUBEN; n+=shape.BASEN){
                for (int k=0; k<shape.CUBEK; k+=shape.BASEK){
                    l1_empty.wait();
                    L1ND2NZ(l1a.get(input_cnt), cmtx[(((((base_b * shape.cmtx_strideB) + (chunk_id * shape.cmtx_strideC)) + (m * shape.cmtx_strideL)) + (base_g * shape.CUBEK)) + k)], shape.BASEM, shape.BASEK, (shape.G * shape.CUBEK), shape.BASEM);
                    L1ND2NZ(l1b.get(input_cnt), state_fp16[((((((ws_cnt % THREE) * GetBlockNum()) * shape.Z) + (get_block_idx() * shape.Z)) + (k * shape.CUBEN)) + n)], shape.BASEK, shape.BASEN, shape.CUBEN, shape.BASEK);
                    l1_ready.set();

                    l0_empty.wait();
                    l1_ready.wait();
                    L0NZ2ZZ(l0a.get(input_cnt), l1a.get(input_cnt), shape.BASEM, shape.BASEK, shape.BASEM, shape.BASEK);
                    L0NZ2ZN(l0b.get(input_cnt), l1b.get(input_cnt), shape.BASEK, shape.BASEN, shape.BASEK, shape.BASEN);
                    l1_empty.set();
                    l0_ready.set();
                    out_empty.wait();
                    l0_ready.wait();
                    MMAD(l0c.get(output_cnt), l0a.get(input_cnt), l0b.get(input_cnt), shape.BASEM, shape.BASEK, shape.BASEN, (k == 0), 0);
                    out_ready.set();
                    l0_empty.set();
                    input_cnt = (input_cnt + 1);

                    out_ready.wait();
                    if (((k + shape.BASEK) >= shape.CUBEK)){
                        L0C2GM_NZ2ND(out_bmm[(((((base_b * shape.out_strideB) + (chunk_id * shape.out_strideC)) + (base_h * shape.out_strideH)) + (m * shape.CUBEN)) + n)], l0c.get(output_cnt), shape.BASEM, shape.BASEN, shape.CUBEN, shape.BASEM, 0);
                        output_cnt = (output_cnt + 1);
                    }
                    out_empty.set();
                }
            }
        }
    }
    
private:
    CustCubeShapeInfo shape;
    // Global Params
    int input_cnt;
    int output_cnt;
    int ws_cnt;
    int len_h;
    // Global Tensors
    GlobalTensor<half> cmtx;
    GlobalTensor<half> state_fp16;
    GlobalTensor<float> out_bmm;
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

} // namespace Mambav2ChunkStatePassing
} // namespace npu_ops_transformer_ext