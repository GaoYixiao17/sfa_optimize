/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_swiglu_impl.hpp
 * \brief MatmulSwiglu kernel(方案一: 标准 matmul -> GM 中间 -> 向量 SwiGLU)
 *
 * y = SiLU(gate) * up, [gate|up] = x @ weight (+bias), weight 打包为 [K, 2N]。
 * 方案一(稳): 用一次标准 matmul 算完整 C=[M,2N] 写到 GM 中间缓冲(user workspace),
 * 跨核同步后, 向量侧按行读回 C, 切分 gate=C[:,0:N]/up=C[:,N:2N], 就地做 SwiGLU 写出 y。
 * 不用取巧的 packed 双 matmul + SetOrgShape 子块寻址(该写法真机实测输出恒 0)。
 */
#ifndef MATMUL_SWIGLU_IMPL_HPP
#define MATMUL_SWIGLU_IMPL_HPP

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

namespace NsMatmulSwiglu {
using namespace AscendC;
using namespace matmul;

constexpr int32_t NUM_BUFFER = 2;  // 输入/输出队列双缓冲深度

// xType: 输入/输出数据类型; cType: matmul 累加与中间数据类型(固定 fp32);
// TransposeW: weight 是否转置(编译期常量, 由 TilingKey 选择, 避免运行时判断)
template <typename xType, typename cType, bool TransposeW>
class MatmulSwigluImpl {
public:
    using AType = MatmulType<TPosition::GM, CubeFormat::ND, xType, false>;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, xType, TransposeW>;
    using CTypeMm = MatmulType<TPosition::GM, CubeFormat::ND, cType>;
    using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, cType>;

    Matmul<AType, BType, CTypeMm, BiasType> mm;  // 单 matmul: x[M,K] @ weight[K,2N] -> [M,2N]
    TPipe pipe;

    __aicore__ inline MatmulSwigluImpl() {}

    // userWs: GetUserWorkspace(workspace) 得到的用户 workspace 指针(存 [M,2N] fp32 中间结果)
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR userWs,
                                const optiling::MatmulSwigluTilingData* tilingData)
    {
        // m/k/2N 从 mmTiling 派生(避免与 TCubeTiling 重复存储): matmul 为 x[M,K] @ weight[K,2N]
        m_ = static_cast<uint64_t>(tilingData->mmTiling.M);
        k_ = static_cast<uint64_t>(tilingData->mmTiling.Ka);
        twoN_ = static_cast<uint64_t>(tilingData->mmTiling.N);
        n_ = twoN_ / 2;  // 单边宽度 = 2N / 2
        hasBias_ = (tilingData->hasBias == 1U);
        tileN_ = tilingData->tileN;  // 行内分块列数, 保证单块 UB 不溢出

        xGm_.SetGlobalBuffer((__gm__ xType*)x, m_ * k_);
        wGm_.SetGlobalBuffer((__gm__ xType*)weight, k_ * twoN_);
        if (hasBias_) {
            biasGm_.SetGlobalBuffer((__gm__ cType*)bias, twoN_);
        }
        yGm_.SetGlobalBuffer((__gm__ xType*)y, m_ * n_);
        cGm_.SetGlobalBuffer((__gm__ cType*)userWs, m_ * twoN_);  // [M, 2N] fp32 中间

        // 向量阶段每次处理一块 tileN 列(而非整行), 避免大 N 时 UB 溢出:
        // gate/up/sig(fp32) + y(xType), 缓冲按 tileN 开。
        pipe.InitBuffer(inQueGate_, NUM_BUFFER, tileN_ * sizeof(cType));
        pipe.InitBuffer(inQueUp_, NUM_BUFFER, tileN_ * sizeof(cType));
        pipe.InitBuffer(sigBuf_, tileN_ * sizeof(cType));
        pipe.InitBuffer(outQueY_, NUM_BUFFER, tileN_ * sizeof(xType));
    }

    __aicore__ inline void Process()
    {
        // ---- Phase 1: 标准 matmul, 完整 [M,2N] 写到 GM 中间 ----
        mm.SetTensorA(xGm_, false);
        mm.SetTensorB(wGm_, TransposeW);
        if (hasBias_) {
            mm.SetBias(biasGm_);
        }
        mm.IterateAll(cGm_);
        mm.End();

        SyncAll();  // 跨核屏障: 确保 matmul(Cube)写完 cGm, 向量核再读

        // ---- Phase 2: 向量 SwiGLU, 按行切分到各核 ----
        const int64_t coreNum = GetBlockNum();
        const int64_t blockIdx = GetBlockIdx();
        const int64_t rowsPerCore = (static_cast<int64_t>(m_) + coreNum - 1) / coreNum;
        const int64_t rowStart = blockIdx * rowsPerCore;
        int64_t rowEnd = rowStart + rowsPerCore;
        rowEnd = rowEnd < static_cast<int64_t>(m_) ? rowEnd : static_cast<int64_t>(m_);
        for (int64_t row = rowStart; row < rowEnd; ++row) {
            ComputeRow(row);
        }
    }

private:
    // 处理第 row 行: 按 tileN 列分块, 逐块做 SwiGLU, 避免大 N 时整行超 UB。
    __aicore__ inline void ComputeRow(int64_t row)
    {
        const int64_t nRow = static_cast<int64_t>(n_);
        const int64_t step = static_cast<int64_t>(tileN_);
        for (int64_t off = 0; off < nRow; off += step) {
            int64_t len = nRow - off;
            if (len > step) {
                len = step;
            }
            ComputeTile(row, off, len);
        }
    }

    // 处理第 row 行 [colOff, colOff+len) 列: gate=cGm[row,colOff:], up=cGm[row,N+colOff:], y=SiLU(gate)*up
    __aicore__ inline void ComputeTile(int64_t row, int64_t colOff, int64_t len)
    {
        DataCopyExtParams copyIn{1, static_cast<uint32_t>(len * sizeof(cType)), 0, 0, 0};
        DataCopyPadExtParams<cType> padIn{false, 0, 0, 0};

        LocalTensor<cType> gate = inQueGate_.AllocTensor<cType>();
        LocalTensor<cType> up = inQueUp_.AllocTensor<cType>();
        DataCopyPad(gate, cGm_[row * twoN_ + colOff], copyIn, padIn);
        DataCopyPad(up, cGm_[row * twoN_ + n_ + colOff], copyIn, padIn);
        inQueGate_.EnQue(gate);
        inQueUp_.EnQue(up);

        gate = inQueGate_.DeQue<cType>();
        up = inQueUp_.DeQue<cType>();
        LocalTensor<cType> sig = sigBuf_.Get<cType>();
        const int32_t n = static_cast<int32_t>(len);
        Sigmoid(sig, gate, n);
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, sig, n);   // SiLU(gate) = gate * sigmoid(gate)
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, up, n);    // * up
        PipeBarrier<PIPE_V>();

        LocalTensor<xType> yUb = outQueY_.AllocTensor<xType>();
        if constexpr (std::is_same<xType, cType>::value) {
            // 同类型直拷 UB->UB, 计数需按 32B 对齐上取整(缓冲已按 tileN 分配, 多读的列在块内)
            constexpr int32_t kBlk = 32 / static_cast<int32_t>(sizeof(cType));
            int32_t cnt = (n + kBlk - 1) / kBlk * kBlk;
            DataCopy(yUb, gate, cnt);
        } else {
            Cast(yUb, gate, RoundMode::CAST_RINT, n);
        }
        inQueGate_.FreeTensor(gate);
        inQueUp_.FreeTensor(up);
        outQueY_.EnQue(yUb);

        yUb = outQueY_.DeQue<xType>();
        DataCopyExtParams copyOut{1, static_cast<uint32_t>(len * sizeof(xType)), 0, 0, 0};
        DataCopyPad(yGm_[row * n_ + colOff], yUb, copyOut);
        outQueY_.FreeTensor(yUb);
    }

private:
    TQue<QuePosition::VECIN, NUM_BUFFER> inQueGate_;
    TQue<QuePosition::VECIN, NUM_BUFFER> inQueUp_;
    TQue<QuePosition::VECOUT, NUM_BUFFER> outQueY_;
    TBuf<TPosition::VECCALC> sigBuf_;

    GlobalTensor<xType> xGm_;
    GlobalTensor<xType> wGm_;
    GlobalTensor<cType> biasGm_;
    GlobalTensor<xType> yGm_;
    GlobalTensor<cType> cGm_;

    uint64_t m_ = 0;
    uint64_t k_ = 0;
    uint64_t n_ = 0;
    uint64_t twoN_ = 0;
    uint64_t tileN_ = 0;
    bool hasBias_ = false;
};

}  // namespace NsMatmulSwiglu
#endif  // MATMUL_SWIGLU_IMPL_HPP
