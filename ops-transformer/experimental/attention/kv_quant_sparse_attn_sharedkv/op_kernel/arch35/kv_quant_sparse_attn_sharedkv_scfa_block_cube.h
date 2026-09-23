/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kv_quant_sparse_attn_sharedkv_scfa_block_cube.h
 * \brief
 */
#ifndef KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
#define KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
#include "common/offset_calculator.h"
#include "common/matmul.h"
#include "common/FixpipeOut.h"
#include "common/CopyInL1.h"
#include "kernel_operator_list_tensor_intf.h"

#include "util_regbase.h"
#include "kv_quant_sparse_attn_sharedkv_common_arch35.h"

using namespace AscendC;
using namespace AscendC::Impl::Detail;
using namespace regbaseutil;
using namespace fa_base_matmul;
namespace BaseApi {

template <SAS_LAYOUT LAYOUT>
__aicore__ inline constexpr GmFormat GetQueryGmFormat()
{
    if constexpr (LAYOUT == SAS_LAYOUT::BSND) {
        return GmFormat::BSNGD;
    } else {
        return GmFormat::TNGD;
    }
}

TEMPLATES_DEF
class SCFABlockCube {
public:
    /* =================编译期常量的基本块信息================= */
    static constexpr uint32_t s1BaseSize = 64;
    static constexpr uint32_t s2BaseSize = 128;
    static constexpr uint32_t dBaseSize = 512;
    static constexpr uint32_t dBaseMatmulSize = 128;
    static constexpr uint32_t l1QBufNum = 3;
    static constexpr uint32_t l1KBufNum = 6;
    static constexpr uint32_t rightBufNum = 3;
    static constexpr uint32_t rightBufSingleSize = s2BaseSize * dBaseSize;
    static constexpr uint32_t rightBufTotalSize = rightBufSingleSize * rightBufNum;

    __aicore__ inline SCFABlockCube() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query);
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo);
    __aicore__ inline void IterateLoadQK(
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop);
    __aicore__ inline void IterateBmm1(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &output,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    __aicore__ inline void IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        RunInfo &runInfo, ConstInfo &constInfo);

private:
    __aicore__ inline void InitLocalBuffer();
    __aicore__ inline void InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo);

    __aicore__ inline void CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo);
    __aicore__ inline void IterateBmm1SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
        bool notLastTwoLoop, RunInfo &runInfoNext,
        RunInfo &runInfo, ConstInfo &constInfo);

    // --------------------Bmm2--------------------------
    __aicore__ inline void IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
        BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
        RunInfo &runInfo, ConstInfo &constInfo);
    TPipe *tPipe;
    /* =====================GM变量==================== */
    static constexpr GmFormat Q_FORMAT = GetQueryGmFormat<LAYOUT_T>();
    FaGmTensor<Q_T, Q_FORMAT> queryGm;

    /* =====================运行时变量==================== */
    uint32_t l1QBufIdx = 0; // 3 buffer,0-2
    uint32_t l1QBufId = 0; // 0-2，用于l1Q
    uint32_t l1KLoadBufIdx = 0; // 6 buffer,0-5
    uint32_t l1KMatmul1BufIdx = 0;
    uint32_t l1KMatmul2BufIdx = 0;
    uint32_t l1KBufId = 3; // 3-8, 用于l1K

    uint32_t l0ABBufIdx = 0; // 2 buffer, 0-1
    uint32_t l0ABBufId = 9; // 9-10, 用于L0AB
    uint32_t l0CBufIdx = 0; // 2 buffer, 0-1
    uint32_t l0CBufId = 11; // 11-12, 用于L0C

    /* =====================LocalBuffer变量==================== */
    TBuf<TPosition::A1> l1QBuffers;
    TBuf<TPosition::B1> l1RightBuffers;
    TBuf<TPosition::A2> mmL0ABuffers;
    TBuf<TPosition::B2> mmL0BBuffers;
    TBuf<TPosition::CO1> mmL0CBuffers;
    LocalTensor<Q_T> l1RightTensor;
    LocalTensor<Q_T> l1QTensor;
    LocalTensor<Q_T> mmL0ATensor;
    LocalTensor<Q_T> mmL0BTensor;
    LocalTensor<T> mmL0CTensor;
};

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeBlock(
    TPipe *pipe, __gm__ uint8_t *query)
{
    if ASCEND_IS_AIC {
        tPipe = pipe;
        this->queryGm.gmTensor.SetGlobalBuffer((__gm__ Q_T *)query);
        InitLocalBuffer();
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo)
{
    if ASCEND_IS_AIC {
        InitGmTensor(cuSeqlensQ, constInfo);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitLocalBuffer()
{
    tPipe->InitBuffer(l1RightBuffers, rightBufTotalSize * sizeof(Q_T));
    l1RightTensor = l1RightBuffers.Get<Q_T>();

    tPipe->InitBuffer(l1QBuffers, BUFFER_SIZE_96K);
    l1QTensor = l1QBuffers.Get<Q_T>();

    tPipe->InitBuffer(mmL0ABuffers, BUFFER_SIZE_32K);
    tPipe->InitBuffer(mmL0BBuffers, BUFFER_SIZE_64K);
    tPipe->InitBuffer(mmL0CBuffers, BUFFER_SIZE_256K);
    mmL0ATensor = mmL0ABuffers.Get<Q_T>();
    mmL0BTensor = mmL0BBuffers.Get<Q_T>();
    mmL0CTensor = mmL0CBuffers.Get<T>();
}

/* 初始化GmTensor,设置shape信息并计算strides */
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::InitGmTensor(__gm__ uint8_t *cuSeqlensQ, const ConstInfo &constInfo)
{
    if constexpr (LAYOUT_T == SAS_LAYOUT::BSND) {
        this->queryGm.offsetCalculator.Init(constInfo.bSize, constInfo.n2Size, constInfo.gSize,
            constInfo.s1Size, constInfo.dSize);
    } else {  // SAS_LAYOUT::TND
        GlobalTensor<int32_t> actualSeqQLen;
        actualSeqQLen.SetGlobalBuffer((__gm__ int32_t *)cuSeqlensQ);
        this->queryGm.offsetCalculator.Init(constInfo.n2Size, constInfo.gSize, constInfo.dSize,
            actualSeqQLen, constInfo.actualSeqLenSize);
    }
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm1SCFA(outputBuf, v0ResGm, notLastTwoLoop, runInfoNext, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    IterateBmm2SCFA(outputBuf, inputLeftBuffers, runInfo, constInfo);
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::CopyQGmToL1(RunInfo &runInfo, ConstInfo &constInfo)
{
    uint64_t gmOffset = this->queryGm.offsetCalculator.GetOffset(runInfo.boIdx, runInfo.n2oIdx, runInfo.goIdx,
        runInfo.s1oIdx * runInfo.qSNumInOneBlock, 0);
    for (uint32_t i = 0; i < 2; i++) {
        uint32_t curL1QBufId = (l1QBufIdx + i) % l1QBufNum;
        get_buf(PIPE_MTE2, l1QBufId + curL1QBufId, false);
        rls_buf(PIPE_MTE2, l1QBufId + curL1QBufId, false);
        uint64_t curGmOffset = gmOffset + i * (constInfo.dSize >> 1);
        CopyToL1Nd2Nz<Q_T>(l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
            this->queryGm.gmTensor[curGmOffset], runInfo.mRealSize, constInfo.dSize >> 1,
            constInfo.mm1Ka);
        get_buf(PIPE_MTE2, l1QBufId + curL1QBufId, true);
        rls_buf(PIPE_MTE2, l1QBufId + curL1QBufId, true);
    }
}
 	 
TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateLoadQK(
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    RunInfo &runInfo, ConstInfo &constInfo, bool isFirstLoop)
{
    if (unlikely(isFirstLoop)) {
        CopyQGmToL1(runInfo, constInfo);
    }

    // 加载当前轮的右矩阵到L1
    get_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx, false);
    rls_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx, false);
    get_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx + 1, false);
    rls_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx + 1, false); // 等待两个子buffer
    LocalTensor<Q_T> dst = l1RightTensor[runInfo.taskIdMod3 * rightBufSingleSize];
    v0ResGm.WaitCrossCore();
    if constexpr (IS_SPLIT_G) {
        CrossCoreSetFlag<0, PIPE_MTE2>(10);
        CrossCoreWaitFlag<0, PIPE_MTE2>(10);
    }
    GlobalTensor<Q_T> v0ResGmTensor = v0ResGm.template GetTensor<Q_T>();
    DataCopy(dst, v0ResGmTensor, Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1));
    get_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx, true);
    rls_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx, true);

    l1KLoadBufIdx = (l1KLoadBufIdx + 1) % l1KBufNum;
    DataCopy(dst[Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1)],
        v0ResGmTensor[Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1)],
        Align16Func(runInfo.s2RealSize) * (constInfo.dSize >> 1));
    get_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx, true);
    rls_buf(PIPE_MTE2, l1KBufId + l1KLoadBufIdx, true);
    l1KLoadBufIdx = (l1KLoadBufIdx + 1) % l1KBufNum;
}
 	

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm1SCFA(
    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    Buffer<BufferType::GM, SyncType::CROSS_CORE_SYNC_BACKWARD> &v0ResGm,
    bool notLastTwoLoop, RunInfo &runInfoNext,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    LocalTensor<Q_T> curL1RightTensor = l1RightTensor[runInfo.taskIdMod3 * rightBufSingleSize];
    get_buf(PIPE_MTE1, l1KBufId + l1KMatmul1BufIdx, false);
    rls_buf(PIPE_MTE1, l1KBufId + l1KMatmul1BufIdx, false);
    get_buf(PIPE_M, l0CBufId + l0CBufIdx, false);
    rls_buf(PIPE_M, l0CBufId + l0CBufIdx, false);

    MMParam param = {static_cast<uint32_t>(runInfo.mRealSize),     // singleM
                     static_cast<uint32_t>(runInfo.s2RealSize),  // singleN
                     static_cast<uint32_t>(constInfo.dSize >> 1),   // singleK
                     0,    // isLeftTranspose
                     1     // isRightTranspose
                    };
    uint32_t curL1QBufId = l1QBufIdx;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        get_buf(PIPE_MTE1, l1QBufId + curL1QBufId, false);
        rls_buf(PIPE_MTE1, l1QBufId + curL1QBufId, false);
    }
    
    // m,n不切，k切128，mm1B直接用tensor的数据
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
        curL1RightTensor,
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx],
        param, l0ABBufId, l0ABBufIdx);

    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % l1KBufNum;
    get_buf(PIPE_MTE1, l1KBufId + l1KMatmul1BufIdx, false);
    rls_buf(PIPE_MTE1, l1KBufId + l1KMatmul1BufIdx, false);
    l1KMatmul1BufIdx = (l1KMatmul1BufIdx + 1) % l1KBufNum;

    curL1QBufId = (curL1QBufId + 1) % l1QBufNum;
    if (unlikely(runInfo.s2LoopCount == 0)) {
        get_buf(PIPE_MTE1, l1QBufId + curL1QBufId, false);
        rls_buf(PIPE_MTE1, l1QBufId + curL1QBufId, false);
    }
    param.singleK = constInfo.dSize - param.singleK;
    param.isOutKFisrt = false;

    // m,n不切，k切128, mm1B直接用tensor的数据
    MatmulK1<Q_T, Q_T, T, s1BaseSize, s2BaseSize, dBaseMatmulSize, ABLayout::MK, ABLayout::KN>(
        l1QTensor[curL1QBufId * BUFFER_SIZE_16K],
        curL1RightTensor[(constInfo.dSize >> 1) * Align16Func(runInfo.s2RealSize)],
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx],
        param, l0ABBufId, l0ABBufIdx);

    if (unlikely(runInfo.s2LoopCount == runInfo.s2LoopLimit)) {
        get_buf(PIPE_MTE1, l1QBufId + l1QBufIdx, true);
        rls_buf(PIPE_MTE1, l1QBufId + l1QBufIdx, true);
        get_buf(PIPE_MTE1, l1QBufId + curL1QBufId, true);
        rls_buf(PIPE_MTE1, l1QBufId + curL1QBufId, true);
        l1QBufIdx = (l1QBufIdx + 2) % l1QBufNum; // 偏移2个buffer
        if (notLastTwoLoop) {
            CopyQGmToL1(runInfoNext, constInfo);
        }
    }

    get_buf(PIPE_M, l0CBufId + l0CBufIdx, true);
    rls_buf(PIPE_M, l0CBufId + l0CBufIdx, true);
    get_buf(PIPE_FIX, l0CBufId + l0CBufIdx, false);
    rls_buf(PIPE_FIX, l0CBufId + l0CBufIdx, false);

    outputBuf.WaitCrossCore();
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB
    fixpipeParams.nSize = Align8Func(runInfo.s2RealSize); // L0C上的bmm1结果矩阵N方向的size大小; 同mmadParams.n; 为什么要8个元素对齐(32B对齐) // 128
    fixpipeParams.mSize = Align2Func(runInfo.mRealSize); // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小(必须为偶数) // 128
    fixpipeParams.srcStride = Align16Func(fixpipeParams.mSize); // L0C上bmm1结果相邻连续数据片段间隔(前面一个数据块的头与后面数据块的头的间隔), 单位为16*sizeof(T) // 源Nz矩阵中相邻大Z排布的起始地址偏移
    fixpipeParams.dstStride = s2BaseSize; // mmResUb上两行之间的间隔，单位：element。 // 128:根据比对dump文件得到, ND方案(S1*S2)时脏数据用mask剔除
    fixpipeParams.dualDstCtl = 1; // 双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;

    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx], fixpipeParams); // 将matmul结果从L0C搬运到UB
    get_buf(PIPE_FIX, l0CBufId + l0CBufIdx, true);
    rls_buf(PIPE_FIX, l0CBufId + l0CBufIdx, true);
    l0CBufIdx ^= 1;
    outputBuf.SetCrossCore();
}

TEMPLATES_DEF_NO_DEFAULT
__aicore__ inline void SCFABlockCube<TEMPLATE_ARGS>::IterateBmm2SCFA(Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> &outputBuf,
    BuffersPolicyDB<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> &inputLeftBuffers,
    RunInfo &runInfo, ConstInfo &constInfo)
{
    LocalTensor<Q_T> curL1RightTensor = l1RightTensor[runInfo.taskIdMod3 * rightBufSingleSize];
    Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffer = inputLeftBuffers.Get(); // P直接用无需搬运
    l1PBuffer.WaitCrossCore();
    get_buf(PIPE_M, l0CBufId + l0CBufIdx, false);
    rls_buf(PIPE_M, l0CBufId + l0CBufIdx, false);
    MMParam param = {static_cast<uint32_t>(s1BaseSize),          // singleM 64
                     static_cast<uint32_t>(constInfo.dSizeV), // singleN 512
                     static_cast<uint32_t>(runInfo.s2RealSize), // singleK 128
                     0,    // isLeftTranspose
                     0     // isRightTranspose
                     };
    MatmulN1<Q_T, Q_T, T, s1BaseSize, dBaseMatmulSize, s2BaseSize, ABLayout::MK, ABLayout::KN>(
        l1PBuffer.GetTensor<Q_T>(),
        curL1RightTensor,
        mmL0ATensor, mmL0BTensor, mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx],
        param, l0ABBufId, l0ABBufIdx);
    get_buf(PIPE_M, l0CBufId + l0CBufIdx, true);
    rls_buf(PIPE_M, l0CBufId + l0CBufIdx, true);
    get_buf(PIPE_FIX, l0CBufId + l0CBufIdx, false);
    rls_buf(PIPE_FIX, l0CBufId + l0CBufIdx, false);

    get_buf(PIPE_MTE1, l1KBufId + l1KMatmul2BufIdx, true);
    rls_buf(PIPE_MTE1, l1KBufId + l1KMatmul2BufIdx, true);
    l1KMatmul2BufIdx = (l1KMatmul2BufIdx + 1) % l1KBufNum;
    get_buf(PIPE_MTE1, l1KBufId + l1KMatmul2BufIdx, true);
    rls_buf(PIPE_MTE1, l1KBufId + l1KMatmul2BufIdx, true);
    l1KMatmul2BufIdx = (l1KMatmul2BufIdx + 1) % l1KBufNum; // 释放两个子buffer

    outputBuf.WaitCrossCore(); //占用
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams; // L0C→UB;FixpipeParamsM300:L0C→UB
    fixpipeParams.nSize = Align8Func(constInfo.dSizeV);    // L0C上的bmm1结果矩阵N方向的size大小, 分档计算且vector2中通过mask筛选出实际有效值
    fixpipeParams.mSize = s1BaseSize;                      // 有效数据不足16行，只需要输出部分行即可; L0C上的bmm1结果矩阵M方向的size大小; 同mmadParams.m
    fixpipeParams.srcStride = Align16Func(s1BaseSize);     // L0C上bmm1结果相邻连续数据片段间隔（前面一个数据块的头与后面数据块的头的间隔）
    fixpipeParams.dstStride = Align16Func(constInfo.dSizeV);
    fixpipeParams.dualDstCtl = 1;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    Fixpipe<T, T, PFA_CFG_ROW_MAJOR_UB>(outputBuf.template GetTensor<T>(),
        mmL0CTensor[BUFFER_SIZE_32K * l0CBufIdx], fixpipeParams); // 将matmul结果从L0C搬运到UB
    get_buf(PIPE_FIX, l0CBufId + l0CBufIdx, true);
    rls_buf(PIPE_FIX, l0CBufId + l0CBufIdx, true);
    l0CBufIdx ^= 1;

    outputBuf.SetCrossCore();
}

TEMPLATES_DEF
class SCFABlockCubeDummy {
public:
    __aicore__ inline SCFABlockCubeDummy() {};
    __aicore__ inline void InitCubeBlock(TPipe *pipe, __gm__ uint8_t *query) {}
    __aicore__ inline void InitCubeInput(__gm__ uint8_t *cuSeqlensQ, const ConstInfo& constInfo) {}
};

template <typename T>
struct CubeBlockTraits;  // 声明

/* 生成CubeBlockTraits */
#define GEN_TRAIT_TYPE(name, ...) using name##_TRAITS = name;
#define GEN_TRAIT_CONST(name, type, ...) static constexpr type name##Traits = name;

#define DEFINE_CUBE_BLOCK_TRAITS(CUBE_BLOCK_CLASS) \
    TEMPLATES_DEF_NO_DEFAULT \
    struct CubeBlockTraits<CUBE_BLOCK_CLASS<TEMPLATE_ARGS>> { \
        CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TRAIT_TYPE) \
        CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_TRAIT_CONST) \
    }

DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCube);
DEFINE_CUBE_BLOCK_TRAITS(SCFABlockCubeDummy);

// /* 生成Arg Traits, kernel中只需要调用ARGS_TRAITS就可以获取所有CubeBlock中的模板参数 */
#define GEN_ARGS_TYPE(name, ...) using name = typename CubeBlockTraits<CubeBlockType>::name##_TRAITS;
#define GEN_ARGS_CONST(name, type, ...) static constexpr type name = CubeBlockTraits<CubeBlockType>::name##Traits;
#define ARGS_TRAITS \
    CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_ARGS_TYPE) \
    CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_ARGS_CONST)
}
#endif // KV_QUANT_SPARSE_ATTN_SHAREDKV_SCFA_BLOCK_CUBE_H
