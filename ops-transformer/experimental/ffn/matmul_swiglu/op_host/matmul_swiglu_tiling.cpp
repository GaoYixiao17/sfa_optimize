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
 * \file matmul_swiglu_tiling.cpp
 * \brief MatmulSwiglu tiling
 *
 * 输出 y 为 [M, N]。gate = x @ weight[:, 0:N], up = x @ weight[:, N:2N], 视作两个
 * 共享 A=x 的 matmul。用 matmul_tiling 为单边 [M, N, K] 生成 TCubeTiling, kernel 按
 * 输出块 [baseM, baseN] 遍历, 中间结果不落 GM, 故仅需系统 workspace。
 *
 * UB 预算: 每块 gate/up/sig (fp32) + y (xType), 即 (3*4 + sizeof(xType)) * baseM *
 * baseN <= UB; 超出时需收缩 baseN。
 */

#include <climits>
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "matmul_swiglu_tiling.h"

using namespace matmul_tiling;

namespace optiling {

namespace {
constexpr size_t IN_X = 0;
constexpr size_t IN_WEIGHT = 1;
constexpr size_t IN_BIAS = 2;
constexpr size_t ATTR_TRANSPOSE_WEIGHT = 0;
constexpr int64_t SPLIT_NUM = 2;
}  // namespace

static ge::DataType GetGeDtype(gert::TilingContext* context, size_t idx)
{
    auto desc = context->GetInputDesc(idx);
    return (desc == nullptr) ? ge::DT_UNDEFINED : desc->GetDataType();
}

static DataType ToMmDataType(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16: return DataType::DT_FLOAT16;
        case ge::DT_BF16:    return DataType::DT_BF16;
        case ge::DT_FLOAT:   return DataType::DT_FLOAT;
        default:             return DataType::DT_FLOAT16;
    }
}

// tiling 中间参数: 由 ParseTilingParams 解析填充, 供后续 matmul tiling 与字段写回使用。
struct SgmTilingParams {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
    int64_t m = 0;
    int64_t k = 0;
    int64_t n = 0;
    int64_t twoN = 0;
    bool hasBias = false;
    bool transposeWeight = false;
    ge::DataType xDtype = ge::DT_FLOAT16;
};

// 解析 shape / attr / dtype 并做合法性校验。
static ge::graphStatus ParseTilingParams(
    gert::TilingContext* context, platform_ascendc::PlatformAscendC& platform, SgmTilingParams& p)
{
    p.coreNum = platform.GetCoreNumAic();
    OP_CHECK_IF(p.coreNum == 0, OP_LOGE(context, "aic core num is 0"), return ge::GRAPH_FAILED);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, p.ubSize);

    auto xShape = context->GetInputShape(IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    auto wShape = context->GetInputShape(IN_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, wShape);
    const auto attrs = context->GetAttrs();
    const bool* transWPtr = (attrs == nullptr) ? nullptr : attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_WEIGHT);
    p.transposeWeight = (transWPtr == nullptr) ? false : *transWPtr;

    const auto& xS = xShape->GetStorageShape();
    const auto& wS = wShape->GetStorageShape();
    const size_t xDim = xS.GetDimNum();
    const size_t wDim = wS.GetDimNum();
    OP_CHECK_IF(xDim < 2 || wDim != 2,
        OP_LOGE(context, "MatmulSwiglu: tiling expects x rank >= 2 and weight rank == 2, got x=%zu w=%zu",
                xDim, wDim),
        return ge::GRAPH_FAILED);

    p.k = xS.GetDim(xDim - 1);
    p.m = 1;
    for (size_t i = 0; i + 1 < xDim; i++) {
        p.m *= xS.GetDim(i);
    }
    p.twoN = p.transposeWeight ? wS.GetDim(0) : wS.GetDim(1);
    int64_t wK = p.transposeWeight ? wS.GetDim(1) : wS.GetDim(0);
    OP_CHECK_IF(p.twoN % SPLIT_NUM != 0,
        OP_LOGE(context, "MatmulSwiglu: weight 2N [%ld] must be divisible by 2", p.twoN),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(wK != p.k,
        OP_LOGE(context, "MatmulSwiglu: K mismatch x.K=%ld weight.K=%ld", p.k, wK),
        return ge::GRAPH_FAILED);
    p.n = p.twoN / SPLIT_NUM;

    // matmul tiling 的 SetShape/SetOrgShape/SetDim 接口取 int32, m/n/k/coreNum 超过 INT32_MAX
    // 会被静默截断导致维度错误; 这里显式拦截(极端大 shape 才会触发)。
    OP_CHECK_IF(p.m > INT32_MAX || p.n > INT32_MAX || p.k > INT32_MAX || p.coreNum > INT32_MAX,
        OP_LOGE(context, "MatmulSwiglu: m/n/k/coreNum [%ld/%ld/%ld/%ld] exceed INT32_MAX, unsupported",
                p.m, p.n, p.k, p.coreNum),
        return ge::GRAPH_FAILED);

    auto biasShape = context->GetOptionalInputShape(IN_BIAS);
    p.hasBias = (biasShape != nullptr);
    if (p.hasBias) {
        const auto& bS = biasShape->GetStorageShape();
        // bias 必须为 1 维且长度 == 2N, 否则 kernel 侧按 twoN 读取会越界
        OP_CHECK_IF(bS.GetDimNum() != 1 || bS.GetDim(0) != p.twoN,
            OP_LOGE(context, "MatmulSwiglu: bias must be 1-D of length 2N=%ld, got rank=%zu dim0=%ld",
                    p.twoN, bS.GetDimNum(), (bS.GetDimNum() > 0 ? bS.GetDim(0) : -1)),
            return ge::GRAPH_FAILED);
    }
    p.xDtype = GetGeDtype(context, IN_X);
    return ge::GRAPH_SUCCESS;
}

// 单边 matmul tiling: [M, N] = x[M,K] @ weightHalf[K,N], 累加 FLOAT32, 结果写入 tiling->mmTiling。
static ge::graphStatus RunMatmulTiling(
    gert::TilingContext* context, platform_ascendc::PlatformAscendC& platform, const SgmTilingParams& p,
    MatmulSwigluTilingData* tiling)
{
    MultiCoreMatmulTiling mmTiling(platform);
    mmTiling.SetDim(static_cast<int32_t>(p.coreNum));
    mmTiling.SetAType(TPosition::GM, CubeFormat::ND, ToMmDataType(p.xDtype), false);
    mmTiling.SetBType(TPosition::GM, CubeFormat::ND, ToMmDataType(p.xDtype), p.transposeWeight);
    mmTiling.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);  // L0C 累加为 fp32
    if (p.hasBias) {
        mmTiling.SetBiasType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT);
    }
    // 方案一: 一次标准 matmul 算完整 C=[M, 2N] = x[M,K] @ weight[K,2N]。
    mmTiling.SetShape(static_cast<int32_t>(p.m), static_cast<int32_t>(p.twoN), static_cast<int32_t>(p.k));
    mmTiling.SetOrgShape(static_cast<int32_t>(p.m), static_cast<int32_t>(p.twoN), static_cast<int32_t>(p.k));
    mmTiling.SetBias(p.hasBias);
    mmTiling.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(mmTiling.GetTiling(tiling->mmTiling) == -1,
        OP_LOGE(context, "MatmulSwiglu: matmul GetTiling failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 计算块数/使用核数并写回 TilingData 各字段、设置 BlockDim 与 TilingKey。
static void FillTilingData(gert::TilingContext* context, const SgmTilingParams& p, MatmulSwigluTilingData* tiling)
{
    // 向量 SwiGLU 行内分块 tileN: 每列元素占 UB = gate(2*fp32) + up(2*fp32) + sig(fp32) +
    // y(2*xType) (in/out 队列双缓冲各 2 份, sig 单份)。留 16KB 余量, 对齐 256 列。
    constexpr int64_t UB_RESERVE = 16 * 1024;
    constexpr int64_t TILE_ALIGN = 256;
    int64_t xTypeSize = (p.xDtype == ge::DT_FLOAT) ? 4 : 2;
    int64_t perColUb = 2 * 4 + 2 * 4 + 4 + 2 * xTypeSize;
    int64_t ubBudget = static_cast<int64_t>(p.ubSize) - UB_RESERVE;
    if (ubBudget < perColUb) {
        ubBudget = perColUb;
    }
    int64_t tileN = (ubBudget / perColUb / TILE_ALIGN) * TILE_ALIGN;  // 对齐下取整
    if (tileN <= 0 || tileN >= p.n) {
        tileN = p.n;  // 小 N 或一块即可容纳: 整行不切分
    }

    // m/k/2N 不再单独写入 TilingData: kernel 侧从 mmTiling(M/Ka/N) 派生
    tiling->set_hasBias(p.hasBias ? 1U : 0U);
    tiling->set_tileN(static_cast<uint32_t>(tileN));

    context->SetBlockDim(static_cast<uint32_t>(p.coreNum));

    // transpose_weight 经 TilingKey 传递(0=非转置, 1=转置), kernel 侧作编译期模板参数, 避免运行时判断。
    // dtype 由 kernel 侧编译期 DTYPE_X 决定, 各 dtype 独立二进制, 不占 TilingKey。
    context->SetTilingKey(p.transposeWeight ? 1U : 0U);
}

static ge::graphStatus MatmulSwigluTilingFunc(gert::TilingContext* context)
{
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);

    SgmTilingParams params;
    OP_CHECK_IF(ParseTilingParams(context, ascendcPlatform, params) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "MatmulSwiglu: parse tiling params failed"), return ge::GRAPH_FAILED);

    // 勿对 TilingData(TilingDef 子类, 非 POD) 做 memset: 会清零内部指针/vtable, 后续
    // set_xxx()/SaveToBuffer() 解引用空指针导致段错误。字段由构造函数零初始化。
    MatmulSwigluTilingData tilingObj;
    MatmulSwigluTilingData* tiling = &tilingObj;
    OP_CHECK_IF(RunMatmulTiling(context, ascendcPlatform, params, tiling) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "MatmulSwiglu: matmul tiling failed"), return ge::GRAPH_FAILED);

    int64_t baseM = tiling->mmTiling.get_baseM();
    int64_t baseN = tiling->mmTiling.get_baseN();
    OP_CHECK_IF(baseM <= 0 || baseN <= 0,
        OP_LOGE(context, "MatmulSwiglu: invalid baseM/baseN %ld/%ld", baseM, baseN),
        return ge::GRAPH_FAILED);

    FillTilingData(context, params, tiling);

    // workspace = 系统 lib workspace + [M,2N] fp32 中间结果(方案一 matmul 输出)
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    ws[0] = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize()) +
            static_cast<size_t>(params.m * params.twoN) * sizeof(float);

    tiling->SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling->GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMatmulSwiglu(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<MatmulSwigluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulSwiglu)
    .Tiling(MatmulSwigluTilingFunc)
    .TilingParse<MatmulSwigluCompileInfo>(TilingPrepareForMatmulSwiglu);

}  // namespace optiling
