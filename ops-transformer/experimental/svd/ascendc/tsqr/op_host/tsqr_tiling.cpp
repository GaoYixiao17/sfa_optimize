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
 * \file tsqr_tiling.cpp
 * \brief
 */
#include <cmath>
#include <utility>
#include "tsqr_tiling.h"

using namespace ge;

namespace optiling {

thread_local static TsqrTilingData tilingData;

struct TsqrCompileInfo {};

class TsqrTiling {
public:
    TsqrTiling() {}
    ~TsqrTiling() {}
    ge::graphStatus RunBigKernelTiling(gert::TilingContext* context);
};

std::pair<int, int> getTmpSize(int M, int N, int blockSize, int numLevels)
{
    int numBlocks = blockSize > 0 ? M / blockSize : 1;
    int numPairs = numBlocks / 2;
    int tail = (numBlocks % 2 > 0);
    int aOffset = 0;
    int qOffset = numBlocks * blockSize * N;
    int rOffset = numBlocks;
    int localM = numBlocks * N;
    for (int lvl = 0; lvl < numLevels - 1; lvl++) {
        aOffset += numBlocks;
        qOffset += (numPairs * 2 + tail) * N * N;
        rOffset += (numPairs + tail);
        numBlocks = numPairs + tail;
        numPairs = numBlocks / 2;
        tail = (numBlocks % 2 > 0);
        localM = numBlocks * N;
    }
    qOffset += 2 * N * N;
    return { qOffset, rOffset * N * N };
}

int getBlockSize(gert::TilingContext* context, int M, int N)
{
    int blockSize = 0;
    auto attrs = context->GetAttrs();
    if (attrs) {
        auto blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
        blockSize = *blockSizePtr;
    }
    if (!blockSize) {
        blockSize = 16;
        if (blockSize < N * 2) blockSize = N * 2;
        if (M / 4 > 1024) blockSize = 1024;
    }
    return blockSize;
}

bool checkLimitations(int M, int N, int blockSize, int numBlocks)
{
    return (
        M >= 128 && N >= 16 // MIN Shape
        && M <= 8 * 1024 * 1024 && N <= 160 // MAX Shape
        && M >= N * 8
        && blockSize > 0
        && N * 2 <= blockSize && blockSize <= M / 4
        && N % 8 == 0 && N <= 168
        && M % blockSize == 0
        && ((numBlocks & (numBlocks - 1)) == 0) // Is Power Of Two
    );
}

void getMatmulTiling(matmul_tiling::MultiCoreMatmulTiling& mmTiling, int N, bool isTransposeB, int coreNum)
{
    int singleM = 2 * N;
    int singleN = N;
    int singleK = N;
    int K = N;
    int M = singleN;

    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      static_cast<matmul_tiling::DataType>(ge::DT_FLOAT), true);
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      static_cast<matmul_tiling::DataType>(ge::DT_FLOAT), isTransposeB);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      static_cast<matmul_tiling::DataType>(ge::DT_FLOAT));
    mmTiling.SetBias(false);
    mmTiling.SetSingleShape(singleM, singleN, singleK);
    mmTiling.SetOrgShape(M, N, K);
    mmTiling.SetBufferSpace(-1, -1, 0, -1);
    mmTiling.SetDim(coreNum);
}

bool setMatmulTilingData(gert::TilingContext* context, TsqrTilingData& tilingData, int coreNum, int N)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling mmTiling(platformInfo);
    getMatmulTiling(mmTiling, N, true, coreNum);

    if (mmTiling.GetTiling(tilingData.mmTilingData) == -1) {
        std::cout << "Matmul tiling data is None" << std::endl;
        return false;
    }

    matmul_tiling::MultiCoreMatmulTiling mmTilingF(platformInfo);
    getMatmulTiling(mmTilingF, N, false, coreNum);

    if (mmTilingF.GetTiling(tilingData.mmTilingDataF) == -1) {
        std::cout << "Matmul tiling data F is None" << std::endl;
        return false;
    }
    return true;
}

int64_t allocWorkspace(TsqrTilingData& tilingData, int M, int N, int blockSize,
                       int numLevels, int batchSize, int coreNum)
{
    auto tmpSize = getTmpSize(M, N, blockSize, numLevels);
    int64_t tmpQSize = tmpSize.first + 2 * N * N;
    int64_t tmpRSize = tmpSize.second;
    int64_t bufferQSize = M * N;
    int64_t maxQrWorkspace = N * blockSize;
    int64_t totalWorkspaceSize = tmpQSize * (batchSize > 1 ? 2 : 1) + bufferQSize + tmpRSize +
                                 maxQrWorkspace * coreNum * 2;

    tilingData.set_tmpQSize(tmpQSize);
    tilingData.set_tmpRSize(tmpRSize);
    tilingData.set_bufferQSize(bufferQSize);
    tilingData.set_maxQrWorkspace(maxQrWorkspace);

    return totalWorkspaceSize;
}

ge::graphStatus TsqrTiling::RunBigKernelTiling(gert::TilingContext* context)
{
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // Last 2 dimentions in the shape are {..., M, N} and all previous numbers accumulated to batch size
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    int32_t dimIdx = (int32_t)inputShape->GetOriginShape().GetDimNum() - 1;
    int32_t N = static_cast<int32_t>(inputShape->GetStorageShape().GetDim(dimIdx--));
    int32_t M = static_cast<int32_t>(inputShape->GetStorageShape().GetDim(dimIdx--));
    int32_t batchSize = 1;
    while (dimIdx >= 0) {
        int32_t dim = static_cast<int32_t>(inputShape->GetStorageShape().GetDim(dimIdx--));
        batchSize *= dim > 0 ? dim : 1;
    }

    int32_t coreNum = platformInfo.GetCoreNum() / 2;
    int32_t blockSize = getBlockSize(context, M, N);
    int32_t numBlocks = blockSize > 0 ? M / blockSize : 1;
    int32_t numLevels = (int32_t)(std::ceil(std::log2(numBlocks)));

    if (!checkLimitations(M, N, blockSize, numBlocks)) {
        std::cout << "Out of shape limitations" << std::endl;
        return ge::GRAPH_FAILED;
    }

    tilingData.set_batchSize(batchSize);
    tilingData.set_m(M);
    tilingData.set_n(N);
    tilingData.set_blockSize(blockSize);
    tilingData.set_numBlocks(numBlocks);
    tilingData.set_numLevels(numLevels);
    tilingData.set_ubSize(ubSize);
    int64_t totalWorkspaceSize = allocWorkspace(tilingData, M, N, blockSize, numLevels, batchSize, coreNum);

    if (!setMatmulTilingData(context, tilingData, coreNum, N)) {
        return ge::GRAPH_FAILED;
    }

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    context->SetBlockDim(coreNum);
    size_t userWorkspaceSize = totalWorkspaceSize * sizeof(float);
    size_t systemWorkspaceSize = static_cast<size_t>(platformInfo.GetLibApiWorkSpaceSize());
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingTsqr(gert::TilingContext* context)
{
    TsqrTiling tsqrTiling;
    auto ret = tsqrTiling.RunBigKernelTiling(context);
    return ret;
}

ge::graphStatus TilingPrepareForTsqr(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(Tsqr)
.Tiling(TilingTsqr)
.TilingParse<TsqrCompileInfo>(TilingPrepareForTsqr);

} // namespace optiling
