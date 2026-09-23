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
 * \file stem_indexer_metadata_aicpu.h
 * \brief
 */

#ifndef STEM_INDEXER_METADATA_AICPU_H
#define STEM_INDEXER_METADATA_AICPU_H

#include <array>
#include <string>
#include <vector>
#include <limits>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "stem_indexer_metadata.h"
#include "../../../../attention/common/op_kernel/load_balance/section_stream_k/section_stream_k.h"
#include "../../../../attention/common/op_kernel/aicpu_common.h"

using namespace optiling;
using namespace std;
using namespace load_balance;

namespace aicpu {

struct StemIndexerBaseInfo : load_balance::BaseInfo {
public:
    [[nodiscard]] int64_t GetNextTokenLeftUp(uint32_t querySeq, uint32_t kvSeq) const override
    {
        auto mode = GetSparseMode();
        switch (mode) {
            case load_balance::SparseMode::DEFAULT_MASK:
            case load_balance::SparseMode::ALL_MASK:
            case load_balance::SparseMode::LEFT_UP_CAUSAL:
                return nextToken - tailSize;
            case load_balance::SparseMode::RIGHT_DOWN_CAUSAL:
                return static_cast<int64_t>(kvSeq) - static_cast<int64_t>(querySeq) - tailSize;
            case load_balance::SparseMode::BAND:
                return static_cast<int64_t>(kvSeq) - static_cast<int64_t>(querySeq) + nextToken - tailSize;
            default:
                return nextToken - tailSize;
        }
    }

public:
    int64_t tailSize { 0 };
};

class StemIndexerMetadataCpuKernel : public CpuKernel {
public:
    StemIndexerMetadataCpuKernel() = default;
    ~StemIndexerMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext &ctx) override;

private:
    bool Prepare(CpuKernelContext &ctx);
    bool GenerateDeviceInfo(DeviceInfo &deviceInfo);
    bool GenerateBaseInfo(StemIndexerBaseInfo &baseInfo);
    bool GenerateSectionStreamKParam(load_balance::SectionStreamKParam &param);
    bool BalanceSchedule(SectionStreamKResult &balanceResult);
    std::vector<int64_t> GetTensorDataAsInt64(Tensor *tensor, size_t size);
    bool GenMetadata(SectionStreamKResult &result);

private:
    CpuKernelContext *context_ = nullptr;
    // input tensor
    Tensor *qSeqLens_ = nullptr;
    Tensor *kvSeqLens_ = nullptr;

    // output tensor
    Tensor *metadata_ = nullptr;

    // input attr
    int64_t numHeadsQ_ = 0;
    int64_t numHeadsKv_ = 0;
    bool causal_ = false;
    int64_t stemBlockSize_ = 0;
    int64_t windowSize_ = 0;
    int64_t headDim_ = 0;

    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36U;
    int32_t aivCoreNum_ = 72U;
private:
    enum class ParamId : uint32_t {
        // input
        qSeqLens = 0,
        kvSeqLens = 1,
        // output
        metadata = 0,
    };
};
} // namespace aicpu

#endif // STEM_INDEXER_METADATA_AICPU_H