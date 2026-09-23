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
 * \file moe_ffn_tiling.h
 */
#ifndef MOE_FFN_TILING_H
#define MOE_FFN_TILING_H
#include <cstdint>
#include <vector>
#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
namespace optiling {

constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t NUM_PER_REPEAT_FLOAT16 = 128;
constexpr int32_t NUM_PER_BLOCK_FLOAT16 = 16;
constexpr int32_t SYS_WORKSPACE_910B = 16 * 1024 * 1024;
constexpr int32_t FRACTAL_FLOAT16 = 16;
enum class MoeFFNTilingKey : uint64_t {
    FP16NORMAL = 10000001,
    BF16NORMAL = 10000027,
    FP16Single = 11000001,
    BF16Single = 11000027,
    UNDFINED = 10000099
};

struct MoeFFNCompileInfo {};
struct MoeFFNParam {
    // platform
    uint64_t CoreNum;
    uint64_t UBSize;
    uint64_t L1Size;
    uint64_t L0ASize;
    uint64_t L0BSize;
    uint64_t L0CSize;

    ge::DataType dataType;
    uint64_t scaleGroupSize;
    uint64_t originE;
    uint64_t originM;
    uint64_t topK;
    uint64_t originN_13;
    uint64_t originK_13;
    uint64_t originN_2;
    uint64_t originK_2;
    uint64_t scaleK_13;
    uint64_t fracN_13;
    uint64_t fracK_13;
    uint64_t scaleK_2;
    uint64_t fracN_2;
    uint64_t fracK_2;

    uint64_t needBias_13;
    uint64_t needBias_2;
};

BEGIN_TILING_DATA_DEF(MoeFFNTilingData)

TILING_DATA_FIELD_DEF(uint32_t, CoreNum);
TILING_DATA_FIELD_DEF(uint32_t, dataType);

TILING_DATA_FIELD_DEF(uint32_t, UBSize);
TILING_DATA_FIELD_DEF(uint32_t, L1Size);
TILING_DATA_FIELD_DEF(uint32_t, L0ASize);
TILING_DATA_FIELD_DEF(uint32_t, L0BSize);
TILING_DATA_FIELD_DEF(uint32_t, L0CSize);

TILING_DATA_FIELD_DEF(uint32_t, scaleGroupSize);
TILING_DATA_FIELD_DEF(uint32_t, originE);
TILING_DATA_FIELD_DEF(uint32_t, originM);
TILING_DATA_FIELD_DEF(uint32_t, topK);
TILING_DATA_FIELD_DEF(uint32_t, originN_13);
TILING_DATA_FIELD_DEF(uint32_t, originK_13);
TILING_DATA_FIELD_DEF(uint32_t, scaleK_13);
TILING_DATA_FIELD_DEF(uint32_t, fracN_13);
TILING_DATA_FIELD_DEF(uint32_t, fracK_13);
TILING_DATA_FIELD_DEF(uint32_t, originN_2);
TILING_DATA_FIELD_DEF(uint32_t, originK_2);
TILING_DATA_FIELD_DEF(uint32_t, scaleK_2);
TILING_DATA_FIELD_DEF(uint32_t, fracN_2);
TILING_DATA_FIELD_DEF(uint32_t, fracK_2);

TILING_DATA_FIELD_DEF(uint32_t, needBias_13);
TILING_DATA_FIELD_DEF(uint32_t, needBias_2);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MoeFFN, MoeFFNTilingData)

class MoeFFNTiling {
public:
    ge::graphStatus runTiling(gert::TilingContext *context);

protected:
    bool GetPlatformInfo(gert::TilingContext *context);

    bool GetCheckAttr(gert::TilingContext *context);

    bool CheckTensorShape(gert::TilingContext *context, gert::Shape &shape, uint64_t ndim, std::vector<uint64_t> dims);

    bool CheckInOutShapes(gert::TilingContext *context);

    bool GetTilingData(gert::TilingContext *context);

    bool SetTilingData(gert::TilingContext *context);

    bool SetLaunchInfo(gert::TilingContext *context);

private:
    MoeFFNTilingKey tilingKey;
    MoeFFNTilingData tilingData;
    MoeFFNParam _Params;
};

ge::graphStatus TilingMoeFFN(gert::TilingContext *context);

} // namespace optiling
#endif // MOE_FFN_TILING_H
