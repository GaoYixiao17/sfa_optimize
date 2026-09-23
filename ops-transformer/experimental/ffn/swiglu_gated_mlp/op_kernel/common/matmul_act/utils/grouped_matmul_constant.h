/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file grouped_matmul_constant.h
 * \brief
 */
#ifndef UTILS_GROUPED_MATMUL_CONSTANT_H
#define UTILS_GROUPED_MATMUL_CONSTANT_H
namespace Act {
namespace Gemm {
namespace GroupedMatmul {
constexpr uint32_t GMM_BUFFER_NUM = 2;
constexpr uint16_t GMM_FLAG_ID_MAX = 16;
constexpr uint16_t GMM_AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t GMM_AIC_SYNC_AIV_FLAG = 8;
constexpr uint8_t GMM_AIC_SYNC_AIV_MODE = 4;
constexpr uint64_t GMM_MAX_STEP_SCALEA_K = 16;
constexpr uint32_t GMM_UB_ALIGN_SIZE = 32;
constexpr uint32_t GMM_MAX_REPEAT_TIMES = 255;

constexpr uint8_t GMM_SPLIT_M = 0;
constexpr uint8_t GMM_SPLIT_K = 2;
constexpr uint64_t GMM_CUBE_BLOCK = 16;
constexpr uint64_t GMM_INNER_AXIS_MIN_SPLIT_VAL = 128; // ND2NZ cacheline 128
constexpr int32_t GMM_MKN_LIST_LEN = 128;              // 128: predefined array legnth

constexpr uint32_t GMM_BMM_BLOCK_NUM = 16;
constexpr uint32_t K0_B8 = 32;
constexpr uint32_t GMM_k0_FLOAT16 = 16;
constexpr uint16_t GMM_DATA_BLOCK = 32;
} // namespace GroupedMatmul
} // namespace Gemm
} // namespace Act
#endif