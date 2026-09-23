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
 * \file sparse_flash_attention_antiquant.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "sparse_flash_attention_antiquant_template_tiling_key.h"
#include "sparse_flash_attention_antiquant_kernel_gqa.h"
#include "sparse_flash_attention_antiquant_metadata.h"
// #include "log.h"

using namespace AscendC;
using namespace optiling::detail;


template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitMetaData(const __gm__ uint8_t *p_metadata, T *metadata)
{
    constexpr uint64_t all_bytes = sizeof(T);
#if defined(ASCENDC_CPU_DEBUG) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__DAV_310R6_CUBE__) || defined(__GET_CODE_CHANNEL__)
    copy_data_align64((uint8_t*)metadata, (__gm__ uint8_t *)p_metadata, all_bytes);
#else
    __ubuf__ uint8_t *metadata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    constexpr uint32_t len_burst = (all_bytes + 31) / 32;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)metadata_in_ub), p_metadata, 0, 1,len_burst, 0, 0);
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    copy_data_align64((uint8_t*)metadata, (__ubuf__ uint8_t *)metadata_in_ub, all_bytes);
#endif
}

#define SFAA_OP_IMPL_GQA(templateClass, tilingdataClass, ...)                                     \
    do {                                                                                          \
        templateClass<SFAAType<__VA_ARGS__>> op;                                                  \
        GET_TILING_DATA_WITH_STRUCT(tilingdataClass, tiling_data_in, tiling);                     \
        const tilingdataClass *__restrict tiling_data = &tiling_data_in;                          \
        SfaMetaData *__restrict meta_data = nullptr;                                              \
        SfaMetaData metaDataTmp;                                                                  \
        if (metaData != nullptr) {                                                                \
            InitMetaData<SfaMetaData>(metaData, &metaDataTmp);                                    \
            meta_data = &metaDataTmp;                                                             \
        }                                                                                         \
        op.Init(query, key, value, sparseIndices, keyScale, valueScale, blocktable,               \
            actualSeqLengthsQuery, actualSeqLengthsKV, sparseSeqLengthsKV, meta_data,             \
	    attentionOut, user, tiling_data, tiling, &tPipe);                                         \
        op.Process();                                                                             \
    } while (0)

template<int ATTENTION_MODE, int FLASH_DECODE, int LAYOUT_T, int KV_LAYOUT_T, int TEMPLATE_MODE>
 __global__ __aicore__ void
sparse_flash_attention_antiquant(__gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *value,
                       __gm__ uint8_t *sparseIndices, __gm__ uint8_t* keyScale, __gm__ uint8_t* valueScale,
                       __gm__ uint8_t *blocktable, __gm__ uint8_t *actualSeqLengthsQuery,
                       __gm__ uint8_t *actualSeqLengthsKV, __gm__ uint8_t *sparseSeqLengthsKV, __gm__ uint8_t *metaData,
                       __gm__ uint8_t *attentionOut, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);

    if constexpr (ATTENTION_MODE == ATTENTION_GQA_MHA) {
        if constexpr (ORIG_DTYPE_QUERY == DT_FLOAT16 && ORIG_DTYPE_KEY == DT_INT8 &&
                    ORIG_DTYPE_ATTENTION_OUT == DT_FLOAT16) {
            SFAA_OP_IMPL_GQA(SparseFlashAttentionAntiquantGqa, SparseFlashAttentionAntiquantTilingDataMla, half, int8_t,
                half, FLASH_DECODE, static_cast<SFAA_LAYOUT>(LAYOUT_T), static_cast<SFAA_LAYOUT>(KV_LAYOUT_T),
                TEMPLATE_MODE, false);
        } else { // bf16
            SFAA_OP_IMPL_GQA(SparseFlashAttentionAntiquantGqa, SparseFlashAttentionAntiquantTilingDataMla, bfloat16_t, int8_t,
                bfloat16_t, FLASH_DECODE, static_cast<SFAA_LAYOUT>(LAYOUT_T), static_cast<SFAA_LAYOUT>(KV_LAYOUT_T),
                TEMPLATE_MODE, true);
        }
    }
}
