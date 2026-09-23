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
 * \file quant_lightning_indexer_metadata_proto.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_METADATA_PROTO_H
#define QUANT_LIGHTNING_INDEXER_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

REG_OP(QuantLightningIndexerMetadata)
    .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actual_seq_lengths_key, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .REQUIRED_ATTR(soc_version, String)
    .REQUIRED_ATTR(num_heads_q, Int)
    .REQUIRED_ATTR(num_heads_k, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .REQUIRED_ATTR(query_quant_mode, Int)
    .REQUIRED_ATTR(key_quant_mode, Int)
    .ATTR(batch_size, Int, 0)
    .ATTR(max_seqlen_q, Int, 0)
    .ATTR(max_seqlen_k, Int, 0)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_key, String, "BSND")
    .ATTR(sparse_count, Int, 2048)
    .ATTR(sparse_mode, Int, 3)
    .ATTR(pre_tokens, Int, 9223372036854775807)
    .ATTR(next_tokens, Int, 9223372036854775807)
    .ATTR(cmp_ratio, Int, 1)
    .OP_END_FACTORY_REG(QuantLightningIndexerMetadata)
} // namespace ge

#endif // QUANT_LIGHTNING_INDEXER_METADATA_PROTO_H

// FD