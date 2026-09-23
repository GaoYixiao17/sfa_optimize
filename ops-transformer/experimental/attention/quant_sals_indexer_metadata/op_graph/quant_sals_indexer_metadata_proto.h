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
 * \file quant_sals_indexer_metadata_proto.h
 * \brief
 */
#ifndef QUANT_SALS_INDEXER_METADATA_PROTO_H
#define QUANT_SALS_INDEXER_METADATA_PROTO_H

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
* @brief Function QuantSalsIndexerMetadata.

* @par Inputs:
* @li actual_seq_lengths_kv: A matrix tensor. The type support int32.
* Effective sequence length of key/value in different batches.

* @par Attributes:
* @li batch_size: An int. batch size of key/value tensor.
* @li kv_seq_size: An int. sequence len of key/value Tensor.
* @li kv_head_num: An int. head-dim of key/value Tensor.
* @li fixed_tail_count: An int. fixed tail count.
* @li sparse_block_size: An int. The block size in the sparse phase. Default: 1.
* @li aic_core_num: An int. cube core num of device
* @li aiv_core_num: An int. vector core num of device

* @par Outputs:
* @li metadata: A matrix tensor. The type support int32.
* The output of attention structure.
*/
REG_OP(QuantSalsIndexerMetadata)
    .OPTIONAL_INPUT(actual_seq_lengths_key, TensorType({DT_INT32}))
    .OUTPUT(metadata, TensorType({DT_INT32}))
    .REQUIRED_ATTR(batch_size, Int)
    .REQUIRED_ATTR(key_seq_size, Int)
    .REQUIRED_ATTR(key_head_num, Int)
    .REQUIRED_ATTR(head_dim, Int)
    .REQUIRED_ATTR(aic_core_num, Int)
    .REQUIRED_ATTR(aiv_core_num, Int)
    .REQUIRED_ATTR(sparse_block_size, Int)
    .REQUIRED_ATTR(sparse_ratio, Float)
    .REQUIRED_ATTR(fixed_tail_count, Int)
    .ATTR(layout_key, String, "BSND")
    .ATTR(soc_version, String, "ascend910B")
    .OP_END_FACTORY_REG(QuantSalsIndexerMetadata)
} // namespace ge

#endif // QUANT_SALS_INDEXER_METADATA_PROTO_H
