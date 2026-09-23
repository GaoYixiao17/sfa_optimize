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
 * \file quant_sals_indexer_tiling.cpp
 * \brief
 */

#include "quant_sals_indexer_tiling.h"
#include "../op_kernel/quant_sals_indexer_template_tiling_key.h"
#include "split_core.h"

using namespace ge;
using namespace AscendC;
using std::map;
using std::string;
using namespace optiling::qsi;
namespace optiling {
// --------------------------QSIInfoParser类成员函数定义-------------------------------------
ge::graphStatus QSIInfoParser::CheckRequiredInOutExistence() const
{
    OPS_ERR_IF(opParamInfo_.query.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor query is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor k is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor k is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.query_dequant_scale.shape == nullptr,
               OPS_LOG_E(opName_, "Shape of tensor query_dequant_scale is nullptr"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.query_dequant_scale.desc == nullptr,
               OPS_LOG_E(opName_, "Desc of tensor query_dequant_scale is nullptr"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key_dequant_scale.shape == nullptr,
               OPS_LOG_E(opName_, "Shape of tensor key_dequant_scale is nullptr"), return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.key_dequant_scale.desc == nullptr,
               OPS_LOG_E(opName_, "Desc of tensor key_dequant_scale is nullptr"), return ge::GRAPH_FAILED);

    OPS_ERR_IF(opParamInfo_.sparseIndices.shape == nullptr, OPS_LOG_E(opName_, "Shape of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(opParamInfo_.sparseIndices.desc == nullptr, OPS_LOG_E(opName_, "Desc of tensor output is nullptr"),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::CheckRequiredAttrExistence() const
{
    OPS_ERR_IF(opParamInfo_.layOutKey == nullptr, OPS_LOG_E(opName_, "attr layout_key is nullptr"),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::CheckRequiredParaExistence() const
{
    if (CheckRequiredInOutExistence() != ge::GRAPH_SUCCESS || CheckRequiredAttrExistence() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetOpName()
{
    if (context_->GetNodeName() == nullptr) {
        OPS_LOG_E("SalsIndexer", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetNpuInfo()
{
    platformInfo_ = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo_ == nullptr, OPS_LOG_E(opName_, "GetPlatformInfo is nullptr."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo_);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    OPS_ERR_IF(aicNum == 0 || aivNum == 0, OPS_LOG_E(opName_, "num of core obtained is 0."), return GRAPH_FAILED);
    OPS_ERR_IF(aicNum != 24, OPS_LOG_E(opName_, "num of core only support 24."), return GRAPH_FAILED);

    OPS_ERR_IF(context_->GetWorkspaceSizes(1) == nullptr, OPS_LOG_E(opName_, "workSpaceSize got from ge is nullptr"),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
               OPS_LOG_E(context_->GetNodeName(), "RawTilingData got from GE context is nullptr."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

void QSIInfoParser::GetOptionalInputParaInfo()
{
    opParamInfo_.actualSeqLengths.tensor = context_->GetOptionalInputTensor(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.actualSeqLengths.desc = context_->GetOptionalInputDesc(ACTUAL_SEQ_K_INDEX);
    opParamInfo_.blockTable.tensor = context_->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    opParamInfo_.blockTable.desc = context_->GetOptionalInputDesc(BLOCK_TABLE_INDEX);
}

void QSIInfoParser::GetInputParaInfo()
{
    opParamInfo_.query.desc = context_->GetInputDesc(QUERY_INDEX);
    opParamInfo_.query.shape = context_->GetInputShape(QUERY_INDEX);
    opParamInfo_.key.desc = context_->GetInputDesc(KEY_INDEX);
    opParamInfo_.key.shape = context_->GetInputShape(KEY_INDEX);
    opParamInfo_.query_dequant_scale.desc = context_->GetInputDesc(QUERY_DEQUANT_SCALE_INDEX);
    opParamInfo_.query_dequant_scale.shape = context_->GetInputShape(QUERY_DEQUANT_SCALE_INDEX);
    opParamInfo_.key_dequant_scale.desc = context_->GetInputDesc(KEY_DEQUANT_SCALE_INDEX);
    opParamInfo_.key_dequant_scale.shape = context_->GetInputShape(KEY_DEQUANT_SCALE_INDEX);

    GetOptionalInputParaInfo();
}

void QSIInfoParser::GetOutputParaInfo()
{
    opParamInfo_.sparseIndices.desc = context_->GetOutputDesc(SPARSE_INDICES_INDEXER);
    opParamInfo_.sparseIndices.shape = context_->GetOutputShape(SPARSE_INDICES_INDEXER);
}

ge::graphStatus QSIInfoParser::GetAndCheckAttrParaInfo()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_REPORT_VECTOR_INNER_ERR(context_->GetNodeName(), "attrs got from ge is nullptr"),
               return ge::GRAPH_FAILED);

    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo start");
    opParamInfo_.maxSeqlenKey = attrs->GetAttrPointer<int32_t>(ATTR_MAX_SEQLEN_KEY_INDEX);
    OPS_ERR_IF(opParamInfo_.maxSeqlenKey == nullptr, OPS_LOG_E(opName_, "input attr max_seqlen_key must not be null"),
               return ge::GRAPH_FAILED);

    opParamInfo_.sparseBlockSize = attrs->GetAttrPointer<int32_t>(ATTR_SPARSE_BLOCK_SIZE_INDEX);
    OPS_ERR_IF(opParamInfo_.sparseBlockSize == nullptr,
               OPS_LOG_E(opName_, "input attr sparse_block_size must not be null"), return ge::GRAPH_FAILED);

    opParamInfo_.sparseRatio = attrs->GetAttrPointer<float>(ATTR_SPARSE_RATIO_INDEX);
    OPS_ERR_IF(opParamInfo_.sparseRatio == nullptr, OPS_LOG_E(opName_, "input attr sparse_ratio must not be null"),
               return ge::GRAPH_FAILED);

    opParamInfo_.fixedTailCount = attrs->GetAttrPointer<int32_t>(ATTR_FIXED_TAIL_COUNT_INDEX);
    OPS_ERR_IF(opParamInfo_.fixedTailCount == nullptr,
               OPS_LOG_E(opName_, "input attr fixed_tail_count must not be null"), return ge::GRAPH_FAILED);

    OPS_ERR_IF(
        (*opParamInfo_.sparseBlockSize) <= 0 || (*opParamInfo_.sparseRatio) <= 0 ||
            (*opParamInfo_.maxSeqlenKey) <= 0 || (*opParamInfo_.fixedTailCount) <= 0,
        OPS_LOG_E(opName_,
                  "input attr max_seqlen_key, sparse_block_size, sparse_ratio,fixed_tail_count must greater than "
                  "0, but max_seqlen_key: %d, sparse_block_size: %d, sparse_ratio: %f, fixed_tail_count: %d.",
                  (*opParamInfo_.maxSeqlenKey), (*opParamInfo_.sparseBlockSize), (*opParamInfo_.sparseRatio),
                  (*opParamInfo_.fixedTailCount)),
        return ge::GRAPH_FAILED);
    
    OPS_ERR_IF(
        (*opParamInfo_.sparseBlockSize) != 16,
        OPS_LOG_E(opName_,
                  "input attr sparse_block_size, must equal %d"
                  ", but sparse_block_size: %d.",
                  16, (*opParamInfo_.sparseBlockSize)),
        return ge::GRAPH_FAILED);

    opParamInfo_.layOutKey = attrs->GetStr(ATTR_KEY_LAYOUT_INDEX);
    if (opParamInfo_.layOutKey != nullptr) {
        OPS_LOG_I(context_->GetNodeName(), "layout_key is:%s", opParamInfo_.layOutKey);
    }

    OPS_ERR_IF(
        ((std::string(opParamInfo_.layOutKey) != "PA_BNSD") && (std::string(opParamInfo_.layOutKey) != "PA_NZ")),
        OPS_LOG_E(opName_, "input attr layout_key only supported PA_BNSD, PA_NZ"), return ge::GRAPH_FAILED);

    OPS_LOG_I(context_->GetNodeName(), "GetAndCheckAttrParaInfo end");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetOpParaInfo()
{
    GetInputParaInfo();
    GetOutputParaInfo();
    if (ge::GRAPH_SUCCESS != GetAndCheckAttrParaInfo()) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetAndCheckInOutDataType()
{
    inputQType_ = opParamInfo_.query.desc->GetDataType();
    inputQType_ = (inputQType_== ge::DT_INT32) ? ge::DT_INT4 : ge::DT_INT8;
    inputKType_ = opParamInfo_.key.desc->GetDataType();
    inputKType_ = (inputKType_== ge::DT_INT32) ? ge::DT_INT4 : ge::DT_INT8;
    inputQueryScaleType_ = opParamInfo_.query_dequant_scale.desc->GetDataType();
    inputKeyScaleType_ = opParamInfo_.key_dequant_scale.desc->GetDataType();

    sparseIndicesType_ = opParamInfo_.sparseIndices.desc->GetDataType();

    bool inDTypeAllEqual = (inputQType_ == inputKType_);
    OPS_ERR_IF(!inDTypeAllEqual,
               OPS_LOG_E(opName_, "The data types of the input query and key must be the same."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF((inputQType_ !=ge::DT_INT4),
               OPS_LOG_E(opName_, "The data types of the input query and key must be int4."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(!(inputQueryScaleType_ == inputKeyScaleType_),
               OPS_LOG_E(opName_, "The data types of the input query_dequant_scale and key_dequant_scale must be the same."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF((inputQueryScaleType_ != ge::DT_FLOAT),
               OPS_LOG_E(opName_, "The data types of the input query_dequant_scale and key_dequant_scale must be float32."),
               return ge::GRAPH_FAILED);

    OPS_ERR_IF(sparseIndicesType_ != ge::DT_INT32,
               OPS_LOG_E(opName_, "The data types of the output sparse_indices must be int32."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetQueryKeyAndOutLayout()
{
    // 获取query,key的Layout基准值
    const map<string, DataLayout> layoutMap = {
        {"BSND", DataLayout::BSND},
        {"PA_BNSD", DataLayout::PA_BNSD},
        {"PA_BSND", DataLayout::PA_BSND},
        {"PA_NZ", DataLayout::PA_NZ},
    };

    std::string layoutKey(opParamInfo_.layOutKey);
    auto itKey = layoutMap.find(layoutKey);
    if (itKey != layoutMap.end()) {
        kLayout_ = itKey->second;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetAndCheckOptionalInput()
{
    if (kLayout_ == DataLayout::PA_BNSD || kLayout_ == DataLayout::PA_NZ) {
        OPS_ERR_IF(opParamInfo_.blockTable.tensor == nullptr,
                   OPS_LOG_E(opName_, "key layout supported PA_BNSD or PA_NZ, input block_table must not be null"),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            opParamInfo_.actualSeqLengths.tensor == nullptr,
            OPS_LOG_E(opName_, "key layout supported PA_BNSD or PA_NZ, input actual_seq_lengths_key must not be null"),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(opParamInfo_.blockTable.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input block_table data type only support int32"), return ge::GRAPH_FAILED);
    }
    OPS_ERR_IF(opParamInfo_.actualSeqLengths.tensor != nullptr &&
               opParamInfo_.actualSeqLengths.desc->GetDataType() != ge::DT_INT32,
                   OPS_LOG_E(opName_, "input actual_seq_lengths_key data type only support int32"),
                   return ge::GRAPH_FAILED);
    OPS_ERR_IF(kLayout_ != DataLayout::PA_NZ && kLayout_ != DataLayout::PA_BNSD && opParamInfo_.blockTable.tensor != nullptr,
                   OPS_LOG_E(opName_, "when key layout is not PA_BNSD or PA_NZ, input block_table must be null"),
                   return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::CheckShapeDim()
{
    OPS_ERR_IF((opParamInfo_.blockTable.tensor != nullptr) &&
                   (opParamInfo_.blockTable.tensor->GetStorageShape().GetDimNum() != DIM_NUM_TWO),
               OPS_LOG_E(opName_, "the dim num of block_table's shape should be 2"), return ge::GRAPH_FAILED);

    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();

    uint32_t qExpectShapeDim = DIM_NUM_THREE;
    uint32_t kExpectShapeDim = DIM_NUM_FOUR;
    if (kLayout_ == DataLayout::PA_NZ) {
        kExpectShapeDim = DIM_NUM_FIVE;
    }

    OPS_ERR_IF(kShapeDim != kExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of key's shape should be %u, but now is %u", kExpectShapeDim, kShapeDim),
               return ge::GRAPH_FAILED);
    OPS_ERR_IF(qShapeDim != qExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of query's shape should be %u, but now is %u",
                qExpectShapeDim, qShapeDim),
               return ge::GRAPH_FAILED);
             
    uint32_t sparseIndicesShapeDim = opParamInfo_.sparseIndices.shape->GetStorageShape().GetDimNum();  

    OPS_ERR_IF(sparseIndicesShapeDim != qExpectShapeDim,
               OPS_LOG_E(opName_, "the dim num of sparse_indices's shape should be %u, but now is %u",
                qExpectShapeDim, sparseIndicesShapeDim),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetActualSeqLenSize(uint32_t &size, const gert::Tensor *tensor,
                                                  const std::string &actualSeqLenName)
{
    size = static_cast<uint32_t>(tensor->GetShapeSize());
    if (size <= 0) {
        OPS_LOG_E(opName_, "%s's shape size is %u, it should be greater than 0.", actualSeqLenName.c_str(), size);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetAndCheckN2Size()
{
    uint32_t n2Index = DIM_IDX_TWO; // PA_BSND/BSND
    if (kLayout_ == DataLayout::PA_BNSD || kLayout_ == DataLayout::PA_NZ) { // PA_BNSD
        n2Index = DIM_IDX_ONE;
    }
    n2Size_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(n2Index));
    OPS_LOG_I(context_->GetNodeName(), "n2Size_ is %d", n2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetBatchSize()
{
    bSize_ = opParamInfo_.query.shape->GetStorageShape().GetDim(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetHeadDim()
{
    // 以query的D维度为基准
    headDim_ = opParamInfo_.query.shape->GetStorageShape().GetDim(DIM_IDX_TWO) * 8; // INT4伪装成IN32, D轴每8个数合成一个数，*8还原shape
    OPS_ERR_IF(headDim_ != 64,
               OPS_LOG_E(opName_, "the head dim of query and key should be %u, but now is %u",
                64, headDim_),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetSparseCount()
{
    sparseCount_ = opParamInfo_.sparseIndices.shape->GetStorageShape().GetDim(DIM_IDX_TWO);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetAndCheckBlockSize()
{
    if (kLayout_ == DataLayout::PA_BSND) {
        blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(1));
    } else if (kLayout_ == DataLayout::PA_BNSD) { // PA_BNSD
        blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(2));
    } else if (kLayout_ == DataLayout::PA_NZ) {
        blockSize_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(3));
    }
    OPS_LOG_I(context_->GetNodeName(), "blockSize_ is %d", blockSize_);

    OPS_ERR_IF(((blockSize_ % 16 != 0) || (blockSize_ == 0) || (blockSize_ > 1024)),
               OPS_LOG_E(opName_, "input key's block_size must be a multiple of 16 and belong to (0, 1024]."),
               return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::CheckBlockCount()
{
    int32_t blockCount_ = static_cast<uint32_t>(opParamInfo_.key.shape->GetStorageShape().GetDim(0));
    OPS_ERR_IF((blockCount_ == 0),
                OPS_LOG_E(opName_, "input key's block_count cannot be 0."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetS2SizeForPageAttention()
{
    if (GetAndCheckBlockSize() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckBlockCount() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    maxBlockNumPerBatch_ = opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(1);
    s2Size_ = maxBlockNumPerBatch_ * blockSize_;
    OPS_LOG_I(context_->GetNodeName(), "maxBlockNumPerBatch_ is %d, blockSize_ is %d, s2Size_ is %d",
              maxBlockNumPerBatch_, blockSize_, s2Size_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::GetS2Size()
{
    // 获取S2基准值
    // 1、BATCH_CONTINUOUS时, 从key的S轴获取
    // 3、PAGE_ATTENTION时, S2 = block_table.dim1 * block_size
    if (kLayout_ == DataLayout::PA_BNSD || kLayout_ == DataLayout::PA_BSND || kLayout_ == DataLayout::PA_NZ) {
        return GetS2SizeForPageAttention();
    } else if (kLayout_ == DataLayout::BSND) {
        s2Size_ = opParamInfo_.key.shape->GetStorageShape().GetDim(1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::ValidateInputShapesMatch()
{
    /*
    BSND:
    query [BatchSize,S1,N1,D],
    key [BlockNum,BlockSize,N2,D],
    block_table [BatchSize, BatchMaxBlockNum],
    act_seq_k [BatchSize]
    act_seq_q [BatchSize] 可选
    out [BatchSize,S1,N2,topk]
    */
    uint32_t outN2Dim = DIM_IDX_ONE;
        // -----------------------check BatchSize-------------------
        // bSize_ 来源于query
        OPS_ERR_IF(
                       ((opParamInfo_.blockTable.tensor != nullptr) &&
                       (opParamInfo_.blockTable.tensor->GetStorageShape().GetDim(0) != bSize_)) ||
                       (opParamInfo_.actualSeqLengths.tensor->GetShapeSize() != bSize_) ||
                       (opParamInfo_.sparseIndices.shape->GetStorageShape().GetDim(0) != bSize_),
                   OPS_LOG_E(opName_, "BSND case input query, actual_seq_lengths_key, block_table, sparse_indices dim 0 must be same."),
                   return ge::GRAPH_FAILED);

    // -----------------------check D-------------------
    uint32_t keyDDim = DIM_IDX_THREE;
    if (kLayout_ == DataLayout::PA_NZ) {
        keyDDim = DIM_IDX_TWO;
        OPS_ERR_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) * 8 * 8 != headDim_),
               OPS_LOG_E(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED); // INT4伪装成IN32, D轴每8个数合成一个数，*8还原shape
    } else {
        OPS_ERR_IF((opParamInfo_.key.shape->GetStorageShape().GetDim(keyDDim) * 8 != headDim_),
               OPS_LOG_E(opName_, "input query, key shape last dim must be same."), return ge::GRAPH_FAILED); // INT4伪装成IN32, D轴每8个数合成一个数，*8还原shape
    }
    // -----------------------check N2-------------------
    OPS_ERR_IF((opParamInfo_.sparseIndices.shape->GetStorageShape().GetDim(outN2Dim) != n2Size_),
               OPS_LOG_E(opName_, "input query and output sparse_indices shape n2 dim must be same."),
               return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QSIInfoParser::CheckScaleShape()
{
    uint32_t qShapeDim = opParamInfo_.query.shape->GetStorageShape().GetDimNum();
    uint32_t kShapeDim = opParamInfo_.key.shape->GetStorageShape().GetDimNum();
    uint32_t qDequantScaleShapeDim = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDimNum();
    uint32_t kDequantScaleShapeDim = opParamInfo_.key_dequant_scale.shape->GetStorageShape().GetDimNum();
    OPS_ERR_IF(qDequantScaleShapeDim != (qShapeDim - 1),
               OPS_LOG_E(opName_, "the dim num of query_dequant_scale's shape should be %u, but now is %u",
                         qShapeDim - 1, qDequantScaleShapeDim),
               return ge::GRAPH_FAILED);
    if (kLayout_ == DataLayout::PA_NZ) {
        OPS_ERR_IF(kDequantScaleShapeDim != (kShapeDim - 2),
               OPS_LOG_E(opName_, "the dim num of key_dequant_scale's shape should be %u, but now is %u", kShapeDim - 1,
                         kDequantScaleShapeDim),
               return ge::GRAPH_FAILED);
    } else {
        OPS_ERR_IF(kDequantScaleShapeDim != (kShapeDim - 1),
               OPS_LOG_E(opName_, "the dim num of key_dequant_scale's shape should be %u, but now is %u", kShapeDim - 1,
                         kDequantScaleShapeDim),
               return ge::GRAPH_FAILED);
    }
    // check q scale
    for (uint32_t i = 0; i < (qShapeDim - 2); i++) {
        uint32_t dimValueQueryScale = opParamInfo_.query_dequant_scale.shape->GetStorageShape().GetDim(i);
        uint32_t dimValueQuery = opParamInfo_.query.shape->GetStorageShape().GetDim(i);
        OPS_ERR_IF(dimValueQueryScale != dimValueQuery,
            OPS_LOG_E(opName_, "query_dequant_scale's shape[%u] %u and query's shape[%u] %u is not same", i,
                      dimValueQueryScale, i, dimValueQuery),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

void QSIInfoParser::GenerateInfo(QSITilingInfo &siInfo)
{
    siInfo.opName = opName_;
    siInfo.platformInfo = platformInfo_;
    siInfo.opParamInfo = opParamInfo_;
    siInfo.socVersion = socVersion_;

    siInfo.bSize = bSize_;
    siInfo.dSize = headDim_;
    siInfo.n2Size = n2Size_;
    siInfo.s2Size = s2Size_;
    siInfo.sparseCount = sparseCount_;
    siInfo.maxSeqlenKey = *opParamInfo_.maxSeqlenKey;
    siInfo.sparseBlockSize = *opParamInfo_.sparseBlockSize;
    siInfo.fixedTailCount = *opParamInfo_.fixedTailCount;
    siInfo.sparseRatio = (int)((1.0 - *opParamInfo_.sparseRatio) * 100.0 + 0.5) / 100.0;

    siInfo.inputQType = inputQType_;
    siInfo.inputKType = inputKType_;
    siInfo.outputType = sparseIndicesType_;

    siInfo.blockSize = blockSize_;
    siInfo.maxBlockNumPerBatch = maxBlockNumPerBatch_;

    std::string layOutKeyStr(opParamInfo_.layOutKey);
    siInfo.pageAttentionFlag = (layOutKeyStr == "PA_BNSD" || layOutKeyStr == "PA_BSND" || layOutKeyStr == "PA_NZ");

    siInfo.inputKLayout = kLayout_;
}

ge::graphStatus QSIInfoParser::ParseAndCheck(QSITilingInfo &siInfo)
{
    if (ge::GRAPH_SUCCESS != GetOpName() || ge::GRAPH_SUCCESS != GetNpuInfo() || ge::GRAPH_SUCCESS != GetOpParaInfo() ||
        ge::GRAPH_SUCCESS != CheckRequiredParaExistence()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetAndCheckInOutDataType() || ge::GRAPH_SUCCESS != GetQueryKeyAndOutLayout() ||
        ge::GRAPH_SUCCESS != GetAndCheckOptionalInput()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != CheckShapeDim() ||
        ge::GRAPH_SUCCESS != GetAndCheckN2Size()) {
        return ge::GRAPH_FAILED;
    }

    if (ge::GRAPH_SUCCESS != GetBatchSize() || ge::GRAPH_SUCCESS != GetHeadDim() ||
        ge::GRAPH_SUCCESS != GetS2Size() || ge::GRAPH_SUCCESS != GetSparseCount()) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != ValidateInputShapesMatch() || ge::GRAPH_SUCCESS != CheckScaleShape()) {
        return ge::GRAPH_FAILED;
    }

    GenerateInfo(siInfo);

    return ge::GRAPH_SUCCESS;
}

void QuantSalsIndexerTiling::SplitCoreBN(uint32_t coreNum, QSITilingInfo *siInfo, SplitParams splitParams) {
    std::vector<uint32_t> s1GBaseNum(siInfo->bSize);
    std::vector<uint32_t> s2BaseNum(siInfo->bSize);
    std::vector<uint32_t> s1Size(siInfo->bSize);
    std::vector<uint32_t> s2Size(siInfo->bSize);
    uint32_t s2BaseSize = 2048;
    // 计算总基本块数
    uint32_t totalBaseNum = 0;
    for (uint32_t bIdx = 0; bIdx < siInfo->bSize; bIdx++) {
        s2Size[bIdx] = siInfo->s2Size;
        s1Size[bIdx] = 1;
        s1GBaseNum[bIdx] = 1;
        s2BaseNum[bIdx] = 1;
        totalBaseNum += s2BaseNum[bIdx] * siInfo->n2Size;
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    }

    uint32_t accumBaseNum = 0; // 当前累计的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t currCoreIdx = 0;
    uint32_t lastValidBIdx = 0;
    // 分核流程，保存分核数据
    for (uint32_t bN2Idx = 0; bN2Idx < siInfo->bSize * siInfo->n2Size; bN2Idx++) {
        uint32_t bIdx = bN2Idx / siInfo->n2Size;
        for (uint32_t s1GIdx = 0; s1GIdx < s1GBaseNum[bIdx]; s1GIdx++) {
            uint32_t sInnerIndexStart = 0;
            uint32_t sInnerIndexEnd = s2BaseNum[bIdx];
            accumBaseNum = accumBaseNum + (sInnerIndexEnd - sInnerIndexStart);
            targetBaseNum = (currCoreIdx + 1) * avgBaseNum;
            if (accumBaseNum >= targetBaseNum) {
                // 更新当前核的End分核信息
                splitParams.bN2End[currCoreIdx] = bN2Idx;
                splitParams.gS1End[currCoreIdx] = s1GIdx;
                splitParams.s2End[currCoreIdx] = sInnerIndexEnd - 1;
                currCoreIdx += 1;
            }
        }
        if ((s1GBaseNum[bIdx] > 0) && (s2BaseNum[bIdx] > 0)) {
            lastValidBIdx = bIdx;
        }
    }
    if (accumBaseNum < targetBaseNum) {
        // 更新最后一个核的End分核信息
        splitParams.bN2End[currCoreIdx] = ((lastValidBIdx + 1) * (siInfo->n2Size)) - 1;
        splitParams.gS1End[currCoreIdx] = s1GBaseNum[lastValidBIdx] - 1;
        splitParams.s2End[currCoreIdx] = s2BaseNum[lastValidBIdx] - 1;
        currCoreIdx += 1;
    }
}

// --------------------------TilingPrepare函数定义-------------------------------------
static ge::graphStatus TilingPrepareForQuantSalsIndexer(gert::TilingParseContext * /* context */)
{
    return ge::GRAPH_SUCCESS;
}

// --------------------------QuantSalsIndexerTiling类成员函数定义-----------------------
ge::graphStatus QuantSalsIndexerTiling::DoTiling(QSITilingInfo *tilingInfo)
{
    // -------------set blockdim-----------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingInfo->platformInfo);
    uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);

    // ------------------------------------------------------------
    if (context_->GetOptionalInputDesc(METADATA_INDEX) == nullptr) {
        SplitParams splitParams;
        splitParams.bN2End = tilingData_.splitParams.get_bN2End();
        splitParams.gS1End = tilingData_.splitParams.get_gS1End();
        splitParams.s2End = tilingData_.splitParams.get_s2End();
        SplitCoreBN(aicNum, tilingInfo, splitParams);
        tilingData_.set_usedCoreNum(blockDim);
    }
    
    // ------------------------------------------------------------


    // -------------set workspacesize-----------------
    constexpr uint32_t MM1_RES_ELEM_SIZE = 4;         // 4: fp32
    constexpr uint32_t DOUBLE_BUFFER = 2;             // 双Buffer
    constexpr uint32_t M_BASE_SIZE = 512;             // m轴基本块大小
    constexpr uint32_t S2_BASE_SIZE = 512;            // S2轴基本块大小
    constexpr uint32_t V1_RES_ELEM_SIZE = 4;          // 4: int32
    constexpr uint32_t V1_RES_ELEM_TYPE = 2;          // 保留Index和Value 2种数据
    constexpr uint32_t V1_DECODE_PARAM_ELEM_SIZE = 8; // 8: int64
    constexpr uint32_t V1_DECODE_PARAM_NUM = 16;      // Decode参数个数
    constexpr uint32_t V1_DECODE_DATA_NUM = 2;        // Decode每个核需要存储头和尾部两块数据
    constexpr uint32_t S1_BASE_SIZE = 8;              // S1轴基本块的大小
    constexpr uint32_t TOPK_MAX_SIZE = 2048;          // TopK选取个数
    uint32_t workspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    // 主流程需Workspace大小
    uint32_t mm1ResSize = M_BASE_SIZE * S2_BASE_SIZE;
    workspaceSize += mm1ResSize * MM1_RES_ELEM_SIZE * DOUBLE_BUFFER * aicNum;
    // Decode流程(LD)需要Workspace大小
    // 临时存储Decode中间结果大小: 2(头/尾)*8(s1Base)*2(idx/value)*2048(K)*sizeof(int32)*24=6M
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_RES_ELEM_TYPE * (TOPK_MAX_SIZE + tilingInfo->fixedTailCount) * V1_RES_ELEM_SIZE * aicNum;
    // 临时存储Decode中间参数信息大小: 2(头/尾)*8(s1Base)*16(paramNum)*sizeof(int64_t)*24=48k
    workspaceSize += V1_DECODE_DATA_NUM * S1_BASE_SIZE * V1_DECODE_PARAM_NUM * V1_DECODE_PARAM_ELEM_SIZE * aicNum;
    size_t *workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize;

    // -------------set tilingdata-----------------
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_n2Size(tilingInfo->n2Size);
    tilingData_.set_fixedTailCount(tilingInfo->fixedTailCount);
    tilingData_.set_maxSeqlenKey(tilingInfo->maxSeqlenKey);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_sparseBlockSize(tilingInfo->sparseBlockSize);
    tilingData_.set_sparseRatio(tilingInfo->sparseRatio);
    tilingData_.set_dSize(tilingInfo->dSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    // -------------set tilingkey-----------------
    // DT_Q, DT_KV, DT_OUT, PAGE_ATTENTION, FLASH_DECODE, LAYOUT_T, KV_LAYOUT_T
    uint32_t inputQType = static_cast<uint32_t>(tilingInfo->inputQType);
    uint32_t inputKType = static_cast<uint32_t>(tilingInfo->inputKType);
    uint32_t outputType = static_cast<uint32_t>(tilingInfo->outputType);
    uint32_t pageAttentionFlag = static_cast<uint32_t>(tilingInfo->pageAttentionFlag);
    uint32_t inputKLayout = static_cast<uint32_t>(tilingInfo->inputKLayout);
    uint32_t tilingKey =
        GET_TPL_TILING_KEY(inputQType, inputKType, outputType, pageAttentionFlag, inputKLayout);
    context_->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

// --------------------------Tiling函数定义---------------------------
ge::graphStatus TilingForQuantSalsIndexer(gert::TilingContext *context)
{
    OPS_ERR_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("SalsIndexer", "Tiling context is null."),
               return ge::GRAPH_FAILED);
    QSITilingInfo siInfo;
    QSIInfoParser QSIInfoParser(context);
    if (QSIInfoParser.ParseAndCheck(siInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    QuantSalsIndexerTiling siTiling(context);
    return siTiling.DoTiling(&siInfo);
}

// --------------------------Tiling函数及TilingPrepare函数注册--------
IMPL_OP_OPTILING(QuantSalsIndexer)
    .Tiling(TilingForQuantSalsIndexer)
    .TilingParse<QSICompileInfo>(TilingPrepareForQuantSalsIndexer);

} // namespace optiling
