/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>
#include <string>
#include <map>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "array_ops.h"  // REG_OP(Data)
#include "ge_ir_build.h"

// #include "experiment_ops.h"
#include "nn_other.h"
#include "../op_graph/sparse_flash_attention_antiquant_metadata_proto.h"
#include "../../sparse_flash_attention_antiquant/op_kernel/sparse_flash_attention_antiquant_metadata.h"

using namespace ge;

static const uint32_t batchSize = 4;
static const uint32_t querySeqSize = 3;
static const uint32_t queryHeadNum = 10;
static const uint32_t kvSeqSize = 10240;
static const uint32_t kvHeadNum = 1;
static const uint32_t headDim = 128;
static const uint32_t topKSize = 128;
static const uint32_t sparseBlockSize = 16;
static const std::string layoutQuery = "BSND";
static const std::string layoutKV = "BSND";
static const uint32_t sparseMode = 0;
static const uint32_t attentionMode = 0;
static const uint32_t ropeHeadDim = 64;
static const uint32_t sparseSharedSize = 3;
static const uint32_t aicCoreNum = 24;
static const uint32_t aivCoreNum = 48;

static const std::vector<int32_t> actSeqLenQuery = {3, 6, 9, 12};
static const std::vector<int32_t> actSeqLenKV = {10240, 10240, 10240, 10240};
static const std::vector<int32_t> sparseSeqLenKV = {2560, 2560, 2560, 2560};
static const std::vector<int64_t> actSeqLenQShape = {batchSize};
static const std::vector<int64_t> actSeqLenKVShape = {batchSize};
static const std::vector<int64_t> sparseSeqLenKVShape = {batchSize};
static const std::vector<int64_t> metadataShape = {optiling::SFA_META_SIZE};
static const std::string dumpFile = "./dump";

static const bool enableActLenQuery = true;
static const bool enableActLenKV = true;
static const bool enablesparseSeqLenKV = true;

using namespace ge;

class GeEnv {
public:
    GeEnv() {
        std::map<AscendString, AscendString> opt = {{"ge.exec.deviceId", "0"},
                                                    {"ge.graphRunMode", "1"}};
        inited_ = GEInitialize(opt) == SUCCESS;
    }
    ~GeEnv() {
        if (inited_)
        GEFinalize();
    }

    bool Ok() { return inited_; }

private:
    bool inited_;
};

int main(int argc, char **argv) {
    GeEnv geEnv;
    if (!geEnv.Ok()) {
        printf("GEInitialize fail\n");
        return -1;
    }

    Graph graph("GraphKvQuantSparseFlashAttentionMetadata");

    auto metaDataOp = op::KvQuantSparseFlashAttentionMetadata(
        "KvQuantSparseFlashAttentionMetadata-0");

    // gen graph
    auto dataOp0 = op::Data("input0").set_attr_index(0); // Data 算子
    if (enableActLenQuery) {
        TensorDesc desc(ge::Shape(actSeqLenQShape), FORMAT_ND, DT_INT32);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        desc.SetRealDimCnt(actSeqLenQShape.size());
        dataOp0.update_input_desc_x(desc);
        graph.AddOp(dataOp0);
        metaDataOp.set_input_actual_seq_lengths_query(dataOp0);
    }

    auto dataOp1 = op::Data("input1").set_attr_index(0); // Data 算子
    if (enableActLenKV) {
        TensorDesc desc(ge::Shape(actSeqLenKVShape), FORMAT_ND, DT_INT32);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        desc.SetRealDimCnt(actSeqLenKVShape.size());
        dataOp1.update_input_desc_x(desc);
        graph.AddOp(dataOp1);
        metaDataOp.set_input_actual_seq_lengths_kv(dataOp1);
    } 

    auto dataOp2 = op::Data("input2").set_attr_index(0); // Data 算子
    if (enablesparseSeqLenKV) {
        TensorDesc desc(ge::Shape(actSeqLenKVShape), FORMAT_ND, DT_INT32);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        desc.SetRealDimCnt(sparseSeqLenKV.size());
        dataOp2.update_input_desc_x(desc);
        graph.AddOp(dataOp2);
        metaDataOp.set_input_sparse_seq_lengths_kv(dataOp2);
    }

    metaDataOp.update_output_desc_metadata(TensorDesc{ge::Shape(metadataShape), FORMAT_ND, DT_INT32});
    metaDataOp.set_attr_batch_size(batchSize);
    metaDataOp.set_attr_query_seq_size(querySeqSize);
    metaDataOp.set_attr_query_head_num(queryHeadNum);
    metaDataOp.set_attr_kv_seq_size(kvSeqSize);
    metaDataOp.set_attr_kv_head_num(kvHeadNum);
    metaDataOp.set_attr_head_dim(headDim);
    metaDataOp.set_attr_topk_size(topKSize);
    metaDataOp.set_attr_sparse_block_size(sparseBlockSize);
    metaDataOp.set_attr_aic_core_num(aicCoreNum);
    metaDataOp.set_attr_aiv_core_num(aivCoreNum);
    metaDataOp.set_attr_layout_query(layoutQuery);
    metaDataOp.set_attr_layout_kv(layoutKV);
    metaDataOp.set_attr_sparse_mode(sparseMode);
    metaDataOp.set_attr_attention_mode(attentionMode);
    metaDataOp.set_attr_rope_head_dim(ropeHeadDim);
    metaDataOp.set_attr_sparse_shared_size(sparseSharedSize);
    graph.AddOp(metaDataOp);

    // run Graph
    std::vector<ge::Operator> inputOps = {dataOp0, dataOp1, dataOp2};
    std::vector<ge::Operator> outputOps = {metaDataOp};
    graph.SetInputs(inputOps).SetOutputs(outputOps);

    aclgrphDumpGraph(graph, dumpFile.c_str(), dumpFile.length());

    std::vector<ge::Tensor> inputTensors;
    std::vector<ge::Tensor> outputTensors;

    if (enableActLenQuery) {
        inputTensors.push_back(
            Tensor{dataOp0.get_input_desc_x(),
                reinterpret_cast<const uint8_t *>(&actSeqLenQuery[0]),
                actSeqLenQuery.size() * sizeof(actSeqLenQuery[0])});
    }

    if (enableActLenKV) {
        inputTensors.push_back(
            Tensor{dataOp1.get_input_desc_x(),
                reinterpret_cast<const uint8_t *>(&actSeqLenKV[0]),
                actSeqLenKV.size() * sizeof(actSeqLenKV[0])});
    }

    if (enablesparseSeqLenKV) {
        inputTensors.push_back(
            Tensor{dataOp2.get_input_desc_x(),
                reinterpret_cast<const uint8_t *>(&sparseSeqLenKV[0]),
                sparseSeqLenKV.size() * sizeof(sparseSeqLenKV[0])});
    }

    std::map<AscendString, AscendString> build_options;
    auto session = std::make_shared<Session>(build_options);
    std::map<AscendString, AscendString> graph_options;
    uint32_t graph_id = 0;
    session->AddGraph(graph_id, graph, graph_options);
    if (session->RunGraph(graph_id, inputTensors, outputTensors)) {
        printf("RunGraph Fail\n");
        return -1;
    }

    auto tensor = outputTensors[0];
    auto data = tensor.GetData();
    auto dataSize = tensor.GetTensorDesc().GetShape().GetShapeSize();
    
    optiling::detail::SfaMetaData *metaDataPtr = (optiling::detail::SfaMetaData*)data;
    printf("mBaseSize: %d \n", metaDataPtr->mBaseSize);
    printf("s2BaseSize: %d \n", metaDataPtr->s2BaseSize);
    printf("gS1BaseSizeOfFd: %d \n", metaDataPtr->gS1BaseSizeOfFd);
    printf("usedCoreNum: %d \n", metaDataPtr->usedCoreNum);
    printf("numOfFdHead: %d \n", metaDataPtr->numOfFdHead);
    printf("usedVecNumOfFd: %d \n", metaDataPtr->usedVecNumOfFd);
    for (uint32_t i = 0; i < metaDataPtr->usedCoreNum; i ++) {
        printf("bN2End[%d]: %d \n", i, metaDataPtr->bN2End[i]);
        printf("gS1End[%d]: %d \n", i, metaDataPtr->gS1End[i]);
        printf("s2End[%d]: %d \n", i, metaDataPtr->s2End[i]);
        printf("s2SplitStartIdxOfCore[%d]: %d \n", i, metaDataPtr->fdRes.s2SplitStartIdxOfCore[i]);
    }

    for (uint32_t i = 0; i < metaDataPtr->numOfFdHead; i ++) {
        printf("bN2IdxOfFdHead[%d]: %d \n", i, metaDataPtr->fdRes.bN2IdxOfFdHead[i]);
        printf("gS1IdxOfFdHead[%d]: %d \n", i, metaDataPtr->fdRes.gS1IdxOfFdHead[i]);
        printf("s2SplitNumOfFdHead[%d]: %d \n", i, metaDataPtr->fdRes.s2SplitNumOfFdHead[i]);
        printf("gS1SplitNumOfFdHead[%d]: %d \n", i, metaDataPtr->fdRes.gS1SplitNumOfFdHead[i]);
        printf("gS1LastPartSizeOfFdHead[%d]: %d \n", i, metaDataPtr->fdRes.gS1LastPartSizeOfFdHead[i]);
    }

    for (uint32_t i = 0; i < metaDataPtr->usedVecNumOfFd; i ++) {
        printf("gS1IdxEndOfFdHead[%d]: %d \n", i, metaDataPtr->fdRes.gS1IdxEndOfFdHead[i]);
        printf("gS1IdxEndOfFdHeadSplit[%d]: %d \n", i, metaDataPtr->fdRes.gS1IdxEndOfFdHeadSplit[i]);
    }

  return 0;
}