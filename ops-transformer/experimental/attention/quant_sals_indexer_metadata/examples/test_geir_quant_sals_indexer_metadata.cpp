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
#include "../op_graph/quant_sals_indexer_metadata_proto.h"
#include "../../quant_sals_indexer/op_kernel/quant_sals_indexer_metadata.h"

using namespace ge;

static const uint32_t batchSize = 3;
static const uint32_t kvSeqSize = 10240;
static const uint32_t kvHeadNum = 4;
static const uint32_t sparseBlockSize = 16;
static const uint32_t fixedTailCount = 16;
static const uint32_t aicCoreNum = 24;
static const uint32_t aivCoreNum = 48;

static const std::vector<int32_t> actSeqLenKV = {10240, 10240, 10240};
static const std::vector<int64_t> actSeqLenKVShape = {batchSize};
static const std::vector<int64_t> metadataShape = {optiling::QSI_META_SIZE};
static const std::string dumpFile = "./dump";

static const bool enableActLenKV = true;

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
        return -1;
    }

    Graph graph("GraphQuantSalsIndexerMetadata");

    auto metaDataOp = op::QuantSalsIndexerMetadata(
        "QuantSalsIndexerMetadata-0");

    // gen graph
    auto dataOp0 = op::Data("input0").set_attr_index(0); // Data 算子
    if (enableActLenKV) {
        TensorDesc desc(ge::Shape(actSeqLenKVShape), FORMAT_ND, DT_INT32);
        desc.SetPlacement(ge::kPlacementHost);
        desc.SetFormat(FORMAT_ND);
        desc.SetRealDimCnt(actSeqLenKVShape.size());
        dataOp0.update_input_desc_x(desc);
        graph.AddOp(dataOp0);
        metaDataOp.set_input_actual_seq_lengths_key(dataOp0);
    } 

    metaDataOp.update_output_desc_metadata(TensorDesc{ge::Shape(metadataShape), FORMAT_ND, DT_INT32});
    metaDataOp.set_attr_batch_size(batchSize);
    metaDataOp.set_attr_key_seq_size(kvSeqSize);
    metaDataOp.set_attr_key_head_num(kvHeadNum);
    metaDataOp.set_attr_sparse_block_size(sparseBlockSize);
    metaDataOp.set_attr_fixed_tail_count(fixedTailCount);
    metaDataOp.set_attr_aic_core_num(aicCoreNum);
    metaDataOp.set_attr_aiv_core_num(aivCoreNum);
    graph.AddOp(metaDataOp);

    // run Graph
    std::vector<ge::Operator> inputOps = {dataOp0};
    std::vector<ge::Operator> outputOps = {metaDataOp};
    graph.SetInputs(inputOps).SetOutputs(outputOps);

    aclgrphDumpGraph(graph, dumpFile.c_str(), dumpFile.length());

    std::vector<ge::Tensor> inputTensors;
    std::vector<ge::Tensor> outputTensors;

    if (enableActLenKV) {
        inputTensors.push_back(
            Tensor{dataOp0.get_input_desc_x(),
                reinterpret_cast<const uint8_t *>(&actSeqLenKV[0]),
                actSeqLenKV.size() * sizeof(actSeqLenKV[0])});
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
    
    optiling::detail::QsiMetaData *metaDataPtr = (optiling::detail::QsiMetaData*)data;
    printf("usedCoreNum: %d \n", metaDataPtr->usedCoreNum);
    for (uint32_t i = 0; i < metaDataPtr->usedCoreNum; i ++) {
        printf("bN2End[%d]: %d \n", i, metaDataPtr->bN2End[i]);
        printf("gS1End[%d]: %d \n", i, metaDataPtr->gS1End[i]);
        printf("s2End[%d]: %d \n", i, metaDataPtr->s2End[i]);
    }

  return 0;
}