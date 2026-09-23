/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file split_core_BN.h
 * \brief
 */

#ifndef SPLIT_CORE_BN_H
#define SPLIT_CORE_BN_H


namespace optiling {

struct BaseInfo {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size = 0;
    uint32_t s2Size = 0;
    bool isAccumSeqS1 = false;
    bool isAccumSeqS2 = false;
    bool sliding = false;
    const int64_t *actualSeqS1Size = nullptr;
    const int64_t *actualSeqS2Size = nullptr;
    uint32_t actualLenDimsQ = 0;
    uint32_t actualLenDimsKV = 0;
    int64_t preToken = 0;
    int64_t nextToken = 0;
};

struct InnerSplitParams {
    uint32_t s1GBaseSize = 1;
    uint32_t s2BaseSize = 1;
};

struct OuterSplitParams {
    uint32_t *bN2End;
    uint32_t *gS1End;
    uint32_t *s2End;
};

struct FlashDecodeParams {
    uint32_t *bN2IdxOfFdHead;
    uint32_t *gS1IdxOfFdHead;
    uint32_t *s2SplitNumOfFdHead;
    uint32_t *s2SplitStartIdxOfCore;
    uint32_t gS1BaseSizeOfFd;
    uint32_t *gS1SplitNumOfFdHead;
    uint32_t *gS1LastPartSizeOfFdHead;
    uint32_t *gS1IdxEndOfFdHead;
    uint32_t *gS1IdxEndOfFdHeadSplit;
};

struct SplitCoreRes {
    uint32_t numOfFdHead;
    uint32_t maxS2SplitNum;
    uint32_t usedCoreNum;
    uint32_t usedVecNumOfFd;
};

int64_t SfaClipSInnerTokenCube(int64_t sInnerToken, int64_t minValue, int64_t maxValue)
{
    sInnerToken = sInnerToken > minValue ? sInnerToken : minValue;
    sInnerToken = sInnerToken < maxValue ? sInnerToken : maxValue;
    return sInnerToken;
}
 
 
void SfaUpDateInnerLoop(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t &s2Start, uint32_t &s2End,
    uint32_t sOuterLoopIdx, int64_t curActualSeqLen,int64_t preTokensLeftUp, int64_t nextTokensLeftUp, uint32_t s2Loop){
    if (!baseInfo.sliding) {
        return;
    }
    uint32_t sOuterSize = innerSplitParams.s1GBaseSize / baseInfo.gSize;  // BNSD需要额外适配
    uint32_t sOuterOffset = sOuterLoopIdx * sOuterSize;
    int64_t sInnerFirstToken = SfaClipSInnerTokenCube(static_cast<int64_t>(sOuterOffset) - preTokensLeftUp,
    0, curActualSeqLen);
    int64_t s2StartIdx = sInnerFirstToken / static_cast<int64_t>(innerSplitParams.s2BaseSize);
        if (s2StartIdx <= 0) {
        s2StartIdx = 0;
    }
    s2Start = s2StartIdx;
    int64_t sInnerLastToken = SfaClipSInnerTokenCube(static_cast<int64_t>(sOuterOffset) + nextTokensLeftUp +
        static_cast<int64_t>(sOuterSize), 0, curActualSeqLen);
    s2End= (sInnerLastToken + static_cast<int64_t>(innerSplitParams.s2BaseSize) - 1) /
        static_cast<int64_t>(innerSplitParams.s2BaseSize);
    if (s2End > s2Loop) {
        s2End = s2Loop;
    }  
}
 
void SfaGetPreNextTokensLeftUp(const BaseInfo &baseInfo,
    int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) {
    if (baseInfo.sliding) {
        preTokensLeftUp = baseInfo.preToken - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = baseInfo.nextToken + actualSeqLengthKV - actualSeqLength;
    }
}
 
uint32_t SfaGetCalcBlockNumsOneHead(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, int64_t outerBlockNums, int64_t innerBlockNums,
  int64_t actualSeqLength, int64_t actualSeqLengthKV, int64_t preTokensLeftUp, int64_t nextTokensLeftUp) {
    if (!baseInfo.sliding) { // 会不会影响原来
      return innerBlockNums * outerBlockNums;
    } else {
      uint32_t toCalcBlockNums = 0;
        for (uint32_t s1OuterIdx = 0; s1OuterIdx <  outerBlockNums; s1OuterIdx++) {
            uint32_t sInnerIndexStart = 0;
            uint32_t sInnerIndexEnd = 0;
            SfaUpDateInnerLoop(baseInfo, innerSplitParams, sInnerIndexStart, sInnerIndexEnd,
                s1OuterIdx, actualSeqLengthKV,preTokensLeftUp, nextTokensLeftUp, innerBlockNums);
            toCalcBlockNums += (sInnerIndexEnd - sInnerIndexStart);
        }
      return toCalcBlockNums;
    }
}
 
void SfaGetActualSeqLength(const BaseInfo &baseInfo, uint32_t &actualSeqLengths, uint32_t &actualSeqLengthsKV, uint32_t bIdx) {
        actualSeqLengths = baseInfo.s1Size;
        actualSeqLengthsKV = baseInfo.s2Size;
}

void SfaSplitCore(const BaseInfo &baseInfo, const InnerSplitParams &innerSplitParams, uint32_t coreNum, OuterSplitParams outerSplitParams) {
    std::vector<uint32_t> s1GBaseNum(baseInfo.bSize);       // S1G方向，切了多少个基本块
    std::vector<uint32_t> s2BaseNum(baseInfo.bSize);        // S2方向，切了多少个基本块
    std::vector<uint32_t> s1Size(baseInfo.bSize);
    std::vector<uint32_t> s2Size(baseInfo.bSize);
    bool seqZeroFlag = true;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        SfaGetActualSeqLength(baseInfo, s1Size[bIdx], s2Size[bIdx], bIdx);
        if (s1Size[bIdx] > 0) {
           seqZeroFlag = false;
        }
    }
    // 计算总基本块数
    uint32_t totalBaseNum = 0;
    for (uint32_t bIdx = 0; bIdx < baseInfo.bSize; bIdx++) {
        //SfaGetActualSeqLength(baseInfo,s1Size[bIdx], s2Size[bIdx], bIdx);
        if (seqZeroFlag == false) {
            s1GBaseNum[bIdx] = (s1Size[bIdx]* baseInfo.gSize  + (innerSplitParams.s1GBaseSize - 1)) / innerSplitParams.s1GBaseSize;
        } else {
            s1GBaseNum[bIdx] = 1;
        }
        
        s2BaseNum[bIdx] = 1;
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        SfaGetPreNextTokensLeftUp(baseInfo, s1Size[bIdx], s2Size[bIdx], preTokensLeftUp, nextTokensLeftUp);
        totalBaseNum += SfaGetCalcBlockNumsOneHead(baseInfo,innerSplitParams,s1GBaseNum[bIdx], s2BaseNum[bIdx],
            s1Size[bIdx], s2Size[bIdx], preTokensLeftUp, nextTokensLeftUp) * baseInfo.n2Size; 
    }
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = (totalBaseNum + coreNum - 1) / coreNum;
    }

    uint32_t accumBaseNum = 0;       // 当前累积的基本块数
    uint32_t targetBaseNum = 0;
    uint32_t currCoreIdx = 0;
    uint32_t lastValidBIdx = 0;
    // res.numOfFdHead = 0;
    // res.maxS2SplitNum = 1;
    // fDParams.s2SplitStartIdxOfCore[0] = 0; //每核头块所处当前线段被切的第几部分
    //分核流程，保存分核数据
    for (uint32_t bN2Idx = 0; bN2Idx < baseInfo.bSize * baseInfo.n2Size; bN2Idx++) { 
        uint32_t bIdx = bN2Idx / baseInfo.n2Size;
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        SfaGetPreNextTokensLeftUp(baseInfo, s1Size[bIdx], s2Size[bIdx], preTokensLeftUp, nextTokensLeftUp); 
        for (uint32_t s1GIdx = 0; s1GIdx < s1GBaseNum[bIdx]; s1GIdx++) {
            uint32_t sInnerIndexStart = 0;
            uint32_t sInnerIndexEnd = s2BaseNum[bIdx];
            SfaUpDateInnerLoop(baseInfo, innerSplitParams,sInnerIndexStart, sInnerIndexEnd,
                s1GIdx, s2Size[bIdx],preTokensLeftUp, nextTokensLeftUp, s2BaseNum[bIdx]);
            uint32_t currKvSplitPart = 1;           // [B,N2,S1]确定后，S2被切了几份

            for (uint32_t s2Idx = sInnerIndexStart; s2Idx < sInnerIndexEnd; s2Idx++) {
                accumBaseNum += 1;
                targetBaseNum = (currCoreIdx + 1) * avgBaseNum;         // 计算当前的目标权重
                if (accumBaseNum >= targetBaseNum) {
                    // 更新当前核的End分核信息
                    outerSplitParams.bN2End[currCoreIdx] = bN2Idx;
                    outerSplitParams.gS1End[currCoreIdx] = s1GIdx;
                    outerSplitParams.s2End[currCoreIdx] = s2Idx;
                    currCoreIdx += 1;
                }
            }
        }
        if ((s1GBaseNum[bIdx] > 0) && (s2BaseNum[bIdx] > 0)) {
            lastValidBIdx = bIdx;
        }
    }
    if (accumBaseNum < targetBaseNum) {
        // 更新最后一个核的End分核信息
        outerSplitParams.bN2End[currCoreIdx] = ((lastValidBIdx + 1) * (baseInfo.n2Size)) - 1;
        outerSplitParams.gS1End[currCoreIdx] = s1GBaseNum[lastValidBIdx] - 1;
        outerSplitParams.s2End[currCoreIdx] = s2BaseNum[lastValidBIdx]  - 1;
        currCoreIdx += 1;
    }
    // res.usedCoreNum = currCoreIdx;
}

void SfaSplitFD(SplitCoreRes &res, FlashDecodeParams fDParams, uint32_t coreNum)
{ 
    uint64_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i <  res.numOfFdHead; i++) {
        totalFDLoad += fDParams.s2SplitNumOfFdHead[i] * fDParams.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += fDParams.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, coreNum * 2);  // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);  // 初始化vector的负载上限
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i <  res.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = fDParams.s2SplitNumOfFdHead[i]; // gs上多少次需要规约的次数
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < fDParams.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - loadOfCurVector;  // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * 2) {
                fDParams.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;  // 记录寄几个上
                fDParams.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit; // 记录有gs/8的end
                curCoreIndex += 1;
                totalFDLoad -= loadOfCurVector;  // 当前未分配的总负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);  // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    fDParams.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    fDParams.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    res.usedVecNumOfFd = curCoreIndex + 1;
}

}
#endif
