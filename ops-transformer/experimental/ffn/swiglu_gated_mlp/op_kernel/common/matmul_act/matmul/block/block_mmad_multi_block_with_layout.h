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
 * \file block_mmad_multi_block_with_layout.h
 * \brief
 */

#ifndef MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_H
#define MATMUL_BLOCK_BLOCK_MMAD_MULTI_BLOCK_WITH_LAYOUT_H

#include "lib/matmul/matmul.h"
#include "lib/matmul/tiling.h"
#include "lib/matmul/constant_tiling.h"
#include "../policy/dispatch_policy.h"
#include "./matmul_impl_traits.h"
#include "./block_mmad_with_layout.h"

namespace Act {
namespace Gemm {
namespace Block {
template <class L1TileShape, class L0TileShape, class AT, class BT, class CT, class BiasT, class TileCopy>
class BlockMmad<MatmulMultiBlockWithLayout<>, L1TileShape, L0TileShape, AT, BT, CT, BiasT, TileCopy,
    AscendC::Std::enable_if_t<IsMatmulLayoutTypeV<AT>>>
    : public BlockMmad<MatmulMultiBlockWithLayout<>, L1TileShape, L0TileShape,
        ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy> {
    using Base = BlockMmad<MatmulMultiBlockWithLayout<>, L1TileShape, L0TileShape,
                           ToMatmulTypeT<AT>, ToMatmulTypeT<BT>, ToMatmulTypeT<CT>, ToMatmulTypeT<BiasT>, TileCopy>;
    using Base::Base;
};

/**
* @class BlockMmad
* @brief A template class for handling matrix multiplication across multiple blocks
*
* This class is specialized base on MatmulMultiBlockWithLayout<>
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType, class TileCopy>
class BlockMmad<MatmulMultiBlockWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy,
    AscendC::Std::enable_if_t<
        !AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitMWithParams>> &&
        !AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitNWithParams>> &&
        !IsMatmulLayoutTypeV<AType>
    >> : public BlockMmadWithLayout<
        BlockMmad<MatmulMultiBlockWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>,
        MatmulMultiBlockWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy
    > {
public:
    using DispatchPolicy = MatmulMultiBlockWithLayout<>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base = BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    friend class BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    static_assert(AscendC::PhyPosIsGM(AType::pos) && AscendC::PhyPosIsGM(BType::pos), "Only support GM input");
    static_assert(
        IsF16OrBf16AB<AType, BType, CType>() || IsI8I8I32<AType, BType, CType>() || IsF32F32F32<AType, BType, CType>(),
        "Unsupported dtype"
    );
    static_assert(IsNDOrAlign<AType>() && IsNDOrAlign<CType>(), "Only support ND format");

private:
    MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType> matmul_;
};

/**
* @class BlockMmad
* @brief A template class for handling matrix multiplication across multiple blocks
*
* This class is specialized base on MatmulMultiBlockWithLayout<>, specifically for splitM/spltN scenarios
*/
template <class L1Shape, class L0Shape, class AType, class BType, class CType, class BiasType, class TileCopy>
class BlockMmad<MatmulMultiBlockWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy,
    AscendC::Std::enable_if_t<
        (AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitMWithParams>> ||
        AscendC::Std::is_same_v<TileCopy, Tile::TileCopy<Arch::Ascend910_95, Tile::CopyOutSplitNWithParams>>) &&
        !IsMatmulLayoutTypeV<AType>
    >> : public BlockMmadWithLayout<
        BlockMmad<MatmulMultiBlockWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>,
        MatmulMultiBlockWithLayout<>, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy
    > {
public:
    using DispatchPolicy = MatmulMultiBlockWithLayout<>;
    using Self = BlockMmad<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    using Base = BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;
    friend class BlockMmadWithLayout<Self, DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy>;

    static_assert(
        AscendC::PhyPosIsGM(AType::pos) && AscendC::PhyPosIsGM(BType::pos) && AscendC::PhyPosIsUB(CType::pos),
        "Only GM input and UB output support splitM or splitN"
    );
    static_assert(IsF16F16F32<AType, BType, CType>() || IsBf16Bf16F32<AType, BType, CType>() ||
                      IsF32F32F32<AType, BType, CType>(),
                  "Unsupported dtype");
    static_assert(IsNDOrAlign<AType>() && IsNDOrAlign<CType>(), "Only support ND format");

private:
    MatmulImplTraitsT<DispatchPolicy, L1Shape, L0Shape, AType, BType, CType, BiasType, TileCopy> matmul_;
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif