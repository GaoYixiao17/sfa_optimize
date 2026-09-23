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
 * \file matmul_layout_type.h
 * \brief
 */

#ifndef UTILS_MATMUL_LAYOUT_TYPE_H
#define UTILS_MATMUL_LAYOUT_TYPE_H

#include "./integral_constant.h"
#include "./layout_utils.h"

namespace Act {
namespace Gemm {

template <typename LAYOUT, typename TYPE, AscendC::TPosition POSITION>
struct MatmulLayoutType {
    using layout = LAYOUT;
    using T = TYPE;
    constexpr static AscendC::TPosition pos = POSITION;
};

template <typename T, typename = void>
struct IsMatmulLayoutType : AscendC::Std::false_type {};

template <typename T>
struct IsMatmulLayoutType<T, std::void_t<typename T::layout, typename T::T, decltype(T::pos)>>
    : AscendC::Std::true_type {};

template <typename T>
constexpr bool IsMatmulLayoutTypeV = IsMatmulLayoutType<T>::value;

template <class LayoutT>
struct ToMatmulType {
    using Type = AscendC::MatmulType<LayoutT::pos, TagToFormat<typename LayoutT::layout>::format,
            typename LayoutT::T, TagToTrans<typename LayoutT::layout>::value>;
};

template <class LayoutT>
using ToMatmulTypeT = typename ToMatmulType<LayoutT>::Type;

} // namespace Gemm
} // namespace Act
#endif
