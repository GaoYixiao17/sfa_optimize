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
 * \file tuple_utils.h
 * \brief
 */
#ifndef UTILS_TUPLE_UTILS_H
#define UTILS_TUPLE_UTILS_H

#include "lib/std/tuple.h"
#include "./integral_constant.h"

namespace Act {
namespace Gemm {
// Base template: handles single-index case
template <size_t I, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return AscendC::Std::get<I>(AscendC::Std::forward<T>(t));
}

// Recursive template: handles multiple index cases
template <size_t First, size_t Second, size_t... Rest, typename T>
__aicore__ constexpr inline decltype(auto) Get(T&& t)
{
    return Get<Second, Rest...>(AscendC::Std::get<First>(AscendC::Std::forward<T>(t)));
}

template <size_t N, typename Tp>
__aicore__ constexpr inline decltype(auto) GetIntegralConstant()
{
    static_assert(AscendC::Std::is_tuple_v<Tp>, "Input must be a AscendC::Std::tuple type");
    return AscendC::Std::tuple_element<N, Tp>::type::value;
}
} // namespace Gemm
} // namespace Act
#endif