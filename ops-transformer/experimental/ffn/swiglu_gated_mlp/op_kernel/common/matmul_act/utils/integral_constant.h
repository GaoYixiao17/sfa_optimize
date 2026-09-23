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
 * \file integral_constant.h
 * \brief
 */
#ifndef UTILS_INTEGRAL_CONSTANT_H
#define UTILS_INTEGRAL_CONSTANT_H
#include "kernel_operator.h"

namespace AscendC {
namespace Std {
template <typename...>
struct always_false : public false_type {};

template <typename... Tp>
constexpr bool always_false_v = always_false<Tp...>::value;
} // namespace Std
} // namespace AscendC

namespace Act {
namespace Gemm {
template <int32_t t>
using Int = AscendC::Std::integral_constant<int32_t, t>;

using _0 = Int<0>;
using _1 = Int<1>;
using _2 = Int<2>;
using _4 = Int<4>;
using _8 = Int<8>;
using _16 = Int<16>;
using _32 = Int<32>;
using _64 = Int<64>;
using _128 = Int<128>;
using _256 = Int<256>;
using _512 = Int<512>;
using _1024 = Int<1024>;
using _2048 = Int<2048>;
using _4096 = Int<4096>;
using _8192 = Int<8192>;
using _16384 = Int<16384>;
using _32768 = Int<32768>;
using _65536 = Int<65536>;

// Unary operator
template <auto t>
__host_aicore__ inline constexpr Int<+t> operator+(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<-t> operator-(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<~t> operator~(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<!t> operator!(Int<t>)
{
    return {};
}

template <auto t>
__host_aicore__ inline constexpr Int<*t> operator*(Int<t>)
{
    return {};
}

// Binary operator
template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t + u)> operator+(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t - u)> operator-(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t * u)> operator*(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t / u)> operator/(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t % u)> operator%(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t & u)> operator&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t | u)> operator|(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t ^ u)> operator^(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t << u)> operator<<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >> u)> operator>>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t && u)> operator&&(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t || u)> operator||(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t == u)> operator==(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t != u)> operator!=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t > u)> operator>(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t < u)> operator<(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t >= u)> operator>=(Int<t>, Int<u>)
{
    return {};
}

template <auto t, auto u>
__host_aicore__ inline constexpr Int<(t <= u)> operator<=(Int<t>, Int<u>)
{
    return {};
}
} // namespace Gemm
} // namespace Act
#endif