/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * MatmulSwiglu aclnn 示例(正确性校验 + 可选计时二合一)。
 *   - 默认: 小形状 + host fp32 参考(matmul + SwiGLU), 逐元素比对, 打印 PASS/FAIL。
 *   - 设环境变量 MATMUL_SWIGLU_PERF=1: 额外跑多形状性能 sweep(warmup+iters, GFLOPS)。
 * 计算: [gate|up] = x @ weight (+bias), weight 打包 [K,2N]; y = SiLU(gate)*up, 输出 [M,N]。
 * 数据类型默认 fp16(x/weight/y), bias 固定 fp32; 改 PERF_FP32=1 可切 fp32。
 */
#include <acl/acl.h>
#include <aclnnop/aclnn_matmul_swiglu.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <vector>

#define PERF_FP32 0  // 0=fp16(默认), 1=fp32

#define CHECK_RET(cond, expr)                    \
    do {                                         \
        if (!(cond)) {                           \
            expr;                                \
        }                                        \
    } while (0)

#define LOG(fmt, ...)               \
    do {                            \
        printf(fmt, ##__VA_ARGS__); \
        fflush(stdout);             \
    } while (0)

namespace {
#if PERF_FP32
using XT = float;
constexpr aclDataType kXDtype = ACL_FLOAT;
inline XT MakeX(float v) { return v; }
inline float XToFloat(XT v) { return v; }
#else
using XT = uint16_t;  // fp16 存储
constexpr aclDataType kXDtype = ACL_FLOAT16;

// float <-> IEEE754 half, 仅用于造输入 / 读输出以做 host 参考比对
uint16_t FloatToHalf(float f)
{
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000U;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xFFU) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFU;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float HalfToFloat(uint16_t h)
{
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1FU;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3FFU;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            int32_t e = 127 - 15 + 1;
            while ((mant & 0x400U) == 0) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3FFU;
            f = sign | (static_cast<uint32_t>(e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1FU) {
        f = sign | 0x7F800000U | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

inline XT MakeX(float v) { return FloatToHalf(v); }
inline float XToFloat(XT v) { return HalfToFloat(v); }
#endif

int64_t ShapeSize(const std::vector<int64_t>& s)
{
    int64_t n = 1;
    for (auto d : s) {
        n *= d;
    }
    return n;
}

template <typename T>
int CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, void** devAddr,
                 aclDataType dtype, aclTensor** tensor)
{
    auto bytes = ShapeSize(shape) * static_cast<int64_t>(sizeof(T));
    auto ret = aclrtMalloc(devAddr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMalloc failed %d\n", ret); return ret);
    ret = aclrtMemcpy(*devAddr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtMemcpy failed %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *devAddr);
    CHECK_RET(*tensor != nullptr, LOG("aclCreateTensor null\n"); return -1);
    return 0;
}

// ---- 正确性校验: 小形状, 确定性输入, host fp32 参考(用 fp16 round-trip 后的输入值以对齐设备) ----
int RunCorrectness(aclrtStream stream)
{
    const int64_t M = 32, K = 128, N = 64, twoN = 2 * N;
    LOG("[正确性] M=%ld K=%ld N=%ld (%s)\n", M, K, N, PERF_FP32 ? "fp32" : "fp16");

    std::vector<XT> xH(M * K), wH(K * twoN);
    std::vector<float> xf(M * K), wf(K * twoN);  // 设备实际看到的输入值(fp16 round-trip 后)
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < K; j++) {
            float v = static_cast<float>((i + j) % 7) * 0.01f;
            xH[i * K + j] = MakeX(v);
            xf[i * K + j] = XToFloat(xH[i * K + j]);
        }
    }
    for (int64_t j = 0; j < K; j++) {
        for (int64_t c = 0; c < twoN; c++) {
            float v = static_cast<float>((j + c) % 5) * 0.01f;
            wH[j * twoN + c] = MakeX(v);
            wf[j * twoN + c] = XToFloat(wH[j * twoN + c]);
        }
    }
    std::vector<float> bH(twoN, 0.0f);  // bias 固定 fp32, 取 0

    // host 参考: acc[m,c] = sum_k xf*wf; y[m,cc] = SiLU(acc[m,cc]) * acc[m,cc+N]
    std::vector<float> ref(M * N, 0.0f);
    for (int64_t m = 0; m < M; m++) {
        for (int64_t cc = 0; cc < N; cc++) {
            float g = 0.0f, u = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                g += xf[m * K + k] * wf[k * twoN + cc];
                u += xf[m * K + k] * wf[k * twoN + cc + N];
            }
            float silu = g / (1.0f + std::exp(-g));
            ref[m * N + cc] = silu * u;
        }
    }

    std::vector<XT> yH(M * N, MakeX(0.0f));
    aclTensor *x = nullptr, *weight = nullptr, *bias = nullptr, *out = nullptr;
    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    int ret = CreateTensor(xH, {M, K}, &xDev, kXDtype, &x);
    ret |= CreateTensor(wH, {K, twoN}, &wDev, kXDtype, &weight);
    ret |= CreateTensor(bH, {twoN}, &bDev, ACL_FLOAT, &bias);
    ret |= CreateTensor(yH, {M, N}, &yDev, kXDtype, &out);
    CHECK_RET(ret == 0, LOG("create tensor failed\n"); return -1);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMatmulSwigluGetWorkspaceSize(x, weight, bias, false, out, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG("GetWorkspaceSize failed %d\n", ret); return ret);
    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG("alloc ws failed %d\n", ret); return ret);
    }
    ret = aclnnMatmulSwiglu(ws, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclnnMatmulSwiglu failed %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("sync failed %d\n", ret); return ret);

    ret = aclrtMemcpy(yH.data(), yH.size() * sizeof(XT), yDev, yH.size() * sizeof(XT), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG("copy out failed %d\n", ret); return ret);

    // 逐元素相对误差比对
    double maxRel = 0.0;
    int64_t worst = 0;
    for (int64_t i = 0; i < M * N; i++) {
        float got = XToFloat(yH[i]);
        float exp = ref[i];
        double rel = std::fabs(got - exp) / (std::fabs(exp) + 1e-3);
        if (rel > maxRel) {
            maxRel = rel;
            worst = i;
        }
    }
    const double tol = PERF_FP32 ? 1e-4 : 1e-2;  // fp16 容差 1%
    bool pass = maxRel <= tol;
    LOG("  y[0..3]=%.5f %.5f %.5f %.5f  ref[0..3]=%.5f %.5f %.5f %.5f\n",
        XToFloat(yH[0]), XToFloat(yH[1]), XToFloat(yH[2]), XToFloat(yH[3]),
        ref[0], ref[1], ref[2], ref[3]);
    LOG("  max rel err = %.4e (@%ld, got=%.5f ref=%.5f), tol=%.1e -> %s\n",
        maxRel, worst, XToFloat(yH[worst]), ref[worst], tol, pass ? "PASS" : "FAIL");

    if (ws != nullptr) {
        aclrtFree(ws);
    }
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclrtFree(xDev);
    aclrtFree(wDev);
    aclrtFree(bDev);
    aclrtFree(yDev);
    return pass ? 0 : -1;
}

// ---- 可选性能: 单形状端到端延迟 + 吞吐 ----
struct Shape {
    int64_t m;
    int64_t k;
    int64_t n;
};

int RunPerfShape(const Shape& sp, int warmup, int iters, aclrtStream stream)
{
    const std::vector<int64_t> xShape = {sp.m, sp.k};
    const std::vector<int64_t> wShape = {sp.k, 2 * sp.n};
    const std::vector<int64_t> bShape = {2 * sp.n};
    const std::vector<int64_t> yShape = {sp.m, sp.n};

    std::vector<XT> xHost(ShapeSize(xShape), MakeX(0.1f));
    std::vector<XT> wHost(ShapeSize(wShape), MakeX(0.1f));
    std::vector<float> bHost(ShapeSize(bShape), 0.0f);
    std::vector<XT> yHost(ShapeSize(yShape), MakeX(0.0f));

    aclTensor *x = nullptr, *weight = nullptr, *bias = nullptr, *out = nullptr;
    void *xDev = nullptr, *wDev = nullptr, *bDev = nullptr, *yDev = nullptr;
    int ret = CreateTensor(xHost, xShape, &xDev, kXDtype, &x);
    ret |= CreateTensor(wHost, wShape, &wDev, kXDtype, &weight);
    ret |= CreateTensor(bHost, bShape, &bDev, ACL_FLOAT, &bias);
    ret |= CreateTensor(yHost, yShape, &yDev, kXDtype, &out);
    CHECK_RET(ret == 0, LOG("create tensor failed\n"); return -1);

    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnMatmulSwigluGetWorkspaceSize(x, weight, bias, false, out, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG("GetWorkspaceSize failed %d\n", ret); return ret);
    void* ws = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG("alloc ws failed %d\n", ret); return ret);
    }

    auto launch = [&]() -> int {
        aclOpExecutor* exec = nullptr;
        uint64_t sz = 0;
        int r = aclnnMatmulSwigluGetWorkspaceSize(x, weight, bias, false, out, &sz, &exec);
        if (r != ACL_SUCCESS) {
            return r;
        }
        return aclnnMatmulSwiglu(ws, wsSize, exec, stream);
    };

    for (int i = 0; i < warmup; i++) {
        CHECK_RET(launch() == ACL_SUCCESS, LOG("warmup failed\n"); return -1);
    }
    aclrtSynchronizeStream(stream);

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        CHECK_RET(launch() == ACL_SUCCESS, LOG("timed failed\n"); return -1);
    }
    aclrtSynchronizeStream(stream);
    auto t1 = std::chrono::steady_clock::now();

    double avgUs = std::chrono::duration<double, std::micro>(t1 - t0).count() / iters;
    double flops = 2.0 * static_cast<double>(sp.m) * (2.0 * static_cast<double>(sp.n)) * static_cast<double>(sp.k);
    LOG("  M=%-5ld K=%-5ld N=%-6ld | avg %8.2f us | %8.1f GFLOPS\n",
        sp.m, sp.k, sp.n, avgUs, flops / (avgUs * 1e-6) / 1e9);

    if (ws != nullptr) {
        aclrtFree(ws);
    }
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclrtFree(xDev);
    aclrtFree(wDev);
    aclrtFree(bDev);
    aclrtFree(yDev);
    return 0;
}
}  // namespace

int main()
{
    const int32_t deviceId = 0;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclInit failed %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtSetDevice failed %d\n", ret); return ret);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG("aclrtCreateStream failed %d\n", ret); return ret);

    LOG("=== MatmulSwiglu 示例 ===\n");
    int rc = RunCorrectness(stream);

    // 可选性能 sweep: 设 MATMUL_SWIGLU_PERF=1 开启
    if (std::getenv("MATMUL_SWIGLU_PERF") != nullptr) {
        LOG("=== 性能 sweep (%s, warmup=10 iters=50) ===\n", PERF_FP32 ? "fp32" : "fp16");
        const std::vector<Shape> shapes = {{512, 2048, 2048}, {2048, 4096, 4096}, {4096, 4096, 11008}};
        for (const auto& sp : shapes) {
            RunPerfShape(sp, 10, 50, stream);
        }
    } else {
        LOG("(设环境变量 MATMUL_SWIGLU_PERF=1 可额外运行性能 sweep)\n");
    }

    if (stream != nullptr) {
        aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
    LOG("=== 结果: %s ===\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
