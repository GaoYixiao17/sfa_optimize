/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
#define TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_

#include <cstring>
#include "ops_base.h"

namespace npu_linalg {
    std::tuple<at::Tensor, at::Tensor> construct_tsqr_output_tensor(const at::Tensor& x);
    std::tuple<at::Tensor, at::Tensor, at::Tensor> construct_svd_output_tensor(const at::Tensor& x);
} // namespace npu_linalg

inline aclScalar* ConvertType(const at::Scalar& at_scalar)
{
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    if (aclCreateScalar == nullptr) {
        return nullptr;
    }

    at::ScalarType scalar_data_type = at_scalar.type();
    aclDataType acl_data_type =
        kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
    TORCH_CHECK(
        acl_data_type != ACL_DT_UNDEFINED,
        std::string(c10::toString(scalar_data_type)) + " has not been supported")
        aclScalar* acl_scalar = nullptr;
    switch (scalar_data_type) {
        case at::ScalarType::Double: {
            double value = at_scalar.toDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Long: {
            int64_t value = at_scalar.toLong();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::Bool: {
            bool value = at_scalar.toBool();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        case at::ScalarType::ComplexDouble: {
            auto value = at_scalar.toComplexDouble();
            acl_scalar = aclCreateScalar(&value, acl_data_type);
            break;
        }
        default:
            acl_scalar = nullptr;
            break;
    }
    return acl_scalar;
}

inline aclBoolArray* ConvertType(const at::ArrayRef<bool>& value)
{
    static const auto aclCreateBoolArray = GET_OP_API_FUNC(aclCreateBoolArray);
    if (aclCreateBoolArray == nullptr) {
        return nullptr;
    }

    auto array = aclCreateBoolArray(value.data(), value.size());
    return array;
}

inline aclIntArray* ConvertType(const at::ArrayRef<c10::SymInt>& at_array)
{
    static const auto aclCreateIntArray = GET_OP_API_FUNC(aclCreateIntArray);
    if (aclCreateIntArray == nullptr) {
        return nullptr;
    }
    auto int_array = c10::asIntArrayRefUnchecked(at_array);
    auto array = aclCreateIntArray(int_array.data(), int_array.size());
    return array;
}

inline aclTensorList* ConvertType(const at::TensorList& at_tensor_list)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor*> tensor_list(at_tensor_list.size());
    for (size_t i = 0; i < at_tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(at_tensor_list[i]);
    }
    auto acl_tensor_list =
        aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}

inline aclScalarList* ConvertType(const at::ArrayRef<at::Scalar>& at_scalar_list)
{
    static const auto aclCreateScalarList = GET_OP_API_FUNC(aclCreateScalarList);
    if (aclCreateScalarList == nullptr) {
        return nullptr;
    }

    std::vector<const aclScalar*> scalar_list(at_scalar_list.size());
    for (size_t i = 0; i < at_scalar_list.size(); i++) {
        scalar_list[i] = ConvertType(at_scalar_list[i]);
    }
    auto acl_scalar_list = aclCreateScalarList(scalar_list.data(), scalar_list.size());
    return acl_scalar_list;
}

inline aclTensor* ConvertType(const c10::optional<at::Tensor>& opt_tensor)
{
    if (opt_tensor.has_value() && opt_tensor.value().defined()) {
        return ConvertType(opt_tensor.value());
    }

    return nullptr;
}

inline aclIntArray* ConvertType(const c10::optional<at::IntArrayRef>& opt_array)
{
    if (opt_array.has_value()) {
        return ConvertType(opt_array.value());
    }
    return nullptr;
}

inline aclScalar* ConvertType(const c10::optional<at::Scalar>& opt_scalar)
{
    if (opt_scalar.has_value()) {
        return ConvertType(opt_scalar.value());
    }
    return nullptr;
}

inline aclDataType ConvertType(const at::ScalarType scalarType)
{
    return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalarType)];
}

// add declare from other hpp
typedef struct {
    const at::Tensor& tensor_;
    aclDataType dtype;
} TensorWrapper;

typedef struct {
    const at::TensorList& tensor_list_;
    aclDataType dtype;
} TensorListWrapper;


c10::SmallVector<int64_t, SIZE> array_to_small_vector(c10::IntArrayRef shape);
// add declare from other hpp

inline aclTensor* ConvertType(const TensorWrapper& tensor_r)
{
    static const auto aclCreateTensor = GET_OP_API_FUNC(aclCreateTensor);
    if (aclCreateTensor == nullptr) return nullptr;

    const at::Tensor& at_tensor = tensor_r.tensor_;
    if (!at_tensor.defined()) return nullptr;
    TORCH_CHECK(torch_npu::utils::is_npu(at_tensor),
        "Expected all tensors to be on the same device. "
        "Expected NPU tensor, please check whether the input tensor device is correct.",
        OPS_ERROR(ErrCode::TYPE));

    aclDataType acl_data_type = tensor_r.dtype;
    c10::SmallVector<int64_t, MAX_DIM_NUM> storageDims;
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperStride = array_to_small_vector(at_tensor.strides());
    c10::SmallVector<int64_t, MAX_DIM_NUM> wrapperShape = array_to_small_vector(at_tensor.sizes());

    const auto dimNum = at_tensor.sizes().size();
    aclFormat format = ACL_FORMAT_ND;
    if (!at_npu::native::FormatHelper::IsOpInputBaseFormat(at_tensor)) {
        format = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.npu_format_;
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                OPS_ERROR(ErrCode::VALUE));
            storageDims = torch_npu::NPUBridge::GetNpuStorageImpl(at_tensor)->npu_desc_.storage_sizes_;
        }
    } else {
        switch (dimNum) {
            case NCL_DIM_NUM:
                format = ACL_FORMAT_NCL;
                break;
            case NCHW_DIM_NUM:
                format = ACL_FORMAT_NCHW;
                break;
            case NCDHW_DIM_NUM:
                format = ACL_FORMAT_NCDHW;
                break;
            default:
                format = ACL_FORMAT_ND;
        }
        // if acl_data_type is ACL_STRING, storageDims is empty.
        if (acl_data_type != ACL_STRING) {
            TORCH_CHECK(at_tensor.itemsize() > 0, "the itemsize of tensor must be greater than 0.",
                OPS_ERROR(ErrCode::VALUE));
            storageDims.push_back(at_tensor.storage().nbytes() / at_tensor.itemsize());
        }
    }

    auto acl_tensor =
        aclCreateTensor(wrapperShape.data(), at_tensor.sizes().size(), acl_data_type, wrapperStride.data(),
            at_tensor.storage_offset(), format, storageDims.data(), storageDims.size(),
            const_cast<void*>(at_tensor.storage().data()));
    return acl_tensor;
}

inline aclTensorList* ConvertType(const TensorListWrapper& tensor_list_wrapper)
{
    static const auto aclCreateTensorList = GET_OP_API_FUNC(aclCreateTensorList);
    if (aclCreateTensorList == nullptr) {
        return nullptr;
    }

    std::vector<const aclTensor*> tensor_list(tensor_list_wrapper.tensor_list_.size());
    for (size_t i = 0; i < tensor_list.size(); i++) {
        tensor_list[i] = ConvertType(TensorWrapper{
            tensor_list_wrapper.tensor_list_[i], tensor_list_wrapper.dtype });
    }
    auto acl_tensor_list = aclCreateTensorList(tensor_list.data(), tensor_list.size());
    return acl_tensor_list;
}


template <typename T>
T ConvertType(T value)
{
    return value;
}


template <typename Tuple, size_t... I>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr, std::index_sequence<I...>)
{
    typedef int (*OpApiFunc)(
        typename std::decay<decltype(std::get<I>(params))>::type...);
    auto func = reinterpret_cast<OpApiFunc>(opApiAddr);
    return func;
}

template <typename Tuple>
auto ConvertToOpApiFunc(const Tuple& params, void* opApiAddr)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return ConvertToOpApiFunc(params, opApiAddr,
        std::make_index_sequence<size>{});
}


// add release for all type
inline void Release(aclTensor* p)
{
    static const auto aclDestroyTensor = GET_OP_API_FUNC(aclDestroyTensor);
    if (aclDestroyTensor == nullptr) {
        return;
    }
    aclDestroyTensor(p);
}

inline void Release(aclScalar* p)
{
    static const auto aclDestroyScalar = GET_OP_API_FUNC(aclDestroyScalar);
    if (aclDestroyScalar == nullptr) {
        return;
    }
    aclDestroyScalar(p);
}

inline void Release(aclIntArray* p)
{
    static const auto aclDestroyIntArray = GET_OP_API_FUNC(aclDestroyIntArray);
    if (aclDestroyIntArray == nullptr) {
        return;
    }

    aclDestroyIntArray(p);
}

inline void Release(aclBoolArray* p)
{
    static const auto aclDestroyBoolArray = GET_OP_API_FUNC(aclDestroyBoolArray);
    if (aclDestroyBoolArray == nullptr) {
        return;
    }

    aclDestroyBoolArray(p);
}

inline void Release(aclTensorList* p)
{
    static const auto aclDestroyTensorList =
        GET_OP_API_FUNC(aclDestroyTensorList);
    if (aclDestroyTensorList == nullptr) {
        return;
    }

    aclDestroyTensorList(p);
}

inline void Release(aclScalarList* p)
{
    static const auto aclDestroyScalarList = GET_OP_API_FUNC(aclDestroyScalarList);
    if (aclDestroyScalarList == nullptr) {
        return;
    }

    aclDestroyScalarList(p);
}

template <typename T>
void Release(T value)
{
    (void)value;
}

template <typename Tuple, size_t... I>
void CallRelease(Tuple t, std::index_sequence<I...>)
{
    (void)std::initializer_list<int>{(Release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void ReleaseConvertTypes(Tuple& t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    CallRelease(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto ConvertTypes(Ts &... args)
{
    return std::make_tuple(ConvertType(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>)
{
    return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t)
{
    static constexpr auto size = std::tuple_size<Tuple>::value;
    return call(f, t, std::make_index_sequence<size>{});
}

template <std::size_t N>
void AddParamToBuf(const std::array<bool, N>&);
template <typename T>
void AddParamToBuf(const T&);
void AddParamToBuf(const at::Tensor&);
void AddParamToBuf(const at::Scalar&);
void AddParamToBuf(const at::IntArrayRef&);
void AddParamToBuf(const at::ArrayRef<bool>&);
void AddParamToBuf(const at::TensorList&);
void AddParamToBuf(const c10::optional<at::Tensor>&);
void AddParamToBuf(const c10::optional<at::IntArrayRef>&);
void AddParamToBuf(const c10::optional<at::Scalar>&);
void AddParamToBuf(const at::ScalarType);
void AddParamToBuf(const string&);
void AddParamToBuf();

template <typename T, typename... Args>
void AddParamToBuf(const T& arg, Args &... args)
{
    AddParamToBuf(arg);
    AddParamToBuf(args...);
}

// for cache
uint64_t CalcHashId();

typedef int (*InitHugeMemThreadLocal)(void*, bool);
typedef void (*UnInitHugeMemThreadLocal)(void*, bool);
typedef void (*ReleaseHugeMem)(void*, bool);
typedef aclOpExecutor* (*PTAGetExecCache)(uint64_t, uint64_t*);
typedef aclOpExecutor* (*PTAFindExecCache)(uint8_t*, size_t, uint64_t*);
typedef void (*InitPTACacheThreadLocal)();
typedef void (*SetPTAHashKey)(uint64_t);
typedef void (*SetPTACacheHashKey)(uint8_t*, size_t);
typedef bool (*CanUsePTACache)(const char*);
typedef void (*UnInitPTACacheThreadLocal)();


inline void UnInitCacheThreadLocal()
{
    static const auto unInitPTACacheThreadLocalAddr = GetOpApiFuncAddr("UnInitPTACacheThreadLocal");
    UnInitPTACacheThreadLocal unInitPTACacheThreadLocalFunc =
        reinterpret_cast<UnInitPTACacheThreadLocal>(unInitPTACacheThreadLocalAddr);
    if (unInitPTACacheThreadLocalFunc) {
        unInitPTACacheThreadLocalFunc();
    }
}


#define EXEC_NPU_CMD_V1(aclnn_api, ...)                                                                            \
    do {                                                                                                           \
        static const auto getWorkspaceSizeFuncAddr = GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");              \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);                                            \
        static const auto initMemAddr = GetOpApiFuncAddr("InitHugeMemThreadLocal");                                \
        static const auto unInitMemAddr = GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                            \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");                                     \
        static const auto initPTACacheThreadLocalAddr = GetOpApiFuncAddr("InitPTACacheThreadLocal");               \
        static const auto setPTAHashKeyAddr = GetOpApiFuncAddr("SetPTAHashKey");                                   \
        TORCH_CHECK(getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr, #aclnn_api, " or ",           \
                    #aclnn_api "GetWorkspaceSize", " not in ", GetOpApiLibName(), ", or ", GetOpApiLibName(),      \
                    "not found.");                                                                                 \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);                                            \
        uint64_t workspace_size = 0;                                                                               \
        uint64_t *workspace_size_addr = &workspace_size;                                                           \
        aclOpExecutor *executor = nullptr;                                                                         \
        aclOpExecutor **executor_addr = &executor;                                                                 \
        InitHugeMemThreadLocal initMemFunc = reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                \
        UnInitHugeMemThreadLocal unInitMemFunc = reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);        \
        InitPTACacheThreadLocal initPTACacheThreadLocalFunc =                                                      \
            reinterpret_cast<InitPTACacheThreadLocal>(initPTACacheThreadLocalAddr);                                \
        SetPTAHashKey setPTAHashKeyFunc = reinterpret_cast<SetPTAHashKey>(setPTAHashKeyAddr);                      \
        if (initPTACacheThreadLocalFunc && setPTAHashKeyFunc) {                                                    \
            initPTACacheThreadLocalFunc();                                                                         \
            setPTAHashKeyFunc(0);                                                                                  \
        }                                                                                                          \
        if (initMemFunc) {                                                                                         \
            initMemFunc(nullptr, false);                                                                           \
        }                                                                                                          \
        auto converted_params = ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);                     \
        static auto getWorkspaceSizeFunc = ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);         \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);                                      \
        TORCH_CHECK(workspace_status== 0, "call " #aclnn_api " failed");                                           \
        void *workspace_addr = nullptr;                                                                            \
        at::Tensor workspace_tensor;                                                                               \
        if (workspace_size != 0) {                                                                                 \
            at::TensorOptions options =                                                                            \
                at::TensorOptions(torch_npu::utils::get_npu_device_type());                                        \
            auto workspace_tensor =                                                                                \
                at::empty({static_cast<int64_t>(workspace_size)}, options.dtype(at::kByte));                       \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data());                                \
        }                                                                                                          \
        auto acl_call = [converted_params, workspace_addr, workspace_size, acl_stream, executor]()->int {          \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);                                      \
            auto api_ret = opApiFunc(workspace_addr, workspace_size, executor, acl_stream);                        \
            TORCH_CHECK(api_ret==0, "call " #aclnn_api " failed");                                                 \
            ReleaseConvertTypes(converted_params);                                                                 \
            ReleaseHugeMem releaseMemFunc = reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                      \
            if (releaseMemFunc) {                                                                                  \
                releaseMemFunc(nullptr, false);                                                                    \
            }                                                                                                      \
            return api_ret;                                                                                        \
        };                                                                                                         \
        at_npu::native::OpCommand::RunOpApiV2(#aclnn_api, acl_call);                                               \
        if (unInitMemFunc) {                                                                                       \
            unInitMemFunc(nullptr, false);                                                                         \
        }                                                                                                          \
        UnInitCacheThreadLocal();                                                                                  \
    } while (false)

#define EXEC_NPU_CMD_v0(aclnn_api, ...)                                             \
    do {                                                                            \
        static const auto getWorkspaceSizeFuncAddr =                                \
            GetOpApiFuncAddr(#aclnn_api "GetWorkspaceSize");                        \
        static const auto opApiFuncAddr = GetOpApiFuncAddr(#aclnn_api);             \
        static const auto initMemAddr =                                             \
            GetOpApiFuncAddr("InitHugeMemThreadLocal");                             \
        static const auto unInitMemAddr =                                           \
            GetOpApiFuncAddr("UnInitHugeMemThreadLocal");                           \
        static const auto releaseMemAddr = GetOpApiFuncAddr("ReleaseHugeMem");      \
        TORCH_CHECK(                                                                \
            getWorkspaceSizeFuncAddr != nullptr && opApiFuncAddr != nullptr,        \
            #aclnn_api, " or ", #aclnn_api "GetWorkspaceSize", " not in ",          \
            GetOpApiLibName(), ", or ", GetOpApiLibName(), "not found.");           \
        auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);             \
        uint64_t workspace_size = 0;                                                \
        uint64_t *workspace_size_addr = &workspace_size;                            \
        aclOpExecutor *executor = nullptr;                                          \
        aclOpExecutor **executor_addr = &executor;                                  \
        InitHugeMemThreadLocal initMemFunc =                                        \
            reinterpret_cast<InitHugeMemThreadLocal>(initMemAddr);                  \
        UnInitHugeMemThreadLocal unInitMemFunc =                                    \
            reinterpret_cast<UnInitHugeMemThreadLocal>(unInitMemAddr);              \
        if (initMemFunc) {                                                          \
            initMemFunc(nullptr, false);                                            \
        }                                                                           \
        auto converted_params =                                                     \
            ConvertTypes(__VA_ARGS__, workspace_size_addr, executor_addr);          \
        static auto getWorkspaceSizeFunc =                                          \
            ConvertToOpApiFunc(converted_params, getWorkspaceSizeFuncAddr);         \
        auto workspace_status = call(getWorkspaceSizeFunc, converted_params);       \
        TORCH_CHECK(workspace_status == 0,                                          \
                  "call " #aclnn_api " failed, detail:", aclGetRecentErrMsg());     \
        void *workspace_addr = nullptr;                                             \
        if (workspace_size != 0) {                                                  \
            at::TensorOptions options =                                             \
                at::TensorOptions(torch_npu::utils::get_npu_device_type());         \
            auto workspace_tensor =                                                 \
                at::empty({workspace_size}, options.dtype(at::kByte));              \
            workspace_addr = const_cast<void *>(workspace_tensor.storage().data()); \
        }                                                                           \
        auto acl_call = [converted_params, workspace_addr, workspace_size,          \
                           acl_stream, executor]()->int {                           \
            typedef int (*OpApiFunc)(void *, uint64_t, aclOpExecutor *,             \
                                     const aclrtStream);                            \
            OpApiFunc opApiFunc = reinterpret_cast<OpApiFunc>(opApiFuncAddr);       \
            auto api_ret =                                                          \
                opApiFunc(workspace_addr, workspace_size, executor, acl_stream);    \
            TORCH_CHECK(api_ret == 0, "call " #aclnn_api " failed, detail:",        \
                        aclGetRecentErrMsg());                                      \
            ReleaseConvertTypes(converted_params);                                  \
            ReleaseHugeMem releaseMemFunc =                                         \
                reinterpret_cast<ReleaseHugeMem>(releaseMemAddr);                   \
            if (releaseMemFunc) {                                                   \
                releaseMemFunc(nullptr, false);                                     \
            }                                                                       \
            return api_ret;                                                         \
        };                                                                          \
        at_npu::native::OpCommand cmd;                                              \
        cmd.Name(#aclnn_api);                                                       \
        cmd.SetCustomHandler(acl_call);                                             \
        cmd.Run();                                                                  \
        if (unInitMemFunc) {                                                        \
            unInitMemFunc(nullptr, false);                                          \
        }                                                                           \
    } while (false)

#endif  // TORCHNPU_TORCH_NPU_CSRC_ATEN_OPS_OP_API_PTA_COMMON_H_
