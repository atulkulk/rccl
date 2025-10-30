/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "nccl.h"
#include <cmath>
#include <hip/hip_runtime.h>
#include <type_traits>
#include <vector>

/**
 * @file RCCLTestBufferHelpers.hpp
 * @brief Template-based buffer transfer utilities for RCCL tests
 *
 * Provides type-safe, reusable functions for common buffer operations:
 * - Initialization with test patterns
 * - Host <-> Device transfers
 * - Data verification
 * - NCCL datatype mapping
 */

namespace RCCLTestHelpers
{

// ============================================================================
// NCCL Datatype Mapping
// ============================================================================

/**
 * @brief Maps C++ types to NCCL data types at compile time
 * @tparam T C++ data type
 */
template<typename T>
struct NcclTypeTraits;

/** @brief Specialization for float */
template<>
struct NcclTypeTraits<float>
{
    static constexpr ncclDataType_t value = ncclFloat;
    static constexpr const char*    name  = "float";
};

/** @brief Specialization for double */
template<>
struct NcclTypeTraits<double>
{
    static constexpr ncclDataType_t value = ncclDouble;
    static constexpr const char*    name  = "double";
};

/** @brief Specialization for int32_t */
template<>
struct NcclTypeTraits<int32_t>
{
    static constexpr ncclDataType_t value = ncclInt32;
    static constexpr const char*    name  = "int32_t";
};

/** @brief Specialization for uint32_t */
template<>
struct NcclTypeTraits<uint32_t>
{
    static constexpr ncclDataType_t value = ncclUint32;
    static constexpr const char*    name  = "uint32_t";
};

/** @brief Specialization for int64_t */
template<>
struct NcclTypeTraits<int64_t>
{
    static constexpr ncclDataType_t value = ncclInt64;
    static constexpr const char*    name  = "int64_t";
};

/** @brief Specialization for uint64_t */
template<>
struct NcclTypeTraits<uint64_t>
{
    static constexpr ncclDataType_t value = ncclUint64;
    static constexpr const char*    name  = "uint64_t";
};

/** @brief Specialization for int8_t */
template<>
struct NcclTypeTraits<int8_t>
{
    static constexpr ncclDataType_t value = ncclInt8;
    static constexpr const char*    name  = "int8_t";
};

/** @brief Specialization for uint8_t */
template<>
struct NcclTypeTraits<uint8_t>
{
    static constexpr ncclDataType_t value = ncclUint8;
    static constexpr const char*    name  = "uint8_t";
};

/**
 * @brief Helper function to get NCCL datatype for a C++ type
 * @tparam T C++ data type
 * @return Corresponding ncclDataType_t
 */
template<typename T>
constexpr ncclDataType_t getNcclDataType()
{
    return NcclTypeTraits<T>::value;
}

/**
 * @brief Helper function to get type name string
 * @tparam T C++ data type
 * @return Type name as string
 */
template<typename T>
constexpr const char* getTypeName()
{
    return NcclTypeTraits<T>::name;
}

// ============================================================================
// Buffer Initialization
// ============================================================================

/**
 * @brief Initialize device buffer with rank-based test pattern
 *
 * Creates a predictable pattern: rank * multiplier + index
 * This pattern is deterministic and easy to verify.
 *
 * @tparam T Element type (float, int, etc.)
 * @param device_buffer Device memory pointer
 * @param num_elements Number of elements
 * @param rank MPI rank (used in pattern generation)
 * @param multiplier Pattern multiplier (default: 1000)
 * @return hipError_t from hipMemcpy, or hipSuccess
 */
template<typename T>
hipError_t initializeBufferWithPattern(void*  device_buffer,
                                       size_t num_elements,
                                       int    rank,
                                       int    multiplier = 1000)
{
    if(!device_buffer || num_elements == 0)
    {
        return hipErrorInvalidValue;
    }

    // Create host buffer with pattern
    std::vector<T> host_data(num_elements);
    for(size_t i = 0; i < num_elements; i++)
    {
        host_data[i] = static_cast<T>(rank * multiplier + i);
    }

    // Copy to device
    return hipMemcpy(device_buffer,
                     host_data.data(),
                     num_elements * sizeof(T),
                     hipMemcpyHostToDevice);
}

/**
 * @brief Initialize device buffer with custom pattern function
 *
 * Allows custom pattern generation via lambda or function pointer.
 *
 * @tparam T Element type
 * @tparam PatternFunc Callable type (lambda, function pointer, functor)
 * @param device_buffer Device memory pointer
 * @param num_elements Number of elements
 * @param pattern_func Function that generates value for each index: T pattern_func(size_t index)
 * @return hipError_t from hipMemcpy, or hipSuccess
 */
template<typename T, typename PatternFunc>
hipError_t initializeBufferWithCustomPattern(void*       device_buffer,
                                             size_t      num_elements,
                                             PatternFunc pattern_func)
{
    if(!device_buffer || num_elements == 0)
    {
        return hipErrorInvalidValue;
    }

    std::vector<T> host_data(num_elements);
    for(size_t i = 0; i < num_elements; i++)
    {
        host_data[i] = pattern_func(i);
    }

    return hipMemcpy(device_buffer,
                     host_data.data(),
                     num_elements * sizeof(T),
                     hipMemcpyHostToDevice);
}

/**
 * @brief Zero-initialize device buffer
 *
 * @tparam T Element type
 * @param device_buffer Device memory pointer
 * @param num_elements Number of elements
 * @return hipError_t from hipMemset
 */
template<typename T>
hipError_t zeroInitializeBuffer(void* device_buffer, size_t num_elements)
{
    if(!device_buffer || num_elements == 0)
    {
        return hipErrorInvalidValue;
    }

    return hipMemset(device_buffer, 0, num_elements * sizeof(T));
}

// ============================================================================
// Buffer Verification
// ============================================================================

/**
 * @brief Verify device buffer contains expected rank-based pattern
 *
 * Downloads data from device and verifies first num_samples elements.
 * Uses appropriate comparison for floating-point vs integer types.
 *
 * @tparam T Element type
 * @param device_buffer Device memory pointer
 * @param num_elements Total number of elements
 * @param expected_rank Rank that generated the pattern
 * @param multiplier Pattern multiplier (must match initialization)
 * @param num_samples Number of elements to verify (default: 10, capped at num_elements)
 * @param tolerance Tolerance for floating-point comparison (default: 1e-5)
 * @param[out] first_error_index If verification fails, set to index of first mismatch
 * @param[out] expected_value If verification fails, set to expected value
 * @param[out] actual_value If verification fails, set to actual value
 * @return true if all samples match, false otherwise
 */
template<typename T>
bool verifyBufferData(const void* device_buffer,
                      size_t      num_elements,
                      int         expected_rank,
                      int         multiplier        = 1000,
                      size_t      num_samples       = 10,
                      double      tolerance         = 1e-5,
                      size_t*     first_error_index = nullptr,
                      T*          expected_value    = nullptr,
                      T*          actual_value      = nullptr)
{
    if(!device_buffer || num_elements == 0)
    {
        return false;
    }

    // Cap num_samples at num_elements
    num_samples = std::min(num_samples, num_elements);

    // Download data from device
    std::vector<T> host_data(num_elements);
    hipError_t     err = hipMemcpy(host_data.data(),
                               device_buffer,
                               num_elements * sizeof(T),
                               hipMemcpyDeviceToHost);
    if(err != hipSuccess)
    {
        return false;
    }

    // Verify samples
    for(size_t i = 0; i < num_samples; i++)
    {
        T expected = static_cast<T>(expected_rank * multiplier + i);
        T actual   = host_data[i];

        bool matches = false;

        // Use appropriate comparison based on type
        if constexpr(std::is_floating_point_v<T>)
        {
            // Floating-point: use tolerance-based comparison
            matches = (std::abs(actual - expected) <= tolerance);
        }
        else
        {
            // Integer: exact comparison
            matches = (actual == expected);
        }

        if(!matches)
        {
            // Record error details
            if(first_error_index)
                *first_error_index = i;
            if(expected_value)
                *expected_value = expected;
            if(actual_value)
                *actual_value = actual;
            return false;
        }
    }

    return true;
}

/**
 * @brief Verify device buffer with custom verification function
 *
 * @tparam T Element type
 * @tparam VerifyFunc Callable type: bool verify_func(size_t index, T actual_value)
 * @param device_buffer Device memory pointer
 * @param num_elements Number of elements
 * @param verify_func Function that verifies each element
 * @param[out] first_error_index If verification fails, set to index of first mismatch
 * @return true if all elements pass verification
 */
template<typename T, typename VerifyFunc>
bool verifyBufferWithCustomCheck(const void* device_buffer,
                                 size_t      num_elements,
                                 VerifyFunc  verify_func,
                                 size_t*     first_error_index = nullptr)
{
    if(!device_buffer || num_elements == 0)
    {
        return false;
    }

    std::vector<T> host_data(num_elements);
    hipError_t     err = hipMemcpy(host_data.data(),
                               device_buffer,
                               num_elements * sizeof(T),
                               hipMemcpyDeviceToHost);
    if(err != hipSuccess)
    {
        return false;
    }

    for(size_t i = 0; i < num_elements; i++)
    {
        if(!verify_func(i, host_data[i]))
        {
            if(first_error_index)
                *first_error_index = i;
            return false;
        }
    }

    return true;
}

// ============================================================================
// Combined Operations
// ============================================================================

/**
 * @brief Allocate, initialize, and return RAII-guarded buffers
 *
 * Convenience function that combines allocation and initialization.
 * Returns host vector for later verification if needed.
 *
 * @tparam T Element type
 * @param[out] device_buffer Pointer to receive device buffer address
 * @param num_elements Number of elements
 * @param rank MPI rank for pattern generation
 * @param multiplier Pattern multiplier
 * @return std::pair<hipError_t, std::vector<T>> - error code and host data copy
 */
template<typename T>
std::pair<hipError_t, std::vector<T>> allocateAndInitialize(void** device_buffer,
                                                            size_t num_elements,
                                                            int    rank,
                                                            int    multiplier = 1000)
{
    if(!device_buffer)
    {
        return {hipErrorInvalidValue, {}};
    }

    // Allocate device memory
    hipError_t err = hipMalloc(device_buffer, num_elements * sizeof(T));
    if(err != hipSuccess)
    {
        return {err, {}};
    }

    // Create host data with pattern
    std::vector<T> host_data(num_elements);
    for(size_t i = 0; i < num_elements; i++)
    {
        host_data[i] = static_cast<T>(rank * multiplier + i);
    }

    // Copy to device
    err = hipMemcpy(*device_buffer,
                    host_data.data(),
                    num_elements * sizeof(T),
                    hipMemcpyHostToDevice);

    return {err, std::move(host_data)};
}

/**
 * @brief Copy data from one device buffer to another
 *
 * @tparam T Element type (used for size calculation)
 * @param dst Destination device buffer
 * @param src Source device buffer
 * @param num_elements Number of elements to copy
 * @return hipError_t from hipMemcpy
 */
template<typename T>
hipError_t copyDeviceBuffer(void* dst, const void* src, size_t num_elements)
{
    if(!dst || !src || num_elements == 0)
    {
        return hipErrorInvalidValue;
    }

    return hipMemcpy(dst, src, num_elements * sizeof(T), hipMemcpyDeviceToDevice);
}

/**
 * @brief Download device buffer to host vector
 *
 * @tparam T Element type
 * @param device_buffer Device memory pointer
 * @param num_elements Number of elements
 * @return std::pair<hipError_t, std::vector<T>> - error code and host data
 */
template<typename T>
std::pair<hipError_t, std::vector<T>> downloadBuffer(const void* device_buffer, size_t num_elements)
{
    std::vector<T> host_data(num_elements);

    if(!device_buffer || num_elements == 0)
    {
        return {hipErrorInvalidValue, {}};
    }

    hipError_t err = hipMemcpy(host_data.data(),
                               device_buffer,
                               num_elements * sizeof(T),
                               hipMemcpyDeviceToHost);

    return {err, std::move(host_data)};
}

} // namespace RCCLTestHelpers
