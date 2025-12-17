/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <cstdlib>
#include <cstdio>

#include "common/ErrCode.hpp"
#include "StandaloneUtils.hpp"

namespace RcclUnitTesting
{

// Helper to check if local registration is enabled via environment variable
static bool isLocalRegisterEnabled() {
    const char* env = std::getenv("NCCL_LOCAL_REGISTER");
    return (env != nullptr && std::string(env) == "1");
}

// Print registration status
static void printRegistrationStatus() {
    if (isLocalRegisterEnabled()) {
        INFO("NCCL_LOCAL_REGISTER=1 (registration enabled)");
    } else {
        INFO("NCCL_LOCAL_REGISTER is not set or 0 (registration disabled, NULL handles expected)");
    }
}

/**
 * @brief Test ncclCommRegister and ncclCommDeregister APIs
 *
 * This test verifies that:
 * 1. A device buffer can be registered with ncclCommRegister (API returns success)
 * 2. When NCCL_LOCAL_REGISTER=1, the registration returns a valid (non-NULL) handle
 * 3. The buffer can be deregistered with ncclCommDeregister
 *
 * Note: NCCL_LOCAL_REGISTER defaults to 0 (disabled) in RCCL.
 * Run with NCCL_LOCAL_REGISTER=1 to test actual registration.
 */
TEST(Register, CommRegisterDeregister)
{
    // Print registration status for clarity
    printRegistrationStatus();

    // Check for GPU availability
    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        GTEST_SKIP() << "This test requires at least 1 GPU device.";
    }

    // Set device and initialize single-rank communicator
    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));

    // Create buffer on device
    const size_t bufferSize = 1024 * 1024; // 1 MB
    void* deviceBuffer = nullptr;
    HIPCALL(hipMalloc(&deviceBuffer, bufferSize));
    ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate device buffer";

    // Register buffer with ncclCommRegister
    void* regHandle = nullptr;
    NCCLCHECK(ncclCommRegister(comm, deviceBuffer, bufferSize, &regHandle));

    // Check registration result based on whether feature is enabled
    if (isLocalRegisterEnabled()) {
        // When NCCL_LOCAL_REGISTER=1, handle should be non-NULL
        EXPECT_NE(regHandle, nullptr) << "Buffer registration failed: regHandle is NULL "
                                      << "even though NCCL_LOCAL_REGISTER=1";
    } else {
        // When disabled (default), NULL handle is expected behavior
        EXPECT_EQ(regHandle, nullptr) << "Expected NULL handle when NCCL_LOCAL_REGISTER is disabled";
    }

    // Deregister the buffer (works for both NULL and non-NULL handles)
    NCCLCHECK(ncclCommDeregister(comm, regHandle));

    // Clean up resources
    HIPCALL(hipFree(deviceBuffer));
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering multiple buffers
 *
 * Verifies that multiple buffers can be registered simultaneously
 * and each gets a unique, valid registration handle (when enabled).
 */
TEST(Register, MultipleBufferRegistration)
{
    printRegistrationStatus();

    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        GTEST_SKIP() << "This test requires at least 1 GPU device.";
    }

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));

    // Create and register multiple buffers
    const int numBuffers = 4;
    const size_t bufferSize = 64 * 1024; // 64 KB each
    void* deviceBuffers[numBuffers] = {nullptr};
    void* regHandles[numBuffers] = {nullptr};

    const bool regEnabled = isLocalRegisterEnabled();

    for (int i = 0; i < numBuffers; i++) {
        HIPCALL(hipMalloc(&deviceBuffers[i], bufferSize));
        ASSERT_NE(deviceBuffers[i], nullptr) << "Failed to allocate buffer " << i;

        NCCLCHECK(ncclCommRegister(comm, deviceBuffers[i], bufferSize, &regHandles[i]));

        if (regEnabled) {
            // Verify each buffer is registered with a valid handle
            EXPECT_NE(regHandles[i], nullptr) << "Registration failed for buffer " << i;
        }
    }

    // Verify all handles are unique (only when registration is enabled)
    if (regEnabled) {
        for (int i = 0; i < numBuffers; i++) {
            for (int j = i + 1; j < numBuffers; j++) {
                if (regHandles[i] != nullptr && regHandles[j] != nullptr) {
                    EXPECT_NE(regHandles[i], regHandles[j])
                        << "Buffers " << i << " and " << j << " have the same registration handle";
                }
            }
        }
    }

    // Deregister all buffers
    for (int i = 0; i < numBuffers; i++) {
        NCCLCHECK(ncclCommDeregister(comm, regHandles[i]));
    }

    // Clean up
    for (int i = 0; i < numBuffers; i++) {
        HIPCALL(hipFree(deviceBuffers[i]));
    }
    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test registering buffers of different sizes
 *
 * Verifies that buffers of various sizes (4KB to 4MB) can be registered
 * and each registration returns a valid handle (when enabled).
 */
TEST(Register, VariableSizeBuffers)
{
    printRegistrationStatus();

    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        GTEST_SKIP() << "This test requires at least 1 GPU device.";
    }

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));

    const bool regEnabled = isLocalRegisterEnabled();

    // Test various buffer sizes: 4KB, 64KB, 1MB, 4MB
    const size_t sizes[] = {4096, 64 * 1024, 1024 * 1024, 4 * 1024 * 1024};
    const int numSizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < numSizes; i++) {
        void* deviceBuffer = nullptr;
        void* regHandle = nullptr;

        HIPCALL(hipMalloc(&deviceBuffer, sizes[i]));
        ASSERT_NE(deviceBuffer, nullptr) << "Failed to allocate buffer of size " << sizes[i];

        NCCLCHECK(ncclCommRegister(comm, deviceBuffer, sizes[i], &regHandle));

        if (regEnabled) {
            // Verify registration succeeded for this buffer size
            EXPECT_NE(regHandle, nullptr) << "Registration failed for buffer size " << sizes[i] << " bytes";
        }

        NCCLCHECK(ncclCommDeregister(comm, regHandle));
        HIPCALL(hipFree(deviceBuffer));
    }

    NCCLCHECK(ncclCommDestroy(comm));
}

/**
 * @brief Test deregistering NULL handle (should succeed as no-op)
 */
TEST(Register, DeregisterNullHandle)
{
    printRegistrationStatus();

    int numDevices;
    HIPCALL(hipGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        GTEST_SKIP() << "This test requires at least 1 GPU device.";
    }

    HIPCALL(hipSetDevice(0));

    ncclComm_t comm;
    ncclUniqueId id;
    NCCLCHECK(ncclGetUniqueId(&id));
    NCCLCHECK(ncclCommInitRank(&comm, 1, id, 0));

    // Deregister NULL handle - should be a no-op
    NCCLCHECK(ncclCommDeregister(comm, nullptr));

    NCCLCHECK(ncclCommDestroy(comm));
}

} // namespace RcclUnitTesting
