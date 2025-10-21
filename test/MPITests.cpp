/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "rccl/rccl.h"
#include <cassert>
#include <cstring>
#include <numeric>

#ifdef MPI_TESTS_ENABLED
#include <mpi.h>
#include "common/RCCLMPIEnvironment.hpp"
#include "common/MPITestBase.hpp"

// Import constants for convenience
using namespace MPITestConstants;

class UnifiedMPITest : public MPITestBase {
protected:
    static bool initialized;

    // Simple test configuration for send/recv tests
    struct SimpleMPIConfig {
        int world_rank;
        int world_size;
        int peer_rank;
        ncclComm_t rccl_comm;
        void* send_buffer;
        void* recv_buffer;
        size_t buffer_size;
        hipStream_t stream;
    };

    SimpleMPIConfig config;
    std::vector<uint32_t> host_send_data;
    std::vector<uint32_t> host_recv_data;

    static void SetUpTestCase() {
        if (initialized) return;

        // Check if RCCLMPIEnvironment was properly initialized
        if (RCCLMPIEnvironment::retCode != 0) {
            GTEST_FAIL() << "RCCLMPIEnvironment initialization failed";
        }

        initialized = true;
        if (RCCLMPIEnvironment::world_rank == 0) {
            printf("Unified MPI Test: Using RCCLMPIEnvironment - Rank %d of %d\n",
                   RCCLMPIEnvironment::world_rank, RCCLMPIEnvironment::world_size);
        }
    }

    static void TearDownTestCase() {
        if (!initialized) return;

        initialized = false;
    }

    void SetUp() override {
        // Each test gets fresh setup
        if (!initialized) {
            SetUpTestCase();
        }

        // Initialize configuration using RCCLMPIEnvironment
        memset(&config, 0, sizeof(config));
        config.world_rank = RCCLMPIEnvironment::world_rank;
        config.world_size = RCCLMPIEnvironment::world_size;
        config.rccl_comm = getActiveCommunicator();
        config.stream = getActiveStream();

        // Require at least 2 MPI processes for send/recv test
        if (config.world_size < 2) {
            GTEST_SKIP() << "This test requires at least 2 MPI processes, got " << config.world_size;
        }

        // Check if RCCLMPIEnvironment was properly initialized
        if (RCCLMPIEnvironment::retCode != 0) {
            GTEST_SKIP() << "RCCLMPIEnvironment initialization failed - insufficient GPUs or other error";
        }
    }

    void TearDown() override {
        // Cleanup is handled by TearDownTestCase
    }
};

// Static member definitions
bool UnifiedMPITest::initialized = false;

// =================== TEST CASES ===================

// Unified MPI Tests - AllReduce Tests
TEST_F(UnifiedMPITest, BasicAllReduce) {
    validateTestPrerequisites(kMinProcessesForMPI, kNoPowerOfTwoRequired);

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const int N = 1024;
    std::vector<float> send_data(N, RCCLMPIEnvironment::world_rank + 1.0f);
    std::vector<float> recv_data(N);

    // Allocate GPU memory
    float *d_send, *d_recv;
    HIPCHECK(hipMalloc(&d_send, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_recv, N * sizeof(float)));

    // Copy data to GPU
    HIPCHECK(hipMemcpy(d_send, send_data.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Perform AllReduce
    NCCLCHECK(ncclAllReduce(d_send, d_recv, N, ncclFloat, ncclSum, getActiveCommunicator(), getActiveStream()));

    // Copy result back
    HIPCHECK(hipMemcpy(recv_data.data(), d_recv, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipStreamSynchronize(getActiveStream()));

    // Verify result
    float expected_sum = 0;
    for (int i = 0; i < RCCLMPIEnvironment::world_size; i++) {
        expected_sum += (i + 1.0f);
    }

    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(recv_data[i], expected_sum) << "Mismatch at index " << i;
    }

    // Cleanup
    HIPCHECK(hipFree(d_send));
    HIPCHECK(hipFree(d_recv));
}

TEST_F(UnifiedMPITest, AllReduceWithDifferentSizes) {
    validateTestPrerequisites(kMinProcessesForMPI, kNoPowerOfTwoRequired);

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    std::vector<int> sizes = {1, 10, 100, 1000, 10000};

    for (int N : sizes) {
        std::vector<int> send_data(N, RCCLMPIEnvironment::world_rank);
        std::vector<int> recv_data(N);

        // Allocate GPU memory
        int *d_send, *d_recv;
        HIPCHECK(hipMalloc(&d_send, N * sizeof(int)));
        HIPCHECK(hipMalloc(&d_recv, N * sizeof(int)));

        // Copy data to GPU
        HIPCHECK(hipMemcpy(d_send, send_data.data(), N * sizeof(int), hipMemcpyHostToDevice));

        // Perform AllReduce
        NCCLCHECK(ncclAllReduce(d_send, d_recv, N, ncclInt, ncclSum, getActiveCommunicator(), getActiveStream()));

        // Copy result back
        HIPCHECK(hipMemcpy(recv_data.data(), d_recv, N * sizeof(int), hipMemcpyDeviceToHost));
        HIPCHECK(hipStreamSynchronize(getActiveStream()));

        // Verify result
        int expected_sum = (RCCLMPIEnvironment::world_size * (RCCLMPIEnvironment::world_size - 1)) / 2;
        for (int i = 0; i < N; i++) {
            EXPECT_EQ(recv_data[i], expected_sum) << "Mismatch at size " << N << ", index " << i;
        }

        // Cleanup
        HIPCHECK(hipFree(d_send));
        HIPCHECK(hipFree(d_recv));
    }
}

TEST_F(UnifiedMPITest, Broadcast) {
    validateTestPrerequisites(kMinProcessesForMPI, kNoPowerOfTwoRequired);

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const int N = 1000;
    std::vector<float> data(N);

    // Initialize data on rank 0
    if (RCCLMPIEnvironment::world_rank == 0) {
        std::iota(data.begin(), data.end(), 1.0f);
    }

    // Allocate GPU memory
    float *d_data;
    HIPCHECK(hipMalloc(&d_data, N * sizeof(float)));

    // Copy data to GPU
    HIPCHECK(hipMemcpy(d_data, data.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // Perform Broadcast
    NCCLCHECK(ncclBroadcast(d_data, d_data, N, ncclFloat, 0, getActiveCommunicator(), getActiveStream()));

    // Copy result back
    HIPCHECK(hipMemcpy(data.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipStreamSynchronize(getActiveStream()));

    // Verify result
    if (RCCLMPIEnvironment::world_rank == 0) {
        for (int i = 0; i < N; i++) {
            EXPECT_FLOAT_EQ(data[i], i + 1.0f) << "Mismatch at index " << i;
        }
    } else {
        // Other ranks should have received the broadcast data
        for (int i = 0; i < N; i++) {
            EXPECT_FLOAT_EQ(data[i], i + 1.0f) << "Mismatch at index " << i;
        }
    }

    // Cleanup
    HIPCHECK(hipFree(d_data));
}

// Unified MPI Tests - Send/Recv Tests
TEST_F(UnifiedMPITest, SimpleSendRecv) {
    validateTestPrerequisites(kMinProcessesForMPI, kNoPowerOfTwoRequired);

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const int N = 1024;
    const int peer_rank = 1 - RCCLMPIEnvironment::world_rank;  // 0 <-> 1

    // Allocate and initialize data
    float* d_send_buff;
    float* d_recv_buff;
    float* h_send_buff = new float[N];
    float* h_recv_buff = new float[N];

    // Initialize host data - each rank sends different values
    for (int i = 0; i < N; i++) {
        h_send_buff[i] = static_cast<float>(RCCLMPIEnvironment::world_rank * 1000 + i);
    }

    // Allocate device memory
    HIPCHECK(hipMalloc(&d_send_buff, N * sizeof(float)));
    HIPCHECK(hipMalloc(&d_recv_buff, N * sizeof(float)));

    // Copy data to device
    HIPCHECK(hipMemcpy(d_send_buff, h_send_buff, N * sizeof(float), hipMemcpyHostToDevice));

    // Initialize receive buffer to zero
    HIPCHECK(hipMemset(d_recv_buff, 0, N * sizeof(float)));

    // Perform Send/Recv operations
    if (RCCLMPIEnvironment::world_rank == 0) {
        // Rank 0 sends to rank 1
        NCCLCHECK(ncclSend(d_send_buff, N, ncclFloat, 1, getActiveCommunicator(), getActiveStream()));
        // Rank 0 receives from rank 1
        NCCLCHECK(ncclRecv(d_recv_buff, N, ncclFloat, 1, getActiveCommunicator(), getActiveStream()));
    } else {
        // Rank 1 receives from rank 0
        NCCLCHECK(ncclRecv(d_recv_buff, N, ncclFloat, 0, getActiveCommunicator(), getActiveStream()));
        // Rank 1 sends to rank 0
        NCCLCHECK(ncclSend(d_send_buff, N, ncclFloat, 0, getActiveCommunicator(), getActiveStream()));
    }

    // Wait for operations to complete
    HIPCHECK(hipStreamSynchronize(getActiveStream()));

    // Copy result back to host
    HIPCHECK(hipMemcpy(h_recv_buff, d_recv_buff, N * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results - each rank should receive data from its peer
    for (int i = 0; i < N; i++) {
        float expected_value = static_cast<float>(peer_rank * 1000 + i);
        EXPECT_FLOAT_EQ(h_recv_buff[i], expected_value)
            << "Element " << i << " mismatch. Expected: " << expected_value
            << ", Got: " << h_recv_buff[i];
    }

    // Cleanup
    HIPCHECK(hipFree(d_send_buff));
    HIPCHECK(hipFree(d_recv_buff));
    delete[] h_send_buff;
    delete[] h_recv_buff;
}

TEST_F(UnifiedMPITest, AllReduceSum) {
    // Skip test if RCCLMPIEnvironment initialization failed
    if (RCCLMPIEnvironment::retCode != 0) {
        GTEST_SKIP() << "RCCLMPIEnvironment initialization failed - insufficient GPUs or other error";
    }

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    int rank = RCCLMPIEnvironment::world_rank;
    int size = RCCLMPIEnvironment::world_size;
    ncclComm_t rccl_comm = getActiveCommunicator();
    hipStream_t stream = getActiveStream();

    int send_val = 1, recv_val = 0;
    int *d_send, *d_recv;

    HIPCHECK(hipMalloc(&d_send, sizeof(int)));
    HIPCHECK(hipMalloc(&d_recv, sizeof(int)));
    HIPCHECK(hipMemcpy(d_send, &send_val, sizeof(int), hipMemcpyHostToDevice));
    NCCLCHECK(ncclAllReduce(d_send, d_recv, 1, ncclInt, ncclSum, rccl_comm, getActiveStream()));
    HIPCHECK(hipStreamSynchronize(getActiveStream()));
    HIPCHECK(hipMemcpy(&recv_val, d_recv, sizeof(int), hipMemcpyDeviceToHost));
    HIPCHECK(hipStreamSynchronize(getActiveStream()));

    if (rank == 0) {
        EXPECT_EQ(recv_val, size);
    }

    HIPCHECK(hipFree(d_send));
    HIPCHECK(hipFree(d_recv));
}

// Unified MPI Tests - Advanced AllReduce Tests
TEST_F(UnifiedMPITest, BasicAllReduceSum) {
    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const int num_elements = 1024;

    // Allocate and initialize data
    float* d_send_buff;
    float* d_recv_buff;
    float* h_send_buff = new float[num_elements];
    float* h_recv_buff = new float[num_elements];

    // Initialize host data - each rank sends rank+1
    for (int i = 0; i < num_elements; i++) {
        h_send_buff[i] = static_cast<float>(RCCLMPIEnvironment::world_rank + 1);
    }

    // Allocate device memory
    HIPCHECK(hipMalloc(&d_send_buff, num_elements * sizeof(float)));
    HIPCHECK(hipMalloc(&d_recv_buff, num_elements * sizeof(float)));

    // Copy data to device
    HIPCHECK(hipMemcpy(d_send_buff, h_send_buff, num_elements * sizeof(float), hipMemcpyHostToDevice));

    // Perform AllReduce
    NCCLCHECK(ncclAllReduce(d_send_buff, d_recv_buff, num_elements, ncclFloat, ncclSum, getActiveCommunicator(), getActiveStream()));
    HIPCHECK(hipStreamSynchronize(getActiveStream()));

    // Copy result back to host
    HIPCHECK(hipMemcpy(h_recv_buff, d_recv_buff, num_elements * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results
    float expected_sum = static_cast<float>(RCCLMPIEnvironment::world_size * (RCCLMPIEnvironment::world_size + 1) / 2); // Sum of 1+2+...+n

    for (int i = 0; i < num_elements; i++) {
        EXPECT_NEAR(h_recv_buff[i], expected_sum, 1e-6)
            << "Element " << i << " mismatch. Expected: " << expected_sum
            << ", Got: " << h_recv_buff[i];
    }

    // Cleanup
    HIPCHECK(hipFree(d_send_buff));
    HIPCHECK(hipFree(d_recv_buff));
    delete[] h_send_buff;
    delete[] h_recv_buff;
}

TEST_F(UnifiedMPITest, AllReduceWithDifferentSizesAdvanced) {
    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    std::vector<int> test_sizes = {1, 10, 100, 1000, 10000};

    for (int num_elements : test_sizes) {
        // Allocate and initialize data
        float* d_send_buff;
        float* d_recv_buff;
        float* h_send_buff = new float[num_elements];
        float* h_recv_buff = new float[num_elements];

        // Initialize host data
        for (int i = 0; i < num_elements; i++) {
            h_send_buff[i] = static_cast<float>(RCCLMPIEnvironment::world_rank);
        }

        // Allocate device memory
        HIPCHECK(hipMalloc(&d_send_buff, num_elements * sizeof(float)));
        HIPCHECK(hipMalloc(&d_recv_buff, num_elements * sizeof(float)));

        // Copy data to device
        HIPCHECK(hipMemcpy(d_send_buff, h_send_buff, num_elements * sizeof(float), hipMemcpyHostToDevice));

        // Perform AllReduce
        NCCLCHECK(ncclAllReduce(d_send_buff, d_recv_buff, num_elements, ncclFloat, ncclSum, getActiveCommunicator(), getActiveStream()));
        HIPCHECK(hipStreamSynchronize(getActiveStream()));

        // Copy result back to host
        HIPCHECK(hipMemcpy(h_recv_buff, d_recv_buff, num_elements * sizeof(float), hipMemcpyDeviceToHost));

        // Verify results
        float expected_sum = static_cast<float>(RCCLMPIEnvironment::world_size * (RCCLMPIEnvironment::world_size - 1) / 2); // Sum of 0+1+...+(n-1)

        for (int i = 0; i < num_elements; i++) {
            EXPECT_NEAR(h_recv_buff[i], expected_sum, 1e-6)
                << "Size " << num_elements << ", Element " << i << " mismatch. Expected: " << expected_sum
                << ", Got: " << h_recv_buff[i];
        }

        // Cleanup
        HIPCHECK(hipFree(d_send_buff));
        HIPCHECK(hipFree(d_recv_buff));
        delete[] h_send_buff;
        delete[] h_recv_buff;
    }
}

TEST_F(UnifiedMPITest, AllReduceMaxOperation) {
    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const int num_elements = 512;

    // Allocate and initialize data
    float* d_send_buff;
    float* d_recv_buff;
    float* h_send_buff = new float[num_elements];
    float* h_recv_buff = new float[num_elements];

    // Initialize host data - each rank sends different values
    for (int i = 0; i < num_elements; i++) {
        h_send_buff[i] = static_cast<float>(RCCLMPIEnvironment::world_rank * 10 + i);
    }

    // Allocate device memory
    HIPCHECK(hipMalloc(&d_send_buff, num_elements * sizeof(float)));
    HIPCHECK(hipMalloc(&d_recv_buff, num_elements * sizeof(float)));

    // Copy data to device
    HIPCHECK(hipMemcpy(d_send_buff, h_send_buff, num_elements * sizeof(float), hipMemcpyHostToDevice));

    // Perform AllReduce with MAX operation
    NCCLCHECK(ncclAllReduce(d_send_buff, d_recv_buff, num_elements, ncclFloat, ncclMax, getActiveCommunicator(), getActiveStream()));
    HIPCHECK(hipStreamSynchronize(getActiveStream()));

    // Copy result back to host
    HIPCHECK(hipMemcpy(h_recv_buff, d_recv_buff, num_elements * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results - should be the maximum value across all ranks
    for (int i = 0; i < num_elements; i++) {
            float expected_max = static_cast<float>((RCCLMPIEnvironment::world_size - 1) * 10 + i);
        EXPECT_NEAR(h_recv_buff[i], expected_max, 1e-6)
            << "Element " << i << " mismatch. Expected: " << expected_max
            << ", Got: " << h_recv_buff[i];
    }

    // Cleanup
    HIPCHECK(hipFree(d_send_buff));
    HIPCHECK(hipFree(d_recv_buff));
    delete[] h_send_buff;
    delete[] h_recv_buff;
}

TEST_F(UnifiedMPITest, AllReduceMinOperation) {
    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const int num_elements = 256;

    // Allocate and initialize data
    float* d_send_buff;
    float* d_recv_buff;
    float* h_send_buff = new float[num_elements];
    float* h_recv_buff = new float[num_elements];

    // Initialize host data - each rank sends different values
    for (int i = 0; i < num_elements; i++) {
        h_send_buff[i] = static_cast<float>(RCCLMPIEnvironment::world_rank * 100 + i);
    }

    // Allocate device memory
    HIPCHECK(hipMalloc(&d_send_buff, num_elements * sizeof(float)));
    HIPCHECK(hipMalloc(&d_recv_buff, num_elements * sizeof(float)));

    // Copy data to device
    HIPCHECK(hipMemcpy(d_send_buff, h_send_buff, num_elements * sizeof(float), hipMemcpyHostToDevice));

    // Perform AllReduce with MIN operation
    NCCLCHECK(ncclAllReduce(d_send_buff, d_recv_buff, num_elements, ncclFloat, ncclMin, getActiveCommunicator(), getActiveStream()));
    HIPCHECK(hipStreamSynchronize(getActiveStream()));

    // Copy result back to host
    HIPCHECK(hipMemcpy(h_recv_buff, d_recv_buff, num_elements * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results - should be the minimum value across all ranks
    for (int i = 0; i < num_elements; i++) {
        float expected_min = static_cast<float>(i); // Rank 0 has the minimum values
        EXPECT_NEAR(h_recv_buff[i], expected_min, 1e-6)
            << "Element " << i << " mismatch. Expected: " << expected_min
            << ", Got: " << h_recv_buff[i];
    }

    // Cleanup
    HIPCHECK(hipFree(d_send_buff));
    HIPCHECK(hipFree(d_recv_buff));
    delete[] h_send_buff;
    delete[] h_recv_buff;
}

TEST_F(UnifiedMPITest, AllReduceWithDifferentDataTypes) {
    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const int num_elements = 128;

    // Test with int32
    {
        int* d_send_buff;
        int* d_recv_buff;
        int* h_send_buff = new int[num_elements];
        int* h_recv_buff = new int[num_elements];

        for (int i = 0; i < num_elements; i++) {
            h_send_buff[i] = RCCLMPIEnvironment::world_rank + 1;
        }

        HIPCHECK(hipMalloc(&d_send_buff, num_elements * sizeof(int)));
        HIPCHECK(hipMalloc(&d_recv_buff, num_elements * sizeof(int)));
        HIPCHECK(hipMemcpy(d_send_buff, h_send_buff, num_elements * sizeof(int), hipMemcpyHostToDevice));

        NCCLCHECK(ncclAllReduce(d_send_buff, d_recv_buff, num_elements, ncclInt, ncclSum, getActiveCommunicator(), getActiveStream()));
        HIPCHECK(hipStreamSynchronize(getActiveStream()));

        HIPCHECK(hipMemcpy(h_recv_buff, d_recv_buff, num_elements * sizeof(int), hipMemcpyDeviceToHost));

        int expected_sum = RCCLMPIEnvironment::world_size * (RCCLMPIEnvironment::world_size + 1) / 2;
        for (int i = 0; i < num_elements; i++) {
            EXPECT_EQ(h_recv_buff[i], expected_sum) << "Int32 element " << i << " mismatch";
        }

        HIPCHECK(hipFree(d_send_buff));
        HIPCHECK(hipFree(d_recv_buff));
        delete[] h_send_buff;
        delete[] h_recv_buff;
    }
}

#endif // MPI_TESTS_ENABLED
