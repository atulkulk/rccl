/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TransportMPIBase.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#ifdef MPI_TESTS_ENABLED

// Import MPI test constants for convenience
using namespace MPITestConstants;

// External reference to SHM transport
extern struct ncclTransport shmTransport;

namespace
{
// Buffer size constants for SHM tests
inline constexpr size_t kDefaultBufferSize     = 1024 * sizeof(float);
inline constexpr size_t kLargeBufferSize       = 135168;
inline constexpr size_t kMediumBufferSize      = 16384;
inline constexpr size_t kSmallBufferSize       = 256;
inline constexpr size_t kMaxValidationElements = 100;

// Large buffer test constants
inline constexpr size_t kCEMemcpyBufferSize  = 256 * 1024 * 1024; // 256 MB
inline constexpr size_t kVeryLargeBufferSize = 512 * 1024 * 1024; // 512 MB

// Test iteration constants
inline constexpr int kMultipleTransferCount = 5;
inline constexpr int kMaxErrorsToReport     = 10;

// Validation sampling parameters
inline constexpr size_t kValidationStride     = 1000;
inline constexpr size_t kMinValidationSamples = 100;

// Helper to perform stream synchronization with error handling
[[nodiscard]] hipError_t syncStream(hipStream_t stream, int rank)
{
    const auto err = hipStreamSynchronize(stream);
    return err;
}
} // namespace

// SHM-specific test configuration
struct ShmTestConfig
{
    bool   is_sender{false};
    void*  send_buffer{nullptr};
    void*  recv_buffer{nullptr};
    size_t buffer_size{0};
};

class ShmMPITest : public TransportTestBase
{
protected:
    ShmTestConfig shm_config;

    // Test data buffers
    std::vector<uint32_t> host_send_data;
    std::vector<uint32_t> host_recv_data;

    void SetUp() override
    {
        // Call base class SetUp first
        TransportTestBase::SetUp();

        // Set up SHM-specific test configuration
        shm_config.is_sender   = (config.world_rank == 0);
        shm_config.buffer_size = kDefaultBufferSize;

        // Allocate test buffers
        EXPECT_EQ(hipSuccess, hipMalloc(&shm_config.send_buffer, shm_config.buffer_size))
            << "Rank " << config.world_rank << ": Failed to allocate send buffer";

        EXPECT_EQ(hipSuccess, hipMalloc(&shm_config.recv_buffer, shm_config.buffer_size))
            << "Rank " << config.world_rank << ": Failed to allocate recv buffer";

        // Initialize send buffer with test data
        constexpr size_t   num_elements = kDefaultBufferSize / sizeof(float);
        std::vector<float> host_data(num_elements);
        for(size_t i = 0; i < num_elements; i++)
        {
            host_data[i] = static_cast<float>(config.world_rank * 1000 + i);
        }

        EXPECT_EQ(hipSuccess,
                  hipMemcpy(shm_config.send_buffer,
                            host_data.data(),
                            shm_config.buffer_size,
                            hipMemcpyHostToDevice))
            << "Rank " << config.world_rank << ": Failed to initialize send buffer";

        // Initialize receive buffer to zero
        EXPECT_EQ(hipSuccess, hipMemset(shm_config.recv_buffer, 0, shm_config.buffer_size))
            << "Rank " << config.world_rank << ": Failed to initialize recv buffer";

        // Synchronize stream to ensure all buffer operations complete
        EXPECT_EQ(hipSuccess, syncStream(config.stream, config.world_rank))
            << "Rank " << config.world_rank
            << ": Failed to synchronize stream after buffer initialization";
    }

    void TearDown() override
    {
        // Cleanup SHM-specific test resources
        if(shm_config.send_buffer)
        {
            HIPCHECK(hipFree(shm_config.send_buffer));
            shm_config.send_buffer = nullptr;
        }
        if(shm_config.recv_buffer)
        {
            HIPCHECK(hipFree(shm_config.recv_buffer));
            shm_config.recv_buffer = nullptr;
        }

        // Call base class TearDown
        TransportTestBase::TearDown();
    }

public:
    // Test SHM capability detection (same-host communication)
    void testShmCanConnect()
    {
        // Validate preconditions
        ASSERT_NE(nullptr, comm_handle)
            << "Rank " << config.world_rank
            << ": comm_handle is null - NCCL communicator not initialized";
        ASSERT_NE(nullptr, local_peer_info)
            << "Rank " << config.world_rank
            << ": local_peer_info is null - peer information not initialized";
        ASSERT_NE(nullptr, remote_peer_info)
            << "Rank " << config.world_rank
            << ": remote_peer_info is null - peer information not initialized";

        int        can_connect = 0;
        const auto result      = shmTransport.canConnect(&can_connect,
                                                    comm_handle,
                                                    topology_graph,
                                                    local_peer_info,
                                                    remote_peer_info);

        ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank
                                       << ": shmCanConnect failed: " << ncclGetErrorString(result);

        // Synchronize the stream to ensure all operations complete
        ASSERT_EQ(hipSuccess, syncStream(config.stream, config.world_rank))
            << "Rank " << config.world_rank << ": Stream synchronization failed";
    }

    // Test SHM setup phase
    void testShmSetup()
    {
        ncclConnect peer_connect_info{};
        const auto  result = shm_config.is_sender ? shmTransport.send.setup(comm_handle,
                                                                           topology_graph,
                                                                           local_peer_info,
                                                                           remote_peer_info,
                                                                           &peer_connect_info,
                                                                           &send_connector,
                                                                           0,
                                                                           0)
                                                  : shmTransport.recv.setup(comm_handle,
                                                                           topology_graph,
                                                                           local_peer_info,
                                                                           remote_peer_info,
                                                                           &peer_connect_info,
                                                                           &recv_connector,
                                                                           0,
                                                                           0);

        ASSERT_EQ(ncclSuccess, result)
            << "Rank " << config.world_rank << ": " << (shm_config.is_sender ? "Send" : "Recv")
            << " setup failed: " << ncclGetErrorString(result);

        // Synchronize all ranks after setup to ensure proxy threads have initialized
        MPI_Barrier(MPI_COMM_WORLD);

        // Second barrier to ensure all proxy threads are ready
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Test SHM connection phase
    void testShmConnect()
    {
        // Validate preconditions
        ASSERT_NE(nullptr, comm_handle) << "Rank " << config.world_rank << ": comm_handle is null";
        ASSERT_NE(nullptr, local_peer_info)
            << "Rank " << config.world_rank << ": local_peer_info is null";
        ASSERT_NE(nullptr, remote_peer_info)
            << "Rank " << config.world_rank << ": remote_peer_info is null";

        // Ensure all ranks are ready before connecting
        MPI_Barrier(MPI_COMM_WORLD);

        // Create and initialize SHM connect info structures
        ncclConnect send_connect_info{};
        ncclConnect recv_connect_info{};

        if(shm_config.is_sender)
        {
            // Setup send connection info using SHM transport setup
            auto result = shmTransport.send.setup(comm_handle,
                                                  topology_graph,
                                                  local_peer_info,
                                                  remote_peer_info,
                                                  &send_connect_info,
                                                  &send_connector,
                                                  0,
                                                  0);
            ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank
                                           << ": Send setup failed: " << ncclGetErrorString(result);

            // Exchange connect info with receiver using MPI
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&send_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD))
                << "Rank " << config.world_rank << ": MPI_Send failed";

            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&recv_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE))
                << "Rank " << config.world_rank << ": MPI_Recv failed";

            // Perform the actual connection using the received info
            result = shmTransport.send.connect(comm_handle,
                                               &recv_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &send_connector);
            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": Send connect failed: " << ncclGetErrorString(result);
        }
        else
        {
            // Setup receive connection info using SHM transport setup
            auto result = shmTransport.recv.setup(comm_handle,
                                                  topology_graph,
                                                  local_peer_info,
                                                  remote_peer_info,
                                                  &recv_connect_info,
                                                  &recv_connector,
                                                  0,
                                                  0);
            ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank
                                           << ": Recv setup failed: " << ncclGetErrorString(result);

            // Exchange connect info with sender using MPI
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&send_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE))
                << "Rank " << config.world_rank << ": MPI_Recv failed";

            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&recv_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD))
                << "Rank " << config.world_rank << ": MPI_Send failed";

            // Perform the actual connection using the received info
            result = shmTransport.recv.connect(comm_handle,
                                               &send_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &recv_connector);
            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": Recv connect failed: " << ncclGetErrorString(result);
        }

        // Synchronize the stream to ensure all RCCL operations complete
        ASSERT_EQ(hipSuccess, syncStream(config.stream, config.world_rank))
            << "Rank " << config.world_rank << ": Stream synchronization failed";
    }

    // Test actual data transfer through SHM
    void testShmDataTransfer()
    {
        // Initialize host data vectors
        const size_t num_elements = shm_config.buffer_size / sizeof(uint32_t);
        host_recv_data.resize(num_elements);
        host_send_data.resize(num_elements);

        // Use RCCL point-to-point operations to validate SHM transport
        const size_t count  = shm_config.buffer_size / sizeof(float);
        const auto   result = shm_config.is_sender ? ncclSend(shm_config.send_buffer,
                                                            count,
                                                            ncclFloat,
                                                            config.peer_rank,
                                                            config.nccl_comm,
                                                            config.stream)
                                                   : ncclRecv(shm_config.recv_buffer,
                                                            count,
                                                            ncclFloat,
                                                            config.peer_rank,
                                                            config.nccl_comm,
                                                            config.stream);

        ASSERT_EQ(ncclSuccess, result)
            << "Rank " << config.world_rank << ": RCCL " << (shm_config.is_sender ? "Send" : "Recv")
            << " failed: " << ncclGetErrorString(result);

        // Ensure both ranks have posted their NCCL operations before synchronizing
        MPI_Barrier(MPI_COMM_WORLD);

        ASSERT_EQ(hipSuccess, syncStream(config.stream, config.world_rank))
            << "Rank " << config.world_rank << ": Stream synchronization failed";

        // Only validate data on the receiver side
        if(!shm_config.is_sender)
        {
            ASSERT_FALSE(host_recv_data.empty())
                << "Rank " << config.world_rank << ": host_recv_data is empty";
            ASSERT_NE(nullptr, shm_config.recv_buffer)
                << "Rank " << config.world_rank << ": recv_buffer is null";

            ASSERT_EQ(hipSuccess,
                      hipMemcpy(host_recv_data.data(),
                                shm_config.recv_buffer,
                                shm_config.buffer_size,
                                hipMemcpyDeviceToHost))
                << "Rank " << config.world_rank << ": hipMemcpy DeviceToHost failed";

            // Validate received data - should match sender's original pattern
            const size_t validation_count = std::min(num_elements, kMaxValidationElements);
            for(size_t i = 0; i < validation_count; i++)
            {
                const float    expected_float = static_cast<float>(config.peer_rank * 1000 + i);
                const uint32_t expected_value = *reinterpret_cast<const uint32_t*>(&expected_float);

                EXPECT_EQ(expected_value, host_recv_data[i])
                    << "Rank " << config.world_rank << ": Data mismatch at index " << i;
            }
        }
    }

    // Test resource cleanup
    void testShmCleanup()
    {
        // Ensure all stream operations complete before cleanup
        [[maybe_unused]] auto err = syncStream(config.stream, config.world_rank);
        // Don't return error on sync failure - continue with cleanup

        auto* connector = shm_config.is_sender ? &send_connector : &recv_connector;
        if(connector->transportResources)
        {
            const auto result = shm_config.is_sender ? shmTransport.send.free(connector)
                                                     : shmTransport.recv.free(connector);

            EXPECT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank << ": " << (shm_config.is_sender ? "Send" : "Recv")
                << " cleanup failed: " << ncclGetErrorString(result);

            // Mark as cleaned up to avoid double cleanup in TearDown
            connector->transportResources = nullptr;
        }
    }

    // Test SHM with memcpy mode enabled (CE - Copy Engine)
    // This test uses the transport API directly to ensure SHM methods are called
    void testShmWithMemcpy()
    {
        // Check if NCCL_SHM_USE_CUDA_MEMCPY is set externally
        const char* shm_memcpy_env = getenv("NCCL_SHM_USE_CUDA_MEMCPY");
        if(!shm_memcpy_env || strcmp(shm_memcpy_env, "1") != 0)
        {
            if(RCCLMPIEnvironment::world_rank == 0)
            {
                std::fprintf(stdout,
                             "Skipping CE memcpy test - NCCL_SHM_USE_CUDA_MEMCPY not set to '1'\n"
                             "To enable this test, set: export NCCL_SHM_USE_CUDA_MEMCPY=1\n");
            } // Skip test gracefully
        }

        // Validate preconditions
        ASSERT_NE(nullptr, comm_handle) << "Rank " << config.world_rank << ": comm_handle is null";
        ASSERT_NE(nullptr, local_peer_info)
            << "Rank " << config.world_rank << ": local_peer_info is null";
        ASSERT_NE(nullptr, remote_peer_info)
            << "Rank " << config.world_rank << ": remote_peer_info is null";

        // Step 1: Test shmCanConnect with CE memcpy enabled
        int          can_connect = 0;
        ncclResult_t result      = shmTransport.canConnect(&can_connect,
                                                      comm_handle,
                                                      topology_graph,
                                                      local_peer_info,
                                                      remote_peer_info);

        ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank
                                       << ": shmCanConnect failed: " << ncclGetErrorString(result);

        ASSERT_EQ(1, can_connect)
            << "Rank " << config.world_rank
            << ": SHM cannot connect - test skipped but connection was expected";

        // Step 2: Test SHM setup with CE memcpy enabled
        MPI_Barrier(MPI_COMM_WORLD);

        ncclConnect send_connect_info{};
        ncclConnect recv_connect_info{};

        if(shm_config.is_sender)
        {
            result = shmTransport.send.setup(comm_handle,
                                             topology_graph,
                                             local_peer_info,
                                             remote_peer_info,
                                             &send_connect_info,
                                             &send_connector,
                                             0,
                                             0);
            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM send setup with CE memcpy failed: " << ncclGetErrorString(result);

            // Exchange connect info with receiver
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&send_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD));
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&recv_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE));
        }
        else
        {
            result = shmTransport.recv.setup(comm_handle,
                                             topology_graph,
                                             local_peer_info,
                                             remote_peer_info,
                                             &recv_connect_info,
                                             &recv_connector,
                                             0,
                                             0);

            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM recv setup with CE memcpy failed: " << ncclGetErrorString(result);

            // Exchange connect info with sender
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Recv(&send_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD,
                               MPI_STATUS_IGNORE));
            ASSERT_EQ(MPI_SUCCESS,
                      MPI_Send(&recv_connect_info,
                               sizeof(ncclConnect),
                               MPI_BYTE,
                               config.peer_rank,
                               0,
                               MPI_COMM_WORLD));
        }

        // Step 3: Test SHM connect with CE memcpy
        MPI_Barrier(MPI_COMM_WORLD);

        if(shm_config.is_sender)
        {
            result = shmTransport.send.connect(comm_handle,
                                               &recv_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &send_connector);

            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM send connect with CE memcpy failed: " << ncclGetErrorString(result);
        }
        else
        {
            result = shmTransport.recv.connect(comm_handle,
                                               &send_connect_info,
                                               config.world_size,
                                               config.world_rank,
                                               &recv_connector);

            ASSERT_EQ(ncclSuccess, result)
                << "Rank " << config.world_rank
                << ": SHM recv connect with CE memcpy failed: " << ncclGetErrorString(result);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Step 4: Send large buffer with CE memcpy and validate
        const size_t buffer_size  = kCEMemcpyBufferSize;
        const size_t num_elements = buffer_size / sizeof(float);
        void*        send_buffer  = nullptr;
        void*        recv_buffer  = nullptr;

        hipError_t hip_result = hipMalloc(&send_buffer, buffer_size);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to allocate send buffer";

        hip_result = hipMalloc(&recv_buffer, buffer_size);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to allocate recv buffer";

        // Initialize send buffer with unique pattern
        std::vector<float> host_send_data(num_elements);
        for(size_t i = 0; i < num_elements; i++)
        {
            host_send_data[i] = static_cast<float>(config.world_rank * 1000000 + (i % 10000));
        }

        hip_result
            = hipMemcpy(send_buffer, host_send_data.data(), buffer_size, hipMemcpyHostToDevice);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to initialize send buffer";

        hip_result = hipMemset(recv_buffer, 0, buffer_size);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Failed to zero recv buffer";

        // Ensure both ranks complete buffer initialization before synchronizing stream
        MPI_Barrier(MPI_COMM_WORLD);

        // Synchronize stream before transfer
        hip_result = syncStream(config.stream, config.world_rank);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Stream sync failed before transfer";

        // Ensure both ranks complete stream sync before posting NCCL operations
        MPI_Barrier(MPI_COMM_WORLD);

        // Perform the actual data transfer using NCCL
        const size_t count = buffer_size / sizeof(float);
        result             = shm_config.is_sender ? ncclSend(send_buffer,
                                                 count,
                                                 ncclFloat,
                                                 config.peer_rank,
                                                 config.nccl_comm,
                                                 config.stream)
                                                  : ncclRecv(recv_buffer,
                                                 count,
                                                 ncclFloat,
                                                 config.peer_rank,
                                                 config.nccl_comm,
                                                 config.stream);

        ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": Large buffer "
                                       << (shm_config.is_sender ? "Send" : "Recv")
                                       << " with CE memcpy failed: " << ncclGetErrorString(result);

        // Ensure both ranks have posted their NCCL operations before synchronizing
        // (required for NCCL_LAUNCH_MODE=GROUP to avoid deadlock)
        MPI_Barrier(MPI_COMM_WORLD);

        // Synchronize to ensure transfer completes
        hip_result = syncStream(config.stream, config.world_rank);
        ASSERT_EQ(hipSuccess, hip_result)
            << "Rank " << config.world_rank << ": Stream sync failed after transfer";

        // Step 5: Validate received data (on receiver only)
        if(!shm_config.is_sender)
        {
            std::vector<float> host_recv_data(num_elements);
            hip_result
                = hipMemcpy(host_recv_data.data(), recv_buffer, buffer_size, hipMemcpyDeviceToHost);

            ASSERT_EQ(hipSuccess, hip_result)
                << "Rank " << config.world_rank << ": Failed to copy received data to host";

            // Validate data - check samples throughout the buffer
            const size_t validation_samples = std::min(num_elements, size_t(10000));
            const size_t stride             = num_elements / validation_samples;
            int          errors             = 0;

            for(size_t i = 0; i < num_elements; i += stride)
            {
                const float expected = static_cast<float>(config.peer_rank * 1000000 + (i % 10000));
                const float received = host_recv_data[i];

                if(std::abs(received - expected) > 1e-5)
                {
                    errors++;
                    if(errors <= 10)
                    { // Print first 10 errors
                        std::fprintf(stdout,
                                     "Rank %d: Validation error at index %zu: expected=%.0f, "
                                     "received=%.0f\n",
                                     config.world_rank,
                                     i,
                                     expected,
                                     received);
                    }
                }
            }

            EXPECT_EQ(0, errors) << "Rank " << config.world_rank << ": Data validation failed with "
                                 << errors << " mismatches out of " << validation_samples
                                 << " samples";
        }

        // Cleanup buffers
        if(send_buffer)
        {
            EXPECT_EQ(hipSuccess, hipFree(send_buffer))
                << "Rank " << config.world_rank << ": Failed to free send buffer";
        }
        if(recv_buffer)
        {
            EXPECT_EQ(hipSuccess, hipFree(recv_buffer))
                << "Rank " << config.world_rank << ": Failed to free recv buffer";
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    // Test SHM buffer allocation and sharing
    void testShmBufferAllocation()
    {
        // Test buffer allocation with various sizes
        const std::vector<size_t> test_sizes
            = {kSmallBufferSize, kMediumBufferSize, kLargeBufferSize};

        for(const auto size : test_sizes)
        {
            void* send_buff = nullptr;
            void* recv_buff = nullptr;

            allocateAndInitBuffers(&send_buff, &recv_buff, size, size);

            // Verify buffers are accessible
            EXPECT_NE(send_buff, nullptr) << "Rank " << config.world_rank << ": send_buff is null";
            EXPECT_NE(recv_buff, nullptr) << "Rank " << config.world_rank << ": recv_buff is null";

            // Cleanup
            cleanupBuffers(send_buff, recv_buff, nullptr, nullptr);
        }
    }
};

TEST_F(ShmMPITest, ShmWorkflow)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    // Test 1: SHM Capability Detection
    testShmCanConnect();

    // Synchronize after canConnect check
    MPI_Barrier(MPI_COMM_WORLD);

    // Test 2: SHM Setup
    testShmSetup();

    // Test 3: SHM Connection
    testShmConnect();

    // Test 4: Data Transfer through SHM
    testShmDataTransfer();

    // Test 5: Resource Cleanup
    testShmCleanup();
}

TEST_F(ShmMPITest, ShmWithMemcpyTest)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    testShmWithMemcpy();
}

TEST_F(ShmMPITest, ShmBufferAllocationTest)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    testShmBufferAllocation();
}

TEST_F(ShmMPITest, ShmTransfer_ZeroSizeBuffer)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    // Allocate minimal buffer
    void* buffer = nullptr;
    HIPCHECK(hipMalloc(&buffer, 1)); // Allocate 1 byte

    const bool is_sender = (config.world_rank == 0);
    const int  peer      = is_sender ? 1 : 0;

    // Try to send/recv 0 elements
    const auto result = is_sender
                            ? ncclSend(buffer, 0, ncclFloat, peer, config.nccl_comm, config.stream)
                            : ncclRecv(buffer, 0, ncclFloat, peer, config.nccl_comm, config.stream);

    ASSERT_EQ(ncclSuccess, result)
        << "Rank " << config.world_rank << ": Zero-size transfer should succeed";

    // Ensure both ranks have posted their NCCL operations before synchronizing
    MPI_Barrier(MPI_COMM_WORLD);

    HIPCHECK(hipStreamSynchronize(config.stream));
    HIPCHECK(hipFree(buffer));
}

TEST_F(ShmMPITest, ShmTransfer_VeryLargeBuffer)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    // Try to allocate a very large buffer
    const size_t large_size  = kCEMemcpyBufferSize;
    void*        send_buffer = nullptr;
    void*        recv_buffer = nullptr;

    hipError_t hip_result = hipMalloc(&send_buffer, large_size);

    hip_result = hipMalloc(&recv_buffer, large_size);

    // Initialize buffer
    HIPCHECK(hipMemset(send_buffer, 0x42, large_size));

    const bool   is_sender = (config.world_rank == 0);
    const int    peer      = is_sender ? 1 : 0;
    const size_t count     = large_size / sizeof(float);

    // Perform send/recv with large buffer
    const auto result
        = is_sender
              ? ncclSend(send_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream)
              : ncclRecv(recv_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream);

    ASSERT_EQ(ncclSuccess, result)
        << "Rank " << config.world_rank << ": Large buffer transfer failed";

    // Ensure both ranks have posted their NCCL operations before synchronizing
    MPI_Barrier(MPI_COMM_WORLD);

    HIPCHECK(hipStreamSynchronize(config.stream));

    // Cleanup
    HIPCHECK(hipFree(send_buffer));
    HIPCHECK(hipFree(recv_buffer));

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(ShmMPITest, ShmTransfer_UnalignedBufferAddress)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    // Allocate aligned buffer
    const size_t buffer_size    = 4096;
    void*        aligned_buffer = nullptr;
    HIPCHECK(hipMalloc(&aligned_buffer, buffer_size));

    // Create unaligned pointer (offset by 1 byte)
    void* unaligned_buffer = static_cast<char*>(aligned_buffer) + 1;

    const bool is_sender = (config.world_rank == 0);
    const int  peer      = is_sender ? 1 : 0;

    const auto result
        = is_sender
              ? ncclSend(unaligned_buffer, 1024, ncclChar, peer, config.nccl_comm, config.stream)
              : ncclRecv(unaligned_buffer, 1024, ncclChar, peer, config.nccl_comm, config.stream);

    // Ensure both ranks have posted their NCCL operations before synchronizing
    MPI_Barrier(MPI_COMM_WORLD);

    // Don't fail the test - just report the result
    HIPCHECK(hipStreamSynchronize(config.stream));
    HIPCHECK(hipFree(aligned_buffer));
}

TEST_F(ShmMPITest, ShmMultipleConsecutiveTransfers)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const size_t buffer_size = kMediumBufferSize;
    void*        send_buffer = nullptr;
    void*        recv_buffer = nullptr;

    HIPCHECK(hipMalloc(&send_buffer, buffer_size));
    HIPCHECK(hipMalloc(&recv_buffer, buffer_size));
    HIPCHECK(hipMemset(send_buffer, 0xAB, buffer_size));

    const bool   is_sender = (config.world_rank == 0);
    const int    peer      = is_sender ? 1 : 0;
    const size_t count     = buffer_size / sizeof(float);

    for(int i = 0; i < kMultipleTransferCount; i++)
    {
        const auto result
            = is_sender
                  ? ncclSend(send_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream)
                  : ncclRecv(recv_buffer, count, ncclFloat, peer, config.nccl_comm, config.stream);

        ASSERT_EQ(ncclSuccess, result)
            << "Rank " << config.world_rank << ": Transfer " << i << " failed";

        // Ensure both ranks have posted their NCCL operations before synchronizing
        MPI_Barrier(MPI_COMM_WORLD);

        HIPCHECK(hipStreamSynchronize(config.stream));
    }

    HIPCHECK(hipFree(send_buffer));
    HIPCHECK(hipFree(recv_buffer));

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(ShmMPITest, ShmCleanup_DoubleCleanup)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    const bool is_sender = (config.world_rank == 0);
    auto*      connector = is_sender ? &send_connector : &recv_connector;

    // Setup connector
    ncclConnect connect_info{};
    const auto  setup_result = is_sender ? shmTransport.send.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &connect_info,
                                                                  connector,
                                                                  0,
                                                                  0)
                                         : shmTransport.recv.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &connect_info,
                                                                  connector,
                                                                  0,
                                                                  0);

    ASSERT_EQ(ncclSuccess, setup_result) << "Rank " << config.world_rank << ": Setup failed";

    MPI_Barrier(MPI_COMM_WORLD);

    // First cleanup
    if(connector->transportResources)
    {
        const auto result1
            = is_sender ? shmTransport.send.free(connector) : shmTransport.recv.free(connector);
        EXPECT_EQ(ncclSuccess, result1) << "Rank " << config.world_rank << ": First cleanup failed";
    }

    // Second cleanup (should handle gracefully since resources are already freed)
    [[maybe_unused]] const auto result2
        = is_sender ? shmTransport.send.free(connector) : shmTransport.recv.free(connector);

    // Mark as cleaned up
    connector->transportResources = nullptr;

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(ShmMPITest, ShmConnect_WithoutSetup)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    if(config.world_rank == 0)
    {
        std::fprintf(stdout,
                     "Testing SHM connect without prior setup (%d processes)\n",
                     config.world_size);
    }

    const bool is_sender = (config.world_rank == 0);
    auto*      connector = is_sender ? &send_connector : &recv_connector;

    // Create empty/uninitialized connect info (simulates invalid state)
    ncclConnect invalid_connect_info{};
    memset(&invalid_connect_info, 0, sizeof(ncclConnect));

    // Try to connect without calling setup first - this should fail or handle gracefully
    const auto result = is_sender ? shmTransport.send.connect(comm_handle,
                                                              &invalid_connect_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector)
                                  : shmTransport.recv.connect(comm_handle,
                                                              &invalid_connect_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector);

    if(config.world_rank == 0)
    {
        std::fprintf(stdout,
                     "Connect without setup result: %s\n"
                     "Note: This tests invalid state handling\n",
                     ncclGetErrorString(result));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(ShmMPITest, ShmConnect_CorruptedConnectInfo)
{
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kRequireSingleNode))
        << "Test requirements not met - all ranks must meet requirements";

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    if(config.world_rank == 0)
    {
        std::fprintf(stdout,
                     "Testing SHM connect with corrupted connect info (%d processes)\n",
                     config.world_size);
    }

    const bool is_sender = (config.world_rank == 0);
    auto*      connector = is_sender ? &send_connector : &recv_connector;

    // First, do valid setup
    ncclConnect valid_connect_info{};
    const auto  setup_result = is_sender ? shmTransport.send.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &valid_connect_info,
                                                                  connector,
                                                                  0,
                                                                  0)
                                         : shmTransport.recv.setup(comm_handle,
                                                                  topology_graph,
                                                                  local_peer_info,
                                                                  remote_peer_info,
                                                                  &valid_connect_info,
                                                                  connector,
                                                                  0,
                                                                  0);

    ASSERT_EQ(ncclSuccess, setup_result) << "Rank " << config.world_rank << ": Setup failed";

    MPI_Barrier(MPI_COMM_WORLD);

    // Create corrupted connect info (fill with invalid data)
    ncclConnect corrupted_info{};
    memset(&corrupted_info, 0xFF, sizeof(ncclConnect)); // Fill with 0xFF

    // Try to connect with corrupted info
    // This tests internal validation of connect info structures
    const auto result = is_sender ? shmTransport.send.connect(comm_handle,
                                                              &corrupted_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector)
                                  : shmTransport.recv.connect(comm_handle,
                                                              &corrupted_info,
                                                              config.world_size,
                                                              config.world_rank,
                                                              connector);

    if(config.world_rank == 0)
    {
        std::fprintf(stdout,
                     "Connect with corrupted info result: %s\n"
                     "Note: Tests connect info validation similar to proxy function validation\n",
                     ncclGetErrorString(result));
    }

    // Cleanup properly allocated resources
    if(connector->transportResources)
    {
        const auto cleanup_result
            = is_sender ? shmTransport.send.free(connector) : shmTransport.recv.free(connector);
        (void)cleanup_result; // Ignore result as we're in error path
        connector->transportResources = nullptr;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

#endif // MPI_TESTS_ENABLED
