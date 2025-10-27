/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TransportMPIBase.hpp"

#ifdef MPI_TESTS_ENABLED

// Import MPI test constants
using namespace MPITestConstants;

namespace
{
// Buffer size constants
inline constexpr size_t kTestBufferSize = 16384;

// NET transport test requirements
inline constexpr int kMinNodesForNET   = 2; // NET transport requires at least 2 nodes
inline constexpr int kExactRanksForNET = 2; // NET transport tests use exactly 2 ranks (1 per node)
} // namespace

class NetTransportMPITest : public TransportTestBase
{
protected:
    void SetUp() override
    {
        TransportTestBase::SetUp();
        if(config.world_rank == 0)
        {
            printf("Rank %d: NetTransport SetUp completed\n", config.world_rank);
        }
    }

    void TearDown() override
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: NetTransport TearDown completed\n", config.world_rank);
        }
        TransportTestBase::TearDown();
    }

public:
    // Test ncclNetGraphRegisterBuffer
    void testNetGraphRegisterBuffer()
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: Testing ncclNetGraphRegisterBuffer...\n", config.world_rank);
        }

        // Verify communicator is ready
        ASSERT_NE(comm_handle, nullptr) << "Rank " << config.world_rank << ": comm_handle is null";

        // Allocate and automatically guard buffers
        void* send_buffer = nullptr;
        void* recv_buffer = nullptr;
        allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, kTestBufferSize, kTestBufferSize);

        // Register and automatically guard handles
        void* send_reg_handle = nullptr;
        void* recv_reg_handle = nullptr;
        preRegisterBuffersGuarded(send_buffer,
                                  recv_buffer,
                                  kTestBufferSize,
                                  kTestBufferSize,
                                  &send_reg_handle,
                                  &recv_reg_handle);

        // Test ncclNetGraphRegisterBuffer
        int                                                       net_reg_flag{};
        void*                                                     net_handle{};
        ncclIntruQueue<ncclCommCallback, &ncclCommCallback::next> cleanup_queue{};
        int                                                       n_cleanup_elts{};

        ncclConnector* send_conn_array[1] = {&send_connector};

        auto nccl_result
            = ncclNetGraphRegisterBuffer(reinterpret_cast<ncclComm*>(getActiveCommunicator()),
                                         send_buffer,
                                         kTestBufferSize,
                                         send_conn_array,
                                         1,
                                         &net_reg_flag,
                                         &net_handle,
                                         &cleanup_queue,
                                         &n_cleanup_elts);

        EXPECT_EQ(ncclSuccess, nccl_result)
            << "Rank " << config.world_rank
            << ": ncclNetGraphRegisterBuffer failed: " << ncclGetErrorString(nccl_result);

        if(config.world_rank == 0)
        {
            printf("Rank %d:     ncclNetGraphRegisterBuffer returned: %s\n"
                   "Rank %d:     Registration flag: %d\n"
                   "Rank %d:     Handle: %p\n"
                   "Rank %d:     Cleanup queue elements: %d\n",
                   config.world_rank,
                   ncclGetErrorString(nccl_result),
                   config.world_rank,
                   net_reg_flag,
                   config.world_rank,
                   net_handle,
                   config.world_rank,
                   n_cleanup_elts);
        }

        if(config.world_rank == 0)
        {
            printf("Rank %d: ncclNetGraphRegisterBuffer test completed\n", config.world_rank);
        }
    }

    // Test ncclNetLocalRegisterBuffer
    void testNetLocalRegisterBuffer()
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: Testing ncclNetLocalRegisterBuffer...\n"
                   "Rank %d: This API internally calls ncclNetLocalRegisterBuffer "
                   "and ncclNetLocalRegisterBuffer\n",
                   config.world_rank,
                   config.world_rank);
        }

        // Verify communicator is ready (NCCL has already initialized NET transport)
        ASSERT_NE(comm_handle, nullptr) << "Rank " << config.world_rank << ": comm_handle is null";

        // Allocate and automatically guard buffers
        void* send_buffer = nullptr;
        void* recv_buffer = nullptr;
        allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, kTestBufferSize, kTestBufferSize);

        // Register and automatically guard handles
        void* send_reg_handle = nullptr;
        void* recv_reg_handle = nullptr;
        preRegisterBuffersGuarded(send_buffer,
                                  recv_buffer,
                                  kTestBufferSize,
                                  kTestBufferSize,
                                  &send_reg_handle,
                                  &recv_reg_handle);

        // Test ncclNetLocalRegisterBuffer
        int   net_reg_flag{};
        void* net_handle{};

        ncclConnector* send_conn_array[1] = {&send_connector};

        auto nccl_result
            = ncclNetLocalRegisterBuffer(reinterpret_cast<ncclComm*>(getActiveCommunicator()),
                                         send_buffer,
                                         kTestBufferSize,
                                         send_conn_array,
                                         1, // nPeers
                                         &net_reg_flag,
                                         &net_handle);

        EXPECT_EQ(ncclSuccess, nccl_result)
            << "Rank " << config.world_rank
            << ": ncclNetLocalRegisterBuffer failed: " << ncclGetErrorString(nccl_result);

        if(config.world_rank == 0)
        {
            printf("Rank %d:     ncclNetLocalRegisterBuffer returned: %s\n"
                   "Rank %d:     Registration flag: %d\n"
                   "Rank %d:     Handle: %p\n",
                   config.world_rank,
                   ncclGetErrorString(nccl_result),
                   config.world_rank,
                   net_reg_flag,
                   config.world_rank,
                   net_handle);
        }
    }

    // Test multiple buffer sizes with actual data transfer
    void testMultipleBufferSizes()
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: Testing multiple buffer sizes (aligned and unaligned) with NET "
                   "transport and data transfer...\n",
                   config.world_rank);
        }

        // Verify communicator is ready
        ASSERT_NE(comm_handle, nullptr) << "Rank " << config.world_rank << ": comm_handle is null";

        // Test both aligned and unaligned buffer sizes to validate edge cases
        std::vector<size_t> sizes = {
            // Small sizes (including unaligned)
            1, // Minimum size
            3, // Unaligned (not power of 2)
            7, // Unaligned
            15, // Unaligned
            63, // Unaligned

            // Medium sizes (mix of aligned and unaligned)
            1024, // 1KB (aligned)
            1025, // 1KB + 1 (unaligned)
            1536, // 1.5KB (unaligned)
            4096, // 4KB (aligned)
            4097, // 4KB + 1 (unaligned)
            5000, // Unaligned
            16384, // 16KB (aligned)
            16385, // 16KB + 1 (unaligned)

            // Large sizes (mix of aligned and unaligned)
            65536, // 64KB (aligned)
            65537, // 64KB + 1 (unaligned)
            100000, // ~98KB (unaligned)
            262144, // 256KB (aligned)
            262145, // 256KB + 1 (unaligned)
            500000, // ~488KB (unaligned)
            1048576, // 1MB (aligned)
            1048577, // 1MB + 1 (unaligned)
            4 * 1024 * 1024, // 4MB (aligned)
            4 * 1024 * 1024 + 1 // 4MB + 1 (unaligned)
        };

        int         peer_rank = (config.world_rank == 0) ? 1 : 0;
        hipStream_t stream    = getActiveStream();
        ASSERT_NE(stream, nullptr) << "Rank " << config.world_rank << ": Stream is null";

        for(size_t size : sizes)
        {
            if(config.world_rank == 0)
            {
                printf("Rank %d:   Testing size: %zu bytes with data transfer\n",
                       config.world_rank,
                       size);
            }

            // Allocate buffers with local guards (per-iteration cleanup)
            void* send_buffer = nullptr;
            void* recv_buffer = nullptr;
            auto [sendGuard, recvGuard]
                = allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, size, size, false);

            ASSERT_NE(send_buffer, nullptr) << "Rank " << config.world_rank
                                            << ": Send buffer allocation failed for size " << size;
            ASSERT_NE(recv_buffer, nullptr) << "Rank " << config.world_rank
                                            << ": Recv buffer allocation failed for size " << size;

            // Initialize send buffer with rank and size-specific pattern
            uint8_t* send_data = static_cast<uint8_t*>(send_buffer);
            for(size_t i = 0; i < size; i++)
            {
                send_data[i] = static_cast<uint8_t>(
                    (config.world_rank * 100 + i)
                    % 256); // Adjusted the pattern to use modulo 256 to fit in a byte
            }

            // Initialize recv buffer with invalid pattern
            uint8_t* recv_data = static_cast<uint8_t*>(recv_buffer);
            for(size_t i = 0; i < size; i++)
            {
                recv_data[i] = 0xFF; // Invalid pattern to detect transfer
            }

            // Perform actual data transfer using NCCL Send/Recv
            ASSERT_EQ(ncclSuccess, ncclGroupStart())
                << "Rank " << config.world_rank << ": Failed to start NCCL group for size " << size;

            ASSERT_EQ(
                ncclSuccess,
                ncclSend(send_buffer, size, ncclInt8, peer_rank, getActiveCommunicator(), stream))
                << "Rank " << config.world_rank << ": ncclSend failed for size " << size;

            ASSERT_EQ(
                ncclSuccess,
                ncclRecv(recv_buffer, size, ncclInt8, peer_rank, getActiveCommunicator(), stream))
                << "Rank " << config.world_rank << ": ncclRecv failed for size " << size;

            ASSERT_EQ(ncclSuccess, ncclGroupEnd())
                << "Rank " << config.world_rank << ": Failed to end NCCL group for size " << size;

            // Wait for transfer to complete
            ASSERT_EQ(hipSuccess, hipStreamSynchronize(stream))
                << "Rank " << config.world_rank << ": Stream synchronization failed for size "
                << size;

            // Verify received data matches peer's send pattern
            int       errors              = 0;
            const int max_errors_to_print = 5;
            for(size_t i = 0; i < size && errors < max_errors_to_print; i++)
            {
                uint8_t expected = static_cast<uint8_t>((peer_rank * 100 + i) % 256);
                if(recv_data[i] != expected)
                {
                    printf("Rank %d: Size %zu - Data mismatch at index %zu: expected %u, got %u\n",
                           config.world_rank,
                           size,
                           i,
                           expected,
                           recv_data[i]);
                    errors++;
                }
            }

            EXPECT_EQ(0, errors) << "Rank " << config.world_rank
                                 << ": Found data mismatches for buffer size " << size;

            if(config.world_rank == 0 && errors == 0)
            {
                printf("Rank %d:   Size %zu - Data transfer successful and verified\n",
                       config.world_rank,
                       size);
            }

            // Guards will automatically cleanup at end of loop iteration
        }

        if(config.world_rank == 0)
        {
            printf(
                "Rank %d: Multiple buffer sizes test completed successfully - all sizes verified\n",
                config.world_rank);
        }
    }
};

// Test cases
TEST_F(NetTransportMPITest, NetGraphRegisterBufferTest)
{
    // NET transport tests require exactly 2 ranks on 2 nodes (1 rank per node)
    if(!validateTestPrerequisites(kExactRanksForNET,
                                  kExactRanksForNET,
                                  kNoPowerOfTwoRequired,
                                  kMinNodesForNET,
                                  kMinNodesForNET))
    {
        GTEST_SKIP() << "NET transport test requires exactly " << kExactRanksForNET << " ranks on "
                     << kMinNodesForNET << " nodes (1 rank per node)";
    }

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    if(config.world_rank == 0)
    {
        printf("Rank %d: Starting ncclNetGraphRegisterBuffer test (multi-node)\n",
               config.world_rank);
    }

    testNetGraphRegisterBuffer();

    if(config.world_rank == 0)
    {
        printf("Rank %d: ncclNetGraphRegisterBuffer test completed successfully\n",
               config.world_rank);
    }
}

TEST_F(NetTransportMPITest, NetLocalRegisterBufferTest)
{
    // NET transport tests require exactly 2 ranks on 2 nodes (1 rank per node)
    if(!validateTestPrerequisites(kExactRanksForNET,
                                  kExactRanksForNET,
                                  kNoPowerOfTwoRequired,
                                  kMinNodesForNET,
                                  kMinNodesForNET))
    {
        GTEST_SKIP() << "NET transport test requires exactly " << kExactRanksForNET << " ranks on "
                     << kMinNodesForNET << " nodes (1 rank per node)";
    }

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    if(config.world_rank == 0)
    {
        printf("Rank %d: Starting ncclNetLocalRegisterBuffer test (multi-node)\n",
               config.world_rank);
    }

    testNetLocalRegisterBuffer();

    if(config.world_rank == 0)
    {
        printf("Rank %d: ncclNetLocalRegisterBuffer test completed successfully\n",
               config.world_rank);
    }
}

TEST_F(NetTransportMPITest, MultipleBufferSizesTest)
{
    // NET transport tests require exactly 2 ranks on 2 nodes (1 rank per node)
    if(!validateTestPrerequisites(kExactRanksForNET,
                                  kExactRanksForNET,
                                  kNoPowerOfTwoRequired,
                                  kMinNodesForNET,
                                  kMinNodesForNET))
    {
        GTEST_SKIP() << "NET transport test requires exactly " << kExactRanksForNET << " ranks on "
                     << kMinNodesForNET << " nodes (1 rank per node)";
    }

    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    if(config.world_rank == 0)
    {
        printf("Rank %d: Starting multiple buffer sizes test (multi-node)\n", config.world_rank);
    }

    testMultipleBufferSizes();

    if(config.world_rank == 0)
    {
        printf("Rank %d: Multiple buffer sizes test completed successfully\n", config.world_rank);
    }
}

#endif // MPI_TESTS_ENABLED
