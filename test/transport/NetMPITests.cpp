/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TransportMPIBase.hpp"

#ifdef MPI_TESTS_ENABLED

// Import MPI test constants for convenience
using namespace MPITestConstants;

namespace
{
// Buffer size constants
inline constexpr size_t kTestBufferSize = 16384;
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

private:
    // Helper to setup buffers for tests
    void setupTestBuffers(void** send_buffer,
                          void** recv_buffer,
                          void** send_reg_handle,
                          void** recv_reg_handle)
    {
        // Initialize NET transport for multi-node capable testing
        // NET transport works across multiple nodes using network communication (IB/Ethernet)
        initializeNETTransport();

        // Allocate and initialize buffers
        allocateAndInitBuffers(send_buffer, recv_buffer, kTestBufferSize, kTestBufferSize);

        // Pre-register buffers
        preRegisterBuffers(*send_buffer,
                           *recv_buffer,
                           kTestBufferSize,
                           kTestBufferSize,
                           send_reg_handle,
                           recv_reg_handle);
    }

public:
    // Test ncclNetGraphRegisterBuffer
    void testNetGraphRegisterBuffer()
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: Testing ncclNetGraphRegisterBuffer...\n", config.world_rank);
        }

        // Setup test buffers
        void* send_buffer     = nullptr;
        void* recv_buffer     = nullptr;
        void* send_reg_handle = nullptr;
        void* recv_reg_handle = nullptr;

        setupTestBuffers(&send_buffer, &recv_buffer, &send_reg_handle, &recv_reg_handle);

        // Test ncclNetGraphRegisterBuffer
        if(send_connector.transportResources)
        {
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

                if(net_reg_flag == 0)
                {
                    printf("Rank %d:     Note: regFlag=0 may occur when:\n"
                           "Rank %d:       - netDeviceType == NCCL_NET_DEVICE_UNPACK\n"
                           "Rank %d:       - NCCL_GRAPH_REGISTER=0 or !comm->planner.persistent\n"
                           "Rank %d:       - Network device doesn't support GDR\n",
                           config.world_rank,
                           config.world_rank,
                           config.world_rank,
                           config.world_rank);
                }
            }
        }

        // Cleanup
        cleanupBuffers(send_buffer, recv_buffer, send_reg_handle, recv_reg_handle);

        if(config.world_rank == 0)
        {
            printf("Rank %d: ncclNetGraphRegisterBuffer test completed\n", config.world_rank);
        }
    }

    // Test ncclRegisterP2pNetBuffer
    void testRegisterP2pNetBuffer()
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: Testing ncclRegisterP2pNetBuffer...\n"
                   "Rank %d: This API internally calls ncclNetGraphRegisterBuffer "
                   "and ncclNetLocalRegisterBuffer\n",
                   config.world_rank,
                   config.world_rank);
        }

        // Setup test buffers
        void* send_buffer     = nullptr;
        void* recv_buffer     = nullptr;
        void* send_reg_handle = nullptr;
        void* recv_reg_handle = nullptr;

        setupTestBuffers(&send_buffer, &recv_buffer, &send_reg_handle, &recv_reg_handle);

        // Test ncclRegisterP2pNetBuffer
        if(send_connector.transportResources)
        {
            int                                                       net_reg_flag{};
            void*                                                     net_handle{};
            ncclIntruQueue<ncclCommCallback, &ncclCommCallback::next> cleanup_queue{};

            auto nccl_result
                = ncclRegisterP2pNetBuffer(reinterpret_cast<ncclComm*>(getActiveCommunicator()),
                                           send_buffer,
                                           kTestBufferSize,
                                           &send_connector,
                                           &net_reg_flag,
                                           &net_handle,
                                           &cleanup_queue);

            EXPECT_EQ(ncclSuccess, nccl_result)
                << "Rank " << config.world_rank
                << ": ncclRegisterP2pNetBuffer failed: " << ncclGetErrorString(nccl_result);

            if(config.world_rank == 0)
            {
                printf("Rank %d:     ncclRegisterP2pNetBuffer returned: %s\n"
                       "Rank %d:     Registration flag: %d\n"
                       "Rank %d:     Handle: %p\n",
                       config.world_rank,
                       ncclGetErrorString(nccl_result),
                       config.world_rank,
                       net_reg_flag,
                       config.world_rank,
                       net_handle);

                if(net_reg_flag == 0)
                {
                    printf("Rank %d:     Note: regFlag=0 may occur when:\n"
                           "Rank %d:       - netDeviceType == NCCL_NET_DEVICE_UNPACK\n"
                           "Rank %d:       - NCCL_GRAPH_REGISTER=0 or !comm->planner.persistent "
                           "(tries local)\n"
                           "Rank %d:       - NCCL_LOCAL_REGISTER=0 (when graph register fails)\n",
                           config.world_rank,
                           config.world_rank,
                           config.world_rank,
                           config.world_rank);
                }
            }
        }

        // Cleanup
        cleanupBuffers(send_buffer, recv_buffer, send_reg_handle, recv_reg_handle);

        if(config.world_rank == 0)
        {
            printf("Rank %d:   ncclRegisterP2pNetBuffer test completed\n"
                   "Rank %d:   - Internally calls ncclNetGraphRegisterBuffer "
                   "(if NCCL_GRAPH_REGISTER=1 && persistent)\n"
                   "Rank %d:   - Falls back to ncclNetLocalRegisterBuffer "
                   "(if graph fails && NCCL_LOCAL_REGISTER=1)\n",
                   config.world_rank,
                   config.world_rank,
                   config.world_rank);
        }
    }
};

// Test cases
TEST_F(NetTransportMPITest, NetGraphRegisterBufferTest)
{
    // NET transport is multi-node capable - works on any node configuration
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kNoNodeLimit))
        << "Test requirements not met - all ranks must meet requirements";

    // Detect if running on single node
    int node_count = MPITestConstants::detectNodeCount();
    if(node_count == 1)
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: Detected single-node execution\n", config.world_rank);
            printf("Rank %d: NET transport is for inter-node (network) communication\n",
                   config.world_rank);
        }
        GTEST_SKIP() << "Rank " << config.world_rank
                     << ": Skipping NET transport test on single-node. "
                     << "NET transport requires multiple nodes for network communication. ";
    }

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    if(config.world_rank == 0)
    {
        printf("Rank %d: Starting ncclNetGraphRegisterBuffer test (multi-node: %d nodes)\n",
               config.world_rank,
               node_count);
    }

    testNetGraphRegisterBuffer();

    if(config.world_rank == 0)
    {
        printf("Rank %d: ncclNetGraphRegisterBuffer test completed successfully\n",
               config.world_rank);
    }
}

TEST_F(NetTransportMPITest, RegisterP2pNetBufferTest)
{
    // NET transport is multi-node capable - works on any node configuration
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI,
                                          kNoProcessLimit,
                                          kRequirePowerOfTwo,
                                          1,
                                          kNoNodeLimit))
        << "Test requirements not met - all ranks must meet requirements";

    // Detect if running on single node
    int node_count = MPITestConstants::detectNodeCount();
    if(node_count == 1)
    {
        if(config.world_rank == 0)
        {
            printf("Rank %d: Detected single-node execution\n", config.world_rank);
            printf("Rank %d: NET transport is for inter-node (network) communication\n",
                   config.world_rank);
        }
        GTEST_SKIP() << "Rank " << config.world_rank
                     << ": Skipping NET transport test on single-node. "
                     << "NET transport requires multiple nodes for network communication. ";
    }

    // Create test-specific communicator for isolation
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    if(config.world_rank == 0)
    {
        printf("Rank %d: Starting ncclRegisterP2pNetBuffer test (multi-node: %d nodes)\n",
               config.world_rank,
               node_count);
    }

    testRegisterP2pNetBuffer();

    if(config.world_rank == 0)
    {
        printf("Rank %d: ncclRegisterP2pNetBuffer test completed successfully\n",
               config.world_rank);
    }
}

#endif // MPI_TESTS_ENABLED
