/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "TransportMPIBase.hpp"

#ifdef MPI_TESTS_ENABLED

// Override createTestCommunicator to also update config and transport components
ncclResult_t TransportTestBase::createTestCommunicator()
{
    // Call base class implementation
    ncclResult_t result = MPITestBase::createTestCommunicator();

    if(result == ncclSuccess)
    {
        // Update config with the new communicator and stream
        config.nccl_comm = getActiveCommunicator();
        config.stream    = getActiveStream();

        // Initialize transport components now that we have a valid communicator
        comm_handle      = config.nccl_comm;
        local_peer_info  = &comm_handle->peerInfo[config.world_rank];
        remote_peer_info = &comm_handle->peerInfo[config.peer_rank];

        if(config.world_rank == 0)
        {
            printf("Rank %d: TransportTestBase config and transport components updated with "
                   "per-test communicator\n",
                   config.world_rank);
        }
    }

    return result;
}

// SetUp: Initialize common transport test components
void TransportTestBase::SetUp()
{
    // Call base class SetUp first
    MPITestBase::SetUp();

    // Initialize test configuration using aggregate initialization
    // Note: rccl_comm and stream are set to nullptr initially; tests must call createTestCommunicator()
    config = {.world_rank = RCCLMPIEnvironment::world_rank,
              .world_size = RCCLMPIEnvironment::world_size,
              .peer_rank  = (RCCLMPIEnvironment::world_rank == 0) ? 1 : 0,
              .nccl_comm  = nullptr,
              .stream     = nullptr};

    // Require at least 2 MPI processes for testing
    if(config.world_size < 2)
    {
        GTEST_SKIP() << "Transport testing requires at least 2 MPI processes";
    }

    // Check if RCCLMPIEnvironment was properly initialized
    if(RCCLMPIEnvironment::retCode != 0)
    {
        GTEST_FAIL() << "RCCLMPIEnvironment initialization failed";
    }

    // Initialize transport component pointers to nullptr
    // They will be set in createTestCommunicator() after the communicator is created
    comm_handle      = nullptr;
    local_peer_info  = nullptr;
    remote_peer_info = nullptr;

    // Create and initialize topology graph
    topology_graph = static_cast<ncclTopoGraph*>(malloc(sizeof(ncclTopoGraph)));
    if(topology_graph)
    {
        *topology_graph = {.id        = 0,
                           .pattern   = NCCL_TOPO_PATTERN_RING,
                           .nChannels = 1,
                           .bwIntra   = 0.0f,
                           .bwInter   = 0.0f,
                           .typeIntra = PATH_SYS,
                           .typeInter = PATH_NET};
    }

    // Set up P2P transport connectors
    send_connector.transportComm = &p2pTransport.send;
    recv_connector.transportComm = &p2pTransport.recv;
}

// TearDown: Cleanup common transport test components
void TransportTestBase::TearDown()
{
    // Cleanup topology graph
    if(topology_graph)
    {
        free(topology_graph);
        topology_graph = nullptr;
    }

    // Cleanup transport resources based on initialized transport type
    if(send_connector.transportResources)
    {
        if(initialized_transport == TransportType::P2P)
        {
            p2pTransport.send.free(&send_connector);
        }
        else if(initialized_transport == TransportType::Network)
        {
            netTransport.send.free(&send_connector);
        }
        send_connector.transportResources = nullptr;
    }
    if(recv_connector.transportResources)
    {
        if(initialized_transport == TransportType::P2P)
        {
            p2pTransport.recv.free(&recv_connector);
        }
        else if(initialized_transport == TransportType::Network)
        {
            netTransport.recv.free(&recv_connector);
        }
        recv_connector.transportResources = nullptr;
    }

    // Reset transport type
    initialized_transport = TransportType::None;

    // Nullify peer info pointers
    local_peer_info  = nullptr;
    remote_peer_info = nullptr;
    comm_handle      = nullptr;

    // Call base class TearDown to cleanup test communicator
    MPITestBase::TearDown();
}

// Initialize P2P transport components
void TransportTestBase::initializeP2PTransport()
{
    if(send_connector.transportResources || recv_connector.transportResources)
    {
        return; // Already initialized
    }

    // Test P2P capability
    int  can_connect{};
    auto result = p2pTransport.canConnect(&can_connect,
                                          comm_handle,
                                          topology_graph,
                                          local_peer_info,
                                          remote_peer_info);
    ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": P2P canConnect failed";
    ASSERT_NE(0, can_connect) << "Rank " << config.world_rank << ": P2P not available";

    // Setup send and recv connectors
    ncclConnect send_connect_info{};
    ncclConnect recv_connect_info{};

    result = p2pTransport.send.setup(comm_handle,
                                     topology_graph,
                                     local_peer_info,
                                     remote_peer_info,
                                     &send_connect_info,
                                     &send_connector,
                                     0,
                                     0);
    ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": P2P send setup failed";

    result = p2pTransport.recv.setup(comm_handle,
                                     topology_graph,
                                     local_peer_info,
                                     remote_peer_info,
                                     &recv_connect_info,
                                     &recv_connector,
                                     0,
                                     0);
    ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": P2P recv setup failed";

    // Mark transport as initialized
    initialized_transport = TransportType::P2P;

    if(config.world_rank == 0)
    {
        printf("Rank %d: P2P transport initialized\n", config.world_rank);
    }
}

// Initialize NET transport components
void TransportTestBase::initializeNETTransport()
{
    if(send_connector.transportResources || recv_connector.transportResources)
    {
        return; // Already initialized
    }

    // Test NET capability
    int  can_connect{};
    auto result = netTransport.canConnect(&can_connect,
                                          comm_handle,
                                          topology_graph,
                                          local_peer_info,
                                          remote_peer_info);
    ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": NET canConnect failed";
    ASSERT_NE(0, can_connect) << "Rank " << config.world_rank << ": NET transport not available";

    // Setup send and recv connectors for NET
    ncclConnect send_connect_info{};
    ncclConnect recv_connect_info{};

    result = netTransport.send.setup(comm_handle,
                                     topology_graph,
                                     local_peer_info,
                                     remote_peer_info,
                                     &send_connect_info,
                                     &send_connector,
                                     0,
                                     0);
    ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": NET send setup failed";

    result = netTransport.recv.setup(comm_handle,
                                     topology_graph,
                                     local_peer_info,
                                     remote_peer_info,
                                     &recv_connect_info,
                                     &recv_connector,
                                     0,
                                     0);
    ASSERT_EQ(ncclSuccess, result) << "Rank " << config.world_rank << ": NET recv setup failed";

    // Mark transport as initialized
    initialized_transport = TransportType::Network;

    if(config.world_rank == 0)
    {
        printf("Rank %d: NET transport initialized (multi-node capable)\n", config.world_rank);
    }
}

// Allocate and initialize test buffers
void TransportTestBase::allocateAndInitBuffers(void** send_buffer,
                                               void** recv_buffer,
                                               size_t send_bytes,
                                               size_t recv_bytes)
{
    // Allocate send buffer
    ASSERT_EQ(hipSuccess, hipMalloc(send_buffer, send_bytes))
        << "Rank " << config.world_rank << ": Failed to allocate send buffer";

    // Allocate recv buffer
    ASSERT_EQ(hipSuccess, hipMalloc(recv_buffer, recv_bytes))
        << "Rank " << config.world_rank << ": Failed to allocate recv buffer";

    // Initialize send buffer with test pattern
    std::vector<float> host_data(send_bytes / sizeof(float));
    for(size_t i = 0; i < host_data.size(); i++)
    {
        host_data[i] = static_cast<float>(config.world_rank * 1000 + i);
    }

    ASSERT_EQ(hipSuccess,
              hipMemcpy(*send_buffer, host_data.data(), send_bytes, hipMemcpyHostToDevice))
        << "Rank " << config.world_rank << ": Failed to initialize send buffer";

    if(config.world_rank == 0)
    {
        printf("Rank %d: Allocated and initialized buffers (%zu bytes each)\n",
               config.world_rank,
               send_bytes);
    }
}

// Pre-register buffers with ncclCommRegister
void TransportTestBase::preRegisterBuffers(void*  send_buffer,
                                           void*  recv_buffer,
                                           size_t send_bytes,
                                           size_t recv_bytes,
                                           void** send_reg_handle,
                                           void** recv_reg_handle)
{
    // Register send buffer
    ASSERT_EQ(ncclSuccess,
              ncclCommRegister(getActiveCommunicator(), send_buffer, send_bytes, send_reg_handle))
        << "Rank " << config.world_rank << ": Failed to pre-register send buffer";

    // Register recv buffer
    ASSERT_EQ(ncclSuccess,
              ncclCommRegister(getActiveCommunicator(), recv_buffer, recv_bytes, recv_reg_handle))
        << "Rank " << config.world_rank << ": Failed to pre-register recv buffer";

    if(config.world_rank == 0)
    {
        printf("Rank %d: Pre-registered buffers with ncclCommRegister\n", config.world_rank);
    }
}

// Buffer allocation with automatic RAII guards
std::pair<BufferGuard, BufferGuard> TransportTestBase::allocateAndInitBuffersGuarded(
    void** send_buffer,
    void** recv_buffer,
    size_t send_bytes,
    size_t recv_bytes,
    bool   store_in_base)
{
    // Allocate buffers using existing method
    allocateAndInitBuffers(send_buffer, recv_buffer, send_bytes, recv_bytes);

    // Create guards
    BufferGuard sendGuard(*send_buffer, false); // Device memory
    BufferGuard recvGuard(*recv_buffer, false); // Device memory

    if(store_in_base)
    {
        // Store guards in base class for cleanup at test end
        buffer_guards_.push_back(std::move(sendGuard));
        buffer_guards_.push_back(std::move(recvGuard));

        // Return empty guards (resources now managed by base class)
        return {BufferGuard(nullptr, false), BufferGuard(nullptr, false)};
    }
    else
    {
        // Return guards for caller to manage (cleanup at caller's scope exit)
        return {std::move(sendGuard), std::move(recvGuard)};
    }
}

// Buffer registration with automatic RAII guards
std::pair<NcclRegHandleGuard, NcclRegHandleGuard> TransportTestBase::preRegisterBuffersGuarded(
    void*  send_buffer,
    void*  recv_buffer,
    size_t send_bytes,
    size_t recv_bytes,
    void** send_reg_handle,
    void** recv_reg_handle,
    bool   store_in_base)
{
    // Register buffers using existing method
    preRegisterBuffers(send_buffer, recv_buffer, send_bytes, recv_bytes, send_reg_handle, recv_reg_handle);

    // Create guards
    NcclRegHandleGuard sendGuard(*send_reg_handle, NcclRegHandleDeleter(getActiveCommunicator()));
    NcclRegHandleGuard recvGuard(*recv_reg_handle, NcclRegHandleDeleter(getActiveCommunicator()));

    if(store_in_base)
    {
        // Store guards in base class for cleanup at test end
        reg_handle_guards_.push_back(std::move(sendGuard));
        reg_handle_guards_.push_back(std::move(recvGuard));

        // Return empty guards (resources now managed by base class)
        return {NcclRegHandleGuard(nullptr, NcclRegHandleDeleter(nullptr)),
                NcclRegHandleGuard(nullptr, NcclRegHandleDeleter(nullptr))};
    }
    else
    {
        // Return guards for caller to manage (cleanup at caller's scope exit)
        return {std::move(sendGuard), std::move(recvGuard)};
    }
}

#endif // MPI_TESTS_ENABLED
