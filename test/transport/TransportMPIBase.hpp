/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TRANSPORT_MPI_BASE_HPP
#define TRANSPORT_MPI_BASE_HPP

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "rccl/rccl.h"
#include "gtest/gtest.h"

#ifdef MPI_TESTS_ENABLED
    #include "MPITestBase.hpp"
    #include "RCCLMPIEnvironment.hpp"
    #include "RCCLTestResourceGuards.hpp"
    #include "comm.h"
    #include "core.h"
    #include "device.h"
    #include "graph.h"
    #include "graph/topo.h"
    #include "nccl_common.h"
    #include "transport.h"

using namespace RCCLTestGuards;

extern struct ncclTransport p2pTransport;
extern struct ncclTransport netTransport;

// Common test configuration
struct TransportTestConfig
{
    int         world_rank{0};
    int         world_size{0};
    int         peer_rank{0};
    ncclComm_t  nccl_comm{nullptr};
    hipStream_t stream{nullptr};
};

// Base class for transport tests with common functionality
// Inherits from MPITestBase to get validation capabilities
class TransportTestBase : public MPITestBase
{
protected:
    TransportTestConfig config;

    // Transport connectors (can be used for P2P or NET)
    ncclConnector send_connector = {};
    ncclConnector recv_connector = {};

    // Track which transport type is initialized
    enum class TransportType
    {
        None,
        P2P,
        Network
    };
    TransportType initialized_transport = TransportType::None;

    // Core NCCL components
    struct ncclComm* comm_handle      = nullptr;
    ncclPeerInfo*    local_peer_info  = nullptr;
    ncclPeerInfo*    remote_peer_info = nullptr;
    ncclTopoGraph*   topology_graph   = nullptr;

    // RAII guards for automatic resource cleanup
    // These are managed by helper methods and cleaned up automatically
    std::vector<BufferGuard>         buffer_guards_;
    std::vector<NcclRegHandleGuard> reg_handle_guards_;

    // Setup and teardown
    void SetUp() override;
    void TearDown() override;

    // Override createTestCommunicator to also update config
    ncclResult_t createTestCommunicator() override;

    // Common helper methods
    void initializeP2PTransport();
    void initializeNETTransport();

    // Buffer allocation (unguarded - for manual management)
    void allocateAndInitBuffers(void** send_buffer,
                                void** recv_buffer,
                                size_t send_bytes,
                                size_t recv_bytes);

    // Buffer allocation with automatic RAII guards
    // store_in_base=true: Guards stored in base class, cleanup at test end
    // store_in_base=false: Guards returned, caller controls cleanup scope
    std::pair<BufferGuard, BufferGuard> allocateAndInitBuffersGuarded(
        void** send_buffer,
        void** recv_buffer,
        size_t send_bytes,
        size_t recv_bytes,
        bool   store_in_base = true);

    // Buffer registration (unguarded - for manual management)
    void preRegisterBuffers(void*  send_buffer,
                            void*  recv_buffer,
                            size_t send_bytes,
                            size_t recv_bytes,
                            void** send_reg_handle,
                            void** recv_reg_handle);

    // Buffer registration with automatic RAII guards
    // store_in_base=true: Guards stored in base class, cleanup at test end
    // store_in_base=false: Guards returned, caller controls cleanup scope
    std::pair<NcclRegHandleGuard, NcclRegHandleGuard> preRegisterBuffersGuarded(
        void*  send_buffer,
        void*  recv_buffer,
        size_t send_bytes,
        size_t recv_bytes,
        void** send_reg_handle,
        void** recv_reg_handle,
        bool   store_in_base = true);
};

#endif // MPI_TESTS_ENABLED

#endif // TRANSPORT_MPI_BASE_HPP
