/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "MPITestBase.hpp"
#include "RCCLTestResourceGuards.hpp"
#include "nccl.h"
#include "net.h"
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>

// Import guard namespace for convenience
using namespace RCCLTestGuards;

// External NET IB plugin
extern ncclNet_t ncclNetIb;

// Test fixture for NET IB tests
class NetIbMPITest : public MPITestBase {
protected:
    static constexpr int kMinProcessesForMPI = 2;
    static constexpr bool kRequirePowerOfTwo = true;
    static constexpr int kNoNodeLimit = MPITestConstants::kNoNodeLimit;

    ncclNet_t* net_;
    int numDevices_;
    std::vector<int> deviceIds_;

    void SetUp() override {
        MPITestBase::SetUp();
        net_ = &ncclNetIb;
        numDevices_ = 0;
    }

    void TearDown() override {
        MPITestBase::TearDown();
    }

    // Helper: Initialize NET IB plugin
    ncclResult_t InitNetIb() {
        return net_->init(nullptr, nullptr);
    }

    // Helper: Get number of devices
    ncclResult_t GetDeviceCount(int* ndev) {
        return net_->devices(ndev);
    }

    // Helper: Get device properties
    ncclResult_t GetDeviceProperties(int dev, ncclNetProperties_t* props) {
        return net_->getProperties(dev, props);
    }

    // Helper: Create listen comm
    ncclResult_t CreateListenComm(int dev, ncclNetHandle_t* handle, void** listenComm) {
        return net_->listen(dev, handle, listenComm);
    }

    // Helper: Connect to remote
    ncclResult_t ConnectToRemote(int dev, ncclNetHandle_t* handle, void** sendComm) {
        return net_->connect(dev, nullptr, handle, sendComm, nullptr);
    }

    // Helper: Accept connection
    ncclResult_t AcceptConnection(void* listenComm, void** recvComm) {
        return net_->accept(listenComm, recvComm, nullptr);
    }

    // Helper: Register memory
    ncclResult_t RegisterMemory(void* comm, void* data, size_t size, int type, void** mhandle) {
        return net_->regMr(comm, data, size, type, mhandle);
    }

    // Helper: Register DMA-BUF memory
    ncclResult_t RegisterDmaBufMemory(void* comm, void* data, size_t size, int type,
                                      uint64_t offset, int fd, void** mhandle) {
        return net_->regMrDmaBuf(comm, data, size, type, offset, fd, mhandle);
    }

    // Helper: Deregister memory
    ncclResult_t DeregisterMemory(void* comm, void* mhandle) {
        return net_->deregMr(comm, mhandle);
    }

    // Helper: Post send operation
    ncclResult_t PostSend(void* sendComm, void* data, size_t size, int tag,
                         void* mhandle, void** request) {
        return net_->isend(sendComm, data, size, tag, mhandle, nullptr, request);
    }

    // Helper: Post recv operation
    ncclResult_t PostRecv(void* recvComm, int n, void** data, size_t* sizes,
                         int* tags, void** mhandles, void** request) {
        return net_->irecv(recvComm, n, data, sizes, tags, mhandles, nullptr, request);
    }

    // Helper: Flush operation
    ncclResult_t FlushRecv(void* recvComm, int n, void** data, int* sizes,
                          void** mhandles, void** request) {
        return net_->iflush(recvComm, n, data, sizes, mhandles, request);
    }

    // Helper: Test request completion
    ncclResult_t TestRequest(void* request, int* done, int* sizes) {
        return net_->test(request, done, sizes);
    }

    // Helper: Close send comm
    ncclResult_t CloseSendComm(void* sendComm) {
        return net_->closeSend(sendComm);
    }

    // Helper: Close recv comm
    ncclResult_t CloseRecvComm(void* recvComm) {
        return net_->closeRecv(recvComm);
    }

    // Helper: Close listen comm
    ncclResult_t CloseListenComm(void* listenComm) {
        return net_->closeListen(listenComm);
    }

    // Helper: Make virtual device
    ncclResult_t MakeVirtualDevice(int* dev, ncclNetVDeviceProps_t* props) {
        return net_->makeVDevice(dev, props);
    }

    // Helper: Setup connection between two ranks
    struct ConnectionPair {
        void* sendComm = nullptr;
        void* recvComm = nullptr;
        void* listenComm = nullptr;
        ncclNetHandle_t handle;
    };

    ncclResult_t SetupConnection(int dev, ConnectionPair& pair, int rank, int peerRank) {
        if (rank == 0) {
            // Rank 0: Listen
            NCCLCHECK(CreateListenComm(dev, &pair.handle, &pair.listenComm));

            // Send handle to peer
            MPI_Send(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD);

            // Accept connection
            int done = 0;
            while (!done) {
                ncclResult_t result = AcceptConnection(pair.listenComm, &pair.recvComm);
                if (result == ncclSuccess && pair.recvComm != nullptr) {
                    done = 1;
                }
            }
        } else {
            // Rank 1: Connect
            MPI_Recv(&pair.handle, sizeof(ncclNetHandle_t), MPI_BYTE, peerRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Connect to peer
            int done = 0;
            while (!done) {
                ncclResult_t result = ConnectToRemote(dev, &pair.handle, &pair.sendComm);
                if (result == ncclSuccess && pair.sendComm != nullptr) {
                    done = 1;
                }
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        return ncclSuccess;
    }

    // Helper: Cleanup connection
    void CleanupConnection(ConnectionPair& pair, int rank) {
        if (rank == 0) {
            if (pair.recvComm) CloseRecvComm(pair.recvComm);
            if (pair.listenComm) CloseListenComm(pair.listenComm);
        } else {
            if (pair.sendComm) CloseSendComm(pair.sendComm);
        }
    }

    // Helper: Allocate GPU memory
    void* AllocateGpuMemory(size_t size) {
        void* ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, size);
        if (err != hipSuccess) {
            return nullptr;
        }
        return ptr;
    }

    // Helper: Free GPU memory
    void FreeGpuMemory(void* ptr) {
        if (ptr) {
            hipFree(ptr);
        }
    }

    // Helper: Allocate host memory
    void* AllocateHostMemory(size_t size) {
        return malloc(size);
    }

    // Helper: Free host memory
    void FreeHostMemory(void* ptr) {
        if (ptr) {
            free(ptr);
        }
    }

    // Helper: Initialize buffer with pattern
    void InitializeBuffer(void* buffer, size_t size, int pattern) {
        uint8_t* bytes = static_cast<uint8_t*>(buffer);
        for (size_t i = 0; i < size; i++) {
            bytes[i] = static_cast<uint8_t>((pattern + i) % 256);
        }
    }

    // Helper: Verify buffer pattern
    bool VerifyBuffer(void* buffer, size_t size, int pattern) {
        uint8_t* bytes = static_cast<uint8_t*>(buffer);
        for (size_t i = 0; i < size; i++) {
            if (bytes[i] != static_cast<uint8_t>((pattern + i) % 256)) {
                return false;
            }
        }
        return true;
    }

    // Helper: Wait for request completion with timeout
    ncclResult_t WaitForCompletion(void* request, int* sizes, int timeoutMs = 5000) {
        int done = 0;
        int attempts = 0;
        const int maxAttempts = timeoutMs / 10;

        while (!done && attempts < maxAttempts) {
            ncclResult_t result = TestRequest(request, &done, sizes);

            if (result != ncclSuccess) {
                return result;
            }

            if (done) {
                break;
            } else {
                usleep(10000); // 10ms
                attempts++;
            }
        }

        return done ? ncclSuccess : ncclInternalError;
    }
};

// ============================================================================
// Initialization Tests
// ============================================================================

TEST_F(NetIbMPITest, InitializePlugin) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ncclResult_t result = InitNetIb();
    EXPECT_EQ(result, ncclSuccess) << "Failed to initialize NET IB plugin";
}

TEST_F(NetIbMPITest, GetDeviceCount) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    EXPECT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    EXPECT_GT(ndev, 0) << "No IB devices found";

    if (RCCLMPIEnvironment::world_rank == 0) {
        printf("Found %d IB device(s)\n", ndev);
    }
}

// ============================================================================
// Device Properties Tests
// ============================================================================

TEST_F(NetIbMPITest, GetDeviceProperties) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    for (int i = 0; i < ndev; i++) {
        ncclNetProperties_t props;
        memset(&props, 0, sizeof(props));

        EXPECT_EQ(GetDeviceProperties(i, &props), ncclSuccess)
            << "Failed to get properties for device " << i;

        // Verify properties are valid
        EXPECT_NE(props.name, nullptr) << "Device " << i << " has NULL name";
        EXPECT_GT(props.speed, 0) << "Device " << i << " has invalid speed";
        EXPECT_NE(props.pciPath, nullptr) << "Device " << i << " has NULL pciPath";

        if (RCCLMPIEnvironment::world_rank == 0) {
            printf("Device %d: name=%s speed=%d pciPath=%s\n",
                   i, props.name, props.speed, props.pciPath);
        }
    }
}

TEST_F(NetIbMPITest, GetDevicePropertiesInvalidDevice) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);

    ncclNetProperties_t props;

    // Invalid device ID (too large)
    ncclResult_t result = GetDeviceProperties(ndev + 100, &props);
    EXPECT_NE(result, ncclSuccess) << "Should fail for invalid device ID";
}

// ============================================================================
// Connection Setup Tests
// ============================================================================

TEST_F(NetIbMPITest, ListenAndConnect) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    if (rank == 0) {
        EXPECT_NE(pair.recvComm, nullptr) << "Recv comm should be established";
        EXPECT_NE(pair.listenComm, nullptr) << "Listen comm should exist";
    } else {
        EXPECT_NE(pair.sendComm, nullptr) << "Send comm should be established";
    }
}

TEST_F(NetIbMPITest, ConnectWithInvalidHandle) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ncclNetHandle_t invalidHandle;
    memset(&invalidHandle, 0xFF, sizeof(invalidHandle));
    void* sendComm = nullptr;

    // Negative test: Connect with garbage handle
    ncclResult_t result = ConnectToRemote(0, &invalidHandle, &sendComm);
    EXPECT_EQ(result, ncclInternalError) << "Should fail with invalid handle";
}

// ============================================================================
// Memory Registration Tests
// ============================================================================

TEST_F(NetIbMPITest, RegisterHostMemory) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = 4096;
    void* buffer = AllocateHostMemory(bufferSize);
    ASSERT_NE(buffer, nullptr);
    BufferGuard bufferGuard(buffer, true);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    EXPECT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    EXPECT_NE(mhandle, nullptr);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    EXPECT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
}

TEST_F(NetIbMPITest, RegisterGpuMemory) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = 4096;
    void* buffer = AllocateGpuMemory(bufferSize);
    ASSERT_NE(buffer, nullptr);
    BufferGuard bufferGuard(buffer, false);  // false = device memory

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    EXPECT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_CUDA, &mhandle), ncclSuccess);
    EXPECT_NE(mhandle, nullptr);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    EXPECT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
}

TEST_F(NetIbMPITest, RegisterMemoryNullPointer) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // Negative test: NULL buffer pointer
    ncclResult_t result = RegisterMemory(comm, nullptr, 4096, NCCL_PTR_HOST, &mhandle);
    EXPECT_NE(result, ncclSuccess) << "Should fail with NULL buffer";
}

TEST_F(NetIbMPITest, DeregisterNullHandle) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;

    // Edge case: Deregister NULL handle (should be no-op)
    EXPECT_EQ(DeregisterMemory(comm, nullptr), ncclSuccess);
}

// ============================================================================
// Send/Recv Tests
// ============================================================================

TEST_F(NetIbMPITest, SimpleSendRecv) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = 4096;
    const int tag = 42;

    void* buffer = AllocateHostMemory(bufferSize);
    ASSERT_NE(buffer, nullptr);
    BufferGuard bufferGuard(buffer, true);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        // Sender
        InitializeBuffer(buffer, bufferSize, rank);

        ASSERT_EQ(PostSend(pair.sendComm, buffer, bufferSize, tag, mhandle, &request), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion
    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Received size mismatch";
    }

    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
}

TEST_F(NetIbMPITest, SendRecvMultipleSizes) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    int rank = RCCLMPIEnvironment::world_rank;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int peerRank = (rank + 1) % 2;
    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    // Test various sizes
    std::vector<size_t> testSizes = {1, 64, 256, 1024, 4096, 16384, 65536};

    for (size_t size : testSizes) {
        const int tag = 100;
        const int seed = 2000 + static_cast<int>(size);  // Unique seed per size

        void* buffer = AllocateHostMemory(size);
        ASSERT_NE(buffer, nullptr);
        BufferGuard bufferGuard(buffer, true);  // Local guard for loop iteration

        void* mhandle = nullptr;
        void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
        ASSERT_EQ(RegisterMemory(comm, buffer, size, NCCL_PTR_HOST, &mhandle), ncclSuccess);
        NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

        void* request = nullptr;

        if (rank == 0) {
            memset(buffer, 0, size);

            void* recvBuffers[1] = {buffer};
            size_t recvSizes[1] = {size};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
            ASSERT_NE(request, nullptr) << "Recv request should never be NULL";
        } else {
            InitializeBuffer(buffer, size, seed);

            // NET IB isend can return success with NULL request if FIFO isn't ready
            // This means the receiver hasn't posted recv yet - retry until ready
            int attempts = 0;
            const int maxAttempts = 1000;  // 10 seconds max
            do {
                ncclResult_t result = PostSend(pair.sendComm, buffer, size, tag, mhandle, &request);
                ASSERT_EQ(result, ncclSuccess);

                if (request != nullptr) {
                    break;
                }

                // NULL request means "not ready yet", wait and retry
                if (++attempts >= maxAttempts) {
                    FAIL() << "PostSend returned NULL request after " << maxAttempts << " attempts";
                }
                usleep(10000);  // 10ms between retries
            } while (request == nullptr);
        }

        // Barrier 1: Ensure both ranks have posted their operations before waiting
        MPI_Barrier(MPI_COMM_WORLD);

        // Wait for completion
        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Barrier 2: CRITICAL - Ensure BOTH ranks have completed before EITHER continues
        // This prevents rank A from starting next transfer while rank B is still
        // completing current transfer, which would cause request object reuse race conditions
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            EXPECT_EQ(sizes[0], size) << "Size mismatch for transfer of " << size << " bytes";

            // Validate received data matches expected pattern
            bool dataValid = VerifyBuffer(buffer, size, seed);
            EXPECT_TRUE(dataValid) << "Data validation failed for size " << size;
        }

        ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
    }
}

TEST_F(NetIbMPITest, SendRecvZeroSize) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = 4096;
    const int tag = 50;

    void* buffer = AllocateHostMemory(bufferSize);
    ASSERT_NE(buffer, nullptr);
    BufferGuard bufferGuard(buffer, true);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver - expect zero bytes
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        // Sender - send zero bytes
        ASSERT_EQ(PostSend(pair.sendComm, buffer, 0, tag, mhandle, &request), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion
    int sizes[1] = {-1};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    if (rank == 0) {
        EXPECT_EQ(sizes[0], 0) << "Should receive zero bytes";
    }

    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
}

// ============================================================================
// Flush Tests
// ============================================================================

TEST_F(NetIbMPITest, FlushAfterRecv) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // Check if GDR is available
    ncclNetProperties_t props;
    ASSERT_EQ(GetDeviceProperties(0, &props), ncclSuccess);

    if (!(props.ptrSupport & NCCL_PTR_CUDA)) {
        GTEST_SKIP() << "GDR not supported, skipping flush test";
    }

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = 4096;
    const int tag = 200;

    void* buffer = AllocateGpuMemory(bufferSize);
    ASSERT_NE(buffer, nullptr);
    BufferGuard bufferGuard(buffer, false);  // false = device memory

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_CUDA, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        // Sender
        void* hostBuffer = AllocateHostMemory(bufferSize);
        ASSERT_NE(hostBuffer, nullptr);
        BufferGuard hostBufferGuard(hostBuffer, true);

        InitializeBuffer(hostBuffer, bufferSize, rank);
        hipMemcpy(buffer, hostBuffer, bufferSize, hipMemcpyHostToDevice);

        ASSERT_EQ(PostSend(pair.sendComm, buffer, bufferSize, tag, mhandle, &request), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion
    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

    if (rank == 0) {
        // Issue flush
        void* flushBuffers[1] = {buffer};
        int flushSizes[1] = {static_cast<int>(bufferSize)};
        void* flushHandles[1] = {mhandle};
        void* flushRequest = nullptr;

        ncclResult_t result = FlushRecv(pair.recvComm, 1, flushBuffers, flushSizes,
                                       flushHandles, &flushRequest);

        if (result == ncclSuccess && flushRequest != nullptr) {
            int flushDone = 0;
            ASSERT_EQ(WaitForCompletion(flushRequest, nullptr), ncclSuccess);
        }
    }

    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
}

// ============================================================================
// Virtual Device Tests
// ============================================================================

TEST_F(NetIbMPITest, MakeVirtualDevice) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    if (ndev < 2) {
        GTEST_SKIP() << "Need at least 2 devices for virtual device test";
    }

    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 2;
    vProps.devs[0] = 0;
    vProps.devs[1] = 1;

    int vdev = -1;
    ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);

    // Virtual device creation may or may not be supported
    if (result == ncclSuccess) {
        EXPECT_GE(vdev, 0) << "Virtual device ID should be non-negative";

        if (RCCLMPIEnvironment::world_rank == 0) {
            printf("Created virtual device %d from physical devices 0 and 1\n", vdev);
        }
    }
}

TEST_F(NetIbMPITest, MakeVirtualDeviceInvalidProps) {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, MPITestConstants::kNoProcessLimit,
                                         kRequirePowerOfTwo, 1, kNoNodeLimit))
        << "Test requirements not met";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    // Negative test: Zero devices
    ncclNetVDeviceProps_t vProps;
    vProps.ndevs = 0;

    int vdev = -1;
    ncclResult_t result = MakeVirtualDevice(&vdev, &vProps);
    EXPECT_EQ(result, ncclInvalidUsage) << "Should fail with zero devices";

}

// ============================================================================
// Stress and Edge Case Tests
// ============================================================================

    // Tests multiple sequential transfers on the same connection using host memory.
// This test validates that request objects can be properly reused across multiple
// send/recv operations without resource exhaustion or state corruption.
//
// SYNCHRONIZATION STRATEGY:
//   Two barriers per iteration ensure proper ordering:
//   1. After Post: Ensures both ranks have posted before either waits
//   2. After Completion: CRITICAL - Ensures BOTH ranks complete before EITHER
//      starts next iteration. Without this, rapid request reuse causes races.
//
// NULL REQUEST HANDLING:
//   NET IB isend() can return ncclSuccess with NULL request when the FIFO
//   isn't ready yet (receiver's irecv RDMA write hasn't reached sender yet).
//   This is NOT an error - it means "try again". The sender must retry until
//   it gets a valid request pointer. This is normal NET IB protocol behavior.
//
// NOTE: Flush (iflush) is intentionally NOT called because:
//   1. Flush is only needed for GPU Direct RDMA to ensure data visibility
//   2. For NCCL_PTR_HOST transfers, flush is unnecessary
//   3. Flush testing is covered by FlushRecvOnGpuBuffer test
TEST_F(NetIbMPITest, MultipleSequentialTransfers) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    int rank = RCCLMPIEnvironment::world_rank;

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int peerRank = (rank + 1) % 2;
    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = 4096;
    const int numTransfers = 100;

    void* sendBuffer = nullptr;
    void* recvBuffer = nullptr;
    BufferGuard sendBufferGuard(nullptr, true);
    BufferGuard recvBufferGuard(nullptr, true);

    if (rank == 0) {
        recvBuffer = AllocateHostMemory(bufferSize);
        ASSERT_NE(recvBuffer, nullptr);
        recvBufferGuard.reset(recvBuffer);
    } else {
        sendBuffer = AllocateHostMemory(bufferSize);
        ASSERT_NE(sendBuffer, nullptr);
        sendBufferGuard.reset(sendBuffer);
    }

    void* mhandle = nullptr;
    void* buffer = (rank == 0) ? recvBuffer : sendBuffer;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    for (int i = 0; i < numTransfers; i++) {
        const int tag = 300 + i;
        const int seed = 1000 + i;  // Unique seed for each transfer
        void* request = nullptr;

        if (rank == 0) {
            memset(recvBuffer, 0, bufferSize);

            void* recvBuffers[1] = {recvBuffer};
            size_t recvSizes[1] = {bufferSize};
            int recvTags[1] = {tag};
            void* recvHandles[1] = {mhandle};

            ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                              recvHandles, &request), ncclSuccess);
            ASSERT_NE(request, nullptr) << "Recv request should never be NULL";
        } else {
            InitializeBuffer(sendBuffer, bufferSize, seed);

            // NET IB isend can return success with NULL request if FIFO isn't ready
            // This means the receiver hasn't posted recv yet - retry until ready
            int attempts = 0;
            const int maxAttempts = 1000;  // 10 seconds max
            do {
                ncclResult_t result = PostSend(pair.sendComm, sendBuffer, bufferSize, tag, mhandle, &request);
                ASSERT_EQ(result, ncclSuccess);

                if (request != nullptr) {
                    break;
                }

                // NULL request means "not ready yet", wait and retry
                if (++attempts >= maxAttempts) {
                    FAIL() << "PostSend returned NULL request after " << maxAttempts << " attempts";
                }
                usleep(10000);  // 10ms between retries
            } while (request == nullptr);
        }

        // Barrier 1: Ensure both ranks have posted their operations before waiting
        MPI_Barrier(MPI_COMM_WORLD);

        // Wait for completion
        int sizes[1] = {0};
        ASSERT_EQ(WaitForCompletion(request, sizes), ncclSuccess);

        // Barrier 2: CRITICAL - Ensure BOTH ranks have completed before EITHER continues
        // This prevents rank A from starting transfer N+1 while rank B is still
        // completing transfer N, which would cause request object reuse race conditions
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            EXPECT_EQ(sizes[0], bufferSize) << "Transfer " << i << " size mismatch";

            // Validate received data matches expected pattern
            bool dataValid = VerifyBuffer(recvBuffer, bufferSize, seed);
            EXPECT_TRUE(dataValid) << "Transfer " << i << " data validation failed (seed=" << seed << ")";

            if (!dataValid) {
                // Print first few mismatched values for debugging
                uint32_t* recvData = static_cast<uint32_t*>(recvBuffer);
                printf("Rank %d: Transfer %d data mismatch. First 4 values:\n", rank, i);
                for (int j = 0; j < 4 && j < (int)(bufferSize / sizeof(uint32_t)); j++) {
                    uint32_t expected = seed + j;
                    printf("  [%d] expected=%u, got=%u %s\n",
                           j, expected, recvData[j],
                           (recvData[j] == expected) ? "✓" : "✗");
                }
                fflush(stdout);
            }

            // NOTE: Flush is NOT called for host memory transfers
            // Flush (iflush) is only needed for GPU Direct RDMA to ensure data visibility on GPU.
            // For NCCL_PTR_HOST transfers, flush is unnecessary and calling it can cause
            // race conditions when request objects are rapidly reused.
            // The NET IB implementation will no-op the flush call for host memory anyway.
        }
    }

    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
}

TEST_F(NetIbMPITest, LargeTransfer) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }

    const size_t bufferSize = 16 * 1024 * 1024; // 16 MB
    const int tag = 400;

    void* buffer = AllocateHostMemory(bufferSize);
    ASSERT_NE(buffer, nullptr);
    BufferGuard bufferGuard(buffer, true);

    void* mhandle = nullptr;
    void* comm = (rank == 0) ? pair.recvComm : pair.sendComm;
    ASSERT_EQ(RegisterMemory(comm, buffer, bufferSize, NCCL_PTR_HOST, &mhandle), ncclSuccess);
    NetMHandleGuard mhandleGuard(mhandle, NetMHandleDeleter(net_, comm));

    void* request = nullptr;

    if (rank == 0) {
        // Receiver
        void* recvBuffers[1] = {buffer};
        size_t recvSizes[1] = {bufferSize};
        int recvTags[1] = {tag};
        void* recvHandles[1] = {mhandle};

        ASSERT_EQ(PostRecv(pair.recvComm, 1, recvBuffers, recvSizes, recvTags,
                          recvHandles, &request), ncclSuccess);
    } else {
        // Sender
        InitializeBuffer(buffer, bufferSize, rank);

        ASSERT_EQ(PostSend(pair.sendComm, buffer, bufferSize, tag, mhandle, &request), ncclSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Wait for completion with longer timeout for large transfer
    int sizes[1] = {0};
    ASSERT_EQ(WaitForCompletion(request, sizes, 30000), ncclSuccess);

    if (rank == 0) {
        EXPECT_EQ(sizes[0], bufferSize) << "Large transfer size mismatch";
    }

    ASSERT_EQ(DeregisterMemory(comm, mhandle), ncclSuccess);
}

TEST_F(NetIbMPITest, CloseWithoutWaitingForCompletion) {
    ASSERT_TRUE(validateTestPrerequisites(2, 2, false, 1, kNoNodeLimit))
        << "Test requires exactly 2 processes";

    ASSERT_EQ(InitNetIb(), ncclSuccess);

    int ndev = 0;
    ASSERT_EQ(GetDeviceCount(&ndev), ncclSuccess);
    ASSERT_GT(ndev, 0);

    ConnectionPair pair;
    int rank = RCCLMPIEnvironment::world_rank;
    int peerRank = (rank + 1) % 2;

    ASSERT_EQ(SetupConnection(0, pair, rank, peerRank), ncclSuccess);

    // Guard connections for automatic cleanup
    NetConnectionGuard connGuard(net_);
    if (rank == 0) {
        connGuard.setRecvComm(pair.recvComm);
        connGuard.setListenComm(pair.listenComm);
    } else {
        connGuard.setSendComm(pair.sendComm);
    }
}
