/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "p2p.h"
#include "rccl/rccl.h"
#include "shm.h"
#include "transport.h"
#include "register.h"

#include "gtest/gtest.h"
#include <iostream>
#include "comm.h"

#include "graph/topo.h"
#include "hip/hip_runtime.h"

enum p2pType { P2P_DIRECT, P2P_INTERMEDIATE, P2P_IPC, P2P_CUMEM };

struct ncclP2pBuff {
  void* directPtr;
  size_t size;
  ncclIpcDesc ipcDesc;
};

struct p2pIpcExpInfo {
  ncclIpcDesc ipcDesc;
  bool legacyIpcCap;
  int impFd;
  size_t size;
  uintptr_t offset;
};

struct p2pShm {
  struct ncclSendMem sendMem;
  struct ncclRecvMem recvMem;
};
struct p2pShmProxyInfo {
  // Shared memory between proxy and receiving GPU
  struct p2pShm* shm;
  struct p2pShm* devShm;
  ncclShmIpcDesc_t desc;

  // Intermediate step for sender
  struct ncclRecvMem* ceRecvMem;
  char* ceDevBuff;

  // Receiver buffer
  char* recvFifo;

  // Used by CE memcpy progress only
  uint64_t step;
  hipStream_t stream;
  hipEvent_t events[NCCL_STEPS];
};

struct p2pConnectInfo {
  int rank;
  int read;
  struct ncclP2pBuff p2pBuff;
  // Used by CE memcpy
  ncclShmIpcDesc_t desc;
};

struct p2pResources {
  enum p2pType type;
  union {
    struct ncclSendMem* sendDevMem;
    struct ncclRecvMem* recvDevMem;
  };
  void* sendMemIpc;
  int sendMemSameProc;
  void* recvMemIpc;
  int recvMemSameProc;
  // CE memcpy support
  struct p2pShmProxyInfo proxyInfo;
  struct p2pShm* shm;
  struct p2pShm* devShm;
  int shmSize;
  ncclShmHandle_t handle;
  ncclShmIpcDesc_t desc;
  uint32_t* next_hdp_reg;  // Next GPU in ring (for p2p transport use only)
};

struct ncclIpcCleanupCallback {
  struct ncclCommCallback base;
  struct ncclComm *comm;
  struct ncclReg *reg;
};

// ncclResult_t p2pCanConnect(int* ret, struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2);

// ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int refcount, ncclIpcDesc *ipcDesc, void **ptr);

// ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int peer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr);

// ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);

// ncclResult_t p2pMap(struct ncclComm *comm, struct ncclProxyConnector* proxyConn, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclP2pBuff* p2pBuff, void** devMem, void** ipcPtr);

// ncclResult_t p2pSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
// 	    struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex);

// ncclResult_t p2pRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo,
//       struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex);

// ncclResult_t ipcRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, struct ncclReg* regRecord, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, bool* isLegacyIpc);

// ncclResult_t cleanupIpc(struct ncclComm* comm, struct ncclCommCallback* cb);

// ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts);

// ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);

class P2pTest : public ::testing::Test {
protected:
  int deviceCount;
  std::vector<hipDeviceProp_t> props;

  void SetUp() override {
    ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
    ASSERT_GE(deviceCount, 3) << "At least three GPU required";
    props.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      ASSERT_EQ(hipGetDeviceProperties(&props[i], i), hipSuccess);
      std::cout << "Device " << i << ": " << props[i].name << std::endl;
      std::cout << "  pciBusID: " << props[i].pciBusID << std::endl;
      std::cout << "  pciDeviceID: " << props[i].pciDeviceID << std::endl;
    }
  }

  void setupCommAndPeers(struct ncclComm* comm, struct ncclTopoSystem* system, uint64_t hostHash, int shmDev, int rank, int cudaDev, bool hasFineGrain) {
    memset(comm, 0, sizeof(struct ncclComm));
    memset(system, 0, sizeof(struct ncclTopoSystem));
    comm->topo = system;
    comm->nRanks = 3;
    comm->rank = rank;
    comm->magic = NCCL_MAGIC; // Replace with the actual macro or value
    comm->regCache.pageSize = 4096;
    comm->peerInfo = (struct ncclPeerInfo*)calloc(3, sizeof(struct ncclPeerInfo));
    ASSERT_NE(comm->peerInfo, nullptr);

    comm->peerInfo[rank].rank = rank;
    comm->peerInfo[rank].hostHash = hostHash;
    comm->peerInfo[rank].pidHash = getpid();
    comm->peerInfo[rank].cudaDev = cudaDev;
    comm->peerInfo[rank].shmDev = shmDev;
    ASSERT_LT(cudaDev, props.size());

    comm->peerInfo[rank].busId = props[cudaDev].pciBusID;
    comm->peerInfo[rank].hasFineGrain = hasFineGrain;

    system->nodes[GPU].count = 3;

    for (int i = 0; i < 3; i++) {
      system->nodes[GPU].nodes[i].type = GPU;
      system->nodes[GPU].nodes[i].id = props[i].pciBusID;
      system->nodes[GPU].nodes[i].gpu.dev = i;
      system->nodes[GPU].nodes[i].gpu.rank = i;
      snprintf(system->nodes[GPU].nodes[i].gpu.gcn, sizeof(system->nodes[GPU].nodes[i].gpu.gcn), "gfx900");
      std::cout << "devBusId=" << props[i].pciBusID << std::endl;
      std::cout << "pciBusId=" << props[cudaDev].pciBusID << std::endl;
    }

    system->nodes[NET].count = 1;
    system->nodes[NET].nodes[0].type = NET;
    system->nodes[NET].nodes[0].id = 0x100; // Arbitrary NET id

    // Connect each GPU to NET
    for (int i = 0; i < 3; i++) {
      system->nodes[GPU].nodes[i].paths[NET] = (struct ncclTopoLinkList*)calloc(1, sizeof(struct ncclTopoLinkList));
      struct ncclTopoLink* link = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
      link->type = PATH_NET;
      link->remNode = &system->nodes[NET].nodes[0];
      system->nodes[GPU].nodes[i].paths[NET][0].count = 1;
      system->nodes[GPU].nodes[i].paths[NET][0].list[0] = link;
      system->nodes[GPU].nodes[i].paths[NET][0].type = PATH_PXB;
      system->nodes[GPU].nodes[i].paths[NET][0].bw = 200.0;
    }

    std::cout << "rank=" << rank << ", cudaDev=" << cudaDev << ", busId=" << comm->peerInfo[rank].busId << std::endl;
    std::cout << "setupCommAndPeers end" << std::endl;
  }

  void setupPaths(struct ncclTopoSystem* system, int pathType = PATH_PXB, float bw = 100.0) {
    int gpuCount = system->nodes[GPU].count;
    for (int i = 0; i < gpuCount; i++) {
      // Allocate paths array for each GPU node
      system->nodes[GPU].nodes[i].paths[GPU] = (struct ncclTopoLinkList*)calloc(gpuCount, sizeof(struct ncclTopoLinkList));
      for (int j = 0; j < gpuCount; j++) {
        if (i == j) continue;
        // Initialize path
        struct ncclTopoLink* link = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
        link->type = pathType;
        link->bw = bw;
        link->remNode = &system->nodes[GPU].nodes[j];

        struct ncclTopoLinkList* path = &system->nodes[GPU].nodes[i].paths[GPU][j];
        path->count = 1;
        path->list[0] = link;
        path->type = pathType;
        path->bw = bw;
      }
    }
  }

  void setupPathsWithIntermediateGpu(struct ncclTopoSystem* system, int gpuSrc, int gpuIntermediate, int gpuDst) {
    for (int i = 0; i < system->nodes[GPU].count; i++) {
      system->nodes[GPU].nodes[i].paths[GPU] = (struct ncclTopoLinkList*)calloc(system->nodes[GPU].count, sizeof(struct ncclTopoLinkList));
      for (int j = 0; j < system->nodes[GPU].count; j++) {
        system->nodes[GPU].nodes[i].paths[GPU][j].count = 0;
        memset(system->nodes[GPU].nodes[i].paths[GPU][j].list, 0, sizeof(system->nodes[GPU].nodes[i].paths[GPU][j].list));
      }
    }

    std::cout << "1. gpuSrc=" << gpuSrc << ", gpuIntermediate=" << gpuIntermediate << ", gpuDst=" << gpuDst << std::endl;

    // gpuSrc -> gpuIntermediate
    struct ncclTopoLink* link1 = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
    link1->type = PATH_PXB;
    link1->remNode = &system->nodes[GPU].nodes[gpuIntermediate];

    std::cout << "2. gpuSrc=" << gpuSrc << ", gpuIntermediate=" << gpuIntermediate << ", gpuDst=" << gpuDst << std::endl;

    // gpuIntermediate -> gpuDst
    struct ncclTopoLink* link2 = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
    link2->type = PATH_PXB;
    link2->remNode = &system->nodes[GPU].nodes[gpuDst];

    std::cout << "3. gpuSrc=" << gpuSrc << ", gpuIntermediate=" << gpuIntermediate << ", gpuDst=" << gpuDst << std::endl;

    // Set path from gpuSrc to gpuDst via gpuIntermediate
    struct ncclTopoLinkList* pathSrcDst = &system->nodes[GPU].nodes[gpuSrc].paths[GPU][gpuDst];
    pathSrcDst->count = 2;
    pathSrcDst->list[0] = link1;
    pathSrcDst->list[1] = link2;
    pathSrcDst->type = PATH_PXB;

    std::cout << "4. gpuSrc=" << gpuSrc << ", gpuIntermediate=" << gpuIntermediate << ", gpuDst=" << gpuDst << std::endl;

    // Set direct path from gpuIntermediate to gpuDst
    struct ncclTopoLinkList* intermediatePath = &system->nodes[GPU].nodes[gpuIntermediate].paths[GPU][gpuDst];
    intermediatePath->count = 2;
    intermediatePath->list[0] = link2;
    intermediatePath->type = PATH_PXB;

    std::cout << "5. gpuSrc=" << gpuSrc << ", gpuIntermediate=" << gpuIntermediate << ", gpuDst=" << gpuDst << std::endl;
  }

  void cleanupPaths(struct ncclTopoSystem* system) {
    for (int src = 0; src < system->nodes[GPU].count; src++) {
      if (system->nodes[GPU].nodes[src].paths[GPU]) {
        for (int dst = 0; dst < system->nodes[GPU].count; dst++) {
          if (src == dst) continue;
          struct ncclTopoLinkList* path = &system->nodes[GPU].nodes[src].paths[GPU][dst];
          for (int linkIdx = 0; linkIdx < path->count; linkIdx++) {
            if (path->list[linkIdx]) {
              // free(path->list[linkIdx]);
              path->list[linkIdx] = nullptr;
            }
          }
        }
        free(system->nodes[GPU].nodes[src].paths[GPU]);
        system->nodes[GPU].nodes[src].paths[GPU] = nullptr;
      }
    }
  }

};

/*
TEST_F(P2pTest, P2pAllocateShareableBuffer) {
  // Setup variables
  size_t size = 1024;
  ncclIpcDesc desc;
  void *hptr = nullptr;
  ncclResult_t result;

  // Test 1: All valid parameters - should succeed
  result = ncclP2pAllocateShareableBuffer(size, 0, &desc, &hptr);
  EXPECT_EQ(result, ncclSuccess);

  // Make sure to free memory from successful allocation
  if (hptr) {
    hipFree(hptr);
    hptr = nullptr;
  }

  // Test 2: NULL desc
  result = ncclP2pAllocateShareableBuffer(size, 0, nullptr, &hptr);
  EXPECT_EQ(result, (ncclResult_t)hipErrorInvalidValue);

}

TEST_F(P2pTest, P2pFreeShareableBuffer) {
  // Setup variables
  ncclIpcDesc desc;

  // Test 1: All valid parameters - should succeed
  ncclResult_t result = ncclP2pFreeShareableBuffer(&desc);
  EXPECT_EQ(result, (ncclResult_t)hipSuccess);

}

TEST_F(P2pTest, P2pMap_DeviceEnablePeerAccessFailure) {
  // Skip test if we don't have at least 2 devices
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);

  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
    return;
  }

  // Setup test structures
  struct ncclComm comm;
  struct ncclProxyConnector proxyConn;
  struct ncclPeerInfo myInfo, peerInfo;
  struct ncclP2pBuff p2pBuff;
  void* devMem = nullptr;
  void* ipcPtr = nullptr;
  ncclResult_t result;

  // Initialize with zeroes
  memset(&comm, 0, sizeof(comm));
  memset(&proxyConn, 0, sizeof(proxyConn));
  memset(&myInfo, 0, sizeof(myInfo));
  memset(&peerInfo, 0, sizeof(peerInfo));
  memset(&p2pBuff, 0, sizeof(p2pBuff));

  // Configure peer info to have same PID but different devices
  myInfo.hostHash = 0x12345678;
  myInfo.pidHash = 0x87654321;
  myInfo.cudaDev = 0;  // First GPU

  peerInfo.hostHash = 0x12345678;  // Same host
  peerInfo.pidHash = 0x87654321;   // Same PID

  // Create some memory for p2pBuff
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMalloc(&p2pBuff.directPtr, 1024), hipSuccess);
  p2pBuff.size = 1024;

  // Get properties of the available GPUs
  std::vector<hipDeviceProp_t> props(deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    ASSERT_EQ(hipGetDeviceProperties(&props[i], i), hipSuccess);
  }

  // Find a device that cannot peer with device 0
  int incompatibleDevice = -1;
  for (int i = 1; i < deviceCount; i++) {
    int canAccessPeer = 0;
    ASSERT_EQ(hipDeviceCanAccessPeer(&canAccessPeer, 0, i), hipSuccess);
    if (!canAccessPeer) {
      incompatibleDevice = i;
      break;
    }
  }

  // If we can't find an incompatible device, force the failure by using an invalid device ID
  if (incompatibleDevice == -1) {
    peerInfo.cudaDev = 999;  // Invalid device ID
  } else {
    peerInfo.cudaDev = incompatibleDevice;
  }

  // Set current device to 0 to ensure we're in the right context
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  // Call the function under test - should fail at the hipDeviceEnablePeerAccess call
  result = p2pMap(&comm, &proxyConn, &myInfo, &peerInfo, &p2pBuff, &devMem, &ipcPtr);

  // Verify that the function returned an error
  EXPECT_EQ(result, ncclInternalError);

  // Clean up
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipFree(p2pBuff.directPtr), hipSuccess);
}

TEST_F(P2pTest, P2pCanConnectForkTest)
{
  // Skip test if we don't have at least 2 devices
  int deviceCount = 0;
  ASSERT_EQ(hipGetDeviceCount(&deviceCount), hipSuccess);
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires at least 2 GPUs";
    return;
  }

  // Create a pipe to communicate between parent and child
  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1) << "Failed to create pipe";

  // Fork the process
  pid_t childPid = fork();
  ASSERT_NE(childPid, -1) << "Failed to fork process";

  // Get device properties for all GPUs before we fork
  std::vector<hipDeviceProp_t> props(deviceCount);
  for (int i = 0; i < deviceCount; i++) {
    ASSERT_EQ(hipGetDeviceProperties(&props[i], i), hipSuccess);
  }

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    // Child process
    close(pipefd[1]); // Close write end

    // Setup structures for child process
    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    memset(&comm, 0, sizeof(comm));
    memset(&graph, 0, sizeof(graph));
    memset(&system, 0, sizeof(system));

    comm.topo = &system;
    comm.nRanks = 2;
    comm.rank = 1; // Child is rank 1

    // Set up peer info
    comm.peerInfo = (struct ncclPeerInfo*)calloc(2, sizeof(struct ncclPeerInfo));

    // Setup my info (rank 1)
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash;
    comm.peerInfo[1].pidHash = getpid(); // Child's PID hash
    comm.peerInfo[1].cudaDev = 1;
    comm.peerInfo[1].shmDev = 0; // Same shmDev for both ranks
    comm.peerInfo[1].busId = props[1].pciBusID;
    comm.peerInfo[1].hasFineGrain = true; // Set hasFineGrain to true

    // Wait for parent process to send their data
    struct ncclPeerInfo peerInfo;
    read(pipefd[0], &peerInfo, sizeof(peerInfo)); // FIXED: was using pipefd[1] incorrectly

    // Set parent's peer info
    comm.peerInfo[0] = peerInfo;

    // Double check hasFineGrain values before testing
    std::cout << "Child process: info1->hasFineGrain=" << comm.peerInfo[0].hasFineGrain
              << ", info2->hasFineGrain=" << comm.peerInfo[1].hasFineGrain << std::endl;

    // Initialize paths between GPUs
    for (int src = 0; src < 2; src++) {
      // Allocate memory for paths
      system.nodes[GPU].nodes[src].paths[GPU] =
          (struct ncclTopoLinkList*)calloc(2, sizeof(struct ncclTopoLinkList));

      for (int dst = 0; dst < 2; dst++) {
        if (src == dst) continue;

        // Create link and path
        struct ncclTopoLink* link = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
        link->type = PATH_PXB;  // PCIe connection
        link->bw = 100.0;
        link->remNode = &system.nodes[GPU].nodes[dst];

        // Set path properties
        system.nodes[GPU].nodes[src].paths[GPU][dst].count = 1;
        system.nodes[GPU].nodes[src].paths[GPU][dst].list[0] = link;
        system.nodes[GPU].nodes[src].paths[GPU][dst].type = PATH_PXB;
        system.nodes[GPU].nodes[src].paths[GPU][dst].bw = 100.0;
      }
    }

    // Test p2pCanConnect between the two processes
    int ret = 0;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1], &comm.peerInfo[0]);

    // Write result back to parent
    write(pipefd[0], &result, sizeof(result)); // FIXED: was using pipefd[1] incorrectly
    write(pipefd[0], &ret, sizeof(ret));       // FIXED: was using pipefd[1] incorrectly
    close(pipefd[0]);

    // Clean up
    for (int src = 0; src < 2; src++) {
      for (int dst = 0; dst < 2; dst++) {
        if (src == dst) continue;
        free(system.nodes[GPU].nodes[src].paths[GPU][dst].list[0]);
      }
      free(system.nodes[GPU].nodes[src].paths[GPU]);
    }
    free(comm.peerInfo);

    exit(0);
  } else {
    // Parent process
    close(pipefd[0]); // Close read end - FIXED: was incorrectly closing write end

    // Setup structures for parent process
    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    memset(&comm, 0, sizeof(comm));
    memset(&graph, 0, sizeof(graph));
    memset(&system, 0, sizeof(system));

    comm.topo = &system;
    comm.nRanks = 2;
    comm.rank = 0; // Parent is rank 0

    // Set up peer info
    comm.peerInfo = (struct ncclPeerInfo*)calloc(2, sizeof(struct ncclPeerInfo));

    // Setup my info (rank 0)
    comm.peerInfo[0].rank = 0;
    comm.peerInfo[0].hostHash = hostHash;
    comm.peerInfo[0].pidHash = getpid(); // Parent's PID hash
    comm.peerInfo[0].cudaDev = 0;
    comm.peerInfo[0].shmDev = 0; // Same shmDev for both ranks
    comm.peerInfo[0].busId = props[0].pciBusID;
    comm.peerInfo[0].hasFineGrain = true; // Set hasFineGrain to true

    // Send my info to child
    write(pipefd[1], &comm.peerInfo[0], sizeof(struct ncclPeerInfo));

    // Set up system nodes (minimal needed for topology traversal)
    system.nodes[GPU].count = 2;

    // Initialize GPU nodes with real device data
    system.nodes[GPU].nodes[0].type = GPU;
    system.nodes[GPU].nodes[0].id = props[0].pciBusID;
    system.nodes[GPU].nodes[0].gpu.dev = 0;
    system.nodes[GPU].nodes[0].gpu.rank = 0;
    snprintf(system.nodes[GPU].nodes[0].gpu.gcn, sizeof(system.nodes[GPU].nodes[0].gpu.gcn), "gfx900");

    system.nodes[GPU].nodes[1].type = GPU;
    system.nodes[GPU].nodes[1].id = props[1].pciBusID;
    system.nodes[GPU].nodes[1].gpu.dev = 1;
    system.nodes[GPU].nodes[1].gpu.rank = 1;
    snprintf(system.nodes[GPU].nodes[1].gpu.gcn, sizeof(system.nodes[GPU].nodes[1].gpu.gcn), "gfx900");

    // Initialize paths between GPUs
    for (int src = 0; src < 2; src++) {
      // Allocate memory for paths
      system.nodes[GPU].nodes[src].paths[GPU] =
          (struct ncclTopoLinkList*)calloc(2, sizeof(struct ncclTopoLinkList));

      for (int dst = 0; dst < 2; dst++) {
        if (src == dst) continue;

        // Create link and path
        struct ncclTopoLink* link = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
        link->type = PATH_PXB;  // PCIe connection
        link->bw = 100.0;
        link->remNode = &system.nodes[GPU].nodes[dst];

        // Set path properties
        system.nodes[GPU].nodes[src].paths[GPU][dst].count = 1;
        system.nodes[GPU].nodes[src].paths[GPU][dst].list[0] = link;
        system.nodes[GPU].nodes[src].paths[GPU][dst].type = PATH_PXB;
        system.nodes[GPU].nodes[src].paths[GPU][dst].bw = 100.0;
      }
    }

    // Child needs to send rank 1 info to parent process
    struct ncclPeerInfo childInfo;
    childInfo.rank = 1;
    childInfo.hostHash = hostHash;
    childInfo.busId = props[1].pciBusID;
    childInfo.cudaDev = 1;
    childInfo.hasFineGrain = true;  // Set hasFineGrain to true for rank 1
    comm.peerInfo[1] = childInfo;

    // Double check hasFineGrain values before testing
    std::cout << "Parent process: info1->hasFineGrain=" << comm.peerInfo[0].hasFineGrain
              << ", info2->hasFineGrain=" << comm.peerInfo[1].hasFineGrain << std::endl;

    // Wait for child results
    ncclResult_t childResult;
    int childRet;
    read(pipefd[1], &childResult, sizeof(childResult)); // FIXED: was using pipefd[0] incorrectly
    read(pipefd[1], &childRet, sizeof(childRet));       // FIXED: was using pipefd[0] incorrectly
    close(pipefd[1]);

    // Now test our p2pCanConnect as well
    int ret = 0;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[0], &comm.peerInfo[1]);

    // Check results
    EXPECT_EQ(result, ncclSuccess) << "Parent process p2pCanConnect failed with " << result;
    EXPECT_EQ(childResult, ncclSuccess) << "Child process p2pCanConnect failed with " << childResult;

    // Clean up
    for (int src = 0; src < 2; src++) {
      for (int dst = 0; dst < 2; dst++) {
        if (src == dst) continue;
        free(system.nodes[GPU].nodes[src].paths[GPU][dst].list[0]);
      }
      free(system.nodes[GPU].nodes[src].paths[GPU]);
    }
    free(comm.peerInfo);

    // Wait for child to complete
    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTest, P2pCanConnectShmDevDiffForkTest)
{
  if (deviceCount < 3) {
    GTEST_SKIP() << "Test requires at least three GPUs";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    close(pipefd[1]); // Child reads from pipefd[0]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 1, 2, true); // child rank 1, shmDev=1, cudaDev=2

    // Read parent's peer info
    read(pipefd[0], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    std::cout << "Parent hostHash=" << comm.peerInfo[0].hostHash
              << ", child hostHash=" << comm.peerInfo[1].hostHash << std::endl;

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 2 -> GPU 1 -> GPU 0)
    setupPathsWithIntermediateGpu(&system, 2, 1, 0);

    std::cout << "Intermediate GPU: " << comm.peerInfo[1].shmDev << std::endl;

    int ret = -1;
    int intermediateRank = -1;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1], &comm.peerInfo[0]);

    // Send results back to parent
    write(pipefd[0], &result, sizeof(result));
    write(pipefd[0], &ret, sizeof(ret));
    write(pipefd[0], &intermediateRank, sizeof(intermediateRank));

    cleanupPaths(&system);
    close(pipefd[0]);
    free(comm.peerInfo);
    exit(0);
  } else {
    close(pipefd[0]); // Parent writes to pipefd[1]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 0, 0, true); // parent rank 0, shmDev=0, cudaDev=0

    // Setup child's peer info explicitly
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash; // Different host hash
    comm.peerInfo[1].pidHash = childPid;
    comm.peerInfo[1].cudaDev = 2;
    // comm.peerInfo[1].shmDev = 1; // Different shmDev
    comm.peerInfo[1].busId = props[2].pciBusID;
    comm.peerInfo[1].hasFineGrain = true;

    // Send both peer infos to child
    write(pipefd[1], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 0 -> GPU 1 -> GPU 2)
    setupPathsWithIntermediateGpu(&system, 0, 1, 2);

    ncclResult_t result;
    int ret;
    int intermediateRank;
    read(pipefd[1], &result, sizeof(result));
    read(pipefd[1], &ret, sizeof(ret));
    read(pipefd[1], &intermediateRank, sizeof(intermediateRank));

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(ret, 0) << "P2P should be disabled due to different host hashes";
    EXPECT_NE(intermediateRank, -1) << "Intermediate GPU should be set";

    cleanupPaths(&system);
    close(pipefd[1]);
    free(comm.peerInfo);

    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTest, P2pCanConnectHostHashDiffForkTest)
{
  if (deviceCount < 1) {
    GTEST_SKIP() << "Test requires at least one GPU";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Use different host hashes for all
  uint64_t parentHostHash = 0x1234567890ABCDEF;
  uint64_t childHostHash  = 0xFEDCBA0987654321;
  uint64_t info2HostHash  = 0x1111111111111111;

  if (childPid == 0) {
    // Child process
    close(pipefd[0]); // Close read end

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    // Child rank 1, with unique hostHash
    setupCommAndPeers(&comm, &system, childHostHash, 0, 1, 0, true);

    // Receive parent's peer info (rank 0)
    read(pipefd[1], comm.peerInfo, sizeof(struct ncclPeerInfo));
    // Set hasFineGrain to true for parent info (rank 0)
    comm.peerInfo[0].hasFineGrain = true;

    // Prepare info2 with a third, different hostHash
    struct ncclPeerInfo info2 = comm.peerInfo[0];
    info2.hostHash = info2HostHash;

    int ret = -1;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1], &info2);

    // Send result back to parent
    write(pipefd[1], &result, sizeof(result));
    write(pipefd[1], &ret, sizeof(ret));

    free(comm.peerInfo);
    close(pipefd[1]);
    exit(0);

  } else {
    // Parent process
    close(pipefd[1]); // Close write end

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    // Parent rank 0, with unique hostHash
    setupCommAndPeers(&comm, &system, parentHostHash, 0, 0, 0, true);
    // Set hasFineGrain to true for child info (rank 1)
    comm.peerInfo[1].hasFineGrain = true;

    // Send parent info to child (only rank 0 info)
    write(pipefd[0], &comm.peerInfo[0], sizeof(struct ncclPeerInfo));

    // Receive result from child
    ncclResult_t result;
    int ret;
    read(pipefd[0], &result, sizeof(result));
    read(pipefd[0], &ret, sizeof(ret));

    std::cout << "Different hostHash test result: " << result << ", ret=" << ret << std::endl;
    EXPECT_EQ(result, ncclSuccess) << "p2pCanConnect should return success with different hostHashes";
    // ret is not set in this branch, so its value is undefined; do not check ret

    free(comm.peerInfo);
    close(pipefd[0]);

    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTest, P2pCanConnectWithIntermediateRankForkTest)
{
  if (deviceCount < 3) {
    GTEST_SKIP() << "Test requires at least three GPUs";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    close(pipefd[1]); // Child reads from pipefd[0]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 1, 2, true); // child rank 1, shmDev=1, cudaDev=2

    // Read parent's peer info
    read(pipefd[0], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    std::cout << "Parent hostHash=" << comm.peerInfo[0].hostHash
              << ", child hostHash=" << comm.peerInfo[1].hostHash << std::endl;

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 2 -> GPU 1 -> GPU 0)
    setupPathsWithIntermediateGpu(&system, 2, 1, 0);

    std::cout << "Intermediate GPU: " << comm.peerInfo[1].shmDev << std::endl;

    int ret = -1;
    int intermediateRank = -1;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1], &comm.peerInfo[0]);

    // Send results back to parent
    write(pipefd[0], &result, sizeof(result));
    write(pipefd[0], &ret, sizeof(ret));
    write(pipefd[0], &intermediateRank, sizeof(intermediateRank));

    cleanupPaths(&system);
    close(pipefd[0]);
    free(comm.peerInfo);
    exit(0);
  } else {
    close(pipefd[0]); // Parent writes to pipefd[1]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 0, 0, 0, true); // parent rank 0, shmDev=0, cudaDev=0

    // Setup child's peer info explicitly
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash; // Different host hash
    comm.peerInfo[1].pidHash = childPid;
    comm.peerInfo[1].cudaDev = 2;
    // comm.peerInfo[1].shmDev = 1; // Different shmDev
    comm.peerInfo[1].busId = props[2].pciBusID;
    comm.peerInfo[1].hasFineGrain = true;

    // Send both peer infos to child
    write(pipefd[1], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup paths explicitly creating an intermediate GPU scenario (GPU 0 -> GPU 1 -> GPU 2)
    setupPathsWithIntermediateGpu(&system, 0, 1, 2);

    ncclResult_t result;
    int ret;
    int intermediateRank;
    read(pipefd[1], &result, sizeof(result));
    read(pipefd[1], &ret, sizeof(ret));
    read(pipefd[1], &intermediateRank, sizeof(intermediateRank));

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(ret, 0) << "P2P should be disabled due to different host hashes";
    EXPECT_NE(intermediateRank, -1) << "Intermediate GPU should be set";

    cleanupPaths(&system);
    close(pipefd[1]);
    free(comm.peerInfo);

    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTest, P2pCanConnectNetPreferredForkTest)
{
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires at least two GPUs";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    close(pipefd[1]); // Child reads from pipefd[0]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 1, 2, true); // child rank 1, shmDev=1, cudaDev=2

    // Read parent's peer info
    read(pipefd[0], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    std::cout << "Parent hostHash=" << comm.peerInfo[0].hostHash
              << ", child hostHash=" << comm.peerInfo[1].hostHash << std::endl;

    setupPaths(&system);

    std::cout << "Intermediate GPU: " << comm.peerInfo[1].shmDev << std::endl;

    int ret = -1;
    int intermediateRank = -1;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1], &comm.peerInfo[0]);

    // Send results back to parent
    write(pipefd[0], &result, sizeof(result));
    write(pipefd[0], &ret, sizeof(ret));
    write(pipefd[0], &intermediateRank, sizeof(intermediateRank));

    cleanupPaths(&system);
    close(pipefd[0]);
    free(comm.peerInfo);
    exit(0);
  } else {
    close(pipefd[0]); // Parent writes to pipefd[1]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 0, 0, 0, true); // parent rank 0, shmDev=0, cudaDev=0

    // Setup child's peer info explicitly
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash; // Different host hash
    comm.peerInfo[1].pidHash = childPid;
    comm.peerInfo[1].cudaDev = 2;
    comm.peerInfo[1].busId = props[2].pciBusID;
    comm.peerInfo[1].hasFineGrain = true;

    // Send both peer infos to child
    write(pipefd[1], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    setupPaths(&system);

    ncclResult_t result;
    int ret;
    int intermediateRank;
    read(pipefd[1], &result, sizeof(result));
    read(pipefd[1], &ret, sizeof(ret));
    read(pipefd[1], &intermediateRank, sizeof(intermediateRank));

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(ret, 0) << "P2P should be disabled due to different host hashes";
    EXPECT_NE(intermediateRank, -1) << "Intermediate GPU should be set";

    cleanupPaths(&system);
    close(pipefd[1]);
    free(comm.peerInfo);

    int status;
    waitpid(childPid, &status, 0);
  }
}

TEST_F(P2pTest, P2pCanConnectInvalidBusID)
{
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires at least two GPUs";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  // Get current host hash (same for parent and child)
  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    close(pipefd[1]); // Child reads from pipefd[0]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 1, 1, 2, true); // child rank 1, shmDev=1, cudaDev=2

    // Read parent's peer info
    read(pipefd[0], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    std::cout << "Parent hostHash=" << comm.peerInfo[0].hostHash
              << ", child hostHash=" << comm.peerInfo[1].hostHash << std::endl;

    setupPaths(&system);

    int ret = -1;
    int intermediateRank = -1;
    ncclResult_t result = p2pCanConnect(&ret, &comm, &graph, &comm.peerInfo[1], &comm.peerInfo[0]);

    // Send results back to parent
    write(pipefd[0], &result, sizeof(result));
    write(pipefd[0], &ret, sizeof(ret));
    write(pipefd[0], &intermediateRank, sizeof(intermediateRank));

    cleanupPaths(&system);
    close(pipefd[0]);
    free(comm.peerInfo);
    exit(0);
  } else {
    close(pipefd[0]); // Parent writes to pipefd[1]

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    setupCommAndPeers(&comm, &system, hostHash, 0, 0, 0, true); // parent rank 0, shmDev=0, cudaDev=0

    // Setup child's peer info explicitly
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash;
    comm.peerInfo[1].pidHash = childPid;
    comm.peerInfo[1].cudaDev = 2;
    comm.peerInfo[1].busId = props[2].pciDeviceID; // Invalid bus ID
    comm.peerInfo[1].hasFineGrain = true;

    // Send both peer infos to child
    write(pipefd[1], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    setupPaths(&system);

    ncclResult_t result;
    int ret;
    int intermediateRank;
    read(pipefd[1], &result, sizeof(result));
    read(pipefd[1], &ret, sizeof(ret));

    EXPECT_EQ(result, ncclSuccess);
    EXPECT_EQ(ret, 0);

    cleanupPaths(&system);
    close(pipefd[1]);
    free(comm.peerInfo);

    int status;
    waitpid(childPid, &status, 0);
  }
}
*/

// TEST_F(P2pTest, P2pSendSetup_LinkTypeError) {
//   // This test covers the error branch in p2pSendSetup when ncclTopoGetLinkType fails (line 407)
//   // Setup comm and topo for two ranks, but intentionally break the topology to trigger the error

//   struct ncclComm comm;
//   struct ncclTopoGraph graph;
//   struct ncclTopoSystem system;
//   setupCommAndPeers(&comm, &system, /*hostHash=*/0x1234, /*shmDev=*/0, /*rank=*/0, /*cudaDev=*/0, /*hasFineGrain=*/true);
//   comm.nRanks = 2;

//   // Setup peer info for rank 1
//   comm.peerInfo[1].rank = 1;
//   comm.peerInfo[1].hostHash = 0x1234;
//   comm.peerInfo[1].pidHash = comm.peerInfo[0].pidHash; // same PID
//   comm.peerInfo[1].cudaDev = 1;
//   comm.peerInfo[1].shmDev = 0;
//   comm.peerInfo[1].busId = props[1].pciBusID;
//   comm.peerInfo[1].hasFineGrain = true;

//   // Setup topology for 2 GPUs, but do NOT create any paths (or use invalid cudaDev above)
//   system.nodes[GPU].count = 2;
//   for (int i = 0; i < 2; i++) {
//     system.nodes[GPU].nodes[i].type = GPU;
//     system.nodes[GPU].nodes[i].id = props[i].pciBusID;
//     system.nodes[GPU].nodes[i].gpu.dev = i;
//     system.nodes[GPU].nodes[i].gpu.rank = i;
//     snprintf(system.nodes[GPU].nodes[i].gpu.gcn, sizeof(system.nodes[GPU].nodes[i].gpu.gcn), "gfx900");
//     // Do NOT allocate paths to simulate missing topology
//     system.nodes[GPU].nodes[i].paths[GPU] = nullptr;
//   }

//   // For a system with 2 GPUs:
//   system.nodes[GPU].count = 2;
//   for (int i = 0; i < 2; i++) {
//       system.nodes[GPU].nodes[i].paths[GPU] = (struct ncclTopoLinkList*)calloc(2, sizeof(struct ncclTopoLinkList));
//       for (int j = 0; j < 2; j++) {
//           if (i == j) continue;
//           struct ncclTopoLink* link = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
//           link->type = LINK_NVL;
//           link->bw = 100.0;
//           link->remNode = &system.nodes[GPU].nodes[j];
//           struct ncclTopoLinkList* path = &system.nodes[GPU].nodes[i].paths[GPU][j];
//           path->count = 1;
//           path->list[0] = link;
//           path->type = LINK_NVL;
//           path->bw = 100.0;
//       }
//   }

//   struct ncclPeerInfo* myInfo = &comm.peerInfo[0];
//   struct ncclPeerInfo* peerInfo = &comm.peerInfo[1];

//   struct ncclConnect connectInfo;
//   struct ncclConnector send;
//   memset(&send, 0, sizeof(send));

//   comm.topParentRanks = (int*)calloc(comm.nRanks, sizeof(int));
//   for (int i = 0; i < comm.nRanks; ++i) comm.topParentRanks[i] = i;

//   // Should fail at ncclTopoGetLinkType and return ncclInternalError
//   ncclResult_t result = p2pSendSetup(&comm, &graph, myInfo, peerInfo, &connectInfo, &send, /*channelId=*/0, /*connIndex=*/0);
//   EXPECT_EQ(result, ncclInternalError);

//   // Clean up
//   free(comm.peerInfo);
//   free(comm.topParentRanks);
// }

/*
TEST_F(P2pTest, P2pSendSetupTest) {
  if (deviceCount < 2) {
    GTEST_SKIP() << "Test requires at least two GPUs";
    return;
  }

  int pipefd[2];
  ASSERT_NE(pipe(pipefd), -1);

  pid_t childPid = fork();
  ASSERT_NE(childPid, -1);

  uint64_t hostHash = gethostid();

  if (childPid == 0) {
    // --- Child process: rank 1 ---
    close(pipefd[1]); // Close write end

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    memset(&comm, 0, sizeof(comm));
    memset(&graph, 0, sizeof(graph));
    memset(&system, 0, sizeof(system));

    comm.nRanks = 2;
    comm.rank = 1;
    comm.topo = &system;
    comm.peerInfo = (struct ncclPeerInfo*)calloc(2, sizeof(struct ncclPeerInfo));

    // Read parent's peer info
    read(pipefd[0], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup my info (rank 1)
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash;
    comm.peerInfo[1].pidHash = getpid();
    comm.peerInfo[1].cudaDev = 1;
    comm.peerInfo[1].shmDev = 0;
    comm.peerInfo[1].busId = props[1].pciBusID;
    comm.peerInfo[1].hasFineGrain = true;

    // Setup topology: direct path between GPU 0 and GPU 1
    system.nodes[GPU].count = 2;
    for (int i = 0; i < 2; i++) {
      system.nodes[GPU].nodes[i].type = GPU;
      system.nodes[GPU].nodes[i].id = props[i].pciBusID;
      system.nodes[GPU].nodes[i].gpu.dev = i;
      system.nodes[GPU].nodes[i].gpu.rank = i;
      snprintf(system.nodes[GPU].nodes[i].gpu.gcn, sizeof(system.nodes[GPU].nodes[i].gpu.gcn), "gfx90a");
      system.nodes[GPU].nodes[i].paths[GPU] = (struct ncclTopoLinkList*)calloc(2, sizeof(struct ncclTopoLinkList));
      for (int j = 0; j < 2; j++) {
        if (i == j) continue;
        struct ncclTopoLink* link = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
        link->type = LINK_NVL;
        link->bw = 100.0;
        link->remNode = &system.nodes[GPU].nodes[j];
        struct ncclTopoLinkList* path = &system.nodes[GPU].nodes[i].paths[GPU][j];
        path->count = 1;
        path->list[0] = link;
        path->type = PATH_PXB;
        path->bw = 100.0;
      }
    }

    comm.topParentRanks = (int*)calloc(2, sizeof(int));
    for (int i = 0; i < 2; ++i) comm.topParentRanks[i] = i;

    // Allocate and initialize proxy state
    comm.proxyState = (struct ncclProxyState*)calloc(1, sizeof(struct ncclProxyState));
    comm.sharedRes = (struct ncclSharedResources*)calloc(1, sizeof(struct ncclSharedResources));
    comm.sharedRes->proxyState = comm.proxyState;
    comm.sharedRes->tpNLocalRanks = comm.nRanks;
    comm.sharedRes->tpRankToLocalRank = (int*)calloc(comm.nRanks, sizeof(int));
    for (int i = 0; i < comm.nRanks; ++i) comm.sharedRes->tpRankToLocalRank[i] = i;
    comm.sharedRes->magic = 0x1234;

    int n = comm.sharedRes->tpNLocalRanks;
    comm.proxyState->peerSocks = (struct ncclSocket*)calloc(n, sizeof(struct ncclSocket));
    comm.proxyState->proxyOps = (struct ncclProxyOps*)calloc(n, sizeof(struct ncclProxyOps));
    comm.proxyState->sharedDevMems = (void**)calloc(n, sizeof(void*));
    for (int i = 0; i < n; ++i) ncclSocketSetFd(-1, &comm.proxyState->peerSocks[i]);

    comm.abortFlag = (uint32_t*)calloc(1, sizeof(uint32_t));
    *comm.abortFlag = 0;

    struct ncclPeerInfo* myInfo = &comm.peerInfo[1];
    struct ncclPeerInfo* peerInfo = &comm.peerInfo[0];

    struct ncclConnect connectInfo;
    struct ncclConnector send;
    memset(&send, 0, sizeof(send));

    ncclResult_t result = p2pSendSetup(&comm, &graph, myInfo, peerInfo, &connectInfo, &send, 0, 0);

    // Send result to parent
    write(pipefd[0], &result, sizeof(result));
    close(pipefd[0]);

    // Clean up
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        if (i == j) continue;
        free(system.nodes[GPU].nodes[i].paths[GPU][j].list[0]);
      }
      free(system.nodes[GPU].nodes[i].paths[GPU]);
    }
    free(comm.peerInfo);
    free(comm.topParentRanks);
    exit(0);
  } else {
    // --- Parent process: rank 0 ---
    close(pipefd[0]); // Close read end

    struct ncclComm comm;
    struct ncclTopoGraph graph;
    struct ncclTopoSystem system;

    memset(&comm, 0, sizeof(comm));
    memset(&graph, 0, sizeof(graph));
    memset(&system, 0, sizeof(system));

    comm.nRanks = 2;
    comm.rank = 0;
    comm.topo = &system;
    comm.peerInfo = (struct ncclPeerInfo*)calloc(2, sizeof(struct ncclPeerInfo));

    // Setup my info (rank 0)
    comm.peerInfo[0].rank = 0;
    comm.peerInfo[0].hostHash = hostHash;
    comm.peerInfo[0].pidHash = getpid();
    comm.peerInfo[0].cudaDev = 0;
    comm.peerInfo[0].shmDev = 0;
    comm.peerInfo[0].busId = props[0].pciBusID;
    comm.peerInfo[0].hasFineGrain = true;

    // Setup peer info for rank 1 (will be overwritten by child, but needed for send)
    comm.peerInfo[1].rank = 1;
    comm.peerInfo[1].hostHash = hostHash;
    comm.peerInfo[1].pidHash = 0;
    comm.peerInfo[1].cudaDev = 1;
    comm.peerInfo[1].shmDev = 0;
    comm.peerInfo[1].busId = props[1].pciBusID;
    comm.peerInfo[1].hasFineGrain = true;

    // Send both peer infos to child
    write(pipefd[1], comm.peerInfo, 2 * sizeof(struct ncclPeerInfo));

    // Setup topology: direct path between GPU 0 and GPU 1
    system.nodes[GPU].count = 2;
    for (int i = 0; i < 2; i++) {
      system.nodes[GPU].nodes[i].type = GPU;
      system.nodes[GPU].nodes[i].id = props[i].pciBusID;
      system.nodes[GPU].nodes[i].gpu.dev = i;
      system.nodes[GPU].nodes[i].gpu.rank = i;
      snprintf(system.nodes[GPU].nodes[i].gpu.gcn, sizeof(system.nodes[GPU].nodes[i].gpu.gcn), "gfx900");
      system.nodes[GPU].nodes[i].paths[GPU] = (struct ncclTopoLinkList*)calloc(2, sizeof(struct ncclTopoLinkList));
      for (int j = 0; j < 2; j++) {
        if (i == j) continue;
        struct ncclTopoLink* link = (struct ncclTopoLink*)calloc(1, sizeof(struct ncclTopoLink));
        link->type = LINK_NVL;
        link->bw = 100.0;
        link->remNode = &system.nodes[GPU].nodes[j];
        struct ncclTopoLinkList* path = &system.nodes[GPU].nodes[i].paths[GPU][j];
        path->count = 1;
        path->list[0] = link;
        path->type = PATH_PXB;
        path->bw = 100.0;
      }
    }

    comm.topParentRanks = (int*)calloc(2, sizeof(int));
    for (int i = 0; i < 2; ++i) comm.topParentRanks[i] = i;

    // Allocate and initialize proxy state
    comm.proxyState = (struct ncclProxyState*)calloc(1, sizeof(struct ncclProxyState));
    comm.sharedRes = (struct ncclSharedResources*)calloc(1, sizeof(struct ncclSharedResources));
    comm.sharedRes->proxyState = comm.proxyState;
    comm.sharedRes->tpNLocalRanks = comm.nRanks;
    comm.sharedRes->tpRankToLocalRank = (int*)calloc(comm.nRanks, sizeof(int));
    for (int i = 0; i < comm.nRanks; ++i) comm.sharedRes->tpRankToLocalRank[i] = i;
    comm.sharedRes->magic = 0x1234;

    int n = comm.sharedRes->tpNLocalRanks;
    comm.proxyState->peerSocks = (struct ncclSocket*)calloc(n, sizeof(struct ncclSocket));
    comm.proxyState->proxyOps = (struct ncclProxyOps*)calloc(n, sizeof(struct ncclProxyOps));
    comm.proxyState->sharedDevMems = (void**)calloc(n, sizeof(void*));
    for (int i = 0; i < n; ++i) ncclSocketSetFd(-1, &comm.proxyState->peerSocks[i]);

    comm.abortFlag = (uint32_t*)calloc(1, sizeof(uint32_t));
    *comm.abortFlag = 0;

    struct ncclPeerInfo* myInfo = &comm.peerInfo[0];
    struct ncclPeerInfo* peerInfo = &comm.peerInfo[1];

    struct ncclConnect connectInfo;
    struct ncclConnector send;
    memset(&send, 0, sizeof(send));

    ncclResult_t result = p2pSendSetup(&comm, &graph, myInfo, peerInfo, &connectInfo, &send, 0, 0);

    // Receive result from child
    ncclResult_t childResult;
    read(pipefd[1], &childResult, sizeof(childResult));
    close(pipefd[1]);

    EXPECT_EQ(result, ncclInvalidArgument); // p2pSendSetup should return ncclInvalidArgument
    EXPECT_EQ(childResult, ncclSuccess);

    // Clean up
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        if (i == j) continue;
        free(system.nodes[GPU].nodes[i].paths[GPU][j].list[0]);
      }
      free(system.nodes[GPU].nodes[i].paths[GPU]);
    }
    free(comm.peerInfo);
    free(comm.topParentRanks);

    int status;
    waitpid(childPid, &status, 0);
  }
}

*/

/*
TEST_F(P2pTest, IpcRegisterBufferFailures) {

  struct ncclComm comm;
  struct ncclTopoSystem system;
  setupCommAndPeers(&comm, &system, gethostid(), 0, 0, 0, true);
  comm.nRanks = 2;
  comm.rankToLocalRank = (int*)calloc(comm.nRanks, sizeof(int));
  for (int i = 0; i < comm.nRanks; ++i) comm.rankToLocalRank[i] = i;

  void* dptr = nullptr;
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  hipError_t err = hipMalloc(&dptr, 32 * sizeof(float));
  ASSERT_EQ(err, hipSuccess);

  struct ncclReg regRecord;
  memset(&regRecord, 0, sizeof(regRecord));
  regRecord.addr = (uintptr_t)dptr;
  regRecord.pages = 1;
  for (int i = 0; i < 2; ++i) regRecord.ipcInfos[i] = nullptr;

  int peerRanks[1] = {1};
  int regBufFlag = 0;
  uintptr_t offsetOut = 0;
  uintptr_t* peerRmtAddrsOut = nullptr;
  bool isLegacyIpc = false;
  ncclIpcRegType type = NCCL_IPC_COLLECTIVE;

  // Test 1: HIP_POINTER_ATTRIBUTE_IS_LEGACY_HIP_IPC_CAPABLE is not supported
  ncclResult_t result = ipcRegisterBuffer(&comm, dptr, 32 * sizeof(float), peerRanks, 1, type, &regRecord, &regBufFlag, &offsetOut, &peerRmtAddrsOut, &isLegacyIpc);
  EXPECT_EQ(result, ncclUnhandledCudaError);

  hipFree(dptr);
  free(comm.peerInfo);
  free(comm.rankToLocalRank);
}

TEST_F(P2pTest, CleanupIpc_SingleProcess) {
  // Setup comm and reg structures
  ncclComm_t comm;
  struct ncclTopoSystem system;
  ncclUniqueId id;
  ncclCommInitAll(&comm, 1, nullptr);

  struct ncclReg* reg = nullptr;
  void* dptr = nullptr;
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  ASSERT_EQ(hipMalloc(&dptr, 32 * sizeof(float)), hipSuccess);

  size_t baseSize = 32 * sizeof(float);
  void* baseAddr = dptr;
  ncclResult_t regResult = ncclCommGraphRegister(comm, baseAddr, baseSize, (void**)&reg);
  ASSERT_EQ(regResult, ncclSuccess);
  ASSERT_NE(reg, nullptr);

  struct ncclIpcCleanupCallback* cb = (struct ncclIpcCleanupCallback*)calloc(1, sizeof(struct ncclIpcCleanupCallback));
  cb->comm = comm;
  cb->reg = reg;

  ncclResult_t cleanupResult = cleanupIpc(comm, (struct ncclCommCallback*)cb);
  EXPECT_EQ(cleanupResult, ncclSuccess);

  // Only free dptr, not reg or cb (already freed)
  hipFree(dptr);
}

*/
