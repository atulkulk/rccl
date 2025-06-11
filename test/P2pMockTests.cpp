#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "HipMockWrappers.hpp"

HipNcclMock* g_mock = nullptr;

// // --- Mock HIP and NCCL functions ---
// extern "C" {
//   hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
//   hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attr, hipDeviceptr_t ptr);
//   hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
//   // ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int peer, struct ncclProxyConnector* conn);
//   // ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* conn, int msgType, void* req, int reqSize, void* resp, int respSize);
// }

// C wrappers for the mock functions
extern "C" {

hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  printf("hipMemGetAddressRange called\n");
  return g_mock->hipMemGetAddressRange(pbase, psize, dptr);
}

hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attr, hipDeviceptr_t ptr) {
  printf("hipPointerGetAttribute called\n");
  return g_mock->hipPointerGetAttribute(data, attr, ptr);
}

hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
  printf("hipIpcGetMemHandle called\n");
  return g_mock->hipIpcGetMemHandle(handle, devPtr);
}

}

ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int peer, struct ncclProxyConnector* conn) {
  std::cout << "ncclProxyConnect called" << std::endl;
  return g_mock->ncclProxyConnect(comm, transport, send, peer, conn);
}

ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* conn, int msgType, void* req, int reqSize, void* resp, int respSize) {
  std::cout << "ncclProxyCallBlocking called" << std::endl;
  return g_mock->ncclProxyCallBlocking(comm, conn, msgType, req, reqSize, resp, respSize);
}

ncclResult_t ncclCudaMemcpyAsync(void* dst, const void* src, size_t count, hipMemcpyKind kind, hipStream_t stream) {
  std::cout << "ncclCudaMemcpyAsync called" << std::endl;
  return g_mock->ncclCudaMemcpyAsync(dst, src, count, kind, stream);
}

ncclResult_t ncclStrongStreamWaitStream(struct ncclStrongStream* ss, hipStream_t stream) {
  std::cout << "ncclStrongStreamWaitStream called" << std::endl;
  return g_mock->ncclStrongStreamWaitStream(ss, stream);
}

ncclResult_t ncclStrongStreamRelease(struct ncclStrongStream* ss) {
  std::cout << "ncclStrongStreamRelease called" << std::endl;
  return g_mock->ncclStrongStreamRelease(ss);
}

// Declare the global mock pointer
// HipNcclMock* g_mock = nullptr;

ncclResult_t ipcRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, struct ncclReg* regRecord, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, bool* isLegacyIpc);

// --- Test fixture using TEST_F ---
class IpcRegisterBufferTest : public ::testing::Test {
protected:
  void SetUp() override {
    g_mock = &mock;
  }
  void TearDown() override {
    g_mock = nullptr;
  }
  HipNcclMock mock;
};

TEST_F(IpcRegisterBufferTest, SuccessPath) {
  // Setup comm and regRecord
  ncclComm comm = {};
  comm.rankToLocalRank = (int*)calloc(2, sizeof(int));
  comm.rankToLocalRank[0] = 0;
  comm.rankToLocalRank[1] = 1;
  comm.nRanks = 2;
  comm.directMode = 0;
  comm.gproxyConn = (struct ncclProxyConnector*)calloc(2, sizeof(struct ncclProxyConnector));
  comm.gproxyConn[1].initialized = false;
  struct ncclStrongStream* strongStream;
  comm.sharedRes = (struct ncclSharedResources*)calloc(1, sizeof(struct ncclSharedResources));
  comm.sharedRes->hostStream = {};
  void* dptr = nullptr;
  hipError_t err = hipMalloc(&dptr, 128);
  ASSERT_EQ(err, hipSuccess);

  struct ncclReg regRecord = {};
  regRecord.addr = 0x1000;
  regRecord.pages = 1;
  // struct ncclIpcRegInfo* ipcInfo0 = new struct ncclIpcRegInfo();
  // struct ncclIpcRegInfo* ipcInfo1 = new struct ncclIpcRegInfo();
  // regRecord.ipcInfos[0] = ipcInfo0;
  // regRecord.ipcInfos[1] = ipcInfo1;

  regRecord.addr = (uintptr_t)dptr;
  regRecord.pages = 1;

  // Allocate and zero out the ipcInfos array (size: comm->nRanks)
  for (int i = 0; i < comm.nRanks; ++i)
      regRecord.ipcInfos[i] = nullptr;

  // Initialize regIpcAddrs.hostPeerRmtAddrs to hold comm.localRanks elements
  regRecord.regIpcAddrs.hostPeerRmtAddrs = (uintptr_t*)calloc(comm.localRanks, sizeof(uintptr_t));
  // Optionally, initialize devPeerRmtAddrs if needed by your code
  regRecord.regIpcAddrs.devPeerRmtAddrs = nullptr;

  // If your code expects regIpcAddrs to be zeroed, do so:
  memset(regRecord.regIpcAddrs.hostPeerRmtAddrs, 0, comm.localRanks * sizeof(uintptr_t));

  int peerRanks[1] = {1};
  int regBufFlag = 0;
  uintptr_t offsetOut = 0;
  uintptr_t* peerRmtAddrsOut = nullptr;
  bool isLegacyIpc = false;
  ncclIpcRegType type = NCCL_IPC_COLLECTIVE;

  // Set up mocks for a successful path
  EXPECT_CALL(mock, hipMemGetAddressRange(testing::_, testing::_, testing::_))
      .WillOnce([](hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
        *pbase = (hipDeviceptr_t)0x1000;
        *psize = 128;
        return hipSuccess;
      });
  EXPECT_CALL(mock, hipPointerGetAttribute(testing::_, testing::_, testing::_))
      .WillOnce([](void* data, hipPointer_attribute, hipDeviceptr_t) {
        *(bool*)data = true;
        return hipSuccess;
      });
  EXPECT_CALL(mock, ncclProxyConnect(testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillOnce(testing::Invoke([](struct ncclComm*, int, int, int, struct ncclProxyConnector* conn) {
        conn->initialized = true;
        conn->sameProcess = true;
        return ncclSuccess;
      }));
  EXPECT_CALL(mock, hipIpcGetMemHandle(testing::_, testing::_))
      .WillOnce(testing::Return(hipSuccess));
  EXPECT_CALL(mock, ncclProxyCallBlocking(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
      .WillOnce([](struct ncclComm*, struct ncclProxyConnector*, int, void*, int, void* resp, int) {
        // Simulate returning a remote address
        void* fakeRemoteAddr = (void*)0x2000;
        memcpy(resp, &fakeRemoteAddr, sizeof(void*));
        return ncclSuccess;
      });

  ncclResult_t result = ipcRegisterBuffer(&comm, (void*)0x1000, 128, peerRanks, 1, type, &regRecord, &regBufFlag, &offsetOut, &peerRmtAddrsOut, &isLegacyIpc);
  EXPECT_EQ(result, ncclSuccess);
  EXPECT_EQ(regBufFlag, 1);
  EXPECT_TRUE(isLegacyIpc);

  free(comm.rankToLocalRank);
  free(comm.gproxyConn);
}

// TEST_F(IpcRegisterBufferTest, HipMemGetAddressRangeFails) {
//   ncclComm comm = {};
//   comm.rankToLocalRank = (int*)calloc(2, sizeof(int));
//   comm.rankToLocalRank[0] = 0;
//   comm.rankToLocalRank[1] = 1;
//   comm.nRanks = 2;
//   comm.gproxyConn = (struct ncclProxyConnector*)calloc(2, sizeof(struct ncclProxyConnector));
//   comm.gproxyConn[1].initialized = false;
//   struct ncclReg regRecord = {};
//   regRecord.addr = 0x1000;
//   regRecord.pages = 1;
//   regRecord.ipcInfos[0] = nullptr;
//   regRecord.ipcInfos[1] = nullptr;
//   int peerRanks[1] = {1};
//   int regBufFlag = 0;
//   uintptr_t offsetOut = 0;
//   uintptr_t* peerRmtAddrsOut = nullptr;
//   bool isLegacyIpc = false;
//   ncclIpcRegType type = NCCL_IPC_COLLECTIVE;

//   // Simulate hipMemGetAddressRange failure
//   EXPECT_CALL(mock, hipMemGetAddressRange(testing::_, testing::_, testing::_))
//       .WillOnce(testing::Return(hipErrorInvalidValue));

//   ncclResult_t result = ipcRegisterBuffer(&comm, (void*)0x1000, 128, peerRanks, 1, type, &regRecord, &regBufFlag, &offsetOut, &peerRmtAddrsOut, &isLegacyIpc);
//   EXPECT_NE(result, ncclSuccess);

//   free(comm.rankToLocalRank);
//   free(comm.gproxyConn);
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleMock(&argc, argv);
  return RUN_ALL_TESTS();
}