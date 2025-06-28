#pragma once
#include "gmock/gmock.h"
#include "hip/hip_runtime.h"

#ifndef warpSize
#define warpSize 64
// #define WARP_SIZE 64
#endif

#include "comm.h"

class HipNcclMock {
public:
  MOCK_METHOD(hipError_t, hipMemGetAddressRange, (hipDeviceptr_t*, size_t*, hipDeviceptr_t), ());
  MOCK_METHOD(hipError_t, hipPointerGetAttribute, (void*, hipPointer_attribute, hipDeviceptr_t), ());
  MOCK_METHOD(hipError_t, hipIpcGetMemHandle, (hipIpcMemHandle_t*, void*), ());
  MOCK_METHOD(ncclResult_t, ncclProxyConnect, (struct ncclComm*, int, int, int, struct ncclProxyConnector*), ());
  MOCK_METHOD(ncclResult_t, ncclProxyCallBlocking, (struct ncclComm*, struct ncclProxyConnector*, int, void*, int, void*, int), ());
  MOCK_METHOD(ncclResult_t, ncclCudaMemcpyAsync, (void*, const void*, size_t, hipMemcpyKind, hipStream_t), ());
  MOCK_METHOD(ncclResult_t, ncclStrongStreamWaitStream, (struct ncclCudaGraph, struct ncclStrongStream*, struct ncclStrongStream*, bool), ());
  MOCK_METHOD(ncclResult_t, ncclStrongStreamWaitStream, (struct ncclCudaGraph, struct ncclStrongStream*, hipStream_t, bool), ());
  MOCK_METHOD(ncclResult_t, ncclStrongStreamWaitStream, (struct ncclCudaGraph, hipStream_t, struct ncclStrongStream*, bool), ());
  MOCK_METHOD(ncclResult_t, ncclStrongStreamRelease, (struct ncclCudaGraph, struct ncclStrongStream*), ());
};


// Forward declaration of the global mock pointer
extern HipNcclMock* g_mock;
