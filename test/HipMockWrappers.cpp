#include "HipMockWrappers.hpp"

// HipNcclMock* g_mock = nullptr;

// // // --- Mock HIP and NCCL functions ---
// // extern "C" {
// //   hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr);
// //   hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attr, hipDeviceptr_t ptr);
// //   hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);
// //   // ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int peer, struct ncclProxyConnector* conn);
// //   // ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* conn, int msgType, void* req, int reqSize, void* resp, int respSize);
// // }

// // C wrappers for the mock functions
// extern "C" {

// hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
//   return g_mock->hipMemGetAddressRange(pbase, psize, dptr);
// }

// hipError_t hipPointerGetAttribute(void* data, hipPointer_attribute attr, hipDeviceptr_t ptr) {
//   printf("hipPointerGetAttribute called\n");
//   return g_mock->hipPointerGetAttribute(data, attr, ptr);
// }

// hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr) {
//   return g_mock->hipIpcGetMemHandle(handle, devPtr);
// }

// }

// ncclResult_t ncclProxyConnect(struct ncclComm* comm, int transport, int send, int peer, struct ncclProxyConnector* conn) {
//   return g_mock->ncclProxyConnect(comm, transport, send, peer, conn);
// }

// ncclResult_t ncclProxyCallBlocking(struct ncclComm* comm, struct ncclProxyConnector* conn, int msgType, void* req, int reqSize, void* resp, int respSize) {
//   return g_mock->ncclProxyCallBlocking(comm, conn, msgType, req, reqSize, resp, respSize);
// }

// ncclResult_t ncclCudaMemcpyAsync(void* dst, const void* src, size_t count, hipMemcpyKind kind, hipStream_t stream) {
//   return g_mock->ncclCudaMemcpyAsync(dst, src, count, kind, stream);
// }

// ncclResult_t ncclStrongStreamWaitStream(struct ncclStrongStream* ss, hipStream_t stream) {
//   return g_mock->ncclStrongStreamWaitStream(ss, stream);
// }

// ncclResult_t ncclStrongStreamRelease(struct ncclStrongStream* ss) {
//   return g_mock->ncclStrongStreamRelease(ss);
// }