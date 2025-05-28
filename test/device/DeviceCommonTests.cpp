#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include "device/common.h"
#include "device/sendrecv.h"

// // Kernel that uses barrier_sync
// __global__ void BarrierSyncTestKernel(int* data, int n) {
//   int tid = threadIdx.x;
//   if (tid < n) {
//     data[tid] = tid;
//   }
//   barrier_sync(0); // Synchronize all threads in the block
//   if (tid < n) {
//     data[tid] += 1;
//   }
// }

class DeviceCommonTests : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(hipMalloc(&d_data, numThreads * sizeof(int)), hipSuccess);
  }
  void TearDown() override {
    hipFree(d_data);
  }
  static constexpr int numThreads = 32;
  int* d_data = nullptr;
};

// TEST_F(DeviceCommonTests, AllThreadsSynchronized) {
//   int h_data[numThreads] = {0};

//   hipLaunchKernelGGL(BarrierSyncTestKernel, dim3(1), dim3(numThreads), 0, 0, d_data, numThreads);
//   ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

//   ASSERT_EQ(hipMemcpy(h_data, d_data, numThreads * sizeof(int), hipMemcpyDeviceToHost), hipSuccess);

//   for (int i = 0; i < numThreads; ++i) {
//     EXPECT_EQ(h_data[i], i + 1);
//   }
// }

// A simple kernel wrapper to call ncclKernelMain
__global__ void NcclKernelMainTestKernel(const ncclDevKernelArgs* args) {
    // Use dummy template parameters for demonstration
    ncclKernelMain<-1, RunWorkBatch<ncclFuncSendRecv, int8_t, FuncSum<int8_t>, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 1>, false, 1>(args);
}

TEST_F(DeviceCommonTests, NcclKernelMainRuns) {
    // Allocate and initialize dummy args
    ncclDevKernelArgs* d_args;
    hipMalloc(&d_args, sizeof(ncclDevKernelArgs));
    hipMemset(d_args, 0, sizeof(ncclDevKernelArgs));

    // Launch the kernel
    NcclKernelMainTestKernel<<<1, 64>>>(d_args);
    hipDeviceSynchronize();

    // Clean up
    hipFree(d_args);

    // If we reach here without a crash, the test passes
    SUCCEED();
}