/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <gtest/gtest.h>

#include "argcheck.h"
#include "comm.h"
#include <hip/hip_runtime.h>

class ArgCheckTest : public ::testing::Test {
protected:
    ncclComm comm;

    void SetUp() override {
        comm.cudaDev = 0;
        comm.startMagic = NCCL_MAGIC;
        comm.endMagic = NCCL_MAGIC;
        comm.nRanks = 4;
        comm.checkPointers = true;
    }
};

TEST_F(ArgCheckTest, CudaPtrCheck_InvalidScenarios) {
    int* devicePtr = nullptr;

    // Case 1: Valid pointer
    hipError_t err = hipMalloc(&devicePtr, sizeof(int));
    ASSERT_EQ(err, hipSuccess); // Ensure allocation was successful
    ncclResult_t result = CudaPtrCheck(devicePtr, &comm, "devicePtr", "TestOp");
    ASSERT_EQ(result, ncclSuccess) << "Failed for valid pointer";
    hipFree(devicePtr);

    // Case 2: Invalid pointer (nullptr)
    devicePtr = nullptr;
    result = CudaPtrCheck(devicePtr, &comm, "invalidPtr", "TestOp");
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for invalid pointer";

    // Case 3: Pointer allocated on a different device
    hipSetDevice(1); // Switch to a different device
    err = hipMalloc(&devicePtr, sizeof(int));
    ASSERT_EQ(err, hipSuccess); // Ensure allocation was successful
    result = CudaPtrCheck(devicePtr, &comm, "devicePtr", "TestOp");
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for pointer on a different device";
    hipFree(devicePtr);

    // Reset device to the original one
    hipSetDevice(comm.cudaDev);
}

TEST_F(ArgCheckTest, PtrCheck_ValidPointer) {
    int value = 42;
    ncclResult_t result = PtrCheck(&value, "TestOp", "value");
    ASSERT_EQ(result, ncclSuccess);
}

TEST_F(ArgCheckTest, PtrCheck_NullPointer) {
    ncclResult_t result = PtrCheck(nullptr, "TestOp", "value");
    ASSERT_EQ(result, ncclInvalidArgument);
}

TEST_F(ArgCheckTest, CommCheck_NullComm) {
    ncclResult_t result = CommCheck(nullptr, "TestOp", "comm");
    ASSERT_EQ(result, ncclInvalidArgument);
}

TEST_F(ArgCheckTest, CommCheck_CorruptedComm) {
    // Corrupt both startMagic and endMagic
    comm.startMagic = 1; // Corrupt startMagic
    comm.endMagic = 1;   // Corrupt endMagic

    // Call CommCheck and verify the result
    ncclResult_t result = CommCheck(&comm, "TestOp", "comm");
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for corrupted comm object";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidInputs) {
    ncclInfo info;

    // Initialize the communicator
    comm.startMagic = NCCL_MAGIC;
    comm.endMagic = NCCL_MAGIC;
    comm.nRanks = 4;           // Set a valid number of ranks
    comm.cudaDev = 0;          // Assume device 0
    comm.checkPointers = true; // Enable pointer checks
    info.comm = &comm;         // Valid communicator

    // Invalid root
    info.root = -1; // Invalid root
    info.datatype = (ncclDataType_t)0;
    info.op = (ncclRedOp_t)0;
    info.coll = ncclFuncBroadcast;
    ncclResult_t result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for invalid root";

    // Invalid datatype
    info.root = 0; // Valid root
    info.datatype = (ncclDataType_t)-1; // Invalid datatype
    result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for invalid datatype";

    // Invalid reduction operation
    info.datatype = (ncclDataType_t)0; // Valid datatype
    info.op = (ncclRedOp_t)-1;         // Invalid reduction operation
    result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for invalid reduction operation";

    // Invalid communicator pointers
    info.op = (ncclRedOp_t)0; // Valid reduction operation
    info.sendbuff = nullptr;  // Invalid send buffer
    info.recvbuff = nullptr;  // Invalid receive buffer
    result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for invalid communicator pointers";
}

TEST_F(ArgCheckTest, ArgsCheck_InvalidReductionOperationAndUserRedOp) {
    int* sendDevicePtr;
    int* recvDevicePtr;

    // Set the active device to match comm->cudaDev
    hipError_t errSetDevice = hipSetDevice(comm.cudaDev);
    ASSERT_EQ(errSetDevice, hipSuccess); // Ensure the device was set successfully

    // Allocate device memory for send and receive buffers
    hipError_t errSend = hipMalloc(&sendDevicePtr, sizeof(int));
    ASSERT_EQ(errSend, hipSuccess); // Ensure allocation was successful
    hipError_t errRecv = hipMalloc(&recvDevicePtr, sizeof(int));
    ASSERT_EQ(errRecv, hipSuccess); // Ensure allocation was successful

    // Initialize the communicator
    comm.startMagic = NCCL_MAGIC;
    comm.endMagic = NCCL_MAGIC;
    comm.nRanks = 4;           // Set a valid number of ranks
    comm.cudaDev = 0;          // Assume device 0
    comm.checkPointers = true; // Enable pointer checks

    ncclInfo info;

    // Initialize the ncclInfo structure with common valid values
    info.root = 0;                     // Valid root
    info.datatype = (ncclDataType_t)0; // Valid datatype
    info.coll = ncclFuncBroadcast;     // Valid collective operation
    info.comm = &comm;                 // Valid communicator
    info.sendbuff = sendDevicePtr;     // Use allocated device pointer for send buffer
    info.recvbuff = recvDevicePtr;     // Use allocated device pointer for receive buffer
    info.count = 10;                   // Valid count
    info.opName = "TestOp";            // Valid operation name

    // Test case 1: Invalid reduction operation (out of range)
    info.op = (ncclRedOp_t)5; // Invalid reduction operation (out of range)
    ncclResult_t result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for invalid reduction operation";

    // Test case 2: User-defined reduction operation with freeNext != -1
    info.op = (ncclRedOp_t)(ncclNumOps + 1); // Set op to a user-defined reduction operation
    result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclInvalidArgument) << "Failed for user-defined reduction operation with freeNext != -1";

    // Free the allocated device memory
    hipFree(sendDevicePtr);
    hipFree(recvDevicePtr);
}

TEST_F(ArgCheckTest, ArgsCheck_SendAndRecvFunction) {
    int* recvDevicePtr;

    // Set the active device to match comm->cudaDev
    hipError_t errSetDevice = hipSetDevice(comm.cudaDev);
    ASSERT_EQ(errSetDevice, hipSuccess); // Ensure the device was set successfully

    // Allocate device memory for the receive buffer
    hipError_t errRecv = hipMalloc(&recvDevicePtr, sizeof(int));
    ASSERT_EQ(errRecv, hipSuccess); // Ensure allocation was successful

    ncclInfo info;

    // Initialize the communicator
    comm.startMagic = NCCL_MAGIC;
    comm.endMagic = NCCL_MAGIC;
    comm.nRanks = 4;           // Set a valid number of ranks
    comm.cudaDev = 0;          // Assume device 0
    comm.checkPointers = true; // Enable pointer checks

    // Initialize the ncclInfo structure with common valid values
    info.root = 0;                     // Valid root
    info.datatype = (ncclDataType_t)0; // Valid datatype
    info.op = (ncclRedOp_t)0;          // Valid reduction operation
    info.comm = &comm;                 // Valid communicator
    info.sendbuff = nullptr;           // Send buffer is not used for ncclFuncSend or ncclFuncRecv
    info.recvbuff = recvDevicePtr;     // Use allocated device pointer for receive buffer
    info.count = 10;                   // Valid count
    info.opName = "TestOp";            // Valid operation name

    // Test both ncclFuncSend and ncclFuncRecv
    for (auto coll : {ncclFuncSend, ncclFuncRecv}) {
        info.coll = coll; // Set the collective operation

        // Call ArgsCheck and verify the result
        ncclResult_t result = ArgsCheck(&info);
        ASSERT_EQ(result, ncclSuccess) << "Failed for coll = " << coll;
    }

    // Free the allocated device memory
    hipFree(recvDevicePtr);
}

TEST_F(ArgCheckTest, ArgsCheck_CollNotReduceOrRankIsRoot) {
    int* sendDevicePtr;
    int* recvDevicePtr;

    // Set the active device to match comm->cudaDev
    hipError_t errSetDevice = hipSetDevice(comm.cudaDev);
    ASSERT_EQ(errSetDevice, hipSuccess); // Ensure the device was set successfully

    // Allocate device memory for send and receive buffers
    hipError_t errSend = hipMalloc(&sendDevicePtr, sizeof(int));
    ASSERT_EQ(errSend, hipSuccess); // Ensure allocation was successful
    hipError_t errRecv = hipMalloc(&recvDevicePtr, sizeof(int));
    ASSERT_EQ(errRecv, hipSuccess); // Ensure allocation was successful

    // Initialize the communicator
    comm.startMagic = NCCL_MAGIC;
    comm.endMagic = NCCL_MAGIC;
    comm.nRanks = 4;           // Set a valid number of ranks
    comm.cudaDev = 0;          // Assume device 0
    comm.checkPointers = true; // Enable pointer checks
    comm.rank = 0;             // Set rank to 0

    ncclInfo info;

    // Initialize the ncclInfo structure with common valid values
    info.root = 0;                     // Valid root
    info.datatype = (ncclDataType_t)0; // Valid datatype
    info.op = (ncclRedOp_t)0;          // Valid reduction operation
    info.comm = &comm;                 // Valid communicator
    info.sendbuff = sendDevicePtr;     // Use allocated device pointer for send buffer
    info.recvbuff = recvDevicePtr;     // Use allocated device pointer for receive buffer
    info.count = 10;                   // Valid count
    info.opName = "TestOp";            // Valid operation name

    // Case 1: info->coll != ncclFuncReduce
    info.coll = ncclFuncBroadcast; // Set coll to ncclFuncBroadcast
    ncclResult_t result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclSuccess) << "Failed for coll != ncclFuncReduce";

    // Case 2: info->coll == ncclFuncReduce and info->comm->rank == info->root
    info.coll = ncclFuncReduce; // Set coll to ncclFuncReduce
    result = ArgsCheck(&info);
    ASSERT_EQ(result, ncclSuccess) << "Failed for coll == ncclFuncReduce and rank == root";

    // Free the allocated device memory
    hipFree(sendDevicePtr);
    hipFree(recvDevicePtr);
}
