// tests/send_test.cpp
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include "../device/primitives.h"  // Using primitives.h instead of prims_simple.h
#include "../device/common.h"
#include "../include/alloc.h"
#include <iostream>
#include <vector>
#include <cstring>

// Define constants for testing
#define TEST_DATA_SIZE 1024
#define TEST_ELEM_COUNT 256
#define MAX_PEERS 16
#define NCCL_STEPS 64
#define CACHE_LINE_SIZE 128
#define NCCL_MAX_CONNS 2
#define NCCL_NUM_PROTOCOLS 3  // Simple/LL/LL128
#define NCCL_PROTO_SIMPLE 0  // Index for simple protocol

// Forward declarations for structures we need to mock
struct ncclShmemData ncclShmem;

// Test fixture for send method
class PrimitivesSendTest : public ::testing::Test {
protected:
    // Host buffers
    float *h_inputBuf;
    float *h_outputBuf;
    int *h_recvPeers;
    int *h_sendPeers;
    ncclDevChannelPeer **h_peers;         // Array of pointers to peers
    ncclShmemData *h_shmem;
    uint64_t *h_sendHead[MAX_PEERS][NCCL_MAX_CONNS];
    uint64_t *h_recvTail[MAX_PEERS][NCCL_MAX_CONNS];

    // Device buffers
    float *d_inputBuf;
    float *d_outputBuf;
    int *d_recvPeers;
    int *d_sendPeers;
    ncclDevChannelPeer **d_peers;         // Array of pointers to peers
    ncclDevChannelPeer **d_peerPtrs;
    void *d_ncclShmem;
    uint64_t *d_sendHead[MAX_PEERS][NCCL_MAX_CONNS];
    uint64_t *d_recvTail[MAX_PEERS][NCCL_MAX_CONNS];

    // Connection buffers - now with 3D structure [MAX_PEERS][NCCL_MAX_CONNS][NCCL_NUM_PROTOCOLS]
    void *h_sendBuff[MAX_PEERS][NCCL_MAX_CONNS][NCCL_NUM_PROTOCOLS];
    void *h_recvBuff[MAX_PEERS][NCCL_MAX_CONNS][NCCL_NUM_PROTOCOLS];
    void *d_sendBuff[MAX_PEERS][NCCL_MAX_CONNS][NCCL_NUM_PROTOCOLS];
    void *d_recvBuff[MAX_PEERS][NCCL_MAX_CONNS][NCCL_NUM_PROTOCOLS];
    ncclConnFifo *h_connFifo[MAX_PEERS][NCCL_MAX_CONNS];
    ncclConnFifo *d_connFifo[MAX_PEERS][NCCL_MAX_CONNS];

    // Test parameters
    int nranks = 2;
    int buffSize;
    int stepSize;
    uint8_t connIndexSend = 0; // Connection index for send operations
    uint8_t connIndexRecv = 0; // Connection index for receive operations
    int testPeer = 1;          // The peer we'll use for testing (index in peers array)

    void SetUp() override {
        // Calculate buffer sizes
        buffSize = TEST_DATA_SIZE * sizeof(float);
        stepSize = buffSize / NCCL_STEPS;

        // Allocate host memory
        h_inputBuf = new float[TEST_DATA_SIZE];
        h_outputBuf = new float[TEST_DATA_SIZE];
        h_recvPeers = new int[MAX_PEERS];
        h_sendPeers = new int[MAX_PEERS];
        h_shmem = (ncclShmemData*)malloc(sizeof(ncclShmemData));

        // Initialize data
        for (int i = 0; i < TEST_DATA_SIZE; i++) {
            h_inputBuf[i] = static_cast<float>(i);
            h_outputBuf[i] = 0.0f;
        }

        // Initialize peer arrays
        memset(h_recvPeers, -1, MAX_PEERS * sizeof(int));
        h_sendPeers[0] = testPeer; // Send to test peer
        for (int i = 1; i < MAX_PEERS; i++) {
            h_sendPeers[i] = -1; // Terminate the list
        }

        // Allocate arrays for peers
        h_peers = new ncclDevChannelPeer*[MAX_PEERS];

        // Allocate device memory
        hipMalloc(&d_inputBuf, buffSize);
        hipMalloc(&d_outputBuf, buffSize);
        hipMalloc(&d_recvPeers, MAX_PEERS * sizeof(int));
        hipMalloc(&d_sendPeers, MAX_PEERS * sizeof(int));
        hipMalloc(&d_ncclShmem, sizeof(ncclShmemData));

        // Allocate device arrays for peers
        hipMalloc(&d_peers, MAX_PEERS * sizeof(ncclDevChannelPeer*));
        hipMalloc(&d_peerPtrs, sizeof(ncclDevChannelPeer**));

        // Initialize all peers
        for (int peer = 0; peer < MAX_PEERS; peer++) {
            // Allocate peer structure
            h_peers[peer] = (ncclDevChannelPeer*)malloc(sizeof(ncclDevChannelPeer));
            memset(h_peers[peer], 0, sizeof(ncclDevChannelPeer));

            // Allocate device memory for peer
            hipMalloc(&d_peers[peer], sizeof(ncclDevChannelPeer));

            // Setup for each connection index
            for (int i = 0; i < NCCL_MAX_CONNS; i++) {
                // Allocate head/tail pointers
                h_sendHead[peer][i] = (uint64_t*)malloc(sizeof(uint64_t));
                h_recvTail[peer][i] = (uint64_t*)malloc(sizeof(uint64_t));

                // Initialize values
                *h_sendHead[peer][i] = 0;
                *h_recvTail[peer][i] = 0;

                // Allocate device memory for head/tail pointers
                hipMalloc(&d_sendHead[peer][i], sizeof(uint64_t));
                hipMalloc(&d_recvTail[peer][i], sizeof(uint64_t));

                // Allocate connection FIFO
                h_connFifo[peer][i] = (ncclConnFifo*)malloc(NCCL_STEPS * sizeof(ncclConnFifo));
                memset(h_connFifo[peer][i], 0, NCCL_STEPS * sizeof(ncclConnFifo));
                hipMalloc(&d_connFifo[peer][i], NCCL_STEPS * sizeof(ncclConnFifo));

                // Allocate multiple buffers for each connection
                for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
                    // Allocate host and device memory for each protocol buffer
                    h_sendBuff[peer][i][p] = malloc(buffSize);
                    h_recvBuff[peer][i][p] = malloc(buffSize);
                    memset(h_sendBuff[peer][i][p], 0, buffSize);
                    memset(h_recvBuff[peer][i][p], 0, buffSize);

                    hipMalloc(&d_sendBuff[peer][i][p], buffSize);
                    hipMalloc(&d_recvBuff[peer][i][p], buffSize);

                    // Initialize protocol buffers with unique patterns
                    float* pattern = new float[TEST_DATA_SIZE];
                    for (int j = 0; j < TEST_DATA_SIZE; j++) {
                        // pattern[j] = static_cast<float>((peer * 10000) + (i * 1000) + (p * 100) + j);
                        pattern[j] = static_cast<float>(0);
                    }
                    hipMemcpy(d_sendBuff[peer][i][p], pattern, buffSize, hipMemcpyHostToDevice);
                    hipMemcpy(d_recvBuff[peer][i][p], pattern, buffSize, hipMemcpyHostToDevice);
                    delete[] pattern;
                }

                // Copy data to device
                hipMemcpy(d_sendHead[peer][i], h_sendHead[peer][i], sizeof(uint64_t), hipMemcpyHostToDevice);
                hipMemcpy(d_recvTail[peer][i], h_recvTail[peer][i], sizeof(uint64_t), hipMemcpyHostToDevice);
                hipMemcpy(d_connFifo[peer][i], h_connFifo[peer][i], NCCL_STEPS * sizeof(ncclConnFifo), hipMemcpyHostToDevice);

                // Setup connection info for each index and protocol - with proper casting
                for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
                    // Cast void* to char* to match the expected type in ncclConnInfo.buffs
                    h_peers[peer]->send[i].buffs[p] = static_cast<char*>(d_sendBuff[peer][i][p]);
                    h_peers[peer]->recv[i].buffs[p] = static_cast<char*>(d_recvBuff[peer][i][p]);
                }

                h_peers[peer]->send[i].tail = d_recvTail[peer][i];  // Connect send to recv tail
                h_peers[peer]->send[i].head = d_sendHead[peer][i];  // Head of send buffer
                h_peers[peer]->recv[i].head = d_recvTail[peer][i];  // Head of recv buffer
                h_peers[peer]->recv[i].connFifo = d_connFifo[peer][i];

                // These properties may not exist in ncclConnInfo, set only if they do
                h_peers[peer]->send[i].step = 0;
                h_peers[peer]->recv[i].step = 0;
                h_peers[peer]->send[i].flags = 0; // No special flags
                h_peers[peer]->recv[i].flags = 0; // No special flags
                h_peers[peer]->send[i].stepSize = stepSize;
                h_peers[peer]->recv[i].stepSize = stepSize;
            }

            // Copy peer structure to device
            hipMemcpy(d_peers[peer], h_peers[peer], sizeof(ncclDevChannelPeer), hipMemcpyHostToDevice);
        }

        // Copy regular data to device
        hipMemcpy(d_inputBuf, h_inputBuf, buffSize, hipMemcpyHostToDevice);
        hipMemcpy(d_outputBuf, h_outputBuf, buffSize, hipMemcpyHostToDevice);
        hipMemcpy(d_recvPeers, h_recvPeers, MAX_PEERS * sizeof(int), hipMemcpyHostToDevice);
        hipMemcpy(d_sendPeers, h_sendPeers, MAX_PEERS * sizeof(int), hipMemcpyHostToDevice);

        // Copy peers array to device
        hipMemcpy(d_peerPtrs, &d_peers, sizeof(ncclDevChannelPeer**), hipMemcpyHostToDevice);

        // Set up shared memory
        memset(h_shmem, 0, sizeof(ncclShmemData));
        h_shmem->channel.peers = (ncclDevChannelPeer**)d_peers;
        // Channel ID is not a member of ncclDevChannel, don't set it
        h_shmem->comm.buffSizes[NCCL_PROTO_SIMPLE] = buffSize;
        h_shmem->comm.nRanks = nranks;
        h_shmem->comm.rank = 0;
        h_shmem->aborted = 0;

        // Setup storage for send/recv connections in shared memory groups
        for (int peer = 0; peer < MAX_PEERS; peer++) {
            for (int i = 0; i < NCCL_MAX_CONNS; i++) {
                int idx = peer * NCCL_MAX_CONNS + i;
                if (idx < MAX_PEERS * NCCL_MAX_CONNS) {  // Ensure we don't overflow
                    h_shmem->groups[0].sendConns[idx] = &h_peers[peer]->send[i];
                    h_shmem->groups[0].recvConns[idx] = &h_peers[peer]->recv[i];
                }
            }
        }

        // Copy shared memory structure to device
        hipMemcpy(d_ncclShmem, h_shmem, sizeof(ncclShmemData), hipMemcpyHostToDevice);
    }

    void TearDown() override {
        // Free host memory
        delete[] h_inputBuf;
        delete[] h_outputBuf;
        delete[] h_recvPeers;
        delete[] h_sendPeers;
        free(h_shmem);

        // Free peer-specific memory
        for (int peer = 0; peer < MAX_PEERS; peer++) {
            // Free connection-specific memory
            for (int i = 0; i < NCCL_MAX_CONNS; i++) {
                // Free protocol buffers
                for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
                    free(h_sendBuff[peer][i][p]);
                    free(h_recvBuff[peer][i][p]);
                    hipFree(d_sendBuff[peer][i][p]);
                    hipFree(d_recvBuff[peer][i][p]);
                }

                free(h_connFifo[peer][i]);
                free(h_sendHead[peer][i]);
                free(h_recvTail[peer][i]);

                hipFree(d_connFifo[peer][i]);
                hipFree(d_sendHead[peer][i]);
                hipFree(d_recvTail[peer][i]);
            }

            free(h_peers[peer]);
            hipFree(d_peers[peer]);
        }

        delete[] h_peers;

        // Free remaining device memory
        hipFree(d_inputBuf);
        hipFree(d_outputBuf);
        hipFree(d_recvPeers);
        hipFree(d_sendPeers);
        hipFree(d_peers);
        hipFree(d_peerPtrs);
        hipFree(d_ncclShmem);
    }
};

// Kernel to test Primitives::send method with specific connection index
template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int MultimemSrcs, int MultimemDsts,
         int P2p, bool isNetOffload>
__global__ void testSendKernel(T* inputBuf, int* recvPeers, int* sendPeers,
                            void* ncclShmemPtr, intptr_t inpIx, int eltN,
                            bool* success, T* outputBuf,
                            uint8_t connIndexSend, uint8_t connIndexRecv) {
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Point global ncclShmem to our allocated memory
    ncclShmem = *((ncclShmemData*)ncclShmemPtr);

    // Initialize the primitives object with specific connection indices
    Primitives<T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>, P2p, isNetOffload>
        prims(tid, nthreads, recvPeers, sendPeers, inputBuf, outputBuf, 0, 0,
              connIndexRecv, connIndexSend);  // Using the provided connection indices

    // Call the send method
    prims.send(inpIx, eltN);
    // prims.recv(inpIx, eltN);

    // For simplicity, indicate success if we reach this point
    if (tid == 0) *success = true;

    // We do a barrier to make sure all threads finish their work
    __syncthreads();
}

// TEST_F(PrimitivesSendTest, SendFromPeer0ToPeer1) {
//     // Set up: peer 0 sends to peer 1
//     int senderPeer = 0;
//     int receiverPeer = 1;
//     connIndexSend = 0;
//     connIndexRecv = 0;

//     // Set sendPeers and recvPeers accordingly
//     int* h_sendPeersTest = new int[MAX_PEERS];
//     int* h_recvPeersTest = new int[MAX_PEERS];
//     for (int i = 0; i < MAX_PEERS; i++) {
//         h_sendPeersTest[i] = -1;
//         h_recvPeersTest[i] = -1;
//     }
//     h_sendPeersTest[0] = receiverPeer; // peer 0 sends to peer 1
//     h_recvPeersTest[0] = senderPeer;   // peer 1 receives from peer 0

//     hipMemcpy(d_sendPeers, h_sendPeersTest, MAX_PEERS * sizeof(int), hipMemcpyHostToDevice);
//     hipMemcpy(d_recvPeers, h_recvPeersTest, MAX_PEERS * sizeof(int), hipMemcpyHostToDevice);

//     // Prepare input buffer for peer 0
//     for (int i = 0; i < TEST_DATA_SIZE; i++) {
//         h_inputBuf[i] = static_cast<float>(i + 1000); // unique pattern
//     }
//     hipMemcpy(d_inputBuf, h_inputBuf, buffSize, hipMemcpyHostToDevice);

//     // Allocate memory for success flag on device
//     bool* d_success;
//     hipMalloc(&d_success, sizeof(bool));
//     bool success = false;
//     hipMemcpy(d_success, &success, sizeof(bool), hipMemcpyHostToDevice);

//     intptr_t inpIx = 0;
//     int eltN = TEST_ELEM_COUNT;

//     using TestRedOp = FuncSum<float>;
//     using TestFan = FanSymmetric<1>;
//     constexpr int slicePerChunk = 1;
//     constexpr int stepPerSlice = 1;
//     constexpr int unroll = 1;

//     dim3 blockSize(128);
//     dim3 gridSize(1);

//     // // // Kernel that does both send and receive
//     // // auto sendRecvKernel = [] __global__ (
//     // //     float* inputBuf, float* outputBuf, int* recvPeers, int* sendPeers,
//     // //     void* ncclShmemPtr, intptr_t inpIx, int eltN, bool* success,
//     // //     uint8_t connIndexSend, uint8_t connIndexRecv
//     // // ) {
//     // //     int tid = threadIdx.x;
//     // //     int nthreads = blockDim.x;
//     // //     ncclShmem = *((ncclShmemData*)ncclShmemPtr);

//     // //     Primitives<float, FuncSum<float>, FanSymmetric<1>, 0,
//     // //                ProtoSimple<slicePerChunk, stepPerSlice, unroll, 0, 0>, 0, false>
//     // //         prims(tid, nthreads, recvPeers, sendPeers, inputBuf, outputBuf, 0, 0,
//     // //               connIndexRecv, connIndexSend);

//     // //     prims.send(inpIx, eltN);
//     // //     prims.recv(inpIx, eltN);

//     // //     if (tid == 0) *success = true;
//     // //     __syncthreads();
//     // // };


//     // Launch the kernel as peer 0 (sender)
//     hipLaunchKernelGGL(
//         (testSendKernel<float, TestRedOp, TestFan, 0, slicePerChunk, stepPerSlice, unroll, 0, 0, 0, false>),
//         gridSize, blockSize, 0, 0,
//         d_inputBuf, d_recvPeers, d_sendPeers, d_ncclShmem, inpIx, eltN, d_success, d_outputBuf,
//         connIndexSend, connIndexRecv
//     );

//     // hipError_t err = hipGetLastError();
//     // EXPECT_EQ(err, hipSuccess) << "Kernel launch failed: " << hipGetErrorString(err);

//     // hipMemcpy(&success, d_success, sizeof(bool), hipMemcpyDeviceToHost);
//     // EXPECT_TRUE(success) << "Send operation failed";

//     // // Verify peer 1's receive buffer (NCCL_PROTO_SIMPLE = 0)
//     // float* h_recvBuffResult = new float[TEST_ELEM_COUNT];
//     // hipMemcpy(h_recvBuffResult, d_recvBuff[receiverPeer][connIndexRecv][NCCL_PROTO_SIMPLE], TEST_ELEM_COUNT * sizeof(float), hipMemcpyDeviceToHost);

//     // // The receive buffer should now contain the data sent from peer 0
//     // for (int i = 0; i < TEST_ELEM_COUNT; i++) {
//     //     EXPECT_EQ(h_recvBuffResult[i], static_cast<float>(i + 1000)) << "Mismatch at element " << i;
//     // }

//     // // Check if step counter was updated for sender
//     // uint64_t sendHead;
//     // hipMemcpy(&sendHead, d_sendHead[senderPeer][connIndexSend], sizeof(uint64_t), hipMemcpyDeviceToHost);
//     // EXPECT_GT(sendHead, 0) << "Sender's step counter was not updated";

//     // // Check if step counter was updated for receiver
//     // uint64_t recvTail;
//     // hipMemcpy(&recvTail, d_recvTail[receiverPeer][connIndexRecv], sizeof(uint64_t), hipMemcpyDeviceToHost);
//     // EXPECT_GT(recvTail, 0) << "Receiver's step counter was not updated";

//     // delete[] h_recvBuffResult;
//     // delete[] h_sendPeersTest;
//     // delete[] h_recvPeersTest;
//     // hipFree(d_success);
// }

// Test that Primitives::send correctly handles basic send operations
TEST_F(PrimitivesSendTest, BasicSendOperation) {
    // Allocate memory for success flag on device
    bool* d_success;
    hipMalloc(&d_success, sizeof(bool));
    bool success = false;
    hipMemcpy(d_success, &success, sizeof(bool), hipMemcpyHostToDevice);

    // Define parameters for the send operation
    intptr_t inpIx = 0;  // Start from beginning of input buffer
    int eltN = TEST_ELEM_COUNT;  // Number of elements to send

    // Launch test kernel with 128 threads
    using TestRedOp = FuncSum<float>;
    using TestFan = FanSymmetric<1>; // Simple fan-out
    constexpr int slicePerChunk = 1;
    constexpr int stepPerSlice = 1;
    constexpr int unroll = 1;

    dim3 blockSize(128);
    dim3 gridSize(1);

    // hipLaunchKernelGGL(
    //     (testSendKernel<float, TestRedOp, TestFan, 0, slicePerChunk, stepPerSlice, unroll, 0, 0, 0, false>),
    //     gridSize, blockSize, 0, 0,
    //     d_inputBuf, d_recvPeers, d_sendPeers, d_ncclShmem, inpIx, eltN, d_success, d_outputBuf,
    //     connIndexSend, connIndexRecv
    // );

    // Check for kernel launch errors
    hipError_t err = hipGetLastError();
    EXPECT_EQ(err, hipSuccess) << "Kernel launch failed: " << hipGetErrorString(err);

    // Copy result back to host
    hipMemcpy(&success, d_success, sizeof(bool), hipMemcpyDeviceToHost);

    // Verify the operation was successful
    EXPECT_TRUE(success) << "Send operation failed";

    // Check if data was correctly put into the send buffer (NCCL_PROTO_SIMPLE = 0)
    // float* h_sendBuffResult = new float[TEST_ELEM_COUNT];
    int* h_sendBuffResult = new int[TEST_ELEM_COUNT];
    hipMemcpy(h_sendBuffResult, d_sendBuff, TEST_ELEM_COUNT * sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < TEST_ELEM_COUNT; i++) {
        std::cout << "h_sendBuffResult[" << i << "] = " << h_sendBuffResult[i] << std::endl;
        std::cout << "h_inputBuf[" << i << "] = " << h_inputBuf[i] << std::endl;
        EXPECT_EQ(h_sendBuffResult[i], h_inputBuf[i]) << "Mismatch at element " << i;
    }

    // hipMemcpy(h_sendBuffResult, d_sendBuff[0][connIndexSend][NCCL_PROTO_SIMPLE], TEST_ELEM_COUNT * sizeof(float), hipMemcpyDeviceToHost);


    // for (int i = 0; i < TEST_ELEM_COUNT; i++) {
    //     std::cout << "h_sendBuffResult[" << i << "] = " << h_sendBuffResult[i] << std::endl;
    //     std::cout << "h_inputBuf[" << i << "] = " << h_inputBuf[i] << std::endl;
    //     EXPECT_EQ(h_sendBuffResult[i], h_inputBuf[i]) << "Mismatch at element " << i;
    // }

    // // Verify at least some data was transferred to the send buffer
    // float sum = 0;
    // for (int i = 0; i < TEST_ELEM_COUNT; i++) {
    //     sum += h_sendBuffResult[i];
    // }
    // EXPECT_NE(sum, 0.0f) << "No data appears to have been transferred to send buffer";

    // // Check if step counter was updated
    // uint64_t sendHead;

    // hipMemcpy(&sendHead, d_sendHead[testPeer][connIndexSend], sizeof(uint64_t), hipMemcpyDeviceToHost);
    // EXPECT_GT(sendHead, 0) << "Step counter was not updated";

    delete[] h_sendBuffResult;
    hipFree(d_success);
}

// // Test with different connection indices
// TEST_F(PrimitivesSendTest, DifferentConnectionIndices) {
//     // Use the second connection index
//     uint8_t testConnIndex = 1;

//     // Allocate memory for success flag on device
//     bool* d_success;
//     hipMalloc(&d_success, sizeof(bool));
//     bool success = false;
//     hipMemcpy(d_success, &success, sizeof(bool), hipMemcpyHostToDevice);

//     // Define parameters for the send operation
//     intptr_t inpIx = 0;  // Start from beginning of input buffer
//     int eltN = TEST_ELEM_COUNT;  // Number of elements to send

//     // Save initial step counter for verification
//     uint64_t initialSendHead;
//     hipMemcpy(&initialSendHead, d_sendHead[testPeer][testConnIndex], sizeof(uint64_t), hipMemcpyDeviceToHost);

//     // Launch test kernel with 128 threads
//     using TestRedOp = FuncSum<float>;
//     using TestFan = FanSymmetric<1>; // Simple fan-out
//     constexpr int slicePerChunk = 1;
//     constexpr int stepPerSlice = 1;
//     constexpr int unroll = 1;

//     dim3 blockSize(128);
//     dim3 gridSize(1);

//     hipLaunchKernelGGL(
//         (testSendKernel<float, TestRedOp, TestFan, 0, slicePerChunk, stepPerSlice, unroll, 0, 0, 0, false>),
//         gridSize, blockSize, 0, 0,
//         d_inputBuf, d_recvPeers, d_sendPeers, d_ncclShmem, inpIx, eltN, d_success, d_outputBuf,
//         testConnIndex, testConnIndex // Use the second connection index
//     );

//     // Check for kernel launch errors
//     hipError_t err = hipGetLastError();
//     EXPECT_EQ(err, hipSuccess) << "Kernel launch failed with alternate connection index: " << hipGetErrorString(err);

//     // Copy result back to host
//     hipMemcpy(&success, d_success, sizeof(bool), hipMemcpyDeviceToHost);

//     // Verify the operation was successful
//     EXPECT_TRUE(success) << "Send operation failed with alternate connection index";

//     // Check if data was correctly put into the send buffer (NCCL_PROTO_SIMPLE = 0)
//     float* h_sendBuffResult = new float[TEST_ELEM_COUNT];
//     hipMemcpy(h_sendBuffResult, d_sendBuff[testPeer][testConnIndex][NCCL_PROTO_SIMPLE], TEST_ELEM_COUNT * sizeof(float), hipMemcpyDeviceToHost);

//     // Verify at least some data was transferred to the send buffer
//     float sum = 0;
//     for (int i = 0; i < TEST_ELEM_COUNT; i++) {
//         sum += h_sendBuffResult[i];
//     }
//     EXPECT_NE(sum, 0.0f) << "No data appears to have been transferred to send buffer";

//     // Check if the correct step counter was updated
//     uint64_t sendHead;
//     hipMemcpy(&sendHead, d_sendHead[testPeer][testConnIndex], sizeof(uint64_t), hipMemcpyDeviceToHost);
//     EXPECT_GT(sendHead, initialSendHead) << "Step counter for connection index " << testConnIndex << " was not updated";

//     // Also check that the other connection's step counter was not affected
//     uint64_t otherSendHead;
//     hipMemcpy(&otherSendHead, d_sendHead[testPeer][0], sizeof(uint64_t), hipMemcpyDeviceToHost);

//     delete[] h_sendBuffResult;
//     hipFree(d_success);
// }

// // Test sending to different peers
// TEST_F(PrimitivesSendTest, MultiPeerSending) {
//     // For this test, we'll try sending to a different peer
//     int alternatePeer = 2;  // Test with peer 2

//     // Allocate memory for success flag on device
//     bool* d_success;
//     hipMalloc(&d_success, sizeof(bool));
//     bool success = false;
//     hipMemcpy(d_success, &success, sizeof(bool), hipMemcpyHostToDevice);

//     // Define parameters for the send operation
//     intptr_t inpIx = 0;  // Start from beginning of input buffer
//     int eltN = TEST_ELEM_COUNT;  // Number of elements to send

//     // Modify h_sendPeers to send to the alternate peer
//     int* modifiedSendPeers = new int[MAX_PEERS];
//     for (int i = 0; i < MAX_PEERS; i++) {
//         modifiedSendPeers[i] = -1;  // Initialize with terminator
//     }
//     modifiedSendPeers[0] = alternatePeer;  // Send to alternate peer

//     // Update the sendPeers on the device
//     hipMemcpy(d_sendPeers, modifiedSendPeers, MAX_PEERS * sizeof(int), hipMemcpyHostToDevice);

//     // Save initial step counter for verification
//     uint64_t initialSendHeadTestPeer, initialSendHeadAltPeer;
//     hipMemcpy(&initialSendHeadTestPeer, d_sendHead[testPeer][connIndexSend], sizeof(uint64_t), hipMemcpyDeviceToHost);
//     hipMemcpy(&initialSendHeadAltPeer, d_sendHead[alternatePeer][connIndexSend], sizeof(uint64_t), hipMemcpyDeviceToHost);

//     // Launch test kernel with 128 threads
//     using TestRedOp = FuncSum<float>;
//     using TestFan = FanSymmetric<1>;
//     constexpr int slicePerChunk = 1;
//     constexpr int stepPerSlice = 1;
//     constexpr int unroll = 1;

//     dim3 blockSize(128);
//     dim3 gridSize(1);

//     hipLaunchKernelGGL(
//         (testSendKernel<float, TestRedOp, TestFan, 0, slicePerChunk, stepPerSlice, unroll, 0, 0, 0, false>),
//         gridSize, blockSize, 0, 0,
//         d_inputBuf, d_recvPeers, d_sendPeers, d_ncclShmem, inpIx, eltN, d_success, d_outputBuf,
//         connIndexSend, connIndexRecv
//     );

//     // Check for kernel launch errors
//     hipError_t err = hipGetLastError();
//     EXPECT_EQ(err, hipSuccess) << "Kernel launch failed with alternate peer: " << hipGetErrorString(err);

//     // Copy result back to host
//     hipMemcpy(&success, d_success, sizeof(bool), hipMemcpyDeviceToHost);

//     // Verify the operation was successful
//     EXPECT_TRUE(success) << "Send operation failed with alternate peer";

//     // Check if step counter for alternate peer was updated
//     uint64_t sendHeadAltPeer, sendHeadTestPeer;
//     hipMemcpy(&sendHeadAltPeer, d_sendHead[alternatePeer][connIndexSend], sizeof(uint64_t), hipMemcpyDeviceToHost);
//     hipMemcpy(&sendHeadTestPeer, d_sendHead[testPeer][connIndexSend], sizeof(uint64_t), hipMemcpyDeviceToHost);

//     EXPECT_GT(sendHeadAltPeer, initialSendHeadAltPeer) << "Step counter for alternate peer was not updated";
//     EXPECT_EQ(sendHeadTestPeer, initialSendHeadTestPeer) << "Step counter for original peer should not have been updated";

//     // Check if data was correctly put into the alternate peer's send buffer
//     float* h_sendBuffResult = new float[TEST_ELEM_COUNT];
//     hipMemcpy(h_sendBuffResult, d_sendBuff[alternatePeer][connIndexSend][NCCL_PROTO_SIMPLE], TEST_ELEM_COUNT * sizeof(float), hipMemcpyDeviceToHost);

//     // Verify at least some data was transferred to the send buffer
//     float sum = 0;
//     for (int i = 0; i < TEST_ELEM_COUNT; i++) {
//         sum += h_sendBuffResult[i];
//     }
//     EXPECT_NE(sum, 0.0f) << "No data appears to have been transferred to alternate peer's send buffer";

//     delete[] h_sendBuffResult;
//     delete[] modifiedSendPeers;
//     hipFree(d_success);
// }

// // Main function to run all the tests
// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }