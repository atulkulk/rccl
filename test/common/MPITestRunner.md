# MPITestBase Framework

A C++ testing framework for multi-process RCCL tests using MPI (Message Passing Interface) and Google Test.

## Table of Contents
- [Overview](#overview)
- [Why Use MPI Testing?](#why-use-mpi-testing)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Per-Rank Logging](#per-rank-logging)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

`MPITestBase` is a base class for writing multi-process tests that verify RCCL features across multiple GPUs. It provides infrastructure for MPI-based distributed testing.

**Key Features:**
- ✅ Multi-process testing with MPI
- ✅ Automatic RCCL communicator management
- ✅ Process count validation (minimum processes, power-of-two requirements)
- ✅ HIP stream lifecycle management
- ✅ Integrated with Google Test framework
- ✅ Test-specific communicators for isolation

**Location:** `test/common/MPITestBase.hpp`

---

## Why Use MPI Testing?

### Problem: Single-Process Testing is Insufficient

RCCL (ROCm Communication Collectives Library) is designed for **multi-GPU, multi-node** communication. Testing these features requires:

1. **Multiple Processes** - Each GPU/rank runs in its own process
2. **Distributed Coordination** - Synchronization across processes
3. **Real Communication** - Actual data transfer between GPUs
4. **Collective Operations** - AllReduce, Broadcast, etc. require all ranks

**Example: Testing AllReduce**

```cpp
// ❌ CANNOT meaningfully test this with single process
void testAllReduce() {
  // AllReduce requires ALL ranks to participate
  ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream);

  // You need multiple processes to test actual collective behavior!
  // Threads within a single process DO NOT work - NCCL operates at process level
}
```

**With MPI Testing:**
```cpp
// ✅ CAN test with multiple MPI processes
TEST_F(MyMPITest, AllReduce) {
  // Rank 0: sends value 1.0
  // Rank 1: sends value 2.0
  // Rank 2: sends value 3.0
  // All ranks receive: 6.0 (sum)

  ncclAllReduce(send, recv, count, ncclFloat, ncclSum, comm, stream);

  // Each rank can verify the result
  EXPECT_EQ(result, 6.0f);
}
```

### Common Use Cases

1. **Collective Operations** - AllReduce, Broadcast, AllGather, ReduceScatter
2. **Point-to-Point Communication** - Send/Recv between specific ranks
3. **Transport Layer Testing** - P2P, SHM (shared memory), NET (network) transports
4. **Multi-Node Scenarios** - Cross-node communication
5. **Scalability Testing** - Behavior with different numbers of processes

---

## Quick Start

### Prerequisites

1. **MPI Implementation** - OpenMPI, MPICH, or similar
2. **Multiple GPUs** - At least 2 GPUs for most tests
3. **Build with MPI Support** - `MPI_TESTS_ENABLED` must be defined

### Basic Example

```cpp
#include "common/MPITestBase.hpp"

// Import constants for convenience
using namespace MPITestConstants;

class MyMPITest : public MPITestBase {
protected:
  void SetUp() override {
    // Optional: Add custom setup
  }
};

TEST_F(MyMPITest, BasicAllReduce) {
  // Validate we have enough processes
  validateTestPrerequisites(kMinProcessesForMPI);  // Requires at least 2

  // Create test-specific communicator
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1024;
  float* d_send;
  float* d_recv;

  // Allocate GPU memory
  HIPCHECK(hipMalloc(&d_send, N * sizeof(float)));
  HIPCHECK(hipMalloc(&d_recv, N * sizeof(float)));

  // Initialize with rank-specific data
  float value = RCCLMPIEnvironment::world_rank + 1.0f;
  HIPCHECK(hipMemset(d_send, value, N * sizeof(float)));

  // Perform AllReduce
  NCCLCHECK(ncclAllReduce(
      d_send, d_recv, N, ncclFloat, ncclSum,
      getActiveCommunicator(),
      getActiveStream()
  ));

  HIPCHECK(hipStreamSynchronize(getActiveStream()));

  // Verify result
  std::vector<float> result(N);
  HIPCHECK(hipMemcpy(result.data(), d_recv, N * sizeof(float),
                     hipMemcpyDeviceToHost));

  float expected = (RCCLMPIEnvironment::world_size *
                    (RCCLMPIEnvironment::world_size + 1)) / 2.0f;

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result[i], expected);
  }

  // Cleanup
  HIPCHECK(hipFree(d_send));
  HIPCHECK(hipFree(d_recv));
}
```

### Running MPI Tests

```bash
# Run with 2 processes on single node
mpirun -np 2 ./test_executable --gtest_filter="*MPI*"

# Run with 4 processes on single node
mpirun -np 4 ./test_executable --gtest_filter="MyMPITest.*"

# Run with specific GPU mapping on single node
mpirun -np 2 --bind-to none -x HIP_VISIBLE_DEVICES=0,1 ./test_executable

# Run across multiple nodes (2 processes per node, 4 total)
mpirun -np 4 -npernode 2 -hostfile hostfile.txt ./test_executable

# Run with 1 process per node across 2 nodes
mpirun -np 2 -npernode 1 --host node1,node2 ./test_executable
```

**Note:** The test validation only checks total process count, not node topology. A test requiring 2 processes will work with:
- 2 processes on 1 node (intra-node, shared memory)
- 1 process on each of 2 nodes (inter-node, network communication)

---

## Core Concepts

### 1. RCCLMPIEnvironment

Global MPI environment that manages MPI initialization and cleanup.

**Static Members:**
```cpp
RCCLMPIEnvironment::world_rank  // Current process rank (0 to N-1)
RCCLMPIEnvironment::world_size  // Total number of processes
RCCLMPIEnvironment::retCode     // Initialization status (0 = success)
```

**Lifecycle:**
- Initialized once before any tests run
- Calls `MPI_Init()` and sets up GPU-to-rank mapping
- Cleaned up after all tests complete
- Each rank is assigned to a unique GPU

### 2. MPITestBase Class

Base class providing common MPI test infrastructure.

**Key Methods:**
```cpp
class MPITestBase : public ::testing::Test {
protected:
  // Validate process count requirements
  void validateTestPrerequisites(int min_processes,
                                 bool require_power_of_two = false);

  // Create isolated communicator for this test
  ncclResult_t createTestCommunicator();

  // Get the active communicator
  ncclComm_t getActiveCommunicator();

  // Get the active HIP stream
  hipStream_t getActiveStream();

  // Cleanup resources (called automatically)
  void cleanupTestCommunicator();
};
```

### 3. Test-Specific Communicators

Each test can create its own RCCL communicator for isolation:

```cpp
TEST_F(MyTest, IsolatedTest) {
  // Create a fresh communicator just for this test
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Use it for operations
  ncclComm_t comm = getActiveCommunicator();
  hipStream_t stream = getActiveStream();

  // Automatically cleaned up in TearDown()
}
```

**Benefits:**
- Tests don't interfere with each other
- Clean state for each test
- Proper resource cleanup
- No shared memory conflicts

### 4. Process Validation

Ensure tests have the right number of processes:

```cpp
// Require at least 2 processes
validateTestPrerequisites(2);

// Require at least 4 processes
validateTestPrerequisites(4);

// Require power-of-two processes (2, 4, 8, 16, ...)
validateTestPrerequisites(2, kRequirePowerOfTwo);

// Minimum 2, but doesn't need to be power-of-two
validateTestPrerequisites(kMinProcessesForMPI, kNoPowerOfTwoRequired);
```

### 5. Synchronization

MPI barriers ensure all ranks reach certain points together:

```cpp
// Explicit barrier (use sparingly)
MPI_Barrier(MPI_COMM_WORLD);

// Barriers are automatically used in:
// - createTestCommunicator() (before and after)
// - cleanupTestCommunicator() (before and after)
```

---

## Per-Rank Logging

### Overview

By default, only **rank 0** output is displayed to the terminal to avoid cluttered logs from multiple processes. However, for debugging and detailed analysis, you can enable **per-rank logging** to capture output from all ranks into separate log files.

### Enabling Per-Rank Logging

Set the environment variable before running tests:

```bash
export RCCL_MPI_LOG_ALL_RANKS=1
mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter="MyTest.*"
```

Or in a single command:

```bash
RCCL_MPI_LOG_ALL_RANKS=1 mpirun -np 4 ./rccl-UnitTestsMPI
```

### Log Files

When enabled, log files are created for all ranks in the **current working directory**:

```
rccl_test_rank_0.log  (contains rank 0 output - also displayed on console)
rccl_test_rank_1.log  (contains all rank 1 output)
rccl_test_rank_2.log  (contains all rank 2 output)
rccl_test_rank_3.log  (contains all rank 3 output)
```

**Important Notes:**
- **Rank 0**: Output goes to BOTH console (for interactive monitoring) AND log file (for later analysis)
- **Rank 1-N**: Output goes only to log files
- **Location**: Log files are created in the directory where you execute the test command
- For multi-node runs, each node creates logs in its local working directory
- Ensure you have write permissions in the execution directory

### What Gets Logged

**For non-zero ranks (Ranks 1-N):**
All output is redirected to log files, including:
- Test progress and results (Google Test output)
- `printf()` and `std::cout` statements
- Error messages from NCCLCHECK, HIPCHECK, MPICHECK macros
- Debug output from RCCL internals
- Warnings and error messages

**For rank 0:**
- Output goes to BOTH console and log file (`rccl_test_rank_0.log`)
  - **Console**: For real-time interactive monitoring
  - **Log file**: For post-run analysis and comparison with other ranks
- Best of both worlds: watch progress live AND have complete logs for debugging

### Implementation Details

**How It Works:**
1. At startup, the framework checks `RCCL_MPI_LOG_ALL_RANKS` environment variable
2. If set to `1`, creates `rccl_test_rank_<N>.log` for each rank
3. **For Rank 0**: Uses "tee" functionality via pipe and background thread
   - Creates a pipe
   - Redirects stdout/stderr to pipe write end
   - Background thread reads from pipe and writes to BOTH original console and log file
4. **For Ranks 1-N**: Redirects stdout/stderr directly to log file
5. Disables buffering for immediate output
6. Restores original output streams on exit

**Output Behavior:**
- **Without per-rank logging** (default): Only rank 0 output shown on terminal
- **With per-rank logging**:
  - **Rank 0**: Output goes to BOTH console AND `rccl_test_rank_0.log`
    - Watch test progress in real-time on console
    - Complete log saved to file for later analysis
  - **Ranks 1-N**: Output redirected to `rccl_test_rank_<N>.log` files only
  - A clear banner message indicates per-rank logging is enabled

### Usage Examples

#### Example 1: Debugging a Specific Test

```bash
# Enable per-rank logging for a specific test
export RCCL_MPI_LOG_ALL_RANKS=1
mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter="P2PTest.DataTransfer"

# You'll see a banner message at startup:
# ========================================================================
#   Per-Rank Logging ENABLED (RCCL_MPI_LOG_ALL_RANKS=1)
# ========================================================================
#   Rank 0     : Output to BOTH console AND rccl_test_rank_0.log
#   Ranks 1-N  : Output redirected to rccl_test_rank_<N>.log
#   Location   : Log files created in current working directory
# ========================================================================
```

### Best Practices

**When to Enable:**
- ✅ Debugging test failures on non-zero ranks
- ✅ Investigating rank-specific behavior differences
- ✅ Analyzing communication patterns across ranks
- ✅ Capturing detailed RCCL internal logs from all processes
- ✅ Troubleshooting deadlocks or hangs

**When to Disable:**
- ✅ Normal test runs (cleaner output)
- ✅ CI/CD pipelines (unless debugging)
- ✅ Large-scale runs (many ranks generate many files)

### Adding Debug Output in Tests

Use rank-aware logging in your tests:

```cpp
TEST_F(MyMPITest, DebugExample) {
  validateTestPrerequisites(2);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  int rank = RCCLMPIEnvironment::world_rank;

  // This will appear in the rank's log file
  printf("Rank %d: Starting test with buffer size %zu\n", rank, buffer_size);

  // Perform operation
  NCCLCHECK(ncclAllReduce(...));

  // More debug output
  printf("Rank %d: AllReduce completed, result=%f\n", rank, result);

  // Only rank 0 prints summary (appears in all modes)
  if (rank == 0) {
    printf("Summary: All ranks completed successfully\n");
  }
}
```
---

## API Reference

### MPITestBase Methods

#### `validateTestPrerequisites()`
Validate that the test has sufficient processes to run.

```cpp
void validateTestPrerequisites(
    int min_processes,
    bool require_power_of_two = false
);
```

**Parameters:**
- `min_processes` - Minimum number of MPI processes required
- `require_power_of_two` - If true, process count must be power of 2

**Behavior:**
- If requirements not met: Test is skipped with `GTEST_SKIP()`
- Rank 0 prints reason for skipping
- Safe to call multiple times
- **Only validates total process count**, not node topology

**Node Topology:**
This validation checks **total process count only** - it works correctly regardless of node distribution:
- ✅ 2 processes on 1 node (single-node)
- ✅ 1 process on each of 2 nodes (multi-node)
- ✅ 4 processes on 2 nodes (2 processes per node)

RCCL automatically handles intra-node (shared memory) and inter-node (network) communication.

**Example:**
```cpp
// Need at least 2 processes (any node configuration)
validateTestPrerequisites(2);

// Need at least 4 processes AND must be power of 2
validateTestPrerequisites(4, kRequirePowerOfTwo);
```

#### `createTestCommunicator()`
Create a test-specific RCCL communicator.

```cpp
ncclResult_t createTestCommunicator();
```

**Returns:** `ncclSuccess` on success, error code otherwise

**What it does:**
1. Rank 0 generates unique ID via `ncclGetUniqueId()`
2. Broadcast ID to all ranks via `MPI_Bcast()`
3. MPI barrier for synchronization
4. Initialize RCCL communicator with `ncclCommInitRank()`
5. Create HIP stream
6. MPI barrier after initialization

**Example:**
```cpp
TEST_F(MyTest, Example) {
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  ncclComm_t comm = getActiveCommunicator();
  // Use comm for RCCL operations
}
```

#### `getActiveCommunicator()`
Get the current test communicator.

```cpp
ncclComm_t getActiveCommunicator();
```

**Returns:** The test-specific communicator, or `nullptr` with test failure

**Important:** Must call `createTestCommunicator()` first!

#### `getActiveStream()`
Get the current HIP stream.

```cpp
hipStream_t getActiveStream();
```

**Returns:** The test-specific stream, or `nullptr` with test failure

**Important:** Must call `createTestCommunicator()` first!

#### `cleanupTestCommunicator()`
Clean up test-specific resources.

```cpp
void cleanupTestCommunicator();
```

**What it does:**
1. MPI barrier before cleanup
2. Destroy HIP stream
3. Destroy RCCL communicator
4. MPI barrier after cleanup

**Note:** Automatically called in `TearDown()` - usually don't need to call manually.

### RCCLMPIEnvironment Static Members

```cpp
// Current process rank (0 to world_size-1)
int RCCLMPIEnvironment::world_rank;

// Total number of processes
int RCCLMPIEnvironment::world_size;

// Initialization return code (0 = success)
int RCCLMPIEnvironment::retCode;
```

### MPITestConstants

```cpp
namespace MPITestConstants {
  // Minimum processes for MPI tests
  constexpr int kMinProcessesForMPI = 2;

  // Flags for validateTestPrerequisites()
  constexpr bool kRequirePowerOfTwo = true;
  constexpr bool kNoPowerOfTwoRequired = false;

  // Helper function
  bool isPowerOfTwo(int n);
}
```

### Helper Macros

```cpp
// MPI error checking (test code)
MPICHECK(MPI_Function());

// RCCL error checking
NCCLCHECK(ncclFunction());

// HIP error checking
HIPCHECK(hipFunction());
```

---

## Examples

### Example 1: Basic AllReduce Test

```cpp
TEST_F(UnifiedMPITest, BasicAllReduce) {
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1024;

  // Each rank contributes its rank+1
  std::vector<float> send_data(N, RCCLMPIEnvironment::world_rank + 1.0f);
  std::vector<float> recv_data(N);

  float *d_send, *d_recv;
  HIPCHECK(hipMalloc(&d_send, N * sizeof(float)));
  HIPCHECK(hipMalloc(&d_recv, N * sizeof(float)));

  HIPCHECK(hipMemcpy(d_send, send_data.data(), N * sizeof(float),
                     hipMemcpyHostToDevice));

  // AllReduce: Sum across all ranks
  NCCLCHECK(ncclAllReduce(d_send, d_recv, N, ncclFloat, ncclSum,
                          getActiveCommunicator(), getActiveStream()));

  HIPCHECK(hipMemcpy(recv_data.data(), d_recv, N * sizeof(float),
                     hipMemcpyDeviceToHost));
  HIPCHECK(hipStreamSynchronize(getActiveStream()));

  // Verify: sum of (1 + 2 + 3 + ... + world_size)
  float expected = (RCCLMPIEnvironment::world_size *
                    (RCCLMPIEnvironment::world_size + 1)) / 2.0f;

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(recv_data[i], expected);
  }

  HIPCHECK(hipFree(d_send));
  HIPCHECK(hipFree(d_recv));
}
```

### Example 2: Broadcast Test

```cpp
TEST_F(UnifiedMPITest, Broadcast) {
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1000;
  std::vector<float> data(N);

  // Only rank 0 initializes data
  if (RCCLMPIEnvironment::world_rank == 0) {
    std::iota(data.begin(), data.end(), 1.0f);  // 1, 2, 3, ..., N
  }

  float *d_data;
  HIPCHECK(hipMalloc(&d_data, N * sizeof(float)));
  HIPCHECK(hipMemcpy(d_data, data.data(), N * sizeof(float),
                     hipMemcpyHostToDevice));

  // Broadcast from rank 0 to all ranks
  NCCLCHECK(ncclBroadcast(d_data, d_data, N, ncclFloat, 0,
                          getActiveCommunicator(), getActiveStream()));

  HIPCHECK(hipMemcpy(data.data(), d_data, N * sizeof(float),
                     hipMemcpyDeviceToHost));
  HIPCHECK(hipStreamSynchronize(getActiveStream()));

  // All ranks should have the same data now
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(data[i], i + 1.0f);
  }

  HIPCHECK(hipFree(d_data));
}
```

### Example 3: Send/Recv Between Ranks

```cpp
TEST_F(UnifiedMPITest, SimpleSendRecv) {
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1024;
  const int peer_rank = 1 - RCCLMPIEnvironment::world_rank;  // 0↔1

  float* d_send;
  float* d_recv;
  std::vector<float> h_send(N);
  std::vector<float> h_recv(N);

  // Each rank sends unique data
  for (int i = 0; i < N; i++) {
    h_send[i] = RCCLMPIEnvironment::world_rank * 1000 + i;
  }

  HIPCHECK(hipMalloc(&d_send, N * sizeof(float)));
  HIPCHECK(hipMalloc(&d_recv, N * sizeof(float)));
  HIPCHECK(hipMemcpy(d_send, h_send.data(), N * sizeof(float),
                     hipMemcpyHostToDevice));

  // Exchange data between ranks
  if (RCCLMPIEnvironment::world_rank == 0) {
    NCCLCHECK(ncclSend(d_send, N, ncclFloat, 1,
                       getActiveCommunicator(), getActiveStream()));
    NCCLCHECK(ncclRecv(d_recv, N, ncclFloat, 1,
                       getActiveCommunicator(), getActiveStream()));
  } else {
    NCCLCHECK(ncclRecv(d_recv, N, ncclFloat, 0,
                       getActiveCommunicator(), getActiveStream()));
    NCCLCHECK(ncclSend(d_send, N, ncclFloat, 0,
                       getActiveCommunicator(), getActiveStream()));
  }

  HIPCHECK(hipStreamSynchronize(getActiveStream()));
  HIPCHECK(hipMemcpy(h_recv.data(), d_recv, N * sizeof(float),
                     hipMemcpyDeviceToHost));

  // Verify received peer's data
  for (int i = 0; i < N; i++) {
    float expected = peer_rank * 1000 + i;
    EXPECT_FLOAT_EQ(h_recv[i], expected);
  }

  HIPCHECK(hipFree(d_send));
  HIPCHECK(hipFree(d_recv));
}
```

### Example 4: Testing Different Reduction Operations

```cpp
TEST_F(UnifiedMPITest, AllReduceMaxOperation) {
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 512;
  float* d_send;
  float* d_recv;
  std::vector<float> h_send(N);
  std::vector<float> h_recv(N);

  // Each rank sends rank*10 + index
  for (int i = 0; i < N; i++) {
    h_send[i] = RCCLMPIEnvironment::world_rank * 10 + i;
  }

  HIPCHECK(hipMalloc(&d_send, N * sizeof(float)));
  HIPCHECK(hipMalloc(&d_recv, N * sizeof(float)));
  HIPCHECK(hipMemcpy(d_send, h_send.data(), N * sizeof(float),
                     hipMemcpyHostToDevice));

  // AllReduce with MAX operation
  NCCLCHECK(ncclAllReduce(d_send, d_recv, N, ncclFloat, ncclMax,
                          getActiveCommunicator(), getActiveStream()));

  HIPCHECK(hipMemcpy(h_recv.data(), d_recv, N * sizeof(float),
                     hipMemcpyDeviceToHost));
  HIPCHECK(hipStreamSynchronize(getActiveStream()));

  // Maximum should be from highest rank
  for (int i = 0; i < N; i++) {
    float expected = (RCCLMPIEnvironment::world_size - 1) * 10 + i;
    EXPECT_FLOAT_EQ(h_recv[i], expected);
  }

  HIPCHECK(hipFree(d_send));
  HIPCHECK(hipFree(d_recv));
}
```

### Example 5: Power-of-Two Requirement

```cpp
TEST_F(MyMPITest, AdvancedAlgorithm) {
  // This algorithm requires power-of-two processes
  validateTestPrerequisites(4, kRequirePowerOfTwo);

  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Test only runs with 4, 8, 16, 32, ... processes
  // Automatically skipped if run with 3, 5, 6, 7, ... processes

  // Your test logic here
}
```

### Example 6: Custom Test Class with Helpers

```cpp
class MyTransportTest : public MPITestBase {
protected:
  void* send_buffer = nullptr;
  void* recv_buffer = nullptr;
  size_t buffer_size = 1024 * 1024;  // 1MB

  void SetUp() override {
    validateTestPrerequisites(kMinProcessesForMPI);
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    // Allocate buffers
    HIPCHECK(hipMalloc(&send_buffer, buffer_size));
    HIPCHECK(hipMalloc(&recv_buffer, buffer_size));
  }

  void TearDown() override {
    // Cleanup buffers
    if (send_buffer) hipFree(send_buffer);
    if (recv_buffer) hipFree(recv_buffer);

    // Base class cleanup (communicator)
    MPITestBase::TearDown();
  }

  void initializeBuffer(void* buffer, uint8_t value) {
    HIPCHECK(hipMemset(buffer, value, buffer_size));
  }
};

TEST_F(MyTransportTest, DataTransfer) {
  initializeBuffer(send_buffer, 0xAB);
  initializeBuffer(recv_buffer, 0x00);

  // Perform transfer
  // ... test logic ...

  // Verify
  std::vector<uint8_t> result(buffer_size);
  HIPCHECK(hipMemcpy(result.data(), recv_buffer, buffer_size,
                     hipMemcpyDeviceToHost));

  for (size_t i = 0; i < buffer_size; i++) {
    EXPECT_EQ(result[i], 0xAB);
  }
}
```

---

## Best Practices

### 1. Always Validate Prerequisites

```cpp
TEST_F(MyTest, SomeTest) {
  // ✅ GOOD: Validate first
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Test logic...
}

// ❌ BAD: No validation
TEST_F(MyTest, SomeTest) {
  // Might crash if only 1 process!
  ncclAllReduce(...);
}
```

### 2. Create Test-Specific Communicators

```cpp
// ✅ GOOD: Isolated communicator per test
TEST_F(MyTest, Test1) {
  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  // Test logic with fresh communicator
}

TEST_F(MyTest, Test2) {
  ASSERT_EQ(ncclSuccess, createTestCommunicator());
  // Another test with its own communicator
}
```

**Why?** Avoids shared memory conflicts and ensures clean state.

### 3. Use Rank-Specific Logic When Needed

```cpp
TEST_F(MyTest, Example) {
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  int rank = RCCLMPIEnvironment::world_rank;

  if (rank == 0) {
    // Only rank 0 prints summary
    printf("Starting test with %d processes\n",
           RCCLMPIEnvironment::world_size);
  }

  // All ranks execute this
  performCollectiveOperation();

  if (rank == 0) {
    // Only rank 0 does final verification
    verifyGlobalState();
  }
}
```

### 4. Clean Up Resources Properly

```cpp
TEST_F(MyTest, Example) {
  float* d_buffer = nullptr;

  try {
    HIPCHECK(hipMalloc(&d_buffer, 1024 * sizeof(float)));

    // Test logic...

    HIPCHECK(hipFree(d_buffer));
    d_buffer = nullptr;
  } catch (...) {
    if (d_buffer) hipFree(d_buffer);
    throw;
  }
}
```

### 5. Use Descriptive Test Names

```cpp
// ✅ GOOD: Clear what's being tested
TEST_F(MyMPITest, AllReduce_WithFloat32_Sum_2Ranks)
TEST_F(MyMPITest, Broadcast_LargeBuffer_FromRank0)
TEST_F(MyMPITest, SendRecv_PeerToPeer_1MBTransfer)

// ❌ BAD: Vague names
TEST_F(MyMPITest, Test1)
TEST_F(MyMPITest, TestAllReduce)
```

### 6. Check Return Codes

```cpp
// ✅ GOOD: Check all return codes
ASSERT_EQ(ncclSuccess, createTestCommunicator());
NCCLCHECK(ncclAllReduce(...));
HIPCHECK(hipMalloc(&ptr, size));

// ❌ BAD: Ignoring return values
createTestCommunicator();  // Might fail silently!
ncclAllReduce(...);         // Could return error
hipMalloc(&ptr, size);      // Allocation might fail
```

### 7. Synchronize Appropriately

```cpp
// ✅ GOOD: Synchronize before checking results
NCCLCHECK(ncclAllReduce(...));
HIPCHECK(hipStreamSynchronize(getActiveStream()));
// Now safe to verify results

// ❌ BAD: Check results without sync
NCCLCHECK(ncclAllReduce(...));
EXPECT_EQ(result, expected);  // Operation might not be done!
```

### 8. Consider Process Count in Expectations

```cpp
TEST_F(MyTest, AllReduceSum) {
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Each rank sends 1
  // Expected result depends on world_size
  float expected = RCCLMPIEnvironment::world_size * 1.0f;

  // ✅ GOOD: Expectation adapts to process count
  EXPECT_FLOAT_EQ(result, expected);

  // ❌ BAD: Hard-coded expectation
  // EXPECT_FLOAT_EQ(result, 2.0f);  // Only works with 2 processes!
}
```

---

## Troubleshooting

### MPI Initialization Failed

**Symptom:** `RCCLMPIEnvironment::retCode != 0`

**Causes:**
- Insufficient GPUs for the number of processes
- MPI not properly installed
- Wrong MPI launcher configuration

**Solutions:**
```cpp
// Check in test
if (RCCLMPIEnvironment::retCode != 0) {
  GTEST_SKIP() << "MPI initialization failed";
}

// Or use validateTestPrerequisites which checks this
validateTestPrerequisites(kMinProcessesForMPI);
```

```bash
# Ensure GPU count matches process count
mpirun -np 2 -x HIP_VISIBLE_DEVICES=0,1 ./test_executable

# Check GPU availability
rocm-smi --showid
```

### Test Hangs / Deadlock

**Symptom:** Test never completes, all ranks waiting.

**Debugging:** Enable per-rank logging to see where each rank gets stuck:
```bash
# Run with per-rank logging
export RCCL_MPI_LOG_ALL_RANKS=1
mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter="HangingTest.*" &

# Monitor logs in real-time to see where each rank stops
tail -f rccl_test_rank_*.log
```

**Common Causes:**

1. **Mismatched Collectives:**
```cpp
// ❌ BAD: Not all ranks participate
if (rank == 0) {
  ncclAllReduce(...);  // Other ranks don't call this!
}

// ✅ GOOD: All ranks participate
ncclAllReduce(...);
```

2. **Missing Synchronization:**
```cpp
// ❌ BAD: Rank 0 waits for data other ranks haven't sent
MPI_Barrier(MPI_COMM_WORLD);  // All ranks must reach this

// ✅ GOOD: createTestCommunicator() includes barriers
ASSERT_EQ(ncclSuccess, createTestCommunicator());
```

3. **Stream Not Synchronized:**
```cpp
// ✅ GOOD: Wait for operations to complete
HIPCHECK(hipStreamSynchronize(getActiveStream()));
MPI_Barrier(MPI_COMM_WORLD);
```

**See:** [Per-Rank Logging](#per-rank-logging) for detailed debugging techniques

### Rank Mismatch in ncclCommInitRank

**Symptom:** Error during communicator creation.

**Solution:**
```cpp
// ✅ GOOD: Use createTestCommunicator()
ASSERT_EQ(ncclSuccess, createTestCommunicator());

// ❌ BAD: Manual initialization can get ranks wrong
ncclCommInitRank(&comm, size, id, wrong_rank);
```

### GPU Out of Memory

**Symptom:** `hipMalloc()` fails with memory error.

**Solutions:**
1. Reduce buffer sizes
2. Ensure proper cleanup of previous allocations
3. Check for memory leaks
4. Run tests sequentially instead of in parallel

```cpp
// Clean up in TearDown
void TearDown() override {
  if (buffer) {
    hipFree(buffer);
    buffer = nullptr;
  }
  MPITestBase::TearDown();
}
```

### Test Skipped: Not Enough Processes

**Symptom:** Test skipped with message about process count.

**Solution:** Run with more processes:
```bash
# Test requires 4 processes
mpirun -np 4 ./test_executable --gtest_filter="MyTest.SomeTest"

# Check what test needs
grep validateTestPrerequisites test_file.cpp
```

### Wrong Results Across Ranks

**Symptom:** Different ranks get different results.

**Solution:** Enable per-rank logging to see output from all ranks:
```bash
# Enable per-rank logging for detailed debugging
export RCCL_MPI_LOG_ALL_RANKS=1
mpirun -np 4 ./rccl-UnitTestsMPI --gtest_filter="FailingTest.*"

# Check logs from each rank
cat rccl_test_rank_0.log
cat rccl_test_rank_1.log
```

**Debugging:**
```cpp
// Print from all ranks to debug
printf("Rank %d: result[0] = %f\n",
       RCCLMPIEnvironment::world_rank, result[0]);
MPI_Barrier(MPI_COMM_WORLD);

// Verify all ranks agree
float local_result = result[0];
float global_result;
MPI_Allreduce(&local_result, &global_result, 1, MPI_FLOAT,
              MPI_MAX, MPI_COMM_WORLD);
EXPECT_FLOAT_EQ(local_result, global_result);
```

**See:** [Per-Rank Logging](#per-rank-logging) for more details

---

## Advanced Topics

### Extending MPITestBase

Create specialized base classes for specific test types:

```cpp
class TransportTestBase : public MPITestBase {
protected:
  ncclConnector send_connector = {};
  ncclConnector recv_connector = {};

  void initializeTransport() {
    // Transport-specific setup
  }

  void SetUp() override {
    validateTestPrerequisites(kMinProcessesForMPI);
    ASSERT_EQ(ncclSuccess, createTestCommunicator());
    initializeTransport();
  }
};

TEST_F(TransportTestBase, P2PTransfer) {
  // Transport already initialized
  // Just write your test logic
}
```

### Testing with Multiple Communicators

```cpp
TEST_F(MyTest, MultipleComms) {
  validateTestPrerequisites(4);

  // Create two separate communicators
  ncclUniqueId id1, id2;
  ncclComm_t comm1, comm2;

  if (RCCLMPIEnvironment::world_rank == 0) {
    ncclGetUniqueId(&id1);
    ncclGetUniqueId(&id2);
  }

  MPI_Bcast(&id1, sizeof(id1), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&id2, sizeof(id2), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclCommInitRank(&comm1, RCCLMPIEnvironment::world_size, id1,
                   RCCLMPIEnvironment::world_rank);
  ncclCommInitRank(&comm2, RCCLMPIEnvironment::world_size, id2,
                   RCCLMPIEnvironment::world_rank);

  // Use both communicators...

  ncclCommDestroy(comm1);
  ncclCommDestroy(comm2);
}
```

### Testing Error Conditions

```cpp
TEST_F(MyTest, InvalidRankHandling) {
  validateTestPrerequisites(kMinProcessesForMPI);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Deliberately use invalid rank
  int invalid_rank = 999;

  // Expect error (don't crash)
  ncclResult_t result = ncclSend(buffer, count, type, invalid_rank,
                                  getActiveCommunicator(), getActiveStream());

  EXPECT_NE(result, ncclSuccess);
  // Test should continue, not deadlock
}
```

---

## FAQ

**Q: When should I use MPITestBase vs ProcessIsolatedTestRunner?**

A:
- **MPITestBase**: For testing multi-process RCCL operations (collectives, transport layers)
- **ProcessIsolatedTestRunner**: For testing single-process code with environment isolation

**Q: How many processes should I test with?**

A:
- Minimum: 2 (for basic collectives and P2P)
- Common: 2, 4, 8 (good coverage)
- For scalability: Test with various counts
- For algorithms: Some require power-of-two (use `kRequirePowerOfTwo`)

**Q: Does `validateTestPrerequisites()` work correctly across multiple nodes?**

A: Yes! The validation only checks total process count, not node distribution:
- Test requiring 2 processes works with: 2 procs on 1 node OR 1 proc on each of 2 nodes
- Test requiring 4 processes works with: 4 procs on 1 node OR 2 procs on 2 nodes, etc.
- RCCL automatically handles both intra-node (shared memory) and inter-node (network) communication

**Q: Can I run MPI tests locally?**

A: Yes, if you have:
- Multiple GPUs in your system
- MPI installed (OpenMPI, MPICH, etc.)
- Tests built with `MPI_TESTS_ENABLED`

**Q: How do I debug a specific rank?**

A:
```bash
# Method 1: Use per-rank logging (easiest)
export RCCL_MPI_LOG_ALL_RANKS=1
mpirun -np 4 ./test_executable
# Check rccl_test_rank_2.log for rank 2 output

# Method 2: GDB with MPI
mpirun -np 2 xterm -e gdb ./test_executable

# Method 3: Attach to specific rank PID
mpirun -np 4 ./test_executable &
gdb -p <rank_pid>

# Method 4: Print debugging from specific rank
if (RCCLMPIEnvironment::world_rank == 2) {
  printf("Debug info from rank 2...\n");
}
```

**See:** [Per-Rank Logging](#per-rank-logging) for comprehensive debugging guide

## See Also

- **MPITestBase.hpp** - Full API documentation
- **MPITestBase.cpp** - Implementation details
- **RCCLMPIEnvironment.hpp** - MPI environment setup
- **MPITests.cpp** - Collective operation examples
- **transport/P2pMPITests.cpp** - P2P transport examples
- **transport/ShmMPITests.cpp** - Shared memory transport examples
- **transport/NetMPITests.cpp** - Network transport examples


