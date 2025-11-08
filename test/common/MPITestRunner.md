# MPI Test Runner (Google Test)

A simple C++ testing framework for multi-process RCCL tests using MPI (Message Passing Interface) and **Google Test**.

> **📝 Note:** This guide mostly covers **Google Test-based** MPI testing. For standalone tests refer to [Standalone Tests](#standalone-tests).

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
- [Standalone Tests](#standalone-tests)

---

## Overview

`MPITestBase` is a Google Test adapter for writing multi-process tests that verify RCCL features across multiple GPUs. It provides infrastructure for MPI-based distributed testing.

**Key Features:**
- ✅ Multi-process testing with MPI
- ✅ Automatic RCCL communicator management
- ✅ Process count validation (minimum processes, power-of-two requirements)
- ✅ Node count validation (single-node vs multi-node constraints)
- ✅ HIP stream lifecycle management
- ✅ Integrated with Google Test framework
- ✅ Test-specific communicators for isolation
- ✅ Framework-agnostic core (can be used without GTest)

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
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"

// Import constants and guards
using namespace MPITestConstants;
using namespace RCCLTestGuards;

class MyMPITest : public MPITestBase {
protected:
  void SetUp() override {
    // Optional: Add custom setup
  }
};

TEST_F(MyMPITest, BasicAllReduce) {
  // Validate we have enough processes (uses defaults for other parameters)
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));  // min=2, no max, any nodes

  // Create test-specific communicator
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1024;
  float* d_send = nullptr;
  float* d_recv = nullptr;

  // Allocate GPU memory with RAII guards for automatic cleanup
  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_send, N * sizeof(float)));
  auto send_guard = makeDeviceBufferAutoGuard(d_send);

  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_recv, N * sizeof(float)));
  auto recv_guard = makeDeviceBufferAutoGuard(d_recv);

  // Initialize with rank-specific data
  float value = MPIEnvironment::world_rank + 1.0f;
  HIP_TEST_CHECK_GTEST_FAIL(hipMemset(d_send, value, N * sizeof(float)));

  // Perform AllReduce
  RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(
      d_send, d_recv, N, ncclFloat, ncclSum,
      getActiveCommunicator(),
      getActiveStream()
  ));

  HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(getActiveStream()));

  // Verify result
  std::vector<float> result(N);
  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(result.data(), d_recv, N * sizeof(float),
                                       hipMemcpyDeviceToHost));

  float expected = (MPIEnvironment::world_size *
                    (MPIEnvironment::world_size + 1)) / 2.0f;

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(result[i], expected);
  }

  // Automatic cleanup via RAII guards - no manual hipFree() needed!
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

# Run across multiple nodes using hostfile (recommended)
# Create hostfile with slots per node:
cat > hostfile.txt << EOF
node-1 slots=8
node-2 slots=8
EOF
mpirun -np 16 --hostfile hostfile.txt ./test_executable

# Or let SLURM auto-generate hostfile (if using test runner script)
salloc -N 2 -n 16 --time=01:00:00
./build_test_coverage.std.sh --config test_config.txt --no-build
# Script automatically detects nodes and creates temporary hostfile
```

**Important Notes:**
- **CPU binding disabled**: Tests run with `--bind-to none` to avoid "more processes than CPUs" errors
- **GPU assignment**: Each node independently assigns GPUs 0-N to local ranks 0-N
- **Multi-node**: Script auto-generates hostfiles with proper slot counts from SLURM allocations

**Important: Node Validation**

Tests can specify node requirements to ensure they run in the correct environment:

| Node Requirement | Constant | Use Case Examples |
|------------------|----------|-------------------|
| Single-node only | `kRequireSingleNode` | Tests requiring direct GPU access, shared memory, or specific hardware topology |
| Any number of nodes | `kNoNodeLimit` (default) | Tests designed for distributed execution or network-based features |

```cpp
// Test that requires single-node execution
validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode);

// Test that works on any number of nodes (default - can omit last parameters)
validateTestPrerequisites(2);  // Uses defaults: no max processes, any nodes
```

**Common Use Cases:**
- **Single-node requirement**: P2P transport, SHM transport, GPU topology tests, local memory tests
- **Multi-node capable**: NET transport, distributed collectives, scalability tests

---

## Core Concepts

### 1. MPIEnvironment

Global MPI environment that manages MPI initialization and cleanup.

**Static Members:**
```cpp
MPIEnvironment::world_rank  // Current process rank (0 to N-1)
MPIEnvironment::world_size  // Total number of processes
MPIEnvironment::retCode     // Initialization status (0 = success)
```

**Lifecycle:**
- Initialized once before any tests run
- Calls `MPI_Init()` and sets up GPU-to-rank mapping
- Cleaned up after all tests complete
- Each rank is assigned to a unique GPU **based on local rank** (rank within node)

**Multi-Node GPU Assignment:**
For multi-node configurations, GPU assignment uses **local rank** (rank within the node) rather than global rank:
- **Node 1**: Global ranks 0-7 → Local ranks 0-7 → GPUs 0-7
- **Node 2**: Global ranks 8-15 → Local ranks 0-7 → GPUs 0-7

This ensures proper GPU mapping across all nodes without requiring unique GPU IDs globally.

**Process Distribution Display:**
At startup, the framework automatically displays detailed process distribution:
```
=== MPI Process Distribution ===
Total processes: 16
Detected nodes:  2

Node 0: node-3 (8 ranks)
  Ranks: 0, 1, 2, 3, 4, 5, 6, 7
Node 1: node-21 (8 ranks)
  Ranks: 8, 9, 10, 11, 12, 13, 14, 15
================================
```

This helps verify correct process placement and node allocation before tests run.

### 2. MPITestBase Class

Base class providing common MPI test infrastructure.

**Key Methods:**
```cpp
class MPITestBase : public ::testing::Test {
protected:
  // Validate process and node count requirements
  bool validateTestPrerequisites(int min_processes = 1,
                                 int max_processes = kNoProcessLimit,
                                 bool require_power_of_two = false,
                                 int min_nodes = 1,
                                 int max_nodes = kNoNodeLimit);

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

### 4. Process and Node Validation

Ensure tests have the right number of processes and correct node configuration:

```cpp
// Require at least 2 processes (uses defaults for other parameters)
validateTestPrerequisites(2);  // min=2, no max, not power-of-two, any nodes

// Require at least 4 processes
validateTestPrerequisites(4);

// Require power-of-two processes (2, 4, 8, 16, ...)
validateTestPrerequisites(2, kNoProcessLimit, kRequirePowerOfTwo);

// Require exactly 2 processes (min=2, max=2)
validateTestPrerequisites(2, 2);  // Uses defaults for other parameters

// Test that requires single-node (e.g., P2P transport, shared memory tests)
validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode);

// Test that requires at least 2 nodes
validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 2);

// Test with 4-16 processes, power-of-two, single-node only
validateTestPrerequisites(4, 16, kRequirePowerOfTwo, 1, kRequireSingleNode);
```

**Node Detection:**
The framework automatically detects the number of unique nodes by gathering hostnames from all MPI ranks. This works transparently with any job scheduler (SLURM, PBS, etc.) and both single-node and multi-node configurations.

**When to Use Node Validation:**
- Use `kRequireSingleNode` when your test requires all processes to be on the same physical node
- Use `kNoNodeLimit` (default) when your test can work across multiple nodes
- The validation automatically skips tests that don't meet node requirements

### 5. Synchronization

MPI barriers ensure all ranks reach certain points together:

```cpp
// Explicit barrier (use sparingly)
ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

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

**Banner Display:**
The per-rank logging banner is displayed using `TEST_INFO` macros, which means:
- **With `NCCL_DEBUG=INFO`**: Full banner with details shown
- **Without `NCCL_DEBUG`**: Banner content not shown (minimal output)
- This allows clean output in production while providing detailed info during debugging

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
3. Displays a banner using `TEST_INFO` macros (visible with `NCCL_DEBUG=INFO`)
4. **For Rank 0**: Uses "tee" functionality via pipe and background thread
   - Creates a pipe
   - Redirects stdout/stderr to pipe write end
   - Background thread reads from pipe and writes to BOTH original console and log file
5. **For Ranks 1-N**: Redirects stdout/stderr directly to log file
6. Disables buffering for immediate output
7. Restores original output streams on exit

**Output Behavior:**
- **Without per-rank logging** (default): Only rank 0 output shown on terminal
- **With per-rank logging**:
  - **Rank 0**: Output goes to BOTH console AND `rccl_test_rank_0.log`
    - Watch test progress in real-time on console
    - Complete log saved to file for later analysis
  - **Ranks 1-N**: Output redirected to `rccl_test_rank_<N>.log` files only
  - Banner visible when `NCCL_DEBUG=INFO` is set (TEST_INFO macros)
  - Without `NCCL_DEBUG`, logging works silently with no banner clutter

### Usage Examples

#### Example 1: Debugging a Specific Test

```bash
# Enable per-rank logging with NCCL_DEBUG for full banner
export RCCL_MPI_LOG_ALL_RANKS=1
NCCL_DEBUG=INFO mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter="P2PTest.DataTransfer"

# You'll see a banner message at startup (with NCCL_DEBUG=INFO):
# [0] TEST INFO Per-Rank Logging ENABLED (RCCL_MPI_LOG_ALL_RANKS=1)
# [0] TEST INFO Rank 0     : Output to BOTH console AND rccl_test_rank_0.log
# [0] TEST INFO Ranks 1-N  : Output redirected to rccl_test_rank_<N>.log
# [0] TEST INFO Location   : Log files created in current working directory

# Without NCCL_DEBUG (minimal output):
export RCCL_MPI_LOG_ALL_RANKS=1
mpirun -np 2 ./rccl-UnitTestsMPI --gtest_filter="P2PTest.DataTransfer"
# (banner content not shown, but logging still enabled)
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

Use `TEST_*` macros for conditional, rank-aware logging that respects `NCCL_DEBUG`:

```cpp
TEST_F(MyMPITest, DebugExample) {
  validateTestPrerequisites(2);
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // TEST_INFO respects NCCL_DEBUG=INFO setting
  // Automatically includes rank (and hostname for multi-node)
  TEST_INFO("Starting test with buffer size %zu", buffer_size);

  // Perform operation
  NCCLCHECK(ncclAllReduce(...));

  // More debug output with automatic rank prefix
  TEST_INFO("AllReduce completed, result=%f", result);

  // Only rank 0 prints summary
  if (MPIEnvironment::world_rank == 0) {
    TEST_INFO("Summary: All ranks completed successfully");
  }
}
```

**Available TEST_* Macros:**
```cpp
TEST_WARN("Warning message");    // NCCL_DEBUG=WARN or higher
TEST_INFO("Info message");       // NCCL_DEBUG=INFO or higher
TEST_ABORT("Abort message");     // NCCL_DEBUG=ABORT or higher
TEST_TRACE("Trace message");     // NCCL_DEBUG=TRACE
```

**Output Format:**
- Single-node: `[rank] TEST INFO <message>`
- Multi-node: `hostname:[rank] TEST INFO <message>`

**Example Output:**
```bash
# Single-node with NCCL_DEBUG=INFO
[0] TEST INFO Starting test with buffer size 1024
[1] TEST INFO Starting test with buffer size 1024

# Multi-node with NCCL_DEBUG=INFO
mi300x-3:[0] TEST INFO Starting test with buffer size 1024
mi300x-4:[1] TEST INFO Starting test with buffer size 1024
```

---

## API Reference

### MPITestBase Methods

#### `validateTestPrerequisites()`
Validate that the test has sufficient processes and correct node configuration to run.

```cpp
bool validateTestPrerequisites(
    int min_processes = 1,
    int max_processes = kNoProcessLimit,
    bool require_power_of_two = false,
    int min_nodes = 1,
    int max_nodes = kNoNodeLimit
);
```

**Parameters:**
- `min_processes` - Minimum number of MPI processes required (default: 1)
- `max_processes` - Maximum number of MPI processes allowed (default: 0 = no limit)
- `require_power_of_two` - If true, process count must be power of 2 (default: false)
- `min_nodes` - Minimum number of nodes required (default: 1)
- `max_nodes` - Maximum number of nodes allowed (default: 0 = no limit)

**Returns:**
- `true` if all requirements are met
- `false` if requirements not met (test should skip)

**Behavior:**
- Displays detailed requirements and current environment on rank 0
- If requirements not met: Returns false, typically used with `GTEST_SKIP()`
- Provides clear error messages explaining what's wrong
- Safe to call multiple times
- Automatically detects node count by gathering hostnames from all ranks
- Only performs node detection when node constraints are specified (min_nodes > 1 or max_nodes > 0)

**Process Validation:**
- `min_processes`: Minimum processes needed (test skips if world_size < min)
- `max_processes = 0` (default): No upper limit on process count
- `max_processes = N`: Test requires at most N processes (e.g., 2 = exactly 2 if min=2)
- When min_processes == max_processes: Test requires exactly that many processes

**Node Validation:**
- `min_nodes = 1` (default): No minimum node constraint
- `min_nodes = N`: Test requires at least N nodes
- `max_nodes = 0` (default): No node limit - test works on any number of nodes
- `max_nodes = N`: Test requires at most N nodes (e.g., 1 = single-node only)
- When min_nodes == max_nodes: Test requires exactly that many nodes

**Common Use Cases:**
- **Flexible process count**: Use only min_processes, let max default to 0
- **Exact process count**: Set min_processes = max_processes
- **Single-node features**: Set max_nodes = 1 (P2P, SHM, shared memory)
- **Multi-node required**: Set min_nodes > 1 (NET transport testing)
- **Power-of-two algorithms**: Set require_power_of_two = true

**Examples:**
```cpp
// Need at least 2 processes (any node configuration)
validateTestPrerequisites(2);

// Exactly 4 processes
validateTestPrerequisites(4, 4);

// At least 4 processes AND must be power of 2
validateTestPrerequisites(4, kNoProcessLimit, kRequirePowerOfTwo);

// Single-node only feature (P2P, SHM, or any intra-node algorithm)
validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode);

// Requires at least 2 nodes (multi-node testing)
validateTestPrerequisites(4, kNoProcessLimit, kNoPowerOfTwoRequired, 2);

// Exactly 8 processes, power-of-two, single-node only
validateTestPrerequisites(8, 8, kRequirePowerOfTwo, 1, kRequireSingleNode);

// 4-16 processes, must be on exactly 2 nodes
validateTestPrerequisites(4, 16, kNoPowerOfTwoRequired, 2, 2);
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

### MPIEnvironment Static Members

```cpp
// Current process rank (0 to world_size-1)
int MPIEnvironment::world_rank;

// Total number of processes
int MPIEnvironment::world_size;

// Initialization return code (0 = success)
int MPIEnvironment::retCode;
```

### MPITestConstants

```cpp
namespace MPITestConstants {
  // Minimum processes for MPI tests
  constexpr int kMinProcessesForMPI = 2;

  // Flags for process count validation
  constexpr bool kRequirePowerOfTwo = true;
  constexpr bool kNoPowerOfTwoRequired = false;
  constexpr int kNoProcessLimit = 0;     // No upper limit on process count

  // Flags for node count validation
  constexpr int kRequireSingleNode = 1;  // For single-node only tests
  constexpr int kNoNodeLimit = 0;        // For multi-node capable tests (default)

  // Helper functions
  bool isPowerOfTwo(int n);
  int detectNodeCount();  // Detects unique nodes via hostnames
}
```

**Node Detection:**
The `detectNodeCount()` function automatically detects the number of unique physical nodes by:
1. Each rank gets its hostname via `MPI_Get_processor_name()`
2. All hostnames are gathered to rank 0
3. Rank 0 counts unique hostnames
4. Count is broadcast to all ranks
5. Returns number of unique nodes

**Usage Example:**
```cpp
// Automatically called by validateTestPrerequisites()
int nodes = MPITestConstants::detectNodeCount();
printf("Detected %d unique node(s)\n", nodes);
```

### Helper Macros

```cpp
// MPI error checking
ASSERT_MPI_SUCCESS(MPI_Function());  // GTest assertion-based

// RCCL error checking
RCCL_TEST_CHECK_GTEST_FAIL(ncclFunction());  // GTest FAIL on error

// HIP error checking
HIP_TEST_CHECK_GTEST_FAIL(hipFunction());  // GTest FAIL on error
```

---

## Examples

### Example 1: Basic AllReduce Test

```cpp
TEST_F(UnifiedMPITest, BasicAllReduce) {
  // Validate we have at least 2 processes (uses defaults for other parameters)
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1024;

  // Each rank contributes its rank+1
  std::vector<float> send_data(N, MPIEnvironment::world_rank + 1.0f);
  std::vector<float> recv_data(N);

  float *d_send = nullptr, *d_recv = nullptr;
  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_send, N * sizeof(float)));
  auto send_guard = makeDeviceBufferAutoGuard(d_send);

  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_recv, N * sizeof(float)));
  auto recv_guard = makeDeviceBufferAutoGuard(d_recv);

  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(d_send, send_data.data(), N * sizeof(float),
                                       hipMemcpyHostToDevice));

  // AllReduce: Sum across all ranks
  RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(d_send, d_recv, N, ncclFloat, ncclSum,
                                            getActiveCommunicator(), getActiveStream()));

  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(recv_data.data(), d_recv, N * sizeof(float),
                                       hipMemcpyDeviceToHost));
  HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(getActiveStream()));

  // Verify: sum of (1 + 2 + 3 + ... + world_size)
  float expected = (MPIEnvironment::world_size *
                    (MPIEnvironment::world_size + 1)) / 2.0f;

  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(recv_data[i], expected);
  }

  // Automatic cleanup via RAII guards
}
```

### Example 2: Broadcast Test

```cpp
TEST_F(UnifiedMPITest, Broadcast) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1000;
  std::vector<float> data(N);

  // Only rank 0 initializes data
  if (MPIEnvironment::world_rank == 0) {
    std::iota(data.begin(), data.end(), 1.0f);  // 1, 2, 3, ..., N
  }

  float *d_data = nullptr;
  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_data, N * sizeof(float)));
  auto data_guard = makeDeviceBufferAutoGuard(d_data);

  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(d_data, data.data(), N * sizeof(float),
                                       hipMemcpyHostToDevice));

  // Broadcast from rank 0 to all ranks
  RCCL_TEST_CHECK_GTEST_FAIL(ncclBroadcast(d_data, d_data, N, ncclFloat, 0,
                                            getActiveCommunicator(), getActiveStream()));

  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(data.data(), d_data, N * sizeof(float),
                                       hipMemcpyDeviceToHost));
  HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(getActiveStream()));

  // All ranks should have the same data now
  for (int i = 0; i < N; i++) {
    EXPECT_FLOAT_EQ(data[i], i + 1.0f);
  }

  // Automatic cleanup via RAII guards
}
```

### Example 3: Send/Recv Between Ranks

```cpp
TEST_F(UnifiedMPITest, SimpleSendRecv) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 1024;
  const int peer_rank = 1 - MPIEnvironment::world_rank;  // 0↔1

  float* d_send = nullptr;
  float* d_recv = nullptr;
  std::vector<float> h_send(N);
  std::vector<float> h_recv(N);

  // Each rank sends unique data
  for (int i = 0; i < N; i++) {
    h_send[i] = MPIEnvironment::world_rank * 1000 + i;
  }

  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_send, N * sizeof(float)));
  auto send_guard = makeDeviceBufferAutoGuard(d_send);

  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_recv, N * sizeof(float)));
  auto recv_guard = makeDeviceBufferAutoGuard(d_recv);

  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(d_send, h_send.data(), N * sizeof(float),
                                       hipMemcpyHostToDevice));

  // Exchange data between ranks
  if (MPIEnvironment::world_rank == 0) {
    RCCL_TEST_CHECK_GTEST_FAIL(ncclSend(d_send, N, ncclFloat, 1,
                                         getActiveCommunicator(), getActiveStream()));
    RCCL_TEST_CHECK_GTEST_FAIL(ncclRecv(d_recv, N, ncclFloat, 1,
                                         getActiveCommunicator(), getActiveStream()));
  } else {
    RCCL_TEST_CHECK_GTEST_FAIL(ncclRecv(d_recv, N, ncclFloat, 0,
                                         getActiveCommunicator(), getActiveStream()));
    RCCL_TEST_CHECK_GTEST_FAIL(ncclSend(d_send, N, ncclFloat, 0,
                                         getActiveCommunicator(), getActiveStream()));
  }

  HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(getActiveStream()));
  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(h_recv.data(), d_recv, N * sizeof(float),
                                       hipMemcpyDeviceToHost));

  // Verify received peer's data
  for (int i = 0; i < N; i++) {
    float expected = peer_rank * 1000 + i;
    EXPECT_FLOAT_EQ(h_recv[i], expected);
  }

  // Automatic cleanup via RAII guards
}
```

### Example 4: Testing Different Reduction Operations

```cpp
TEST_F(UnifiedMPITest, AllReduceMaxOperation) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const int N = 512;
  float* d_send = nullptr;
  float* d_recv = nullptr;
  std::vector<float> h_send(N);
  std::vector<float> h_recv(N);

  // Each rank sends rank*10 + index
  for (int i = 0; i < N; i++) {
    h_send[i] = MPIEnvironment::world_rank * 10 + i;
  }

  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_send, N * sizeof(float)));
  auto send_guard = makeDeviceBufferAutoGuard(d_send);

  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&d_recv, N * sizeof(float)));
  auto recv_guard = makeDeviceBufferAutoGuard(d_recv);

  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(d_send, h_send.data(), N * sizeof(float),
                                       hipMemcpyHostToDevice));

  // AllReduce with MAX operation
  RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(d_send, d_recv, N, ncclFloat, ncclMax,
                                            getActiveCommunicator(), getActiveStream()));

  HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(h_recv.data(), d_recv, N * sizeof(float),
                                       hipMemcpyDeviceToHost));
  HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(getActiveStream()));

  // Maximum should be from highest rank
  for (int i = 0; i < N; i++) {
    float expected = (MPIEnvironment::world_size - 1) * 10 + i;
    EXPECT_FLOAT_EQ(h_recv[i], expected);
  }

  // Automatic cleanup via RAII guards
}
```

### Example 5: Power-of-Two Requirement

```cpp
TEST_F(MyMPITest, AdvancedAlgorithm) {
  // This algorithm requires power-of-two processes (at least 4)
  ASSERT_TRUE(validateTestPrerequisites(4, kNoProcessLimit, kRequirePowerOfTwo));

  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Test only runs with 4, 8, 16, 32, ... processes
  // Automatically skipped if run with 3, 5, 6, 7, ... processes

  // Your test logic here
}
```

### Example 6: Single-Node Only Test

Tests that require all processes on the same physical node should use `kRequireSingleNode`:

```cpp
TEST_F(P2pMPITest, P2pWorkflow) {
  // This test requires single-node execution
  // Common reasons: direct GPU access, shared memory, local IPC, hardware topology
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode));

  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Test runs on single node:
  // Test skips on multi-node with informative message:
  //   "Error: REQUIREMENT NOT MET: Need at most 1 node(s), detected 2 nodes"
  //   "This test uses P2P/SHM transport (single-node only)"
  //   "For multi-node testing, use NET transport tests"

  // Your single-node test logic here...
  // Examples: P2P transport, SHM transport, GPU topology tests,
  //           shared memory algorithms, local IPC features
}
```

### Example 7: Multi-Node Capable Test

Tests that work across multiple nodes should use `kNoNodeLimit` (or omit the parameter):

```cpp
TEST_F(NetMPITest, NetWorkflow) {
  // This test works on any number of nodes
  // Common reasons: network-based, distributed features, scalability tests
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  // Uses defaults: no max processes, not power-of-two, any nodes
  // (kNoNodeLimit is the default for max_nodes)

  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Test runs on single node:
  // Test runs on multi-node:
  // Works with any node configuration

  // Your multi-node test logic here...
  // Examples: NET transport, distributed collectives, RDMA features,
  //           scalability tests, network-based algorithms
}
```

### Example 8: Custom Test Class with RAII Resource Guards

```cpp
class MyTransportTest : public TransportTestBase {
protected:
  void* send_buffer = nullptr;
  void* recv_buffer = nullptr;
  size_t buffer_size = 1024 * 1024;  // 1MB

  void SetUp() override {
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
    ASSERT_EQ(ncclSuccess, createTestCommunicator());

    // Allocate buffers with automatic RAII guards
    // Guards stored in base class, cleanup automatic at test end
    allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, buffer_size, buffer_size);
  }

  // No need for manual cleanup in TearDown()
  // Base class automatically cleans up guarded resources

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

  // Resources automatically cleaned up at test end
}
```

### Example 9: Loop with Per-Iteration Cleanup

For tests that allocate resources in loops, use `store_in_base=false` to get local guards:

```cpp
TEST_F(MyTransportTest, TestMultipleSizes) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  const std::vector<size_t> test_sizes = {1024, 4096, 16384, 65536};

  for(const auto size : test_sizes) {
    void* send_buff = nullptr;
    void* recv_buff = nullptr;

    // Get local guards - cleanup at end of each iteration
    auto [sendGuard, recvGuard] = allocateAndInitBuffersGuarded(
        &send_buff, &recv_buff, size, size, false);  // false = local guards

    // Test with this buffer size
    testTransfer(send_buff, recv_buff, size);

    // Buffers automatically freed here at end of iteration
  }

  // All buffers already cleaned up, minimal memory footprint maintained
}

---

## Best Practices

### 1. Use RAII Guards for Automatic Resource Cleanup

**TransportTestBase** provides RAII-based resource management to prevent leaks:

```cpp
// ✅ GOOD: Use guarded allocation (default - cleanup at test end)
TEST_F(MyTransportTest, Example) {
  void* send_buffer = nullptr;
  void* recv_buffer = nullptr;

  // Allocate with automatic guards
  allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, size, size);

  // Use buffers...

  // Automatic cleanup even if assertions fail!
}

// ✅ GOOD: Loop with per-iteration cleanup
TEST_F(MyTransportTest, LoopExample) {
  for(const auto size : test_sizes) {
    void* send_buff = nullptr;
    void* recv_buff = nullptr;

    // Local guards - cleanup per iteration
    auto [sg, rg] = allocateAndInitBuffersGuarded(&send_buff, &recv_buff, size, size, false);

    // Test logic...
    // Cleanup happens here automatically
  }
}

// ❌ BAD: Manual cleanup can leak on assertion failure
TEST_F(MyTransportTest, Example) {
  void* send_buffer = nullptr;
  hipMalloc(&send_buffer, size);

  ASSERT_TRUE(condition);  // If this fails, send_buffer leaks!

  hipFree(send_buffer);  // Never reached if assertion fails
}
```

**RAII Guard Benefits:**
- ✅ Resources cleaned up even if `ASSERT_*` or `EXPECT_*` fails
- ✅ Exception-safe cleanup
- ✅ No manual cleanup code needed
- ✅ Prevents memory leaks in test failures
- ✅ Supports both test-scoped and loop-scoped cleanup

**API:**
```cpp
// Store guards in base class (cleanup at test end) - DEFAULT
allocateAndInitBuffersGuarded(&send, &recv, size, size);

// Get local guards (cleanup at scope exit) - FOR LOOPS
auto [sendGuard, recvGuard] = allocateAndInitBuffersGuarded(&send, &recv, size, size, false);

// Registration with guards
preRegisterBuffersGuarded(send, recv, size, size, &send_handle, &recv_handle);
auto [sendRegGuard, recvRegGuard] = preRegisterBuffersGuarded(..., false);
```

### 2. Always Validate Prerequisites

```cpp
TEST_F(MyTest, SomeTest) {
  // ✅ GOOD: Validate first
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Test logic...
}

// ❌ BAD: No validation
TEST_F(MyTest, SomeTest) {
  // Might crash if only 1 process!
  ncclAllReduce(...);
}
```

### 3. Create Test-Specific Communicators

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

### 4. Use RAII Guards for Resource Management

**TransportTestBase** provides RAII-based automatic resource cleanup with proper ordering:

```cpp
// ✅ GOOD: Automatic cleanup even on test failure
TEST_F(MyTransportTest, Example) {
  void* send_buffer = nullptr;
  void* recv_buffer = nullptr;

  // Allocate with automatic guards (default: cleanup at test end)
  allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, size, size);

  ASSERT_TRUE(condition);  // Even if this fails, buffers cleaned up!

  // Use buffers...
  // Automatic cleanup at test end
}

// ✅ GOOD: Registration handles with guards
TEST_F(MyTransportTest, RegistrationExample) {
  void* send_buffer = nullptr;
  void* recv_buffer = nullptr;
  allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, size, size);

  // Pre-register with guards - handles deregistered before comm destroyed
  void* send_handle = nullptr;
  void* recv_handle = nullptr;
  preRegisterBuffersGuarded(send_buffer, recv_buffer, size, size,
                           &send_handle, &recv_handle);

  // Use registered buffers...
  // Cleanup order: handles deregistered → buffers freed → comm destroyed
}

// ✅ GOOD: Loop with per-iteration cleanup
TEST_F(MyTransportTest, LoopTest) {
  for(const auto size : test_sizes) {
    void* send_buff = nullptr;
    void* recv_buff = nullptr;

    // Get local guards (store_in_base=false for per-iteration cleanup)
    auto [sendGuard, recvGuard] = allocateAndInitBuffersGuarded(
        &send_buff, &recv_buff, size, size, false);

    // Test logic...
    // Cleanup happens here automatically at end of iteration
  }
}

// ❌ BAD: Manual cleanup leaks on assertion failure
TEST_F(MyTest, Example) {
  void* buffer = nullptr;
  hipMalloc(&buffer, size);

  ASSERT_TRUE(condition);  // If fails, buffer leaks!

  hipFree(buffer);  // Never reached
}
```

**RAII Guard API:**
```cpp
// Allocate with guards (cleanup at test end) - DEFAULT
allocateAndInitBuffersGuarded(&send, &recv, size, size);

// Allocate with local guards (cleanup at scope exit) - FOR LOOPS
auto [sendGuard, recvGuard] = allocateAndInitBuffersGuarded(&send, &recv, size, size, false);

// Register buffers with guards (cleanup at test end) - DEFAULT
preRegisterBuffersGuarded(send, recv, size, size, &send_handle, &recv_handle);

// Register with local guards (for loops)
auto [sRegGuard, rRegGuard] = preRegisterBuffersGuarded(send, recv, size, size,
                                                         &send_handle, &recv_handle, false);
```

**Benefits:**
- ✅ Resources cleaned up even if `ASSERT_*` or `EXPECT_*` fails
- ✅ **Correct cleanup order**: Guards destroyed before communicator
- ✅ Exception-safe cleanup
- ✅ No manual cleanup code needed
- ✅ Prevents memory leaks in test failures
- ✅ Prevents "corrupted comm object" errors
- ✅ Supports both test-scoped and loop-scoped cleanup

**Critical: Cleanup Order**
The framework ensures proper cleanup order to prevent "corrupted comm object" errors:
```
1. Registration handles deregistered (ncclCommDeregister with valid comm)
2. Buffers freed (hipFree)
3. Transport resources cleaned up
4. Communicator destroyed (ncclCommDestroy)
```

This is handled automatically by `TransportTestBase::TearDown()` which explicitly clears
guard vectors before destroying the communicator.

### 5. Use Rank-Specific Logic When Needed

```cpp
TEST_F(MyTest, Example) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  int rank = MPIEnvironment::world_rank;

  if (rank == 0) {
    // Only rank 0 prints summary (TEST_INFO automatically adds rank prefix)
    TEST_INFO("Starting test with %d processes", MPIEnvironment::world_size);
  }

  // All ranks execute this
  performCollectiveOperation();

  if (rank == 0) {
    // Only rank 0 does final verification
    verifyGlobalState();
    TEST_INFO("Test completed successfully");
  }
}
```

### 6. Use Descriptive Test Names

```cpp
// ✅ GOOD: Clear what's being tested
TEST_F(MyMPITest, AllReduce_WithFloat32_Sum_2Ranks)
TEST_F(MyMPITest, Broadcast_LargeBuffer_FromRank0)
TEST_F(MyMPITest, SendRecv_PeerToPeer_1MBTransfer)

// ❌ BAD: Vague names
TEST_F(MyMPITest, Test1)
TEST_F(MyMPITest, TestAllReduce)
```

### 7. Check Return Codes

```cpp
// ✅ GOOD: Check all return codes with appropriate macros
ASSERT_EQ(ncclSuccess, createTestCommunicator());
RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(...));
HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&ptr, size));

// BAD: Ignoring return values
createTestCommunicator();  // Might fail silently!
ncclAllReduce(...);         // Could return error
hipMalloc(&ptr, size);      // Allocation might fail
```

### 8. Synchronize Appropriately

```cpp
// ✅ GOOD: Synchronize before checking results
RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(...));
HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(getActiveStream()));
// Now safe to verify results

// ❌ BAD: Check results without sync
RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(...));
EXPECT_EQ(result, expected);  // Operation might not be done!
```

### 9. Consider Process Count in Expectations

```cpp
TEST_F(MyTest, AllReduceSum) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  // Each rank sends 1
  // Expected result depends on world_size
  float expected = MPIEnvironment::world_size * 1.0f;

  // ✅ GOOD: Expectation adapts to process count
  EXPECT_FLOAT_EQ(result, expected);

  // ❌ BAD: Hard-coded expectation
  // EXPECT_FLOAT_EQ(result, 2.0f);  // Only works with 2 processes!
}
```

---

## RAII Resource Guards

The test infrastructure provides comprehensive RAII (Resource Acquisition Is Initialization) guards for automatic resource cleanup. These guards ensure resources are cleaned up even when tests fail via `ASSERT_*` or `EXPECT_*` failures.

### Overview

Two types of guards are provided:

1. **`AutoGuard<T, auto DeleterFunc>`** - Modern C++17 guard using function pointers (for simple cleanup)
2. **`ResourceGuard<T, Deleter>`** - Functor-based guard (for complex stateful cleanup)

### AutoGuard - Simple Cleanup

For resources with simple, stateless cleanup functions:

```cpp
// Device memory
void* buffer;
hipMalloc(&buffer, size);
auto guard = makeDeviceBufferAutoGuard(buffer);
// buffer automatically freed on scope exit

// HIP stream
hipStream_t stream;
hipStreamCreate(&stream);
auto guard = makeStreamAutoGuard(stream);
// stream automatically destroyed on scope exit

// HIP event
hipEvent_t event;
hipEventCreate(&event);
auto guard = makeEventAutoGuard(event);
// event automatically destroyed on scope exit

// Host memory
void* host_buf = malloc(size);
auto guard = makeHostBufferAutoGuard(host_buf);
// host_buf automatically freed on scope exit

// NCCL communicator
ncclComm_t comm;
ncclCommInitRank(&comm, ...);
auto guard = makeCommAutoGuard(comm);
// comm automatically destroyed on scope exit
```

### ResourceGuard - Complex Cleanup

For resources requiring additional context (stateful deleters):

```cpp
// NCCL registration handle (needs communicator)
void* reg_handle;
ncclCommRegister(comm, buffer, size, &reg_handle);
auto guard = makeRegHandleGuard(reg_handle, comm);
// reg_handle automatically deregistered on scope exit

// NET plugin memory handle (needs net plugin + comm)
void* mhandle;
net->regMr(comm, buffer, size, type, &mhandle);
auto guard = makeNetMHandleGuard(mhandle, net, comm);
// mhandle automatically deregistered on scope exit

// NET send communicator (needs net plugin)
void* send_comm;
net->connect(dev, handle, &send_comm, &send_dev_handle);
auto guard = makeNetSendCommGuard(send_comm, net);
// send_comm automatically closed on scope exit
```

### Guard Operations

All guards support these operations:

```cpp
auto guard = makeDeviceBufferAutoGuard(buffer);

// Get the resource handle
void* buf = guard.get();

// Get pointer to handle (for API calls that take T*)
void** buf_ptr = guard.ptr();

// Set a new resource
guard.set(new_buffer);

// Release ownership (prevent cleanup)
void* released = guard.release();

// Dismiss without returning (prevent cleanup)
guard.dismiss();
```

### Specialized Guards

#### BufferGuard - Host or Device Memory

Manages both host and device memory with runtime discrimination:

```cpp
void* device_buf;
hipMalloc(&device_buf, size);
BufferGuard dev_guard(device_buf, false);  // false = device memory

void* host_buf = malloc(size);
BufferGuard host_guard(host_buf, true);  // true = host memory

// Both automatically freed on scope exit with correct function
```

#### NetConnectionGuard - Multiple Network Resources

Manages listen, send, and recv communicators together:

```cpp
NetConnectionGuard conn_guard(net_plugin);

// Set resources as they're created
conn_guard.setListenComm(listen_comm);
conn_guard.setSendComm(send_comm);
conn_guard.setRecvComm(recv_comm);

// All automatically closed on scope exit in correct order
```

#### TransportResourceGuard - Send and Recv Together

Manages paired send/recv transport resources:

```cpp
ncclConnector send_conn, recv_conn;
TransportResourceGuard guard(&send_conn, &recv_conn, transport);

// Both connectors automatically cleaned up on scope exit
```

### Factory Methods

Prefer factory methods for type deduction and cleaner syntax:

```cpp
// ✅ GOOD: Factory method (type deduced)
auto guard = makeDeviceBufferAutoGuard(buffer);

// ❌ VERBOSE: Explicit type (harder to read)
AutoGuard<void*, hipFreeWrapper> guard(buffer);

// ✅ GOOD: Complex guard with factory
auto guard = makeRegHandleGuard(handle, comm);

// ❌ VERBOSE: Explicit type
ResourceGuard<void*, NcclRegHandleDeleter> guard(handle, NcclRegHandleDeleter(comm));
```

### Available Factory Methods

**Simple Resources (AutoGuard):**
- `makeHostBufferAutoGuard(void* buffer)` - Host memory
- `makeDeviceBufferAutoGuard(void* buffer)` - Device memory
- `makeStreamAutoGuard(hipStream_t stream)` - HIP stream
- `makeEventAutoGuard(hipEvent_t event)` - HIP event
- `makeCommAutoGuard(ncclComm_t comm)` - NCCL communicator

**Complex Resources (ResourceGuard):**
- `makeRegHandleGuard(void* handle, ncclComm_t comm)` - NCCL registration
- `makeNetMHandleGuard(void* handle, ncclNet_t* net, void* comm)` - NET memory
- `makeNetSendCommGuard(void* comm, ncclNet_t* net)` - NET send comm
- `makeNetRecvCommGuard(void* comm, ncclNet_t* net)` - NET recv comm
- `makeNetListenCommGuard(void* comm, ncclNet_t* net)` - NET listen comm
- `makeTransportSendGuard(ncclConnector* conn, ncclTransport* trans)` - Transport send
- `makeTransportRecvGuard(ncclConnector* conn, ncclTransport* trans)` - Transport recv

**Generic:**
- `makeGuard(T resource, Deleter deleter)` - Generic ResourceGuard
- `makeCustomGuard(T resource, Deleter deleter)` - Custom deleter (alias)


### Best Practices with Guards

**1. Use Guards for All Resources:**
```cpp
// ✅ GOOD: Guards ensure cleanup even on assertion failure
TEST_F(MyTest, Example) {
  void* buffer;
  hipMalloc(&buffer, size);
  auto guard = makeDeviceBufferAutoGuard(buffer);

  ASSERT_TRUE(condition);  // If fails, buffer still freed!

  // Use buffer...
  // Automatic cleanup on scope exit
}

// ❌ BAD: Manual cleanup leaks on assertion failure
TEST_F(MyTest, Example) {
  void* buffer;
  hipMalloc(&buffer, size);

  ASSERT_TRUE(condition);  // If fails, buffer leaks!

  hipFree(buffer);  // Never reached
}
```

**2. Respect Cleanup Order:**
```cpp
// ✅ GOOD: Correct cleanup order (handles before comm)
TEST_F(MyTest, Example) {
  ncclComm_t comm;
  ncclCommInitRank(&comm, ...);
  auto comm_guard = makeCommAutoGuard(comm);

  void* reg_handle;
  ncclCommRegister(comm, buffer, size, &reg_handle);
  auto handle_guard = makeRegHandleGuard(reg_handle, comm);

  // Cleanup order: handle_guard destroyed first (correct!)
  // Then comm_guard destroyed
}

// ❌ BAD: Wrong order causes "corrupted comm object" error
TEST_F(MyTest, Example) {
  void* reg_handle;
  ncclCommRegister(comm, buffer, size, &reg_handle);
  auto handle_guard = makeRegHandleGuard(reg_handle, comm);

  ncclComm_t comm;
  ncclCommInitRank(&comm, ...);
  auto comm_guard = makeCommAutoGuard(comm);

  // Cleanup order: comm_guard destroyed first - ERROR!
  // handle_guard tries to use destroyed comm
}
```

**3. Use Local Guards for Loops:**
```cpp
// ✅ GOOD: Local guards for per-iteration cleanup
for (const auto size : test_sizes) {
  void* buffer;
  hipMalloc(&buffer, size);
  auto guard = makeDeviceBufferAutoGuard(buffer);

  // Test with this size...

  // Buffer freed here at end of iteration
}

// ❌ BAD: Accumulating allocations
std::vector<void*> buffers;
for (const auto size : test_sizes) {
  void* buffer;
  hipMalloc(&buffer, size);
  buffers.push_back(buffer);  // Memory accumulates!
}
// All buffers freed only at end - high memory usage
```

**4. Use Custom Guards for Lambdas:**
```cpp
// Complex cleanup with lambda
FILE* file = fopen("test.txt", "w");
auto guard = makeCustomGuard(file, [](FILE* f) {
  if (f) {
    fflush(f);
    fclose(f);
  }
});
// file automatically flushed and closed on scope exit
```

### Implementation Details

**AutoGuard:**
- Uses C++17 `auto` non-type template parameters
- Zero overhead - deleter is a compile-time constant
- No functor object stored - just the resource handle
- Smaller memory footprint than ResourceGuard

**ResourceGuard:**
- Stores both resource and deleter functor
- Supports stateful deleters with additional context
- Move-only semantics (non-copyable)
- Slightly larger memory footprint due to deleter storage

**When to Use Which:**
- Use **AutoGuard** (via factory methods like `makeDeviceBufferAutoGuard`) for simple cleanup
- Use **ResourceGuard** (via factory methods like `makeRegHandleGuard`) for cleanup requiring context

### See Also

- **ResourceGuards.hpp** - Full guard implementation (includes ScopeGuard, AutoGuard, ResourceGuard)
- **TransportMPIBase.hpp** - Transport test base with guarded resource management
- **Best Practices** section above for RAII usage patterns

---

## Troubleshooting

### MPI Initialization Failed

**Symptom:** `MPIEnvironment::retCode != 0` or "Only X GPUs available for Y ranks"

**Causes:**
- Insufficient GPUs for the number of **local** processes (per node)
- MPI not properly installed
- Wrong MPI launcher configuration

**Solutions:**
```cpp
// Check in test
if (MPIEnvironment::retCode != 0) {
  GTEST_SKIP() << "MPI initialization failed";
}

// Or use validateTestPrerequisites which checks this
ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
```

```bash
# For single-node: GPU count must match process count
mpirun -np 8 -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./test_executable

# For multi-node: GPU count per node must match processes per node
# Example: 16 ranks on 2 nodes = 8 ranks/node, need 8 GPUs/node
salloc -N 2 --gres=gpu:8 --ntasks-per-node=8
mpirun -np 16 ./test_executable

# Check GPU availability on each node
rocm-smi --showid
```

**Multi-Node GPU Assignment:**
The framework now uses **local rank** (rank within node) for GPU assignment, not global rank. This means:
- Node 1: Ranks 0-7 use GPUs 0-7
- Node 2: Ranks 8-15 use GPUs 0-7
- Each node independently assigns its local ranks to its local GPUs

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
RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(...));
```

2. **Missing Synchronization:**
```cpp
// ❌ BAD: Rank 0 waits for data other ranks haven't sent
ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));  // All ranks must reach this

// ✅ GOOD: createTestCommunicator() includes barriers
ASSERT_EQ(ncclSuccess, createTestCommunicator());
```

3. **Stream Not Synchronized:**
```cpp
// ✅ GOOD: Wait for operations to complete
HIP_TEST_CHECK_GTEST_FAIL(hipStreamSynchronize(getActiveStream()));
ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));
```

**See:** [Per-Rank Logging](#per-rank-logging) for detailed debugging techniques

### "More Processes Than CPUs" Error

**Symptom:**
```
A request was made to bind to that would result in binding more
processes than cpus available in your allocation:
   #processes:      16
   Binding policy:  CORE
```

**Cause:** OpenMPI trying to bind each process to a dedicated CPU core, but insufficient cores.

**Solution:** The test runner automatically uses `--bind-to none` to disable CPU binding.

```bash
# Automatic (if using test runner script)
./build_test_coverage.std.sh --config test_config.txt

# Manual fix: add --bind-to none
mpirun -np 16 --bind-to none ./test_executable
```

**Why this works:** For GPU tests, CPU binding is not critical since GPUs do the computation. Disabling binding allows any number of processes regardless of CPU count.

### Hostfile Issues / "Host Not Found"

**Symptom:**
```
Missing requested host: node-name
At least one of the requested hosts is not included in the current allocation
```

**Cause:** Hostfile doesn't match allocated nodes or has wrong format.

**Solution:** The test runner automatically generates hostfiles from SLURM allocations.

```bash
# Automatic (if using test runner with SLURM)
salloc -N 2 -n 16
./build_test_coverage.std.sh --config test_config.txt --no-build
# Creates temporary hostfile:
#   node-3 slots=8
#   node-21 slots=8

# Manual hostfile creation
cat > hostfile.txt << EOF
node-3 slots=8
node-21 slots=8
EOF
mpirun -np 16 --hostfile hostfile.txt ./test_executable
```

**Hostfile format:**
- `hostname slots=N` where N is max processes per node
- Script auto-calculates: `slots = total_ranks / num_nodes`

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
3. Use RAII guards to prevent leaks on test failure
4. Run tests sequentially instead of in parallel

```cpp
// ✅ GOOD: Use RAII guards (automatic cleanup even on failure)
TEST_F(MyTest, Example) {
  void* buffer = nullptr;
  allocateAndInitBuffersGuarded(&buffer, nullptr, size, 0);
  // Automatic cleanup even if test fails
}

// ❌ BAD: Manual cleanup in TearDown (leaks if test fails)
void TearDown() override {
  if (buffer) {
    hipFree(buffer);
    buffer = nullptr;
  }
  MPITestBase::TearDown();
}
```

### Corrupted Comm Object Error

**Symptom:** Test passes but logs "corrupted comm object detected" errors during cleanup:
```
NCCL WARN Error: corrupted comm object detected
/home/.../src/register/register.cc:171 -> 4
```

**Cause:** Registration handles being deregistered after communicator was destroyed.

**Solution:** Use `preRegisterBuffersGuarded()` instead of manual registration:

```cpp
// ✅ GOOD: Guards ensure proper cleanup order
TEST_F(MyTransportTest, Example) {
  void* send_buffer = nullptr;
  void* recv_buffer = nullptr;
  allocateAndInitBuffersGuarded(&send_buffer, &recv_buffer, size, size);

  void* send_handle = nullptr;
  void* recv_handle = nullptr;
  preRegisterBuffersGuarded(send_buffer, recv_buffer, size, size,
                           &send_handle, &recv_handle);

  // Guards automatically deregister handles BEFORE comm is destroyed
}

// ❌ BAD: Manual deregistration may happen after comm destroyed
TEST_F(MyTest, Example) {
  void* handle = nullptr;
  ncclCommRegister(comm, buffer, size, &handle);
  // ... test logic ...
  ncclCommDeregister(comm, handle);  // May fail if comm destroyed first!
}
```

**Why it works:** `TransportTestBase::TearDown()` explicitly clears guard vectors before
destroying the communicator, ensuring handles are deregistered while the communicator is
still valid.

### Test Skipped: Not Enough Processes

**Symptom:** Test skipped with message about process count.

**Solution:** Run with more processes:
```bash
# Test requires 4 processes
mpirun -np 4 ./test_executable --gtest_filter="MyTest.SomeTest"

# Check what test needs
grep validateTestPrerequisites test_file.cpp
```

### Test Skipped: Node Requirement Not Met

**Symptom:** Test skipped with message like:
```
Skipping test - requires at most 1 node(s), detected 2 nodes
This test requires single-node execution
To run on single node, allocate all processes on the same host
```

**Cause:** Test requires single-node execution but was run on multiple nodes.

**Solutions:**

**Option 1: Run on single node with multiple GPUs**
```bash
# Run 2 processes on single node
mpirun -np 2 ./test_executable --gtest_filter="SingleNodeTest.*"

# Or specify GPUs explicitly
mpirun -np 2 -x HIP_VISIBLE_DEVICES=0,1 ./test_executable
```

**Option 2: Use multi-node capable tests instead**
```bash
# Run tests that work across multiple nodes
mpirun -np 4 --host node1,node2 ./test_executable --gtest_filter="MultiNodeTest.*"
```

**Option 3: Check your hostfile/job allocation**
```bash
# Verify you're actually on different nodes
mpirun -np 2 hostname

# If both print same hostname, you're on one node
# If different hostnames, you're on multiple nodes
```

### Test Fails on Multi-Node But Works on Single-Node

**Symptom:**
- Test passes when all processes are on one node
- Test fails or crashes when processes are on different nodes
- May see errors like "invalid device pointer", "segmentation fault", or hangs

**Cause:** Test is using features that require single-node execution but lacks node validation.

**Common single-node features:**
- Direct GPU-to-GPU memory access (P2P transport)
- Shared memory between processes (SHM transport)
- Inter-process communication (IPC) handles
- Assumptions about memory locality or GPU topology

**Solution:**
```cpp
// ✅ GOOD: Add node validation
TEST_F(MyTest, LocalFeature) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode));
  // Test will skip gracefully on multi-node
}

// ✅ GOOD: Or redesign to use network-capable features
TEST_F(MyTest, DistributedFeature) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  // Uses defaults: any nodes allowed
  // Use NET transport or network-based communication
}
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
// Use TEST_INFO for debug output (respects NCCL_DEBUG=INFO)
TEST_INFO("result[0] = %f", result[0]);
ASSERT_MPI_SUCCESS(MPI_Barrier(MPI_COMM_WORLD));

// Verify all ranks agree
float local_result = result[0];
float global_result;
ASSERT_MPI_SUCCESS(MPI_Allreduce(&local_result, &global_result, 1, MPI_FLOAT,
                                  MPI_MAX, MPI_COMM_WORLD));
EXPECT_FLOAT_EQ(local_result, global_result);
```

**Debug with NCCL_DEBUG:**
```bash
# Enable detailed logging from tests and library
NCCL_DEBUG=INFO RCCL_MPI_LOG_ALL_RANKS=1 mpirun -np 4 ./test_executable

# Output shows rank/hostname automatically:
# [0] TEST INFO result[0] = 1.000000
# [1] TEST INFO result[0] = 2.000000
# [2] TEST INFO result[0] = 3.000000
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
    ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
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
  ASSERT_TRUE(validateTestPrerequisites(4));

  // Create two separate communicators
  ncclUniqueId id1, id2;
  ncclComm_t comm1 = nullptr, comm2 = nullptr;

  if (MPIEnvironment::world_rank == 0) {
    RCCL_TEST_CHECK_GTEST_FAIL(ncclGetUniqueId(&id1));
    RCCL_TEST_CHECK_GTEST_FAIL(ncclGetUniqueId(&id2));
  }

  ASSERT_MPI_SUCCESS(MPI_Bcast(&id1, sizeof(id1), MPI_BYTE, 0, MPI_COMM_WORLD));
  ASSERT_MPI_SUCCESS(MPI_Bcast(&id2, sizeof(id2), MPI_BYTE, 0, MPI_COMM_WORLD));

  RCCL_TEST_CHECK_GTEST_FAIL(ncclCommInitRank(&comm1, MPIEnvironment::world_size, id1,
                                               MPIEnvironment::world_rank));
  auto comm1_guard = makeCommAutoGuard(comm1);

  RCCL_TEST_CHECK_GTEST_FAIL(ncclCommInitRank(&comm2, MPIEnvironment::world_size, id2,
                                               MPIEnvironment::world_rank));
  auto comm2_guard = makeCommAutoGuard(comm2);

  // Use both communicators...

  // Automatic cleanup via RAII guards
}
```

### Testing Error Conditions

```cpp
TEST_F(MyTest, InvalidRankHandling) {
  ASSERT_TRUE(validateTestPrerequisites(kMinProcessesForMPI));
  ASSERT_EQ(ncclSuccess, createTestCommunicator());

  void* buffer = nullptr;
  HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buffer, 1024));
  auto buffer_guard = makeDeviceBufferAutoGuard(buffer);

  // Deliberately use invalid rank
  int invalid_rank = 999;

  // Expect error (don't crash)
  ncclResult_t result = ncclSend(buffer, 256, ncclFloat, invalid_rank,
                                  getActiveCommunicator(), getActiveStream());

  EXPECT_NE(result, ncclSuccess);
  // Test should continue, not deadlock
}
```

---

## FAQ

**Q: When should I use MPITestBase vs ProcessIsolatedTestRunner?**

A: Choose based on your testing requirements:

**Use MPITestBase when:**
- ✅ Testing **multi-process** RCCL operations (collectives, point-to-point)
- ✅ Testing **multi-node** distributed execution
- ✅ Validating communication between multiple GPUs/ranks
- ✅ Testing transport layers (P2P, SHM, NET)
- ✅ Verifying scalability across processes and nodes
- ✅ Testing MPI-based coordination and synchronization
- **Examples:** AllReduce, Broadcast, Send/Recv, multi-GPU collectives, cross-node communication

**Use ProcessIsolatedTestRunner when:**
- ✅ Testing **single-process** code with clean environment isolation
- ✅ Need to **set environment variables programmatically** for each test
- ✅ Testing environment-dependent behavior without affecting other tests
- ✅ Validating RCCL configuration with different environment settings
- ✅ Testing initialization/cleanup with isolated state
- ✅ Running tests that require specific environment variables
- **Examples:** Testing NCCL_DEBUG levels, NCCL tuning parameters, plugin loading, environment-specific initialization

**Key Differences:**

| Feature | MPITestBase | ProcessIsolatedTestRunner |
|---------|-------------|---------------------------|
| **Process Count** | Multiple (2+) | Single |
| **Node Support** | Single or multi-node | Single node only |
| **Environment Control** | Inherited from shell | Programmatic per test |
| **Use Case** | Multi-GPU/multi-node operations | Environment-dependent single-process tests |
| **Coordination** | MPI barriers and communication | None (isolated process) |

**Q: How many processes should I test with?**

A:
- Minimum: 2 (for basic collectives and P2P)
- Common: 2, 4, 8 (good coverage)
- For scalability: Test with various counts
- For algorithms: Some require power-of-two (use `kRequirePowerOfTwo`)

**Q: Does `validateTestPrerequisites()` work correctly across multiple nodes?**

A: Yes! It validates both process count AND node count:
- **Process count validation**: Checks total number of processes (any node distribution)
- **Node count validation**: Detects number of unique nodes via hostnames
- Tests can specify node requirements based on what they need

Example:
```cpp
// Test that works on any number of nodes (uses defaults)
validateTestPrerequisites(2);

// Test that requires single-node execution
validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 1, kRequireSingleNode);
```

**Q: How does node detection work?**

A: Node detection uses MPI to gather hostnames from all ranks:
1. Each rank gets its hostname via `MPI_Get_processor_name()`
2. All hostnames are gathered to rank 0
3. Rank 0 counts unique hostnames
4. Count is broadcast to all ranks

This automatically works with any job scheduler (SLURM, PBS, etc.) and requires no configuration.

**Q: When should I use `kRequireSingleNode` vs `kNoNodeLimit`?**

A: Based on what your test requires:

**Use `kRequireSingleNode` when your test:**
- Uses direct GPU-to-GPU memory access
- Requires shared memory between processes
- Uses inter-process communication (IPC) handles
- Makes assumptions about memory locality or GPU topology
- Uses features that only work within a single physical node
- Examples: P2P transport, SHM transport, GPU topology tests, local memory tests

**Use `kNoNodeLimit` (default) when your test:**
- Works with network-based communication
- Uses distributed features or algorithms
- Should work regardless of node configuration
- Tests scalability across multiple nodes
- Examples: NET transport, collective operations, RDMA features, scalability tests

**General guidance:**
- If your test uses intra-node features → use `kRequireSingleNode`
- If your test works across nodes → use `kNoNodeLimit` or omit (it's the default)
- If unsure, leave it as default (`kNoNodeLimit`) and add validation if you encounter multi-node issues

**Q: Can I run MPI tests locally?**

A: Yes, if you have:
- Multiple GPUs in your system
- MPI installed (OpenMPI, MPICH, etc.)
- Tests built with `MPI_TESTS_ENABLED`

**Q: How do I debug a specific rank?**

A:
```bash
# Method 1: Use per-rank logging with NCCL_DEBUG (easiest)
NCCL_DEBUG=INFO RCCL_MPI_LOG_ALL_RANKS=1 mpirun -np 4 ./test_executable
# Check rccl_test_rank_2.log for rank 2 output
# TEST_INFO messages will appear automatically

# Method 2: GDB with MPI
mpirun -np 2 xterm -e gdb ./test_executable

# Method 3: Attach to specific rank PID
mpirun -np 4 ./test_executable &
gdb -p <rank_pid>

# Method 4: Use TEST_INFO for conditional debugging
if (MPIEnvironment::world_rank == 2) {
  TEST_INFO("Debug info from rank 2...");
  // Only appears when NCCL_DEBUG=INFO
}
```

**Q: How do TEST_* macros work with NCCL_DEBUG?**

A: TEST_* macros respect the `NCCL_DEBUG` environment variable:
```bash
# No output from TEST_* macros (clean)
mpirun -np 2 ./test_executable

# TEST_INFO and higher appear
NCCL_DEBUG=INFO mpirun -np 2 ./test_executable

# All TEST_* macros appear
NCCL_DEBUG=TRACE mpirun -np 2 ./test_executable
```

**Available levels (least to most verbose):**
- `NCCL_DEBUG=WARN` → TEST_WARN
- `NCCL_DEBUG=INFO` → TEST_WARN, TEST_INFO (recommended for debugging)
- `NCCL_DEBUG=ABORT` → All above + TEST_ABORT
- `NCCL_DEBUG=TRACE` → All macros including TEST_TRACE

**Benefits:**
- ✅ Same verbosity control as RCCL library
- ✅ Automatic rank prefixes (no manual "Rank %d:")
- ✅ Hostname included for multi-node tests
- ✅ Clean output in production (no NCCL_DEBUG)
- ✅ Detailed debugging on demand (NCCL_DEBUG=INFO)

---

## Standalone Tests

The MPI test infrastructure now supports **framework-agnostic testing**, allowing you to write tests
without Google Test. This is ideal for:

- **Performance benchmarks** (bandwidth, latency)
- **Low-level API tests** without GTest overhead
- **Production utilities** using MPI infrastructure
- **Custom test harnesses**

### Quick Comparison

| Feature | GTest (MPITestBase) | Standalone (MPIStandaloneTest) |
|---------|--------------------|---------------------------------|
| Requires GTest | ✅ Yes | ❌ No |
| Assertions | ASSERT_*, EXPECT_* | Return codes |
| Setup/Teardown | Automatic | Manual `cleanup()` |
| Best For | Unit/integration tests | Performance benchmarks |
| Overhead | Higher | Minimal |

### Example: Standalone Test

```cpp
#include "MPIStandaloneTest.hpp"
#include "MPIHelpers.hpp"

class MyBenchmark : public MPIStandaloneTest {
public:
    int run() override {
        // Validate prerequisites
        if (!validateTestPrerequisites(2, 2)) return 0; // Skip

        // Create communicator
        if (createTestCommunicator() != ncclSuccess) return 1; // Error

        // Your benchmark code here...
        // Use getActiveCommunicator() and getActiveStream()

        return 0; // Success
    }
};

int main(int argc, char** argv) {
    // Initialize MPI and setup GPU
    auto mpi_ctx = MPIHelpers::initializeMPI(&argc, &argv);
    MPIHelpers::setupGPU(mpi_ctx.world_rank);

    // Run test with automatic cleanup via RAII
    int result = 0;
    {
        MyBenchmark test;
        MPIStandaloneTestRAII raii(&test);  // Automatic cleanup on scope exit
        result = test.run();
    }

    MPI_Finalize();
    return result;
}
```

---

## See Also

**Core Test Infrastructure:**
- **MPITestCore.hpp** - Framework-agnostic base class
- **MPITestBase.hpp** - Google Test adapter (full API documentation)
- **MPIStandaloneTest.hpp** - Standalone test adapter
- **MPIEnvironment.hpp** - MPI environment setup
- **MPIEnvironment.cpp** - Multi-node GPU assignment implementation

**Test Examples:**
- **transport/P2pMPITests.cpp** - P2P transport tests (demonstrate single-node validation)
- **transport/ShmMPITests.cpp** - Shared memory transport tests (demonstrate single-node validation)
- **transport/NetMPITests.cpp** - Network transport tests (demonstrate multi-node capable tests)
