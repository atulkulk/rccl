# Process Isolated Test Runner

A lightweight C++ testing framework for running Google Test cases in isolated processes with clean environment settings.

## Table of Contents
- [Overview](#overview)
- [Why Use Process Isolation?](#why-use-process-isolation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

`ProcessIsolatedTestRunner` is a framework that executes tests in separate processes using `fork()`. This ensures complete isolation between tests, particularly useful when testing code with static variables or environment-dependent behavior.

**Key Features:**
- ✅ Process-based test isolation (each test runs in its own process)
- ✅ Per-test environment variable management
- ✅ Configurable timeouts
- ✅ Sequential or stop-on-failure execution
- ✅ Thread-safe test registration
- ✅ Detailed test result reporting

**Location:** `test/common/ProcessIsolatedTestRunner.hpp`

---

## Why use Process Isolation?

### Problem: Static Variable Pollution

Consider this RCCL code with static variables:

```cpp
void rcclSetP2pNetChunkSize(struct ncclComm* comm, int& chunkSize) {
  static int p2pNetChunkSize = RCCL_VALUE_UNSET;  // ← Static variable!

  if (p2pNetChunkSize == RCCL_VALUE_UNSET) {
    const char* inputStr = getenv("NCCL_P2P_NET_CHUNKSIZE");
    if (!inputStr) {
      // Calculate value based on architecture...
      p2pNetChunkSize = calculateValue();
    }
  }
  chunkSize = p2pNetChunkSize;
}
```

**Without Process Isolation:**
```cpp
TEST(MyTest, FirstTest) {
  setenv("NCCL_P2P_NET_CHUNKSIZE", "12345", 1);
  // Static variable gets set to 12345
}

TEST(MyTest, SecondTest) {
  unsetenv("NCCL_P2P_NET_CHUNKSIZE");
  // ❌ Static variable STILL holds 12345 from previous test!
  // This test will fail or produce incorrect results
}
```

**With Process Isolation:**
```cpp
// Each test runs in a separate process
// Static variables are reset for each test
// ✅ Tests are truly independent
```

### Common Use Cases

1. **Testing environment variable behavior** - When code reads env vars into static variables
2. **Testing architecture-specific logic** - Different GPU architectures with cached state
3. **Testing initialization code** - One-time initialization patterns
4. **Testing configuration changes** - When config is cached statically

---

## Quick Start

### Basic Example

```cpp
#include "common/ProcessIsolatedTestRunner.hpp"

TEST(Rcclwrap, MyIsolatedTests) {
  // Clear any previous test registrations
  ProcessIsolatedTestRunner::clear();

  // Register a test with environment variables
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "TestWithCleanEnvironment",
          []() {
            // This runs in a separate process
            const char* value = getenv("MY_VARIABLE");
            EXPECT_STREQ(value, "test_value");

            // Your test logic here
            EXPECT_TRUE(someFunction());
          })
          .withEnvironment({{"MY_VARIABLE", "test_value"}})
  );

  // Execute all registered tests
  bool allTestsPassed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(allTestsPassed);
}
```

---

## Core Concepts

### 1. Test Configuration (`TestConfig`)

Defines how a test should be executed:

```cpp
TestConfig config(
    "TestName",           // Test name (for reporting)
    []() { /* logic */ }  // Test function (lambda or function pointer)
);

// Optional configurations
config.withEnvironment({{"VAR1", "value1"}, {"VAR2", "value2"}})
      .withTimeout(std::chrono::seconds(60))
      .withCleanEnvironment(false);  // Inherit parent environment
```

### 2. Test Registration

Tests must be registered before execution:

```cpp
// Method 1: Full configuration
ProcessIsolatedTestRunner::registerTest(config);

// Method 2: Simple (name + logic only)
ProcessIsolatedTestRunner::registerTest("SimplTest", []() {
  EXPECT_TRUE(true);
});

// Method 3: With environment
ProcessIsolatedTestRunner::registerTest(
    "EnvTest",
    []() { /* logic */ },
    {{"ENV_VAR", "value"}}
);
```

### 3. Test Execution

Execute all registered tests:

```cpp
// Default options (continue on failure, no verbose logging)
bool passed = ProcessIsolatedTestRunner::executeAllTests();

// Custom options
ProcessIsolatedTestRunner::ExecutionOptions options;
options.stopOnFirstFailure = true;   // Stop after first failure
options.verboseLogging = true;       // Print detailed logs

bool passed = ProcessIsolatedTestRunner::executeAllTests(options);
```

### 4. Test Results

Each test produces a `TestResult`:

```cpp
struct TestResult {
  std::string testName;               // Name of the test
  bool passed;                        // Whether the test passed
  bool skipped;                       // Whether the test was skipped
  int exitCode;                       // Process exit code
  pid_t processId;                    // Process ID that ran the test
  std::chrono::milliseconds duration; // Execution duration
  std::string errorMessage;           // Error message if failed
  std::unordered_map<std::string, std::string> environment;  // Env used
};
```

---

## API Reference

### Main Methods

#### `registerTest()`
Register a test for later execution.

```cpp
// Variant 1: Full configuration
static void registerTest(const TestConfig& config);

// Variant 2: Simple registration
static void registerTest(
    const std::string& name,
    std::function<void()> testLogic
);

// Variant 3: With environment
static void registerTest(
    const std::string& name,
    std::function<void()> testLogic,
    const std::unordered_map<std::string, std::string>& env
);
```

#### `executeAllTests()`
Execute all registered tests sequentially.

```cpp
static bool executeAllTests(
    const ExecutionOptions& options = ExecutionOptions()
);
```

**Returns:** `true` if all tests passed, `false` if any failed.

#### `getTestResults()`
Retrieve detailed results from the last execution.

```cpp
static std::vector<TestResult> getTestResults();
```

#### `clear()`
Clear all registered tests (call at the start of each TEST block).

```cpp
static void clear();
```

### TestConfig Methods

#### `withEnvironment()`
Set environment variables for the test.

```cpp
TestConfig& withEnvironment(
    const std::unordered_map<std::string, std::string>& env
);
```

**Note:** Variables are set in the child process only.

#### `withTimeout()`
Set a timeout for test execution.

```cpp
TestConfig& withTimeout(std::chrono::seconds timeoutSeconds);
```

**Default:** 30 seconds

#### `withCleanEnvironment()`
Control whether to inherit parent process environment.

```cpp
TestConfig& withCleanEnvironment(bool inherit = true);
```

**Default:** `true` (inherits parent environment)

---

## Examples

**Note:** The examples below use helper functions from `RcclWrapTests.cpp`:

```cpp
// Helper to create a mock NCCL communicator with specified architecture and ranks
static void CreateMockComm(ncclComm_t &mockComm,
                           struct ncclTopoSystem &mockTopo,
                           struct ncclTopoNode &mockGpuNode,
                           const char *arch,
                           int nRanks);

// Helper to cleanup a mock communicator
static void CleanupMockComm(ncclComm_t &mockComm);
```

### Example 1: Testing Environment Variable Behavior

```cpp
TEST(Rcclwrap, EnvironmentVariableTests) {
  ProcessIsolatedTestRunner::clear();

  // Test 1: With environment variable set
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "WithEnvVarSet",
          []() {
            ncclComm_t mockComm = nullptr;
            struct ncclTopoSystem mockTopo;
            struct ncclTopoNode mockGpuNode;
            CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 128);

            int chunkSize = RCCL_VALUE_UNSET;
            rcclSetP2pNetChunkSize(mockComm, chunkSize);

            // Should use default architecture-based value
            EXPECT_EQ(chunkSize, 1 << 19);

            CleanupMockComm(mockComm);
          })
          .withEnvironment({{"NCCL_P2P_NET_CHUNKSIZE", "999999"}})
  );

  // Test 2: Without environment variable (clean state)
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "WithoutEnvVar",
          []() {
            // Verify environment is clean
            const char* value = getenv("NCCL_P2P_NET_CHUNKSIZE");
            EXPECT_EQ(value, nullptr);

            // Test default behavior
            ncclComm_t mockComm = nullptr;
            struct ncclTopoSystem mockTopo;
            struct ncclTopoNode mockGpuNode;
            CreateMockComm(mockComm, mockTopo, mockGpuNode, "gfx942", 32);

            int chunkSize = RCCL_VALUE_UNSET;
            rcclSetP2pNetChunkSize(mockComm, chunkSize);
            EXPECT_EQ(chunkSize, 1 << 17);  // Default for < 64 ranks

            CleanupMockComm(mockComm);
          })
  );

  // Execute both tests in isolated processes
  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

### Example 2: Testing Multiple Architectures

```cpp
TEST(Rcclwrap, ArchitectureTests) {
  ProcessIsolatedTestRunner::clear();

  struct TestCase {
    std::string name;
    std::string arch;
    int ranks;
    int expectedChunkSize;
  };

  std::vector<TestCase> testCases = {
    {"GFX942_SmallRanks", "gfx942", 32, 1 << 17},
    {"GFX942_LargeRanks", "gfx942", 128, 1 << 19},
    {"GFX950_SmallRanks", "gfx950", 8, 1 << 17},
    {"GFX950_MediumRanks", "gfx950", 24, 1 << 18},
    {"GFX950_LargeRanks", "gfx950", 64, 1 << 19},
  };

  for (const auto& tc : testCases) {
    ProcessIsolatedTestRunner::registerTest(
        ProcessIsolatedTestRunner::TestConfig(
            tc.name,
            [tc]() {
              ncclComm_t mockComm = nullptr;
              struct ncclTopoSystem mockTopo;
              struct ncclTopoNode mockGpuNode;
              CreateMockComm(mockComm, mockTopo, mockGpuNode, tc.arch.c_str(), tc.ranks);

              int chunkSize = RCCL_VALUE_UNSET;
              rcclSetP2pNetChunkSize(mockComm, chunkSize);

              EXPECT_EQ(chunkSize, tc.expectedChunkSize)
                  << "Failed for " << tc.arch << " with " << tc.ranks << " ranks";

              CleanupMockComm(mockComm);
            })
    );
  }

  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.verboseLogging = true;
  options.stopOnFirstFailure = false;  // Run all tests even if one fails

  bool passed = ProcessIsolatedTestRunner::executeAllTests(options);
  EXPECT_TRUE(passed);
}
```

### Example 3: Testing with Timeouts

```cpp
TEST(Rcclwrap, TimeoutHandling) {
  ProcessIsolatedTestRunner::clear();

  // Test that completes quickly
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "FastTest",
          []() {
            EXPECT_TRUE(true);
          })
          .withTimeout(std::chrono::seconds(5))
  );

  // Test with longer timeout for complex operations
  ProcessIsolatedTestRunner::registerTest(
      ProcessIsolatedTestRunner::TestConfig(
          "SlowTest",
          []() {
            // Simulate slow operation
            std::this_thread::sleep_for(std::chrono::seconds(2));
            EXPECT_TRUE(true);
          })
          .withTimeout(std::chrono::seconds(10))
  );

  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

### Example 4: Stop on First Failure

```cpp
TEST(Rcclwrap, CriticalTests) {
  ProcessIsolatedTestRunner::clear();

  // Register multiple critical tests
  ProcessIsolatedTestRunner::registerTest(
      "CriticalTest1", []() { EXPECT_TRUE(checkCriticalCondition1()); });

  ProcessIsolatedTestRunner::registerTest(
      "CriticalTest2", []() { EXPECT_TRUE(checkCriticalCondition2()); });

  ProcessIsolatedTestRunner::registerTest(
      "CriticalTest3", []() { EXPECT_TRUE(checkCriticalCondition3()); });

  // Stop on first failure - don't waste time if critical tests fail
  ProcessIsolatedTestRunner::ExecutionOptions options;
  options.stopOnFirstFailure = true;

  bool passed = ProcessIsolatedTestRunner::executeAllTests(options);
  EXPECT_TRUE(passed) << "Critical test suite failed";
}
```

---

## Best Practices

### 1. Always Clear Before Registration

```cpp
TEST(MyTest, IsolatedTests) {
  // ✅ GOOD: Clear previous registrations
  ProcessIsolatedTestRunner::clear();

  ProcessIsolatedTestRunner::registerTest(/* ... */);
}
```

### 2. Use Descriptive Test Names

```cpp
// ❌ BAD: Vague name
ProcessIsolatedTestRunner::registerTest("Test1", []() { /* ... */ });

// ✅ GOOD: Descriptive name
ProcessIsolatedTestRunner::registerTest(
    "GFX942_LargeRanks_P2PChunkSize_ExpectHighValue",
    []() { /* ... */ }
);
```

### 3. Group Related Tests

```cpp
TEST(Rcclwrap, AllP2PChunkSizeTests) {
  ProcessIsolatedTestRunner::clear();

  // Register all related tests together
  registerGFX942Tests();
  registerGFX950Tests();
  registerUnsupportedArchTests();

  bool passed = ProcessIsolatedTestRunner::executeAllTests();
  EXPECT_TRUE(passed);
}
```

### 4. Use Options for Better Control

```cpp
// For debugging: verbose + stop on failure
ProcessIsolatedTestRunner::ExecutionOptions debugOptions;
debugOptions.stopOnFirstFailure = true;
debugOptions.verboseLogging = true;

// For CI: run all tests, collect all failures
ProcessIsolatedTestRunner::ExecutionOptions ciOptions;
ciOptions.stopOnFirstFailure = false;
debugOptions.verboseLogging = false;
```

### 5. Set Appropriate Timeouts

```cpp
// ✅ GOOD: Different timeouts for different test types
quickTest.withTimeout(std::chrono::seconds(5));
normalTest.withTimeout(std::chrono::seconds(30));
slowTest.withTimeout(std::chrono::seconds(120));

// ❌ BAD: Same long timeout for everything
allTests.withTimeout(std::chrono::seconds(300));
```

### 6. Clean Up Resources in Tests

```cpp
ProcessIsolatedTestRunner::registerTest("ResourceTest", []() {
  ncclComm_t comm = nullptr;
  struct ncclTopoSystem topo;
  struct ncclTopoNode gpuNode;
  CreateMockComm(comm, topo, gpuNode, "gfx942", 32);

  try {
    // Your test logic
    EXPECT_TRUE(someTest(comm));

    // ✅ GOOD: Clean up in all paths
    CleanupMockComm(comm);
  } catch (...) {
    CleanupMockComm(comm);
    throw;
  }
});
```

---

## Troubleshooting

### Test Hangs / Times Out

**Symptom:** Test never completes, eventually times out.

**Solutions:**
1. Increase timeout: `.withTimeout(std::chrono::seconds(120))`
2. Check for deadlocks in test logic
3. Enable verbose logging to see where it hangs:
   ```cpp
   options.verboseLogging = true;
   ```

### Environment Variables Not Being Set

**Symptom:** `getenv()` returns `nullptr` in test.

**Solutions:**
1. Verify environment variable name is correct
2. Check that you're calling `withEnvironment()`:
   ```cpp
   config.withEnvironment({{"VAR_NAME", "value"}})
   ```
3. Verify the test is actually executing (check test name)

### Tests Pass Individually but Fail Together

**Symptom:** Individual tests pass, but fail when run in a suite.

**Cause:** This is the **exact problem** that ProcessIsolatedTestRunner solves!

**Solution:** Already solved - each test runs in isolated process. If you're still seeing this, check:
1. Are you using `executeAllTests()` correctly?
2. Are there shared external resources (files, network, etc.)?

### Fork Failures

**Symptom:** Error messages about fork() failing.

**Solutions:**
1. Check system resource limits: `ulimit -u` (max processes)
2. Reduce number of tests or run in smaller batches
3. Check for resource leaks in parent process

### Test Results Not Available

**Symptom:** `getTestResults()` returns empty vector.

**Solution:**
```cpp
// Call executeAllTests() first
ProcessIsolatedTestRunner::executeAllTests();

// Then get results
auto results = ProcessIsolatedTestRunner::getTestResults();
```

---

## Implementation Details

### How It Works

1. **Registration Phase:**
   - Tests are registered into a static vector
   - Each test gets a `TestConfig` with name, logic, and environment

2. **Execution Phase:**
   - Parent process iterates through registered tests
   - For each test:
     - `fork()` creates a child process
     - Child applies environment variables
     - Child executes test logic
     - Parent waits for child to complete
     - Result is collected and stored

3. **Result Collection:**
   - Exit codes are captured from child processes
   - Timing information is recorded
   - All results stored in static vector

### Exit Codes

```cpp
enum RcclTestCode {
  RCCL_TEST_SUCCESS = 0,           // Test passed
  RCCL_TEST_FAILURE = 1,           // Test failed (assertion)
  RCCL_TEST_UNKNOWN_EXCEPTION = 2, // Uncaught exception
  RCCL_TEST_TIMEOUT = 3,           // Test timed out
  RCCL_TEST_SKIPPED = 4            // Test was skipped
};
```

### Thread Safety

The framework uses mutexes for thread-safe operations:
- Test registration (write)
- Result recording (write)
- Result retrieval (read)

---

## Limitations

1. **Process Overhead:** Each test creates a new process (fork overhead)
2. **Sequential Execution:** Tests run one at a time (not parallel)
3. **Linux/Unix Only:** Uses `fork()` - not available on Windows
4. **Memory Duplication:** Each forked process duplicates memory
5. **No Shared State:** Tests cannot share data between processes

---

## FAQ

**Q: When should I use ProcessIsolatedTestRunner vs regular Google Test?**

A: Use ProcessIsolatedTestRunner when:
- Testing code with static variables
- Testing environment variable behavior
- Testing one-time initialization
- Need guaranteed clean state between tests

Use regular Google Test when:
- Tests are truly independent
- No static state concerns
- Need parallel execution
- Testing simple units

**Q: Can I use this with MPI tests?**

A: Not directly. Process Isolated test runner is for single-process tests. For MPI tests, use `MPI Test Runner` instead. Process Isolated test runner is currently hooked into `rccl-UnitTestsFixtures` binary and MPI test runner is hooked into `rccl-UnitTestsMPI` binary. These are two independent implementation.

**Q: How do I debug a test that's running in an isolated process?**

A:
1. Enable verbose logging
2. Add print statements in your test lambda
3. Temporarily run the test logic outside the framework
4. Use GDB

**Q: Can I run tests in parallel?**

A: No, the current implementation only supports sequential execution.

**Q: Does this work with CTest/CMake?**

A: Yes! The tests are still Google Test cases, so they work with standard test runners.

---

## See Also

- **ProcessIsolatedTestRunner.hpp** - Full API documentation
- **ProcessIsolatedTestRunner.cpp** - Implementation details
- **RcclWrapTests.cpp** - Usage examples
- **MPITestBase.hpp** - For multi-process MPI-based tests
