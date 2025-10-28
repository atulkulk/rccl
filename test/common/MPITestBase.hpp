/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPITestBase.hpp
 * @brief Base class infrastructure for MPI-based RCCL testing
 *
 * Provides a common test base class for writing multi-process distributed tests
 * using MPI and RCCL. Handles communicator creation, process validation, and
 * resource cleanup automatically.
 *
 * @see MPITestBase for the main base class
 * @see RCCLMPIEnvironment for global MPI setup
 */

#ifndef MPI_TEST_BASE_HPP
#define MPI_TEST_BASE_HPP

#include "gtest/gtest.h"

#ifdef MPI_TESTS_ENABLED
    #include "RCCLMPIEnvironment.hpp"
    #include "rccl/rccl.h"
    #include "utils.h" // For getHostName() from RCCL
    #include <cstdio>
    #include <cstdlib>
    #include <cstring>
    #include <hip/hip_runtime.h>
    #include <mpi.h>
    #include <string>

/**
 * @brief Test logging macros that respect NCCL_DEBUG environment variable
 *
 * These macros mirror NCCL's debug levels and provide conditional logging for MPI tests:
 * - TEST_NONE: No logging (always disabled)
 * - TEST_VERSION: Version information (enabled when NCCL_DEBUG=VERSION or higher)
 * - TEST_WARN: Warning messages (enabled when NCCL_DEBUG=WARN or higher)
 * - TEST_INFO: Informational messages (enabled when NCCL_DEBUG=INFO or higher)
 * - TEST_ABORT: Abort-level messages (enabled when NCCL_DEBUG=ABORT or higher)
 * - TEST_TRACE: Trace-level messages (enabled when NCCL_DEBUG=TRACE)
 *
 * Debug levels (from least to most verbose):
 *   NONE < VERSION < WARN < INFO < ABORT < TRACE
 *
 * Usage (hostname and rank are added automatically):
 *   TEST_VERSION("Test suite version: %s", version);
 *   TEST_WARN("Unexpected condition detected");
 *   TEST_INFO("Starting test with buffer size %zu", size);
 *   TEST_TRACE("Entering function %s", __func__);
 */

// Forward declaration of NCCL's global debug level
extern int ncclDebugLevel;

// Helper function to get numeric debug level from RCCL's internal variable
// Uses RCCL's ncclDebugLevel which is set during RCCL initialization from NCCL_DEBUG env var
// This ensures consistency with RCCL library's own logging
inline int getTestDebugLevel()
{
    // ncclDebugLevel values (from NCCL's ncclDebugLogLevel enum):
    // NCCL_LOG_NONE = 0, NCCL_LOG_VERSION = 1, NCCL_LOG_WARN = 2,
    // NCCL_LOG_INFO = 3, NCCL_LOG_ABORT = 4, NCCL_LOG_TRACE = 5
    return ncclDebugLevel;
}

// Helper function to get MPI rank for logging
inline int getTestMpiRank()
{
    int rank = -1;
    #ifdef MPI_TESTS_ENABLED
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif
    return rank;
}

// Helper function to get hostname for logging (uses RCCL's getHostName)
inline const char* getTestHostname()
{
    static char hostname[256] = {0};
    static bool initialized   = false;

    if(!initialized)
    {
        // Use RCCL's getHostName() which handles hostname retrieval and uses '.' as delimiter
        // This matches how RCCL's debug.cc initializes its hostname
        if(getHostName(hostname, sizeof(hostname), '.') != ncclSuccess)
        {
            strncpy(hostname, "unknown", sizeof(hostname) - 1);
        }
        initialized = true;
    }
    return hostname;
}

// Helper function to check if running in multi-node configuration
inline bool isMultiNodeTest()
{
    static int cached_result = -1; // -1 = not computed, 0 = single node, 1 = multi-node

    if(cached_result != -1)
    {
        return cached_result == 1;
    }

    #ifdef MPI_TESTS_ENABLED
    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if(world_size <= 1)
    {
        cached_result = 0; // Single process = single node
        return false;
    }

    // Split by shared memory to detect nodes
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

    int node_size = 0;
    MPI_Comm_size(node_comm, &node_size);
    MPI_Comm_free(&node_comm);

    // If node_size < world_size, we have multiple nodes
    cached_result = (node_size < world_size) ? 1 : 0;
    return cached_result == 1;
    #else
    cached_result = 0;
    return false;
    #endif
}

    // TEST_NONE: Never logs
    #define TEST_NONE(...) \
        do                 \
        {                  \
        }                  \
        while(0)

    // TEST_VERSION: Logs when NCCL_DEBUG=VERSION or higher
    #define TEST_VERSION(...)                                                 \
        do                                                                    \
        {                                                                     \
            if(getTestDebugLevel() >= 1)                                      \
            {                                                                 \
                int rank = getTestMpiRank();                                  \
                if(isMultiNodeTest())                                         \
                {                                                             \
                    printf("%s:[%d] TEST VERSION ", getTestHostname(), rank); \
                }                                                             \
                else                                                          \
                {                                                             \
                    printf("[%d] TEST VERSION ", rank);                       \
                }                                                             \
                printf(__VA_ARGS__);                                          \
                printf("\n");                                                 \
                fflush(stdout);                                               \
            }                                                                 \
        }                                                                     \
        while(0)

    // TEST_WARN: Logs when NCCL_DEBUG=WARN or higher
    #define TEST_WARN(...)                                                 \
        do                                                                 \
        {                                                                  \
            if(getTestDebugLevel() >= 2)                                   \
            {                                                              \
                int rank = getTestMpiRank();                               \
                if(isMultiNodeTest())                                      \
                {                                                          \
                    printf("%s:[%d] TEST WARN ", getTestHostname(), rank); \
                }                                                          \
                else                                                       \
                {                                                          \
                    printf("[%d] TEST WARN ", rank);                       \
                }                                                          \
                printf(__VA_ARGS__);                                       \
                printf("\n");                                              \
                fflush(stdout);                                            \
            }                                                              \
        }                                                                  \
        while(0)

    // TEST_INFO: Logs when NCCL_DEBUG=INFO or higher
    #define TEST_INFO(...)                                                 \
        do                                                                 \
        {                                                                  \
            if(getTestDebugLevel() >= 3)                                   \
            {                                                              \
                int rank = getTestMpiRank();                               \
                if(isMultiNodeTest())                                      \
                {                                                          \
                    printf("%s:[%d] TEST INFO ", getTestHostname(), rank); \
                }                                                          \
                else                                                       \
                {                                                          \
                    printf("[%d] TEST INFO ", rank);                       \
                }                                                          \
                printf(__VA_ARGS__);                                       \
                printf("\n");                                              \
                fflush(stdout);                                            \
            }                                                              \
        }                                                                  \
        while(0)

    // TEST_ABORT: Logs when NCCL_DEBUG=ABORT or higher
    #define TEST_ABORT(...)                                                 \
        do                                                                  \
        {                                                                   \
            if(getTestDebugLevel() >= 4)                                    \
            {                                                               \
                int rank = getTestMpiRank();                                \
                if(isMultiNodeTest())                                       \
                {                                                           \
                    printf("%s:[%d] TEST ABORT ", getTestHostname(), rank); \
                }                                                           \
                else                                                        \
                {                                                           \
                    printf("[%d] TEST ABORT ", rank);                       \
                }                                                           \
                printf(__VA_ARGS__);                                        \
                printf("\n");                                               \
                fflush(stdout);                                             \
            }                                                               \
        }                                                                   \
        while(0)

    // TEST_TRACE: Logs when NCCL_DEBUG=TRACE
    #define TEST_TRACE(...)                                                 \
        do                                                                  \
        {                                                                   \
            if(getTestDebugLevel() >= 5)                                    \
            {                                                               \
                int rank = getTestMpiRank();                                \
                if(isMultiNodeTest())                                       \
                {                                                           \
                    printf("%s:[%d] TEST TRACE ", getTestHostname(), rank); \
                }                                                           \
                else                                                        \
                {                                                           \
                    printf("[%d] TEST TRACE ", rank);                       \
                }                                                           \
                printf(__VA_ARGS__);                                        \
                printf("\n");                                               \
                fflush(stdout);                                             \
            }                                                               \
        }                                                                   \
        while(0)

/**
 * @namespace MPITestConstants
 * @brief Common constants and utilities for MPI-based tests
 */
namespace MPITestConstants
{
/**
   * @brief Minimum number of MPI processes required for distributed testing
   */
inline constexpr int kMinProcessesForMPI = 2;

/**
   * @brief Flag indicating power-of-two process count is required
   */
inline constexpr bool kRequirePowerOfTwo = true;

/**
   * @brief Flag indicating power-of-two process count is not required
   */
inline constexpr bool kNoPowerOfTwoRequired = false;

/**
   * @brief Flag indicating single-node execution is required
   */
inline constexpr int kRequireSingleNode = 1;

/**
   * @brief Flag indicating no node limit (multi-node is allowed)
   */
inline constexpr int kNoNodeLimit = 0;

/**
   * @brief Flag indicating no maximum process count limit
   */
inline constexpr int kNoProcessLimit = 0;

/**
   * @brief Check if a number is a power of 2
   * @param n The number to check
   * @return true if n is a power of 2, false otherwise
   */
inline bool isPowerOfTwo(int n)
{
    return (n > 0) && ((n & (n - 1)) == 0);
}

/**
   * @brief Detect the number of unique nodes in the MPI communicator
   *
   * Uses MPI_Comm_split_type(MPI_COMM_TYPE_SHARED) to detect nodes.
   * Assigns unique node IDs via MPI_Exscan and broadcasts within each node.
   * Respects MPI's internal node allocation (hostfiles, SLURM, PBS, etc.).
   * Displays detailed process distribution from rank 0.
   *
   * @return Number of unique nodes detected by MPI
   *
   * @note This is a collective operation - all ranks must call it
   * @note Works for N nodes with any rank ordering
   * @note Displays detailed rank-to-node mapping
   */
int detectNodeCount();
} // namespace MPITestConstants

/**
 * @class MPITestBase
 * @brief Base class for all MPI-based RCCL tests
 *
 * Provides infrastructure for writing multi-process distributed tests with:
 * - Process count validation (minimum processes, power-of-two requirements)
 * - Test-specific NCCL communicator creation and management
 * - HIP stream management for each test
 * - Automatic resource cleanup
 *
 * @par Usage Example:
 * @code
 * class MyMPITest : public MPITestBase {};
 *
 * TEST_F(MyMPITest, BasicAllReduce) {
 *   validateTestPrerequisites(2);  // Need at least 2 processes
 *   NCCLCHECK(createTestCommunicator());
 *
 *   ncclComm_t comm = getActiveCommunicator();
 *   hipStream_t stream = getActiveStream();
 *
 *   // Your test logic here...
 *   // Cleanup happens automatically in TearDown()
 * }
 * @endcode
 *
 * @see RCCLMPIEnvironment for global MPI initialization
 */
class MPITestBase : public ::testing::Test
{
protected:
    /**
   * @brief Test-specific NCCL communicator handle
   *
   * Created by createTestCommunicator() for isolated testing.
   * Automatically destroyed in TearDown().
   */
    ncclComm_t test_comm = nullptr;

    /**
   * @brief Test-specific HIP stream handle
   *
   * Created alongside the test communicator for GPU operations.
   * Automatically destroyed in TearDown().
   */
    hipStream_t test_stream = nullptr;

    /**
   * @brief Flag indicating whether test-specific resources have been created
   */
    bool using_test_comm = false;

    /**
   * @brief Validate test prerequisites and return whether requirements are met
   *
   * Checks if the current MPI world size meets the test requirements.
   * Displays what the test requires and whether the environment satisfies those requirements.
   * Returns true if all requirements met, false otherwise.
   *
   * Parameters are organized by category:
   * - Process requirements: min_processes, max_processes, require_power_of_two
   * - Node requirements: min_nodes, max_nodes
   *
   * @param min_processes Minimum number of MPI processes required (default: 1)
   * @param max_processes Maximum number of MPI processes allowed (0 = no limit) (default: 0)
   * @param require_power_of_two If true, world size must be a power of 2 (default: false)
   * @param min_nodes Minimum number of nodes required (default: 1)
   * @param max_nodes Maximum number of nodes allowed (0 = no limit) (default: 0)
   *
   * @return true if all requirements are met, false otherwise
   *
   * @par Example:
   * @code
   * // Single process test (uses all defaults)
   * TEST_F(MyTest, SingleProcessTest) {
   *     if(!validateTestPrerequisites()) {
   *         GTEST_SKIP() << "Test validation failed";
   *     }
   *     // Test logic...
   * }
   *
   * // Exactly 2 processes on exactly 1 node
   * TEST_F(MyTest, TwoRankSingleNodeTest) {
   *     if(!validateTestPrerequisites(2, 2, false, 1, 1)) {
   *         GTEST_SKIP() << "Need exactly 2 ranks on exactly 1 node";
   *     }
   *     // Test logic...
   * }
   *
   * // 2-8 processes, single node only
   * TEST_F(MyTest, FlexibleSingleNodeTest) {
   *     if(!validateTestPrerequisites(2, 8, false, 1, 1)) {
   *         GTEST_SKIP() << "Need 2-8 ranks on single node";
   *     }
   *     // Test logic...
   * }
   *
   * // At least 4 processes, power-of-two, multi-node allowed
   * TEST_F(MyTest, PowerOfTwoTest) {
   *     if(!validateTestPrerequisites(4, 0, true)) {
   *         GTEST_SKIP() << "Need power-of-two processes (4+)";
   *     }
   *     // Test logic...
   * }
   * @endcode
   */
    bool validateTestPrerequisites(int  min_processes        = 1,
                                   int  max_processes        = MPITestConstants::kNoProcessLimit,
                                   bool require_power_of_two = false,
                                   int  min_nodes            = 1,
                                   int  max_nodes            = MPITestConstants::kNoNodeLimit);

    /**
   * @brief Create a test-specific NCCL communicator and HIP stream
   *
   * Creates isolated NCCL communicator and HIP stream for this test.
   * Uses ncclGroupStart/End for proper initialization and MPI barriers
   * for synchronization across all ranks.
   *
   * @return ncclSuccess on success, or NCCL error code on failure
   *
   * @note This function is idempotent - calling it multiple times is safe
   * @note Communicator is automatically destroyed in TearDown()
   *
   * @par Example:
   * @code
   * TEST_F(MyMPITest, BasicTest) {
   *   NCCLCHECK(createTestCommunicator());
   *   ncclComm_t comm = getActiveCommunicator();
   *   // Use comm for NCCL operations...
   * }
   * @endcode
   */
    virtual ncclResult_t createTestCommunicator();

    /**
   * @brief Get the active NCCL communicator for this test
   *
   * Returns the test-specific communicator. Fails if createTestCommunicator()
   * has not been called first.
   *
   * @return The active NCCL communicator handle, or nullptr if not created
   *
   * @note Always call createTestCommunicator() before this method
   */
    virtual ncclComm_t getActiveCommunicator();

    /**
   * @brief Get the active HIP stream for this test
   *
   * Returns the test-specific HIP stream. Fails if createTestCommunicator()
   * has not been called first.
   *
   * @return The active HIP stream handle, or nullptr if not created
   *
   * @note Always call createTestCommunicator() before this method
   */
    virtual hipStream_t getActiveStream();

    /**
   * @brief Cleanup test-specific NCCL communicator and HIP stream
   *
   * Destroys the test communicator and stream with proper MPI synchronization.
   * Safe to call multiple times or if resources were never created.
   *
   * @note This is automatically called by TearDown()
   */
    virtual void cleanupTestCommunicator();

    /**
   * @brief Google Test TearDown hook - ensures cleanup of test resources
   *
   * Automatically called after each test completes. Calls cleanupTestCommunicator()
   * to ensure proper resource cleanup.
   */
    void TearDown() override;
};

#endif // MPI_TESTS_ENABLED

#endif // MPI_TEST_BASE_HPP
