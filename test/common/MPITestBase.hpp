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

#include "MPITestCore.hpp"
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
 * These macros mirror RCCL's debug levels and provide conditional logging for MPI tests:
 * - TEST_WARN: Warning messages (enabled when NCCL_DEBUG=WARN or higher)
 * - TEST_INFO: Informational messages (enabled when NCCL_DEBUG=INFO or higher)
 * - TEST_ABORT: Abort-level messages (enabled when NCCL_DEBUG=ABORT or higher)
 * - TEST_TRACE: Trace-level messages (enabled when NCCL_DEBUG=TRACE)
 *
 * Debug levels (from least to most verbose):
 *   NONE < WARN < INFO < ABORT < TRACE
 *
 * Usage (hostname and rank are added automatically):
 *   TEST_WARN("Unexpected condition detected");
 *   TEST_INFO("Starting test with buffer size %zu", size);
 *   TEST_TRACE("Entering function %s", __func__);
 */

/**
 * @class MPITestBase
 * @brief Google Test adapter for MPI tests
 *
 * Integrates MPITestCore with Google Test framework for seamless MPI testing.
 * Inherits from both ::testing::Test (for GTest integration) and MPITestCore
 * (for MPI/RCCL functionality).
 *
 * **Features:**
 * - Process count validation (minimum processes, power-of-two requirements)
 * - Node count validation (single-node vs multi-node)
 * - Test-specific RCCL communicator creation and management
 * - HIP stream management for each test
 * - Automatic resource cleanup via GTest TearDown
 *
 * **Usage Example:**
 * @code
 * class MyMPITest : public MPITestBase {};
 *
 * TEST_F(MyMPITest, BasicAllReduce) {
 *   if (!validateTestPrerequisites(2)) {
 *     GTEST_SKIP() << "Need at least 2 processes";
 *   }
 *   ASSERT_EQ(ncclSuccess, createTestCommunicator());
 *
 *   ncclComm_t comm = getActiveCommunicator();
 *   hipStream_t stream = getActiveStream();
 *
 *   // Your test logic here...
 *   // Cleanup happens automatically in TearDown()
 * }
 * @endcode
 *
 * @note For standalone tests without GTest, use MPIStandaloneTest instead
 * @see MPITestCore for the base framework-agnostic functionality
 * @see RCCLMPIEnvironment for global MPI initialization
 */
/**
 * @brief Google Test adapter for MPI tests
 *
 * Integrates MPITestCore with Google Test framework by inheriting from both
 * ::testing::Test and MPITestCore.
 *
 * @note For standalone tests (without GTest), use MPIStandaloneTest instead
 */
class MPITestBase
    : public ::testing::Test
    , public MPITestCore
{
public:
    /**
   * @brief Google Test SetUp hook - initializes test resources
   *
   * Automatically called before each test runs. Calls initializeTest()
   * from MPITestCore for any custom initialization.
   *
   * @note No ambiguity with MPITestCore::initializeTest() - different names
   */
    void SetUp() override
    {
        initializeTest();
    }

    /**
   * @brief Google Test TearDown hook - ensures cleanup of test resources
   *
   * Automatically called after each test completes. Calls cleanupTest()
   * from MPITestCore to ensure proper resource cleanup.
   */
    void TearDown() override
    {
        cleanupTest();
    }
};

    /**
 * @defgroup MPI_AWARE_ASSERTIONS MPI-Aware Assertion Macros
 * @brief Assertion macros that synchronize all MPI ranks before failing
 *
 * These macros prevent deadlocks by ensuring all ranks reach the same assertion
 * point together. When any rank fails, ALL ranks are notified and skip together.
 *
 * **Problem Solved:**
 * - Regular ASSERT_* macros cause only the failing rank to exit to TearDown()
 * - Other ranks continue in the test body
 * - Results in deadlock when ranks call different MPI collectives
 *
 * **Solution:**
 * - Use MPI_Allreduce to check if ANY rank failed
 * - If any rank fails, ALL ranks skip together
 * - Prevents deadlock by keeping ranks synchronized
 *
 * **Usage:**
 * @code
 * TEST_F(MyMPITest, Example) {
 *   // Use MPI-aware assertions for operations that can fail on some ranks
 *   ASSERT_MPI_SUCCESS(createTestCommunicator());
 *   ASSERT_MPI_EQ(expected, actual);
 *   ASSERT_MPI_TRUE(condition);
 *
 *   // If any rank fails, ALL ranks skip to TearDown together
 * }
 * @endcode
 *
 * @note Adds MPI overhead (MPI_Allreduce) on each assertion
 * @note Use for critical assertions that can fail differently across ranks
 * @note For performance-critical code, use regular ASSERT_* if safe
 *
 * @{
 */

    /**
 * @brief MPI-aware version of ASSERT_TRUE
 *
 * Checks if condition is true on all ranks. If ANY rank fails,
 * ALL ranks skip together to prevent deadlock.
 *
 * @param condition The condition to test
 */
    #define ASSERT_MPI_TRUE(condition)                                                            \
        do                                                                                        \
        {                                                                                         \
            bool _local_pass    = static_cast<bool>(condition);                                   \
            int  _local_status  = _local_pass ? 1 : 0;                                            \
            int  _global_status = 0;                                                              \
            MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                                  \
            if(_global_status == 0)                                                               \
            {                                                                                     \
                /* At least one rank failed */                                                    \
                if(!_local_pass)                                                                  \
                {                                                                                 \
                    /* This rank failed - show the actual error */                                \
                    EXPECT_TRUE(condition)                                                        \
                        << "Rank " << RCCLMPIEnvironment::world_rank << " failed assertion";      \
                }                                                                                 \
                /* All ranks skip together */                                                     \
                GTEST_SKIP()                                                                      \
                    << "Rank " << RCCLMPIEnvironment::world_rank                                  \
                    << ": Skipping test due to failure on at least one rank (synchronized exit)"; \
            }                                                                                     \
        }                                                                                         \
        while(0)

    /**
 * @brief MPI-aware version of ASSERT_FALSE
 */
    #define ASSERT_MPI_FALSE(condition) ASSERT_MPI_TRUE(!(condition))

    /**
 * @brief MPI-aware version of ASSERT_EQ
 *
 * Checks if val1 == val2 on all ranks. If ANY rank fails,
 * ALL ranks skip together to prevent deadlock.
 *
 * @param val1 First value
 * @param val2 Second value
 */
    #define ASSERT_MPI_EQ(val1, val2)                                                             \
        do                                                                                        \
        {                                                                                         \
            auto _v1            = (val1);                                                         \
            auto _v2            = (val2);                                                         \
            bool _local_pass    = (_v1 == _v2);                                                   \
            int  _local_status  = _local_pass ? 1 : 0;                                            \
            int  _global_status = 0;                                                              \
            MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                                  \
            if(_global_status == 0)                                                               \
            {                                                                                     \
                if(!_local_pass)                                                                  \
                {                                                                                 \
                    EXPECT_EQ(_v1, _v2)                                                           \
                        << "Rank " << RCCLMPIEnvironment::world_rank << " failed assertion";      \
                }                                                                                 \
                GTEST_SKIP()                                                                      \
                    << "Rank " << RCCLMPIEnvironment::world_rank                                  \
                    << ": Skipping test due to failure on at least one rank (synchronized exit)"; \
            }                                                                                     \
        }                                                                                         \
        while(0)

    /**
 * @brief MPI-aware version of ASSERT_NE
 */
    #define ASSERT_MPI_NE(val1, val2)                                                             \
        do                                                                                        \
        {                                                                                         \
            auto _v1            = (val1);                                                         \
            auto _v2            = (val2);                                                         \
            bool _local_pass    = (_v1 != _v2);                                                   \
            int  _local_status  = _local_pass ? 1 : 0;                                            \
            int  _global_status = 0;                                                              \
            MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                                  \
            if(_global_status == 0)                                                               \
            {                                                                                     \
                if(!_local_pass)                                                                  \
                {                                                                                 \
                    EXPECT_NE(_v1, _v2)                                                           \
                        << "Rank " << RCCLMPIEnvironment::world_rank << " failed assertion";      \
                }                                                                                 \
                GTEST_SKIP()                                                                      \
                    << "Rank " << RCCLMPIEnvironment::world_rank                                  \
                    << ": Skipping test due to failure on at least one rank (synchronized exit)"; \
            }                                                                                     \
        }                                                                                         \
        while(0)

    /**
 * @brief MPI-aware version for checking ncclResult_t == ncclSuccess
 *
 * Special macro for RCCL operations that provides better error messages.
 * If ANY rank fails, ALL ranks skip together.
 *
 * @param expr Expression that returns ncclResult_t
 */
    #define ASSERT_MPI_SUCCESS(expr)                                                              \
        do                                                                                        \
        {                                                                                         \
            ncclResult_t _result        = (expr);                                                 \
            bool         _local_pass    = (_result == ncclSuccess);                               \
            int          _local_status  = _local_pass ? 1 : 0;                                    \
            int          _global_status = 0;                                                      \
            MPI_Allreduce(&_local_status, &_global_status, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);  \
                                                                                                  \
            if(_global_status == 0)                                                               \
            {                                                                                     \
                if(!_local_pass)                                                                  \
                {                                                                                 \
                    EXPECT_EQ(ncclSuccess, _result) << "Rank " << RCCLMPIEnvironment::world_rank  \
                                                    << ": " << ncclGetErrorString(_result);       \
                }                                                                                 \
                GTEST_SKIP()                                                                      \
                    << "Rank " << RCCLMPIEnvironment::world_rank                                  \
                    << ": Skipping test due to failure on at least one rank (synchronized exit)"; \
            }                                                                                     \
        }                                                                                         \
        while(0)

    /** @} */ // end of MPI_AWARE_ASSERTIONS group

#endif // MPI_TESTS_ENABLED

#endif // MPI_TEST_BASE_HPP
