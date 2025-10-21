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
#include <cstdio>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include "rccl/rccl.h"

/**
 * @namespace MPITestConstants
 * @brief Common constants and utilities for MPI-based tests
 */
namespace MPITestConstants {
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
   * @brief Check if a number is a power of 2
   * @param n The number to check
   * @return true if n is a power of 2, false otherwise
   */
  inline bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
  }
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
class MPITestBase : public ::testing::Test {
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
   * @brief Validate test prerequisites before running the test
   *
   * Checks if the current MPI world size meets the test requirements.
   * Automatically skips the test using GTEST_SKIP() if requirements are not met.
   *
   * @param min_processes Minimum number of MPI processes required
   * @param require_power_of_two If true, world size must be a power of 2
   *
   * @par Example:
   * @code
   * TEST_F(MyMPITest, AllReduce) {
   *   validateTestPrerequisites(4, MPITestConstants::kRequirePowerOfTwo);
   *   // Test will be skipped if world_size < 4 or not power of 2
   * }
   * @endcode
   */
  virtual void validateTestPrerequisites(int min_processes,
                                         bool require_power_of_two = false);

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

