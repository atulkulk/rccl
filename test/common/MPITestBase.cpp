/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPITestBase.cpp
 * @brief Implementation of MPI-based test infrastructure
 */

#include "MPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

/**
 * @brief Validate that the MPI environment meets test requirements
 *
 * Checks world size against minimum process count and power-of-two requirements.
 * If validation fails, the test is automatically skipped with GTEST_SKIP().
 * This ensures tests don't fail due to insufficient resources.
 */
void MPITestBase::validateTestPrerequisites(int min_processes,
                                            bool require_power_of_two) {
  int world_rank = RCCLMPIEnvironment::world_rank;
  int world_size = RCCLMPIEnvironment::world_size;

  if (world_size < min_processes) {
    if (world_rank == 0) {
      printf("Rank %d: Skipping test - requires at least %d processes, got %d\n",
             world_rank, min_processes, world_size);
    }
    GTEST_SKIP() << "Test requires at least " << min_processes
                 << " processes, got " << world_size;
  }

  if (require_power_of_two && !MPITestConstants::isPowerOfTwo(world_size)) {
    if (world_rank == 0) {
      printf("Rank %d: Skipping test - requires power of 2 processes, got %d\n",
             world_rank, world_size);
    }
    GTEST_SKIP() << "Test requires power of 2 processes, got " << world_size;
  }
}

/**
 * @brief Create and initialize test-specific NCCL communicator and HIP stream
 *
 * This method performs the following steps:
 * 1. Generates a unique NCCL ID on rank 0
 * 2. Broadcasts the ID to all ranks via MPI
 * 3. Creates NCCL communicator using ncclGroupStart/End pattern
 * 4. Creates a dedicated HIP stream for the test
 * 5. Synchronizes all ranks before returning
 *
 * @return ncclSuccess if successful, NCCL error code otherwise
 *
 * @note Uses ncclGroupStart/End which is critical for shared memory setup
 * @note Idempotent - safe to call multiple times (returns immediately if already created)
 * @note All ranks must call this function for proper synchronization
 */
ncclResult_t MPITestBase::createTestCommunicator() {
  if (using_test_comm) {
    return ncclSuccess; // Already created
  }

  int world_rank = RCCLMPIEnvironment::world_rank;
  int world_size = RCCLMPIEnvironment::world_size;

  if (world_rank == 0) {
    printf("Rank %d: Creating test-specific communicator (isolated from shared comm)\n",
           world_rank);
  }

  // Get unique ID
  ncclUniqueId nccl_id;
  if (world_rank == 0) {
    ncclResult_t result = ncclGetUniqueId(&nccl_id);
    if (result != ncclSuccess) {
      printf("Rank %d: Failed to get unique ID: %s\n",
             world_rank, ncclGetErrorString(result));
      return result;
    }
  }

  // Broadcast unique ID
  int mpi_result = MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (mpi_result != MPI_SUCCESS) {
    printf("Rank %d: MPI_Bcast failed\n", world_rank);
    return ncclSystemError;
  }

  // Barrier before comm init
  MPI_Barrier(MPI_COMM_WORLD);

  // Use ncclGroupStart/End for proper initialization (critical for shared memory setup)
  ncclResult_t result = ncclGroupStart();
  if (result != ncclSuccess) {
    printf("Rank %d: Failed to start NCCL group: %s\n",
           world_rank, ncclGetErrorString(result));
    return result;
  }

  // Initialize test communicator
  result = ncclCommInitRank(&test_comm, world_size, nccl_id, world_rank);
  if (result != ncclSuccess) {
    printf("Rank %d: Failed to initialize test communicator: %s\n",
           world_rank, ncclGetErrorString(result));
    ncclGroupEnd();  // Clean up group
    return result;
  }

  result = ncclGroupEnd();
  if (result != ncclSuccess) {
    printf("Rank %d: Failed to end NCCL group: %s\n",
           world_rank, ncclGetErrorString(result));
    if (test_comm) {
      ncclCommDestroy(test_comm);
      test_comm = nullptr;
    }
    return result;
  }

  // Create test stream
  hipError_t hip_result = hipStreamCreate(&test_stream);
  if (hip_result != hipSuccess) {
    printf("Rank %d: Failed to create test stream: %s\n",
           world_rank, hipGetErrorString(hip_result));
    ncclCommDestroy(test_comm);
    test_comm = nullptr;
    return ncclSystemError;
  }

  // Barrier after comm init - ensure all ranks have completed group init
  MPI_Barrier(MPI_COMM_WORLD);

  using_test_comm = true;

  if (world_rank == 0) {
    printf("Rank %d: Test-specific communicator created successfully\n", world_rank);
  }

  return ncclSuccess;
}

/**
 * @brief Retrieve the active NCCL communicator for test operations
 *
 * Returns the test-specific communicator that was created by createTestCommunicator().
 * If no communicator has been created, this will trigger a test failure.
 *
 * @return The active NCCL communicator handle, or nullptr with test failure
 *
 * @warning Always call createTestCommunicator() before calling this method
 */
ncclComm_t MPITestBase::getActiveCommunicator() {
  if (!using_test_comm) {
    // No test communicator created - this is an error
    ADD_FAILURE() << "No test communicator created. Call createTestCommunicator() first.";
    return nullptr;
  }
  return test_comm;
}

/**
 * @brief Retrieve the active HIP stream for GPU operations
 *
 * Returns the test-specific HIP stream that was created by createTestCommunicator().
 * If no stream has been created, this will trigger a test failure.
 *
 * @return The active HIP stream handle, or nullptr with test failure
 *
 * @warning Always call createTestCommunicator() before calling this method
 */
hipStream_t MPITestBase::getActiveStream() {
  if (!using_test_comm) {
    // No test stream created - this is an error
      ADD_FAILURE() << "No test stream created. Call createTestCommunicator() first.";
    return nullptr;
  }
  return test_stream;
}

/**
 * @brief Clean up test-specific NCCL and HIP resources
 *
 * Destroys the test communicator and stream with proper synchronization:
 * 1. MPI barrier before cleanup (ensures all ranks are ready)
 * 2. Destroys HIP stream
 * 3. Destroys NCCL communicator
 * 4. MPI barrier after cleanup (ensures all ranks completed)
 *
 * @note Safe to call multiple times or if resources were never created
 * @note Automatically called by TearDown() after each test
 */
void MPITestBase::cleanupTestCommunicator() {
  if (!using_test_comm) {
    return; // Nothing to cleanup
  }

  // Synchronize before cleanup
  MPI_Barrier(MPI_COMM_WORLD);

  // Destroy test stream
  if (test_stream) {
    hipStreamDestroy(test_stream);
    test_stream = nullptr;
  }

  // Destroy test communicator
  if (test_comm) {
    ncclCommDestroy(test_comm);
    test_comm = nullptr;
  }

  using_test_comm = false;

  // Synchronize after cleanup
  MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Google Test TearDown hook - automatically called after each test
 *
 * Ensures proper cleanup of test resources by calling cleanupTestCommunicator().
 * This guarantees that each test starts with a clean slate and prevents
 * resource leaks between tests.
 */
void MPITestBase::TearDown() {
  cleanupTestCommunicator();
}

#endif // MPI_TESTS_ENABLED

