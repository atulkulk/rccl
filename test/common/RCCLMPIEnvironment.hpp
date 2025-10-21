/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RCCLMPIEnvironment.hpp
 * @brief Global MPI environment and error checking macros for RCCL testing
 *
 * Provides a Google Test Environment for managing MPI initialization/finalization
 * and error checking macros for MPI, NCCL, and HIP operations in tests.
 *
 * @see RCCLMPIEnvironment for the main environment class
 * @see MPICHECK for MPI error checking
 * @see NCCLCHECK for NCCL error checking
 * @see HIPCHECK for HIP error checking
 */

#ifndef RCCL_MPI_ENVIRONMENT_HPP
#define RCCL_MPI_ENVIRONMENT_HPP

#include <gtest/gtest.h>

// Conditionally include MPI headers for MPI-based tests
#ifdef MPI_TESTS_ENABLED
#include <mpi.h>
#include <hip/hip_runtime.h>
#include "rccl/rccl.h"

/**
 * @def MPICHECK
 * @brief Context-aware MPI error checking macro with overloaded behavior
 *
 * Provides three usage modes depending on context:
 *
 * @par Usage Modes:
 * - `MPICHECK(cmd)` - Normal test code: Fails test with FAIL() on error
 * - `MPICHECK(cmd, rank)` - Cleanup code: Calls MPI_Abort() on error
 * - `MPICHECK(cmd, rank, true)` - MPI_Finalize: Calls std::exit() on error
 *
 * @par Example:
 * @code
 * // In test body
 * MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
 *
 * // In cleanup code
 * MPICHECK(MPI_Barrier(MPI_COMM_WORLD), world_rank);
 *
 * // During finalization
 * MPICHECK(MPI_Finalize(), world_rank, true);
 * @endcode
 *
 * @note Prints detailed error message including file, line, and MPI error string
 */

// Helper macros for argument counting
#define MPICHECK_GET_MACRO(_1,_2,_3,NAME,...) NAME
#define MPICHECK(...) MPICHECK_GET_MACRO(__VA_ARGS__, MPICHECK_3, MPICHECK_2, MPICHECK_1)(__VA_ARGS__)

/**
 * @def MPICHECK_1
 * @brief 1-argument version: Normal test code (uses FAIL())
 * @hideinitializer
 */
#define MPICHECK_1(cmd) do {                                    \
   int err = cmd;                                              \
   if (err != MPI_SUCCESS) {                                   \
       char error_string[MPI_MAX_ERROR_STRING];                \
       int length;                                             \
       MPI_Error_string(err, error_string, &length);          \
       printf("MPI Error at %s:%d - %s\n", __FILE__, __LINE__, error_string); \
       FAIL() << "MPI Error: " << error_string;               \
   }                                                           \
} while(0)

/**
 * @def MPICHECK_2
 * @brief 2-argument version: Cleanup code (uses MPI_Abort())
 * @hideinitializer
 */
#define MPICHECK_2(cmd, rank) do {                              \
   int err = cmd;                                              \
   if (err != MPI_SUCCESS) {                                   \
       char error_string[MPI_MAX_ERROR_STRING];                \
       int length;                                             \
       MPI_Error_string(err, error_string, &length);          \
       std::fprintf(stderr, "Rank %d: MPI Error at %s:%d - %s\n", rank, __FILE__, __LINE__, error_string); \
       std::fflush(stderr);                                    \
       MPI_Abort(MPI_COMM_WORLD, err);                         \
   }                                                           \
} while(0)

/**
 * @def MPICHECK_3
 * @brief 3-argument version: MPI_Finalize (uses std::exit())
 * @hideinitializer
 */
#define MPICHECK_3(cmd, rank, is_finalize) do {                 \
   int err = cmd;                                              \
   if (err != MPI_SUCCESS) {                                   \
       char error_string[MPI_MAX_ERROR_STRING];                \
       int length;                                             \
       MPI_Error_string(err, error_string, &length);          \
       std::fprintf(stderr, "Rank %d: MPI_Finalize Error at %s:%d - %s\n", rank, __FILE__, __LINE__, error_string); \
       std::fflush(stderr);                                    \
       std::exit(err);                                         \
   }                                                           \
} while(0)

/**
 * @def NCCLCHECK
 * @brief NCCL error checking macro
 *
 * Checks the result of NCCL function calls and fails the test if an error occurs.
 * Prints file location, line number, and NCCL error string.
 *
 * @par Example:
 * @code
 * NCCLCHECK(ncclAllReduce(sendbuf, recvbuf, count, datatype, op, comm, stream));
 * @endcode
 */
#define NCCLCHECK(cmd) do {                                 \
   ncclResult_t res = cmd;                                 \
   if (res != ncclSuccess) {                               \
       printf("RCCL Error at %s:%d - %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
       FAIL() << "RCCL Error: " << ncclGetErrorString(res); \
   }                                                       \
} while(0)

/**
 * @def HIPCHECK
 * @brief HIP error checking macro
 *
 * Checks the result of HIP function calls and fails the test if an error occurs.
 * Prints file location, line number, and HIP error string.
 *
 * @par Example:
 * @code
 * HIPCHECK(hipMalloc(&d_ptr, size));
 * HIPCHECK(hipMemcpy(d_ptr, h_ptr, size, hipMemcpyHostToDevice));
 * @endcode
 */
#define HIPCHECK(cmd) do {                                  \
   hipError_t err = cmd;                                   \
   if (err != hipSuccess) {                                \
       printf("HIP Error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
       FAIL() << "HIP Error: " << hipGetErrorString(err); \
   }                                                       \
} while(0)

/**
 * @class RCCLMPIEnvironment
 * @brief Google Test Environment for global MPI setup and teardown
 *
 * Manages the global MPI state for all MPI-based tests:
 * - One-time MPI initialization (MPI_Init_thread)
 * - GPU device initialization and assignment
 * - MPI finalization and result aggregation across ranks
 *
 * @par Usage:
 * @code
 * int main(int argc, char** argv) {
 *   ::testing::InitGoogleTest(&argc, argv);
 *   ::testing::AddGlobalTestEnvironment(new RCCLMPIEnvironment());
 *   return RUN_ALL_TESTS();
 * }
 * @endcode
 *
 * @note MPI_Init can only be called once, so this uses static flags
 * @note Each MPI rank is assigned to a unique GPU
 * @see MPITestBase for test-level functionality
 */
class RCCLMPIEnvironment : public ::testing::Environment {
public:
    /**
     * @brief Current MPI rank in MPI_COMM_WORLD
     *
     * Valid after MPI initialization. Each rank corresponds to one GPU.
     */
    inline static int world_rank{0};

    /**
     * @brief Total number of MPI processes in MPI_COMM_WORLD
     *
     * Valid after MPI initialization. Must not exceed number of available GPUs.
     */
    inline static int world_size{0};

    /**
     * @brief Aggregated return code for test results
     *
     * Set to non-zero on test failure. Aggregated across all ranks during cleanup.
     */
    inline static int retCode{0};

    /**
     * @brief Flag indicating MPI has been initialized
     *
     * Prevents multiple MPI_Init calls (only allowed once per process).
     */
    inline static bool mpi_initialized{false};

    /**
     * @brief Flag indicating GPU devices have been initialized
     *
     * Prevents redundant device setup across multiple test runs.
     */
    inline static bool devices_initialized{false};

    /**
     * @brief Initialize MPI with thread support
     *
     * Calls MPI_Init_thread() with MPI_THREAD_MULTIPLE support and sets
     * world_rank and world_size. Safe to call multiple times (idempotent).
     *
     * @note Should be called before any MPI operations
     * @see mpi_initialized flag
     */
    static void initialize_mpi();

    /**
     * @brief Initialize and assign GPU devices to MPI ranks
     *
     * Performs the following:
     * 1. Queries available GPU count
     * 2. Validates sufficient GPUs for all ranks
     * 3. Assigns one GPU per rank (rank N → GPU N)
     * 4. Resets and sets HIP device context
     * 5. Synchronizes all ranks
     *
     * @note Requires world_size ≤ number of available GPUs
     * @see devices_initialized flag
     */
    static void initialize_devices();

    /**
     * @brief Clean up MPI resources and finalize
     *
     * Performs the following cleanup:
     * 1. Synchronizes all ranks with MPI_Barrier
     * 2. Aggregates test results across ranks with MPI_Allreduce
     * 3. Prints final results from rank 0
     * 4. Calls MPI_Finalize()
     *
     * @note Uses static guard to prevent multiple cleanup attempts
     * @note Safe to call from signal handlers or error paths
     */
    static void cleanup_mpi();

    /**
     * @brief Google Test SetUp hook - called once before all tests
     *
     * Initializes MPI and GPU devices for the entire test suite.
     */
    void SetUp() override;

    /**
     * @brief Google Test TearDown hook - called once after all tests
     *
     * Synchronizes all ranks and calls cleanup_mpi() to finalize MPI.
     */
    void TearDown() override;
};

#endif // MPI_TESTS_ENABLED

#endif // RCCL_MPI_ENVIRONMENT_HPP
