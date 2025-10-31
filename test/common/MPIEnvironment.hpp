/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPIEnvironment.hpp
 * @brief Global MPI environment and error checking macros for RCCL testing
 *
 * Provides a Google Test Environment for managing MPI initialization/finalization
 * and error checking macros for MPI, NCCL, and HIP operations in tests.
 *
 * @see MPIEnvironment for the main environment class
 * @see MPICHECK for MPI error checking
 * @see NCCLCHECK for NCCL error checking
 * @see HIPCHECK for HIP error checking
 */

#ifndef RCCL_MPI_ENVIRONMENT_HPP
#define RCCL_MPI_ENVIRONMENT_HPP

#include <gtest/gtest.h>

// Conditionally include MPI headers for MPI-based tests
#ifdef MPI_TESTS_ENABLED
    #include "rccl/rccl.h"
    #include <hip/hip_runtime.h>
    #include <mpi.h>

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
    #define MPICHECK_GET_MACRO(_1, _2, _3, NAME, ...) NAME
    #define MPICHECK(...) \
        MPICHECK_GET_MACRO(__VA_ARGS__, MPICHECK_3, MPICHECK_2, MPICHECK_1)(__VA_ARGS__)

    /**
 * @def MPICHECK_1
 * @brief 1-argument version: Normal test code (uses FAIL())
 * @hideinitializer
 */
    #define MPICHECK_1(cmd)                                                            \
        do                                                                             \
        {                                                                              \
            int err = cmd;                                                             \
            if(err != MPI_SUCCESS)                                                     \
            {                                                                          \
                char error_string[MPI_MAX_ERROR_STRING];                               \
                int  length;                                                           \
                MPI_Error_string(err, error_string, &length);                          \
                printf("MPI Error at %s:%d - %s\n", __FILE__, __LINE__, error_string); \
                FAIL() << "MPI Error: " << error_string;                               \
            }                                                                          \
        }                                                                              \
        while(0)

    /**
 * @def MPICHECK_2
 * @brief 2-argument version: Cleanup code (uses MPI_Abort())
 * @hideinitializer
 */
    #define MPICHECK_2(cmd, rank)                                  \
        do                                                         \
        {                                                          \
            int err = cmd;                                         \
            if(err != MPI_SUCCESS)                                 \
            {                                                      \
                char error_string[MPI_MAX_ERROR_STRING];           \
                int  length;                                       \
                MPI_Error_string(err, error_string, &length);      \
                std::fprintf(stderr,                               \
                             "Rank %d: MPI Error at %s:%d - %s\n", \
                             rank,                                 \
                             __FILE__,                             \
                             __LINE__,                             \
                             error_string);                        \
                std::fflush(stderr);                               \
                MPI_Abort(MPI_COMM_WORLD, err);                    \
            }                                                      \
        }                                                          \
        while(0)

    /**
 * @def MPICHECK_3
 * @brief 3-argument version: MPI_Finalize (uses std::exit())
 * @hideinitializer
 */
    #define MPICHECK_3(cmd, rank, is_finalize)                              \
        do                                                                  \
        {                                                                   \
            int err = cmd;                                                  \
            if(err != MPI_SUCCESS)                                          \
            {                                                               \
                char error_string[MPI_MAX_ERROR_STRING];                    \
                int  length;                                                \
                MPI_Error_string(err, error_string, &length);               \
                std::fprintf(stderr,                                        \
                             "Rank %d: MPI_Finalize Error at %s:%d - %s\n", \
                             rank,                                          \
                             __FILE__,                                      \
                             __LINE__,                                      \
                             error_string);                                 \
                std::fflush(stderr);                                        \
                std::exit(err);                                             \
            }                                                               \
        }                                                                   \
        while(0)

    /**
 * @def RCCL_TEST_CHECK
 * @brief RCCL error checking macro for test infrastructure code
 *
 * Similar to the library's NCCLCHECK macro but for test infrastructure.
 * Returns ncclResult_t on error instead of calling exit().
 * Use this in test infrastructure/framework code (e.g., setup/teardown methods)
 * that needs to return errors gracefully.
 *
 * Behavior:
 * - Checks NCCL function result
 * - Logs error with fprintf (visible even without NCCL_DEBUG)
 * - Returns error code to caller
 *
 * @note Requires enclosing function to return ncclResult_t
 * @note For test bodies, use RCCL_TEST_CHECK_GTEST_FAIL instead (calls FAIL())
 * @note For code with cleanup, use ScopeGuard pattern
 *
 * Example (in infrastructure code):
 *   ncclResult_t setupComm() {
 *     RCCL_TEST_CHECK(ncclCommInitRank(&comm, size, id, rank));
 *     RCCL_TEST_CHECK(ncclGroupStart());
 *     return ncclSuccess;
 *   }
 */
    #define RCCL_TEST_CHECK(cmd)                            \
        do                                                  \
        {                                                   \
            ncclResult_t res = cmd;                         \
            if(res != ncclSuccess && res != ncclInProgress) \
            {                                               \
                fprintf(stderr,                             \
                        "RCCL Error at %s:%d - %s\n",       \
                        __FILE__,                           \
                        __LINE__,                           \
                        ncclGetErrorString(res));           \
                return res;                                 \
            }                                               \
        }                                                   \
        while(0)

    /**
 * @def RCCL_TEST_CHECK_GTEST_FAIL
 * @brief RCCL error checking macro for GTest test bodies
 *
 * Test-specific version of NCCLCHECK that integrates with Google Test.
 * Checks the result of NCCL function calls and fails the test if an error occurs.
 * Prints file location, line number, and NCCL error string, then calls FAIL()
 * from Google Test to mark the test as failed.
 *
 * @note Use this in TEST_F/TEST_P test bodies
 * @note For infrastructure code (setup/teardown), use RCCL_TEST_CHECK instead
 *
 * Example (in test body):
 *   TEST_F(MyTest, Example) {
 *     RCCL_TEST_CHECK_GTEST_FAIL(ncclAllReduce(...));
 *   }
 */
    #define RCCL_TEST_CHECK_GTEST_FAIL(cmd)                                                        \
        do                                                                                         \
        {                                                                                          \
            ncclResult_t res = cmd;                                                                \
            if(res != ncclSuccess)                                                                 \
            {                                                                                      \
                printf("RCCL Error at %s:%d - %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
                FAIL() << "RCCL Error: " << ncclGetErrorString(res);                               \
            }                                                                                      \
        }                                                                                          \
        while(0)

    /**
 * @def HIP_TEST_CHECK
 * @brief HIP error checking macro for test infrastructure code
 *
 * Similar to the library's CUDACHECK macro - returns an error code instead of exiting.
 * Use this in test infrastructure/framework code (e.g., setup/teardown methods) that
 * needs to return errors gracefully rather than calling exit().
 *
 * Behavior:
 * - Checks HIP function result
 * - Logs error with fprintf (visible even without NCCL_DEBUG)
 * - Returns ncclUnhandledCudaError to caller
 *
 * @note Requires enclosing function to return ncclResult_t
 * @note For test bodies, use HIP_TEST_CHECK_GTEST_FAIL instead (calls FAIL())
 * @note For code with cleanup, use ScopeGuard pattern
 *
 * Example (in infrastructure code):
 *   ncclResult_t setupResources() {
 *     void* buffer = nullptr;
 *     HIP_TEST_CHECK(hipMalloc(&buffer, size));
 *     HIP_TEST_CHECK(hipMemset(buffer, 0, size));
 *     return ncclSuccess;
 *   }
 */
    #define HIP_TEST_CHECK(cmd)                                      \
        do                                                           \
        {                                                            \
            hipError_t err = cmd;                                    \
            if(err != hipSuccess)                                    \
            {                                                        \
                fprintf(stderr,                                      \
                        "HIP Error at %s:%d - %s (hipError_t=%d)\n", \
                        __FILE__,                                    \
                        __LINE__,                                    \
                        hipGetErrorString(err),                      \
                        static_cast<int>(err));                      \
                return ncclUnhandledCudaError;                       \
            }                                                        \
        }                                                            \
        while(0)

    /**
 * @def HIPCHECK
 * @brief HIP error checking macro (library-style)
 *
 * Similar to RCCL library's CUDACHECK macro. Returns ncclUnhandledCudaError on error.
 * Use in any code that returns ncclResult_t.
 *
 * Behavior:
 * - Checks HIP function result
 * - Logs error with fprintf (visible even without NCCL_DEBUG)
 * - Returns ncclUnhandledCudaError to caller
 *
 * @note Requires enclosing function to return ncclResult_t
 * @note For GTest test bodies, use HIP_TEST_CHECK_GTEST_FAIL instead
 * @note This mirrors the library's CUDACHECK behavior
 *
 * Example:
 *   ncclResult_t setupBuffer() {
 *     HIPCHECK(hipMalloc(&buffer, size));
 *     HIPCHECK(hipMemcpy(buffer, data, size, hipMemcpyHostToDevice));
 *     return ncclSuccess;
 *   }
 */
    #ifndef HIPCHECK
        #define HIPCHECK(cmd)                                            \
            do                                                           \
            {                                                            \
                hipError_t err = cmd;                                    \
                if(err != hipSuccess)                                    \
                {                                                        \
                    fprintf(stderr,                                      \
                            "HIP Error at %s:%d - %s (hipError_t=%d)\n", \
                            __FILE__,                                    \
                            __LINE__,                                    \
                            hipGetErrorString(err),                      \
                            static_cast<int>(err));                      \
                    return ncclUnhandledCudaError;                       \
                }                                                        \
            }                                                            \
            while(0)
    #endif // HIPCHECK

    /**
 * @def HIP_TEST_CHECK_GTEST_FAIL
 * @brief HIP error checking for GTest test bodies
 *
 * Checks HIP function calls and fails the test if an error occurs.
 * Use in TEST_F/TEST_P test bodies.
 *
 * Behavior:
 * - Checks HIP function result
 * - Prints error to stdout (always visible)
 * - Calls FAIL() to mark test as failed
 *
 * Example (in test body):
 *   TEST_F(MyTest, Example) {
 *     HIP_TEST_CHECK_GTEST_FAIL(hipMalloc(&buffer, size));
 *     HIP_TEST_CHECK_GTEST_FAIL(hipMemcpy(buffer, data, size, hipMemcpyHostToDevice));
 *   }
 *
 * @note Use HIPCHECK or HIP_TEST_CHECK in infrastructure code that returns ncclResult_t
 */
    #define HIP_TEST_CHECK_GTEST_FAIL(cmd)                                                       \
        do                                                                                       \
        {                                                                                        \
            hipError_t err = cmd;                                                                \
            if(err != hipSuccess)                                                                \
            {                                                                                    \
                printf("HIP Error at %s:%d - %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
                FAIL() << "HIP Error: " << hipGetErrorString(err);                               \
            }                                                                                    \
        }                                                                                        \
        while(0)

    // ============================================================================
    // Generic RAII Scope Guard for C++11+
    // ============================================================================

    /**
 * @brief Include generic RAII scope guard utilities
 *
 * Provides ScopeGuard template class, makeScopeGuard() helper, and SCOPE_EXIT macro
 * for exception-safe automatic cleanup - the C++ alternative to goto-based cleanup.
 *
 * @par Key Components:
 * - ScopeGuard<Func> - Generic RAII guard template
 * - makeScopeGuard(lambda) - Helper for creating guards with type deduction
 * - SCOPE_EXIT(code) - Macro for quick one-liner cleanup
 *
 * @par Quick Example:
 * @code
 * void* buffer = nullptr;
 * hipMalloc(&buffer, size);
 * auto guard = makeScopeGuard([&]() { if(buffer) hipFree(buffer); });
 * // Automatic cleanup on scope exit
 * @endcode
 *
 * @see GenericScopeGuard.hpp for full documentation and examples
 * @see RAII_CLEANUP_PATTERNS.md for comprehensive usage guide
 */
    #include "GenericScopeGuard.hpp"

/**
 * @class MPIEnvironment
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
 *   ::testing::AddGlobalTestEnvironment(new MPIEnvironment());
 *   return RUN_ALL_TESTS();
 * }
 * @endcode
 *
 * @note MPI_Init can only be called once, so this uses static flags
 * @note Each MPI rank is assigned to a unique GPU
 * @see MPITestBase for test-level functionality
 */
class MPIEnvironment : public ::testing::Environment
{
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
     * @brief Cached result of multi-node detection
     *
     * Computed once during SetUp() using MPI_Comm_split_type().
     * -1 = not computed, 0 = single node, 1 = multi-node
     *
     * @note MUST be initialized before any TEST_* macros are called
     * @note Prevents nested MPI collective operations in isMultiNodeTest()
     */
    inline static int cached_multi_node_result{-1};

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
