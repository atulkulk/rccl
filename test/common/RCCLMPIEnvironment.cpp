/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RCCLMPIEnvironment.cpp
 * @brief Implementation of global MPI environment for RCCL testing
 */

#include "RCCLMPIEnvironment.hpp"

#ifdef MPI_TESTS_ENABLED

/**
 * @brief Initialize the global test environment
 *
 * Performs one-time setup for the entire test suite:
 * - Initializes MPI with thread support
 * - Sets up GPU devices for each rank
 *
 * @note Called automatically by Google Test framework before any tests run
 */
void RCCLMPIEnvironment::SetUp()
{
    // One-time initialization (MPI_Init can only be called once)
    initialize_mpi();
    initialize_devices();
}

/**
 * @brief Initialize MPI with multi-threading support
 *
 * Calls MPI_Init_thread() with MPI_THREAD_MULTIPLE to support concurrent
 * MPI operations. Sets world_rank and world_size for use by all tests.
 *
 * Idempotent - safe to call multiple times (uses mpi_initialized flag).
 * Typically called from main_mpi.cpp, but provides fallback initialization.
 */
void RCCLMPIEnvironment::initialize_mpi()
{
    if(mpi_initialized)
    {
        // Already initialized in main_mpi.cpp
        if(world_rank == 0)
        {
            printf("Rank %d: MPI already initialized - skipping re-initialization\n", world_rank);
        }
        return;
    }

    // This path should not be reached when using main_mpi.cpp
    // but kept for compatibility with other test mains
    auto provided = int{};
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

    mpi_initialized = true;

    if(world_rank == 0)
    {
        printf("Rank %d: MPI initialized - World size: %d, Thread support: %d\n",
               world_rank,
               world_size,
               provided);
    }
}

/**
 * @brief Initialize GPU devices and assign one GPU per MPI rank
 *
 * Performs comprehensive GPU setup:
 * 1. Queries number of available GPUs
 * 2. Validates sufficient GPUs for world_size
 * 3. Assigns GPU ID = rank (rank-based assignment)
 * 4. Resets HIP context for clean state
 * 5. Sets active device
 * 6. Verifies device assignment
 * 7. Synchronizes all ranks
 *
 * @note Requires at least world_size GPUs
 * @note Sets retCode=1 on error (insufficient GPUs, assignment failure)
 * @note Idempotent - safe to call multiple times (uses devices_initialized flag)
 */
void RCCLMPIEnvironment::initialize_devices()
{
    if(devices_initialized)
    {
        return; // Already initialized
    }

    auto numDevices = int{};
    HIPCHECK(hipGetDeviceCount(&numDevices));

    // Calculate local rank (rank within this node) for multi-node support
    // Split MPI_COMM_WORLD by node using MPI_Comm_split_type
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        world_rank,
                        MPI_INFO_NULL,
                        &node_comm);

    int local_rank, local_size;
    MPI_Comm_rank(node_comm, &local_rank);
    MPI_Comm_size(node_comm, &local_size);

    if(world_rank == 0)
    {
        printf("Rank %d: Detected %d GPU(s) for %d MPI rank(s)\n",
               world_rank,
               numDevices,
               world_size);
        printf("Rank %d: Local configuration: %d ranks per node\n", world_rank, local_size);
    }

    // Check if we have enough GPUs for ranks on THIS node
    if(numDevices < local_size)
    {
        printf("ERROR: Rank %d (local rank %d): Only %d GPUs available on this node for %d local "
               "ranks.\n"
               "RCCL requires unique GPUs per rank on each node.\n"
               "Please run with fewer ranks per node (e.g., --ntasks-per-node=%d) "
               "or ensure more GPUs are available.\n",
               world_rank,
               local_rank,
               numDevices,
               local_size,
               numDevices);
        retCode             = 1;
        devices_initialized = true;
        MPI_Comm_free(&node_comm);
        return;
    }

    // Use LOCAL rank for device assignment (not global rank)
    // This ensures ranks 0-7 on each node use GPUs 0-7
    const auto assigned_device = local_rank;

    // Validate device assignment
    if(assigned_device < 0 || assigned_device >= numDevices)
    {
        printf("ERROR: Rank %d (local rank %d): Invalid device assignment! "
               "assigned_device=%d, numDevices=%d\n",
               world_rank,
               local_rank,
               assigned_device,
               numDevices);
        retCode             = 1;
        devices_initialized = true;
        MPI_Comm_free(&node_comm);
        return;
    }

    // Complete HIP context reset and isolation
    HIPCHECK(hipDeviceReset());
    HIPCHECK(hipSetDevice(assigned_device));

    // Force HIP context creation and synchronization
    auto prop = hipDeviceProp_t{};
    HIPCHECK(hipGetDeviceProperties(&prop, assigned_device));
    HIPCHECK(hipDeviceSynchronize());

    // Verify device assignment
    auto current_device = int{};
    HIPCHECK(hipGetDevice(&current_device));
    if(current_device != assigned_device)
    {
        printf("ERROR: Rank %d (local rank %d) device assignment failed! Expected %d, got %d\n",
               world_rank,
               local_rank,
               assigned_device,
               current_device);
        retCode = 1;
        MPI_Comm_free(&node_comm);
        return;
    }

    // Print device info (only from rank 0 to reduce output)
    if(world_rank == 0)
    {
        printf("Rank %d (local rank %d): Device assignment: global rank %d -> GPU %d\n"
               "Rank %d: PCI Bus ID = 0x%x, Device Name = %s\n"
               "Rank %d: Total GPUs available per node: %d\n"
               "Rank %d: Multi-node: Each node's local ranks (0-%d) mapped to GPUs (0-%d)\n",
               world_rank,
               local_rank,
               world_rank,
               assigned_device,
               world_rank,
               prop.pciBusID,
               prop.name,
               world_rank,
               numDevices,
               world_rank,
               local_size - 1,
               numDevices - 1);
    }

    // Clean up node communicator
    MPI_Comm_free(&node_comm);

    // Ensure all ranks have set their devices before proceeding
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

    devices_initialized = true;

    if(world_rank == 0)
    {
        printf("Rank %d: Device initialization completed\n"
               "Rank %d: Each test will create its own NCCL communicator for isolation\n",
               world_rank,
               world_rank);
    }
}

/**
 * @brief Tear down the global test environment
 *
 * Ensures all ranks have completed their tests before cleanup:
 * 1. Synchronizes all ranks with MPI_Barrier
 * 2. Calls cleanup_mpi() to finalize MPI
 *
 * @note Critical synchronization point - ensures all test cleanup is complete
 * @note Called automatically by Google Test framework after all tests complete
 */
void RCCLMPIEnvironment::TearDown()
{
    // CRITICAL: Synchronize ALL ranks BEFORE calling cleanup_mpi()
    // This ensures all ranks complete their test-level teardown before starting global cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    cleanup_mpi();
}

/**
 * @brief Clean up MPI resources and finalize
 *
 * Performs coordinated cleanup across all ranks:
 * 1. Guards against multiple cleanup attempts
 * 2. Synchronizes all ranks
 * 3. Aggregates test results using MPI_Allreduce
 * 4. Prints final results from rank 0
 * 5. Calls MPI_Finalize()
 * 6. Resets initialization flags
 *
 * Uses context-aware error handling:
 * - MPI_Barrier/Allreduce: MPICHECK with rank (aborts on error)
 * - MPI_Finalize: MPICHECK with rank and true flag (exits on error)
 *
 * @note Uses static guard to prevent multiple cleanup attempts
 * @note Safe to call from signal handlers or error paths
 * @note All ranks must call this function for proper finalization
 */
void RCCLMPIEnvironment::cleanup_mpi()
{
    // Use static guard to prevent multiple cleanup attempts
    static bool cleanup_in_progress_or_done = false;

    if(cleanup_in_progress_or_done)
    {
        return; // Already cleaned up or currently cleaning up
    }

    if(!mpi_initialized)
    {
        return; // Never initialized
    }

    cleanup_in_progress_or_done = true;

    // Synchronize all ranks before MPI finalization
    MPICHECK(MPI_Barrier(MPI_COMM_WORLD), world_rank);

    MPICHECK(MPI_Finalize(), world_rank, true);

    mpi_initialized     = false;
    devices_initialized = false;
}

#endif // MPI_TESTS_ENABLED
