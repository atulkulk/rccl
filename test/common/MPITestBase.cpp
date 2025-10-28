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
#include <map>
#include <set>
#include <string>
#include <vector>

#ifdef MPI_TESTS_ENABLED

/**
 * @brief Detect the number of unique nodes in the MPI communicator
 *
 * Implementation of MPITestConstants::detectNodeCount().
 * Uses MPI_Comm_split_type with MPI_COMM_TYPE_SHARED to detect nodes.
 * Each node gets a unique ID via MPI_Exscan, which is broadcast to all
 * ranks on that node. This accurately respects MPI's internal node
 * allocation (hostfiles, SLURM, PBS, etc.) for any number of nodes.
 */
int MPITestConstants::detectNodeCount()
{
    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Use MPI_Comm_split_type to detect actual nodes (respects hostfile allocation)
    // MPI_COMM_TYPE_SHARED groups processes that can share memory (same physical node)
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        world_rank,
                        MPI_INFO_NULL,
                        &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // Assign a unique node ID to each node
    // Node leaders (node_rank == 0) do an exclusive scan to get unique IDs
    int is_node_leader = (node_rank == 0) ? 1 : 0;
    int node_id        = 0;
    MPI_Exscan(&is_node_leader, &node_id, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // node_id is now: 0 for first leader, 1 for second leader, etc.

    // Broadcast node_id within each node so all ranks know their node ID
    MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm);

    // Count total number of nodes
    int unique_node_count = 0;
    MPI_Allreduce(&is_node_leader, &unique_node_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // Get hostname for display
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int  name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Gather all node IDs and hostnames to rank 0 for display
    std::vector<int>  all_node_ids;
    std::vector<char> all_names;
    if(world_rank == 0)
    {
        all_node_ids.resize(world_size);
        all_names.resize(world_size * MPI_MAX_PROCESSOR_NAME);
    }

    MPI_Gather(&node_id, 1, MPI_INT, all_node_ids.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gather(processor_name,
               MPI_MAX_PROCESSOR_NAME,
               MPI_CHAR,
               all_names.data(),
               MPI_MAX_PROCESSOR_NAME,
               MPI_CHAR,
               0,
               MPI_COMM_WORLD);

    if(world_rank == 0)
    {
        // Group ranks by their node ID
        std::map<int, std::vector<int>> node_id_to_ranks;
        std::map<int, std::string>      node_id_to_hostname;

        for(int i = 0; i < world_size; i++)
        {
            int nid = all_node_ids[i];
            node_id_to_ranks[nid].push_back(i);
            if(node_id_to_hostname.find(nid) == node_id_to_hostname.end())
            {
                node_id_to_hostname[nid] = std::string(&all_names[i * MPI_MAX_PROCESSOR_NAME]);
            }
        }

        // Print detailed node and rank distribution
        TEST_INFO("");
        TEST_INFO("=== MPI Process Distribution ===");
        TEST_INFO("Total processes: %d", world_size);
        TEST_INFO("Detected nodes:  %d (via MPI_COMM_TYPE_SHARED)", unique_node_count);
        TEST_INFO("");

        for(const auto& pair : node_id_to_ranks)
        {
            int                     node_id  = pair.first;
            const std::vector<int>& ranks    = pair.second;
            const std::string&      hostname = node_id_to_hostname[node_id];

            TEST_INFO("Node %d: %s (%zu ranks)", node_id, hostname.c_str(), ranks.size());

            // Build ranks string
            std::string ranks_str = "  Ranks: ";
            for(size_t i = 0; i < ranks.size(); i++)
            {
                ranks_str += std::to_string(ranks[i]);
                if(i < ranks.size() - 1)
                    ranks_str += ", ";
            }
            TEST_INFO("%s", ranks_str.c_str());
        }
        TEST_INFO("================================");
        TEST_INFO("");
    }

    // Clean up node communicator
    MPI_Comm_free(&node_comm);

    return unique_node_count;
}

/**
 * @brief Validate that the MPI environment meets test requirements
 *
 * Checks world size against minimum/maximum process count, power-of-two, and node count
 * requirements. Displays what the test requires and whether those requirements
 * are met. Returns true if all requirements satisfied, false otherwise.
 *
 * Parameters organized by category for clarity:
 * - Process requirements: min_processes, max_processes, require_power_of_two
 * - Node requirements: min_nodes, max_nodes
 *
 * @param min_processes Minimum number of MPI processes required (default: 1)
 * @param max_processes Maximum number of MPI processes allowed (0 = no limit)
 * @param require_power_of_two If true, world size must be a power of 2
 * @param min_nodes Minimum number of nodes required (default: 1)
 * @param max_nodes Maximum number of nodes allowed (0 = no limit)
 * @return true if all requirements met, false otherwise
 */
bool MPITestBase::validateTestPrerequisites(
    int min_processes, int max_processes, bool require_power_of_two, int min_nodes, int max_nodes)
{
    int world_rank = RCCLMPIEnvironment::world_rank;
    int world_size = RCCLMPIEnvironment::world_size;

    // Display test requirements (rank 0 only)
    if(world_rank == 0)
    {
        TEST_INFO("");
        TEST_INFO("=== Test Requirements ===");

        // Process count requirements
        if(max_processes > 0 && max_processes == min_processes)
        {
            TEST_INFO("Processes:         exactly %d", min_processes);
        }
        else if(max_processes > 0)
        {
            TEST_INFO("Processes:         %d-%d", min_processes, max_processes);
        }
        else
        {
            TEST_INFO("Min processes:     %d", min_processes);
        }

        TEST_INFO("Power-of-two:      %s", require_power_of_two ? "required" : "not required");

        // Node count requirements
        if(min_nodes > 1 && max_nodes > 0 && min_nodes == max_nodes)
        {
            TEST_INFO("Nodes:             exactly %d", min_nodes);
        }
        else if(min_nodes == 1 && max_nodes == 1)
        {
            TEST_INFO("Nodes:             exactly 1 (single-node only)");
        }
        else if(min_nodes > 1 && max_nodes > 0)
        {
            TEST_INFO("Nodes:             %d-%d", min_nodes, max_nodes);
        }
        else if(max_nodes > 0)
        {
            TEST_INFO("Max nodes:         %d%s",
                      max_nodes,
                      max_nodes == 1 ? " (single-node only)" : "");
        }
        else if(min_nodes > 1)
        {
            TEST_INFO("Min nodes:         %d", min_nodes);
        }
        else
        {
            TEST_INFO("Nodes:             unlimited");
        }

        TEST_INFO("");
        TEST_INFO("=== Current Environment ===");
        TEST_INFO("Actual processes:  %d", world_size);
        TEST_INFO("Is power-of-two:   %s",
                  MPITestConstants::isPowerOfTwo(world_size) ? "yes" : "no");
    }

    // Check minimum process count
    if(world_size < min_processes)
    {
        if(world_rank == 0)
        {
            TEST_INFO("Actual nodes:      (checking...)");
            TEST_INFO("");
            printf("❌ REQUIREMENT NOT MET: Need at least %d processes, got %d\n",
                   min_processes,
                   world_size);
            printf("   For test details, set: NCCL_DEBUG=INFO\n");
            printf("===========================\n\n");
        }
        return false;
    }

    // Check maximum process count
    if(max_processes > 0 && world_size > max_processes)
    {
        if(world_rank == 0)
        {
            TEST_INFO("Actual nodes:      (checking...)");
            TEST_INFO("");
            printf("❌ REQUIREMENT NOT MET: Need at most %d processes, got %d\n",
                   max_processes,
                   world_size);
            if(min_processes == max_processes)
            {
                printf("   This test requires exactly %d processes\n", min_processes);
            }
            printf("   For test details, set: NCCL_DEBUG=INFO\n");
            printf("===========================\n\n");
        }
        return false;
    }

    // Check power-of-two requirement
    if(require_power_of_two && !MPITestConstants::isPowerOfTwo(world_size))
    {
        if(world_rank == 0)
        {
            TEST_INFO("Actual nodes:      (checking...)");
            TEST_INFO("");
            printf("❌ REQUIREMENT NOT MET: Need power-of-two processes, got %d\n", world_size);
            printf("   For test details, set: NCCL_DEBUG=INFO\n");
            printf("===========================\n\n");
        }
        return false;
    }

    // Check node count if any node constraint is specified
    // min_nodes=1 with max_nodes=0 means "no constraint" (default)
    if(min_nodes > 1 || max_nodes > 0)
    {
        int node_count = MPITestConstants::detectNodeCount();

        if(world_rank == 0)
        {
            TEST_INFO("Actual nodes:      %d", node_count);
            TEST_INFO("");
        }

        // Check minimum nodes
        if(min_nodes > 1 && node_count < min_nodes)
        {
            if(world_rank == 0)
            {
                printf("❌ REQUIREMENT NOT MET: Need at least %d node(s), detected %d nodes\n",
                       min_nodes,
                       node_count);
                printf("   For test details, set: NCCL_DEBUG=INFO\n");
                printf("===========================\n\n");
            }
            return false;
        }

        // Check maximum nodes
        if(max_nodes > 0 && node_count > max_nodes)
        {
            if(world_rank == 0)
            {
                printf("❌ REQUIREMENT NOT MET: Need at most %d node(s), detected %d nodes\n",
                       max_nodes,
                       node_count);
                if(max_nodes == 1)
                {
                    printf("   This test uses P2P/SHM transport (single-node only)\n");
                    printf("   For multi-node testing, use NET transport tests\n");
                }
                printf("   For test details, set: NCCL_DEBUG=INFO\n");
                printf("===========================\n\n");
            }
            return false;
        }

        if(world_rank == 0)
        {
            TEST_INFO("✅ ALL REQUIREMENTS MET");
            if(node_count == 1)
            {
                TEST_INFO("   Single-node execution (%d processes on 1 node)", world_size);
            }
            else
            {
                TEST_INFO("   Multi-node execution (%d processes on %d nodes)",
                          world_size,
                          node_count);
            }
            TEST_INFO("===========================");
            TEST_INFO("");
        }
    }
    else
    {
        // No node constraints specified (min_nodes=1, max_nodes=0 means unlimited)
        if(world_rank == 0)
        {
            TEST_INFO("Actual nodes:      (not checked - no constraints specified)");
            TEST_INFO("");
            TEST_INFO("✅ ALL REQUIREMENTS MET");
            TEST_INFO("===========================");
            TEST_INFO("");
        }
    }

    return true;
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
 * @note Idempotent - safe to call multiple times (returns immediately if
 * already created)
 * @note All ranks must call this function for proper synchronization
 */
ncclResult_t MPITestBase::createTestCommunicator()
{
    if(using_test_comm)
    {
        return ncclSuccess; // Already created
    }

    int world_rank = RCCLMPIEnvironment::world_rank;
    int world_size = RCCLMPIEnvironment::world_size;

    TEST_INFO("Creating test-specific communicator");

    // Get unique ID on rank 0 and broadcast error status to all ranks
    ncclUniqueId nccl_id;
    ncclResult_t init_result = ncclSuccess;

    if(world_rank == 0)
    {
        init_result = ncclGetUniqueId(&nccl_id);
        if(init_result != ncclSuccess)
        {
            TEST_WARN("Failed to get unique ID: %s", ncclGetErrorString(init_result));
        }
    }

    // Broadcast error status to all ranks so they can exit together
    int error_code = static_cast<int>(init_result);
    MPI_Bcast(&error_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // All ranks check the error status
    if(error_code != ncclSuccess)
    {
        if(world_rank != 0)
        {
            TEST_WARN("Rank 0 failed to get unique ID, aborting communicator creation");
        }
        return static_cast<ncclResult_t>(error_code);
    }

    // Broadcast unique ID
    int mpi_result = MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    if(mpi_result != MPI_SUCCESS)
    {
        TEST_WARN("MPI_Bcast failed");
        return ncclSystemError;
    }

    // Barrier before comm init
    MPI_Barrier(MPI_COMM_WORLD);

    // Use ncclGroupStart/End for proper initialization (critical for shared
    // memory setup)
    ncclResult_t result = ncclGroupStart();
    if(result != ncclSuccess)
    {
        TEST_WARN("Failed to start NCCL group: %s", ncclGetErrorString(result));
    }

    // Broadcast error status - all ranks must know if any rank failed
    error_code = static_cast<int>(result);
    MPI_Allreduce(MPI_IN_PLACE, &error_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(error_code != ncclSuccess)
    {
        if(world_rank == 0)
        {
            TEST_WARN("One or more ranks failed to start NCCL group, aborting");
        }
        return static_cast<ncclResult_t>(error_code);
    }

    // Initialize test communicator
    result = ncclCommInitRank(&test_comm, world_size, nccl_id, world_rank);
    if(result != ncclSuccess)
    {
        TEST_WARN("Failed to initialize test communicator: %s", ncclGetErrorString(result));
        ncclGroupEnd(); // Clean up group
    }

    // Broadcast error status - all ranks must know if any rank failed
    error_code = static_cast<int>(result);
    MPI_Allreduce(MPI_IN_PLACE, &error_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(error_code != ncclSuccess)
    {
        if(world_rank == 0)
        {
            TEST_WARN("One or more ranks failed to initialize communicator, aborting");
        }
        // Clean up group on all ranks
        ncclGroupEnd();
        if(test_comm)
        {
            ncclCommDestroy(test_comm);
            test_comm = nullptr;
        }
        return static_cast<ncclResult_t>(error_code);
    }

    result = ncclGroupEnd();
    if(result != ncclSuccess)
    {
        TEST_WARN("Failed to end NCCL group: %s", ncclGetErrorString(result));
        if(test_comm)
        {
            ncclCommDestroy(test_comm);
            test_comm = nullptr;
        }
    }

    // Broadcast error status - all ranks must know if any rank failed
    error_code = static_cast<int>(result);
    MPI_Allreduce(MPI_IN_PLACE, &error_code, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(error_code != ncclSuccess)
    {
        if(world_rank == 0)
        {
            TEST_WARN("One or more ranks failed to end NCCL group, aborting");
        }
        if(test_comm)
        {
            ncclCommDestroy(test_comm);
            test_comm = nullptr;
        }
        return static_cast<ncclResult_t>(error_code);
    }

    // Create test stream
    hipError_t hip_result = hipStreamCreate(&test_stream);
    if(hip_result != hipSuccess)
    {
        TEST_WARN("Failed to create test stream: %s", hipGetErrorString(hip_result));
        ncclCommDestroy(test_comm);
        test_comm = nullptr;
    }

    // Broadcast error status - all ranks must know if any rank failed
    int hip_error = static_cast<int>(hip_result);
    MPI_Allreduce(MPI_IN_PLACE, &hip_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if(hip_error != hipSuccess)
    {
        if(world_rank == 0)
        {
            TEST_WARN("One or more ranks failed to create HIP stream, aborting");
        }
        if(test_comm)
        {
            ncclCommDestroy(test_comm);
            test_comm = nullptr;
        }
        return ncclSystemError;
    }

    // Barrier after comm init - ensure all ranks have completed group init
    MPI_Barrier(MPI_COMM_WORLD);

    using_test_comm = true;

    TEST_INFO("Test-specific communicator created successfully");

    return ncclSuccess;
}

/**
 * @brief Retrieve the active NCCL communicator for test operations
 *
 * Returns the test-specific communicator that was created by
 * createTestCommunicator(). If no communicator has been created, this will
 * trigger a test failure.
 *
 * @return The active NCCL communicator handle, or nullptr with test failure
 *
 * @warning Always call createTestCommunicator() before calling this method
 */
ncclComm_t MPITestBase::getActiveCommunicator()
{
    if(!using_test_comm)
    {
        // No test communicator created - this is an error
        ADD_FAILURE() << "No test communicator created. Call createTestCommunicator() first.";
        return nullptr;
    }
    return test_comm;
}

/**
 * @brief Retrieve the active HIP stream for GPU operations
 *
 * Returns the test-specific HIP stream that was created by
 * createTestCommunicator(). If no stream has been created, this will trigger a
 * test failure.
 *
 * @return The active HIP stream handle, or nullptr with test failure
 *
 * @warning Always call createTestCommunicator() before calling this method
 */
hipStream_t MPITestBase::getActiveStream()
{
    if(!using_test_comm)
    {
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
void MPITestBase::cleanupTestCommunicator()
{
    if(!using_test_comm)
    {
        return; // Nothing to cleanup
    }

    // Synchronize before cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Destroy test stream
    if(test_stream)
    {
        hipStreamDestroy(test_stream);
        test_stream = nullptr;
    }

    // Destroy test communicator
    if(test_comm)
    {
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
 * Ensures proper cleanup of test resources by calling
 * cleanupTestCommunicator(). This guarantees that each test starts with a clean
 * slate and prevents resource leaks between tests.
 */
void MPITestBase::TearDown()
{
    cleanupTestCommunicator();
}

#endif // MPI_TESTS_ENABLED
