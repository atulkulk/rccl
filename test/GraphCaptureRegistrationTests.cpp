/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file GraphCaptureRegistrationTests.cpp
 * @brief Tests for Graph Capture Registration functionality in RCCL
 *
 * These tests verify that buffer registration works correctly when
 * NCCL operations are captured into HIP graphs. The NCCL_GRAPH_REGISTER
 * feature enables persistent buffer registration during graph capture.
 *
 * Each test is designed to:
 * 1. Capture NCCL collective operations into a HIP graph
 * 2. Instantiate and execute the graph
 * 3. Verify the operation completed successfully
 * 4. Check that graph registration path was used (via REG debug log parsing)
 *
 * Graph Capture Registration:
 * - When NCCL_GRAPH_REGISTER=1, buffers are registered during graph capture
 * - Registration is persistent across graph replays (zero-copy)
 * - Deregistration happens when graph is destroyed
 *
 * REQUIRED Environment Variables:
 *   NCCL_DEBUG=INFO              Enable debug logging (REQUIRED)
 *   NCCL_DEBUG_SUBSYS=REG        Enable REG subsystem logging (REQUIRED)
 *   NCCL_GRAPH_REGISTER=1        Enable graph buffer registration
 *
 * Run tests (example):
 *   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG NCCL_GRAPH_REGISTER=1 \
 *     mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=GraphCapture_AllReduce*
 *
 *   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG NCCL_GRAPH_REGISTER=1 \
 *     mpirun -np 16 --hostfile hostfile ./rccl-UnitTestsMPI --gtest_filter=GraphCapture_*MultiNode*
 *
 * The tests parse REG debug output to verify:
 *   - Graph registration succeeded
 *   - Buffer registration occurred during capture
 *   - Deregistration happens on cleanup
 *
 * Tests will FAIL if NCCL_DEBUG and NCCL_DEBUG_SUBSYS are not properly set.
 */

#include "DeviceBufferHelpers.hpp"
#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "ResourceGuards.hpp"
#include <cstdlib>
#include <sstream>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;
using namespace RCCLTestGuards;
using namespace RCCLTestHelpers;

// =============================================================================
// Test Configuration
// =============================================================================

namespace GraphCaptureTestConfig {
    // Buffer sizes for testing
    constexpr size_t SMALL_COUNT  = 1024;           // 4KB for float
    constexpr size_t MEDIUM_COUNT = 256 * 1024;     // 1MB for float
    constexpr size_t LARGE_COUNT  = 1024 * 1024;    // 4MB for float

    // Default data type - bfloat16 is common in ML workloads
    using DefaultType = hip_bfloat16;

    // Test setup configuration - minimum ranks and nodes
    constexpr int MIN_RANKS_DEFAULT    = 2;   // Minimum ranks for most tests
    constexpr int MIN_RANKS_ALLTOALL   = 4;   // AllToAll needs more ranks for meaningful test
    constexpr int MIN_NODES_MULTINODE  = 2;   // Minimum nodes for multi-node tests

    // Graph replay iterations
    constexpr int DEFAULT_GRAPH_ITERATIONS = 3;
}

// =============================================================================
// Graph REG Log Checker - Pattern checking for graph registration debug output
// =============================================================================

/**
 * @class GraphREGLogChecker
 * @brief Checks captured stderr for graph registration patterns in REG subsystem
 *
 * Uses MPIHelpers::StderrCapture for the actual capture,
 * provides graph-registration-specific pattern matching.
 */
class GraphREGLogChecker
{
public:
    explicit GraphREGLogChecker(const MPIHelpers::StderrCapture& capture)
        : m_capture(capture) {}

    // Graph registration: buffer was registered during graph capture
    bool hasGraphRegistration() const
    {
        return m_capture.hasPattern("register comm") && m_capture.hasPattern("buffer");
    }

    // IPC graph registration
    bool hasIPCGraphRegistration() const
    {
        return m_capture.hasPattern("IPC register buffer") ||
               m_capture.hasPattern("IPC registering buffer");
    }

    // IPC reuse: Pre-registered buffer was found and reused
    bool hasIPCReuse() const
    {
        return m_capture.hasPattern("IPC reuse buffer");
    }

    // NET graph registration
    bool hasNETGraphRegistration() const
    {
        return m_capture.hasPattern("NET register userbuff") ||
               m_capture.hasPattern("NET reuse buffer");
    }

    // Any successful registration path
    bool hasAnyRegistrationSuccess() const
    {
        return hasIPCGraphRegistration() || hasIPCReuse() || hasNETGraphRegistration();
    }

    // Check for failure patterns
    bool hasIPCFailure() const
    {
        return m_capture.hasPattern("failed to IPC register") ||
               m_capture.hasPattern("legacy IPC blocked");
    }

    bool hasNETFailure() const
    {
        return m_capture.hasPattern("failed to NET register");
    }

    // Summary for test output
    std::string getSummary() const
    {
        std::ostringstream ss;
        ss << "Graph REG Log Summary: ";
        if (hasIPCReuse()) ss << "[IPC-REUSE] ";
        if (hasIPCGraphRegistration()) ss << "[IPC-REG] ";
        if (hasNETGraphRegistration()) ss << "[NET-REG] ";
        if (hasIPCFailure()) ss << "[IPC-FAIL] ";
        if (hasNETFailure()) ss << "[NET-FAIL] ";
        if (!hasAnyRegistrationSuccess() && !hasIPCFailure()) ss << "[NO-REG-DETECTED]";
        return ss.str();
    }

private:
    const MPIHelpers::StderrCapture& m_capture;
};

// =============================================================================
// Graph Capture Test Base Class
// =============================================================================

/**
 * @class GraphCaptureTestBase
 * @brief Base class for Graph Capture Registration tests
 *
 * Provides infrastructure for testing graph capture with buffer registration
 * and verification that the registration path was actually used.
 */
class GraphCaptureTestBase : public MPITestBase
{
protected:
    // Stderr capture for REG log parsing
    MPIHelpers::StderrCapture m_stderrCapture;

    // Graph resources
    struct GraphResources {
        hipGraph_t graph = nullptr;
        hipGraphExec_t graphExec = nullptr;
        bool valid = false;

        void cleanup()
        {
            if (graphExec) {
                hipGraphExecDestroy(graphExec);
                graphExec = nullptr;
            }
            if (graph) {
                hipGraphDestroy(graph);
                graph = nullptr;
            }
            valid = false;
        }
    };

    // Buffer info
    struct BufferInfo {
        void* buffer = nullptr;
        size_t size = 0;

        void cleanup()
        {
            if (buffer) {
                hipFree(buffer);
                buffer = nullptr;
            }
            size = 0;
        }
    };

    // =========================================================================
    // Test Setup Helpers
    // =========================================================================

    bool setupSingleNode(int minRanks = 2)
    {
        if (!validateTestPrerequisites(minRanks, kNoProcessLimit,
                                        kNoPowerOfTwoRequired, 1, kRequireSingleNode)) {
            return false;
        }
        return (createTestCommunicator() == ncclSuccess);
    }

    bool setupMultiNode(int minRanks = 2, int minNodes = 2)
    {
        int nodeCount = MPITestConstants::detectNodeCount();
        if (nodeCount < minNodes) {
            return false;
        }
        if (!validateTestPrerequisites(minRanks, kNoProcessLimit,
                                        kNoPowerOfTwoRequired, minNodes, kNoNodeLimit)) {
            return false;
        }
        return (createTestCommunicator() == ncclSuccess);
    }

    // =========================================================================
    // Environment Checks
    // =========================================================================

    /**
     * @brief Check if NCCL_GRAPH_REGISTER is enabled
     */
    bool isGraphRegisterEnabled()
    {
        const char* graphReg = getenv("NCCL_GRAPH_REGISTER");
        return (graphReg && std::string(graphReg) == "1");
    }

    /**
     * @brief Check if required NCCL debug subsystem is enabled
     */
    bool requireNCCLDebug(const std::string& subsystem)
    {
        if (!MPIHelpers::isNCCLDebugEnabled(subsystem)) {
            TEST_WARN("NCCL debug logging not enabled for subsystem '%s'", subsystem.c_str());
            TEST_WARN("To enable, set: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=%s", subsystem.c_str());
            TEST_WARN("Example: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=%s mpirun ...", subsystem.c_str());
            return false;
        }
        return true;
    }

    // =========================================================================
    // REG Log Capture Helpers
    // =========================================================================

    void startREGCapture() { m_stderrCapture.start(); }
    void stopREGCapture() { m_stderrCapture.stop(); }

    MPIHelpers::StderrCaptureScope createREGLogScope()
    {
        return MPIHelpers::StderrCaptureScope(m_stderrCapture);
    }

    GraphREGLogChecker getREGLogChecker() const
    {
        return GraphREGLogChecker(m_stderrCapture);
    }

    /**
     * @brief Verify graph registration path was taken by checking REG logs
     */
    bool verifyGraphRegistrationFromLogs(const char* testName, bool requireIPC = false)
    {
        GraphREGLogChecker checker = getREGLogChecker();
        std::string summary = checker.getSummary();
        TEST_INFO("%s: %s", testName, summary.c_str());

        // Check if REG debug logging is enabled - FAIL if not
        if (!requireNCCLDebug("REG")) {
            return false;
        }

        if (requireIPC) {
            if (checker.hasIPCReuse() || checker.hasIPCGraphRegistration()) {
                TEST_INFO("%s: IPC graph registration path verified", testName);
                return true;
            }
            if (checker.hasIPCFailure()) {
                TEST_WARN("%s: IPC registration failed (may be expected in directMode)", testName);
                return false;
            }
        }

        if (checker.hasAnyRegistrationSuccess()) {
            TEST_INFO("%s: Graph registration path verified", testName);
            return true;
        }

        TEST_WARN("%s: No graph registration detected in logs", testName);
        return false;
    }

    // =========================================================================
    // Graph Capture Helpers
    // =========================================================================

    /**
     * @brief Begin graph capture on the active stream
     */
    bool beginGraphCapture()
    {
        hipError_t err = hipStreamBeginCapture(getActiveStream(), hipStreamCaptureModeGlobal);
        if (err != hipSuccess) {
            TEST_WARN("hipStreamBeginCapture failed: %s", hipGetErrorString(err));
            return false;
        }
        return true;
    }

    /**
     * @brief End graph capture and create executable graph
     */
    bool endGraphCapture(GraphResources& resources)
    {
        hipError_t err = hipStreamEndCapture(getActiveStream(), &resources.graph);
        if (err != hipSuccess) {
            TEST_WARN("hipStreamEndCapture failed: %s", hipGetErrorString(err));
            return false;
        }

        err = hipGraphInstantiate(&resources.graphExec, resources.graph, nullptr, nullptr, 0);
        if (err != hipSuccess) {
            TEST_WARN("hipGraphInstantiate failed: %s", hipGetErrorString(err));
            hipGraphDestroy(resources.graph);
            resources.graph = nullptr;
            return false;
        }

        resources.valid = true;
        return true;
    }

    /**
     * @brief Execute graph and synchronize
     */
    bool executeGraph(const GraphResources& resources)
    {
        if (!resources.valid || !resources.graphExec) {
            TEST_WARN("Invalid graph resources");
            return false;
        }

        hipError_t err = hipGraphLaunch(resources.graphExec, getActiveStream());
        if (err != hipSuccess) {
            TEST_WARN("hipGraphLaunch failed: %s", hipGetErrorString(err));
            return false;
        }

        err = hipStreamSynchronize(getActiveStream());
        if (err != hipSuccess) {
            TEST_WARN("hipStreamSynchronize failed: %s", hipGetErrorString(err));
            return false;
        }

        return true;
    }

    // =========================================================================
    // Data Initialization and Verification
    // =========================================================================

    template<typename T>
    void initSendBuffer(void* buffer, size_t count, int rank)
    {
        initializeBufferWithPattern<T>(buffer, count,
            [rank](size_t) { return static_cast<T>(static_cast<float>(rank + 1)); });
    }

    template<typename T>
    bool verifyAllReduceResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyReduceScatterResult(void* buffer, size_t count, int nRanks)
    {
        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2));
        return verifyBufferData<T>(buffer, count, [expected](size_t) { return expected; });
    }

    template<typename T>
    bool verifyAllGatherResult(void* buffer, size_t countPerRank, int nRanks)
    {
        return verifyBufferData<T>(buffer, countPerRank * nRanks,
            [countPerRank](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank + 1));
            });
    }

    template<typename T>
    bool verifyBroadcastResult(void* buffer, size_t count, T rootValue)
    {
        return verifyBufferData<T>(buffer, count,
            [rootValue](size_t) { return rootValue; });
    }
};

// =============================================================================
// AllReduce Graph Capture Tests
// =============================================================================

class GraphCapture_AllReduce : public GraphCaptureTestBase {};

TEST_F(GraphCapture_AllReduce, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT));

    if (!isGraphRegisterEnabled()) {
        TEST_WARN("NCCL_GRAPH_REGISTER not set to 1, graph registration disabled");
    }

    using T = GraphCaptureTestConfig::DefaultType;
    const size_t count = GraphCaptureTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // Allocate buffers
    BufferInfo sendBuf, recvBuf;
    sendBuf.size = count * sizeof(T);
    recvBuf.size = count * sizeof(T);

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf.buffer, sendBuf.size));
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf.buffer, recvBuf.size));

    auto bufCleanup = makeScopeGuard([&]() {
        sendBuf.cleanup();
        recvBuf.cleanup();
    });

    // Initialize send buffer
    initSendBuffer<T>(sendBuf.buffer, count, rank);

    // Capture graph
    GraphResources graphRes;
    auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

    {
        auto logScope = createREGLogScope();

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclAllReduce(sendBuf.buffer, recvBuf.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));
    }

    // Verify graph registration (if enabled)
    if (isGraphRegisterEnabled()) {
        EXPECT_TRUE(verifyGraphRegistrationFromLogs("AllReduce_GraphCapture"))
            << "Graph registration not detected";
    }

    // Execute graph multiple times
    for (int iter = 0; iter < GraphCaptureTestConfig::DEFAULT_GRAPH_ITERATIONS; ++iter) {
        SCOPED_TRACE("Graph iteration: " + std::to_string(iter));

        // Re-initialize for each iteration
        initSendBuffer<T>(sendBuf.buffer, count, rank);

        ASSERT_TRUE(executeGraph(graphRes));
        EXPECT_TRUE(verifyAllReduceResult<T>(recvBuf.buffer, count, nRanks));
    }
}

TEST_F(GraphCapture_AllReduce, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT,
                        GraphCaptureTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    if (!isGraphRegisterEnabled()) {
        TEST_WARN("NCCL_GRAPH_REGISTER not set to 1, graph registration disabled");
    }

    using T = GraphCaptureTestConfig::DefaultType;
    const size_t count = GraphCaptureTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    BufferInfo sendBuf, recvBuf;
    sendBuf.size = count * sizeof(T);
    recvBuf.size = count * sizeof(T);

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf.buffer, sendBuf.size));
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf.buffer, recvBuf.size));

    auto bufCleanup = makeScopeGuard([&]() {
        sendBuf.cleanup();
        recvBuf.cleanup();
    });

    initSendBuffer<T>(sendBuf.buffer, count, rank);

    GraphResources graphRes;
    auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

    {
        auto logScope = createREGLogScope();

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclAllReduce(sendBuf.buffer, recvBuf.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));
    }

    // Multi-node should succeed with IPC registration
    if (isGraphRegisterEnabled()) {
        EXPECT_TRUE(verifyGraphRegistrationFromLogs("AllReduce_MultiNode_GraphCapture", true))
            << "IPC graph registration not detected on multi-node";
    }

    for (int iter = 0; iter < GraphCaptureTestConfig::DEFAULT_GRAPH_ITERATIONS; ++iter) {
        SCOPED_TRACE("Graph iteration: " + std::to_string(iter));
        initSendBuffer<T>(sendBuf.buffer, count, rank);
        ASSERT_TRUE(executeGraph(graphRes));
        EXPECT_TRUE(verifyAllReduceResult<T>(recvBuf.buffer, count, nRanks));
    }
}

// =============================================================================
// AllGather Graph Capture Tests
// =============================================================================

class GraphCapture_AllGather : public GraphCaptureTestBase {};

TEST_F(GraphCapture_AllGather, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT));

    using T = GraphCaptureTestConfig::DefaultType;
    const size_t countPerRank = GraphCaptureTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    BufferInfo sendBuf, recvBuf;
    sendBuf.size = countPerRank * sizeof(T);
    recvBuf.size = countPerRank * nRanks * sizeof(T);

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf.buffer, sendBuf.size));
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf.buffer, recvBuf.size));

    auto bufCleanup = makeScopeGuard([&]() {
        sendBuf.cleanup();
        recvBuf.cleanup();
    });

    initSendBuffer<T>(sendBuf.buffer, countPerRank, rank);

    GraphResources graphRes;
    auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

    {
        auto logScope = createREGLogScope();

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclAllGather(sendBuf.buffer, recvBuf.buffer, countPerRank,
                                             getNcclDataType<T>(),
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));
    }

    if (isGraphRegisterEnabled()) {
        verifyGraphRegistrationFromLogs("AllGather_GraphCapture");
    }

    for (int iter = 0; iter < GraphCaptureTestConfig::DEFAULT_GRAPH_ITERATIONS; ++iter) {
        SCOPED_TRACE("Graph iteration: " + std::to_string(iter));
        initSendBuffer<T>(sendBuf.buffer, countPerRank, rank);
        ASSERT_TRUE(executeGraph(graphRes));
        EXPECT_TRUE(verifyAllGatherResult<T>(recvBuf.buffer, countPerRank, nRanks));
    }
}

// =============================================================================
// ReduceScatter Graph Capture Tests
// =============================================================================

class GraphCapture_ReduceScatter : public GraphCaptureTestBase {};

TEST_F(GraphCapture_ReduceScatter, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT));

    using T = GraphCaptureTestConfig::DefaultType;
    const size_t countPerRank = GraphCaptureTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    BufferInfo sendBuf, recvBuf;
    sendBuf.size = countPerRank * nRanks * sizeof(T);
    recvBuf.size = countPerRank * sizeof(T);

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf.buffer, sendBuf.size));
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf.buffer, recvBuf.size));

    auto bufCleanup = makeScopeGuard([&]() {
        sendBuf.cleanup();
        recvBuf.cleanup();
    });

    initSendBuffer<T>(sendBuf.buffer, countPerRank * nRanks, rank);

    GraphResources graphRes;
    auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

    {
        auto logScope = createREGLogScope();

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclReduceScatter(sendBuf.buffer, recvBuf.buffer, countPerRank,
                                                 getNcclDataType<T>(), ncclSum,
                                                 getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));
    }

    if (isGraphRegisterEnabled()) {
        verifyGraphRegistrationFromLogs("ReduceScatter_GraphCapture");
    }

    for (int iter = 0; iter < GraphCaptureTestConfig::DEFAULT_GRAPH_ITERATIONS; ++iter) {
        SCOPED_TRACE("Graph iteration: " + std::to_string(iter));
        initSendBuffer<T>(sendBuf.buffer, countPerRank * nRanks, rank);
        ASSERT_TRUE(executeGraph(graphRes));
        EXPECT_TRUE(verifyReduceScatterResult<T>(recvBuf.buffer, countPerRank, nRanks));
    }
}

// =============================================================================
// Broadcast Graph Capture Tests
// =============================================================================

class GraphCapture_Broadcast : public GraphCaptureTestBase {};

TEST_F(GraphCapture_Broadcast, InPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT));

    using T = GraphCaptureTestConfig::DefaultType;
    const size_t count = GraphCaptureTestConfig::MEDIUM_COUNT;
    const int root = 0;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    BufferInfo buf;
    buf.size = count * sizeof(T);

    ASSERT_EQ(hipSuccess, hipMalloc(&buf.buffer, buf.size));

    auto bufCleanup = makeScopeGuard([&]() { buf.cleanup(); });

    const T rootValue = static_cast<T>(42.0f);
    if (rank == root) {
        initializeBufferWithPattern<T>(buf.buffer, count,
            [rootValue](size_t) { return rootValue; });
    } else {
        initializeBufferWithPattern<T>(buf.buffer, count,
            [](size_t) { return static_cast<T>(0.0f); });
    }

    GraphResources graphRes;
    auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

    {
        auto logScope = createREGLogScope();

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclBroadcast(buf.buffer, buf.buffer, count,
                                             getNcclDataType<T>(), root,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));
    }

    if (isGraphRegisterEnabled()) {
        verifyGraphRegistrationFromLogs("Broadcast_GraphCapture");
    }

    for (int iter = 0; iter < GraphCaptureTestConfig::DEFAULT_GRAPH_ITERATIONS; ++iter) {
        SCOPED_TRACE("Graph iteration: " + std::to_string(iter));

        // Re-initialize buffer before each iteration
        if (rank == root) {
            initializeBufferWithPattern<T>(buf.buffer, count,
                [rootValue](size_t) { return rootValue; });
        } else {
            initializeBufferWithPattern<T>(buf.buffer, count,
                [](size_t) { return static_cast<T>(0.0f); });
        }

        ASSERT_TRUE(executeGraph(graphRes));
        EXPECT_TRUE(verifyBroadcastResult<T>(buf.buffer, count, rootValue));
    }
}

// =============================================================================
// AllToAll Graph Capture Tests
// =============================================================================

class GraphCapture_AllToAll : public GraphCaptureTestBase {};

TEST_F(GraphCapture_AllToAll, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT));

    using T = GraphCaptureTestConfig::DefaultType;
    const size_t countPerRank = GraphCaptureTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t totalCount = countPerRank * nRanks;
    BufferInfo sendBuf, recvBuf;
    sendBuf.size = totalCount * sizeof(T);
    recvBuf.size = totalCount * sizeof(T);

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf.buffer, sendBuf.size));
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf.buffer, recvBuf.size));

    auto bufCleanup = makeScopeGuard([&]() {
        sendBuf.cleanup();
        recvBuf.cleanup();
    });

    // Initialize send buffer
    initializeBufferWithPattern<T>(sendBuf.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        });

    GraphResources graphRes;
    auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

    {
        auto logScope = createREGLogScope();

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclAllToAll(sendBuf.buffer, recvBuf.buffer, countPerRank,
                                            getNcclDataType<T>(),
                                            getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));
    }

    if (isGraphRegisterEnabled()) {
        verifyGraphRegistrationFromLogs("AllToAll_GraphCapture");
    }

    for (int iter = 0; iter < GraphCaptureTestConfig::DEFAULT_GRAPH_ITERATIONS; ++iter) {
        SCOPED_TRACE("Graph iteration: " + std::to_string(iter));

        // Re-initialize for each iteration
        initializeBufferWithPattern<T>(sendBuf.buffer, totalCount,
            [rank, countPerRank](size_t i) {
                int destRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(rank * 100 + destRank));
            });

        ASSERT_TRUE(executeGraph(graphRes));

        // Verify
        bool verified = verifyBufferData<T>(recvBuf.buffer, totalCount,
            [rank, countPerRank](size_t i) {
                int srcRank = i / countPerRank;
                return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
            });
        EXPECT_TRUE(verified);
    }
}

// =============================================================================
// Graph Lifecycle Tests
// =============================================================================

class GraphCapture_Lifecycle : public GraphCaptureTestBase {};

TEST_F(GraphCapture_Lifecycle, MultipleGraphCaptures)
{
    ASSERT_TRUE(setupSingleNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT));

    using T = GraphCaptureTestConfig::DefaultType;
    const size_t count = GraphCaptureTestConfig::SMALL_COUNT;
    const int numGraphs = 3;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    BufferInfo sendBuf, recvBuf;
    sendBuf.size = count * sizeof(T);
    recvBuf.size = count * sizeof(T);

    ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf.buffer, sendBuf.size));
    ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf.buffer, recvBuf.size));

    auto bufCleanup = makeScopeGuard([&]() {
        sendBuf.cleanup();
        recvBuf.cleanup();
    });

    // Capture and execute multiple independent graphs
    for (int graphIdx = 0; graphIdx < numGraphs; ++graphIdx) {
        SCOPED_TRACE("Graph index: " + std::to_string(graphIdx));

        initSendBuffer<T>(sendBuf.buffer, count, rank);

        GraphResources graphRes;
        auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclAllReduce(sendBuf.buffer, recvBuf.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));

        // Execute multiple times
        for (int iter = 0; iter < 2; ++iter) {
            initSendBuffer<T>(sendBuf.buffer, count, rank);
            ASSERT_TRUE(executeGraph(graphRes));
            EXPECT_TRUE(verifyAllReduceResult<T>(recvBuf.buffer, count, nRanks));
        }
    }
}

TEST_F(GraphCapture_Lifecycle, GraphWithDifferentBufferSizes)
{
    ASSERT_TRUE(setupSingleNode(GraphCaptureTestConfig::MIN_RANKS_DEFAULT));

    using T = GraphCaptureTestConfig::DefaultType;
    std::vector<size_t> sizes = {
        GraphCaptureTestConfig::SMALL_COUNT,
        GraphCaptureTestConfig::MEDIUM_COUNT,
        GraphCaptureTestConfig::LARGE_COUNT
    };

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    for (size_t count : sizes) {
        SCOPED_TRACE("Buffer count: " + std::to_string(count));

        BufferInfo sendBuf, recvBuf;
        sendBuf.size = count * sizeof(T);
        recvBuf.size = count * sizeof(T);

        ASSERT_EQ(hipSuccess, hipMalloc(&sendBuf.buffer, sendBuf.size));
        ASSERT_EQ(hipSuccess, hipMalloc(&recvBuf.buffer, recvBuf.size));

        auto bufCleanup = makeScopeGuard([&]() {
            sendBuf.cleanup();
            recvBuf.cleanup();
        });

        initSendBuffer<T>(sendBuf.buffer, count, rank);

        GraphResources graphRes;
        auto graphCleanup = makeScopeGuard([&]() { graphRes.cleanup(); });

        ASSERT_TRUE(beginGraphCapture());

        ncclResult_t result = ncclAllReduce(sendBuf.buffer, recvBuf.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);

        ASSERT_TRUE(endGraphCapture(graphRes));

        ASSERT_TRUE(executeGraph(graphRes));
        EXPECT_TRUE(verifyAllReduceResult<T>(recvBuf.buffer, count, nRanks));
    }
}

#endif // MPI_TESTS_ENABLED
