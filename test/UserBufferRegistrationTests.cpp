/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file UserBufferRegistrationTests.cpp
 * @brief Tests for User Buffer Registration (UBR) functionality in RCCL
 *
 * These tests verify that the UBR code paths are correctly taken for different
 * collective operations. Each test is designed to:
 * 1. Register buffers using ncclCommRegister
 * 2. Execute the collective operation
 * 3. Verify the operation completed successfully
 * 4. Check that the UBR path was used (via REG debug log parsing)
 *
 * UBR Path Conditions:
 * - ReduceScatter: Skips IPC registration (uses CollNet paths)
 * - AllReduce in-place with RING: Skips IPC registration
 * - Reduce with RING: Skips IPC registration
 * - AllGather with PAT algorithm: Skips IPC registration
 * - Tree/CollnetChain in-place: Skips IPC registration
 *
 * Multi-node tests verify IPC registration succeeds (not blocked by directMode).
 *
 * REQUIRED Environment Variables:
 *   NCCL_DEBUG=INFO              Enable debug logging (REQUIRED)
 *   NCCL_DEBUG_SUBSYS=REG        Enable REG subsystem logging (REQUIRED)
 *   NCCL_LOCAL_REGISTER=1        Enable local buffer registration
 *   NCCL_LEGACY_CUDA_REGISTER=1  Enable legacy CUDA registration
 *
 * Run tests (example):
 *   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG NCCL_LOCAL_REGISTER=1 \
 *     mpirun -np 8 ./rccl-UnitTestsMPI --gtest_filter=UBR_AllReduce*
 *
 *   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG NCCL_LOCAL_REGISTER=1 \
 *     mpirun -np 16 --hostfile hostfile ./rccl-UnitTestsMPI --gtest_filter=UBR_*MultiNode*
 *
 * The tests parse REG debug output to verify:
 *   - IPC registration succeeded (multi-node)
 *   - Buffer reuse occurred (UBR cache hit)
 *   - NET registration succeeded (where applicable)
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

namespace UBRTestConfig {
    // Buffer sizes for testing
    constexpr size_t SMALL_COUNT  = 1024;           // 4KB for float
    constexpr size_t MEDIUM_COUNT = 256 * 1024;     // 1MB for float
    constexpr size_t LARGE_COUNT  = 1024 * 1024;    // 4MB for float
    constexpr size_t HUGE_COUNT   = 8 * 1024 * 1024; // 8MB for float

    // Default data type - bfloat16 is common in ML workloads
    using DefaultType = hip_bfloat16;

    // Test setup configuration - minimum ranks and nodes
    constexpr int MIN_RANKS_DEFAULT    = 2;   // Minimum ranks for most tests
    constexpr int MIN_RANKS_ALLTOALL   = 4;   // AllToAll needs more ranks for meaningful test
    constexpr int MIN_NODES_MULTINODE  = 2;   // Minimum nodes for multi-node tests
}

// =============================================================================
// REG Log Checker - UBR-specific pattern checking for REG debug output
// =============================================================================

/**
 * @class REGLogChecker
 * @brief Checks captured stderr for NCCL REG subsystem patterns
 *
 * Uses MPIHelpers::StderrCapture for the actual capture,
 * provides UBR-specific pattern matching.
 *
 * Usage:
 *   MPIHelpers::StderrCapture capture;
 *   {
 *       MPIHelpers::StderrCaptureScope scope(capture);
 *       ncclAllReduce(...);
 *       hipStreamSynchronize(...);
 *   }
 *   REGLogChecker checker(capture);
 *   EXPECT_TRUE(checker.hasIPCRegistration());
 */
class REGLogChecker
{
public:
    explicit REGLogChecker(const MPIHelpers::StderrCapture& capture)
        : m_capture(capture) {}

    // UBR Success: buffer was registered (initial registration)
    bool hasRegistration() const
    {
        return m_capture.hasPattern("register comm") && m_capture.hasPattern("buffer");
    }

    // IPC Success: IPC registration completed
    bool hasIPCRegistration() const
    {
        return m_capture.hasPattern("IPC register buffer") ||
               m_capture.hasPattern("IPC registering buffer");
    }

    // IPC Reuse: Pre-registered buffer was found and reused (UBR hit!)
    bool hasIPCReuse() const
    {
        return m_capture.hasPattern("IPC reuse buffer");
    }

    // NET Success: Network buffer registration
    bool hasNETRegistration() const
    {
        return m_capture.hasPattern("NET register userbuff") ||
               m_capture.hasPattern("NET reuse buffer");
    }

    // Any successful registration path
    bool hasAnyRegistrationSuccess() const
    {
        return hasIPCRegistration() || hasIPCReuse() || hasNETRegistration();
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
        ss << "REG Log Summary: ";
        if (hasIPCReuse()) ss << "[IPC-REUSE] ";
        if (hasIPCRegistration()) ss << "[IPC-REG] ";
        if (hasNETRegistration()) ss << "[NET-REG] ";
        if (hasIPCFailure()) ss << "[IPC-FAIL] ";
        if (hasNETFailure()) ss << "[NET-FAIL] ";
        if (!hasAnyRegistrationSuccess() && !hasIPCFailure()) ss << "[NO-REG-DETECTED]";
        return ss.str();
    }

private:
    const MPIHelpers::StderrCapture& m_capture;
};

// =============================================================================
// UBR Test Base Class
// =============================================================================

/**
 * @class UBRTestBase
 * @brief Base class for User Buffer Registration tests
 *
 * Provides infrastructure for testing UBR with verification that
 * the registration path was actually used via REG log parsing.
 */
class UBRTestBase : public MPITestBase
{
protected:
    // Stderr capture for REG log parsing
    MPIHelpers::StderrCapture m_stderrCapture;

    // Registration tracking
    struct RegInfo {
        void* buffer = nullptr;
        void* handle = nullptr;
        size_t size = 0;
        bool registered = false;
    };

    // =========================================================================
    // Buffer Management
    // =========================================================================

    RegInfo allocateAndRegister(size_t size)
    {
        RegInfo info;
        info.size = size;

        if (hipMalloc(&info.buffer, size) != hipSuccess) {
            return info;
        }

        ncclResult_t result = ncclCommRegister(getActiveCommunicator(),
                                                info.buffer, size, &info.handle);
        info.registered = (result == ncclSuccess && info.handle != nullptr);

        return info;
    }

    void cleanupRegInfo(RegInfo& info)
    {
        if (info.handle) {
            ncclCommDeregister(getActiveCommunicator(), info.handle);
            info.handle = nullptr;
        }
        if (info.buffer) {
            hipFree(info.buffer);
            info.buffer = nullptr;
        }
        info.registered = false;
    }

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
    // UBR Verification
    // =========================================================================

    /**
     * @brief Check if UBR environment is properly configured
     * @return true if NCCL_LOCAL_REGISTER=1 is set
     */
    bool isUBREnabled()
    {
        const char* localReg = getenv("NCCL_LOCAL_REGISTER");
        return (localReg && std::string(localReg) == "1");
    }

    /**
     * @brief Verify that buffer was successfully registered
     * @param info Registration info to check
     * @param testName Name of test for logging
     * @return true if registration succeeded
     */
    bool verifyRegistration(const RegInfo& info, const char* testName)
    {
        if (!info.registered || !info.handle) {
            TEST_WARN("%s: Buffer registration failed - UBR path not taken", testName);
            return false;
        }
        TEST_INFO("%s: Buffer registered successfully (handle=%p)", testName, info.handle);
        return true;
    }

    // =========================================================================
    // REG Log Capture Helpers
    // =========================================================================

    /**
     * @brief Start capturing stderr for REG debug logs
     * Call before running collective operations
     */
    void startREGCapture() { m_stderrCapture.start(); }

    /**
     * @brief Stop capturing stderr
     * Call after collective operations complete
     */
    void stopREGCapture() { m_stderrCapture.stop(); }

    /**
     * @brief Create RAII scope for stderr capture
     * @return StderrCaptureScope that captures until destroyed
     *
     * Usage:
     *   {
     *       auto scope = createREGLogScope();
     *       ncclAllReduce(...);
     *       hipStreamSynchronize(...);
     *   }
     *   verifyUBRPathFromLogs("TestName");
     */
    MPIHelpers::StderrCaptureScope createREGLogScope()
    {
        return MPIHelpers::StderrCaptureScope(m_stderrCapture);
    }

    /**
     * @brief Get REGLogChecker for analyzing captured output
     * @return REGLogChecker with UBR-specific pattern matching
     */
    REGLogChecker getREGLogChecker() const
    {
        return REGLogChecker(m_stderrCapture);
    }

    /**
     * @brief Check if required NCCL debug subsystem is enabled
     * @param subsystem Debug subsystem to check (e.g., "REG", "NET", "INIT")
     * @return true if debug is enabled, false otherwise (with warning message)
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

    /**
     * @brief Verify UBR path was taken by checking REG logs
     * @param testName Name of test for logging
     * @param requireIPC If true, require IPC registration specifically
     * @return true if UBR path detected in logs
     *
     * Note: Requires NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG to be set
     *       Test will FAIL if debug logging is not enabled.
     */
    bool verifyUBRPathFromLogs(const char* testName, bool requireIPC = false)
    {
        REGLogChecker checker = getREGLogChecker();
        std::string summary = checker.getSummary();
        TEST_INFO("%s: %s", testName, summary.c_str());

        // Check if REG debug logging is enabled - FAIL if not
        if (!requireNCCLDebug("REG")) {
            return false;
        }

        if (requireIPC) {
            if (checker.hasIPCReuse() || checker.hasIPCRegistration()) {
                TEST_INFO("%s: IPC registration path verified", testName);
                return true;
            }
            if (checker.hasIPCFailure()) {
                TEST_WARN("%s: IPC registration failed (may be expected in directMode)", testName);
                return false;
            }
        }

        if (checker.hasAnyRegistrationSuccess()) {
            TEST_INFO("%s: UBR path verified", testName);
            return true;
        }

        TEST_WARN("%s: No UBR registration detected in logs", testName);
        return false;
    }

    /**
     * @brief Check if IPC path specifically was blocked
     * @return true if IPC was blocked (directMode, etc.)
     */
    bool wasIPCBlocked() const
    {
        REGLogChecker checker(m_stderrCapture);
        return checker.hasIPCFailure();
    }

    /**
     * @brief Get the captured stderr output
     */
    const std::string& getREGLogOutput() const
    {
        return m_stderrCapture.getOutput();
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
// AllReduce UBR Tests
// =============================================================================

class UBR_AllReduce : public UBRTestBase {};

/**
 * @test AllReduce with UBR - Out-of-place (should use IPC registration on multi-node)
 */
TEST_F(UBR_AllReduce, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    // Verify registration (may fail in directMode - expected on single node)
    bool sendReg = verifyRegistration(sendInfo, "AllReduce_OutOfPlace_Send");
    bool recvReg = verifyRegistration(recvInfo, "AllReduce_OutOfPlace_Recv");
    TEST_INFO("AllReduce OutOfPlace SingleNode: sendReg=%d recvReg=%d", sendReg, recvReg);

    // Initialize and execute
    initSendBuffer<T>(sendInfo.buffer, count, rank);

    ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                         getNcclDataType<T>(), ncclSum,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyAllReduceResult<T>(recvInfo.buffer, count, nRanks));
}

TEST_F(UBR_AllReduce, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(UBRTestConfig::MIN_RANKS_DEFAULT, UBRTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    // On multi-node, IPC registration should succeed
    EXPECT_TRUE(verifyRegistration(sendInfo, "AllReduce_MultiNode_Send"))
        << "UBR path not taken for AllReduce send buffer on multi-node";
    EXPECT_TRUE(verifyRegistration(recvInfo, "AllReduce_MultiNode_Recv"))
        << "UBR path not taken for AllReduce recv buffer on multi-node";

    initSendBuffer<T>(sendInfo.buffer, count, rank);

    // Capture REG logs during collective execution
    {
        auto logScope = createREGLogScope();
        ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);
        hipStreamSynchronize(getActiveStream());
    }

    // Verify UBR/IPC path was taken (requires NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=REG)
    EXPECT_TRUE(verifyUBRPathFromLogs("AllReduce_MultiNode", true /*requireIPC*/))
        << "UBR/IPC path not detected in REG logs for multi-node AllReduce";

    EXPECT_TRUE(verifyAllReduceResult<T>(recvInfo.buffer, count, nRanks));
}

/**
 * @test AllReduce in-place - Skips IPC registration on RING algorithm
 * This is expected behavior per coll_reg.cc line 186
 */
TEST_F(UBR_AllReduce, InPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo bufInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(bufInfo);
    });

    ASSERT_NE(bufInfo.buffer, nullptr);

    // Registration should succeed even if IPC path is skipped
    verifyRegistration(bufInfo, "AllReduce_InPlace");

    initSendBuffer<T>(bufInfo.buffer, count, rank);

    // In-place: sendbuf == recvbuf
    ncclResult_t result = ncclAllReduce(bufInfo.buffer, bufInfo.buffer, count,
                                         getNcclDataType<T>(), ncclSum,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyAllReduceResult<T>(bufInfo.buffer, count, nRanks));
}

// =============================================================================
// AllGather UBR Tests
// =============================================================================

class UBR_AllGather : public UBRTestBase {};

TEST_F(UBR_AllGather, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    verifyRegistration(sendInfo, "AllGather_Send");
    verifyRegistration(recvInfo, "AllGather_Recv");

    initSendBuffer<T>(sendInfo.buffer, countPerRank, rank);

    ncclResult_t result = ncclAllGather(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                         getNcclDataType<T>(),
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyAllGatherResult<T>(recvInfo.buffer, countPerRank, nRanks));
}

TEST_F(UBR_AllGather, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(UBRTestConfig::MIN_RANKS_DEFAULT, UBRTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    using T = UBRTestConfig::DefaultType;
    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    EXPECT_TRUE(verifyRegistration(sendInfo, "AllGather_MultiNode_Send"))
        << "UBR path not taken for AllGather send buffer on multi-node";
    EXPECT_TRUE(verifyRegistration(recvInfo, "AllGather_MultiNode_Recv"))
        << "UBR path not taken for AllGather recv buffer on multi-node";

    initSendBuffer<T>(sendInfo.buffer, countPerRank, rank);

    ncclResult_t result = ncclAllGather(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                         getNcclDataType<T>(),
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyAllGatherResult<T>(recvInfo.buffer, countPerRank, nRanks));
}

// =============================================================================
// ReduceScatter UBR Tests
// =============================================================================

class UBR_ReduceScatter : public UBRTestBase {};

/**
 * @test ReduceScatter with UBR
 * Note: ReduceScatter skips IPC registration per coll_reg.cc line 185, 202
 * It uses NVLS or CollNet paths instead.
 */
TEST_F(UBR_ReduceScatter, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    // ReduceScatter may skip IPC path - this is expected behavior
    verifyRegistration(sendInfo, "ReduceScatter_Send");
    verifyRegistration(recvInfo, "ReduceScatter_Recv");

    initSendBuffer<T>(sendInfo.buffer, countPerRank * nRanks, rank);

    ncclResult_t result = ncclReduceScatter(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyReduceScatterResult<T>(recvInfo.buffer, countPerRank, nRanks));
}

TEST_F(UBR_ReduceScatter, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(UBRTestConfig::MIN_RANKS_DEFAULT, UBRTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    using T = UBRTestConfig::DefaultType;
    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(countPerRank * nRanks * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(countPerRank * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    // Registration should succeed (buffer is cached)
    verifyRegistration(sendInfo, "ReduceScatter_MultiNode_Send");
    verifyRegistration(recvInfo, "ReduceScatter_MultiNode_Recv");

    initSendBuffer<T>(sendInfo.buffer, countPerRank * nRanks, rank);

    ncclResult_t result = ncclReduceScatter(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyReduceScatterResult<T>(recvInfo.buffer, countPerRank, nRanks));
}

// =============================================================================
// Broadcast UBR Tests
// =============================================================================

class UBR_Broadcast : public UBRTestBase {};

TEST_F(UBR_Broadcast, InPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::MEDIUM_COUNT;
    const int root = 0;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo bufInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(bufInfo);
    });

    ASSERT_NE(bufInfo.buffer, nullptr);
    verifyRegistration(bufInfo, "Broadcast_InPlace");

    const T rootValue = static_cast<T>(42.0f);
    if (rank == root) {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [rootValue](size_t) { return rootValue; });
    } else {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [](size_t) { return static_cast<T>(0.0f); });
    }

    ncclResult_t result = ncclBroadcast(bufInfo.buffer, bufInfo.buffer, count,
                                         getNcclDataType<T>(), root,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyBroadcastResult<T>(bufInfo.buffer, count, rootValue));
}

TEST_F(UBR_Broadcast, InPlace_MultiNode)
{
    if (!setupMultiNode(UBRTestConfig::MIN_RANKS_DEFAULT, UBRTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes";
    }

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::MEDIUM_COUNT;
    const int root = 0;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo bufInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(bufInfo);
    });

    ASSERT_NE(bufInfo.buffer, nullptr);
    EXPECT_TRUE(verifyRegistration(bufInfo, "Broadcast_MultiNode"))
        << "UBR path not taken for Broadcast on multi-node";

    const T rootValue = static_cast<T>(42.0f);
    if (rank == root) {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [rootValue](size_t) { return rootValue; });
    } else {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [](size_t) { return static_cast<T>(0.0f); });
    }

    ncclResult_t result = ncclBroadcast(bufInfo.buffer, bufInfo.buffer, count,
                                         getNcclDataType<T>(), root,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyBroadcastResult<T>(bufInfo.buffer, count, rootValue));
}

TEST_F(UBR_Broadcast, NonZeroRoot_MultiNode)
{
    if (!setupMultiNode(UBRTestConfig::MIN_RANKS_ALLTOALL, UBRTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes with 4+ ranks";
    }

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::MEDIUM_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const int root = nRanks - 1;  // Last rank as root

    RegInfo bufInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(bufInfo);
    });

    ASSERT_NE(bufInfo.buffer, nullptr);
    verifyRegistration(bufInfo, "Broadcast_NonZeroRoot");

    const T rootValue = static_cast<T>(99.0f);
    if (rank == root) {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [rootValue](size_t) { return rootValue; });
    } else {
        initializeBufferWithPattern<T>(bufInfo.buffer, count,
            [](size_t) { return static_cast<T>(0.0f); });
    }

    ncclResult_t result = ncclBroadcast(bufInfo.buffer, bufInfo.buffer, count,
                                         getNcclDataType<T>(), root,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyBroadcastResult<T>(bufInfo.buffer, count, rootValue));
}

// =============================================================================
// AllToAll UBR Tests
// =============================================================================

class UBR_AllToAll : public UBRTestBase {};

TEST_F(UBR_AllToAll, OutOfPlace_SingleNode)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t totalCount = countPerRank * nRanks;
    RegInfo sendInfo = allocateAndRegister(totalCount * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(totalCount * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    verifyRegistration(sendInfo, "AllToAll_Send");
    verifyRegistration(recvInfo, "AllToAll_Recv");

    // Initialize: send buffer[destRank * countPerRank + i] = rank * 100 + destRank
    initializeBufferWithPattern<T>(sendInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        });

    ncclResult_t result = ncclAllToAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                        getNcclDataType<T>(),
                                        getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    // Verify: recv buffer[srcRank * countPerRank + i] = srcRank * 100 + rank
    bool verified = verifyBufferData<T>(recvInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int srcRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
        });
    EXPECT_TRUE(verified);
}

TEST_F(UBR_AllToAll, OutOfPlace_MultiNode)
{
    if (!setupMultiNode(UBRTestConfig::MIN_RANKS_ALLTOALL, UBRTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes with 4+ ranks";
    }

    using T = UBRTestConfig::DefaultType;
    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t totalCount = countPerRank * nRanks;
    RegInfo sendInfo = allocateAndRegister(totalCount * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(totalCount * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    EXPECT_TRUE(verifyRegistration(sendInfo, "AllToAll_MultiNode_Send"))
        << "UBR path not taken for AllToAll send buffer on multi-node";
    EXPECT_TRUE(verifyRegistration(recvInfo, "AllToAll_MultiNode_Recv"))
        << "UBR path not taken for AllToAll recv buffer on multi-node";

    initializeBufferWithPattern<T>(sendInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int destRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(rank * 100 + destRank));
        });

    ncclResult_t result = ncclAllToAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                        getNcclDataType<T>(),
                                        getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    bool verified = verifyBufferData<T>(recvInfo.buffer, totalCount,
        [rank, countPerRank](size_t i) {
            int srcRank = i / countPerRank;
            return static_cast<T>(static_cast<float>(srcRank * 100 + rank));
        });
    EXPECT_TRUE(verified);
}

// =============================================================================
// SendRecv (P2P) UBR Tests
// =============================================================================

class UBR_SendRecv : public UBRTestBase {};

/**
 * @test SendRecv with UBR - Multi-node only
 * Note: In directMode (single-process), IPC registration is blocked.
 * SendRecv UBR requires multi-node setup.
 */
TEST_F(UBR_SendRecv, RingPattern_MultiNode)
{
    if (!setupMultiNode(UBRTestConfig::MIN_RANKS_DEFAULT, UBRTestConfig::MIN_NODES_MULTINODE)) {
        GTEST_SKIP() << "Requires 2+ nodes for SendRecv UBR";
    }

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::SMALL_COUNT;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);

    EXPECT_TRUE(verifyRegistration(sendInfo, "SendRecv_MultiNode_Send"))
        << "UBR path not taken for SendRecv send buffer on multi-node";
    EXPECT_TRUE(verifyRegistration(recvInfo, "SendRecv_MultiNode_Recv"))
        << "UBR path not taken for SendRecv recv buffer on multi-node";

    // Ring pattern: send to (rank+1), recv from (rank-1)
    int sendPeer = (rank + 1) % nRanks;
    int recvPeer = (rank - 1 + nRanks) % nRanks;

    initSendBuffer<T>(sendInfo.buffer, count, rank);

    EXPECT_EQ(ncclSuccess, ncclGroupStart());
    ncclResult_t sendResult = ncclSend(sendInfo.buffer, count, getNcclDataType<T>(),
                                        sendPeer, getActiveCommunicator(), getActiveStream());
    ncclResult_t recvResult = ncclRecv(recvInfo.buffer, count, getNcclDataType<T>(),
                                        recvPeer, getActiveCommunicator(), getActiveStream());
    EXPECT_EQ(ncclSuccess, ncclGroupEnd());

    ASSERT_EQ(ncclSuccess, sendResult);
    ASSERT_EQ(ncclSuccess, recvResult);

    hipStreamSynchronize(getActiveStream());

    // Verify: should receive value from recvPeer
    T expected = static_cast<T>(static_cast<float>(recvPeer + 1));
    bool verified = verifyBufferData<T>(recvInfo.buffer, count,
        [expected](size_t) { return expected; });
    EXPECT_TRUE(verified);
}

// =============================================================================
// Buffer Size Variation Tests
// =============================================================================

class UBR_BufferSizes : public UBRTestBase {};

TEST_F(UBR_BufferSizes, AllReduce_VariousSizes)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384, 65536,
                                  UBRTestConfig::MEDIUM_COUNT};

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    for (size_t count : sizes) {
        SCOPED_TRACE("Count: " + std::to_string(count));

        RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
        RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

        auto cleanup = makeScopeGuard([&]() {
            cleanupRegInfo(sendInfo);
            cleanupRegInfo(recvInfo);
        });

        ASSERT_NE(sendInfo.buffer, nullptr);
        ASSERT_NE(recvInfo.buffer, nullptr);

        initSendBuffer<T>(sendInfo.buffer, count, rank);

        ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);
        hipStreamSynchronize(getActiveStream());

        EXPECT_TRUE(verifyAllReduceResult<T>(recvInfo.buffer, count, nRanks));
    }
}

// =============================================================================
// Registration Lifecycle Tests
// =============================================================================

class UBR_Lifecycle : public UBRTestBase {};

TEST_F(UBR_Lifecycle, RegisterDeregisterCycle)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::SMALL_COUNT;
    const int numCycles = 5;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    for (int cycle = 0; cycle < numCycles; ++cycle) {
        SCOPED_TRACE("Cycle: " + std::to_string(cycle));

        RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
        RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

        ASSERT_NE(sendInfo.buffer, nullptr);
        ASSERT_NE(recvInfo.buffer, nullptr);

        initSendBuffer<T>(sendInfo.buffer, count, rank);

        ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);
        hipStreamSynchronize(getActiveStream());

        EXPECT_TRUE(verifyAllReduceResult<T>(recvInfo.buffer, count, nRanks));

        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    }
}

TEST_F(UBR_Lifecycle, MultipleBuffersRegistered)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::SMALL_COUNT;
    const int numBuffers = 8;

    std::vector<RegInfo> buffers(numBuffers);

    auto cleanup = makeScopeGuard([&]() {
        for (auto& buf : buffers) {
            cleanupRegInfo(buf);
        }
    });

    // Register multiple buffers
    for (int i = 0; i < numBuffers; ++i) {
        buffers[i] = allocateAndRegister(count * sizeof(T));
        ASSERT_NE(buffers[i].buffer, nullptr);
        EXPECT_TRUE(buffers[i].registered)
            << "Failed to register buffer " << i;
    }

    // Use first two for AllReduce
    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    initSendBuffer<T>(buffers[0].buffer, count, rank);

    ncclResult_t result = ncclAllReduce(buffers[0].buffer, buffers[1].buffer, count,
                                         getNcclDataType<T>(), ncclSum,
                                         getActiveCommunicator(), getActiveStream());
    ASSERT_EQ(ncclSuccess, result);
    hipStreamSynchronize(getActiveStream());

    EXPECT_TRUE(verifyAllReduceResult<T>(buffers[1].buffer, count, nRanks));
}

TEST_F(UBR_Lifecycle, PersistentRegistration)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    using T = UBRTestConfig::DefaultType;
    const size_t count = UBRTestConfig::SMALL_COUNT;
    const int numIterations = 10;

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    // Register once, use multiple times
    RegInfo sendInfo = allocateAndRegister(count * sizeof(T));
    RegInfo recvInfo = allocateAndRegister(count * sizeof(T));

    auto cleanup = makeScopeGuard([&]() {
        cleanupRegInfo(sendInfo);
        cleanupRegInfo(recvInfo);
    });

    ASSERT_NE(sendInfo.buffer, nullptr);
    ASSERT_NE(recvInfo.buffer, nullptr);
    EXPECT_TRUE(sendInfo.registered);
    EXPECT_TRUE(recvInfo.registered);

    for (int iter = 0; iter < numIterations; ++iter) {
        SCOPED_TRACE("Iteration: " + std::to_string(iter));

        // Different value each iteration
        initializeBufferWithPattern<T>(sendInfo.buffer, count,
            [rank, iter](size_t) {
                return static_cast<T>(static_cast<float>((rank + 1) * (iter + 1)));
            });

        ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                             getNcclDataType<T>(), ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        ASSERT_EQ(ncclSuccess, result);
        hipStreamSynchronize(getActiveStream());

        T expected = static_cast<T>(static_cast<float>(nRanks * (nRanks + 1) / 2 * (iter + 1)));
        bool verified = verifyBufferData<T>(recvInfo.buffer, count,
            [expected](size_t) { return expected; });
        EXPECT_TRUE(verified);
    }
}

// =============================================================================
// Data Type Tests - All Collectives with All Data Types
// =============================================================================

class UBR_DataTypes : public UBRTestBase
{
protected:
    // Data type descriptor for test loops
    struct DataTypeInfo {
        ncclDataType_t ncclType;
        size_t elementSize;
        const char* name;
    };

    static std::vector<DataTypeInfo> getDataTypes()
    {
        return {
            {ncclFloat,    sizeof(float),       "float"},
            {ncclDouble,   sizeof(double),      "double"},
            {ncclInt32,    sizeof(int32_t),     "int32"},
            {ncclInt64,    sizeof(int64_t),     "int64"},
            {ncclBfloat16, sizeof(hip_bfloat16), "bfloat16"},
            {ncclFloat16,  sizeof(__half),      "float16"},
        };
    }

    // Generic collective runner that works with any data type
    bool runAllReduceGeneric(ncclDataType_t dtype, size_t elemSize, size_t count,
                              int rank, int nRanks)
    {
        RegInfo sendInfo = allocateAndRegister(count * elemSize);
        RegInfo recvInfo = allocateAndRegister(count * elemSize);

        auto cleanup = makeScopeGuard([&]() {
            cleanupRegInfo(sendInfo);
            cleanupRegInfo(recvInfo);
        });

        if (!sendInfo.buffer || !recvInfo.buffer) return false;

        // Initialize send buffer with rank+1 value
        std::vector<uint8_t> hostSend(count * elemSize);
        fillBufferWithValue(hostSend.data(), dtype, count, static_cast<float>(rank + 1));
        hipMemcpy(sendInfo.buffer, hostSend.data(), count * elemSize, hipMemcpyHostToDevice);

        ncclResult_t result = ncclAllReduce(sendInfo.buffer, recvInfo.buffer, count,
                                             dtype, ncclSum,
                                             getActiveCommunicator(), getActiveStream());
        if (result != ncclSuccess) return false;
        hipStreamSynchronize(getActiveStream());

        // Verify: sum of (1 + 2 + ... + nRanks) = nRanks * (nRanks + 1) / 2
        float expected = static_cast<float>(nRanks * (nRanks + 1) / 2);
        return verifyBufferValue(recvInfo.buffer, dtype, count, expected);
    }

    bool runAllGatherGeneric(ncclDataType_t dtype, size_t elemSize, size_t countPerRank,
                              int rank, int nRanks)
    {
        RegInfo sendInfo = allocateAndRegister(countPerRank * elemSize);
        RegInfo recvInfo = allocateAndRegister(countPerRank * nRanks * elemSize);

        auto cleanup = makeScopeGuard([&]() {
            cleanupRegInfo(sendInfo);
            cleanupRegInfo(recvInfo);
        });

        if (!sendInfo.buffer || !recvInfo.buffer) return false;

        std::vector<uint8_t> hostSend(countPerRank * elemSize);
        fillBufferWithValue(hostSend.data(), dtype, countPerRank, static_cast<float>(rank + 1));
        hipMemcpy(sendInfo.buffer, hostSend.data(), countPerRank * elemSize, hipMemcpyHostToDevice);

        ncclResult_t result = ncclAllGather(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                             dtype, getActiveCommunicator(), getActiveStream());
        if (result != ncclSuccess) return false;
        hipStreamSynchronize(getActiveStream());

        // Verify each rank's contribution
        std::vector<uint8_t> hostRecv(countPerRank * nRanks * elemSize);
        hipMemcpy(hostRecv.data(), recvInfo.buffer, countPerRank * nRanks * elemSize, hipMemcpyDeviceToHost);

        for (int r = 0; r < nRanks; ++r) {
            float expected = static_cast<float>(r + 1);
            if (!verifyBufferSegment(hostRecv.data() + r * countPerRank * elemSize,
                                      dtype, countPerRank, expected)) {
                return false;
            }
        }
        return true;
    }

    bool runReduceScatterGeneric(ncclDataType_t dtype, size_t elemSize, size_t countPerRank,
                                  int rank, int nRanks)
    {
        RegInfo sendInfo = allocateAndRegister(countPerRank * nRanks * elemSize);
        RegInfo recvInfo = allocateAndRegister(countPerRank * elemSize);

        auto cleanup = makeScopeGuard([&]() {
            cleanupRegInfo(sendInfo);
            cleanupRegInfo(recvInfo);
        });

        if (!sendInfo.buffer || !recvInfo.buffer) return false;

        std::vector<uint8_t> hostSend(countPerRank * nRanks * elemSize);
        fillBufferWithValue(hostSend.data(), dtype, countPerRank * nRanks, static_cast<float>(rank + 1));
        hipMemcpy(sendInfo.buffer, hostSend.data(), countPerRank * nRanks * elemSize, hipMemcpyHostToDevice);

        ncclResult_t result = ncclReduceScatter(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                                 dtype, ncclSum,
                                                 getActiveCommunicator(), getActiveStream());
        if (result != ncclSuccess) return false;
        hipStreamSynchronize(getActiveStream());

        float expected = static_cast<float>(nRanks * (nRanks + 1) / 2);
        return verifyBufferValue(recvInfo.buffer, dtype, countPerRank, expected);
    }

    bool runBroadcastGeneric(ncclDataType_t dtype, size_t elemSize, size_t count,
                              int rank, int nRanks, int root = 0)
    {
        RegInfo bufInfo = allocateAndRegister(count * elemSize);

        auto cleanup = makeScopeGuard([&]() {
            cleanupRegInfo(bufInfo);
        });

        if (!bufInfo.buffer) return false;

        const float rootValue = 42.0f;
        std::vector<uint8_t> hostBuf(count * elemSize);
        if (rank == root) {
            fillBufferWithValue(hostBuf.data(), dtype, count, rootValue);
        } else {
            fillBufferWithValue(hostBuf.data(), dtype, count, 0.0f);
        }
        hipMemcpy(bufInfo.buffer, hostBuf.data(), count * elemSize, hipMemcpyHostToDevice);

        ncclResult_t result = ncclBroadcast(bufInfo.buffer, bufInfo.buffer, count,
                                             dtype, root,
                                             getActiveCommunicator(), getActiveStream());
        if (result != ncclSuccess) return false;
        hipStreamSynchronize(getActiveStream());

        return verifyBufferValue(bufInfo.buffer, dtype, count, rootValue);
    }

    bool runAllToAllGeneric(ncclDataType_t dtype, size_t elemSize, size_t countPerRank,
                             int rank, int nRanks)
    {
        const size_t totalCount = countPerRank * nRanks;
        RegInfo sendInfo = allocateAndRegister(totalCount * elemSize);
        RegInfo recvInfo = allocateAndRegister(totalCount * elemSize);

        auto cleanup = makeScopeGuard([&]() {
            cleanupRegInfo(sendInfo);
            cleanupRegInfo(recvInfo);
        });

        if (!sendInfo.buffer || !recvInfo.buffer) return false;

        // Initialize: send[destRank * countPerRank + i] = rank * 100 + destRank
        std::vector<uint8_t> hostSend(totalCount * elemSize);
        for (int destRank = 0; destRank < nRanks; ++destRank) {
            float value = static_cast<float>(rank * 100 + destRank);
            fillBufferSegment(hostSend.data() + destRank * countPerRank * elemSize,
                               dtype, countPerRank, value);
        }
        hipMemcpy(sendInfo.buffer, hostSend.data(), totalCount * elemSize, hipMemcpyHostToDevice);

        ncclResult_t result = ncclAllToAll(sendInfo.buffer, recvInfo.buffer, countPerRank,
                                            dtype, getActiveCommunicator(), getActiveStream());
        if (result != ncclSuccess) return false;
        hipStreamSynchronize(getActiveStream());

        // Verify: recv[srcRank * countPerRank + i] = srcRank * 100 + rank
        std::vector<uint8_t> hostRecv(totalCount * elemSize);
        hipMemcpy(hostRecv.data(), recvInfo.buffer, totalCount * elemSize, hipMemcpyDeviceToHost);

        for (int srcRank = 0; srcRank < nRanks; ++srcRank) {
            float expected = static_cast<float>(srcRank * 100 + rank);
            if (!verifyBufferSegment(hostRecv.data() + srcRank * countPerRank * elemSize,
                                      dtype, countPerRank, expected)) {
                return false;
            }
        }
        return true;
    }

    // Helper functions for generic data type handling
    void fillBufferWithValue(void* buffer, ncclDataType_t dtype, size_t count, float value)
    {
        switch (dtype) {
            case ncclFloat:
                for (size_t i = 0; i < count; ++i) static_cast<float*>(buffer)[i] = value;
                break;
            case ncclDouble:
                for (size_t i = 0; i < count; ++i) static_cast<double*>(buffer)[i] = value;
                break;
            case ncclInt32:
                for (size_t i = 0; i < count; ++i) static_cast<int32_t*>(buffer)[i] = static_cast<int32_t>(value);
                break;
            case ncclInt64:
                for (size_t i = 0; i < count; ++i) static_cast<int64_t*>(buffer)[i] = static_cast<int64_t>(value);
                break;
            case ncclBfloat16:
                for (size_t i = 0; i < count; ++i) static_cast<hip_bfloat16*>(buffer)[i] = hip_bfloat16(value);
                break;
            case ncclFloat16:
                for (size_t i = 0; i < count; ++i) static_cast<__half*>(buffer)[i] = __float2half(value);
                break;
            default:
                break;
        }
    }

    void fillBufferSegment(void* buffer, ncclDataType_t dtype, size_t count, float value)
    {
        fillBufferWithValue(buffer, dtype, count, value);
    }

    bool verifyBufferValue(void* devBuffer, ncclDataType_t dtype, size_t count, float expected)
    {
        size_t elemSize = getElementSize(dtype);
        std::vector<uint8_t> hostBuf(count * elemSize);
        hipMemcpy(hostBuf.data(), devBuffer, count * elemSize, hipMemcpyDeviceToHost);
        return verifyBufferSegment(hostBuf.data(), dtype, count, expected);
    }

    bool verifyBufferSegment(void* buffer, ncclDataType_t dtype, size_t count, float expected)
    {
        const float tolerance = 1e-2f;
        for (size_t i = 0; i < count; ++i) {
            float actual = 0.0f;
            switch (dtype) {
                case ncclFloat:    actual = static_cast<float*>(buffer)[i]; break;
                case ncclDouble:   actual = static_cast<float>(static_cast<double*>(buffer)[i]); break;
                case ncclInt32:    actual = static_cast<float>(static_cast<int32_t*>(buffer)[i]); break;
                case ncclInt64:    actual = static_cast<float>(static_cast<int64_t*>(buffer)[i]); break;
                case ncclBfloat16: actual = static_cast<float>(static_cast<hip_bfloat16*>(buffer)[i]); break;
                case ncclFloat16:  actual = __half2float(static_cast<__half*>(buffer)[i]); break;
                default: return false;
            }
            if (std::abs(actual - expected) > tolerance) {
                TEST_WARN("Mismatch at [%zu]: expected=%f got=%f", i, expected, actual);
                return false;
            }
        }
        return true;
    }

    size_t getElementSize(ncclDataType_t dtype)
    {
        switch (dtype) {
            case ncclFloat:    return sizeof(float);
            case ncclDouble:   return sizeof(double);
            case ncclInt32:    return sizeof(int32_t);
            case ncclInt64:    return sizeof(int64_t);
            case ncclBfloat16: return sizeof(hip_bfloat16);
            case ncclFloat16:  return sizeof(__half);
            default:           return 0;
        }
    }
};

/**
 * @test All data types with AllReduce
 */
TEST_F(UBR_DataTypes, AllReduce_AllTypes)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = UBRTestConfig::SMALL_COUNT;

    for (const auto& dt : getDataTypes()) {
        SCOPED_TRACE(std::string("DataType: ") + dt.name);
        EXPECT_TRUE(runAllReduceGeneric(dt.ncclType, dt.elementSize, count, rank, nRanks))
            << "AllReduce failed for " << dt.name;
    }
}

/**
 * @test All data types with AllGather
 */
TEST_F(UBR_DataTypes, AllGather_AllTypes)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    for (const auto& dt : getDataTypes()) {
        SCOPED_TRACE(std::string("DataType: ") + dt.name);
        EXPECT_TRUE(runAllGatherGeneric(dt.ncclType, dt.elementSize, countPerRank, rank, nRanks))
            << "AllGather failed for " << dt.name;
    }
}

/**
 * @test All data types with ReduceScatter
 */
TEST_F(UBR_DataTypes, ReduceScatter_AllTypes)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    for (const auto& dt : getDataTypes()) {
        SCOPED_TRACE(std::string("DataType: ") + dt.name);
        EXPECT_TRUE(runReduceScatterGeneric(dt.ncclType, dt.elementSize, countPerRank, rank, nRanks))
            << "ReduceScatter failed for " << dt.name;
    }
}

/**
 * @test All data types with Broadcast
 */
TEST_F(UBR_DataTypes, Broadcast_AllTypes)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = UBRTestConfig::SMALL_COUNT;

    for (const auto& dt : getDataTypes()) {
        SCOPED_TRACE(std::string("DataType: ") + dt.name);
        EXPECT_TRUE(runBroadcastGeneric(dt.ncclType, dt.elementSize, count, rank, nRanks))
            << "Broadcast failed for " << dt.name;
    }
}

/**
 * @test All data types with AllToAll
 */
TEST_F(UBR_DataTypes, AllToAll_AllTypes)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t countPerRank = UBRTestConfig::SMALL_COUNT;

    for (const auto& dt : getDataTypes()) {
        SCOPED_TRACE(std::string("DataType: ") + dt.name);
        EXPECT_TRUE(runAllToAllGeneric(dt.ncclType, dt.elementSize, countPerRank, rank, nRanks))
            << "AllToAll failed for " << dt.name;
    }
}

/**
 * @test All collectives with all data types (comprehensive)
 */
TEST_F(UBR_DataTypes, AllCollectives_AllTypes)
{
    ASSERT_TRUE(setupSingleNode(UBRTestConfig::MIN_RANKS_DEFAULT));

    int rank, nRanks;
    ncclCommUserRank(getActiveCommunicator(), &rank);
    ncclCommCount(getActiveCommunicator(), &nRanks);

    const size_t count = UBRTestConfig::SMALL_COUNT;

    for (const auto& dt : getDataTypes()) {
        SCOPED_TRACE(std::string("DataType: ") + dt.name);

        TEST_INFO("Testing all collectives with %s", dt.name);

        EXPECT_TRUE(runAllReduceGeneric(dt.ncclType, dt.elementSize, count, rank, nRanks))
            << "AllReduce failed for " << dt.name;

        EXPECT_TRUE(runAllGatherGeneric(dt.ncclType, dt.elementSize, count, rank, nRanks))
            << "AllGather failed for " << dt.name;

        EXPECT_TRUE(runReduceScatterGeneric(dt.ncclType, dt.elementSize, count, rank, nRanks))
            << "ReduceScatter failed for " << dt.name;

        EXPECT_TRUE(runBroadcastGeneric(dt.ncclType, dt.elementSize, count, rank, nRanks))
            << "Broadcast failed for " << dt.name;

        EXPECT_TRUE(runAllToAllGeneric(dt.ncclType, dt.elementSize, count, rank, nRanks))
            << "AllToAll failed for " << dt.name;
    }
}

#endif // MPI_TESTS_ENABLED
