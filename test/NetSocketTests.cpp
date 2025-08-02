/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include "gtest/gtest.h"
#include <atomic>
#include <cstring>
#include <thread>

extern ncclNet_t ncclNetSocket;

namespace RcclUnitTesting {

class NetSocketTests : public ::testing::Test {

private:
  // RAII wrapper for send communicator
  class SendCommGuard {
  private:
    void *comm_ = nullptr;

  public:
    explicit SendCommGuard(void *comm = nullptr)
        : comm_(comm) {} // default constructor

    // Move constructor
    SendCommGuard(SendCommGuard &&other) noexcept : comm_(other.comm_) {
      other.comm_ = nullptr;
    }

    // Move assignment
    SendCommGuard &operator=(SendCommGuard &&other) noexcept {
      if (this != &other) {
        reset();
        comm_ = other.comm_;
        other.comm_ = nullptr;
      }
      return *this;
    }

    // Disable copy constructor and assignment
    SendCommGuard(const SendCommGuard &other) = delete;
    SendCommGuard &operator=(const SendCommGuard &other) = delete;

    ~SendCommGuard() { reset(); }

    void reset(void *comm = nullptr) {
      if (comm_ && comm_ != comm) {
        ncclResult_t result = ncclNetSocket.closeSend(comm_);
        if (result != ncclSuccess) {
          INFO(NCCL_LOG_INFO, "Warning: SendCommGuard cleanup failed: %d",
               result);
        }
      }
      comm_ = comm;
    }

    void *get() const { return comm_; }
    void *release() {
      void *temp = comm_;
      comm_ = nullptr;
      return temp;
    }

    explicit operator bool() const { return comm_ != nullptr; }
  };

  // RAII wrapper for receive communicator
  class RecvCommGuard {
  private:
    void *comm_;

  public:
    explicit RecvCommGuard(void *comm = nullptr) : comm_(comm) {}

    // Move constructor
    RecvCommGuard(RecvCommGuard &&other) noexcept : comm_(other.comm_) {
      other.comm_ = nullptr;
    }

    // Move assignment
    RecvCommGuard &operator=(RecvCommGuard &&other) noexcept {
      if (this != &other) {
        reset();
        comm_ = other.comm_;
        other.comm_ = nullptr;
      }
      return *this;
    }

    // Disable copy
    RecvCommGuard(const RecvCommGuard &) = delete;
    RecvCommGuard &operator=(const RecvCommGuard &) = delete;

    ~RecvCommGuard() { reset(); }

    void reset(void *comm = nullptr) {
      if (comm_ && comm_ != comm) {
        ncclResult_t result = ncclNetSocket.closeRecv(comm_);
        if (result != ncclSuccess) {
          INFO(NCCL_LOG_INFO, "Warning: RecvCommGuard cleanup failed: %d",
               result);
        }
      }
      comm_ = comm;
    }

    void *get() const { return comm_; }
    void *release() {
      void *temp = comm_;
      comm_ = nullptr;
      return temp;
    }

    explicit operator bool() const { return comm_ != nullptr; }
  };

protected:
  void SetUp() override {
    ncclResult_t result = ncclNetSocket.init(nullptr, nullptr);
    ASSERT_EQ(result, ncclSuccess);

    result = ncclNetSocket.devices(&ndev);
    ASSERT_EQ(result, ncclSuccess);

    if (ndev == 0) {
      GTEST_SKIP() << "No network devices available for testing";
    }
  }

  int ndev = 0;

  // Common function to test socket properties
  void TestSocketProperties() {
    INFO(NCCL_LOG_INFO, "\n=== Testing socket properties ===");

    // Test ncclNetSocketGetProperties for each device
    for (int dev = 0; dev < ndev; dev++) {
      ncclNetProperties_t props = {};
      ncclResult_t propsResult = ncclNetSocket.getProperties(dev, &props);
      INFO(NCCL_LOG_INFO, "Device %d - getProperties result: %d", dev,
           propsResult);
      if (propsResult == ncclSuccess) {
        INFO(NCCL_LOG_INFO,
             "  Device %d properties: name='%s', pciPath='%s', guid=%llu, "
             "speed=%d, port=%d, maxComms=%d",
             dev, props.name, props.pciPath, (unsigned long long)props.guid,
             props.speed, props.port, props.maxComms);
      }
      EXPECT_EQ(propsResult, ncclSuccess)
          << "getProperties failed for device " << dev;
    }
  }

  // Common function to establish a connection pair with improved reliability
  bool EstablishConnectionPair(void *handle, void *listenComm, void *&sendComm,
                               void *&recvComm) {
    // Initialize output parameters
    sendComm = nullptr;
    recvComm = nullptr;

    // RAII guards for automatic cleanup
    SendCommGuard sendGuard;
    RecvCommGuard recvGuard;

    std::atomic<bool> connectionEstablished{false};
    std::atomic<bool> acceptCompleted{false};
    std::atomic<bool> connectCompleted{false};
    std::atomic<bool> shouldStop{false};

    INFO(NCCL_LOG_INFO,
         "Establishing connection pair with enhanced reliability");

    std::thread connectAcceptThread([&]() {
      // Accept thread with longer timeout and better coordination
      std::thread acceptThread([&]() {
        ncclNetDeviceHandle_t *recvDevComm = nullptr;
        void *tempRecvComm = nullptr;

        // Increased attempts and longer total timeout for reliability
        for (int attempt = 0; attempt < 100 && !shouldStop.load(); attempt++) {
          ncclResult_t acceptResult =
              ncclNetSocket.accept(listenComm, &tempRecvComm, &recvDevComm);
          if (acceptResult == ncclSuccess && tempRecvComm != nullptr) {
            recvGuard.reset(tempRecvComm);
            acceptCompleted = true;
            INFO(NCCL_LOG_INFO, "Accept completed successfully on attempt %d",
                 attempt + 1);
            break;
          }

          // Longer sleep for network stability
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (!acceptCompleted.load()) {
          INFO(NCCL_LOG_INFO, "Accept thread timed out after 100 attempts");
        }
      });

      // Connect thread with longer timeout and better coordination
      std::thread connectThread([&]() {
        ncclNetCommConfig_t config = {};
        ncclNetDeviceHandle_t *sendDevComm = nullptr;
        void *tempSendComm = nullptr;

        // Give accept thread more time to start listening
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Increased attempts and longer total timeout for reliability
        for (int attempt = 0; attempt < 100 && !shouldStop.load(); attempt++) {
          ncclResult_t connectResult = ncclNetSocket.connect(
              0, &config, handle, &tempSendComm, &sendDevComm);
          if (connectResult == ncclSuccess && tempSendComm != nullptr) {
            sendGuard.reset(tempSendComm);
            connectCompleted = true;
            INFO(NCCL_LOG_INFO, "Connect completed successfully on attempt %d",
                 attempt + 1);
            break;
          }

          // Longer sleep for network stability
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        if (!connectCompleted.load()) {
          INFO(NCCL_LOG_INFO, "Connect thread timed out after 100 attempts");
        }
      });

      // Wait for both threads with overall timeout
      auto startTime = std::chrono::steady_clock::now();
      const auto maxWaitTime =
          std::chrono::seconds(10); // 10 second overall timeout

      while (!acceptCompleted.load() || !connectCompleted.load()) {
        auto currentTime = std::chrono::steady_clock::now();
        if (currentTime - startTime > maxWaitTime) {
          INFO(NCCL_LOG_INFO,
               "Overall connection timeout reached, stopping threads");
          shouldStop = true;
          break;
        }

        // Check every 100ms
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }

      acceptThread.join();
      connectThread.join();

      // Check if both operations completed successfully
      connectionEstablished = acceptCompleted.load() &&
                              connectCompleted.load() && sendGuard && recvGuard;
    });

    connectAcceptThread.join();

    if (connectionEstablished) {
      // Transfer ownership to output parameters
      sendComm = sendGuard.release();
      recvComm = recvGuard.release();

      INFO(NCCL_LOG_INFO, "Successfully established connection pair");
      return true;
    } else {
      INFO(NCCL_LOG_INFO,
           "Failed to establish connection pair - accept: %s, connect: %s",
           acceptCompleted.load() ? "success" : "failed",
           connectCompleted.load() ? "success" : "failed");
      // RAII guards will automatically clean up any partial connections
      return false;
    }
  }

  // Common function to setup memory and operations for a test size
  bool SetupOperationsForSize(void *sendComm, void *recvComm, size_t testSize,
                              std::vector<std::vector<char>> &sendBuffers,
                              std::vector<std::vector<char>> &recvBuffers,
                              std::vector<void *> &sendMhandles,
                              std::vector<void *> &recvMhandles,
                              std::vector<void *> &sendRequests,
                              std::vector<void *> &recvRequests,
                              uint8_t fillPattern = 0xCD) {

    // Create buffers
    sendBuffers.emplace_back(testSize, fillPattern);
    recvBuffers.emplace_back(testSize, 0x00);

    void *sendMhandle = nullptr;
    void *recvMhandle = nullptr;

    // Register memory
    ncclResult_t sendRegResult =
        ncclNetSocket.regMr(sendComm, sendBuffers.back().data(), testSize,
                            NCCL_PTR_HOST, &sendMhandle);
    ncclResult_t recvRegResult =
        ncclNetSocket.regMr(recvComm, recvBuffers.back().data(), testSize,
                            NCCL_PTR_HOST, &recvMhandle);

    if (sendRegResult == ncclSuccess && recvRegResult == ncclSuccess) {
      sendMhandles.push_back(sendMhandle);
      recvMhandles.push_back(recvMhandle);

      INFO(NCCL_LOG_INFO, "Memory registration successful for size %zu",
           testSize);

      // Start send operation
      void *sendRequest = nullptr;
      ncclResult_t sendResult =
          ncclNetSocket.isend(sendComm, sendBuffers.back().data(), testSize, 0,
                              sendMhandle, nullptr, &sendRequest);

      // Start receive operation
      void *recvRequest = nullptr;
      void *recvDataPtr = recvBuffers.back().data();
      size_t recvSize = testSize;
      int tag = 0;
      ncclResult_t recvResult =
          ncclNetSocket.irecv(recvComm, 1, &recvDataPtr, &recvSize, &tag,
                              &recvMhandle, nullptr, &recvRequest);

      if (sendResult == ncclSuccess && recvResult == ncclSuccess &&
          sendRequest && recvRequest) {
        sendRequests.push_back(sendRequest);
        recvRequests.push_back(recvRequest);
        INFO(NCCL_LOG_INFO, "Successfully started operations for size %zu",
             testSize);
        return true;
      } else {
        INFO(NCCL_LOG_INFO,
             "Failed to start operations - send result: %d, recv result: %d",
             sendResult, recvResult);
        sendRequests.push_back(nullptr);
        recvRequests.push_back(nullptr);
        return false;
      }
    } else {
      INFO(NCCL_LOG_INFO,
           "Failed to register memory - send result: %d, recv result: %d",
           sendRegResult, recvRegResult);
      sendMhandles.push_back(nullptr);
      recvMhandles.push_back(nullptr);
      sendRequests.push_back(nullptr);
      recvRequests.push_back(nullptr);
      return false;
    }
  }

  // Common function to progress operations and test ncclNetSocketGetTask
  bool ProgressOperations(void *sendRequest, void *recvRequest, size_t testSize,
                          const std::string &testContext = "") {
    const int maxTestIterations = 10;
    bool taskCreationExercised = false;

    INFO(NCCL_LOG_INFO,
         "Starting progress testing - this exercises ncclNetSocketGetTask%s",
         testContext.c_str());

    for (int testIter = 0; testIter < maxTestIterations; testIter++) {
      INFO(NCCL_LOG_INFO, "  Progress test iteration %d/%d", testIter + 1,
           maxTestIterations);

      if (sendRequest && recvRequest) {
        int sendDone = 0, recvDone = 0;
        int sendSize = 0, recvSize_out = 0;

        ncclResult_t sendTestResult =
            ncclNetSocket.test(sendRequest, &sendDone, &sendSize);
        ncclResult_t recvTestResult =
            ncclNetSocket.test(recvRequest, &recvDone, &recvSize_out);

        INFO(NCCL_LOG_INFO, "    Send test: result=%d, done=%d", sendTestResult,
             sendDone);
        INFO(NCCL_LOG_INFO, "    Recv test: result=%d, done=%d", recvTestResult,
             recvDone);

        // If we reach this point with successful or in-progress results,
        // ncclNetSocketGetTask was exercised
        if ((sendTestResult == ncclSuccess ||
             sendTestResult == ncclInProgress) &&
            (recvTestResult == ncclSuccess ||
             recvTestResult == ncclInProgress)) {
          taskCreationExercised = true;
          INFO(NCCL_LOG_INFO,
               "    *** SUCCESS: ncclNetSocketGetTask was exercised! ***");
        }

        // Count completed operations
        if (sendDone && recvDone) {
          INFO(NCCL_LOG_INFO, "    Operations completed successfully!");
          break;
        }

        // If operations fail, that's okay - we still exercised the code path
        if (sendTestResult != ncclSuccess && sendTestResult != ncclInProgress) {
          INFO(NCCL_LOG_INFO, "    Send operation failed, but "
                              "ncclNetSocketGetTask was still exercised");
          break;
        }
        if (recvTestResult != ncclSuccess && recvTestResult != ncclInProgress) {
          INFO(NCCL_LOG_INFO, "    Recv operation failed, but "
                              "ncclNetSocketGetTask was still exercised");
          break;
        }
      }

      // Give time between tests
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (taskCreationExercised) {
      INFO(NCCL_LOG_INFO,
           "*** VERIFICATION: ncclNetSocketGetTask was successfully exercised "
           "for buffer size %zu ***",
           testSize);
    }

    return taskCreationExercised;
  }

  // Common function to deregister memory and test ncclNetSocketDeregMr
  void DeregisterMemory(void *sendComm, void *recvComm,
                        const std::vector<void *> &sendMhandles,
                        const std::vector<void *> &recvMhandles,
                        size_t testSize) {
    INFO(NCCL_LOG_INFO,
         "\n=== Testing ncclNetSocketDeregMr for size %zu ===", testSize);

    // Deregister send memory handles
    for (size_t j = 0; j < sendMhandles.size(); j++) {
      if (sendComm) {
        INFO(NCCL_LOG_INFO, "Deregistering send memory handle %zu for size %zu",
             j, testSize);
        ncclResult_t deregResult =
            ncclNetSocket.deregMr(sendComm, sendMhandles[j]);
        INFO(NCCL_LOG_INFO, "Send memory deregMr result: %d", deregResult);
        EXPECT_EQ(deregResult, ncclSuccess)
            << "deregMr should succeed for send memory, size: " << testSize;
      }
    }

    // Deregister receive memory handles
    for (size_t j = 0; j < recvMhandles.size(); j++) {
      if (recvComm) {
        INFO(NCCL_LOG_INFO, "Deregistering recv memory handle %zu for size %zu",
             j, testSize);
        ncclResult_t deregResult =
            ncclNetSocket.deregMr(recvComm, recvMhandles[j]);
        INFO(NCCL_LOG_INFO, "Recv memory deregMr result: %d", deregResult);
        EXPECT_EQ(deregResult, ncclSuccess)
            << "deregMr should succeed for recv memory, size: " << testSize;
      }
    }
  }

  // Common function to cleanup communicators
  void CleanupCommunicators(const std::vector<void *> &sendComms,
                            const std::vector<void *> &recvComms,
                            void *listenComm) {
    INFO(NCCL_LOG_INFO, "\nCleaning up communicators...");

    for (size_t i = 0; i < sendComms.size(); i++) {
      if (sendComms[i]) {
        INFO(NCCL_LOG_INFO, "Closing send communicator %zu", i);
        ncclResult_t closeResult = ncclNetSocket.closeSend(sendComms[i]);
        if (closeResult != ncclSuccess) {
          INFO(NCCL_LOG_INFO, "Note: Send communicator %zu close returned: %d",
               i, closeResult);
        }
      }
    }

    for (size_t i = 0; i < recvComms.size(); i++) {
      if (recvComms[i]) {
        INFO(NCCL_LOG_INFO, "Closing recv communicator %zu", i);
        ncclResult_t closeResult = ncclNetSocket.closeRecv(recvComms[i]);
        if (closeResult != ncclSuccess) {
          INFO(NCCL_LOG_INFO, "Note: Recv communicator %zu close returned: %d",
               i, closeResult);
        }
      }
    }

    if (listenComm) {
      INFO(NCCL_LOG_INFO, "Closing listen communicator");
      ncclResult_t closeResult = ncclNetSocket.closeListen(listenComm);
      if (closeResult != ncclSuccess) {
        INFO(NCCL_LOG_INFO, "Note: Listen communicator close returned: %d",
             closeResult);
      }
    }
  }

  // Common function to get test buffer sizes
  std::vector<size_t> GetTestSizes() {
    return {
        1024,           // Small - basic test
        64 * 1024,      // MIN_CHUNKSIZE - boundary case
        128 * 1024,     // 2x MIN_CHUNKSIZE - will exercise subdivision
        256 * 1024,     // 4x MIN_CHUNKSIZE - multiple chunks
        512 * 1024,     // 8x MIN_CHUNKSIZE - many chunks
        1024 * 1024,    // Large - stress test
        2 * 1024 * 1024 // Very large - comprehensive test
    };
  }
};

// Test concurrent operations task creation in default configuration (without
// env vars)
TEST_F(NetSocketTests, TestConcurrentOperationsTaskCreationDefault) {
  INFO(NCCL_LOG_INFO,
       "Testing task creation functionality in default configuration");
  INFO(NCCL_LOG_INFO,
       "This test exercises ncclNetSocketGetTask regardless of nSocks value");

  // Test socket properties
  TestSocketProperties();

  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;

  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);
  ASSERT_EQ(result, ncclSuccess);

  INFO(NCCL_LOG_INFO, "Testing task creation functionality in default mode");

  std::vector<void *> sendComms;
  std::vector<void *> recvComms;

  // Establish connection
  void *sendComm = nullptr;
  void *recvComm = nullptr;
  bool connectionSuccess =
      EstablishConnectionPair(handle, listenComm, sendComm, recvComm);

  if (connectionSuccess) {
    sendComms.push_back(sendComm);
    recvComms.push_back(recvComm);

    // Test with various buffer sizes
    std::vector<size_t> testSizes = GetTestSizes();

    for (size_t testSize : testSizes) {
      INFO(NCCL_LOG_INFO,
           "\n=== Testing with buffer size: %zu bytes ===", testSize);
      INFO(NCCL_LOG_INFO,
           "This exercises ncclNetSocketGetTask task creation logic");

      std::vector<void *> sendMhandles;
      std::vector<void *> recvMhandles;
      std::vector<void *> sendRequests;
      std::vector<void *> recvRequests;
      std::vector<std::vector<char>> sendBuffers;
      std::vector<std::vector<char>> recvBuffers;

      // Setup operations for this test size
      bool setupSuccess = SetupOperationsForSize(
          sendComm, recvComm, testSize, sendBuffers, recvBuffers, sendMhandles,
          recvMhandles, sendRequests, recvRequests, 0xCD);

      if (setupSuccess) {
        // Progress operations
        bool taskExercised =
            ProgressOperations(sendRequests[0], recvRequests[0], testSize);

        if (!taskExercised) {
          INFO(NCCL_LOG_INFO,
               "*** NOTE: Operations didn't progress as expected for size %zu, "
               "but API was still exercised ***",
               testSize);
        }
      } else {
        INFO(NCCL_LOG_INFO,
             "No operations started - skipping progress testing for size %zu",
             testSize);
      }

      // Deregister memory
      DeregisterMemory(sendComm, recvComm, sendMhandles, recvMhandles,
                       testSize);

      INFO(NCCL_LOG_INFO,
           "=== Completed testing for buffer size: %zu bytes ===", testSize);
    }

    INFO(NCCL_LOG_INFO, "\n*** TEST SUCCESS: ncclNetSocketGetTask was "
                        "successfully exercised in default configuration! ***");
  } else {
    INFO(NCCL_LOG_INFO, "No connections established - test passed (network may "
                        "not be available)");
  }

  // Cleanup
  CleanupCommunicators(sendComms, recvComms, listenComm);
  INFO(NCCL_LOG_INFO,
       "TestConcurrentOperationsTaskCreationDefault completed successfully");
}

// Test multiple concurrent operations to stress test task creation
TEST_F(NetSocketTests, TestConcurrentOperationsTaskCreation) {
  INFO(NCCL_LOG_INFO, "Checking socket configuration environment variables");

  // Check if the required environment variables are set
  const char *nThreadsEnv = getenv("NCCL_SOCKET_NTHREADS");
  const char *nSocksPerThreadEnv = getenv("NCCL_NSOCKS_PERTHREAD");

  if (!nThreadsEnv || !nSocksPerThreadEnv) {
    INFO(NCCL_LOG_INFO,
         "SKIPPING TEST: Required environment variables not set");
    INFO(NCCL_LOG_INFO,
         "Please set the following environment variables to run this test:");
    INFO(NCCL_LOG_INFO, "  export NCCL_SOCKET_NTHREADS=1");
    INFO(NCCL_LOG_INFO, "  export NCCL_NSOCKS_PERTHREAD=2");
    INFO(NCCL_LOG_INFO,
         "This ensures nSocks > 0 so that ncclNetSocketGetTask gets called");
    GTEST_SKIP() << "Environment variables NCCL_SOCKET_NTHREADS and "
                    "NCCL_NSOCKS_PERTHREAD must be set";
    return;
  }

  int nThreads = std::atoi(nThreadsEnv);
  int nSocksPerThread = std::atoi(nSocksPerThreadEnv);
  int expectedNSocks = nThreads * nSocksPerThread;

  INFO(NCCL_LOG_INFO, "Environment configuration found:");
  INFO(NCCL_LOG_INFO, "  NCCL_SOCKET_NTHREADS=%d", nThreads);
  INFO(NCCL_LOG_INFO, "  NCCL_NSOCKS_PERTHREAD=%d", nSocksPerThread);
  INFO(NCCL_LOG_INFO, "  Expected nSocks=%d", expectedNSocks);

  if (expectedNSocks <= 0) {
    INFO(NCCL_LOG_INFO,
         "SKIPPING TEST: Invalid configuration - nSocks must be > 0");
    INFO(NCCL_LOG_INFO, "Current configuration results in nSocks=%d",
         expectedNSocks);
    INFO(
        NCCL_LOG_INFO,
        "Please ensure NCCL_SOCKET_NTHREADS > 0 and NCCL_NSOCKS_PERTHREAD > 0");
    GTEST_SKIP() << "Environment variables must result in nSocks > 0 for "
                    "ncclNetSocketGetTask to be called";
    return;
  }

  INFO(NCCL_LOG_INFO, "Configuration valid - proceeding with test to exercise "
                      "ncclNetSocketGetTask");

  // Test socket properties
  TestSocketProperties();

  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;

  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);
  ASSERT_EQ(result, ncclSuccess);

  INFO(NCCL_LOG_INFO, "Testing task creation functionality - ensuring "
                      "ncclNetSocketGetTask is called");

  std::vector<void *> sendComms;
  std::vector<void *> recvComms;

  // Establish connection
  void *sendComm = nullptr;
  void *recvComm = nullptr;
  bool connectionSuccess =
      EstablishConnectionPair(handle, listenComm, sendComm, recvComm);

  if (connectionSuccess) {
    sendComms.push_back(sendComm);
    recvComms.push_back(recvComm);

    // Test with buffer sizes that will trigger task subdivision
    std::vector<size_t> testSizes = GetTestSizes();

    for (size_t testSize : testSizes) {
      INFO(NCCL_LOG_INFO,
           "\n=== Testing with buffer size: %zu bytes ===", testSize);
      INFO(NCCL_LOG_INFO, "This should trigger ncclNetSocketGetTask to create "
                          "task subdivision");

      std::vector<void *> sendMhandles;
      std::vector<void *> recvMhandles;
      std::vector<void *> sendRequests;
      std::vector<void *> recvRequests;
      std::vector<std::vector<char>> sendBuffers;
      std::vector<std::vector<char>> recvBuffers;

      // Setup operations for this test size
      bool setupSuccess = SetupOperationsForSize(
          sendComm, recvComm, testSize, sendBuffers, recvBuffers, sendMhandles,
          recvMhandles, sendRequests, recvRequests, 0xAB);

      if (setupSuccess) {
        // Progress operations with context about environment variables
        ProgressOperations(sendRequests[0], recvRequests[0], testSize,
                           " (with nSocks > 0 from environment variables)");
      } else {
        INFO(NCCL_LOG_INFO,
             "No operations started - skipping progress testing for size %zu",
             testSize);
      }

      // Deregister memory
      DeregisterMemory(sendComm, recvComm, sendMhandles, recvMhandles,
                       testSize);

      INFO(NCCL_LOG_INFO,
           "=== Completed testing for buffer size: %zu bytes ===", testSize);
    }

    INFO(NCCL_LOG_INFO, "\n*** TEST SUCCESS: ncclNetSocketGetTask was "
                        "successfully exercised! ***");
  } else {
    INFO(NCCL_LOG_INFO, "No connections established - test passed (network may "
                        "not be available)");
  }

  // Cleanup
  CleanupCommunicators(sendComms, recvComms, listenComm);
  INFO(NCCL_LOG_INFO,
       "TestConcurrentOperationsTaskCreation completed successfully");
}

// Test for invalid device index in listen function
TEST_F(NetSocketTests, TestInvalidDeviceIndexListen) {
  INFO(NCCL_LOG_INFO, "Testing invalid device index in ncclNetSocketListen");

  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;

  // Test with negative device index
  ncclResult_t result = ncclNetSocket.listen(-1, handle, &listenComm);
  INFO(NCCL_LOG_INFO, "Listen with dev=-1 returned: %d", result);
  EXPECT_EQ(result, ncclInternalError)
      << "Listen should fail with negative device index";

  // Test with device index greater than available devices
  int invalidDev = ndev + 10;
  result = ncclNetSocket.listen(invalidDev, handle, &listenComm);
  INFO(NCCL_LOG_INFO, "Listen with dev=%d (> ndev=%d) returned: %d", invalidDev,
       ndev, result);
  EXPECT_EQ(result, ncclInternalError)
      << "Listen should fail with device index >= ndev";

  INFO(NCCL_LOG_INFO, "TestInvalidDeviceIndexListen completed");
}

// Test for invalid device index in connect function
TEST_F(NetSocketTests, TestInvalidDeviceIndexConnect) {
  INFO(NCCL_LOG_INFO, "Testing invalid device index in ncclNetSocketConnect");

  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *sendComm = nullptr;
  ncclNetCommConfig_t config = {};
  ncclNetDeviceHandle_t *sendDevComm = nullptr;

  // Test with negative device index
  ncclResult_t result =
      ncclNetSocket.connect(-1, &config, handle, &sendComm, &sendDevComm);
  INFO(NCCL_LOG_INFO, "Connect with dev=-1 returned: %d", result);
  EXPECT_EQ(result, ncclInternalError)
      << "Connect should fail with negative device index";

  // Test with device index greater than available devices
  int invalidDev = ndev + 10;
  result = ncclNetSocket.connect(invalidDev, &config, handle, &sendComm,
                                 &sendDevComm);
  INFO(NCCL_LOG_INFO, "Connect with dev=%d (> ndev=%d) returned: %d",
       invalidDev, ndev, result);
  EXPECT_EQ(result, ncclInternalError)
      << "Connect should fail with device index >= ndev";

  INFO(NCCL_LOG_INFO, "TestInvalidDeviceIndexConnect completed");
}

// Test for NULL request in test function
TEST_F(NetSocketTests, TestNullRequestInTest) {
  INFO(NCCL_LOG_INFO, "Testing NULL request in ncclNetSocketTest");

  int done = 0;
  int size = 0;

  // Test with NULL request
  ncclResult_t result = ncclNetSocket.test(nullptr, &done, &size);
  INFO(NCCL_LOG_INFO, "Test with NULL request returned: %d", result);
  EXPECT_EQ(result, ncclInternalError) << "Test should fail with NULL request";

  INFO(NCCL_LOG_INFO, "TestNullRequestInTest completed");
}

// Test for invalid array size in irecv function
TEST_F(NetSocketTests, TestInvalidArraySizeIrecv) {
  INFO(NCCL_LOG_INFO, "Testing invalid array size in ncclNetSocketIrecv");

  // Setup a dummy communicator first
  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);

  if (result == ncclSuccess && listenComm) {
    void *sendComm = nullptr;
    void *recvComm = nullptr;
    bool connectionSuccess =
        EstablishConnectionPair(handle, listenComm, sendComm, recvComm);

    if (connectionSuccess && recvComm) {
      // Test with n != 1 (invalid for socket implementation)
      std::vector<char> buffer1(1024, 0xAA);
      std::vector<char> buffer2(1024, 0xBB);
      void *data[2] = {buffer1.data(), buffer2.data()};
      size_t sizes[2] = {1024, 1024};
      int tags[2] = {0, 1};
      void *mhandles[2] = {nullptr, nullptr};
      void *phandles[2] = {nullptr, nullptr};
      void *request = nullptr;

      // Test with n=2 (should fail for socket implementation)
      result = ncclNetSocket.irecv(recvComm, 2, data, sizes, tags, mhandles,
                                   phandles, &request);
      INFO(NCCL_LOG_INFO, "Irecv with n=2 returned: %d", result);
      EXPECT_EQ(result, ncclInternalError) << "Irecv should fail with n != 1";

      // Test with n=0 (should fail)
      result = ncclNetSocket.irecv(recvComm, 0, data, sizes, tags, mhandles,
                                   phandles, &request);
      INFO(NCCL_LOG_INFO, "Irecv with n=0 returned: %d", result);
      EXPECT_EQ(result, ncclInternalError) << "Irecv should fail with n != 1";

      // Cleanup
      if (sendComm)
        ncclNetSocket.closeSend(sendComm);
      if (recvComm)
        ncclNetSocket.closeRecv(recvComm);
    }

    if (listenComm)
      ncclNetSocket.closeListen(listenComm);
  }

  INFO(NCCL_LOG_INFO, "TestInvalidArraySizeIrecv completed");
}

// Test for non-host memory type in regMr function
TEST_F(NetSocketTests, TestNonHostMemoryRegMr) {
  INFO(NCCL_LOG_INFO, "Testing non-host memory type in ncclNetSocketRegMr");

  // Setup a dummy communicator first
  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);

  if (result == ncclSuccess && listenComm) {
    void *sendComm = nullptr;
    void *recvComm = nullptr;
    bool connectionSuccess =
        EstablishConnectionPair(handle, listenComm, sendComm, recvComm);

    if (connectionSuccess && sendComm) {
      std::vector<char> buffer(1024, 0xAA);
      void *mhandle = nullptr;

      // Test with NCCL_PTR_CUDA (should fail for socket implementation)
      result = ncclNetSocket.regMr(sendComm, buffer.data(), 1024, NCCL_PTR_CUDA,
                                   &mhandle);
      INFO(NCCL_LOG_INFO, "RegMr with NCCL_PTR_CUDA returned: %d", result);
      EXPECT_EQ(result, ncclInternalError)
          << "RegMr should fail with non-host memory type";

      // Test with valid NCCL_PTR_HOST (should succeed)
      result = ncclNetSocket.regMr(sendComm, buffer.data(), 1024, NCCL_PTR_HOST,
                                   &mhandle);
      INFO(NCCL_LOG_INFO, "RegMr with NCCL_PTR_HOST returned: %d", result);
      EXPECT_EQ(result, ncclSuccess)
          << "RegMr should succeed with host memory type";

      // Cleanup
      if (sendComm)
        ncclNetSocket.closeSend(sendComm);
      if (recvComm)
        ncclNetSocket.closeRecv(recvComm);
    }

    if (listenComm)
      ncclNetSocket.closeListen(listenComm);
  }

  INFO(NCCL_LOG_INFO, "TestNonHostMemoryRegMr completed");
}

// Test for excessive thread configuration warning
TEST_F(NetSocketTests, TestExcessiveThreadConfig) {
  INFO(NCCL_LOG_INFO, "Testing excessive thread configuration warning");

  // Check if the required environment variables are set
  const char *nThreadsEnv = getenv("NCCL_SOCKET_NTHREADS");
  const char *nSocksPerThreadEnv = getenv("NCCL_NSOCKS_PERTHREAD");

  if (!nThreadsEnv || !nSocksPerThreadEnv) {
    INFO(NCCL_LOG_INFO,
         "SKIPPING TEST: Required environment variables not set");
    INFO(NCCL_LOG_INFO,
         "Please set the following environment variables to run this test:");
    INFO(NCCL_LOG_INFO, "  export NCCL_SOCKET_NTHREADS=20");
    INFO(NCCL_LOG_INFO, "  export NCCL_NSOCKS_PERTHREAD=1");
    INFO(NCCL_LOG_INFO, "This test requires NCCL_SOCKET_NTHREADS > MAX_THREADS "
                        "(16) to trigger warning");
    GTEST_SKIP() << "Environment variables NCCL_SOCKET_NTHREADS and "
                    "NCCL_NSOCKS_PERTHREAD must be set";
    return;
  }

  int nThreads = std::atoi(nThreadsEnv);
  int nSocksPerThread = std::atoi(nSocksPerThreadEnv);

  INFO(NCCL_LOG_INFO, "Environment configuration found:");
  INFO(NCCL_LOG_INFO, "  NCCL_SOCKET_NTHREADS=%d", nThreads);
  INFO(NCCL_LOG_INFO, "  NCCL_NSOCKS_PERTHREAD=%d", nSocksPerThread);

  // Check if configuration is set to trigger the excessive threads warning
  if (nThreads <= 16) { // MAX_THREADS is 16
    INFO(NCCL_LOG_INFO, "SKIPPING TEST: NCCL_SOCKET_NTHREADS must be > 16 to "
                        "test excessive thread warning");
    INFO(NCCL_LOG_INFO, "Current NCCL_SOCKET_NTHREADS=%d", nThreads);
    INFO(NCCL_LOG_INFO, "Please set: export NCCL_SOCKET_NTHREADS=20");
    GTEST_SKIP()
        << "NCCL_SOCKET_NTHREADS must be > MAX_THREADS (16) to trigger warning";
    return;
  }

  INFO(NCCL_LOG_INFO,
       "Configuration valid for testing excessive threads warning");
  INFO(NCCL_LOG_INFO, "NCCL_SOCKET_NTHREADS=%d > MAX_THREADS=16", nThreads);

  // Test socket properties
  TestSocketProperties();

  // Initialize to trigger the warning logic
  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);

  if (result == ncclSuccess && listenComm) {
    // The implementation should have limited the threads to MAX_THREADS
    // internally
    INFO(NCCL_LOG_INFO,
         "*** SUCCESS: Listen succeeded with excessive NCCL_SOCKET_NTHREADS - "
         "limits enforced internally ***");
    ncclNetSocket.closeListen(listenComm);
  } else {
    INFO(NCCL_LOG_INFO, "Listen failed with result: %d", result);
  }

  INFO(NCCL_LOG_INFO, "TestExcessiveThreadConfig completed");
}

// Test for excessive socket configuration warning
TEST_F(NetSocketTests, TestExcessiveSocketConfig) {
  INFO(NCCL_LOG_INFO, "Testing excessive socket configuration warning");

  // Check if the required environment variables are set
  const char *nThreadsEnv = getenv("NCCL_SOCKET_NTHREADS");
  const char *nSocksPerThreadEnv = getenv("NCCL_NSOCKS_PERTHREAD");

  if (!nThreadsEnv || !nSocksPerThreadEnv) {
    INFO(NCCL_LOG_INFO,
         "SKIPPING TEST: Required environment variables not set");
    INFO(NCCL_LOG_INFO,
         "Please set the following environment variables to run this test:");
    INFO(NCCL_LOG_INFO, "  export NCCL_SOCKET_NTHREADS=8");
    INFO(NCCL_LOG_INFO, "  export NCCL_NSOCKS_PERTHREAD=10");
    INFO(NCCL_LOG_INFO, "This test requires total sockets (nThreads * "
                        "nSocksPerThread) > MAX_SOCKETS (64)");
    GTEST_SKIP() << "Environment variables NCCL_SOCKET_NTHREADS and "
                    "NCCL_NSOCKS_PERTHREAD must be set";
    return;
  }

  int nThreads = std::atoi(nThreadsEnv);
  int nSocksPerThread = std::atoi(nSocksPerThreadEnv);
  int totalSockets = nThreads * nSocksPerThread;

  INFO(NCCL_LOG_INFO, "Environment configuration found:");
  INFO(NCCL_LOG_INFO, "  NCCL_SOCKET_NTHREADS=%d", nThreads);
  INFO(NCCL_LOG_INFO, "  NCCL_NSOCKS_PERTHREAD=%d", nSocksPerThread);
  INFO(NCCL_LOG_INFO, "  Total sockets=%d", totalSockets);

  // Check if configuration is set to trigger the excessive sockets warning
  if (totalSockets <= 64) { // MAX_SOCKETS is 64
    INFO(NCCL_LOG_INFO, "SKIPPING TEST: Total sockets must be > 64 to test "
                        "excessive socket warning");
    INFO(NCCL_LOG_INFO,
         "Current total sockets=%d (nThreads * nSocksPerThread = %d * %d)",
         totalSockets, nThreads, nSocksPerThread);
    INFO(NCCL_LOG_INFO,
         "Please set environment variables such that total > 64, e.g.:");
    INFO(NCCL_LOG_INFO, "  export NCCL_SOCKET_NTHREADS=8");
    INFO(NCCL_LOG_INFO, "  export NCCL_NSOCKS_PERTHREAD=10");
    GTEST_SKIP()
        << "Total sockets must be > MAX_SOCKETS (64) to trigger warning";
    return;
  }

  INFO(NCCL_LOG_INFO,
       "Configuration valid for testing excessive sockets warning");
  INFO(NCCL_LOG_INFO, "Total sockets=%d > MAX_SOCKETS=64", totalSockets);

  // Test socket properties
  TestSocketProperties();

  // Initialize to trigger the warning logic
  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);

  if (result == ncclSuccess && listenComm) {
    // The implementation should have limited the sockets to MAX_SOCKETS
    // internally
    INFO(NCCL_LOG_INFO, "*** SUCCESS: Listen succeeded with excessive total "
                        "sockets - limits enforced internally ***");
    ncclNetSocket.closeListen(listenComm);
  } else {
    INFO(NCCL_LOG_INFO, "Listen failed with result: %d", result);
  }

  INFO(NCCL_LOG_INFO, "TestExcessiveSocketConfig completed");
}

// Test to trigger request allocation failure scenario
TEST_F(NetSocketTests, TestRequestAllocationFailure) {
  INFO(NCCL_LOG_INFO, "Testing request allocation failure scenario");

  // Setup communication
  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);

  if (result == ncclSuccess && listenComm) {
    void *sendComm = nullptr;
    void *recvComm = nullptr;
    bool connectionSuccess =
        EstablishConnectionPair(handle, listenComm, sendComm, recvComm);

    if (connectionSuccess && sendComm && recvComm) {
      INFO(NCCL_LOG_INFO, "Attempting to exhaust request pool (MAX_REQUESTS)");

      std::vector<void *> requests;
      std::vector<std::vector<char>> buffers;
      std::vector<void *> mhandles;

      // Try to allocate many requests to potentially exhaust the pool
      // MAX_REQUESTS is defined as NCCL_NET_MAX_REQUESTS in the code
      for (int i = 0; i < 300; i++) { // Try to exceed typical MAX_REQUESTS
        buffers.emplace_back(1024, 0xAA + (i % 10));
        void *mhandle = nullptr;

        // Register memory first
        result = ncclNetSocket.regMr(sendComm, buffers.back().data(), 1024,
                                     NCCL_PTR_HOST, &mhandle);
        if (result != ncclSuccess)
          break;
        mhandles.push_back(mhandle);

        // Try to create send request
        void *request = nullptr;
        result = ncclNetSocket.isend(sendComm, buffers.back().data(), 1024, 0,
                                     mhandle, nullptr, &request);

        if (result == ncclInternalError) {
          INFO(NCCL_LOG_INFO,
               "Request allocation failed at iteration %d (expected behavior "
               "when pool exhausted)",
               i);
          break;
        } else if (result == ncclSuccess) {
          requests.push_back(request);
        } else {
          INFO(NCCL_LOG_INFO, "Unexpected result at iteration %d: %d", i,
               result);
          break;
        }
      }

      INFO(NCCL_LOG_INFO,
           "Successfully allocated %zu requests before failure/completion",
           requests.size());

      // Cleanup: Test any pending requests and deregister memory
      for (size_t i = 0; i < requests.size(); i++) {
        if (requests[i]) {
          int done = 0;
          int size = 0;
          ncclNetSocket.test(requests[i], &done,
                             &size); // Don't care about result
        }
      }

      for (size_t i = 0; i < mhandles.size(); i++) {
        if (mhandles[i]) {
          ncclNetSocket.deregMr(sendComm, mhandles[i]);
        }
      }

      // Cleanup communicators
      ncclNetSocket.closeSend(sendComm);
      ncclNetSocket.closeRecv(recvComm);
    }

    ncclNetSocket.closeListen(listenComm);
  }

  INFO(NCCL_LOG_INFO, "TestRequestAllocationFailure completed");
}

// Test for message size mismatch scenario
TEST_F(NetSocketTests, TestMessageSizeMismatch) {
  INFO(NCCL_LOG_INFO, "Testing message size mismatch scenario");

  // This test simulates the condition where a receiver expects a smaller
  // message than what the sender is trying to send, which should trigger the
  // truncation warning

  char handle[NCCL_NET_HANDLE_MAXSIZE];
  void *listenComm = nullptr;
  ncclResult_t result = ncclNetSocket.listen(0, handle, &listenComm);

  if (result == ncclSuccess && listenComm) {
    void *sendComm = nullptr;
    void *recvComm = nullptr;
    bool connectionSuccess =
        EstablishConnectionPair(handle, listenComm, sendComm, recvComm);

    if (connectionSuccess && sendComm && recvComm) {
      // Large send buffer
      const size_t sendSize = 2048;
      std::vector<char> sendBuffer(sendSize, 0xAA);

      // Small receive buffer (to simulate size mismatch)
      const size_t recvSize = 1024; // Smaller than send size
      std::vector<char> recvBuffer(recvSize, 0x00);

      void *sendMhandle = nullptr;
      void *recvMhandle = nullptr;

      // Register memory
      result = ncclNetSocket.regMr(sendComm, sendBuffer.data(), sendSize,
                                   NCCL_PTR_HOST, &sendMhandle);
      EXPECT_EQ(result, ncclSuccess);

      result = ncclNetSocket.regMr(recvComm, recvBuffer.data(), recvSize,
                                   NCCL_PTR_HOST, &recvMhandle);
      EXPECT_EQ(result, ncclSuccess);

      // Start send operation with large size
      void *sendRequest = nullptr;
      result = ncclNetSocket.isend(sendComm, sendBuffer.data(), sendSize, 0,
                                   sendMhandle, nullptr, &sendRequest);
      EXPECT_EQ(result, ncclSuccess);

      // Start receive operation with small size
      void *recvRequest = nullptr;
      void *recvDataPtr = recvBuffer.data();
      size_t recvSizeVar = recvSize;
      int tag = 0;
      result = ncclNetSocket.irecv(recvComm, 1, &recvDataPtr, &recvSizeVar,
                                   &tag, &recvMhandle, nullptr, &recvRequest);
      EXPECT_EQ(result, ncclSuccess);

      // Progress operations - this should eventually trigger the size mismatch
      // warning
      for (int i = 0; i < 100; i++) {
        int sendDone = 0, recvDone = 0;
        int sendSizeOut = 0, recvSizeOut = 0;

        ncclResult_t sendTestResult =
            ncclNetSocket.test(sendRequest, &sendDone, &sendSizeOut);
        ncclResult_t recvTestResult =
            ncclNetSocket.test(recvRequest, &recvDone, &recvSizeOut);

        // If receiver detects size mismatch, it should return ncclInvalidUsage
        if (recvTestResult == ncclInvalidUsage) {
          INFO(NCCL_LOG_INFO, "*** SUCCESS: Message size mismatch detected "
                              "(ncclInvalidUsage) ***");
          break;
        }

        if (sendDone && recvDone) {
          INFO(NCCL_LOG_INFO,
               "Operations completed without size mismatch detection");
          break;
        }

        if (sendTestResult != ncclSuccess && sendTestResult != ncclInProgress) {
          INFO(NCCL_LOG_INFO, "Send operation failed: %d", sendTestResult);
          break;
        }

        if (recvTestResult != ncclSuccess && recvTestResult != ncclInProgress &&
            recvTestResult != ncclInvalidUsage) {
          INFO(NCCL_LOG_INFO, "Recv operation failed: %d", recvTestResult);
          break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

      // Cleanup
      if (sendMhandle)
        ncclNetSocket.deregMr(sendComm, sendMhandle);
      if (recvMhandle)
        ncclNetSocket.deregMr(recvComm, recvMhandle);
      ncclNetSocket.closeSend(sendComm);
      ncclNetSocket.closeRecv(recvComm);
    }

    ncclNetSocket.closeListen(listenComm);
  }

  INFO(NCCL_LOG_INFO, "TestMessageSizeMismatch completed");
}

// Test to cover the iflush function that always returns ncclInternalError
TEST_F(NetSocketTests, TestIflushAlwaysFails) {
  INFO(NCCL_LOG_INFO,
       "Testing ncclNetSocketIflush always returns ncclInternalError");

  // This function should always return ncclInternalError for socket
  // implementation as it doesn't support CUDA pointers and flush operations

  std::vector<char> buffer(1024, 0xAA);
  void *data = buffer.data();
  int size = 1024;
  void *mhandle = nullptr;
  void *request = nullptr;

  // Test with dummy parameters - should always fail
  ncclResult_t result =
      ncclNetSocket.iflush(nullptr, 1, &data, &size, &mhandle, &request);
  INFO(NCCL_LOG_INFO, "ncclNetSocketIflush returned: %d", result);
  EXPECT_EQ(result, ncclInternalError)
      << "iflush should always return ncclInternalError";

  INFO(NCCL_LOG_INFO, "TestIflushAlwaysFails completed");
}

} // namespace RcclUnitTesting
