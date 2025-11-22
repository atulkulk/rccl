/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "ProcessIsolatedTestRunner.hpp"
#include <atomic>
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

namespace RcclUnitTesting {

// Exit codes for test process results
enum RcclTestCode {
  RCCL_TEST_INVALID = -1,
  RCCL_TEST_SUCCESS = 0,
  RCCL_TEST_FAILURE = 1,
  RCCL_TEST_UNKNOWN_EXCEPTION = 2,
  RCCL_TEST_TIMEOUT = 3,
  RCCL_TEST_SKIPPED = 4
};

// Define static members
std::mutex ProcessIsolatedTestRunner::testConfigsMutex_;
std::vector<ProcessIsolatedTestRunner::TestConfig>
    ProcessIsolatedTestRunner::testConfigs_;
std::mutex ProcessIsolatedTestRunner::resultsMutex_;
std::vector<ProcessIsolatedTestRunner::TestResult>
    ProcessIsolatedTestRunner::testResults_;

// TestResult implementation
ProcessIsolatedTestRunner::TestResult::TestResult()
    : passed(false), skipped(false), exitCode(-1), processId(-1), duration(0) {}

// TestConfig implementation
ProcessIsolatedTestRunner::TestConfig::TestConfig(const std::string &testName,
                                                  std::function<void()> logic)
    : name(testName), testLogic(logic), timeout(30), inheritParentEnv(true) {}

ProcessIsolatedTestRunner::TestConfig &
ProcessIsolatedTestRunner::TestConfig::withEnvironment(
    const std::unordered_map<std::string, std::string> &env) {
  environmentVariables = env;
  return *this;
}

ProcessIsolatedTestRunner::TestConfig &
ProcessIsolatedTestRunner::TestConfig::withTimeout(
    std::chrono::seconds timeoutSeconds) {
  timeout = timeoutSeconds;
  return *this;
}

ProcessIsolatedTestRunner::TestConfig &
ProcessIsolatedTestRunner::TestConfig::withCleanEnvironment(bool inherit) {
  inheritParentEnv = inherit;
  return *this;
}

ProcessIsolatedTestRunner::TestConfig &
ProcessIsolatedTestRunner::TestConfig::clearVariable(
    const std::string &varName) {
  clearEnvVars.push_back(varName);
  return *this;
}

ProcessIsolatedTestRunner::TestConfig &
ProcessIsolatedTestRunner::TestConfig::setVariable(const std::string &name,
                                                   const std::string &value) {
  environmentVariables[name] = value;
  return *this;
}

// ExecutionOptions implementation
ProcessIsolatedTestRunner::ExecutionOptions::ExecutionOptions()
    : stopOnFirstFailure(false), verboseLogging(true) {}

// Apply environment variables to current process
void ProcessIsolatedTestRunner::applyEnvironmentVariables(
    const TestConfig &config) {
  // Clear specified environment variables first
  for (const auto &varName : config.clearEnvVars) {
    unsetenv(varName.c_str());
  }

  // If not inheriting parent environment, clear all environment variables
  if (!config.inheritParentEnv) {
    extern char **environ;
    // Save the variables we want to set
    std::unordered_map<std::string, std::string> varsToSet =
        config.environmentVariables;

    // Set our variables
    for (const auto &[name, value] : varsToSet) {
      setenv(name.c_str(), value.c_str(), 1);
    }
  } else {
    // Just set/override the specified variables
    for (const auto &[name, value] : config.environmentVariables) {
      setenv(name.c_str(), value.c_str(), 1);
    }
  }
}

// Execute a single test in a separate process
int ProcessIsolatedTestRunner::runTestInProcess(const TestConfig &config) {
  pid_t processId = getpid();

  if (config.name.empty()) {
    std::cerr << "Error: Test name is empty for process " << processId
              << std::endl;
    return RCCL_TEST_FAILURE;
  }

  try {
    // Apply environment variables
    applyEnvironmentVariables(config);

    if (getenv("VERBOSE_LOGGING") || getenv("PROCESS_RUNNER_VERBOSE")) {
      std::cout << "Process " << processId << " executing test '" << config.name
                << "' with environment modifications" << std::endl;

      // Log environment variables being set
      for (const auto &[name, value] : config.environmentVariables) {
        std::cout << "  " << name << "=" << value << std::endl;
      }
    }

    // Thread-safe test execution with timeout protection
    std::atomic<bool> testCompleted{false};
    std::exception_ptr testException = nullptr;
    bool testPassed = true;
    bool testSkipped = false;

    // Run test in a separate thread to allow timeout handling
    std::thread testThread([&]() {
      try {
        // Get initial test state
        const ::testing::UnitTest *unitTest =
            ::testing::UnitTest::GetInstance();
        size_t initialFailureCount = unitTest->failed_test_count();
        size_t initialSkippedCount = unitTest->skipped_test_count();

        // Execute the test logic
        config.testLogic();

        // Check if any new test failures occurred
        size_t finalFailureCount = unitTest->failed_test_count();
        size_t finalSkippedCount = unitTest->skipped_test_count();

        testPassed = (finalFailureCount == initialFailureCount);
        testSkipped = (finalSkippedCount > initialSkippedCount);

        testCompleted = true;
      } catch (...) {
        testException = std::current_exception();
        testPassed = false;
        testCompleted = true;
      }
    });

    // Wait for test completion with timeout
    auto start = std::chrono::steady_clock::now();
    const auto timeout = config.timeout;

    while (!testCompleted.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      if (std::chrono::steady_clock::now() - start > timeout) {
        // Test timed out
        std::cout << "Test '" << config.name << "' timed out after "
                  << timeout.count() << " seconds in process " << processId
                  << std::endl;

        // Detach thread and exit
        testThread.detach();
        return RCCL_TEST_TIMEOUT; // Timeout exit code
      }
    }

    // Wait for thread completion
    if (testThread.joinable()) {
      testThread.join();
    }

    // Check if test threw an exception
    if (testException) {
      std::rethrow_exception(testException);
    }

    // Return appropriate exit code based on test result
    if (testSkipped) {
      std::cout << "Test '" << config.name << "' SKIPPED in process "
                << processId << std::endl;
      return RCCL_TEST_SKIPPED; // Skipped exit code
    } else if (testPassed) {
      std::cout << "Test '" << config.name
                << "' completed successfully in process " << processId
                << std::endl;
      return RCCL_TEST_SUCCESS; // Success
    } else {
      std::cout << "Test '" << config.name
                << "' failed with assertion failures in process " << processId
                << std::endl;
      return RCCL_TEST_FAILURE; // Test assertion failure
    }

  } catch (const std::exception &e) {
    std::cerr << "Test '" << config.name << "' failed in process " << processId
              << ": " << e.what() << std::endl;
    return RCCL_TEST_FAILURE; // Failure
  } catch (...) {
    std::cerr << "Test '" << config.name << "' failed in process " << processId
              << " with unknown exception" << std::endl;
    return RCCL_TEST_UNKNOWN_EXCEPTION; // Unknown failure
  }
}

// Register a test configuration
void ProcessIsolatedTestRunner::registerTest(const TestConfig &config) {
  std::lock_guard<std::mutex> lock(testConfigsMutex_);
  testConfigs_.push_back(config);
}

// Register a simple test with just name and logic
void ProcessIsolatedTestRunner::registerTest(const std::string &name,
                                             std::function<void()> testLogic) {
  registerTest(TestConfig(name, testLogic));
}

// Register a test with environment variables
void ProcessIsolatedTestRunner::registerTest(
    const std::string &name, std::function<void()> testLogic,
    const std::unordered_map<std::string, std::string> &env) {
  registerTest(TestConfig(name, testLogic).withEnvironment(env));
}

// Record test result (thread-safe)
void ProcessIsolatedTestRunner::recordTestResult(const TestResult &result) {
  std::lock_guard<std::mutex> lock(resultsMutex_);
  testResults_.push_back(result);
}

// Execute all registered tests (simplified sequential execution only)
bool ProcessIsolatedTestRunner::executeAllTests(
    const ExecutionOptions &options) {
  std::cout << "\n=== Starting Process-Isolated Test Execution (Sequential) ==="
            << std::endl;

  // Get test configurations to run
  std::vector<TestConfig> testsToRun;
  {
    std::lock_guard<std::mutex> lock(testConfigsMutex_);
    testsToRun = testConfigs_;
  }

  // Clear previous results
  {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    testResults_.clear();
  }

  std::cout << "Executing " << testsToRun.size() << " tests" << std::endl;

  // Sequential execution
  for (const auto &testConfig : testsToRun) {
    auto startTime = std::chrono::steady_clock::now();

    pid_t pid = fork();

    if (pid == 0) {
      // Child process - execute the test
      int result = runTestInProcess(testConfig);
      exit(result);
    } else if (pid > 0) {
      // Parent process - wait for child
      int status;
      pid_t result = waitpid(pid, &status, 0);
      auto endTime = std::chrono::steady_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
          endTime - startTime);

      TestResult testResult;
      testResult.testName = testConfig.name;
      testResult.processId = pid;
      testResult.duration = duration;

      if (result == pid) {
        if (WIFEXITED(status)) {
          int exitCode = WEXITSTATUS(status);
          testResult.exitCode = exitCode;
          testResult.passed = (exitCode == RCCL_TEST_SUCCESS);
          testResult.skipped = (exitCode == RCCL_TEST_SKIPPED);

          if (exitCode == RCCL_TEST_SUCCESS) {
            std::cout << "Test '" << testConfig.name << "' (PID: " << pid
                      << ") PASSED in " << duration.count() << " ms"
                      << std::endl;
          } else if (exitCode == RCCL_TEST_TIMEOUT) {
            std::cout << "Test '" << testConfig.name << "' (PID: " << pid
                      << ") TIMED OUT after " << duration.count() << " ms"
                      << std::endl;
            testResult.errorMessage = "Test timed out";
          } else if (exitCode == RCCL_TEST_SKIPPED) {
            std::cout << "Test '" << testConfig.name << "' (PID: " << pid
                      << ") SKIPPED in " << duration.count() << " ms"
                      << std::endl;
            testResult.errorMessage = "Test skipped";
          } else {
            std::cout << "Test '" << testConfig.name << "' (PID: " << pid
                      << ") FAILED with exit code " << exitCode << " after "
                      << duration.count() << " ms" << std::endl;
            testResult.errorMessage =
                "Test failed with exit code " + std::to_string(exitCode);
          }
        } else if (WIFSIGNALED(status)) {
          int signal = WTERMSIG(status);
          testResult.passed = false;
          testResult.skipped = false;
          testResult.exitCode = -signal;
          testResult.errorMessage =
              "Terminated by signal " + std::to_string(signal);

          std::cout << "Test '" << testConfig.name << "' (PID: " << pid
                    << ") terminated by signal " << signal << " after "
                    << duration.count() << " ms" << std::endl;
        }
      } else {
        testResult.passed = false;
        testResult.skipped = false;
        testResult.exitCode = RCCL_TEST_INVALID;
        testResult.errorMessage = "Failed to wait for process";
        std::cout << "Test '" << testConfig.name
                  << "' failed to wait for process" << std::endl;
      }

      recordTestResult(testResult);

      // Stop on first failure if requested
      if (options.stopOnFirstFailure && !testResult.passed &&
          !testResult.skipped) {
        std::cout << "Stopping execution due to first failure." << std::endl;
        break;
      }
    } else {
      // Fork failed
      TestResult testResult;
      testResult.testName = testConfig.name;
      testResult.passed = false;
      testResult.skipped = false;
      testResult.exitCode = RCCL_TEST_INVALID;
      testResult.processId = RCCL_TEST_INVALID;
      testResult.duration = std::chrono::milliseconds(0);
      testResult.errorMessage = "Failed to fork process";

      recordTestResult(testResult);
      std::cout << "Failed to fork process for test '" << testConfig.name << "'"
                << std::endl;

      if (options.stopOnFirstFailure) {
        std::cout << "Stopping execution due to fork failure." << std::endl;
        break;
      }
    }
  }

  bool result = generateReport(options);

  // Automatically clear test configurations and results after execution
  // This ensures a clean state for the next test suite without requiring
  // explicit clear() calls from test cases
  {
    std::lock_guard<std::mutex> lock(testConfigsMutex_);
    testConfigs_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    testResults_.clear();
  }

  return result;
}

// Generate and display test report
bool ProcessIsolatedTestRunner::generateReport(
    const ExecutionOptions &options) {
  int totalTests = 0;
  int passedTests = 0;
  int failedTests = 0;
  int skippedTests = 0;
  std::chrono::milliseconds totalDuration{0};

  {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    totalTests = testResults_.size();

    for (const auto &result : testResults_) {
      if (result.skipped) {
        skippedTests++;
      } else if (result.passed) {
        passedTests++;
      } else {
        failedTests++;
      }
      totalDuration += result.duration;
    }
  }

  // Report final results
  std::cout << "\n=== Process-Isolated Test Summary ===" << std::endl;
  std::cout << "Total Tests: " << totalTests
            << "  |  Passed: " << passedTests
            << "  |  Skipped: " << skippedTests
            << "  |  Failed: " << failedTests << std::endl;
  std::cout << "Total Duration: " << totalDuration.count() << " ms"
            << "  |  Execution Mode: Sequential" << std::endl;

  if (options.verboseLogging && failedTests > 0) {
    std::cout << "\nFailed Tests Details:" << std::endl;
    std::lock_guard<std::mutex> lock(resultsMutex_);
    for (const auto &result : testResults_) {
      if (!result.passed && !result.skipped) {
        std::cout << "  - " << result.testName << ": " << result.errorMessage
                  << std::endl;
      }
    }
  }

  std::cout << "======================================\n" << std::endl;

  return failedTests == 0;
}

// Get detailed test results (thread-safe)
std::vector<ProcessIsolatedTestRunner::TestResult>
ProcessIsolatedTestRunner::getTestResults() {
  std::lock_guard<std::mutex> lock(resultsMutex_);
  return testResults_;
}

// Clear test registry and results (thread-safe)
void ProcessIsolatedTestRunner::clear() {
  size_t registeredCount = 0;
  size_t executedCount = 0;

  // Check for unexecuted tests before clearing
  {
    std::lock_guard<std::mutex> lock(testConfigsMutex_);
    registeredCount = testConfigs_.size();
  }
  {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    executedCount = testResults_.size();
  }

  // Warn if tests were registered but not all executed
  if (registeredCount > 0 && executedCount < registeredCount) {
    std::cerr << "\n⚠️  WARNING: ProcessIsolatedTestRunner::clear() called with "
              << (registeredCount - executedCount) << " unexecuted test(s)!\n"
              << "   Registered: " << registeredCount << " test(s)\n"
              << "   Executed:   " << executedCount << " test(s)\n"
              << "   Did you forget to call executeAllTests()?\n" << std::endl;
  }

  // Clear the registrations and results
  {
    std::lock_guard<std::mutex> lock(testConfigsMutex_);
    testConfigs_.clear();
  }
  {
    std::lock_guard<std::mutex> lock(resultsMutex_);
    testResults_.clear();
  }
}

// Get number of registered tests
size_t ProcessIsolatedTestRunner::getTestCount() {
  std::lock_guard<std::mutex> lock(testConfigsMutex_);
  return testConfigs_.size();
}

} // namespace RcclUnitTesting
