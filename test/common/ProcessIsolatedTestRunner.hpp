/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

#include <chrono>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

namespace RcclUnitTesting {

/**
 * @brief Generic thread-safe process isolated test runner
 *
 * This class provides a framework for running tests in isolated processes
 * with clean environment settings and sequential execution.
 *
 */
class ProcessIsolatedTestRunner {
public:
  /**
   * @brief Test execution result structure
   */
  struct TestResult {
    std::string testName;               ///< Name of the test
    bool passed;                        ///< Whether the test passed
    bool skipped;                       ///< Whether the test skipped
    int exitCode;                       ///< Process exit code
    pid_t processId;                    ///< Process ID that ran the test
    std::chrono::milliseconds duration; ///< Test execution duration
    std::string errorMessage;           ///< Error message if test failed
    std::unordered_map<std::string, std::string>
        environment; ///< Environment variables used

    /**
     * @brief Default constructor
     */
    TestResult();
  };

  /**
   * @brief Test configuration structure
   */
  struct TestConfig {
    std::string name;                ///< Test name
    std::function<void()> testLogic; ///< Test function to execute
    std::unordered_map<std::string, std::string>
        environmentVariables;     ///< Environment variables to set
    std::chrono::seconds timeout; ///< Test timeout
    bool inheritParentEnv;        ///< Whether to inherit parent environment
    std::vector<std::string>
        clearEnvVars; ///< Environment variables to explicitly clear

    /**
     * @brief Constructor
     * @param testName Name of the test
     * @param logic Test function to execute
     */
    TestConfig(const std::string &testName, std::function<void()> logic);

    /**
     * @brief Set environment variables for this test
     * @param env Map of environment variable name-value pairs
     * @return Reference to this TestConfig for method chaining
     */
    TestConfig &
    withEnvironment(const std::unordered_map<std::string, std::string> &env);

    /**
     * @brief Set timeout for this test
     * @param timeoutSeconds Timeout in seconds
     * @return Reference to this TestConfig for method chaining
     */
    TestConfig &withTimeout(std::chrono::seconds timeoutSeconds);

    /**
     * @brief Configure environment inheritance
     * @param inherit Whether to inherit parent environment variables
     * @return Reference to this TestConfig for method chaining
     */
    TestConfig &withCleanEnvironment(bool inherit = false);

    /**
     * @brief Clear a specific environment variable
     * @param varName Name of the variable to clear
     * @return Reference to this TestConfig for method chaining
     */
    TestConfig &clearVariable(const std::string &varName);

    /**
     * @brief Set a specific environment variable
     * @param name Variable name
     * @param value Variable value
     * @return Reference to this TestConfig for method chaining
     */
    TestConfig &setVariable(const std::string &name, const std::string &value);
  };

  /**
   * @brief Execution options for test runner
   */
  struct ExecutionOptions {
    bool stopOnFirstFailure; ///< Stop execution on first test failure
    bool verboseLogging;     ///< Enable verbose logging

    /**
     * @brief Default constructor with sensible defaults
     */
    ExecutionOptions();
  };

private:
  // Thread-safe static members for test management
  static std::mutex testConfigsMutex_;
  static std::vector<TestConfig> testConfigs_;
  static std::mutex resultsMutex_;
  static std::vector<TestResult> testResults_;

  /**
   * @brief Apply environment variables to current process
   * @param config Test configuration containing environment settings
   */
  static void applyEnvironmentVariables(const TestConfig &config);

  /**
   * @brief Execute a single test in the child process
   * @param config Test configuration
   * @return Exit code (0 for success, non-zero for failure)
   */
  static int runTestInProcess(const TestConfig &config);

public:
  /**
   * @brief Register a test configuration
   * @param config Complete test configuration
   */
  static void registerTest(const TestConfig &config);

  /**
   * @brief Register a simple test with just name and logic
   * @param name Test name
   * @param testLogic Test function to execute
   */
  static void registerTest(const std::string &name,
                           std::function<void()> testLogic);

  /**
   * @brief Register a test with environment variables
   * @param name Test name
   * @param testLogic Test function to execute
   * @param env Environment variables to set for this test
   */
  static void
  registerTest(const std::string &name, std::function<void()> testLogic,
               const std::unordered_map<std::string, std::string> &env);

  /**
   * @brief Record a test result (thread-safe)
   * @param result Test result to record
   */
  static void recordTestResult(const TestResult &result);

  /**
   * @brief Execute all registered tests sequentially
   * @param options Execution options (defaults to continue on failure)
   * @return True if all tests passed, false otherwise
   * @note This method automatically clears all test registrations and results
   *       after execution, ensuring a clean state for the next test suite.
   */
  static bool
  executeAllTests(const ExecutionOptions &options = ExecutionOptions());

  /**
   * @brief Generate and display test report
   * @param options Execution options used for the test run
   * @return True if all tests passed, false otherwise
   */
  static bool generateReport(const ExecutionOptions &options);

  /**
   * @brief Get detailed test results (thread-safe)
   * @return Vector of all test results
   */
  static std::vector<TestResult> getTestResults();

  /**
   * @brief Clear test registry and results (thread-safe)
   * @note Calling this method manually is typically not necessary, as
   *       executeAllTests() automatically clears registrations after execution.
   *       This method is primarily useful for advanced use cases or when tests
   *       are registered but not executed.
   */
  static void clear();

  /**
   * @brief Get number of registered tests
   * @return Number of registered tests
   */
  static size_t getTestCount();
};

} // namespace RcclUnitTesting
