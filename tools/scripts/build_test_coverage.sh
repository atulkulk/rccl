#!/bin/bash
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE.txt for license information

set -e
# set -x

readonly EXIT_SUCCESS=0
readonly EXIT_FAILURE=1
readonly EXIT_TIMEOUT=124

readonly TEST_RESULT_PASSED="PASSED"
readonly TEST_RESULT_FAILED="FAILED"
readonly TEST_RESULT_TIMEOUT="TIMEOUT"

readonly DEFAULT_TIMEOUT=0
readonly DEFAULT_NUM_RANKS=1
readonly VERBOSE_OFF=0
readonly VERBOSE_ON=1

readonly ENV_NONE="NONE"
readonly GTEST_FILTER_ALL="ALL"

check_environment() {
  local BUILD_DIR="$1"
  echo "Checking environment..."

  # Check ROCm installation
  if [[ ! -d "$ROCM_PATH" ]]; then
    echo "ERROR: ROCm not found at $ROCM_PATH"
    return $EXIT_FAILURE
  fi

  # Check MPI installation (when MPI tests are enabled)
  if [ -n "$MPI_PATH" ]; then
    if [[ ! -d "$MPI_PATH" ]]; then
      echo "ERROR: MPI path not found: $MPI_PATH"
      return $EXIT_FAILURE
    fi
    if [[ ! -f "$MPI_PATH/bin/mpirun" ]]; then
      echo "ERROR: mpirun not found in MPI path: $MPI_PATH/bin/mpirun"
      return $EXIT_FAILURE
    fi
    echo "Using MPI installation at: $MPI_PATH"
  fi

  # Check RCCL library
  if [[ ! -f "${BUILD_DIR}/librccl.so" ]]; then
    echo "ERROR: RCCL library not found at ${BUILD_DIR}/librccl.so"
    return $EXIT_FAILURE
  fi

  echo "Environment validation passed"
  return $EXIT_SUCCESS
}

run_tests_from_config() {
  local CONFIG_FILE="$1"
  local BUILD_DIR="$2"
  local LOG_DIR="$3"
  local TEST_NAME_FILTER="$4"  # Optional: run specific test by name
  local VERBOSE="${5:-$VERBOSE_OFF}"  # Optional: verbose mode (default: off)

  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Test configuration file not found: $CONFIG_FILE"
    return $EXIT_FAILURE
  fi

  # Check environment first
  if ! check_environment "$BUILD_DIR"; then
    echo "Environment check failed"
    return $EXIT_FAILURE
  fi

  echo "========================================"
  echo "Running tests from configuration file"
  echo "Config: $CONFIG_FILE"
  echo "Build:  $BUILD_DIR"
  echo "========================================"
  echo ""

  cd "${BUILD_DIR}/test"

  # Arrays to track test results
  declare -a TEST_NAMES=()
  declare -a TEST_RESULTS=()
  declare -a TEST_DURATIONS=()
  local TOTAL_TESTS=0
  local PASSED_TESTS=0
  local FAILED_TESTS=0

  # Arrays to store parsed test configurations
  local TEST_NAMES_PARSED=()
  local TEST_BINARIES=()
  local TEST_NUM_RANKS=()
  local TEST_TIMEOUTS=()
  local TEST_ENV_VARS=()
  local TEST_GTEST_FILTERS=()
  local TEST_NUM_NODES=()

  echo ""
  echo "========================================"
  echo "PARSING CONFIG FILE"
  echo "========================================"
  echo ""

  # Read and parse config file
  local LINE_NUM=0
  local LINES_PROCESSED=0
  local LINES_SKIPPED_COMMENT=0
  local LINES_SKIPPED_EMPTY=0

  while IFS='|' read -r TEST_NAME ENV_VARS BINARY GTEST_FILTER NUM_NODES NUM_RANKS TIMEOUT || [ -n "$TEST_NAME" ]; do
    LINE_NUM=$((LINE_NUM + 1))
    LINES_PROCESSED=$((LINES_PROCESSED + 1))

    # Skip comments and empty lines
    if [[ "$TEST_NAME" =~ ^[[:space:]]*# ]]; then
      LINES_SKIPPED_COMMENT=$((LINES_SKIPPED_COMMENT + 1))
      continue
    fi

    if [[ -z "${TEST_NAME// }" ]]; then
      LINES_SKIPPED_EMPTY=$((LINES_SKIPPED_EMPTY + 1))
      continue
    fi

    # Trim whitespace from all fields
    TEST_NAME=$(echo "$TEST_NAME" | xargs)
    ENV_VARS=$(echo "$ENV_VARS" | xargs)
    BINARY=$(echo "$BINARY" | xargs)
    GTEST_FILTER=$(echo "$GTEST_FILTER" | xargs)
    NUM_NODES=$(echo "$NUM_NODES" | xargs)
    NUM_RANKS=$(echo "$NUM_RANKS" | xargs)
    TIMEOUT=$(echo "$TIMEOUT" | xargs)

    # Set defaults for optional fields
    NUM_NODES=${NUM_NODES:-1}
    NUM_RANKS=${NUM_RANKS:-$DEFAULT_NUM_RANKS}
    TIMEOUT=${TIMEOUT:-$DEFAULT_TIMEOUT}

    if [[ "$VERBOSE" -eq 1 ]]; then
      echo "----------------------------------------"
      echo "Line $LINE_NUM: Parsing test line"
      echo "  Test Name:    '$TEST_NAME'"
      echo "  Env Vars:     '$ENV_VARS'"
      echo "  Binary:       '$BINARY'"
      echo "  Gtest Filter: '$GTEST_FILTER'"
      echo "  Num Nodes:    '$NUM_NODES'"
      echo "  Num Ranks:    '$NUM_RANKS'"
      echo "  Timeout:      '$TIMEOUT'"
    fi

    # Skip if test name filter is set and doesn't match
    if [[ -n "$TEST_NAME_FILTER" ]] && [[ "$TEST_NAME" != "$TEST_NAME_FILTER" ]]; then
      if [[ "$VERBOSE" -eq 1 ]]; then
        echo "  Status: SKIPPED (filter '$TEST_NAME_FILTER' doesn't match)"
        echo ""
      fi
      continue
    fi

    # Validate required fields (NUM_NODES, NUM_RANKS, and TIMEOUT have defaults, so not required)
    if [[ -z "$TEST_NAME" ]] || [[ -z "$BINARY" ]] || [[ -z "$GTEST_FILTER" ]]; then
      if [[ "$VERBOSE" -eq $VERBOSE_ON ]]; then
        echo "  Status: SKIPPED (Incomplete configuration - missing required field)"
        echo ""
      fi
      continue
    fi

    # Validate binary exists
    if [[ ! -f "$BINARY" ]]; then
      if [[ "$VERBOSE" -eq $VERBOSE_ON ]]; then
        echo "  Status: SKIPPED (Binary not found: $BINARY)"
        echo ""
      fi
      continue
    fi

    # Validate NUM_RANKS is a positive integer
    if ! [[ "$NUM_RANKS" =~ ^[0-9]+$ ]] || [[ "$NUM_RANKS" -lt 1 ]]; then
      if [[ "$VERBOSE" -eq $VERBOSE_ON ]]; then
        echo "  Status: SKIPPED (Invalid NUM_RANKS: $NUM_RANKS)"
        echo ""
      fi
      continue
    fi

    # Validate TIMEOUT is a non-negative integer
    if ! [[ "$TIMEOUT" =~ ^[0-9]+$ ]]; then
      if [[ "$VERBOSE" -eq $VERBOSE_ON ]]; then
        echo "  Status: SKIPPED (Invalid TIMEOUT: $TIMEOUT)"
        echo ""
      fi
      continue
    fi

    # Validate NUM_NODES is a positive integer >= 1
    if ! [[ "$NUM_NODES" =~ ^[0-9]+$ ]] || [[ "$NUM_NODES" -lt 1 ]]; then
      if [[ "$VERBOSE" -eq $VERBOSE_ON ]]; then
        echo "  Status: SKIPPED (Invalid NUM_NODES: $NUM_NODES - must be >= 1)"
        echo ""
      fi
      continue
    fi

    if [[ "$VERBOSE" -eq $VERBOSE_ON ]]; then
      echo "  Status: PARSED SUCCESSFULLY - Added to test queue"
      echo ""
    fi

    # Store parsed test configuration
    TEST_NAMES_PARSED+=("$TEST_NAME")
    TEST_BINARIES+=("$BINARY")
    TEST_NUM_RANKS+=("$NUM_RANKS")
    TEST_TIMEOUTS+=("$TIMEOUT")
    TEST_ENV_VARS+=("$ENV_VARS")
    TEST_GTEST_FILTERS+=("$GTEST_FILTER")
    TEST_NUM_NODES+=("$NUM_NODES")
    TOTAL_TESTS=$((TOTAL_TESTS + 1))

  done < "$CONFIG_FILE"

  # Print parsing summary
  echo ""
  echo "========================================"
  echo "PARSING COMPLETE"
  echo "========================================"
  echo "Configuration:           $CONFIG_FILE"
  echo "Lines read:              $LINES_PROCESSED"
  echo "Lines skipped (comment): $LINES_SKIPPED_COMMENT"
  echo "Lines skipped (empty):   $LINES_SKIPPED_EMPTY"
  echo "Tests parsed & queued:   $TOTAL_TESTS"
  echo ""

  # Check if zero tests were found
  if [[ $TOTAL_TESTS -eq 0 ]]; then
    echo "========================================"
    echo "WARNING: NO TESTS FOUND!"
    echo "========================================"
    echo ""
    echo "Possible reasons:"
    echo "  1. All test lines are commented out (lines starting with #)"
    echo "  2. Config file is empty or only contains comments"
    echo "  3. All tests were filtered out by --test-name filter"
    echo "  4. All tests failed validation (check paths, parameters)"
    echo ""
    echo "To enable tests:"
    echo "  - Open $CONFIG_FILE"
    echo "  - Remove the # at the start of test lines you want to run"
    echo "  - Ensure all required fields are present and valid"
    echo ""
    return $EXIT_SUCCESS
  fi

  echo "========================================"
  echo "EXECUTING TESTS"
  echo "========================================"
  echo ""

  # Now execute all parsed tests
  for (( i=0; i<$TOTAL_TESTS; i++ )); do
    local TEST_NAME="${TEST_NAMES_PARSED[$i]}"
    local BINARY="${TEST_BINARIES[$i]}"
    local NUM_RANKS="${TEST_NUM_RANKS[$i]}"
    local TIMEOUT="${TEST_TIMEOUTS[$i]}"
    local ENV_VARS="${TEST_ENV_VARS[$i]}"
    local GTEST_FILTER="${TEST_GTEST_FILTERS[$i]}"
    local NUM_NODES="${TEST_NUM_NODES[$i]}"

    TEST_NAMES+=("$TEST_NAME")

    echo ""
    echo "========================================"
    echo "Test $((i+1))/$TOTAL_TESTS: $TEST_NAME"
    echo "========================================"
    echo "  Binary:       $BINARY"
    echo "  Ranks:        $NUM_RANKS"
    echo "  Nodes:        $NUM_NODES"
    if [[ "$TIMEOUT" -eq 0 ]]; then
      echo "  Timeout:      None (unlimited)"
    else
    echo "  Timeout:      $TIMEOUT seconds"
    fi
    echo "  Environment:  $ENV_VARS"
    echo "  Filter:       $GTEST_FILTER"
    echo "  Started:      $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Build environment variable string for the command
    local ENV_PREFIX=""

    # Add base environment variables
    ENV_PREFIX="LD_LIBRARY_PATH=${BUILD_DIR}:${LD_LIBRARY_PATH} "
    ENV_PREFIX+="HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "

    # Parse and add additional environment variables
    # Supports both comma (,) and semicolon (;) as delimiters
    # Use semicolon when values contain commas (e.g., NCCL_SOCKET_IFNAME=eth0,eth1)
    if [[ "$ENV_VARS" != "$ENV_NONE" ]] && [[ -n "$ENV_VARS" ]]; then
      # Detect delimiter: use semicolon if present, otherwise comma
      local DELIMITER=','
      if [[ "$ENV_VARS" =~ \; ]]; then
        DELIMITER=';'
      fi

      IFS="$DELIMITER" read -ra ENV_ARRAY <<< "$ENV_VARS"
      for ENV_VAR in "${ENV_ARRAY[@]}"; do
        # Trim leading/trailing whitespace
        ENV_VAR=$(echo "$ENV_VAR" | xargs)
        if [[ "$ENV_VAR" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
          ENV_PREFIX+="$ENV_VAR "
        fi
      done
    fi

    # Build the command
    local TEST_CMD=""
    local FULL_CMD=""

    if [[ "$NUM_RANKS" -eq $DEFAULT_NUM_RANKS ]]; then
      # Non-MPI test
      if [[ "$GTEST_FILTER" == "$GTEST_FILTER_ALL" ]]; then
        TEST_CMD="./$BINARY"
      else
        TEST_CMD="./$BINARY --gtest_filter=$GTEST_FILTER"
      fi
      # Prepend environment variables to the command
      FULL_CMD="${ENV_PREFIX}${TEST_CMD}"
    else
      # MPI test
      if [[ -z "$MPI_PATH" ]]; then
        echo "ERROR: MPI_PATH not set, cannot run MPI test '$TEST_NAME'"
        echo "Skipping test and continuing..."
        continue
      fi

      MPI_CMD="${MPI_PATH}/bin/mpirun"
      MPI_ARGS="-np $NUM_RANKS"
      MPI_ARGS="$MPI_ARGS --mca btl ^vader,openib"
      MPI_ARGS="$MPI_ARGS --mca pml ucx"

      # Add multi-node arguments if needed
      if [[ "$NUM_NODES" -gt 1 ]]; then
        # User should set up hostfile or add appropriate MPI arguments
        echo "  Note: Multi-node execution requested ($NUM_NODES nodes) - ensure MPI hostfile is configured"
      fi

      # Add -x flags for MPI to pass environment variables
      local MPI_ENV_ARGS=""

      # Add base environment variables to MPI
      MPI_ENV_ARGS="-x LD_LIBRARY_PATH=${BUILD_DIR}:${LD_LIBRARY_PATH}"
      MPI_ENV_ARGS="$MPI_ENV_ARGS -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

      # Add additional environment variables for MPI
      if [[ "$ENV_VARS" != "$ENV_NONE" ]] && [[ -n "$ENV_VARS" ]]; then
        # Detect delimiter: use semicolon if present, otherwise comma
        local DELIMITER=','
        if [[ "$ENV_VARS" =~ \; ]]; then
          DELIMITER=';'
        fi

        IFS="$DELIMITER" read -ra ENV_ARRAY <<< "$ENV_VARS"
        for ENV_VAR in "${ENV_ARRAY[@]}"; do
          # Trim leading/trailing whitespace
          ENV_VAR=$(echo "$ENV_VAR" | xargs)
          if [[ "$ENV_VAR" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            # Pass both variable name and value to mpirun
            MPI_ENV_ARGS="$MPI_ENV_ARGS -x ${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
          fi
        done
      fi

      if [[ "$GTEST_FILTER" == "$GTEST_FILTER_ALL" ]]; then
        TEST_CMD="./$BINARY"
      else
        TEST_CMD="./$BINARY --gtest_filter=$GTEST_FILTER"
      fi

      FULL_CMD="$MPI_CMD $MPI_ARGS $MPI_ENV_ARGS $TEST_CMD"
    fi

    echo "  Command: $FULL_CMD"
    echo ""

    # Execute the test
    local START_TIME=$(date +%s)
    local TEST_EXIT_CODE=0

    if [[ "$TIMEOUT" -gt $DEFAULT_TIMEOUT ]]; then
      # Run with timeout (timeout command will kill after specified seconds)
      timeout "$TIMEOUT" bash -c "$FULL_CMD" || TEST_EXIT_CODE=$?
    else
      # Run without timeout (timeout = 0 means unlimited, test runs until completion or manual interrupt)
      bash -c "$FULL_CMD" || TEST_EXIT_CODE=$?
    fi

    local END_TIME=$(date +%s)
    local DURATION=$((END_TIME - START_TIME))
    TEST_DURATIONS+=("$DURATION")

    # Evaluate result
    echo ""
    if [[ $TEST_EXIT_CODE -eq $EXIT_SUCCESS ]]; then
      echo "$TEST_RESULT_PASSED ($DURATION seconds)"
      TEST_RESULTS+=("$TEST_RESULT_PASSED")
      PASSED_TESTS=$((PASSED_TESTS + 1))
    elif [[ $TEST_EXIT_CODE -eq $EXIT_TIMEOUT ]]; then
      echo "$TEST_RESULT_TIMEOUT after $TIMEOUT seconds"
      TEST_RESULTS+=("$TEST_RESULT_TIMEOUT")
      FAILED_TESTS=$((FAILED_TESTS + 1))
    else
      echo "$TEST_RESULT_FAILED (exit code: $TEST_EXIT_CODE, duration: $DURATION seconds)"
      TEST_RESULTS+=("$TEST_RESULT_FAILED")
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi

  done  # End of test execution loop

  # Print final test execution summary
  echo ""
  echo "========================================"
  echo "TEST EXECUTION SUMMARY"
  echo "========================================"
  echo "Total Tests:   $TOTAL_TESTS"
  echo "Passed:        $PASSED_TESTS"
  echo "Failed:        $FAILED_TESTS"
  echo ""

  # Print detailed results
  if [[ ${#TEST_NAMES[@]} -gt 0 ]]; then
    echo "Detailed Results:"
    echo "----------------------------------------"
    printf "%-40s %-10s %s\n" "Test Name" "Result" "Duration"
    echo "----------------------------------------"
    for i in "${!TEST_NAMES[@]}"; do
      printf "%-40s %-10s %s seconds\n" "${TEST_NAMES[$i]}" "${TEST_RESULTS[$i]}" "${TEST_DURATIONS[$i]}"
    done
    echo "========================================"
  fi

  # Save results to log file if LOG_DIR is set
  if [[ -n "$LOG_DIR" ]]; then
    local RESULT_LOG="$LOG_DIR/test_config_results.log"
    echo "Saving results to: $RESULT_LOG"
    {
      echo "Test Execution Results - $(date)"
      echo "Configuration: $CONFIG_FILE"
      echo "Total: $TOTAL_TESTS, Passed: $PASSED_TESTS, Failed: $FAILED_TESTS"
      echo ""
      printf "%-40s %-10s %s\n" "Test Name" "Result" "Duration"
      for i in "${!TEST_NAMES[@]}"; do
        printf "%-40s %-10s %s seconds\n" "${TEST_NAMES[$i]}" "${TEST_RESULTS[$i]}" "${TEST_DURATIONS[$i]}"
      done
    } > "$RESULT_LOG"
  fi

  # Return non-zero if any tests failed
  if [[ $FAILED_TESTS -gt 0 ]]; then
    return $EXIT_FAILURE
  fi

  return $EXIT_SUCCESS
}

generate_coverage_report() {
  local LOG_DIR="$1"
  local BUILD_DIR="$2"
  local REPORT_DIR="$3"
  local WORKDIR="$4"

  mkdir -p "$REPORT_DIR"

  # Move all the rawfiles into a single location
  mkdir -p "${LOG_DIR}/rawfiles"
  find "${BUILD_DIR}" -type f -name "*.profraw" -exec cp {} "${LOG_DIR}/rawfiles" \;

  # Create a list of raw files to merge
  find "${LOG_DIR}/rawfiles" -type f -name "*.profraw" > "${LOG_DIR}/rawprofiles.list"

  # Create the merged profdata
  /opt/rocm/lib/llvm/bin/llvm-profdata merge --sparse --input-files="${LOG_DIR}/rawprofiles.list" --output="${LOG_DIR}/merged.profdata"

  # Create the HTML report
  /opt/rocm/lib/llvm/bin/llvm-cov show \
    --instr-profile="${LOG_DIR}/merged.profdata" \
    --format=html \
    --Xdemangler=c++filt \
    --output-dir="${REPORT_DIR}" \
    --project-title="RCCL_Lib_Coverage_Report" \
    --ignore-filename-regex=".*tuner_v.*|.*profiler_v.*|.*net_v.*|.*_deps.*|ext.*|.*coll_net.*|.*nvls.*|.*nvml.*|.*nvtx.*|test/|.*gtest.*" \
    --object "${BUILD_DIR}/librccl.so" \
    --object "${BUILD_DIR}/test/rccl-UnitTestsFixtures" \
    --object "${BUILD_DIR}/test/rccl-UnitTests"

  # Generate function coverage summary (text report)
  /opt/rocm/lib/llvm/bin/llvm-cov report \
    --instr-profile="${LOG_DIR}/merged.profdata" \
    --Xdemangler=c++filt \
    --object "${BUILD_DIR}/librccl.so" \
    --object "${BUILD_DIR}/test/rccl-UnitTestsFixtures" \
    --object "${BUILD_DIR}/test/rccl-UnitTests" \
    --ignore-filename-regex=".*tuner_v.*|.*profiler_v.*|.*net_v.*|.*_deps.*|ext.*|.*coll_net.*|.*nvls.*|.*nvml.*|.*nvtx.*|test/|.*gtest.*" \
    --show-functions \
    --sources "${BUILD_DIR}" \
    > "${REPORT_DIR}/function_coverage_report.txt"

  echo "Function coverage report written to ${REPORT_DIR}/function_coverage_report.txt"
}

build_rccl() {
  local BUILD_TYPE="$1"
  local CODE_COV="$2"
  local BUILD_TESTS="$3"
  local BUILD_DIR="$4"
  local LOG_DIR="$5"
  local ROCM_PATH="$6"

  cd "$WORKDIR"

  # -DONLY_FUNCS="SendRecv|AllReduce" \
  CMAKE_SUCCESS=0
  CXX=${ROCM_PATH}/bin/amdclang++ cmake -S . -B "$BUILD_DIR" \
    -DCMAKE_CXX_FLAGS="-Wl,--build-id=sha1" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--build-id=sha1" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DENABLE_CODE_COVERAGE="$CODE_COV" \
    -DBUILD_TESTS="$BUILD_TESTS" \
    -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
    -DTRACE=ON \
    -DCOLLTRACE=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_VERBOSE_MAKEFILE=1 \
    -DENABLE_MPI_TESTS=ON \
    -DMPI_PATH="${MPI_PATH}" \
    -G"Unix Makefiles"

  if [ $? -eq 0 ]; then
    export LD_LIBRARY_PATH=${BUILD_DIR}:${LD_LIBRARY_PATH}
    cmake --build "$BUILD_DIR" --parallel 64 2>&1 | tee "$LOG_DIR/rccl_build_log.txt"
    if [ $? -ne 0 ]; then
      CMAKE_SUCCESS=1
      echo "ERROR: cmake build failed for BUILD_TYPE=${BUILD_TYPE}, CODE_COV=${CODE_COV}, BUILD_TESTS=${BUILD_TESTS}" | tee -a "$LOG_DIR/rccl_build_log.txt"
    fi
  else
    CMAKE_SUCCESS=1
    echo "ERROR: cmake configuration failed for BUILD_TYPE=${BUILD_TYPE}, CODE_COV=${CODE_COV}, BUILD_TESTS=${BUILD_TESTS}" | tee -a "$LOG_DIR/rccl_build_log.txt"
    cd "$WORKDIR"
  fi
  return $CMAKE_SUCCESS

}

validate_options() {
  local errors=0

  echo "Validating script options..."

  # Check 1: --test-name requires --config
  if [[ -n "$TEST_NAME_FILTER" && -z "$CONFIG_FILE" ]]; then
    echo "ERROR: --test-name requires --config option"
    errors=$((errors + 1))
  fi

  # Check 2: --no-build with no --config is pointless (but allow it)
  if [[ "$NO_BUILD" -eq 1 && -z "$CONFIG_FILE" ]]; then
    echo "WARNING: --no-build without --config will not run any tests"
    echo "         Consider using --config to run tests with existing build"
  fi

  # Check 3: --skip-tests with --config doesn't make sense
  if [[ "$SKIP_TESTS" -eq 1 && -n "$CONFIG_FILE" ]]; then
    echo "ERROR: --skip-tests and --config are mutually exclusive"
    echo "       --config is used to run tests, --skip-tests skips them"
    errors=$((errors + 1))
  fi

  # Check 4: Validate --report-suffix is not empty
  if [[ -z "$REPORT_SUFFIX" ]]; then
    echo "ERROR: --report-suffix cannot be empty"
    errors=$((errors + 1))
  fi

  # Check 5: Check if ROCm installation exists
  if [[ ! -d "$ROCM_PATH" ]]; then
    echo "ERROR: ROCm installation not found at: $ROCM_PATH"
    echo "       Available ROCm installations in /opt/:"
    ls -d /opt/rocm* 2>/dev/null || echo "       No ROCm installations found"
    errors=$((errors + 1))
  fi

  # Check 6: Check if MPI installation exists (only warn)
  if [[ ! -d "$MPI_PATH" ]]; then
    echo "WARNING: MPI installation not found at: $MPI_PATH"
    echo "         MPI tests may fail if ENABLE_MPI_TESTS=ON"
    echo "         Set MPI_PATH environment variable to override: export MPI_PATH=/path/to/mpi"
  fi

  # Check 7: Verify compiler exists
  if [[ ! -f "$ROCM_PATH/bin/amdclang++" ]]; then
    echo "ERROR: AMD clang++ compiler not found at: $ROCM_PATH/bin/amdclang++"
    errors=$((errors + 1))
  fi

  # Check 8: If --no-cov-report is used without build, warn
  if [[ "$NO_COV_REPORT" -eq 1 && "$NO_BUILD" -eq 1 ]]; then
    echo "WARNING: --no-cov-report with --no-build has no effect"
  fi

  # Check 9: Validate config file exists (if specified)
  if [[ -n "$CONFIG_FILE" ]]; then
    # This will be checked again later with better path resolution, but do basic check
    if [[ "$CONFIG_FILE" =~ ^/ ]]; then
      # Absolute path
      if [[ ! -f "$CONFIG_FILE" ]]; then
        echo "ERROR: Config file not found: $CONFIG_FILE"
        errors=$((errors + 1))
      fi
    fi
  fi

  # Exit if there are errors
  if [[ $errors -gt 0 ]]; then
    echo ""
    echo "Found $errors error(s). Please fix the issues and try again."
    echo "Run '$0 --help' for usage information."
    exit 1
  fi

  echo "Option validation passed successfully"
  echo ""
}

show_help() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --overwrite              Overwrite previous build and report directories (default: append timestamp)
  --no-build               Skip the build step and use existing build artifacts
  --no-cov-report          Skip code coverage report generation
  --skip-tests             Skip unit test execution (generates coverage report if profraw files exist)
  --report-suffix SUFFIX   Suffix to append to the report directory name (default: "test")
  --config FILE            Run tests from configuration file (builds first if needed)
  --test-name NAME         Filter tests by name when using --config
  --verbose, -v            Enable verbose output (shows detailed parsing information)
  --help                   Show this help message and exit

Description:
  This script builds RCCL with various options, runs unit tests, and generates code coverage reports.
  By default, it builds in Debug mode with code coverage and unit tests enabled.

  Test Execution:
  - All tests are now managed through configuration files
  - Use --config to specify test configuration file
  - Script will build first (if needed), then execute tests
  - Config file format: TEST_NAME | ENV_VARS | BINARY | GTEST_FILTER | NUM_NODES | NUM_RANKS | TIMEOUT
  - NUM_NODES, NUM_RANKS, and TIMEOUT default to 1, 1, and 0 respectively if not specified
  - Tests must be uncommented (not start with #) to run
  - Supports both MPI and non-MPI tests
  - Environment variables are included in the command for easy copy-paste execution

  Coverage Report Generation:
  - Coverage reports are generated even if some tests fail (uses available profraw files)
  - With --skip-tests: generates report if profraw files exist from previous test runs
  - Without profraw files: exits gracefully with informative message
  - This allows flexible workflows and partial coverage analysis

  Advanced Options:
  - Use --test-name to run a specific test from the config
  - Use --verbose/-v to see detailed parsing information
  - Use --no-build with --config to skip building
  - Set timeout to 0 for unlimited test execution time

Examples:
  # Build only
  $0

  # Build and run all tests from config (config file in same directory as script)
  $0 --config all_tests_config.txt

  # Run specific test with verbose output (builds first)
  $0 --config all_tests_config.txt --test-name P2P_IpcBufferRegistration_EdgeCases -v

  # Use existing build, run all tests
  $0 --config all_tests_config.txt --no-build

  # Use existing build, run specific test with verbose output
  $0 --config all_tests_config.txt --no-build --test-name MyTest --verbose

  # Use absolute path for config file
  $0 --config /path/to/custom_tests_config.txt

  # Generate coverage report from existing profraw files (skip running tests)
  $0 --config all_tests_config.txt --no-build --skip-tests

  # Build and skip tests (useful if you want to run tests manually later)
  $0 --skip-tests
EOF
}

main() {
  # Determine script location and set WORKDIR
  # Script is in rccl/tools/scripts/, so RCCL root is 2 levels up
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  RCCL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  # WORKDIR is set to RCCL root directory
  export WORKDIR="$RCCL_ROOT"

  echo "Script location: $SCRIPT_DIR"
  echo "RCCL root: $RCCL_ROOT"
  echo "Working directory: $WORKDIR"

  ROCM_DIR=$(ls /opt/ | grep rocm- | sort -V | tail -1)
  ROCM_VERSION="${ROCM_DIR#rocm-}"
  ROCM_PATCH_VERSION=$(echo $ROCM_VERSION | awk -F. '{printf "%01d%02d%02d\n", $1, $2, $3}')
  export ROCM_PATH=/opt/${ROCM_DIR}
  export MPI_PATH=${MPI_PATH:-${HOME}/softwares/ompi}
  export LD_LIBRARY_PATH=${MPI_PATH}/lib:${ROCM_PATH}/lib:${LD_LIBRARY_PATH}
  export PATH=${MPI_PATH}/bin:${ROCM_PATH}/bin:${PATH}

  # Example of how to run:
  # ./build_test_coverage.sh --overwrite
  # If --overwrite is specified, the build and report directories will be overwritten
  # Otherwise, a timestamp will be appended to the directory names to avoid overwriting previous results
  # The final report will be written to the latest timestamped directory

  # Build all combinations of build type, code coverage, and unit tests
  # BUILD_TYPES=("Debug" "RelWithDebInfo" "Release")
  # CODE_COVERAGE_FLAGS=("ON" "OFF")
  # BUILD_TESTS_FLAGS=("ON" "OFF")

  BUILD_TYPES=("Debug")
  CODE_COVERAGE_FLAGS=("ON")
  BUILD_TESTS_FLAGS=("ON")

  OVERWRITE=$EXIT_SUCCESS
  NO_BUILD=$EXIT_SUCCESS
  NO_COV_REPORT=$EXIT_SUCCESS
  CMAKE_SUCCESS=$EXIT_SUCCESS
  UNITTEST_SUCCESS=$EXIT_SUCCESS
  SKIP_TESTS=$EXIT_SUCCESS
  REPORT_SUFFIX="test"
  CONFIG_FILE=""
  TEST_NAME_FILTER=""
  VERBOSE=$VERBOSE_OFF

  while [[ $# -gt 0 ]]; do
    case $1 in
      --overwrite)
        OVERWRITE=1
        shift
        ;;
      --no-build)
        NO_BUILD=1
        shift
        ;;
      --no-cov-report)
        NO_COV_REPORT=1
        shift
        ;;
      --skip-tests)
        SKIP_TESTS=1
        shift
        ;;
      --report-suffix)
        REPORT_SUFFIX=$2
        shift 2
        ;;
      --config)
        CONFIG_FILE=$2
        shift 2
        ;;
      --test-name)
        TEST_NAME_FILTER=$2
        shift 2
        ;;
      --verbose|-v)
        VERBOSE=$VERBOSE_ON
        shift
        ;;
      --help)
        show_help
        exit $EXIT_SUCCESS
        ;;
      *)
        echo "ERROR: Unknown option: $1"
        echo ""
        show_help
        exit $EXIT_FAILURE
        ;;
    esac
  done

  # Validate options before proceeding
  validate_options

  # If --config is specified, handle config-based test execution
  if [[ -n "$CONFIG_FILE" ]]; then
    # Resolve config file path
    if [[ "$CONFIG_FILE" =~ ^/ ]]; then
      # Absolute path - use as is
      :
    elif [[ -f "$CONFIG_FILE" ]]; then
      # Relative path exists from current directory
      CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
    elif [[ -f "$SCRIPT_DIR/$CONFIG_FILE" ]]; then
      # Look in script directory (where config files are typically stored)
      CONFIG_FILE="$SCRIPT_DIR/$CONFIG_FILE"
    else
      echo "ERROR: Config file not found: $CONFIG_FILE"
      echo "Searched in:"
      echo "  - Current directory: $(pwd)/$CONFIG_FILE"
      echo "  - Script directory: $SCRIPT_DIR/$CONFIG_FILE"
      exit 1
    fi

    echo "Using test configuration file: $CONFIG_FILE"

    # If --no-build is set, use existing build
    if [[ "$NO_BUILD" -eq 1 ]]; then
      # Find the most recent build directory
      BUILD_DIR=$(find "${WORKDIR}" -maxdepth 1 -type d -name "build_*" | sort -r | head -n 1)
      if [[ -z "$BUILD_DIR" ]]; then
        echo "ERROR: No build directory found. Please build first."
        exit 1
      fi
      echo "Using existing build directory: $BUILD_DIR"

      # Set up log directory
      if [[ "$OVERWRITE" -eq 1 ]]; then
        LOG_DIR="${WORKDIR}/rccl_codecov_${REPORT_SUFFIX}"
      else
        TIMESTAMP=$(date +"%Y_%m_%d_%H%M%S")
        LOG_DIR="${WORKDIR}/rccl_codecov_${REPORT_SUFFIX}_${TIMESTAMP}"
      fi
      mkdir -p "$LOG_DIR"
    fi
    # If --no-build is NOT set, continue to build section and run tests after build
  fi

  for BUILD_TYPE in "${BUILD_TYPES[@]}"; do
    for CODE_COV in "${CODE_COVERAGE_FLAGS[@]}"; do
      for BUILD_TESTS in "${BUILD_TESTS_FLAGS[@]}"; do

        if [ "$CODE_COV" == "ON" ]; then
          export HIPCC_COMPILE_FLAGS_APPEND="-g -Wno-format-nonliteral -Xarch_host -fprofile-instr-generate -Xarch_host -fcoverage-mapping -parallel-jobs=16"
          export HIPCC_LINK_FLAGS_APPEND="-fprofile-instr-generate -fcoverage-mapping -parallel-jobs=16"
          export LLVM_PROFILE_FILE=rccl_tests_%9999m.profraw
        fi

        if [ "$OVERWRITE" -eq 1 ]; then
          WORKSPACE_NAME="rccl_codecov_${REPORT_SUFFIX}"
          DIR_SUFFIX="build_${BUILD_TYPE,,}_cov_${CODE_COV,,}_tests_${BUILD_TESTS,,}"
          BUILD_DIR="${WORKDIR}/${DIR_SUFFIX}"
          LOG_DIR="${WORKDIR}/${WORKSPACE_NAME}"
          REPORT_DIR="${WORKDIR}/${WORKSPACE_NAME}/${WORKSPACE_NAME}_report"
        else
          TIMESTAMP=$(date +"%Y_%m_%d_%H%M%S")
          WORKSPACE_NAME="rccl_codecov_${REPORT_SUFFIX}_${TIMESTAMP}"
          DIR_SUFFIX="build_${BUILD_TYPE,,}_cov_${CODE_COV,,}_tests_${BUILD_TESTS,,}_${TIMESTAMP}"
          BUILD_DIR="${WORKDIR}/${DIR_SUFFIX}"
          LOG_DIR="${WORKDIR}/${WORKSPACE_NAME}"
          REPORT_DIR="${WORKDIR}/${WORKSPACE_NAME}/${WORKSPACE_NAME}_report"
        fi

        mkdir -p "$BUILD_DIR"
        mkdir -p "$LOG_DIR"

        # Try-catch style: if any command fails, print error and continue to next build
        # Build RCCL (skip if --no-build)
        if [ "$NO_BUILD" -eq 0 ]; then
          build_rccl "$BUILD_TYPE" "$CODE_COV" "$BUILD_TESTS" "$BUILD_DIR" "$LOG_DIR" "$ROCM_PATH"
          CMAKE_SUCCESS=$?
        else
          echo "SKIP: build_rccl step skipped due to --no-build option." | tee -a "$LOG_DIR/rccl_build_log.txt"
          CMAKE_SUCCESS=0
        fi

        # Run tests from config file if specified
        if [ "$CMAKE_SUCCESS" -eq 0 ] && [ "$BUILD_TESTS" == "ON" ] && [ "$SKIP_TESTS" -eq 0 ]; then
          if [[ -n "$CONFIG_FILE" ]]; then
            # Run tests from config file
            echo "===================================================================" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "Running tests from configuration file: $CONFIG_FILE" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "===================================================================" | tee -a "$LOG_DIR/rccl_build_log.txt"

            if [[ -n "$TEST_NAME_FILTER" ]]; then
              echo "Filtering for test: $TEST_NAME_FILTER" | tee -a "$LOG_DIR/rccl_build_log.txt"
            fi
            if [[ "$VERBOSE" -eq 1 ]]; then
              echo "Verbose mode enabled" | tee -a "$LOG_DIR/rccl_build_log.txt"
            fi

            if run_tests_from_config "$CONFIG_FILE" "$BUILD_DIR" "$LOG_DIR" "$TEST_NAME_FILTER" "$VERBOSE"; then
              echo "All tests from config passed!" | tee -a "$LOG_DIR/rccl_build_log.txt"
              UNITTEST_SUCCESS=0
            else
              echo "Some tests from config failed!" | tee -a "$LOG_DIR/rccl_build_log.txt"
              echo "NOTE: Continuing to generate coverage report with available profraw files." | tee -a "$LOG_DIR/rccl_build_log.txt"
              UNITTEST_SUCCESS=0
            fi
          else
            # No config file specified - show usage message
            echo "===================================================================" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "To run tests, use:" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "  $0 --config all_tests_config.txt" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "  $0 --config all_tests_config.txt --verbose" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "  $0 --config all_tests_config.txt --test-name TEST_NAME" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "Or use --no-build with an existing build:" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "  $0 --config all_tests_config.txt --no-build" | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "===================================================================" | tee -a "$LOG_DIR/rccl_build_log.txt"
            UNITTEST_SUCCESS=0
          fi
        elif [ "$SKIP_TESTS" -eq 1 ]; then
          echo "SKIP: Unit tests skipped due to --skip-tests option." | tee -a "$LOG_DIR/rccl_build_log.txt"

          # Check if profraw files are available for coverage report generation
          PROFRAW_COUNT=$(find "${BUILD_DIR}" -type f -name "*.profraw" 2>/dev/null | wc -l)
          if [ "$PROFRAW_COUNT" -gt 0 ]; then
            echo "Found $PROFRAW_COUNT profraw file(s). Will generate coverage report with available data." | tee -a "$LOG_DIR/rccl_build_log.txt"
            UNITTEST_SUCCESS=0
          else
            echo "WARNING: No profraw files found in ${BUILD_DIR}." | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "Cannot generate coverage report without coverage data files." | tee -a "$LOG_DIR/rccl_build_log.txt"
            echo "Please run tests first to generate profraw files, or use this script without --skip-tests." | tee -a "$LOG_DIR/rccl_build_log.txt"
            UNITTEST_SUCCESS=1
          fi
        else
          echo "SKIP: Build failed for BUILD_TYPE=${BUILD_TYPE}, CODE_COV=${CODE_COV}, BUILD_TESTS=${BUILD_TESTS}" | tee -a "$LOG_DIR/rccl_build_log.txt"
          UNITTEST_SUCCESS=1
        fi

        # Only run code coverage if code coverage is ON, unit tests succeeded, and not --no-cov-report
        if [ "$CODE_COV" == "ON" ] && [ "$UNITTEST_SUCCESS" -eq 0 ] && [ "$NO_COV_REPORT" -eq 0 ]; then
          generate_coverage_report "$LOG_DIR" "$BUILD_DIR" "$REPORT_DIR" "$WORKDIR"
        elif [ "$CODE_COV" == "ON" ] && [ "$UNITTEST_SUCCESS" -eq 0 ] && [ "$NO_COV_REPORT" -eq 1 ] && [ "$SKIP_TESTS" -eq 0 ]; then
          echo "SKIP: Generating code coverage report step skipped due to --no-cov-report option." | tee -a "$LOG_DIR/rccl_build_log.txt"
        fi

        echo "Build with BUILD_TYPE=${BUILD_TYPE}, CODE_COV=${CODE_COV}, BUILD_TESTS=${BUILD_TESTS} completed."
        cd "$WORKDIR"
      done
    done
  done

  echo "All builds and tests completed."
}

main "$@"
