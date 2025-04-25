# Copyright (c) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.
# Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

cmake_minimum_required(VERSION 2.8.12)

if(BUILD_TESTS)

  option(OPENMP_TESTS_ENABLED "Enable OpenMP for unit tests" OFF)

  message("Building rccl unit tests (Installed in /test/rccl-UnitTests)")

  find_package(hsa-runtime64 PATHS /opt/rocm )
  if(${hsa-runtime64_FOUND})
    message("hsa-runtime64 found @  ${hsa-runtime64_DIR} ")
  else()
    message("find_package did NOT find hsa-runtime64, finding it the OLD Way")
    message("Looking for header files in ${ROCR_INC_DIR}")
    message("Looking for library files in ${ROCR_LIB_DIR}")

    # Search for ROCr header file in user defined locations
    find_path(ROCR_HDR hsa/hsa.h PATHS ${ROCR_INC_DIR} "/opt/rocm" PATH_SUFFIXES include REQUIRED)
    INCLUDE_DIRECTORIES(${ROCR_HDR})

    # Search for ROCr library file in user defined locations
    find_library(ROCR_LIB ${CORE_RUNTIME_TARGET} PATHS ${ROCR_LIB_DIR} "/opt/rocm" PATH_SUFFIXES lib lib64 REQUIRED)
  endif()

  if(OPENMP_TESTS_ENABLED)
    find_package(OpenMP REQUIRED)
  endif()

  include_directories(${GTEST_INCLUDE_DIRS} ./common)

  # Collect testing framework source files
  set(TEST_SOURCE_FILES
    AllGatherTests.cpp
    AllReduceTests.cpp
    AllToAllTests.cpp
    AllToAllVTests.cpp
    BroadcastTests.cpp
    GatherTests.cpp
    GroupCallTests.cpp
    NonBlockingTests.cpp
    ReduceScatterTests.cpp
    ReduceTests.cpp
    RcclWrapTests.cpp
    ScatterTests.cpp
    SendRecvTests.cpp
    StandaloneTests.cpp
    _RecorderTests.cpp
    common/main.cpp
    common/CallCollectiveForked.cpp
    common/CollectiveArgs.cpp
    common/EnvVars.cpp
    common/PrepDataFuncs.cpp
    common/PtrUnion.cpp
    common/TestBed.cpp
    common/TestBedChild.cpp
    common/StandaloneUtils.cpp
    )

  # Append file if build type is not Debug.
  # Visibility is hidden by default, so we need to explicitly add the following file(s)
  # to the unit tests to ensure it is included for the existing rccl-UnitTests execution
  if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND TEST_SOURCE_FILES
      ../src/misc/recorder.cc
    )
  endif()

  # Add rccl-UnitTests binary
  add_executable(rccl-UnitTests ${TEST_SOURCE_FILES})

  set(RCCL_TEST_EXECUTABLES rccl-UnitTests)

  # Create rccl-UnitTestsFixtures binary if ROCm version is 4.6.0 or greater
  # and build type is Debug
  if (ROCM_VERSION VERSION_GREATER_EQUAL "60400" AND CMAKE_BUILD_TYPE MATCHES "Debug")

    set(TEST_FIXTURE_SOURCE_FILES
      AltRsmiTests.cpp
      ArgCheckTests.cpp
      common/main_fixtures.cpp
      common/EnvVars.cpp
    )

    set(HIPIFY_SRC_DIR "${PROJECT_BINARY_DIR}/hipify/src")
    set(REPLACE_SCRIPT "${CMAKE_SOURCE_DIR}/tools/scripts/replace_static.sh")
    message ("Replacing static functions in ${HIPIFY_SRC_DIR} with ${REPLACE_SCRIPT}")

    # Create a list of files which needs to be modified to remove static
    set(TEST_NONSTATIC_SOURCE_FILES
      ${HIPIFY_SRC_DIR}/misc/alt_rsmi.cc
    )

    # Create a custom command to backup the original files and remove static
    foreach(srcfile ${TEST_NONSTATIC_SOURCE_FILES})
      add_custom_command(
        OUTPUT "${srcfile}.staticbak"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${srcfile}" "${srcfile}.staticbak"
        COMMAND ${CMAKE_COMMAND} -E echo "Replacing static in ${srcfile}"
        COMMAND bash -c "\"${REPLACE_SCRIPT}\" \"${srcfile}\" --replace-vars 2>&1"
        DEPENDS "${srcfile}" "${REPLACE_SCRIPT}"
        COMMENT "Removing static from ${srcfile} (backup at ${srcfile}.staticbak)"
        VERBATIM
      )
      list(APPEND STATIC_BAK_FILES "${srcfile}.staticbak")
    endforeach()
    add_custom_target(replace_static_in_hipify ALL DEPENDS ${STATIC_BAK_FILES})
    add_dependencies(replace_static_in_hipify hipify_all)

    # Add rccl-UnitTestsFixtures binary
    add_executable(rccl-UnitTestsFixtures ${TEST_FIXTURE_SOURCE_FILES})

    list(APPEND RCCL_TEST_EXECUTABLES rccl-UnitTestsFixtures)

    add_dependencies(rccl-UnitTestsFixtures replace_static_in_hipify)
  endif()

  ## Set include directories for both binaries
  foreach(target ${RCCL_TEST_EXECUTABLES})
    target_include_directories(${target} PRIVATE ${ROCM_PATH} ${GTEST_INCLUDE_DIRS})
    target_include_directories(${target} PRIVATE ${PROJECT_BINARY_DIR}/include)            # for generated rccl.h header
    target_include_directories(${target} PRIVATE ${PROJECT_BINARY_DIR}/hipify/src/include) # for rccl_bfloat16.h
    target_include_directories(${target} PRIVATE ${PROJECT_BINARY_DIR}/hipify/src) 

    ## Set compile definitions
    if(LL128_ENABLED)
      target_compile_definitions(${target} PRIVATE ENABLE_LL128)
    endif()
    if(OPENMP_TESTS_ENABLED)
      target_compile_definitions(${target} PRIVATE ENABLE_OPENMP)
    endif()
    target_compile_definitions(${target} PRIVATE ROCM_PATH="${ROCM_PATH}")

    ## Set rccl unittests linked libraries
    target_link_libraries(${target} PRIVATE ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(${target} PRIVATE hip::host hip::device hsa-runtime64::hsa-runtime64)
    target_link_libraries(${target} PRIVATE Threads::Threads)
    target_link_libraries(${target} PRIVATE dl)
    if(OPENMP_TESTS_ENABLED)
      target_link_libraries(${target} PRIVATE "${OpenMP_CXX_FLAGS}")
    endif()

    # Link rccl library
    if(BUILD_SHARED_LIBS)
      target_link_libraries(${target} PRIVATE rccl)
      set_property(TARGET ${target} PROPERTY INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${ROCM_PATH}/lib;${CMAKE_BINARY_DIR}")
    else()
      add_dependencies(${target} rccl)
      target_link_libraries(${target} PRIVATE dl rt numa -lrccl -L${CMAKE_BINARY_DIR} -lrocm_smi64 -L${ROCM_PATH}/lib -L${ROCM_PATH}/rocm_smi/lib)
    endif()
    set_property(TARGET ${target} PROPERTY BUILD_RPATH "${CMAKE_BINARY_DIR};${ROCM_PATH}/lib")

    # Install the binary
    rocm_install(TARGETS ${target} COMPONENT tests)
  endforeach()

else()
  message("Not building rccl unit tests")
  # If not building tests, restore originals in the build directory if backup exists
  foreach(srcfile ${HIPIFY_SRC_FILES})
    if(EXISTS "${srcfile}.staticbak")
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy "${srcfile}.staticbak" "${srcfile}"
      )
      execute_process(
        COMMAND ${CMAKE_COMMAND} -E remove "${srcfile}.staticbak"
      )
    endif()
  endforeach()
endif()

