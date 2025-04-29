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

  set(TEST_FIXTURE_SOURCE_FILES
    AltRsmiTests.cpp
    ArgCheckTests.cpp
    common/main_fixtures.cpp
    common/EnvVars.cpp
  )

  # Add rccl-UnitTests binary
  add_executable(rccl-UnitTests ${TEST_SOURCE_FILES})

  # Add rccl-UnitTestsFixtures binary
  add_executable(rccl-UnitTestsFixtures ${TEST_FIXTURE_SOURCE_FILES})

  ## Set include directories for both binaries
  foreach(target rccl-UnitTests rccl-UnitTestsFixtures)
    target_include_directories(${target} PRIVATE ${ROCM_PATH} ${GTEST_INCLUDE_DIRS})
    target_include_directories(${target} PRIVATE ${PROJECT_BINARY_DIR}/include)            # for generated rccl.h header
    target_include_directories(${target} PRIVATE ${PROJECT_BINARY_DIR}/hipify/src/include) # for rccl_bfloat16.h

    ## Set compile definitions
    if(LL128_ENABLED)
      target_compile_definitions(${target} PRIVATE ENABLE_LL128)
    endif()
    if(OPENMP_TESTS_ENABLED)
      target_compile_definitions(${target} PRIVATE ENABLE_OPENMP)
    endif()
    target_compile_definitions(${target} PRIVATE ROCM_PATH="${ROCM_PATH}")

    ## Set rccl-UnitTests linked libraries
    target_link_libraries(rccl-UnitTests PRIVATE ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(rccl-UnitTests PRIVATE hip::host hip::device hsa-runtime64::hsa-runtime64)
    target_link_libraries(rccl-UnitTests PRIVATE Threads::Threads)
    target_link_libraries(rccl-UnitTests PRIVATE dl)
    if(OPENMP_TESTS_ENABLED)
      target_link_libraries(rccl-UnitTests PRIVATE "${OpenMP_CXX_FLAGS}")
    endif()

    ## Set linked libraries
    target_link_libraries(${target} PRIVATE ${GTEST_BOTH_LIBRARIES})
    target_link_libraries(${target} PRIVATE hip::host hip::device hsa-runtime64::hsa-runtime64)
    target_link_libraries(${target} PRIVATE Threads::Threads)
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
  endforeach()

  # Install the binaries
  rocm_install(TARGETS rccl-UnitTests rccl-UnitTestsFixtures COMPONENT tests)
else()
  message("Not building rccl unit tests")
endif()

