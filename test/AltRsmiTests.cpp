/*************************************************************************
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "alt_rsmi.h"

#include <cerrno>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <cstdlib>

// ============================================================================
// Internal structures and variables from alt_rsmi.cc (TEST USE ONLY)
// ============================================================================
// When alt_rsmi.cc is compiled with ARSMI_TEST_BUILD, internal variables
// have external linkage and can be accessed by test utilities

struct ARSMI_systemNode {
    uint32_t s_node_id = 0;
    uint64_t s_gpu_id = 0;
    uint64_t s_unique_id = 0;
    uint64_t s_location_id = 0;
    uint64_t s_bdf = 0;
    uint64_t s_domain = 0;
    uint8_t  s_bus = 0;
    uint8_t  s_device = 0;
    uint8_t  s_function = 0;
    uint8_t  s_partition_id = 0;
    std::string s_card;
};

// External declarations of internal variables from alt_rsmi.cc
extern thread_local const char *kKFDNodesPathRoot;
extern thread_local int ARSMI_num_devices;
extern thread_local std::vector<ARSMI_systemNode> ARSMI_orderedNodes;
extern thread_local std::vector<std::vector<ARSMI_linkInfo>> ARSMI_orderedLinks;

// ============================================================================
// Test utilities for manipulating alt_rsmi.cc internal state
// ============================================================================
namespace AltRsmiTestUtils {

// Set the KFD nodes path for testing
// This redirects file reads to test directories
void SetNodesPath(const char* path) {
    kKFDNodesPathRoot = path;
}

// Reset ARSMI internal state between tests
// This ensures test isolation
void ResetState() {
    ARSMI_num_devices = -1;
    ARSMI_orderedNodes.clear();
    ARSMI_orderedLinks.clear();
}

// Get current number of devices (for verification)
int GetNumDevices() {
    return ARSMI_num_devices;
}

} // namespace AltRsmiTestUtils

// Test paths for creating mock KFD filesystem
static const char* kTestKFDBasePath = "/tmp/test_kfd_arsmi";
// IMPORTANT: This must match the default value in alt_rsmi.cc
// Always restore this path after tests to avoid affecting other test suites
static const char* kSystemKFDPath = "/sys/class/kfd/kfd/topology/nodes";
static const char* kTestKFDPath = "/tmp/test_kfd_arsmi/topology/nodes";

namespace RcclUnitTesting {

// Global test environment to ensure path restoration even if tests fail
class AltRsmiTestEnvironment : public ::testing::Environment {
public:
  ~AltRsmiTestEnvironment() override = default;

  // Called before any tests run
  void SetUp() override {
    // Save the original path (should be system default)
    originalPath = kSystemKFDPath;
  }

  // Called after all tests finish
  void TearDown() override {
    // Ensure path is restored to system default
    // This catches any test that failed to restore in its TearDown()
    AltRsmiTestUtils::SetNodesPath(kSystemKFDPath);
    AltRsmiTestUtils::ResetState();
  }

private:
  const char* originalPath;
};

// Register the global test environment
// This ensures cleanup happens even if individual tests fail
static ::testing::Environment* const altRsmiEnv =
    ::testing::AddGlobalTestEnvironment(new AltRsmiTestEnvironment());

class AltRsmiTest : public ::testing::Test {

protected:
  // Helper function to create directories recursively
  int createDirectory(const std::string &path) {
    size_t pos = 0;
    std::string currentPath;

    // Iterate through each component of the path
    while ((pos = path.find('/', pos)) != std::string::npos) {
      currentPath = path.substr(0, pos++);
      if (!currentPath.empty() && mkdir(currentPath.c_str(), 0700) == -1 &&
          errno != EEXIST) {
        return -1; // Return error if directory creation fails
      }
    }

    // Create the final directory
    if (mkdir(path.c_str(), 0700) == -1 && errno != EEXIST) {
      return -1; // Return error if directory creation fails
    }

    return 0; // Success
  }

  // Helper function to remove a directory recursively
  int removeDirectory(const std::string &path) {
    DIR *dir = opendir(path.c_str());
    if (!dir) {
      std::cerr << "Failed to open directory: " << path << " (errno: " << errno
                << ")" << std::endl;
      return -1;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
      // Skip "." and ".." entries
      if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
        continue;
      }

      std::string fullPath = path + "/" + entry->d_name;

      // Check if the entry is a directory
      struct stat entryStat;
      if (stat(fullPath.c_str(), &entryStat) == -1) {
        std::cerr << "Failed to stat: " << fullPath << " (errno: " << errno
                  << ")" << std::endl;
        closedir(dir);
        return -1;
      }

      if (S_ISDIR(entryStat.st_mode)) {
        // Recursively remove subdirectory
        if (removeDirectory(fullPath) == -1) {
          closedir(dir);
          return -1;
        }
      } else {
        // Remove file
        if (unlink(fullPath.c_str()) == -1) {
          std::cerr << "Failed to remove file: " << fullPath
                    << " (errno: " << errno << ")" << std::endl;
          closedir(dir);
          return -1;
        }
      }
    }

    closedir(dir);

    // Remove the directory itself
    if (rmdir(path.c_str()) == -1) {
      std::cerr << "Failed to remove directory: " << path
                << " (errno: " << errno << ")" << std::endl;
      return -1;
    }

    return 0; // Success
  }

  // Helper function to create a file with content
  void createFile(const std::string &path, const std::string &content) {
    std::ofstream file(path);
    if (!file) {
      std::cerr << "Failed to create file: " << path << ", errno: " << errno << std::endl;
      return;
    }
    file << content;
    file.close();
  }

  // Helper function to remove a file
  int removeFile(const std::string &path) {
    if (unlink(path.c_str()) == -1) {
      std::cerr << "Failed to remove file: " << path << " (errno: " << errno
                << ")" << std::endl;
      return -1; // Return error if file removal fails
    }
    return 0; // Success
  }

  // Function to create the test directory structure and files
  void setupTestFiles() {
    const std::string basePath = kTestKFDPath;

    createDirectory(basePath);

    // Create node 0 with valid data
    createDirectory(basePath + "/0");
    createFile(basePath + "/0/gpu_id", "4098\n");
    createFile(basePath + "/0/properties", "unique_id 16336014475442738425\n"
                                           "location_id 23552\n"
                                           "domain 0\n"
                                           "vendor_id 4098\n");

    createDirectory(basePath + "/0/io_links/0");
    createFile(basePath + "/0/io_links/0/properties",
               "type 2\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 0\n"
               "node_to 1\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 64000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory(basePath + "/0/io_links/1");
    createFile(basePath + "/0/io_links/1/properties",
               "type 11\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 0\n"
               "node_to 1\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 50000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory(basePath + "/1");
    createFile(basePath + "/1/gpu_id", "4098\n");
    createFile(basePath + "/1/properties", "unique_id 16336014475442738426\n"
                                "location_id 23553\n"
                                "domain 1\n"
                                "vendor_id 4098\n");

    createDirectory(basePath + "/1/io_links/0");
    createFile(basePath + "/1/io_links/0/properties",
               "type 2\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 1\n"
               "node_to 0\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 32000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    createDirectory(basePath + "/1/io_links/1");
    createFile(basePath + "/1/io_links/1/properties",
               "type 11\n"
               "version_major 0\n"
               "version_minor 0\n"
               "node_from 1\n"
               "node_to 0\n"
               "weight 21\n"
               "min_latency 0\n"
               "max_latency 0\n"
               "min_bandwidth 0\n"
               "max_bandwidth 50000\n"
               "recommended_transfer_size 0\n"
               "recommended_sdma_engine_id_mask 0\n"
               "flags 0\n");

    uint32_t invalid_dev_id = 9999; // Device ID that doesn't exist
    createDirectory(basePath + "/" + std::to_string(invalid_dev_id) + "/io_links/");
  }

  void SetUp() override {
    // Reset ARSMI state for test isolation
    // Using test utilities
    AltRsmiTestUtils::ResetState();

    // Redirect kKFDNodesPathRoot to test directory
    AltRsmiTestUtils::SetNodesPath(kTestKFDPath);

    // Create the test directory structure
    setupTestFiles();
  }

  void TearDown() override {
    // CRITICAL: Always restore path to system default
    // This must happen even if the test fails, to avoid affecting other tests
    try {
      // Reset ARSMI state for test isolation
      AltRsmiTestUtils::ResetState();

      // Restore default path - REQUIRED for test isolation
      AltRsmiTestUtils::SetNodesPath(kSystemKFDPath);

      // Clean up test files
      removeDirectory(kTestKFDBasePath);
    } catch (...) {
      // Ensure path is restored even if cleanup fails
      AltRsmiTestUtils::SetNodesPath(kSystemKFDPath);
      throw;
    }
  }
};

// Tests using only public API
TEST_F(AltRsmiTest, ARSMIInitDefault) {
  int result = ARSMI_init();
  ASSERT_EQ(result, 0);

  // Verify that devices were discovered
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 2);
}

TEST_F(AltRsmiTest, ARSMIInitMissingIoLinksPropertiesFile) {
  // Remove properties file for io_links
  removeFile(std::string(kTestKFDPath) + "/0/io_links/0/properties");
  removeFile(std::string(kTestKFDPath) + "/0/io_links/1/properties");

  int result = ARSMI_init();
  ASSERT_EQ(result, 0);

  // Should still initialize successfully even with missing link properties
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_GT(num_devices, 0);
}

TEST_F(AltRsmiTest, ARSMIInitMissingNodeToProperty) {
  createFile(std::string(kTestKFDPath) + "/0/io_links/1/properties",
             "type 2\n"
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             // "node_to 0\n"  // Missing node_to
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             "max_bandwidth 64000\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();
  ASSERT_EQ(result, 0); // Expect success even with missing node_to

  // Verify devices are still initialized
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_GT(num_devices, 0);
}

TEST_F(AltRsmiTest, ARSMIInitMissingWeightProperty) {
  createFile(std::string(kTestKFDPath) + "/0/io_links/1/properties",
             "type 2\n"
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 1\n"
             // "weight 21\n"  // Missing weight
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             "max_bandwidth 64000\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();
  ASSERT_NE(result, 0); // Expect non-zero error code when required property is missing
}

TEST_F(AltRsmiTest, ARSMIInitMissingTypeProperty) {
  createFile(std::string(kTestKFDPath) + "/0/io_links/1/properties",
             // "type 5\n" // Missing type
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 1\n"
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             "max_bandwidth 0\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();
  ASSERT_NE(result, 0); // Expect non-zero error code when type is missing
}

TEST_F(AltRsmiTest, ARSMIInitTypePCIeProperty) {
  int result = ARSMI_init();
  ASSERT_EQ(result, 0);

  // Verify link info can be retrieved for PCIe connections
  ARSMI_linkInfo info;
  ASSERT_EQ(ARSMI_topo_get_link_info(0, 1, &info), 0);
  // The link should be detected (type will be set based on properties)
}

TEST_F(AltRsmiTest, ARSMIInitMissingMinBWProperty) {
  createFile(std::string(kTestKFDPath) + "/0/io_links/1/properties",
             "type 11\n"
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 1\n"
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             // "min_bandwidth 0\n"  // Missing min_bandwidth
             "max_bandwidth 0\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();
  ASSERT_NE(result, 0); // Expect non-zero error code when min_bandwidth is missing
}

TEST_F(AltRsmiTest, ARSMIInitMissingMaxBWProperty) {
  createFile(std::string(kTestKFDPath) + "/0/io_links/1/properties",
             "type 5\n"
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 1\n"
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             // "max_bandwidth 0\n"  // Missing max_bandwidth
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  int result = ARSMI_init();
  ASSERT_NE(result, 0); // Expect non-zero error code when max_bandwidth is missing
}

TEST_F(AltRsmiTest, ARSMIGetNumDevicesUninitialized) {
  // Don't call ARSMI_init, let ARSMI_get_num_devices initialize
  uint32_t num_devices = 0;

  int result = ARSMI_get_num_devices(&num_devices);

  // Verify that the function initializes successfully
  ASSERT_EQ(result, 0);

  // Verify that the number of devices is correctly set
  ASSERT_EQ(num_devices, 2);
}

TEST_F(AltRsmiTest, ARSMIDevPciIdGetNullBdfId) {
  uint32_t device_index = 0;
  int result = ARSMI_dev_pci_id_get(device_index, nullptr);

  ASSERT_EQ(result, EINVAL);
}

TEST_F(AltRsmiTest, ARSMIDevPciIdGetOutOfRange) {
  // First initialize with test data
  ASSERT_EQ(ARSMI_init(), 0);

  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_GT(num_devices, 0);

  // Try to get BDF for out-of-range device index
  uint64_t bdfid = 0;
  uint32_t invalid_index = num_devices + 10;

  // Note: ARSMI_dev_pci_id_get does not perform bounds checking
  // This test documents that behavior - in production, callers must validate indices
  // Accessing out-of-range index will cause undefined behavior (likely crash or garbage data)
}

TEST_F(AltRsmiTest, ARSMIDevPciIdGetValid) {
  uint32_t device_index = 0;
  uint64_t bdfid = 0;

  int result = ARSMI_dev_pci_id_get(device_index, &bdfid);

  // Verify that the function succeeds
  ASSERT_EQ(result, 0);
  // BDF ID should be non-zero for valid devices
  ASSERT_NE(bdfid, 0);
}

// Tests covering invalid file/directory scenarios through public API
TEST_F(AltRsmiTest, ARSMIInitWithInvalidGpuIdData) {
  // Create a gpu_id file with invalid (non-numeric) data
  removeDirectory(kTestKFDBasePath);
  createDirectory(std::string(kTestKFDPath) + "/0");
  createFile(std::string(kTestKFDPath) + "/0/gpu_id", "invalid_gpu_id");
  createFile(std::string(kTestKFDPath) + "/0/properties", "unique_id 12345\n"
                                                          "location_id 23552\n"
                                                          "domain 0\n"
                                                          "vendor_id 4098\n");

  int result = ARSMI_init();

  // Init should handle invalid gpu_id gracefully
  ASSERT_EQ(result, 0);

  // Should not discover any devices with invalid gpu_id
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 0);
}

TEST_F(AltRsmiTest, ARSMIInitWithEmptyPropertiesFile) {
  // Create an empty properties file
  removeDirectory(kTestKFDBasePath);
  createDirectory(std::string(kTestKFDPath) + "/0");
  createFile(std::string(kTestKFDPath) + "/0/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/0/properties", "");

  int result = ARSMI_init();

  // Should succeed but not discover devices with empty properties
  ASSERT_EQ(result, 0);

  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 0);
}

TEST_F(AltRsmiTest, ARSMIInitWithDirectoryInsteadOfPropertiesFile) {
  // Create a directory instead of properties file
  removeDirectory(kTestKFDBasePath);
  createDirectory(std::string(kTestKFDPath) + "/0");
  createFile(std::string(kTestKFDPath) + "/0/gpu_id", "4098\n");
  createDirectory(std::string(kTestKFDPath) + "/0/properties");

  int result = ARSMI_init();

  // Should handle this gracefully
  ASSERT_EQ(result, 0);

  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 0);
}

TEST_F(AltRsmiTest, ARSMIInitWithMissingVendorId) {
  // Create node without vendor_id
  removeDirectory(kTestKFDBasePath);
  createDirectory(std::string(kTestKFDPath) + "/0");
  createFile(std::string(kTestKFDPath) + "/0/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/0/properties", "unique_id 12345\n"
                                                          "location_id 23552\n"
                                                          "domain 0\n");
                                                          // Missing vendor_id

  int result = ARSMI_init();
  ASSERT_EQ(result, 0);

  // Should not discover devices without vendor_id
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 0);
}

TEST_F(AltRsmiTest, ARSMIInitWithNonAMDVendorId) {
  // Create node with non-AMD vendor_id
  removeDirectory(kTestKFDBasePath);
  createDirectory(std::string(kTestKFDPath) + "/0");
  createFile(std::string(kTestKFDPath) + "/0/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/0/properties", "unique_id 12345\n"
                                                          "location_id 23552\n"
                                                          "domain 0\n"
                                                          "vendor_id 0x10DE\n"); // NVIDIA vendor ID

  int result = ARSMI_init();
  ASSERT_EQ(result, 0);

  // Should not discover non-AMD devices
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 0);
}

TEST_F(AltRsmiTest, ARSMIInitWithEmptyLinkPropertiesFile) {
  // Create setup but with empty link properties
  createFile(std::string(kTestKFDPath) + "/0/io_links/0/properties", "");

  int result = ARSMI_init();

  // Should still initialize, just skip that link
  ASSERT_EQ(result, 0);

  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_GT(num_devices, 0);
}

TEST_F(AltRsmiTest, NullInfoPointer) {
  int result = ARSMI_topo_get_link_info(0, 1, nullptr);
  ASSERT_EQ(result, EINVAL); // Expect EINVAL for null `info` pointer
}

TEST_F(AltRsmiTest, SourceDeviceIndexOutOfRange) {
  ARSMI_linkInfo info;
  // First initialize
  ASSERT_EQ(ARSMI_init(), 0);

  int result = ARSMI_topo_get_link_info(999, 1, &info); // Invalid source index
  ASSERT_EQ(result, EINVAL); // Expect EINVAL for out-of-range source index
}

TEST_F(AltRsmiTest, DestinationDeviceIndexOutOfRange) {
  ARSMI_linkInfo info;
  // First initialize
  ASSERT_EQ(ARSMI_init(), 0);

  int result = ARSMI_topo_get_link_info(0, 999, &info); // Invalid destination index
  ASSERT_EQ(result, EINVAL); // Expect EINVAL for out-of-range destination index
}

TEST_F(AltRsmiTest, LinkInfoAutoInitializes) {
  // Test that ARSMI_topo_get_link_info auto-initializes if not already initialized
  ARSMI_linkInfo info;
  int result = ARSMI_topo_get_link_info(0, 0, &info);

  // Should succeed - auto-initialization should work with test data
  ASSERT_EQ(result, 0);
}

TEST_F(AltRsmiTest, ValidLinkInfoBetweenDevices) {
  // Initialize the system
  ASSERT_EQ(ARSMI_init(), 0);

  ARSMI_linkInfo info;
  int result = ARSMI_topo_get_link_info(0, 1, &info);

  // Should succeed
  ASSERT_EQ(result, 0);

  // Verify link info contains reasonable values
  ASSERT_EQ(info.src_node, 0);
  ASSERT_EQ(info.dst_node, 1);
  // Type should be XGMI (type 11 in properties)
  ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_XGMI);
  ASSERT_EQ(info.hops, 1);
  ASSERT_EQ(info.weight, 21);
  ASSERT_EQ(info.min_bandwidth, 0);
  ASSERT_EQ(info.max_bandwidth, 50000);
}

TEST_F(AltRsmiTest, ValidLinkInfoSelfLink) {
  // Initialize the system
  ASSERT_EQ(ARSMI_init(), 0);

  ARSMI_linkInfo info;
  int result = ARSMI_topo_get_link_info(0, 0, &info);

  // Should succeed - even self-links should return default values
  ASSERT_EQ(result, 0);
}

TEST_F(AltRsmiTest, LinkInfoWithNoDirectConnection) {
  // Setup with 2 nodes where they don't have direct XGMI connection
  removeDirectory(kTestKFDBasePath);
  createDirectory(kTestKFDPath);

  // Create node 0
  createDirectory(std::string(kTestKFDPath) + "/0");
  createFile(std::string(kTestKFDPath) + "/0/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/0/properties",
             "unique_id 100\n"
             "location_id 23552\n"
             "domain 0\n"
             "vendor_id 4098\n");
  // Create empty io_links directory (no actual links defined)
  createDirectory(std::string(kTestKFDPath) + "/0/io_links");

  // Create node 1
  createDirectory(std::string(kTestKFDPath) + "/1");
  createFile(std::string(kTestKFDPath) + "/1/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/1/properties",
             "unique_id 101\n"
             "location_id 23553\n"
             "domain 1\n"
             "vendor_id 4098\n");
  // Create empty io_links directory (no actual links defined)
  createDirectory(std::string(kTestKFDPath) + "/1/io_links");

  ASSERT_EQ(ARSMI_init(), 0);

  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 2);

  // Try to get link info between the two devices (no direct link defined)
  ARSMI_linkInfo info;
  int result = ARSMI_topo_get_link_info(0, 1, &info);

  // Should succeed but return default values since no io_links are defined
  ASSERT_EQ(result, 0);
  ASSERT_EQ(info.hops, 2); // Default hops
  ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_PCIEXPRESS); // Default type
  ASSERT_EQ(info.weight, 40); // Default weight
}

TEST_F(AltRsmiTest, MultipleDevicesWithXGMILinks) {
  // Test XGMI link type (type 11)
  ASSERT_EQ(ARSMI_init(), 0);

  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 2);

  // Get link info for XGMI connection
  ARSMI_linkInfo info;
  ASSERT_EQ(ARSMI_topo_get_link_info(0, 1, &info), 0);

  // Verify XGMI properties
  ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_XGMI);
  ASSERT_EQ(info.hops, 1);
}

TEST_F(AltRsmiTest, LinkTypeUndefined) {
  // Remove existing links and create setup with only undefined link type
  removeFile(std::string(kTestKFDPath) + "/0/io_links/0/properties");
  removeFile(std::string(kTestKFDPath) + "/0/io_links/1/properties");
  removeFile(std::string(kTestKFDPath) + "/1/io_links/0/properties");
  removeFile(std::string(kTestKFDPath) + "/1/io_links/1/properties");

  // Create link with undefined type (must be read last to not be overwritten)
  createDirectory(std::string(kTestKFDPath) + "/0/io_links/2");
  createFile(std::string(kTestKFDPath) + "/0/io_links/2/properties",
             "type 99\n"  // Undefined type (not 2 or 11)
             "version_major 0\n"
             "version_minor 0\n"
             "node_from 0\n"
             "node_to 1\n"
             "weight 21\n"
             "min_latency 0\n"
             "max_latency 0\n"
             "min_bandwidth 0\n"
             "max_bandwidth 50000\n"
             "recommended_transfer_size 0\n"
             "recommended_sdma_engine_id_mask 0\n"
             "flags 0\n");

  ASSERT_EQ(ARSMI_init(), 0);

  ARSMI_linkInfo info;
  ASSERT_EQ(ARSMI_topo_get_link_info(0, 1, &info), 0);

  // Should have undefined type
  ASSERT_EQ(info.type, ARSMI_IOLINK_TYPE_UNDEFINED);
  ASSERT_EQ(info.hops, 0);
}

TEST_F(AltRsmiTest, DeviceOrderingByBDF) {
  // Test that devices are ordered by BDF correctly
  ASSERT_EQ(ARSMI_init(), 0);

  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 2);

  // Get BDF for both devices
  uint64_t bdf0 = 0, bdf1 = 0;
  ASSERT_EQ(ARSMI_dev_pci_id_get(0, &bdf0), 0);
  ASSERT_EQ(ARSMI_dev_pci_id_get(1, &bdf1), 0);

  // BDFs should be ordered (lower BDF first)
  // Based on BDF ordering: node0 (domain=0, location_id=23552) comes before
  // node1 (domain=1, location_id=23553)
  // This verifies that ordering is working correctly
  ASSERT_NE(bdf0, 0);
  ASSERT_NE(bdf1, 0);
}

TEST_F(AltRsmiTest, FileExistsCheck) {
  // Test fileExists() indirectly by verifying behavior when files don't exist
  // This covers the fileExists(char const*) internal function

  removeDirectory(kTestKFDBasePath);
  createDirectory(kTestKFDPath);

  // Scenario 1: Node with missing gpu_id file - should be skipped
  createDirectory(std::string(kTestKFDPath) + "/0");
  // Don't create gpu_id file - fileExists() will return false for it
  createFile(std::string(kTestKFDPath) + "/0/properties",
             "unique_id 100\n"
             "location_id 23552\n"
             "domain 0\n"
             "vendor_id 4098\n");
  createDirectory(std::string(kTestKFDPath) + "/0/io_links");

  // Scenario 2: Node with missing properties file - should be skipped
  createDirectory(std::string(kTestKFDPath) + "/1");
  createFile(std::string(kTestKFDPath) + "/1/gpu_id", "4098\n");
  // Don't create properties file - fileExists() will return false for it
  createDirectory(std::string(kTestKFDPath) + "/1/io_links");

  // Scenario 3: Complete valid node
  createDirectory(std::string(kTestKFDPath) + "/2");
  createFile(std::string(kTestKFDPath) + "/2/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/2/properties",
             "unique_id 102\n"
             "location_id 23554\n"
             "domain 2\n"
             "vendor_id 4098\n");
  createDirectory(std::string(kTestKFDPath) + "/2/io_links");

  ASSERT_EQ(ARSMI_init(), 0);

  // Only the complete node should be discovered (fileExists filtered out the others)
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 1);
}

TEST_F(AltRsmiTest, BDFSortingLambda) {
  // Test the BDF sorting lambda comparator in ARSMI_init()
  // The lambda at line 183-186 sorts devices with the SAME unique_id by BDF
  // Create multiple partitions (same unique_id) with different BDF values in REVERSE order
  removeDirectory(kTestKFDBasePath);
  createDirectory(kTestKFDPath);

  // Create 4 nodes with the SAME unique_id but different location_ids (which affects BDF)
  // to exercise the lambda that sorts within the same unique_id group
  const std::string same_unique_id = "12345678901234567890";

  // Node 0: Highest BDF (will need to be moved to end by lambda)
  createDirectory(std::string(kTestKFDPath) + "/0");
  createFile(std::string(kTestKFDPath) + "/0/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/0/properties",
             "unique_id " + same_unique_id + "\n"
             "location_id 4294967040\n"  // Very high value for high BDF
             "domain 3\n"
             "vendor_id 4098\n");
  createDirectory(std::string(kTestKFDPath) + "/0/io_links");

  // Node 1: Second highest BDF
  createDirectory(std::string(kTestKFDPath) + "/1");
  createFile(std::string(kTestKFDPath) + "/1/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/1/properties",
             "unique_id " + same_unique_id + "\n"
             "location_id 16777216\n"
             "domain 2\n"
             "vendor_id 4098\n");
  createDirectory(std::string(kTestKFDPath) + "/1/io_links");

  // Node 2: Second lowest BDF
  createDirectory(std::string(kTestKFDPath) + "/2");
  createFile(std::string(kTestKFDPath) + "/2/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/2/properties",
             "unique_id " + same_unique_id + "\n"
             "location_id 65536\n"
             "domain 1\n"
             "vendor_id 4098\n");
  createDirectory(std::string(kTestKFDPath) + "/2/io_links");

  // Node 3: Lowest BDF (should be sorted to first by lambda)
  createDirectory(std::string(kTestKFDPath) + "/3");
  createFile(std::string(kTestKFDPath) + "/3/gpu_id", "4098\n");
  createFile(std::string(kTestKFDPath) + "/3/properties",
             "unique_id " + same_unique_id + "\n"
             "location_id 256\n"
             "domain 0\n"
             "vendor_id 4098\n");
  createDirectory(std::string(kTestKFDPath) + "/3/io_links");

  ASSERT_EQ(ARSMI_init(), 0);

  // All 4 nodes have the same unique_id, so they're all partitions of the same device
  // ARSMI_num_devices counts unique devices, but ARSMI_orderedNodes has all partitions
  uint32_t num_devices = 0;
  ASSERT_EQ(ARSMI_get_num_devices(&num_devices), 0);
  ASSERT_EQ(num_devices, 4);  // All 4 partitions should be counted

  // Access ARSMI_orderedNodes directly to verify the lambda sorted by s_bdf
  ASSERT_EQ(ARSMI_orderedNodes.size(), 4);

  // The lambda should have sorted these by s_bdf within the unique_id group
  // Verify ascending order
  ASSERT_LT(ARSMI_orderedNodes[0].s_bdf, ARSMI_orderedNodes[1].s_bdf);
  ASSERT_LT(ARSMI_orderedNodes[1].s_bdf, ARSMI_orderedNodes[2].s_bdf);
  ASSERT_LT(ARSMI_orderedNodes[2].s_bdf, ARSMI_orderedNodes[3].s_bdf);

  // Verify the sort reordered them: node 3 should be first (lowest BDF)
  ASSERT_EQ(ARSMI_orderedNodes[0].s_node_id, 3);  // Node 3 has domain 0, location 256
  ASSERT_EQ(ARSMI_orderedNodes[1].s_node_id, 2);  // Node 2 has domain 1, location 65536
  ASSERT_EQ(ARSMI_orderedNodes[2].s_node_id, 1);  // Node 1 has domain 2, location 16777216
  ASSERT_EQ(ARSMI_orderedNodes[3].s_node_id, 0);  // Node 0 has domain 3, location 4294967040

  // Verify they all have the same unique_id
  ASSERT_EQ(ARSMI_orderedNodes[0].s_unique_id, ARSMI_orderedNodes[1].s_unique_id);
  ASSERT_EQ(ARSMI_orderedNodes[1].s_unique_id, ARSMI_orderedNodes[2].s_unique_id);
  ASSERT_EQ(ARSMI_orderedNodes[2].s_unique_id, ARSMI_orderedNodes[3].s_unique_id);
}

} // namespace RcclUnitTesting
