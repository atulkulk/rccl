/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "gtest/gtest.h"

#include "core.h"
#include "graph.h"
#include "ibvcore.h"
#include "nccl.h"
#include "net.h"
#include "param.h"
#include "socket.h"
#include "utils.h"
#include <assert.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#define ENABLE_TIMER 0
#include "timer.h"
#include <sys/utsname.h>

#include "ibvwrap.h"

#define MAXNAMESIZE (64)

struct ncclIbStats {
  int fatalErrorCount;
};
struct ncclIbMrCache {
  struct ncclIbMr *slots;
  int capacity, population;
};
struct alignas(64) ncclIbDev {
  pthread_mutex_t lock;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context *context;
  int pdRefs;
  ibv_pd *pd;
  char devName[MAXNAMESIZE];
  char *pciPath;
  char *virtualPciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
};
// fwd declaration of function
void *envIbAddrRange(sa_family_t af, int *mask);
void ncclIbStatsFatalError(struct ncclIbStats *stat);
void ncclIbDevFatalError(struct ncclIbDev *dev);
void ncclIbQpFatalError(struct ibv_qp *qp);
void ncclIbCqFatalError(struct ibv_cq *cq);
sa_family_t envIbAddrFamily(void);
sa_family_t getGidAddrFamily(union ibv_gid *gid);
bool configuredGid(union ibv_gid *gid);
bool linkLocalGid(union ibv_gid *gid);
bool validGid(union ibv_gid *gid);
ncclResult_t ncclIbRoceGetVersionNum(const char *deviceName, int portNum,
                                     int gidIndex, int *version);
void *envIbAddrRange(sa_family_t af, int *mask);
bool matchGidAddrPrefix(sa_family_t af, void *prefix, int prefixlen,
                        union ibv_gid *gid);
ncclResult_t ncclUpdateGidIndex(struct ibv_context *context, uint8_t portNum,
                                sa_family_t af, void *prefix, int prefixlen,
                                int roceVer, int gidIndexCandidate,
                                int *gidIndex);

namespace RcclUnitTesting {
TEST(IbTests, GidAddrPrefix) {
  INFO(NCCL_LOG_INFO, "[IbTests] Test begin \n");
  struct in_addr prefix;
  inet_pton(AF_INET, "192.168.1.0", &prefix);

  union ibv_gid gid;
  inet_pton(AF_INET, "192.168.1.10",
            &gid.raw[12]); // IPv4 address is stored in the last 4 bytes of the
                           // IPv6 address

  bool result_ipv4 = matchGidAddrPrefix(AF_INET, &prefix, 24, &gid);
  INFO(NCCL_LOG_INFO, "[IbTests] Match result ipv4: %s\n",
       result_ipv4 ? "true" : "false");
  ASSERT_EQ(result_ipv4, true);

  struct in6_addr prefix2;
  inet_pton(AF_INET6, "2001:0db8:85a3::", &prefix2);

  union ibv_gid gid2;
  inet_pton(AF_INET6, "2001:0db8:85a3:0000:0000:8a2e:0370:7334", &gid2.raw);

  bool result_ipv6 = matchGidAddrPrefix(AF_INET6, &prefix2, 64, &gid2);
  INFO(NCCL_LOG_INFO, "Match result ipv6: %s\n",
       result_ipv6 ? "true" : "false");
  ASSERT_EQ(result_ipv6, true);
  INFO(NCCL_LOG_INFO, "[IbTests] Test complete \n");
}
TEST(IbTests, AddrRangeIPV4) // Should be run with 'export
                             // NCCL_IB_ADDR_RANGE=192.168.1.1/24' for ipv4 test
{
  int mask = 0;
  INFO(NCCL_LOG_INFO, "[IbTests] Test begin \n");
  struct in_addr *ptr4 = (struct in_addr *)envIbAddrRange(AF_INET, &mask);
  INFO(NCCL_LOG_INFO, "[IbTests] IP Address 4: %s\n", inet_ntoa(*ptr4));

  ASSERT_EQ(mask, 24); // Check if the mask is set correctly for IPv4
  INFO(NCCL_LOG_INFO, "[IbTests] Test complete \n");
}
TEST(IbTests,
     AddrRangeIPV6) // Should be run with 'export
                    // NCCL_IB_ADDR_RANGE=2001:0db8:85a3::/64' for ipv6 test
{
  int mask = 0;

  INFO(NCCL_LOG_INFO, "[IbTests] Test complete \n");
  struct in6_addr *ptr6 = (struct in6_addr *)envIbAddrRange(AF_INET6, &mask);
  char ipv6_str[INET6_ADDRSTRLEN];

  if (inet_ntop(AF_INET6, ptr6, ipv6_str, INET6_ADDRSTRLEN) == NULL) {
    INFO(NCCL_LOG_INFO, "[IbTests] inet_ntop failed");
    assert(0);
  }
  INFO(NCCL_LOG_INFO, "IP Address 6: %s\n", ipv6_str);
  ASSERT_EQ(mask, 64); // Check if the mask is set correctly for IPv6

  INFO(NCCL_LOG_INFO, "[IbTests] Test complete \n");
}

TEST(IbTests, Devftl) {
  struct ncclIbDev test_structure;

  // Config the test struct
  test_structure.stats.fatalErrorCount = 0;
  INFO(NCCL_LOG_INFO, "[IbTests] Init value %u \n",
       test_structure.stats.fatalErrorCount);

  ncclIbDevFatalError(&test_structure);

  INFO(NCCL_LOG_INFO, "[IbTests] new value %u \n",
       test_structure.stats.fatalErrorCount);
  ASSERT_EQ(test_structure.stats.fatalErrorCount, 1);

  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, Cqftl) {
  struct ibv_cq test_structure;
  struct ncclIbStats test_structure2;

  test_structure.cq_context = (void *)&test_structure2;
  struct ncclIbStats *t_ptr = (struct ncclIbStats *)test_structure.cq_context;

  // Config the test struct
  t_ptr->fatalErrorCount = 0; // counter
  INFO(NCCL_LOG_INFO, "[IbTests] Init value %u \n", t_ptr->fatalErrorCount);

  ncclIbCqFatalError(
      &test_structure); // This function will atomically increase the counter

  INFO(NCCL_LOG_INFO, "[IbTests] new value %u \n", t_ptr->fatalErrorCount);
  ASSERT_EQ(t_ptr->fatalErrorCount, 1);

  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, Qpftl) {

  struct ibv_qp test_structure;
  struct ncclIbStats test_structure2;

  test_structure.qp_context = (void *)&test_structure2;
  struct ncclIbStats *t_ptr = (struct ncclIbStats *)test_structure.qp_context;

  // Config the test struct
  t_ptr->fatalErrorCount = 0; // Counter
  INFO(NCCL_LOG_INFO, "[IbTests] Init value %u \n", t_ptr->fatalErrorCount);

  ncclIbQpFatalError(
      &test_structure); // This function will atomically increase the counter

  INFO(NCCL_LOG_INFO, "[IbTests] new value %u \n", t_ptr->fatalErrorCount);
  ASSERT_EQ(t_ptr->fatalErrorCount, 1);

  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, EnvAddrFamily) {
  INFO(NCCL_LOG_INFO, "[IbTests] test begin\n");
  sa_family_t res = 0;

  INFO(NCCL_LOG_INFO, "[IbTests] init val %u \n", res);
  res = envIbAddrFamily();
  INFO(NCCL_LOG_INFO, "\n [IbTests] final val %u \n", res);

  assert(res == AF_INET);
  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, GetGidAddrFamily) {
  INFO(NCCL_LOG_INFO, "[IbTests] test begin for ipv4\n");
  sa_family_t res = 0;

  struct in6_addr tst;
  tst.s6_addr32[1] = 0;
  tst.s6_addr32[0] = 0;
  tst.s6_addr32[2] =
      htonl(0x0000ffff); // Configuring such that the address is IPv4 mapped
  union ibv_gid *tst_cast = (ibv_gid *)&tst;

  INFO(NCCL_LOG_INFO, "[IbTests] init val %u \n", res);
  res = getGidAddrFamily(tst_cast);
  INFO(NCCL_LOG_INFO, "[IbTests] final val %u \n", res);
  INFO(NCCL_LOG_INFO, "[IbTests] AF_INET %u AF_INET6 %u \n", AF_INET, AF_INET6);

  ASSERT_EQ(res, AF_INET);

  INFO(NCCL_LOG_INFO, "[IbTests] test begin for ipv6\n");
  union ibv_gid gid;
  // Set the first 96 bits to a value that is not 0:0:0:ffff (for IPv4-mapped)
  // or ff0e:0:0:ffff (for IPv4-mapped multicast)
  gid.raw[0] = 0x20; // Example value
  gid.raw[1] = 0x01; // Example value
  gid.raw[2] = 0x0d; // Example value
  gid.raw[3] = 0xb8; // Example value
  gid.raw[4] = 0x85; // Example value
  gid.raw[5] = 0xa3; // Example value
  gid.raw[6] = 0x00; // Example value
  gid.raw[7] = 0x00; // Example value
  // The rest can be any value
  gid.raw[8] = 0x8a;  // Example value
  gid.raw[9] = 0x2e;  // Example value
  gid.raw[10] = 0x03; // Example value
  gid.raw[11] = 0x70; // Example value
  gid.raw[12] = 0x73; // Example value
  gid.raw[13] = 0x34; // Example value
  gid.raw[14] = 0x00; // Example value
  gid.raw[15] = 0x00; // Example value

  res = 0;
  INFO(NCCL_LOG_INFO, "[IbTests] init val %u \n", res);
  res = getGidAddrFamily(&gid);
  INFO(NCCL_LOG_INFO, "[IbTests] final val %u \n", res);
  ASSERT_EQ(res, AF_INET6);
  INFO(NCCL_LOG_INFO, "[IbTests] AF_INET %u AF_INET6 %u \n", AF_INET, AF_INET6);
  ASSERT_EQ(res, AF_INET6);
  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, ConfiguredGid) {
  INFO(NCCL_LOG_INFO, "[IbTests] test begin\n");
  bool res = false;

  union ibv_gid gid;
  // Set the first 32 bits to a non-zero value that is not 0xfe800000
  gid.raw[0] = 0x20; // Example value
  gid.raw[1] = 0x01; // Example value
  gid.raw[2] = 0x0d; // Example value
  gid.raw[3] = 0xb8; // Example value
  // Set the next 96 bits to any non-zero value
  gid.raw[4] = 0x85;  // Example value
  gid.raw[5] = 0xa3;  // Example value
  gid.raw[6] = 0x00;  // Example value
  gid.raw[7] = 0x00;  // Example value
  gid.raw[8] = 0x8a;  // Example value
  gid.raw[9] = 0x2e;  // Example value
  gid.raw[10] = 0x03; // Example value
  gid.raw[11] = 0x70; // Example value
  gid.raw[12] = 0x73; // Example value
  gid.raw[13] = 0x34; // Example value
  gid.raw[14] = 0x00; // Example value
  gid.raw[15] = 0x00; // Example value

  INFO(NCCL_LOG_INFO, "[IbTests] init val %u \n", res);
  res = configuredGid(&gid);
  INFO(NCCL_LOG_INFO, "[IbTests] final val %u \n", res);

  ASSERT_EQ(res, true);

  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, LinkLocalGid) {
  INFO(NCCL_LOG_INFO, "[IbTests] test begin\n");
  bool res = false;

  union ibv_gid gid;
  // Set the first 32 bits to 0xfe800000
  gid.raw[0] = 0xfe;
  gid.raw[1] = 0x80;
  gid.raw[2] = 0x00;
  gid.raw[3] = 0x00;
  // Set the next 32 bits to 0x00000000
  gid.raw[4] = 0x00;
  gid.raw[5] = 0x00;
  gid.raw[6] = 0x00;
  gid.raw[7] = 0x00;
  // The rest can be any value
  for (int i = 8; i < 16; i++) {
    gid.raw[i] = 0x00;
  }

  INFO(NCCL_LOG_INFO, "[IbTests] init val %u \n", res);
  res = linkLocalGid(&gid);
  INFO(NCCL_LOG_INFO, "[IbTests] final val %u \n", res);
  ASSERT_EQ(res, true);

  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, ValidGid) {
  INFO(NCCL_LOG_INFO, "[IbTests] test begin\n");
  bool res = false;

  union ibv_gid gid;
  // Set the first 32 bits to a value that is not 0xfe800000
  gid.raw[0] = 0x20; // Example value
  gid.raw[1] = 0x01; // Example value
  gid.raw[2] = 0x0d; // Example value
  gid.raw[3] = 0xb8; // Example value
  // Set the next 32 bits to any non-zero value
  gid.raw[4] = 0x00;
  gid.raw[5] = 0x00;
  gid.raw[6] = 0x00;
  gid.raw[7] = 0x01; // Non-zero value
  // The rest can be any value
  gid.raw[8] = 0x00;
  gid.raw[9] = 0x00;
  gid.raw[10] = 0x00;
  gid.raw[11] = 0x00;
  gid.raw[12] = 0x00;
  gid.raw[13] = 0x00;
  gid.raw[14] = 0x00;
  gid.raw[15] = 0x00;

  INFO(NCCL_LOG_INFO, "init val %u \n", res);
  res = validGid(&gid);
  INFO(NCCL_LOG_INFO, "[IbTests] final val %u \n", res);
  ASSERT_EQ(res, true);

  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}

TEST(IbTests, RoceVer) {
  INFO(NCCL_LOG_INFO, "[IbTests] test begin\n");
  ncclResult_t res = ncclRemoteError;
  const char *deviceName = "mlx5_0";
  int portNum = 1;
  int gidIndex = 0;
  int version = -1;

  INFO(NCCL_LOG_INFO, "[IbTests] init val %u \n init version %d \n", res,
       version);
  res = ncclIbRoceGetVersionNum(deviceName, portNum, gidIndex, &version);
  INFO(NCCL_LOG_INFO, "[IbTests] Final val %u \n Final version %d \n", res,
       version);

  assert(res == ncclSuccess);

  INFO(NCCL_LOG_INFO, "[IbTests] test complete\n");
}
} // namespace RcclUnitTesting
