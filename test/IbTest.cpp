#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"
#include "graph.h"
#include "utils.h"
#include "param.h"
#include "ibvcore.h"
#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <poll.h>
#include <sys/types.h>
#include <unistd.h>
#define ENABLE_TIMER 0
#include "timer.h"
#include <sys/utsname.h>

#include "ibvwrap.h"


#include "TestBed.hpp"
#define MAXNAMESIZE (64)

struct ncclIbStats
{
  int fatalErrorCount;
};
struct ncclIbMrCache 
{
  struct ncclIbMr *slots;
  int capacity, population;
};
struct alignas(64) ncclIbDev 
{
  pthread_mutex_t lock;
  int device;
  uint64_t guid;
  uint8_t portNum;
  uint8_t link;
  int speed;
  ibv_context* context;
  int pdRefs;
  ibv_pd* pd;
  char devName[MAXNAMESIZE];
  char* pciPath;
  char* virtualPciPath;
  int realPort;
  int maxQp;
  float latency;
  struct ncclIbMrCache mrCache;
  int ar; // ADAPTIVE_ROUTING
  struct ibv_port_attr portAttr;
  struct ncclIbStats stats;
  int dmaBufSupported;
};
//fwd declaration of function
void* envIbAddrRange(sa_family_t af, int* mask);
void ncclIbStatsFatalError(struct ncclIbStats* stat);
void ncclIbDevFatalError(struct ncclIbDev* dev);
void ncclIbQpFatalError(struct ibv_qp* qp);
void ncclIbCqFatalError(struct ibv_cq* cq);
sa_family_t envIbAddrFamily(void);
sa_family_t getGidAddrFamily(union ibv_gid* gid);
bool configuredGid(union ibv_gid* gid);
bool linkLocalGid(union ibv_gid* gid);
bool validGid(union ibv_gid* gid);
ncclResult_t ncclIbRoceGetVersionNum(const char* deviceName, int portNum, int gidIndex, int* version);
void* envIbAddrRange(sa_family_t af, int* mask);
bool matchGidAddrPrefix(sa_family_t af, void* prefix, int prefixlen, union ibv_gid* gid);
ncclResult_t ncclUpdateGidIndex(struct ibv_context* context, uint8_t portNum, sa_family_t af, void* prefix, int prefixlen, int roceVer, int gidIndexCandidate, int* gidIndex);

namespace RcclUnitTesting
{
  TEST(IbTest, UpdateGidIndex)
  {
    INFO("[IbTest] Test begin \n");

    uint8_t portNum = 1;
    INFO("[IbTest] portNum %u \n", portNum);

    
    struct ibv_context *context;


    int nIbDevs = 0;
    struct ibv_device** devices = NULL;

    ncclResult_t res0 = wrap_ibv_symbols();
    if(res0 != ncclSuccess)
    {
      INFO("[IbTest]Init Ibv verbs fail\n");
    }
    INFO("[IbTest]Init Ibv verbs success \n");

    ncclResult_t res1 = wrap_ibv_get_device_list(&devices, &nIbDevs);
    if(ncclSuccess != res1)
    {
      INFO("[IbTest] NET/IB : No devices res1 %u \n", res1);
      //assert(0);
    }

    INFO("[IbTest] NET/IB : Total present devices %u \n", nIbDevs);

    ncclResult_t res2 = wrap_ibv_open_device(&context, devices[0]);
    if (ncclSuccess != res2 || context == NULL)
    {
      INFO("[IbTest] NET/IB : Unable to open device %s  res2 %u\n", devices[0]->name, res2);
      //assert(0);
    }

    INFO("[IbTest] here\n");

    sa_family_t userAddrFamily = AF_INET;
    INFO("[IbTest] userAddrFamily %u \n", userAddrFamily);

    int prefixlen;
    void *prefix = envIbAddrRange(userAddrFamily, &prefixlen);
    INFO("[IbTest] prefixlen %u \n", prefixlen);

    int userRoceVersion = 1;//ncclParamIbRoceVersionNum();
    INFO("[IbTest] userRoceVersion %u \n", userRoceVersion);
    int *gidIndex = 0;
    int gidIndexNext = 1;
    
    ncclResult_t res = ncclRemoteError; //Dummy value
    res = ncclUpdateGidIndex(context, portNum, userAddrFamily, prefix, prefixlen, userRoceVersion, gidIndexNext, gidIndex);

    INFO("[IbTest] Result %u \n", res);

    INFO("[IbTest] Test complete \n");
  }

  TEST(IbTest, GidAddrPrefix)
  {
    INFO("[IbTest] Test begin \n");
    sa_family_t af = 2;
    struct in_addr t_struct;
    t_struct.s_addr = 0x12345678;

    void* prefix = (void*)&t_struct;
    int prefixlen = 0;

    union ibv_gid dummy_gid;
    for(int i=0;i<16;i++)
    {
      dummy_gid.raw[i] = 1;
    }
    bool res = false;

    INFO("[IbTest] init res %u \n", res);
    res = matchGidAddrPrefix(af, prefix, prefixlen, &dummy_gid);
    INFO("[IbTest] Final res %u \n", res);
    assert(res == true);
    INFO("[IbTest] Test complete \n");

  }
	TEST(IbTest, AddrRange) // Should be run with 'export NCCL_IB_ADDR_RANGE=192.168.1.1/'
  {
    INFO("[IbTest] Test begin \n");
    int mask = 64;
    sa_family_t af = 2;

    INFO("[IbTest] Init mask %u \n", mask);
    void *prefix = nullptr;
    //char* prefix = new char[512];
    //INFO("[IbTest] Init prefix %s \n", prefix);
    //INFO("[IbTest] Init prefix address %p \n", prefix);
    prefix = envIbAddrRange(af, &mask);
    //INFO("[IbTest] Final prefix %s \n", prefix);
    INFO("[IbTest] Final mask %s \n", mask);
    INFO("[IbTest] Final prefix address %p \n", prefix);

    printf("F char val = %d %d %d %d\n", ((char*)prefix)[0], ((char*)prefix)[1], ((char*)prefix)[2], ((char*)prefix)[3] );
    assert(prefix != nullptr);
    INFO("[IbTest] Test complete \n");
  }

  TEST(IbTest, Devftl)
	{
		//TestBed tb;
		struct ncclIbDev test_structure;

		//Config the test struct
		test_structure.stats.fatalErrorCount = 0;
		INFO("[IbTest] Init value %u \n", test_structure.stats.fatalErrorCount);

		ncclIbDevFatalError(&test_structure);
		
		INFO("[IbTest] new value %u \n", test_structure.stats.fatalErrorCount);
		assert(test_structure.stats.fatalErrorCount == 1);

		INFO("[IbTest] test complete\n");

		//tb.Finalize();
	}

  TEST(IbTest, Cqftl)
	{
		//TestBed tb;
		struct ibv_cq test_structure;
    struct ncclIbStats test_structure2;

    test_structure.cq_context = (void*)&test_structure2;
		struct ncclIbStats* t_ptr = (struct ncclIbStats*)test_structure.cq_context;

		//Config the test struct
		t_ptr->fatalErrorCount = 0;//counter
		INFO("[IbTest] Init value %u \n", t_ptr->fatalErrorCount);

		ncclIbCqFatalError(&test_structure);//This function will atomically increase the counter
		
		INFO("[IbTest] new value %u \n", t_ptr->fatalErrorCount);
		assert(t_ptr->fatalErrorCount == 1);

		INFO("[IbTest] test complete\n");

		//tb.Finalize();
	}

	TEST(IbTest, Qpftl)
	{
		//TestBed tb;
		struct ibv_qp test_structure;
    struct ncclIbStats test_structure2;

    test_structure.qp_context = (void*)&test_structure2;
		struct ncclIbStats* t_ptr = (struct ncclIbStats*)test_structure.qp_context;

		//Config the test struct
		t_ptr->fatalErrorCount = 0;//Counter
		INFO("[IbTest] Init value %u \n", t_ptr->fatalErrorCount);

		ncclIbQpFatalError(&test_structure);//This function will atomically increase the counter
		
		INFO("[IbTest] new value %u \n", t_ptr->fatalErrorCount);
		assert(t_ptr->fatalErrorCount == 1);

		INFO("[IbTest] test complete\n");

		//tb.Finalize();
	}

  
  TEST(IbTest, EnvAddrFamily)
  {
     INFO("[IbTest] test begin\n");
     sa_family_t res = 0;

     INFO("[IbTest] init val %u \n", res);
     res = envIbAddrFamily();
     INFO("\n [IbTest] final val %u \n", res);

     assert(res == AF_INET);
     INFO("[IbTest] test complete\n");
  }

  TEST(IbTest, GetGidAddrFamily)
  {
     INFO("[IbTest] test begin\n");
     sa_family_t res = 0;
     
     union ibv_gid dummy_gid;
     for(int i=0;i<16;i++)
     {
        dummy_gid.raw[i] = 1;
     }

     INFO("[IbTest] init val %u \n", res);
     res = getGidAddrFamily(&dummy_gid);
     INFO("[IbTest] final val %u \n", res);

     assert(res != 0);

     INFO("[IbTest] test complete\n");
  }

  TEST(IbTest, ConfiguredGid)
  {
     INFO("[IbTest] test begin\n");
     bool res = false;
     
     union ibv_gid dummy_gid;
     for(int i=0;i<16;i++)
     {
        dummy_gid.raw[i] = 1;
     }

     INFO("[IbTest] init val %u \n", res);
     res = configuredGid(&dummy_gid);
     INFO("[IbTest] final val %u \n", res);

     assert(res == true);

     INFO("[IbTest] test complete\n");
  }

  

  TEST(IbTest, LinkLocalGid)
  {
     INFO("[IbTest] test begin\n");
     bool res = false;
     
     union ibv_gid* dummy_gid = new ibv_gid;
     struct in6_addr *dummy_a = (struct in6_addr *)dummy_gid->raw;
     dummy_a->s6_addr32[0] = htonl(0xfe800000);
     dummy_a->s6_addr32[1] = 0UL;
     /*for(int i=0;i<16;i++)
     {
        dummy_gid.raw[i] = 1;
     }*/

     INFO("[IbTest] init val %u \n", res);
     res = linkLocalGid(dummy_gid);
     INFO("[IbTest] final val %u \n", res);
     
     delete dummy_gid;
     INFO("[IbTest] test complete\n");
  }

  TEST(IbTest, ValidGid)
  {
     INFO("[IbTest] test begin\n");
     bool res = false;
     
     union ibv_gid* dummy_gid = new ibv_gid;
     struct in6_addr *dummy_a = (struct in6_addr *)dummy_gid->raw;
     dummy_a->s6_addr32[0] = htonl(0xfe800000);
     dummy_a->s6_addr32[1] = 0UL;

     INFO("init val %u \n", res);
     res = validGid(dummy_gid);
     INFO("[IbTest] final val %u \n", res);
    
     delete dummy_gid;
     INFO("[IbTest] test complete\n");
  }

  

  TEST(IbTest, RoceVer)
  {
    INFO("[IbTest] test begin\n");
    ncclResult_t res = ncclRemoteError;
    const char* deviceName = "mlx5_0";
    int portNum = 1;
    int gidIndex = 0;
    int version = -1;

    INFO("[IbTest] init val %u \n init version %d \n", res, version);
    res = ncclIbRoceGetVersionNum(deviceName, portNum, gidIndex, &version);
    INFO("[IbTest] Final val %u \n Final version %d \n", res, version);

    assert(res == ncclSuccess);

    INFO("[IbTest] test complete\n");

  }
}













