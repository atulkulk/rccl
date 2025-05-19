#ifndef RCCL_MOCK_FUNCS_HPP
#define RCCL_MOCK_FUNCS_HPP

#ifndef RCCL_EXPOSE_STATIC

#include "info.h"
#include "comm.h"

void ncclDebugLog(ncclDebugLogLevel, unsigned long, char const*, int, char const*, ...) {};
ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  return ncclSuccess;
}
#endif  // RCCL_EXPOSE_STATIC

#endif  // RCCL_MOCK_FUNCS_HPP
