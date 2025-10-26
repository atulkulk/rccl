/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <hip/hip_runtime.h>
#include "nccl.h"
#include "net.h"
#include "transport.h"
#include <cstdlib>
#include <utility>

/**
 * @file RCCLTestResourceGuards.hpp
 * @brief Templatized RAII resource guards for automatic cleanup in tests
 *
 * This file provides generic RAII (Resource Acquisition Is Initialization)
 * guards using C++ templates to minimize code duplication. All guards ensure
 * resources are cleaned up even when ASSERT_* fails in tests.
 *
 * Usage Example:
 * @code
 * TEST_F(MyTest, Example) {
 *     BufferGuard bufferGuard(malloc(4096), true);
 *     ASSERT_NE(bufferGuard.get(), nullptr);  // If fails, buffer still freed!
 *
 *     NetConnectionGuard connGuard(net_);
 *     connGuard.setListenComm(listenComm);  // Auto-closed on any failure!
 *
 *     ASSERT_TRUE(SomeOperation());  // All resources cleaned up if this fails!
 * }
 * @endcode
 */

namespace RCCLTestGuards {

// ============================================================================
// Generic Resource Guard Template
// ============================================================================

/**
 * @class ResourceGuard
 * @brief Generic RAII guard template for single resources
 *
 * @tparam T Resource handle type
 * @tparam Deleter Functor type for cleanup
 */
template<typename T, typename Deleter>
class ResourceGuard {
private:
    T resource_;
    Deleter deleter_;
    bool owns_;

public:
    /**
     * @brief Construct a resource guard
     * @param resource Resource handle (can be nullptr/0)
     * @param deleter Cleanup function/functor
     */
    explicit ResourceGuard(T resource = T{}, Deleter deleter = Deleter{})
        : resource_(resource), deleter_(std::move(deleter)), owns_(true) {}

    /**
     * @brief Destructor - automatically cleans up resource
     */
    ~ResourceGuard() {
        if (owns_ && resource_) {
            deleter_(resource_);
        }
    }

    /**
     * @brief Get the resource handle
     * @return Resource handle
     */
    T get() const { return resource_; }

    /**
     * @brief Get pointer to resource handle (for API calls)
     * @return Pointer to resource handle
     */
    T* ptr() { return &resource_; }

    /**
     * @brief Set the resource handle
     * @param resource New resource handle
     */
    void set(T resource) { resource_ = resource; }

    /**
     * @brief Reset to a new resource (cleans up old resource if different)
     * @param resource New resource handle
     */
    void reset(T resource = T{}) {
        if (owns_ && resource_ && resource_ != resource) {
            deleter_(resource_);
        }
        resource_ = resource;
        owns_ = true;
    }

    /**
     * @brief Release ownership without cleanup
     * @return The resource handle (caller takes ownership)
     */
    T release() {
        owns_ = false;
        return resource_;
    }

    // Delete copy constructor and assignment operator
    ResourceGuard(const ResourceGuard&) = delete;
    ResourceGuard& operator=(const ResourceGuard&) = delete;

    // Allow move
    ResourceGuard(ResourceGuard&& other) noexcept
        : resource_(other.resource_), deleter_(std::move(other.deleter_)), owns_(other.owns_) {
        other.owns_ = false;
    }

    ResourceGuard& operator=(ResourceGuard&& other) noexcept {
        if (this != &other) {
            // Clean up current resource
            if (owns_ && resource_) {
                deleter_(resource_);
            }
            // Take ownership of other's resource
            resource_ = other.resource_;
            deleter_ = std::move(other.deleter_);
            owns_ = other.owns_;
            other.owns_ = false;
        }
        return *this;
    }
};

// ============================================================================
// Deleters (Cleanup Functors)
// ============================================================================

/**
 * @brief Deleter for host memory (free)
 */
struct HostMemoryDeleter {
    void operator()(void* ptr) const {
        if (ptr) free(ptr);
    }
};

/**
 * @brief Deleter for device memory (hipFree)
 */
struct DeviceMemoryDeleter {
    void operator()(void* ptr) const {
        if (ptr) hipFree(ptr);
    }
};

/**
 * @brief Deleter for HIP streams
 */
struct HipStreamDeleter {
    void operator()(hipStream_t stream) const {
        if (stream) hipStreamDestroy(stream);
    }
};

/**
 * @brief Deleter for HIP events
 */
struct HipEventDeleter {
    void operator()(hipEvent_t event) const {
        if (event) hipEventDestroy(event);
    }
};

/**
 * @brief Deleter for NCCL communicators
 */
struct NcclCommDeleter {
    void operator()(ncclComm_t comm) const {
        if (comm) ncclCommDestroy(comm);
    }
};

/**
 * @brief Deleter for NCCL registration handles
 */
struct NcclRegHandleDeleter {
    ncclComm_t comm;

    explicit NcclRegHandleDeleter(ncclComm_t c = nullptr) : comm(c) {}

    void operator()(void* reg_handle) const {
        if (reg_handle && comm) {
            ncclCommDeregister(comm, reg_handle);
        }
    }
};

/**
 * @brief Deleter for NET plugin memory handles
 */
struct NetMHandleDeleter {
    ncclNet_t* net;
    void* comm;

    NetMHandleDeleter(ncclNet_t* n = nullptr, void* c = nullptr)
        : net(n), comm(c) {}

    void operator()(void* mhandle) const {
        if (mhandle && comm && net) {
            net->deregMr(comm, mhandle);
        }
    }
};

/**
 * @brief Deleter for NET send communicators
 */
struct NetSendCommDeleter {
    ncclNet_t* net;

    explicit NetSendCommDeleter(ncclNet_t* n = nullptr) : net(n) {}

    void operator()(void* comm) const {
        if (comm && net) net->closeSend(comm);
    }
};

/**
 * @brief Deleter for NET recv communicators
 */
struct NetRecvCommDeleter {
    ncclNet_t* net;

    explicit NetRecvCommDeleter(ncclNet_t* n = nullptr) : net(n) {}

    void operator()(void* comm) const {
        if (comm && net) net->closeRecv(comm);
    }
};

/**
 * @brief Deleter for NET listen communicators
 */
struct NetListenCommDeleter {
    ncclNet_t* net;

    explicit NetListenCommDeleter(ncclNet_t* n = nullptr) : net(n) {}

    void operator()(void* comm) const {
        if (comm && net) net->closeListen(comm);
    }
};

/**
 * @brief Deleter for transport send resources
 */
struct TransportSendResourceDeleter {
    ncclTransport* transport;

    explicit TransportSendResourceDeleter(ncclTransport* t = nullptr) : transport(t) {}

    void operator()(ncclConnector* connector) const {
        if (connector && transport) {
            transport->send.free(connector);
        }
    }
};

/**
 * @brief Deleter for transport recv resources
 */
struct TransportRecvResourceDeleter {
    ncclTransport* transport;

    explicit TransportRecvResourceDeleter(ncclTransport* t = nullptr) : transport(t) {}

    void operator()(ncclConnector* connector) const {
        if (connector && transport) {
            transport->recv.free(connector);
        }
    }
};

// ============================================================================
// Type Aliases (Convenience Names)
// ============================================================================

/** @brief Guard for host memory buffers */
using HostBufferGuard = ResourceGuard<void*, HostMemoryDeleter>;

/** @brief Guard for device memory buffers */
using DeviceBufferGuard = ResourceGuard<void*, DeviceMemoryDeleter>;

/** @brief Guard for HIP streams */
using HipStreamGuard = ResourceGuard<hipStream_t, HipStreamDeleter>;

/** @brief Guard for HIP events */
using HipEventGuard = ResourceGuard<hipEvent_t, HipEventDeleter>;

/** @brief Guard for NCCL communicators */
using NcclCommGuard = ResourceGuard<ncclComm_t, NcclCommDeleter>;

/** @brief Guard for NCCL registration handles */
using NcclRegHandleGuard = ResourceGuard<void*, NcclRegHandleDeleter>;

/** @brief Guard for NET plugin memory handles */
using NetMHandleGuard = ResourceGuard<void*, NetMHandleDeleter>;

/** @brief Guard for NET send communicators */
using NetSendCommGuard = ResourceGuard<void*, NetSendCommDeleter>;

/** @brief Guard for NET recv communicators */
using NetRecvCommGuard = ResourceGuard<void*, NetRecvCommDeleter>;

/** @brief Guard for NET listen communicators */
using NetListenCommGuard = ResourceGuard<void*, NetListenCommDeleter>;

/** @brief Guard for transport send resources */
using TransportSendResourceGuard = ResourceGuard<ncclConnector*, TransportSendResourceDeleter>;

/** @brief Guard for transport recv resources */
using TransportRecvResourceGuard = ResourceGuard<ncclConnector*, TransportRecvResourceDeleter>;

// ============================================================================
// Specialized Guards (Complex Cases)
// ============================================================================

/**
 * @class BufferGuard
 * @brief RAII guard for host or device memory buffers
 *
 * Combines host and device memory management in one guard for convenience.
 */
class BufferGuard {
private:
    void* buffer_;
    bool is_host_;
    bool owns_;

public:
    /**
     * @brief Construct a buffer guard
     * @param buffer Pointer to buffer (can be nullptr)
     * @param is_host true for host memory, false for device memory
     */
    explicit BufferGuard(void* buffer = nullptr, bool is_host = true)
        : buffer_(buffer), is_host_(is_host), owns_(true) {}

    ~BufferGuard() {
        if (owns_ && buffer_) {
            if (is_host_) {
                free(buffer_);
            } else {
                hipFree(buffer_);
            }
        }
    }

    void* get() const { return buffer_; }
    void** ptr() { return &buffer_; }

    void reset(void* new_buffer = nullptr) {
        if (owns_ && buffer_ && buffer_ != new_buffer) {
            if (is_host_) {
                free(buffer_);
            } else {
                hipFree(buffer_);
            }
        }
        buffer_ = new_buffer;
        owns_ = true;
    }

    void* release() {
        owns_ = false;
        return buffer_;
    }

    // Delete copy, allow move
    BufferGuard(const BufferGuard&) = delete;
    BufferGuard& operator=(const BufferGuard&) = delete;

    BufferGuard(BufferGuard&& other) noexcept
        : buffer_(other.buffer_), is_host_(other.is_host_), owns_(other.owns_) {
        other.owns_ = false;
    }

    BufferGuard& operator=(BufferGuard&& other) noexcept {
        if (this != &other) {
            if (owns_ && buffer_) {
                if (is_host_) free(buffer_); else hipFree(buffer_);
            }
            buffer_ = other.buffer_;
            is_host_ = other.is_host_;
            owns_ = other.owns_;
            other.owns_ = false;
        }
        return *this;
    }
};

/**
 * @class NetConnectionGuard
 * @brief RAII guard for multiple NET plugin connections
 *
 * Manages send, recv, and listen communicators together.
 */
class NetConnectionGuard {
private:
    NetSendCommGuard send_guard_;
    NetRecvCommGuard recv_guard_;
    NetListenCommGuard listen_guard_;

public:
    explicit NetConnectionGuard(ncclNet_t* net)
        : send_guard_(nullptr, NetSendCommDeleter(net))
        , recv_guard_(nullptr, NetRecvCommDeleter(net))
        , listen_guard_(nullptr, NetListenCommDeleter(net)) {}

    // Setters
    void setSendComm(void* comm) { send_guard_.set(comm); }
    void setRecvComm(void* comm) { recv_guard_.set(comm); }
    void setListenComm(void* comm) { listen_guard_.set(comm); }

    // Getters
    void* getSendComm() const { return send_guard_.get(); }
    void* getRecvComm() const { return recv_guard_.get(); }
    void* getListenComm() const { return listen_guard_.get(); }

    // Release ownership
    void releaseSendComm() { send_guard_.release(); }
    void releaseRecvComm() { recv_guard_.release(); }
    void releaseListenComm() { listen_guard_.release(); }

    NetConnectionGuard(const NetConnectionGuard&) = delete;
    NetConnectionGuard& operator=(const NetConnectionGuard&) = delete;
};

/**
 * @class TransportResourceGuard
 * @brief RAII guard for transport send/recv resources
 *
 * Manages both send and recv transport resources together.
 */
class TransportResourceGuard {
private:
    TransportSendResourceGuard send_guard_;
    TransportRecvResourceGuard recv_guard_;

public:
    explicit TransportResourceGuard(ncclTransport* transport)
        : send_guard_(nullptr, TransportSendResourceDeleter(transport))
        , recv_guard_(nullptr, TransportRecvResourceDeleter(transport)) {}

    // Setters
    void setSendResources(ncclConnector* res) { send_guard_.set(res); }
    void setRecvResources(ncclConnector* res) { recv_guard_.set(res); }

    // Getters
    ncclConnector* getSendResources() const { return send_guard_.get(); }
    ncclConnector* getRecvResources() const { return recv_guard_.get(); }

    // Release ownership
    void releaseSendResources() { send_guard_.release(); }
    void releaseRecvResources() { recv_guard_.release(); }

    TransportResourceGuard(const TransportResourceGuard&) = delete;
    TransportResourceGuard& operator=(const TransportResourceGuard&) = delete;
};

} // namespace RCCLTestGuards
