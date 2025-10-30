/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file RCCLGenericScopeGuard.hpp
 * @brief Generic RAII scope guard for automatic resource cleanup
 *
 * Provides a generic ScopeGuard template for exception-safe cleanup of any resource.
 * This is the C++ idiomatic alternative to goto-based cleanup patterns.
 *
 * @par Key Features:
 * - Exception-safe: Cleanup runs even if exception is thrown
 * - Dismissible: Can disable cleanup for ownership transfer
 * - Move-only: Prevents accidental double-cleanup
 * - Zero overhead: Fully inline and optimizable
 *
 * @par Usage Examples:
 * @code
 * // Basic cleanup
 * void* buffer = nullptr;
 * hipMalloc(&buffer, size);
 * auto guard = makeScopeGuard([&]() { if(buffer) hipFree(buffer); });
 * // Automatic cleanup on scope exit
 *
 * // Ownership transfer
 * auto guard = makeScopeGuard([&]() { cleanup(); });
 * // ... success path ...
 * guard.dismiss();  // Don't cleanup - caller owns it now
 *
 * // Quick one-liner
 * SCOPE_EXIT(if(resource) freeResource(resource));
 * @endcode
 *
 * @see makeScopeGuard for the recommended way to create guards
 * @see SCOPE_EXIT for quick one-liner cleanup
 * @see RCCLTestResourceGuards.hpp for dedicated resource guards
 */

#ifndef RCCL_GENERIC_SCOPE_GUARD_HPP
#define RCCL_GENERIC_SCOPE_GUARD_HPP

#include <utility> // For std::move

/**
 * @class ScopeGuard
 * @brief Generic RAII scope guard for custom cleanup logic
 *
 * Executes a cleanup function on scope exit (normal return, early return, or exception).
 * Useful for resources that don't have dedicated RAII guards or for one-off cleanup needs.
 *
 * **Features:**
 * - Exception-safe: Cleanup runs even if exception is thrown
 * - Dismissible: Can disable cleanup if resource ownership is transferred
 * - Move-only: Cannot be copied (prevents accidental double-cleanup)
 * - Zero overhead: Inline and optimizable
 *
 * **Advantages over goto-based cleanup:**
 * - Exception-safe (goto doesn't handle exceptions)
 * - Automatic (can't forget to cleanup)
 * - Correct cleanup order (LIFO guaranteed by C++ destructors)
 * - Composable (multiple guards work together seamlessly)
 * - Idiomatic C++ (standard RAII pattern)
 *
 * @par Basic Example:
 * @code
 * void example() {
 *     void* buffer = nullptr;
 *     hipMalloc(&buffer, size);
 *     auto cleanup = makeScopeGuard([&]() {
 *         if(buffer) hipFree(buffer);
 *     });
 *
 *     // Use buffer...
 *     // Cleanup automatically happens on scope exit
 * }
 * @endcode
 *
 * @par Dismissible Guard Example:
 * @code
 * ncclResult_t transferOwnership(ncclComm_t* out_comm) {
 *     ncclComm_t comm = nullptr;
 *     HIP_TEST_CHECK(ncclCommInitRank(&comm, ...));
 *     auto guard = makeScopeGuard([&]() {
 *         if(comm) ncclCommDestroy(comm);
 *     });
 *
 *     // Do work...
 *
 *     // Success - transfer ownership
 *     *out_comm = comm;
 *     guard.dismiss();  // Don't cleanup - caller owns it now
 *     return ncclSuccess;
 * }
 * @endcode
 *
 * @par Multiple Resources Example:
 * @code
 * ncclResult_t setupResources() {
 *     // Each resource gets its own guard
 *     ncclComm_t comm = nullptr;
 *     HIP_TEST_CHECK(ncclCommInitRank(&comm, ...));
 *     auto commGuard = makeScopeGuard([&]() { if(comm) ncclCommDestroy(comm); });
 *
 *     void* buf1 = nullptr;
 *     HIP_TEST_CHECK(hipMalloc(&buf1, size));
 *     auto buf1Guard = makeScopeGuard([&]() { if(buf1) hipFree(buf1); });
 *
 *     void* buf2 = nullptr;
 *     HIP_TEST_CHECK(hipMalloc(&buf2, size));
 *     auto buf2Guard = makeScopeGuard([&]() { if(buf2) hipFree(buf2); });
 *
 *     // All guards cleanup in LIFO order: buf2 -> buf1 -> comm
 *     return ncclSuccess;
 * }
 * @endcode
 *
 * @par Comparison with goto:
 * @code
 * // Error: goto-based (C-style, not exception-safe)
 * ncclResult_t setupResources() {
 *     ncclResult_t result = ncclSuccess;
 *     ncclComm_t comm = nullptr;
 *     void* buf = nullptr;
 *
 *     NCCLCHECK_GOTO(ncclCommInitRank(&comm, ...), result, cleanup);
 *     HIP_TEST_CHECK_GOTO(hipMalloc(&buf, size), result, cleanup);
 *
 *     return ncclSuccess;
 *
 * cleanup:  // Won't run if exception thrown!
 *     if(buf) hipFree(buf);
 *     if(comm) ncclCommDestroy(comm);
 *     return result;
 * }
 *
 * // Good: ScopeGuard (C++ RAII, exception-safe)
 * ncclResult_t setupResources() {
 *     ncclComm_t comm = nullptr;
 *     HIP_TEST_CHECK(ncclCommInitRank(&comm, ...));
 *     auto commGuard = makeScopeGuard([&]() { if(comm) ncclCommDestroy(comm); });
 *
 *     void* buf = nullptr;
 *     HIP_TEST_CHECK(hipMalloc(&buf, size));
 *     auto bufGuard = makeScopeGuard([&]() { if(buf) hipFree(buf); });
 *
 *     return ncclSuccess;  // Automatic cleanup even on exception!
 * }
 * @endcode
 *
 * @note Cleanup function should be noexcept or handle exceptions internally
 * @note Use dismiss() to prevent cleanup when transferring ownership
 * @note For common resources, prefer dedicated guards from RCCLTestResourceGuards.hpp
 *
 * @tparam Func Callable type (lambda, function pointer, functor)
 */
template<typename Func>
class ScopeGuard
{
    Func cleanup_; ///< Cleanup function to execute on scope exit
    bool dismissed_; ///< If true, skip cleanup (for ownership transfer)

public:
    /**
     * @brief Construct scope guard with cleanup function
     * @param f Cleanup function (usually a lambda)
     */
    explicit ScopeGuard(Func f) noexcept : cleanup_(std::move(f)), dismissed_(false) {}

    /**
     * @brief Destructor - executes cleanup if not dismissed
     *
     * Marked noexcept to ensure cleanup happens even during stack unwinding.
     * If cleanup throws, program will terminate (std::terminate).
     */
    ~ScopeGuard() noexcept
    {
        if(!dismissed_)
        {
            cleanup_();
        }
    }

    /**
     * @brief Dismiss the guard - prevent cleanup from running
     *
     * Use when transferring ownership of the resource to another scope.
     *
     * @par Example:
     * @code
     * auto guard = makeScopeGuard([&]() { cleanup(); });
     * // ... success path ...
     * guard.dismiss();  // Caller takes ownership, don't cleanup
     * @endcode
     */
    void dismiss() noexcept
    {
        dismissed_ = true;
    }

    /**
     * @brief Re-enable the guard after dismissal
     *
     * Rarely needed, but available if ownership transfer is cancelled.
     */
    void restore() noexcept
    {
        dismissed_ = false;
    }

    // Move-only semantics (prevent double-cleanup)

    /**
     * @brief Move constructor - transfers ownership of cleanup
     * @param other Guard to move from (will be dismissed)
     */
    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup_(std::move(other.cleanup_)), dismissed_(other.dismissed_)
    {
        other.dismissed_ = true; // Prevent moved-from guard from running cleanup
    }

    /**
     * @brief Move assignment operator - transfers ownership of cleanup
     * @param other Guard to move from (will be dismissed)
     * @return Reference to this guard
     */
    ScopeGuard& operator=(ScopeGuard&& other) noexcept
    {
        if(this != &other)
        {
            // Run our existing cleanup first (if not dismissed)
            if(!dismissed_)
            {
                cleanup_();
            }

            cleanup_   = std::move(other.cleanup_);
            dismissed_ = other.dismissed_;

            other.dismissed_ = true; // Prevent moved-from guard from running cleanup
        }
        return *this;
    }

    // Non-copyable (prevent double-cleanup)
    ScopeGuard(const ScopeGuard&)            = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
};

/**
 * @brief Helper function to create ScopeGuard with type deduction
 *
 * Creates a ScopeGuard without needing to specify the template parameter.
 * This is the recommended way to create scope guards.
 *
 * @par Example:
 * @code
 * auto guard = makeScopeGuard([&]() {
 *     // cleanup code
 * });
 * @endcode
 *
 * @par Real-World Example:
 * @code
 * ncclResult_t allocateResources() {
 *     void* buffer = nullptr;
 *     hipMalloc(&buffer, 1024);
 *     auto bufGuard = makeScopeGuard([&]() {
 *         if(buffer) hipFree(buffer);
 *     });
 *
 *     ncclComm_t comm = nullptr;
 *     ncclCommInitRank(&comm, ...);
 *     auto commGuard = makeScopeGuard([&]() {
 *         if(comm) ncclCommDestroy(comm);
 *     });
 *
 *     // Use resources...
 *     return ncclSuccess;  // Automatic cleanup!
 * }
 * @endcode
 *
 * @tparam Func Callable type (deduced from argument)
 * @param f Cleanup function
 * @return ScopeGuard<Func> instance
 */
template<typename Func>
ScopeGuard<Func> makeScopeGuard(Func f)
{
    return ScopeGuard<Func>(std::move(f));
}

/**
 * @def SCOPE_EXIT
 * @brief Convenience macro for creating anonymous scope guards
 *
 * Creates a scope guard with automatic variable name generation.
 * Useful for quick one-liners without needing to name the guard variable.
 *
 * @par Example:
 * @code
 * void* buffer = nullptr;
 * hipMalloc(&buffer, size);
 * SCOPE_EXIT(if(buffer) hipFree(buffer));
 *
 * // Use buffer...
 * // Automatically freed on scope exit
 * @endcode
 *
 * @par Multi-statement Example:
 * @code
 * FILE* f = fopen("file.txt", "r");
 * SCOPE_EXIT({
 *     if(f) {
 *         fflush(f);
 *         fclose(f);
 *     }
 * });
 * @endcode
 *
 * @par Real-World Example:
 * @code
 * void processData() {
 *     int fd = open("data.bin", O_RDONLY);
 *     SCOPE_EXIT(if(fd >= 0) close(fd));
 *
 *     void* mapped = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
 *     SCOPE_EXIT(if(mapped != MAP_FAILED) munmap(mapped, size));
 *
 *     // Process data...
 *     // Both resources automatically cleaned up
 * }
 * @endcode
 *
 * @note Cannot be dismissed (use makeScopeGuard for dismissible guards)
 * @note Variable name is auto-generated, so cannot be referenced later
 */
#define SCOPE_EXIT_CONCAT_IMPL(a, b) a##b
#define SCOPE_EXIT_CONCAT(a, b) SCOPE_EXIT_CONCAT_IMPL(a, b)
#define SCOPE_EXIT(code) \
    auto SCOPE_EXIT_CONCAT(scope_guard_, __LINE__) = makeScopeGuard([&]() { code; })

/**
 * @example ScopeGuard Usage Examples
 *
 * @par Example 1: Basic Cleanup
 * @code
 * void basicExample() {
 *     void* buffer = nullptr;
 *     hipMalloc(&buffer, 1024);
 *     auto guard = makeScopeGuard([&]() {
 *         if(buffer) hipFree(buffer);
 *     });
 *     // Automatic cleanup on scope exit
 * }
 * @endcode
 *
 * @par Example 2: Multiple Resources
 * @code
 * ncclResult_t multipleResources() {
 *     ncclComm_t comm = nullptr;
 *     HIP_TEST_CHECK(ncclCommInitRank(&comm, ...));
 *     auto commGuard = makeScopeGuard([&]() { if(comm) ncclCommDestroy(comm); });
 *
 *     void* buf = nullptr;
 *     HIP_TEST_CHECK(hipMalloc(&buf, size));
 *     auto bufGuard = makeScopeGuard([&]() { if(buf) hipFree(buf); });
 *
 *     return ncclSuccess;  // Cleanup: buf -> comm (LIFO)
 * }
 * @endcode
 *
 * @par Example 3: Ownership Transfer
 * @code
 * ncclResult_t createResource(ncclComm_t* out) {
 *     ncclComm_t comm = nullptr;
 *     HIP_TEST_CHECK(ncclCommInitRank(&comm, ...));
 *     auto guard = makeScopeGuard([&]() { if(comm) ncclCommDestroy(comm); });
 *
 *     // Do work...
 *
 *     *out = comm;
 *     guard.dismiss();  // Transfer ownership
 *     return ncclSuccess;
 * }
 * @endcode
 *
 * @par Example 4: SCOPE_EXIT Macro
 * @code
 * void quickCleanup() {
 *     void* buf = nullptr;
 *     hipMalloc(&buf, 1024);
 *     SCOPE_EXIT(if(buf) hipFree(buf));
 *
 *     // Use buf...
 *     // Automatic cleanup
 * }
 * @endcode
 */

#endif // RCCL_GENERIC_SCOPE_GUARD_HPP
