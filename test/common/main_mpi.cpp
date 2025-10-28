/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>

#ifdef MPI_TESTS_ENABLED

    #include "MPITestBase.hpp" // For TEST_INIT macro
    #include "RCCLMPIEnvironment.hpp"
    #include <array>
    #include <atomic>
    #include <fcntl.h>
    #include <memory>
    #include <mpi.h>
    #include <optional>
    #include <string>
    #include <string_view>
    #include <thread>
    #include <unistd.h>

namespace
{

// Thread for tee-ing output (rank 0 only)
class TeeThread
{
public:
    TeeThread(int read_fd, int console_fd, int log_fd)
        : read_fd_(read_fd), console_fd_(console_fd), log_fd_(log_fd), running_(true)
    {
        thread_ = std::thread([this]() { this->tee_loop(); });
    }

    ~TeeThread()
    {
        running_ = false;
        if(thread_.joinable())
        {
            thread_.join();
        }
    }

    // Delete copy/move operations
    TeeThread(const TeeThread&)            = delete;
    TeeThread& operator=(const TeeThread&) = delete;
    TeeThread(TeeThread&&)                 = delete;
    TeeThread& operator=(TeeThread&&)      = delete;

private:
    void tee_loop()
    {
        std::array<char, 4096> buffer;
        while(running_)
        {
            const auto bytes_read = ::read(read_fd_, buffer.data(), buffer.size());
            if(bytes_read <= 0)
            {
                if(bytes_read == 0 || errno != EINTR)
                {
                    break; // EOF or error
                }
                continue;
            }

            // Write to console
            [[maybe_unused]] auto console_written = ::write(console_fd_, buffer.data(), bytes_read);

            // Write to log file
            [[maybe_unused]] auto log_written = ::write(log_fd_, buffer.data(), bytes_read);
        }
    }

    int               read_fd_;
    int               console_fd_;
    int               log_fd_;
    std::atomic<bool> running_;
    std::thread       thread_;
};

// RAII wrapper for file descriptors
class FileDescriptor
{
public:
    explicit FileDescriptor(int fd = -1) noexcept : fd_(fd) {}

    ~FileDescriptor()
    {
        if(fd_ >= 0)
        {
            ::close(fd_);
        }
    }

    // Move-only semantics
    FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.fd_)
    {
        other.fd_ = -1;
    }

    FileDescriptor& operator=(FileDescriptor&& other) noexcept
    {
        if(this != &other)
        {
            if(fd_ >= 0)
            {
                ::close(fd_);
            }
            fd_       = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    // Delete copy operations
    FileDescriptor(const FileDescriptor&)            = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    [[nodiscard]] int get() const noexcept
    {
        return fd_;
    }
    [[nodiscard]] bool is_valid() const noexcept
    {
        return fd_ >= 0;
    }

    int release() noexcept
    {
        const auto fd = fd_;
        fd_           = -1;
        return fd;
    }

private:
    int fd_;
};

// Per-rank logging configuration
struct RankLogConfig
{
    std::optional<FileDescriptor> log_fd;
    std::optional<FileDescriptor> saved_stdout;
    std::optional<FileDescriptor> saved_stderr;
    std::optional<FileDescriptor> pipe_read_fd;
    std::optional<FileDescriptor> pipe_write_fd;
    std::unique_ptr<TeeThread>    tee_thread;
    bool                          logging_enabled{false};
    bool                          is_rank_zero{false};
};

// Setup per-rank logging if environment variable is set
[[nodiscard]] std::optional<RankLogConfig> setup_rank_logging(int rank)
{
    const auto* env_value                = std::getenv("RCCL_MPI_LOG_ALL_RANKS");
    const bool  per_rank_logging_enabled = (env_value && std::string_view{env_value} == "1");

    RankLogConfig config;
    config.logging_enabled = per_rank_logging_enabled;
    config.is_rank_zero    = (rank == 0);

    // Non-zero ranks: Always redirect output (either to log file or /dev/null)
    if(rank != 0)
    {
        // Save original stdout/stderr
        config.saved_stdout = FileDescriptor{::dup(STDOUT_FILENO)};
        config.saved_stderr = FileDescriptor{::dup(STDERR_FILENO)};

        if(!config.saved_stdout->is_valid() || !config.saved_stderr->is_valid())
        {
            std::cerr << "Rank " << rank << ": Failed to duplicate stdout/stderr\n";
            return std::nullopt;
        }

        if(per_rank_logging_enabled)
        {
            // Per-rank logging enabled: Redirect to log file
            const auto log_filename
                = std::string{"rccl_test_rank_"} + std::to_string(rank) + ".log";
            const auto log_fd = ::open(log_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

            if(log_fd < 0)
            {
                std::cerr << "Rank " << rank << ": Failed to create log file: " << log_filename
                          << '\n';
                return std::nullopt;
            }

            config.log_fd = FileDescriptor{log_fd};

            // Redirect stdout/stderr to log file
            if(::dup2(log_fd, STDOUT_FILENO) < 0 || ::dup2(log_fd, STDERR_FILENO) < 0)
            {
                std::cerr << "Rank " << rank << ": Failed to redirect to log file\n";
                return std::nullopt;
            }
        }
        else
        {
            // Default: Suppress all output by redirecting to /dev/null
            const auto null_fd = ::open("/dev/null", O_WRONLY);
            if(null_fd < 0)
            {
                std::cerr << "Rank " << rank << ": Failed to open /dev/null\n";
                return std::nullopt;
            }

            // Redirect stdout/stderr to /dev/null
            if(::dup2(null_fd, STDOUT_FILENO) < 0 || ::dup2(null_fd, STDERR_FILENO) < 0)
            {
                std::cerr << "Rank " << rank << ": Failed to redirect to /dev/null\n";
                ::close(null_fd);
                return std::nullopt;
            }

            ::close(null_fd);
        }

        // Disable buffering for immediate output
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::setvbuf(stderr, nullptr, _IONBF, 0);

        return config;
    }

    // Rank 0: Only redirect if per-rank logging is enabled (for tee functionality)
    if(!per_rank_logging_enabled)
    {
        return std::nullopt; // Rank 0 outputs to console normally
    }

    // Create log file for this rank
    const auto log_filename = std::string{"rccl_test_rank_"} + std::to_string(rank) + ".log";
    const auto log_fd       = ::open(log_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if(log_fd < 0)
    {
        std::cerr << "Rank " << rank << ": Failed to create log file: " << log_filename << '\n';
        return std::nullopt;
    }

    config.log_fd = FileDescriptor{log_fd};

    // Rank 0 with per-rank logging: Output to BOTH console AND log file (tee behavior)
    // Print banner before redirection
    TEST_INFO("Per-Rank Logging ENABLED (RCCL_MPI_LOG_ALL_RANKS=1)");
    TEST_INFO("Rank 0     : Output to BOTH console AND %s", log_filename.c_str());
    TEST_INFO("Ranks 1-N  : Output redirected to rccl_test_rank_<N>.log");
    TEST_INFO("Location   : Log files created in current working directory");

    // Save original stdout/stderr for tee thread
    config.saved_stdout = FileDescriptor{::dup(STDOUT_FILENO)};
    config.saved_stderr = FileDescriptor{::dup(STDERR_FILENO)};

    if(!config.saved_stdout->is_valid() || !config.saved_stderr->is_valid())
    {
        std::cerr << "Rank " << rank << ": Failed to duplicate stdout/stderr\n";
        return std::nullopt;
    }

    // Create pipes for tee functionality
    int pipe_fds[2];
    if(::pipe(pipe_fds) < 0)
    {
        std::cerr << "Rank " << rank << ": Failed to create pipe\n";
        return std::nullopt;
    }

    config.pipe_read_fd  = FileDescriptor{pipe_fds[0]};
    config.pipe_write_fd = FileDescriptor{pipe_fds[1]};

    // Start tee thread to duplicate output to both console and log file
    try
    {
        config.tee_thread = std::make_unique<TeeThread>(config.pipe_read_fd->get(),
                                                        config.saved_stdout->get(),
                                                        log_fd);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Rank " << rank << ": Failed to start tee thread: " << e.what() << '\n';
        return std::nullopt;
    }

    // Redirect stdout/stderr to the pipe write end
    if(::dup2(config.pipe_write_fd->get(), STDOUT_FILENO) < 0
       || ::dup2(config.pipe_write_fd->get(), STDERR_FILENO) < 0)
    {
        std::cerr << "Rank " << rank << ": Failed to redirect to pipe\n";
        return std::nullopt;
    }

    // Disable buffering for immediate output
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    return config;
}

// Restore original stdout/stderr
void restore_rank_logging(RankLogConfig& config)
{
    // Only restore if we actually redirected (have saved stdout/stderr)
    if(!config.saved_stdout || !config.saved_stdout->is_valid())
    {
        return;
    }

    // Flush any pending output
    std::fflush(stdout);
    std::fflush(stderr);

    // CRITICAL: Restore stdout/stderr BEFORE closing pipe to release all references to pipe write end
    // When dup2() redirected stdout/stderr to the pipe, those became additional references
    // The tee thread won't get EOF until ALL write ends are closed
    if(config.saved_stdout && config.saved_stdout->is_valid())
    {
        ::dup2(config.saved_stdout->get(), STDOUT_FILENO);
    }

    if(config.saved_stderr && config.saved_stderr->is_valid())
    {
        ::dup2(config.saved_stderr->get(), STDERR_FILENO);
    }

    if(config.is_rank_zero && config.tee_thread)
    {
        // For rank 0 with per-rank logging: Stop the tee thread
        // Close the pipe write end to signal EOF to the tee thread
        // (stdout/stderr already restored above, so this closes the last reference)
        config.pipe_write_fd.reset();

        // Wait for tee thread to finish processing (should exit quickly now that it gets EOF)
        config.tee_thread.reset();

        // Close pipe read end
        config.pipe_read_fd.reset();
    }
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    // Early MPI initialization to get rank information
    auto provided = int{};
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &RCCLMPIEnvironment::world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &RCCLMPIEnvironment::world_size);
    RCCLMPIEnvironment::mpi_initialized = true;

    const auto world_rank = RCCLMPIEnvironment::world_rank;
    const auto world_size = RCCLMPIEnvironment::world_size;

    // Check if per-rank logging is enabled
    auto       rank_log_config          = setup_rank_logging(world_rank);
    const auto per_rank_logging_enabled = rank_log_config && rank_log_config->logging_enabled;

    // Print initialization message (banner already printed by setup_rank_logging for rank 0)
    if(world_rank == 0 && !per_rank_logging_enabled)
    {
        TEST_INFO("MPI initialized - World size: %d, Thread support: %d", world_size, provided);
    }

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Suppress GTest output for non-zero ranks (unless per-rank logging is enabled)
    // This is done by deleting GTest listeners for non-zero ranks
    // Note: stdout/stderr are already redirected for non-zero ranks by setup_rank_logging
    if(world_rank != 0 && !per_rank_logging_enabled)
    {
        auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
        delete listeners.Release(listeners.default_xml_generator());
    }

    // Set up the RCCL MPI environment for all tests
    ::testing::AddGlobalTestEnvironment(new RCCLMPIEnvironment());

    // Run all tests
    const auto ret_code = RUN_ALL_TESTS();

    // Restore original output if per-rank logging was enabled
    if(rank_log_config)
    {
        restore_rank_logging(*rank_log_config);
    }

    return ret_code;
}

#else // MPI_TESTS_ENABLED not defined

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    std::fprintf(stderr,
                 "ERROR: MPI tests are not enabled. Please build with ENABLE_MPI_TESTS=ON\n");
    std::fprintf(stderr, "Usage: cmake -DENABLE_MPI_TESTS=ON -DMPI_PATH=/path/to/mpi ..\n");
    return 1;
}

#endif // MPI_TESTS_ENABLED
