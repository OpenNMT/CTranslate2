/**
 * ██████  ███████       ████████ ██   ██ ██████  ███████  █████  ██████          ██████   ██████   ██████  ██
 * ██   ██ ██      ██ ██    ██    ██   ██ ██   ██ ██      ██   ██ ██   ██         ██   ██ ██    ██ ██    ██ ██
 * ██████  ███████          ██    ███████ ██████  █████   ███████ ██   ██         ██████  ██    ██ ██    ██ ██
 * ██   ██      ██ ██ ██    ██    ██   ██ ██   ██ ██      ██   ██ ██   ██         ██      ██    ██ ██    ██ ██
 * ██████  ███████          ██    ██   ██ ██   ██ ███████ ██   ██ ██████  ███████ ██       ██████   ██████  ███████
 *
 * @file BS_thread_pool.hpp
 * @author Barak Shoshany (baraksh@gmail.com) (https://baraksh.com/)
 * @version 5.1.0
 * @date 2026-01-03
 * @copyright Copyright (c) 2021-2026 Barak Shoshany. Licensed under the MIT license. If you found this project useful, please consider starring it on GitHub! If you use this library in software of any kind, please provide a link to the GitHub repository https://github.com/bshoshany/thread-pool in the source code and documentation. If you use this library in published research, please cite it as follows: Barak Shoshany, "A C++17 Thread Pool for High-Performance Scientific Computing", doi:10.1016/j.softx.2024.101687, SoftwareX 26 (2024) 101687, arXiv:2105.00613
 *
 * @brief `BS::thread_pool`: a fast, lightweight, modern, and easy-to-use C++17/C++20/C++23 thread pool library. This header file contains the entire library, and is the only file needed to use the library.
 */

#ifndef BS_THREAD_POOL_HPP
#define BS_THREAD_POOL_HPP

// We need to include <version> since if we're using `import std` it will not define any feature-test macros.
#ifdef __has_include
    #if __has_include(<version>)
        #include <version> // NOLINT(misc-include-cleaner)
    #endif
#endif

// At the time of this release, there is a bug in Clang with libc++ where using `std::jthread` in a C++20 module causes a compilation error. As a workaround, until the bug is fixed, the thread pool library automatically falls back to `std::thread` if it detects that Clang and libc++ are being used together with C++20 modules. This workaround can be disabled by defining `BS_THREAD_POOL_DISABLE_WORKAROUNDS` when compiling the module. TODO: Remove this workaround when the bug is fixed.
#if defined(__clang__) && defined(_LIBCPP_VERSION) && defined(BS_THREAD_POOL_MODULE) && (__cplusplus >= 202002L) && !defined(BS_THREAD_POOL_DISABLE_WORKAROUNDS)
    #ifdef __cpp_lib_jthread
        #undef __cpp_lib_jthread
    #endif
#endif

// At the time of this release, there is a bug when using GCC with libstdc++ on Windows via MSYS2 where the `BS.thread_pool` module doesn't compile if both native extensions and `import std` are enabled. As a workaround, until the bug is fixed, the thread pool library automatically falls back to header files if it detects that GCC and libstdc++ are being used together with the C++23 `std` module on Windows. This workaround can be disabled by defining `BS_THREAD_POOL_DISABLE_WORKAROUNDS` when compiling the module. TODO: Remove this workaround when the bug is fixed.
#if (defined(__GNUC__) && defined(_GLIBCXX_RELEASE) && defined(_WIN32)) && !defined(BS_THREAD_POOL_DISABLE_WORKAROUNDS)
    #ifdef BS_THREAD_POOL_IMPORT_STD
        #undef BS_THREAD_POOL_IMPORT_STD
    #endif
#endif

// In GCC with libstdc++ on Linux, loading the system headers after `import std` causes compilation errors, so we load them first.
#ifdef BS_THREAD_POOL_NATIVE_EXTENSIONS
    #if defined(_WIN32)
        #ifndef WIN32_LEAN_AND_MEAN
            #define WIN32_LEAN_AND_MEAN
        #endif
        #ifndef NOMINMAX
            #define NOMINMAX
        #endif
        #include <windows.h>
    #elif defined(__linux__) || defined(__APPLE__)
        #include <pthread.h>
        #include <sched.h>
        #include <sys/resource.h>
        #include <unistd.h>
        #if defined(__linux__)
            #include <sys/syscall.h>
            #include <sys/sysinfo.h>
        #endif
    #else
        #undef BS_THREAD_POOL_NATIVE_EXTENSIONS
    #endif
#endif

// If the macro `BS_THREAD_POOL_IMPORT_STD` is defined, import the C++ Standard Library as a module. Otherwise, include the relevant Standard Library header files.
#if defined(BS_THREAD_POOL_IMPORT_STD) && (__cplusplus >= 202004L)
    // Only allow importing the `std` module if the library itself is imported as a module. If the library is included as a header file, this will force the program that included the header file to also import `std`, which is not desirable and can lead to compilation errors if the program `#include`s any Standard Library header files.
    #ifdef BS_THREAD_POOL_MODULE
import std;
    #else
        #error "The thread pool library cannot import the C++ Standard Library as a module using `import std` if the library itself is not imported as a module. Either use `import BS.thread_pool` to import the library, or remove the `BS_THREAD_POOL_IMPORT_STD` macro. Aborting compilation."
    #endif
#else
    #undef BS_THREAD_POOL_IMPORT_STD

    #include <algorithm>
    #include <chrono>
    #include <condition_variable>
    #include <cstddef>
    #include <cstdint>
    #include <functional>
    #include <future>
    #include <iostream>
    #include <limits>
    #include <memory>
    #include <mutex>
    #include <optional>
    #include <queue>
    #include <string>
    #include <thread>
    #include <tuple>
    #include <type_traits>
    #include <utility>
    #include <variant>
    #include <vector>

    #ifdef __cpp_concepts
        #include <concepts>
    #endif
    #ifdef __cpp_exceptions
        #include <exception>
        #include <stdexcept>
    #endif
    #ifdef __cpp_impl_three_way_comparison
        #include <compare>
    #endif
    #ifdef __cpp_lib_int_pow2
        #include <bit>
    #endif
    #ifdef __cpp_lib_jthread
        #include <stop_token>
    #endif
#endif

// On Linux, <sys/sysmacros.h> defines macros called `major` and `minor`, which we undefine here to prevent conflicts.
#ifdef major
    #undef major
#endif
#ifdef minor
    #undef minor
#endif

// On Windows, <windows.h> defines macros called `min` and `max`, which we undefine here to prevent conflicts.
#ifdef min
    #undef min
#endif
#ifdef max
    #undef max
#endif

/**
 * @brief A namespace used by Barak Shoshany's projects.
 */
namespace BS {
// Macros indicating the version of the thread pool library.
#define BS_THREAD_POOL_VERSION_MAJOR 5
#define BS_THREAD_POOL_VERSION_MINOR 1
#define BS_THREAD_POOL_VERSION_PATCH 0

/**
 * @brief A struct used to store a version number, which can be checked and compared at compilation time.
 */
struct [[nodiscard]] version
{
    constexpr version(const std::uint64_t major_, const std::uint64_t minor_, const std::uint64_t patch_) noexcept : major(major_), minor(minor_), patch(patch_) {}

// In C++20 and later we can use the spaceship operator `<=>` to automatically generate comparison operators. In C++17 we have to define them manually.
#ifdef __cpp_impl_three_way_comparison
    std::strong_ordering operator<=>(const version&) const = default;
#else
    [[nodiscard]] constexpr friend bool operator==(const version& lhs, const version& rhs) noexcept
    {
        return std::tuple(lhs.major, lhs.minor, lhs.patch) == std::tuple(rhs.major, rhs.minor, rhs.patch);
    }

    [[nodiscard]] constexpr friend bool operator!=(const version& lhs, const version& rhs) noexcept
    {
        return !(lhs == rhs);
    }

    [[nodiscard]] constexpr friend bool operator<(const version& lhs, const version& rhs) noexcept
    {
        return std::tuple(lhs.major, lhs.minor, lhs.patch) < std::tuple(rhs.major, rhs.minor, rhs.patch);
    }

    [[nodiscard]] constexpr friend bool operator>=(const version& lhs, const version& rhs) noexcept
    {
        return !(lhs < rhs);
    }

    [[nodiscard]] constexpr friend bool operator>(const version& lhs, const version& rhs) noexcept
    {
        return std::tuple(lhs.major, lhs.minor, lhs.patch) > std::tuple(rhs.major, rhs.minor, rhs.patch);
    }

    [[nodiscard]] constexpr friend bool operator<=(const version& lhs, const version& rhs) noexcept
    {
        return !(lhs > rhs);
    }
#endif

    [[nodiscard]] std::string to_string() const
    {
        return std::to_string(major) + '.' + std::to_string(minor) + '.' + std::to_string(patch);
    }

    friend std::ostream& operator<<(std::ostream& stream, const version& ver)
    {
        stream << ver.to_string();
        return stream;
    }

    std::uint64_t major;
    std::uint64_t minor;
    std::uint64_t patch;
}; // struct version

/**
 * @brief The version of the thread pool library.
 */
inline constexpr version thread_pool_version(BS_THREAD_POOL_VERSION_MAJOR, BS_THREAD_POOL_VERSION_MINOR, BS_THREAD_POOL_VERSION_PATCH);

#ifdef BS_THREAD_POOL_MODULE
// If the library is being compiled as a module, ensure that the version of the module file matches the version of the header file.
static_assert(thread_pool_version == version(BS_THREAD_POOL_MODULE), "The versions of BS.thread_pool.cppm and BS_thread_pool.hpp do not match. Aborting compilation.");
/**
 * @brief A flag indicating whether the thread pool library was compiled as a C++20 module.
 */
inline constexpr bool thread_pool_module = true;
#else
/**
 * @brief A flag indicating whether the thread pool library was compiled as a C++20 module.
 */
inline constexpr bool thread_pool_module = false;
#endif

#ifdef BS_THREAD_POOL_IMPORT_STD
/**
 * @brief A flag indicating whether the thread pool library imported the C++23 Standard Library module using `import std`.
 */
inline constexpr bool thread_pool_import_std = true;
#else
/**
 * @brief A flag indicating whether the thread pool library imported the C++23 Standard Library module using `import std`.
 */
inline constexpr bool thread_pool_import_std = false;
#endif

#ifdef BS_THREAD_POOL_NATIVE_EXTENSIONS
/**
 * @brief A flag indicating whether the thread pool library's native extensions are enabled.
 */
inline constexpr bool thread_pool_native_extensions = true;
#else
/**
 * @brief A flag indicating whether the thread pool library's native extensions are enabled.
 */
inline constexpr bool thread_pool_native_extensions = false;
#endif

/**
 * @brief The type used for the bitmask template parameter of the thread pool.
 */
using opt_t = std::uint8_t;

/**
 * @brief An enumeration class of flags to be used in the bitmask template parameter of `BS::thread_pool` to enable optional features.
 */
enum class tp : opt_t
{
    /**
     * @brief No optional features enabled.
     */
    none = 0,

    /**
     * @brief Enable task priority.
     */
    priority = 1 << 0,

    /**
     * @brief Enable pausing.
     */
    pause = 1 << 1,

    /**
     * @brief Enable wait deadlock checks.
     */
    wait_deadlock_checks = 1 << 2
};

// NOLINTBEGIN(bugprone-macro-parentheses)
#define BS_THREAD_POOL_DEFINE_BITWISE_OPERATOR(ENUM, OP) \
    constexpr ENUM operator OP(const ENUM lhs, const ENUM rhs) noexcept \
    { \
        return static_cast<ENUM>(static_cast<std::underlying_type_t<ENUM>>(lhs) OP static_cast<std::underlying_type_t<ENUM>>(rhs)); \
    } \
    constexpr ENUM& operator OP##=(ENUM& lhs, const ENUM rhs) noexcept \
    { \
        return lhs = lhs OP rhs; \
    }
// NOLINTEND(bugprone-macro-parentheses)

BS_THREAD_POOL_DEFINE_BITWISE_OPERATOR(tp, &)
BS_THREAD_POOL_DEFINE_BITWISE_OPERATOR(tp, |)
BS_THREAD_POOL_DEFINE_BITWISE_OPERATOR(tp, ^)

constexpr tp operator~(const tp value) noexcept
{
    return static_cast<tp>(~static_cast<std::underlying_type_t<tp>>(value));
}

template <tp>
class thread_pool;

#ifdef __cpp_lib_move_only_function
/**
 * @brief The template to use to store functions in the task queue and other places. In C++23 and later we use `std::move_only_function`.
 */
using std::move_only_function;
#else
template <typename...>
class move_only_function;

/**
 * @brief A simple polyfill for `std::move_only_function`, to be used if C++23 features are not available. Note that it does not have all the features of `std::move_only_function`, only the minimum needed for the thread pool library.
 *
 * @tparam R The return type of the function.
 * @tparam Args The argument types of the function.
 */
template <typename R, typename... Args>
class move_only_function<R(Args...)>
{
public:
    move_only_function() = default;
    move_only_function(move_only_function&&) noexcept = default;
    move_only_function& operator=(move_only_function&&) noexcept = default;
    move_only_function(const move_only_function&) = delete;
    move_only_function& operator=(const move_only_function&) = delete;
    ~move_only_function() = default;

    template <typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, move_only_function> && std::is_invocable_r_v<R, F&, Args...>>>
    move_only_function(F&& func) : ptr(std::make_unique<func_model<std::decay_t<F>>>(std::forward<F>(func))) {} // NOLINT(hicpp-explicit-conversions)

    R operator()(Args... args)
    {
        return ptr->call(std::forward<Args>(args)...);
    }

private:
    struct func_concept
    {
        virtual ~func_concept() = default;
        virtual R call(Args... args) = 0;
    };

    template <typename F>
    struct func_model final : func_concept
    {
        template <typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<T>, func_model>>>
        explicit func_model(T&& func) : stored_func(std::forward<T>(func)) {}

        R call(Args... args) override
        {
            if constexpr (std::is_void_v<R>)
            {
                std::invoke(stored_func, std::forward<Args>(args)...);
            }
            else
            {
                return std::invoke(stored_func, std::forward<Args>(args)...);
            }
        }

        F stored_func;
    };

    std::unique_ptr<func_concept> ptr = nullptr;
};
#endif

/**
 * @brief The type of tasks in the task queue.
 */
using task_t = move_only_function<void()>;

#ifdef __cpp_lib_jthread
/**
 * @brief The type of threads to use. In C++20 and later we use `std::jthread`.
 */
using thread_t = std::jthread;
    // The following macros are used to determine how to stop the workers. In C++20 and later we can use `std::stop_token`.
    #define BS_THREAD_POOL_WORKER_TOKEN const std::stop_token &stop_token,
    #define BS_THREAD_POOL_WAIT_TOKEN , stop_token
    #define BS_THREAD_POOL_STOP_CONDITION stop_token.stop_requested()
    #define BS_THREAD_POOL_OR_STOP_CONDITION
#else
/**
 * @brief The type of threads to use. In C++17 we use `std::thread`.
 */
using thread_t = std::thread;
    // The following macros are used to determine how to stop the workers. In C++17 we use a manual flag `workers_running`.
    #define BS_THREAD_POOL_WORKER_TOKEN
    #define BS_THREAD_POOL_WAIT_TOKEN
    #define BS_THREAD_POOL_STOP_CONDITION !workers_running
    #define BS_THREAD_POOL_OR_STOP_CONDITION || !workers_running
#endif

/**
 * @brief A type used to indicate the priority of a task. Defined to be a signed integer with a width of exactly 8 bits (-128 to +127).
 */
using priority_t = std::int8_t;

/**
 * @brief An enum containing some pre-defined priorities for convenience.
 */
enum pr : priority_t // NOLINT(cppcoreguidelines-use-enum-class) This cannot be an `enum class` because we need the numerical values.
{
    lowest = -128,
    low = -64,
    normal = 0,
    high = +64,
    highest = +127
};

/**
 * @brief A helper struct to store a task with an assigned priority.
 */
struct [[nodiscard]] pr_task
{
    /**
     * @brief Construct a new task with an assigned priority.
     *
     * @param task_ The task.
     * @param priority_ The desired priority.
     */
    explicit pr_task(task_t&& task_, const priority_t priority_ = 0) noexcept(std::is_nothrow_move_constructible_v<task_t>) : task(std::move(task_)), priority(priority_) {}

    /**
     * @brief Compare the priority of two tasks.
     *
     * @param lhs The first task.
     * @param rhs The second task.
     * @return `true` if the first task has a lower priority than the second task, `false` otherwise.
     */
    [[nodiscard]] friend bool operator<(const pr_task& lhs, const pr_task& rhs) noexcept
    {
        return lhs.priority < rhs.priority;
    }

    /**
     * @brief The task. It is `mutable` so it can be moved out of the `const` reference returned by `std::priority_queue::top()`.
     */
    mutable task_t task;

    /**
     * @brief The priority of the task.
     */
    priority_t priority = 0;
}; // struct pr_task

// In C++20 and later we can use concepts. In C++17 we instead use SFINAE ("Substitution Failure Is Not An Error") with `std::enable_if_t`.
#ifdef __cpp_concepts
    #define BS_THREAD_POOL_IF_PAUSE_ENABLED template <bool P = pause_enabled> requires(P)
template <typename F>
concept init_func_c = std::invocable<F> || std::invocable<F, std::size_t>;
    #define BS_THREAD_POOL_INIT_FUNC_CONCEPT(F) init_func_c F
#else
    #define BS_THREAD_POOL_IF_PAUSE_ENABLED template <bool P = pause_enabled, typename = std::enable_if_t<P>>
    #define BS_THREAD_POOL_INIT_FUNC_CONCEPT(F) typename F, typename = std::enable_if_t<std::is_invocable_v<F> || std::is_invocable_v<F, std::size_t>> // NOLINT(bugprone-macro-parentheses)
#endif

/**
 * @brief A helper class to facilitate waiting for and/or getting the results of multiple futures at once.
 *
 * @tparam T The return type of the futures.
 */
template <typename T>
class [[nodiscard]] multi_future : public std::vector<std::future<T>>
{
public:
    // Inherit all constructors from the base class `std::vector`.
    using std::vector<std::future<T>>::vector;

    /**
     * @brief Get the results from all the futures stored in this `BS::multi_future`, rethrowing any stored exceptions.
     *
     * @return If the futures return `void`, this function returns `void` as well. Otherwise, it returns a vector containing the results.
     */
    [[nodiscard]] std::conditional_t<std::is_void_v<T>, void, std::vector<T>> get()
    {
        if constexpr (std::is_void_v<T>)
        {
            for (std::future<T>& future : *this)
                future.get();
            return;
        }
        else
        {
            std::vector<T> results;
            results.reserve(this->size());
            for (std::future<T>& future : *this)
                results.push_back(future.get());
            return results;
        }
    }

    /**
     * @brief Check how many of the futures stored in this `BS::multi_future` are ready.
     *
     * @return The number of ready futures.
     */
    [[nodiscard]] std::size_t ready_count() const
    {
        std::size_t count = 0;
        for (const std::future<T>& future : *this)
        {
            if (future.wait_for(std::chrono::duration<double>::zero()) == std::future_status::ready)
                ++count;
        }
        return count;
    }

    /**
     * @brief Check if all the futures stored in this `BS::multi_future` are valid.
     *
     * @return `true` if all futures are valid, `false` if at least one of the futures is not valid.
     */
    [[nodiscard]] bool valid() const noexcept
    {
        bool is_valid = true;
        for (const std::future<T>& future : *this)
            is_valid = is_valid && future.valid();
        return is_valid;
    }

    /**
     * @brief Wait for all the futures stored in this `BS::multi_future`.
     */
    void wait() const
    {
        for (const std::future<T>& future : *this)
            future.wait();
    }

    /**
     * @brief Wait for all the futures stored in this `BS::multi_future`, but stop waiting after the specified duration has passed. This function first waits for the first future for the desired duration. If that future is ready before the duration expires, this function waits for the second future for whatever remains of the duration. It continues similarly until the duration expires.
     *
     * @tparam R An arithmetic type representing the number of ticks to wait.
     * @tparam P An `std::ratio` representing the length of each tick in seconds.
     * @param duration The amount of time to wait.
     * @return `true` if all futures have been waited for before the duration expired, `false` otherwise.
     */
    template <typename R, typename P>
    bool wait_for(const std::chrono::duration<R, P>& duration) const
    {
        const std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
        for (const std::future<T>& future : *this)
        {
            future.wait_for(duration - (std::chrono::steady_clock::now() - start_time));
            if (duration < std::chrono::steady_clock::now() - start_time)
                return false;
        }
        return true;
    }

    /**
     * @brief Wait for all the futures stored in this `BS::multi_future`, but stop waiting after the specified time point has been reached. This function first waits for the first future until the desired time point. If that future is ready before the time point is reached, this function waits for the second future until the desired time point. It continues similarly until the time point is reached.
     *
     * @tparam C The type of the clock used to measure time.
     * @tparam D An `std::chrono::duration` type used to indicate the time point.
     * @param timeout_time The time point at which to stop waiting.
     * @return `true` if all futures have been waited for before the time point was reached, `false` otherwise.
     */
    template <typename C, typename D>
    bool wait_until(const std::chrono::time_point<C, D>& timeout_time) const
    {
        for (const std::future<T>& future : *this)
        {
            future.wait_until(timeout_time);
            if (timeout_time < C::now())
                return false;
        }
        return true;
    }
}; // class multi_future

/**
 * @brief A helper class to divide a range into blocks. Used by `detach_blocks()`, `submit_blocks()`, `detach_loop()`, and `submit_loop()`.
 *
 * @tparam T The type of the indices. Should be a signed or unsigned integer.
 */
template <typename T>
class [[nodiscard]] blocks
{
public:
    /**
     * @brief Construct a `blocks` object with the given specifications.
     *
     * @param first_index_ The first index in the range.
     * @param index_after_last_ The index after the last index in the range.
     * @param num_blocks_ The desired number of blocks to divide the range into.
     */
    blocks(const T first_index_, const T index_after_last_, const std::size_t num_blocks_) noexcept : num_blocks(num_blocks_), first_index(first_index_), index_after_last(index_after_last_)
    {
        if (index_after_last > first_index)
        {
            const std::size_t total_size = static_cast<std::size_t>(index_after_last - first_index);
            num_blocks = std::min(num_blocks, total_size);
            block_size = total_size / num_blocks;
            remainder = total_size % num_blocks;
            if (block_size == 0)
            {
                block_size = 1;
                num_blocks = (total_size > 1) ? total_size : 1;
            }
        }
        else
        {
            num_blocks = 0;
        }
    }

    /**
     * @brief Get the index after the last index of a block.
     *
     * @param block The block number.
     * @return The index after the last index.
     */
    [[nodiscard]] T end(const std::size_t block) const noexcept
    {
        return (block == num_blocks - 1) ? index_after_last : start(block + 1);
    }

    /**
     * @brief Get the number of blocks. Note that this may be different than the desired number of blocks that was passed to the constructor.
     *
     * @return The number of blocks.
     */
    [[nodiscard]] std::size_t get_num_blocks() const noexcept
    {
        return num_blocks;
    }

    /**
     * @brief Get the first index of a block.
     *
     * @param block The block number.
     * @return The first index.
     */
    [[nodiscard]] T start(const std::size_t block) const noexcept
    {
        return first_index + static_cast<T>(block * block_size) + static_cast<T>(block < remainder ? block : remainder);
    }

private:
    /**
     * @brief The size of each block (except possibly the last block).
     */
    std::size_t block_size = 0;

    /**
     * @brief The number of blocks.
     */
    std::size_t num_blocks = 0;

    /**
     * @brief The remainder obtained after dividing the total size by the number of blocks.
     */
    std::size_t remainder = 0;

    /**
     * @brief The first index in the range.
     */
    T first_index = 0;

    /**
     * @brief The index after the last index in the range.
     */
    T index_after_last = 0;
}; // class blocks

/**
 * @brief A function object class used by `detach_blocks()` and `submit_blocks()` to execute a block function over a specified range of indices.
 *
 * @tparam T The type of the indices.
 * @tparam F The type of the function.
 * @tparam R The return type of the function (can be `void`).
 */
template <typename T, typename F, typename R>
struct block_task
{
    R operator()()
    {
        return (*block_ptr)(start, end);
    }

    std::shared_ptr<std::decay_t<F>> block_ptr;
    T start;
    T end;
}; // struct block_task

/**
 * @brief A function object class used by `detach_loop()` and `submit_loop()` to execute a loop function over a specified range of indices.
 *
 * @tparam T The type of the indices.
 * @tparam F The type of the function.
 */
template <typename T, typename F>
struct loop_task
{
    void operator()()
    {
        for (T i = start; i < end; ++i)
            (*loop_ptr)(i);
    }

    std::shared_ptr<std::decay_t<F>> loop_ptr;
    T start;
    T end;
}; // struct loop_task

/**
 * @brief A function object class used by `detach_sequence()` and `submit_sequence()` to execute a sequence function over a specified index.
 *
 * @tparam T The type of the index.
 * @tparam F The type of the function.
 * @tparam R The return type of the function (can be `void`).
 */
template <typename T, typename F, typename R>
struct sequence_task
{
    R operator()()
    {
        return (*sequence_ptr)(i);
    }

    std::shared_ptr<std::decay_t<F>> sequence_ptr;
    T i;
}; // struct sequence_task

/**
 * @brief A class that takes a function with a return value (but no arguments), and constructs a task with no return value along with a future used to retrieve the function's return value once the task is executed. Used by `submit_task()` and `submit_bulk()`.
 *
 * @tparam R The return type of the function (can be `void`).
 */
template <typename R>
struct task_and_future
{
    template <typename F, typename = std::enable_if_t<!std::is_same_v<std::decay_t<F>, task_and_future<R>>>>
    explicit task_and_future(F&& func)
    {
        std::promise<R> promise;
        future = promise.get_future();
        task = [task = std::forward<F>(func), promise = std::move(promise)]() mutable
        {
#ifdef __cpp_exceptions
            try
            {
#endif
                if constexpr (std::is_void_v<R>)
                {
                    task();
                    promise.set_value();
                }
                else
                {
                    promise.set_value(task());
                }
#ifdef __cpp_exceptions
            }
            catch (...)
            {
                try
                {
                    promise.set_exception(std::current_exception());
                }
                catch (...)
                {
                }
            }
#endif
        };
    }

    std::future<R> future;
    task_t task;
}; // struct task_and_future

#ifdef __cpp_exceptions
/**
 * @brief An exception that will be thrown by `wait()`, `wait_for()`, and `wait_until()` if the user tries to call them from within a thread of the same pool, which would result in a deadlock. Only used if the flag `BS::tp::wait_deadlock_checks` is enabled in the template parameter of `BS::thread_pool`.
 */
struct [[nodiscard]] wait_deadlock : public std::runtime_error
{
    wait_deadlock() : std::runtime_error("BS::wait_deadlock") {};
};
#endif

#ifdef BS_THREAD_POOL_NATIVE_EXTENSIONS
    #if defined(_WIN32)
/**
 * @brief An enum containing pre-defined OS-specific process priority values for portability.
 */
enum class os_process_priority
{
    idle = IDLE_PRIORITY_CLASS,
    below_normal = BELOW_NORMAL_PRIORITY_CLASS,
    normal = NORMAL_PRIORITY_CLASS,
    above_normal = ABOVE_NORMAL_PRIORITY_CLASS,
    high = HIGH_PRIORITY_CLASS,
    realtime = REALTIME_PRIORITY_CLASS
};

/**
 * @brief An enum containing pre-defined OS-specific thread priority values for portability.
 */
enum class os_thread_priority
{
    idle = THREAD_PRIORITY_IDLE,
    lowest = THREAD_PRIORITY_LOWEST,
    below_normal = THREAD_PRIORITY_BELOW_NORMAL,
    normal = THREAD_PRIORITY_NORMAL,
    above_normal = THREAD_PRIORITY_ABOVE_NORMAL,
    highest = THREAD_PRIORITY_HIGHEST,
    realtime = THREAD_PRIORITY_TIME_CRITICAL
};
    #elif defined(__linux__) || defined(__APPLE__)
/**
 * @brief An enum containing pre-defined OS-specific process priority values for portability.
 */
enum class os_process_priority
{
    idle = PRIO_MAX - 2,
    below_normal = PRIO_MAX / 2,
    normal = 0,
    above_normal = PRIO_MIN / 3,
    high = PRIO_MIN * 2 / 3,
    realtime = PRIO_MIN
};

/**
 * @brief An enum containing pre-defined OS-specific thread priority values for portability.
 */
enum class os_thread_priority
{
    idle,
    lowest,
    below_normal,
    normal,
    above_normal,
    highest,
    realtime
};
    #endif

/**
 * @brief Get the processor affinity of the current process using the current platform's native API. This should work on Windows and Linux, but is not possible on macOS as the native API does not allow it.
 *
 * @return An `std::optional` object, optionally containing the processor affinity of the current process as an `std::vector<bool>` where each element corresponds to a logical processor. If the returned object does not contain a value, then the affinity could not be determined. On macOS, this function always returns `std::nullopt`.
 */
[[nodiscard]] inline std::optional<std::vector<bool>> get_os_process_affinity()
{
    #if defined(_WIN32)
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0)
        return std::nullopt;
        #ifdef __cpp_lib_int_pow2
    const std::size_t num_cpus = static_cast<std::size_t>(std::bit_width(system_mask));
        #else
    std::size_t num_cpus = 0;
    if (system_mask != 0)
    {
        num_cpus = 1;
        while ((system_mask >>= 1U) != 0U)
            ++num_cpus;
    }
        #endif
    std::vector<bool> affinity(num_cpus);
    for (std::size_t i = 0; i < num_cpus; ++i)
        affinity[i] = ((process_mask & (1ULL << i)) != 0ULL);
    return affinity;
    #elif defined(__linux__)
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if (sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpu_set) != 0)
        return std::nullopt;
    const int num_cpus = get_nprocs();
    if (num_cpus < 1)
        return std::nullopt;
    std::vector<bool> affinity(static_cast<std::size_t>(num_cpus));
    for (std::size_t i = 0; i < affinity.size(); ++i)
        affinity[i] = CPU_ISSET(i, &cpu_set);
    return affinity;
    #elif defined(__APPLE__)
    return std::nullopt;
    #endif
}

/**
 * @brief Set the processor affinity of the current process using the current platform's native API. This should work on Windows and Linux, but is not possible on macOS as the native API does not allow it.
 *
 * @param affinity The processor affinity to set, as an `std::vector<bool>` where each element corresponds to a logical processor.
 * @return `true` if the affinity was set successfully, `false` otherwise. On macOS, this function always returns `false`.
 */
inline bool set_os_process_affinity([[maybe_unused]] const std::vector<bool>& affinity)
{
    #if defined(_WIN32)
    DWORD_PTR process_mask = 0;
    for (std::size_t i = 0; i < std::min<std::size_t>(affinity.size(), sizeof(DWORD_PTR) * 8); ++i)
        process_mask |= (affinity[i] ? (1ULL << i) : 0ULL);
    return SetProcessAffinityMask(GetCurrentProcess(), process_mask) != 0;
    #elif defined(__linux__)
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    for (std::size_t i = 0; i < std::min<std::size_t>(affinity.size(), CPU_SETSIZE); ++i)
    {
        if (affinity[i])
            CPU_SET(i, &cpu_set);
    }
    return sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpu_set) == 0;
    #elif defined(__APPLE__)
    return false;
    #endif
}

/**
 * @brief Get the priority of the current process using the current platform's native API. This should work on Windows, Linux, and macOS.
 *
 * @return An `std::optional` object, optionally containing the priority of the current process, as a member of the enum `BS::os_process_priority`. If the returned object does not contain a value, then either the priority could not be determined, or it is not one of the pre-defined values and therefore cannot be represented in a portable way.
 */
[[nodiscard]] inline std::optional<os_process_priority> get_os_process_priority()
{
    #if defined(_WIN32)
    // On Windows, this is straightforward.
    const DWORD priority = GetPriorityClass(GetCurrentProcess());
    if (priority == 0)
        return std::nullopt;
    return static_cast<os_process_priority>(priority);
    #elif defined(__linux__) || defined(__APPLE__)
    // On Linux/macOS there is no direct analogue of `GetPriorityClass()` on Windows, so instead we get the "nice" value. The usual range is -20 to 19 or 20, with higher values corresponding to lower priorities. However, we are only using 6 pre-defined values for portability, so if the value was set via any means other than `BS::set_os_process_priority()`, it may not match one of our pre-defined values. Note that `getpriority()` returns -1 on error, but since this does not correspond to any of our pre-defined values, this function will return `std::nullopt` anyway.
    const int nice_val = getpriority(PRIO_PROCESS, static_cast<id_t>(getpid()));
    switch (nice_val)
    {
    case static_cast<int>(os_process_priority::idle):
        return os_process_priority::idle;
    case static_cast<int>(os_process_priority::below_normal):
        return os_process_priority::below_normal;
    case static_cast<int>(os_process_priority::normal):
        return os_process_priority::normal;
    case static_cast<int>(os_process_priority::above_normal):
        return os_process_priority::above_normal;
    case static_cast<int>(os_process_priority::high):
        return os_process_priority::high;
    case static_cast<int>(os_process_priority::realtime):
        return os_process_priority::realtime;
    default:
        return std::nullopt;
    }
    #endif
}

/**
 * @brief Set the priority of the current process using the current platform's native API. This should work on Windows, Linux, and macOS. However, note that higher priorities might require elevated permissions.
 *
 * @param priority The priority to set. Must be a value from the enum `BS::os_process_priority`.
 * @return `true` if the priority was set successfully, `false` otherwise. Usually, `false` means that the user does not have the necessary permissions to set the desired priority.
 */
inline bool set_os_process_priority(const os_process_priority priority)
{
    #if defined(_WIN32)
    // On Windows, this is straightforward.
    return SetPriorityClass(GetCurrentProcess(), static_cast<DWORD>(priority)) != 0;
    #elif defined(__linux__) || defined(__APPLE__)
    // On Linux/macOS there is no direct analogue of `SetPriorityClass()` on Windows, so instead we set the "nice" value. The usual range is -20 to 19 or 20, with higher values corresponding to lower priorities. However, we are only using 6 pre-defined values for portability. Note that the "nice" values are only relevant for the `SCHED_OTHER` policy, but we do not set that policy here, as it is per-thread rather than per-process.
    // Also, it's important to note that a non-root user cannot decrease the nice value (i.e. increase the process priority), only increase it. This can cause confusing behavior. For example, if the current priority is `BS::os_process_priority::normal` and the user sets it to `BS::os_process_priority::idle`, they cannot change it back `BS::os_process_priority::normal`.
    return setpriority(PRIO_PROCESS, static_cast<id_t>(getpid()), static_cast<int>(priority)) == 0;
    #endif
}
#endif

/**
 * @brief A class used to obtain information about the current thread and, if native extensions are enabled, get/set its priority, affinity, or name.
 */
class [[nodiscard]] this_thread
{
    template <tp>
    friend class thread_pool;

public:
    /**
     * @brief Get the index of the current thread. If this thread belongs to a `BS::thread_pool` object, the return value will be an index in the range `[0, N)` where `N == BS::thread_pool::get_thread_count()`. Otherwise, for example if this thread is the main thread or an independent thread not in any pools, `std::nullopt` will be returned.
     *
     * @return An `std::optional` object, optionally containing a thread index.
     */
    [[nodiscard]] static std::optional<std::size_t> get_index() noexcept
    {
        return my_index;
    }

    /**
     * @brief Get a pointer to the thread pool that owns the current thread. If this thread belongs to a `BS::thread_pool` object, the return value will be a `void` pointer to that object. Otherwise, for example if this thread is the main thread or an independent thread not in any pools, `std::nullopt` will be returned.
     *
     * @return An `std::optional` object, optionally containing a pointer to a thread pool. Note that this will be a `void` pointer, so it must be cast to the desired instantiation of the `BS::thread_pool` template in order to use any member functions.
     */
    [[nodiscard]] static std::optional<void*> get_pool() noexcept
    {
        return my_pool;
    }

#ifdef BS_THREAD_POOL_NATIVE_EXTENSIONS
    /**
     * @brief Get the processor affinity of the current thread using the current platform's native API. This should work on Windows and Linux, but is not possible on macOS and Android as the native API does not allow it.
     *
     * @return An `std::optional` object, optionally containing the processor affinity of the current thread as an `std::vector<bool>` where each element corresponds to a logical processor. If the returned object does not contain a value, then the affinity could not be determined. On macOS and Android, this function always returns `std::nullopt`.
     */
    [[nodiscard]] static std::optional<std::vector<bool>> get_os_thread_affinity()
    {
    #if defined(_WIN32)
        // Windows does not have a `GetThreadAffinityMask()` function, but `SetThreadAffinityMask()` returns the previous affinity mask, so we can use that to get the current affinity and then restore it. It's a bit of a hack, but it works. Since the thread affinity must be a subset of the process affinity, we use the process affinity as the temporary value.
        DWORD_PTR process_mask = 0;
        DWORD_PTR system_mask = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask) == 0)
            return std::nullopt;
        const DWORD_PTR previous_mask = SetThreadAffinityMask(GetCurrentThread(), process_mask);
        if (previous_mask == 0)
            return std::nullopt;
        SetThreadAffinityMask(GetCurrentThread(), previous_mask);
        #ifdef __cpp_lib_int_pow2
        const std::size_t num_cpus = static_cast<std::size_t>(std::bit_width(system_mask));
        #else
        std::size_t num_cpus = 0;
        if (system_mask != 0)
        {
            num_cpus = 1;
            while ((system_mask >>= 1U) != 0U)
                ++num_cpus;
        }
        #endif
        std::vector<bool> affinity(num_cpus);
        for (std::size_t i = 0; i < num_cpus; ++i)
            affinity[i] = ((previous_mask & (1ULL << i)) != 0ULL);
        return affinity;
    #elif defined(__linux__) && !defined(__ANDROID__)
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) != 0)
            return std::nullopt;
        const int num_cpus = get_nprocs();
        if (num_cpus < 1)
            return std::nullopt;
        std::vector<bool> affinity(static_cast<std::size_t>(num_cpus));
        for (std::size_t i = 0; i < affinity.size(); ++i)
            affinity[i] = CPU_ISSET(i, &cpu_set);
        return affinity;
    #else
        return std::nullopt;
    #endif
    }

    /**
     * @brief Set the processor affinity of the current thread using the current platform's native API. This should work on Windows and Linux, but is not possible on macOS and Android as the native API does not allow it. Note that the thread affinity must be a subset of the process affinity (as obtained using `BS::get_os_process_affinity()`) for the containing process of a thread.
     *
     * @param affinity The processor affinity to set, as an `std::vector<bool>` where each element corresponds to a logical processor.
     * @return `true` if the affinity was set successfully, `false` otherwise. On macOS and Android, this function always returns `false`.
     */
    static bool set_os_thread_affinity([[maybe_unused]] const std::vector<bool>& affinity)
    {
    #if defined(_WIN32)
        DWORD_PTR thread_mask = 0;
        for (std::size_t i = 0; i < std::min<std::size_t>(affinity.size(), sizeof(DWORD_PTR) * 8); ++i)
            thread_mask |= (affinity[i] ? (1ULL << i) : 0ULL);
        return SetThreadAffinityMask(GetCurrentThread(), thread_mask) != 0;
    #elif defined(__linux__) && !defined(__ANDROID__)
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        for (std::size_t i = 0; i < std::min<std::size_t>(affinity.size(), CPU_SETSIZE); ++i)
        {
            if (affinity[i])
                CPU_SET(i, &cpu_set);
        }
        return pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) == 0;
    #else
        return false;
    #endif
    }

    /**
     * @brief Get the name of the current thread using the current platform's native API. This should work on Windows, Linux, and macOS.
     *
     * @return An `std::optional` object, optionally containing the name of the current thread. If the returned object does not contain a value, then the name could not be determined.
     */
    [[nodiscard]] static std::optional<std::string> get_os_thread_name()
    {
    #if defined(_WIN32)
        // On Windows thread names are wide strings, so we need to convert them to normal strings.
        PWSTR data = nullptr;
        const HRESULT hr = GetThreadDescription(GetCurrentThread(), &data);
        if (FAILED(hr))
            return std::nullopt;
        if (data == nullptr)
            return std::nullopt;
        const int size = WideCharToMultiByte(CP_UTF8, 0, data, -1, nullptr, 0, nullptr, nullptr);
        if (size == 0)
        {
            LocalFree(data);
            return std::nullopt;
        }
        std::string name(static_cast<std::size_t>(size) - 1, 0);
        const int result = WideCharToMultiByte(CP_UTF8, 0, data, -1, name.data(), size, nullptr, nullptr);
        LocalFree(data);
        if (result == 0)
            return std::nullopt;
        return name;
    #elif defined(__linux__) || defined(__APPLE__)
        #ifdef __linux__
        // On Linux thread names are limited to 16 characters, including the null terminator.
        constexpr std::size_t buffer_size = 16;
        #else
        // On macOS thread names are limited to 64 characters, including the null terminator.
        constexpr std::size_t buffer_size = 64;
        #endif
        char name[buffer_size] = {};
        if (pthread_getname_np(pthread_self(), name, buffer_size) != 0)
            return std::nullopt;
        return std::string(name);
    #endif
    }

    /**
     * @brief Set the name of the current thread using the current platform's native API. This should work on Windows, Linux, and macOS. Note that on Linux thread names are limited to 16 characters, including the null terminator.
     *
     * @param name The name to set.
     * @return `true` if the name was set successfully, `false` otherwise.
     */
    static bool set_os_thread_name(const std::string& name)
    {
    #if defined(_WIN32)
        // On Windows thread names are wide strings, so we need to convert them from normal strings.
        const int size = MultiByteToWideChar(CP_UTF8, 0, name.data(), -1, nullptr, 0);
        if (size == 0)
            return false;
        std::wstring wide(static_cast<std::size_t>(size), 0);
        if (MultiByteToWideChar(CP_UTF8, 0, name.data(), -1, wide.data(), size) == 0)
            return false;
        const HRESULT hr = SetThreadDescription(GetCurrentThread(), wide.data());
        return SUCCEEDED(hr);
    #elif defined(__linux__)
        // On Linux this is straightforward.
        return pthread_setname_np(pthread_self(), name.data()) == 0;
    #elif defined(__APPLE__)
        // On macOS, unlike Linux, a thread can only set a name for itself, so the signature is different.
        return pthread_setname_np(name.data()) == 0;
    #endif
    }

    /**
     * @brief Get the priority of the current thread using the current platform's native API. This should work on Windows, Linux, and macOS.
     *
     * @return An `std::optional` object, optionally containing the priority of the current thread, as a member of the enum `BS::os_thread_priority`. If the returned object does not contain a value, then either the priority could not be determined, or it is not one of the pre-defined values.
     */
    [[nodiscard]] static std::optional<os_thread_priority> get_os_thread_priority()
    {
    #if defined(_WIN32)
        // On Windows, this is straightforward.
        const int priority = GetThreadPriority(GetCurrentThread());
        if (priority == THREAD_PRIORITY_ERROR_RETURN)
            return std::nullopt;
        return static_cast<os_thread_priority>(priority);
    #elif defined(__linux__)
        // On Linux, we distill the choices of scheduling policy, priority, and "nice" value into 7 pre-defined levels, for simplicity and portability. The total number of possible combinations of policies and priorities is much larger, so if the value was set via any means other than `BS::this_thread::set_os_thread_priority()`, it may not match one of our pre-defined values.
        int policy = 0;
        struct sched_param param = {};
        if (pthread_getschedparam(pthread_self(), &policy, &param) != 0)
            return std::nullopt;
        if (policy == SCHED_FIFO && param.sched_priority == sched_get_priority_max(SCHED_FIFO))
        {
            // The only pre-defined priority that uses SCHED_FIFO and the maximum available priority value is the "realtime" priority.
            return os_thread_priority::realtime;
        }
        if (policy == SCHED_RR && param.sched_priority == sched_get_priority_min(SCHED_RR) + ((sched_get_priority_max(SCHED_RR) - sched_get_priority_min(SCHED_RR)) / 2))
        {
            // The only pre-defined priority that uses SCHED_RR and a priority in the middle of the available range is the "highest" priority.
            return os_thread_priority::highest;
        }
        #ifdef __linux__
        if (policy == SCHED_IDLE)
        {
            // The only pre-defined priority that uses SCHED_IDLE is the "idle" priority. Note that this scheduling policy is not available on macOS.
            return os_thread_priority::idle;
        }
        #endif
        if (policy == SCHED_OTHER)
        {
            // For SCHED_OTHER, the result depends on the "nice" value. The usual range is -20 to 19 or 20, with higher values corresponding to lower priorities. Note that `getpriority()` returns -1 on error, but since this does not correspond to any of our pre-defined values, this function will return `std::nullopt` anyway.
            const int nice_val = getpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)));
            switch (nice_val)
            {
            case PRIO_MIN + 2:
                return os_thread_priority::above_normal;
            case 0:
                return os_thread_priority::normal;
            case (PRIO_MAX / 2) + (PRIO_MAX % 2):
                return os_thread_priority::below_normal;
            case PRIO_MAX - 3:
                return os_thread_priority::lowest;
        #ifdef __APPLE__
            // `SCHED_IDLE` doesn't exist on macOS, so we use the policy `SCHED_OTHER` with a "nice" value of `PRIO_MAX - 2`.
            case PRIO_MAX - 2:
                return os_thread_priority::idle;
        #endif
            default:
                return std::nullopt;
            }
        }
        return std::nullopt;
    #elif defined(__APPLE__)
        // On macOS, we distill the choices of scheduling policy and priority into 7 pre-defined levels, for simplicity and portability. The total number of possible combinations of policies and priorities is much larger, so if the value was set via any means other than `BS::this_thread::set_os_thread_priority()`, it may not match one of our pre-defined values.
        int policy = 0;
        struct sched_param param = {};
        if (pthread_getschedparam(pthread_self(), &policy, &param) != 0)
            return std::nullopt;
        if (policy == SCHED_FIFO && param.sched_priority == sched_get_priority_max(SCHED_FIFO))
        {
            // The only pre-defined priority that uses SCHED_FIFO and the maximum available priority value is the "realtime" priority.
            return os_thread_priority::realtime;
        }
        if (policy == SCHED_RR && param.sched_priority == sched_get_priority_min(SCHED_RR) + (sched_get_priority_max(SCHED_RR) - sched_get_priority_min(SCHED_RR)) / 2)
        {
            // The only pre-defined priority that uses SCHED_RR and a priority in the middle of the available range is the "highest" priority.
            return os_thread_priority::highest;
        }
        if (policy == SCHED_OTHER)
        {
            // For SCHED_OTHER, the result depends on the specific value of the priority.
            if (param.sched_priority == sched_get_priority_max(SCHED_OTHER))
                return os_thread_priority::above_normal;
            if (param.sched_priority == sched_get_priority_min(SCHED_OTHER) + (sched_get_priority_max(SCHED_OTHER) - sched_get_priority_min(SCHED_OTHER)) / 2)
                return os_thread_priority::normal;
            if (param.sched_priority == sched_get_priority_min(SCHED_OTHER) + (sched_get_priority_max(SCHED_OTHER) - sched_get_priority_min(SCHED_OTHER)) * 2 / 3)
                return os_thread_priority::below_normal;
            if (param.sched_priority == sched_get_priority_min(SCHED_OTHER) + (sched_get_priority_max(SCHED_OTHER) - sched_get_priority_min(SCHED_OTHER)) / 3)
                return os_thread_priority::lowest;
            if (param.sched_priority == sched_get_priority_min(SCHED_OTHER))
                return os_thread_priority::idle;
            return std::nullopt;
        }
        return std::nullopt;
    #endif
    }

    /**
     * @brief Set the priority of the current thread using the current platform's native API. This should work on Windows, Linux, and macOS. However, note that higher priorities might require elevated permissions.
     *
     * @param priority The priority to set. Must be a value from the enum `BS::os_thread_priority`.
     * @return `true` if the priority was set successfully, `false` otherwise. Usually, `false` means that the user does not have the necessary permissions to set the desired priority.
     */
    static bool set_os_thread_priority(const os_thread_priority priority)
    {
    #if defined(_WIN32)
        // On Windows, this is straightforward.
        return SetThreadPriority(GetCurrentThread(), static_cast<int>(priority)) != 0;
    #elif defined(__linux__)
        // On Linux, we distill the choices of scheduling policy, priority, and "nice" value into 7 pre-defined levels, for simplicity and portability. The total number of possible combinations of policies and priorities is much larger, but allowing more fine-grained control would not be portable.
        int policy = 0;
        struct sched_param param = {};
        std::optional<int> nice_val = std::nullopt;
        switch (priority)
        {
        case os_thread_priority::realtime:
            // "Realtime" pre-defined priority: We use the policy `SCHED_FIFO` with the highest possible priority.
            policy = SCHED_FIFO;
            param.sched_priority = sched_get_priority_max(SCHED_FIFO);
            break;
        case os_thread_priority::highest:
            // "Highest" pre-defined priority: We use the policy `SCHED_RR` ("round-robin") with a priority in the middle of the available range.
            policy = SCHED_RR;
            param.sched_priority = sched_get_priority_min(SCHED_RR) + ((sched_get_priority_max(SCHED_RR) - sched_get_priority_min(SCHED_RR)) / 2);
            break;
        case os_thread_priority::above_normal:
            // "Above normal" pre-defined priority: We use the policy `SCHED_OTHER` (the default). This policy does not accept a priority value, so priority must be 0. However, we set the "nice" value to the minimum value as given by `PRIO_MIN`, plus 2 (which should evaluate to -18). The usual range is -20 to 19 or 20, with higher values corresponding to lower priorities.
            policy = SCHED_OTHER;
            param.sched_priority = 0;
            nice_val = PRIO_MIN + 2;
            break;
        case os_thread_priority::normal:
            // "Normal" pre-defined priority: We use the policy `SCHED_OTHER`, priority must be 0, and we set the "nice" value to 0 (the default).
            policy = SCHED_OTHER;
            param.sched_priority = 0;
            nice_val = 0;
            break;
        case os_thread_priority::below_normal:
            // "Below normal" pre-defined priority: We use the policy `SCHED_OTHER`, priority must be 0, and we set the "nice" value to half the maximum value as given by `PRIO_MAX`, rounded up (which should evaluate to 10).
            policy = SCHED_OTHER;
            param.sched_priority = 0;
            nice_val = (PRIO_MAX / 2) + (PRIO_MAX % 2);
            break;
        case os_thread_priority::lowest:
            // "Lowest" pre-defined priority: We use the policy `SCHED_OTHER`, priority must be 0, and we set the "nice" value to the maximum value as given by `PRIO_MAX`, minus 3 (which should evaluate to 17).
            policy = SCHED_OTHER;
            param.sched_priority = 0;
            nice_val = PRIO_MAX - 3;
            break;
        case os_thread_priority::idle:
            // "Idle" pre-defined priority on Linux: We use the policy `SCHED_IDLE`, priority must be 0, and we don't touch the "nice" value.
            policy = SCHED_IDLE;
            param.sched_priority = 0;
            break;
        default:
            return false;
        }
        bool success = (pthread_setschedparam(pthread_self(), policy, &param) == 0);
        if (nice_val.has_value())
            success = success && (setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), nice_val.value()) == 0);
        return success;
    #elif defined(__APPLE__)
        // On macOS, unlike Linux, the "nice" value is per-process, not per-thread (in compliance with the POSIX standard). However, unlike Linux, `SCHED_OTHER` on macOS does have a range of priorities. So for `realtime` and `highest` priorities we use `SCHED_FIFO` and `SCHED_RR` respectively as for Linux, but for the other priorities we use `SCHED_OTHER` with a priority in the range given by `sched_get_priority_min(SCHED_OTHER)` to `sched_get_priority_max(SCHED_OTHER)`.
        int policy = 0;
        struct sched_param param = {};
        switch (priority)
        {
        case os_thread_priority::realtime:
            // "Realtime" pre-defined priority: We use the policy `SCHED_FIFO` with the highest possible priority.
            policy = SCHED_FIFO;
            param.sched_priority = sched_get_priority_max(SCHED_FIFO);
            break;
        case os_thread_priority::highest:
            // "Highest" pre-defined priority: We use the policy `SCHED_RR` ("round-robin") with a priority in the middle of the available range.
            policy = SCHED_RR;
            param.sched_priority = sched_get_priority_min(SCHED_RR) + (sched_get_priority_max(SCHED_RR) - sched_get_priority_min(SCHED_RR)) / 2;
            break;
        case os_thread_priority::above_normal:
            // "Above normal" pre-defined priority: We use the policy `SCHED_OTHER` (the default) with the highest possible priority.
            policy = SCHED_OTHER;
            param.sched_priority = sched_get_priority_max(SCHED_OTHER);
            break;
        case os_thread_priority::normal:
            // "Normal" pre-defined priority: We use the policy `SCHED_OTHER` (the default) with a priority in the middle of the available range (which appears to be the default?).
            policy = SCHED_OTHER;
            param.sched_priority = sched_get_priority_min(SCHED_OTHER) + (sched_get_priority_max(SCHED_OTHER) - sched_get_priority_min(SCHED_OTHER)) / 2;
            break;
        case os_thread_priority::below_normal:
            // "Below normal" pre-defined priority: We use the policy `SCHED_OTHER` (the default) with a priority equal to 2/3rds of the normal value.
            policy = SCHED_OTHER;
            param.sched_priority = sched_get_priority_min(SCHED_OTHER) + (sched_get_priority_max(SCHED_OTHER) - sched_get_priority_min(SCHED_OTHER)) * 2 / 3;
            break;
        case os_thread_priority::lowest:
            // "Lowest" pre-defined priority: We use the policy `SCHED_OTHER` (the default) with a priority equal to 1/3rd of the normal value.
            policy = SCHED_OTHER;
            param.sched_priority = sched_get_priority_min(SCHED_OTHER) + (sched_get_priority_max(SCHED_OTHER) - sched_get_priority_min(SCHED_OTHER)) / 3;
            break;
        case os_thread_priority::idle:
            // "Idle" pre-defined priority on macOS: We use the policy `SCHED_OTHER` (the default) with the lowest possible priority.
            policy = SCHED_OTHER;
            param.sched_priority = sched_get_priority_min(SCHED_OTHER);
            break;
        default:
            return false;
        }
        return pthread_setschedparam(pthread_self(), policy, &param) == 0;
    #endif
    }
#endif

private:
    inline static thread_local std::optional<std::size_t> my_index = std::nullopt;
    inline static thread_local std::optional<void*> my_pool = std::nullopt;
}; // class this_thread

/**
 * @brief A meta-programming template to determine the common type of two integer types. Unlike `std::common_type`, this template maintains correct signedness.
 *
 * @tparam T1 The first type.
 * @tparam T2 The second type.
 * @tparam Enable A dummy parameter to enable SFINAE in specializations.
 */
template <typename T1, typename T2, typename Enable = void>
struct common_index_type
{
    // Fallback to `std::common_type_t` if no specialization matches.
    using type = std::common_type_t<T1, T2>;
};

// The common type of two signed integers is the larger of the integers, with the same signedness.
template <typename T1, typename T2>
struct common_index_type<T1, T2, std::enable_if_t<std::is_signed_v<T1> && std::is_signed_v<T2>>>
{
    using type = std::conditional_t<(sizeof(T1) >= sizeof(T2)), T1, T2>;
};

// The common type of two unsigned integers is the larger of the integers, with the same signedness.
template <typename T1, typename T2>
struct common_index_type<T1, T2, std::enable_if_t<std::is_unsigned_v<T1> && std::is_unsigned_v<T2>>>
{
    using type = std::conditional_t<(sizeof(T1) >= sizeof(T2)), T1, T2>;
};

// The common type of a signed and an unsigned integer is a signed integer that can hold the full ranges of both integers.
template <typename T1, typename T2>
struct common_index_type<T1, T2, std::enable_if_t<(std::is_signed_v<T1> && std::is_unsigned_v<T2>) || (std::is_unsigned_v<T1> && std::is_signed_v<T2>)>>
{
    using S = std::conditional_t<std::is_signed_v<T1>, T1, T2>;
    using U = std::conditional_t<std::is_unsigned_v<T1>, T1, T2>;
    static constexpr std::size_t larger_size = (sizeof(S) > sizeof(U)) ? sizeof(S) : sizeof(U);
    using type = std::conditional_t<larger_size <= 4,
        // If both integers are 32 bits or less, the common type should be a signed type that can hold both of them. If both are 8 bits, or the signed type is 16 bits and the unsigned type is 8 bits, the common type is `std::int16_t`. Otherwise, if both are 16 bits, or the signed type is 32 bits and the unsigned type is smaller, the common type is `std::int32_t`. Otherwise, if both are 32 bits or less, the common type is `std::int64_t`.
        std::conditional_t<larger_size == 1 || (sizeof(S) == 2 && sizeof(U) == 1), std::int16_t, std::conditional_t<larger_size == 2 || (sizeof(S) == 4 && sizeof(U) < 4), std::int32_t, std::int64_t>>,
        // If the unsigned integer is 64 bits, the common type should also be an unsigned 64-bit integer, that is, `std::uint64_t`. The reason is that the most common scenario where this might happen is where the indices go from 0 to `x` where `x` has been previously defined as `std::size_t`, e.g. the size of a vector. Note that this will fail if the first index is negative; in that case, the user must cast the indices explicitly to the desired common type. If the unsigned integer is not 64 bits, then the signed integer must be 64 bits, hence the common type is `std::int64_t`.
        std::conditional_t<sizeof(U) == 8, std::uint64_t, std::int64_t>>;
};

/**
 * @brief A helper type alias to obtain the common type from the template `BS::common_index_type`.
 *
 * @tparam T1 The first type.
 * @tparam T2 The second type.
 */
template <typename T1, typename T2>
using common_index_type_t = typename common_index_type<T1, T2>::type;

/**
 * @brief A fast, lightweight, modern, and easy-to-use C++17/C++20/C++23 thread pool class. This alias defines a thread pool with all optional features disabled.
 */
using light_thread_pool = thread_pool<tp::none>;

/**
 * @brief A fast, lightweight, modern, and easy-to-use C++17/C++20/C++23 thread pool class. This alias defines a thread pool with task priority enabled.
 */
using priority_thread_pool = thread_pool<tp::priority>;

/**
 * @brief A fast, lightweight, modern, and easy-to-use C++17/C++20/C++23 thread pool class. This alias defines a thread pool with pausing enabled.
 */
using pause_thread_pool = thread_pool<tp::pause>;

/**
 * @brief A fast, lightweight, modern, and easy-to-use C++17/C++20/C++23 thread pool class. This alias defines a thread pool with wait deadlock checks enabled.
 */
using wdc_thread_pool = thread_pool<tp::wait_deadlock_checks>;

/**
 * @brief A fast, lightweight, modern, and easy-to-use C++17/C++20/C++23 thread pool class.
 *
 * @tparam OptFlags A bitmask of flags which can be used to enable optional features. The flags are members of the `BS::tp` enumeration: `BS::tp::priority`, `BS::tp::pause`, and `BS::tp::wait_deadlock_checks`. The default is `BS::tp::none`, which disables all optional features. To enable multiple features, use the bitwise OR operator `|`, e.g. `BS::tp::priority | BS::tp::pause`.
 */
template <tp OptFlags = tp::none>
class [[nodiscard]] thread_pool
{
public:
    /**
     * @brief A flag indicating whether task priority is enabled.
     */
    static constexpr bool priority_enabled = (OptFlags & tp::priority) != tp::none;

    /**
     * @brief A flag indicating whether pausing is enabled.
     */
    static constexpr bool pause_enabled = (OptFlags & tp::pause) != tp::none;

    /**
     * @brief A flag indicating whether wait deadlock checks are enabled.
     */
    static constexpr bool wait_deadlock_checks_enabled = (OptFlags & tp::wait_deadlock_checks) != tp::none;

#ifndef __cpp_exceptions
    static_assert(!wait_deadlock_checks_enabled, "Wait deadlock checks cannot be enabled if exception handling is disabled.");
#endif

    // ============================
    // Constructors and destructors
    // ============================

    /**
     * @brief Construct a new thread pool. The number of threads will be the total number of hardware threads available, as reported by the implementation. This is usually determined by the number of cores in the CPU. If a core is hyperthreaded, it will count as two threads. If the native extensions are enabled, the pool will instead use the number of threads available to the process, as obtained from `BS::get_os_process_affinity()`, which can be less than the number of hardware threads.
     */
    thread_pool() : thread_pool(0, [] {}) {}

    /**
     * @brief Construct a new thread pool with the specified number of threads.
     *
     * @param num_threads The number of threads to use.
     */
    explicit thread_pool(const std::size_t num_threads) : thread_pool(num_threads, [] {}) {}

    /**
     * @brief Construct a new thread pool with the specified initialization function and the default number of threads.
     *
     * @param init An initialization function to run in each thread before it starts executing any submitted tasks. The function must have no return value, and can either take one argument, the thread index of type `std::size_t`, or zero arguments. It will be executed exactly once per thread, when the thread is first constructed. The initialization function must not throw any exceptions, as that will result in program termination. Any exceptions must be handled explicitly within the function.
     */
    template <BS_THREAD_POOL_INIT_FUNC_CONCEPT(F)>
    explicit thread_pool(F&& init) : thread_pool(0, std::forward<F>(init)) {}

    /**
     * @brief Construct a new thread pool with the specified number of threads and initialization function.
     *
     * @param num_threads The number of threads to use.
     * @param init An initialization function to run in each thread before it starts executing any submitted tasks. The function must have no return value, and can either take one argument, the thread index of type `std::size_t`, or zero arguments. It will be executed exactly once per thread, when the thread is first constructed. The initialization function must not throw any exceptions, as that will result in program termination. Any exceptions must be handled explicitly within the function.
     */
    template <BS_THREAD_POOL_INIT_FUNC_CONCEPT(F)>
    thread_pool(const std::size_t num_threads, F&& init)
    {
        create_threads(num_threads, std::forward<F>(init));
    }

    // The copy and move constructors and assignment operators are deleted. The thread pool cannot be copied or moved.
    thread_pool(const thread_pool&) = delete;
    thread_pool(thread_pool&&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool& operator=(thread_pool&&) = delete;

    /**
     * @brief Destruct the thread pool. Waits for all tasks to complete, then destroys all threads. If a cleanup function was set, it will run in each thread right before it is destroyed. Note that if the pool is paused, then any tasks still in the queue will never be executed.
     */
    ~thread_pool() noexcept
    {
#ifdef __cpp_exceptions
        try
        {
#endif
            wait();
#ifndef __cpp_lib_jthread
            destroy_threads();
#endif
#ifdef __cpp_exceptions
        }
        catch (...)
        {
        }
#endif
    }

    // =======================
    // Public member functions
    // =======================

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The block function takes two arguments, the start and end of the block, so that it is only called once per block, but it is up to the user to make sure the block function correctly deals with all the indices in each block. Does not return a `BS::multi_future`, so the user must use `wait()` or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam T1 The type of the first index. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index. Should be a signed or unsigned integer.
     * @tparam T The common type of the indices, as determined by `BS::common_index_type_t<T1, T2>`.
     * @tparam F The type of the block function.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted.
     * @param block A function that will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. `block(start, end)` should typically involve a loop of the form `for (T i = start; i < end; ++i)`. Must not return a value.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     */
    template <typename T1, typename T2, typename T = common_index_type_t<T1, T2>, typename F>
    void detach_blocks(const T1 first_index, const T2 index_after_last, F&& block, const std::size_t num_blocks = 0, const priority_t priority = 0)
    {
        enqueue_blocks<T, F, void, false>(static_cast<T>(first_index), static_cast<T>(index_after_last), std::forward<F>(block), num_blocks, priority);
    }

    /**
     * @brief Submit an iterator range containing functions with no arguments and no return values into the task queue, with the specified priority. To submit functions with arguments, enclose them in lambda expressions. Does not return a `BS::multi_future`, so the user must use `wait()` or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam I The type of the iterators.
     * @param first An iterator to the first function.
     * @param last An iterator to one past the last function.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     */
    template <typename I>
    void detach_bulk(const I first, const I last, const priority_t priority = 0)
    {
        if (first != last)
        {
            bool notify = false;
            {
                const std::scoped_lock tasks_lock(tasks_mutex);
                if constexpr (pause_enabled)
                    notify = tasks.empty() && !paused;
                else
                    notify = tasks.empty();
                for (I it = first; it != last; ++it)
                {
                    if constexpr (priority_enabled)
                        tasks.emplace(std::move(*it), priority);
                    else
                        tasks.emplace(std::move(*it));
                }
            }
            if (notify)
                task_available_cv.notify_all();
        }
    }

    /**
     * @brief Submit a container of functions with no arguments and no return values into the task queue, with the specified priority. To submit functions with arguments, enclose them in lambda expressions. Does not return a `BS::multi_future`, so the user must use `wait()` or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam C The type of the container. Must either be an array or have `begin()` and `end()` member functions.
     * @param container The container.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     */
    template <typename C>
    void detach_bulk(C& container, const priority_t priority = 0)
    {
        detach_bulk(std::begin(container), std::end(container), priority);
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The loop function takes one argument, the loop index, and it is called exactly once per index, but many times per block. Does not return a `BS::multi_future`, so the user must use `wait()` or some other method to ensure that the loop finishes executing, otherwise bad things will happen.
     *
     * @tparam T1 The type of the first index. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index. Should be a signed or unsigned integer.
     * @tparam T The common type of the indices, as determined by `BS::common_index_type_t<T1, T2>`.
     * @tparam F The type of the loop function.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted.
     * @param loop A function that will be called once per index, many times per block. Should take exactly one argument: the loop index. Must not return a value.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     */
    template <typename T1, typename T2, typename T = common_index_type_t<T1, T2>, typename F>
    void detach_loop(const T1 first_index, const T2 index_after_last, F&& loop, const std::size_t num_blocks = 0, const priority_t priority = 0)
    {
        enqueue_loop<T, F, false>(static_cast<T>(first_index), static_cast<T>(index_after_last), std::forward<F>(loop), num_blocks, priority);
    }

    /**
     * @brief Submit a sequence of tasks enumerated by indices to the queue, with the specified priority. The sequence function takes one argument, the task index, and will be called once per index. Does not return a `BS::multi_future`, so the user must use `wait()` or some other method to ensure that the sequence finishes executing, otherwise bad things will happen.
     *
     * @tparam T1 The type of the first index. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index. Should be a signed or unsigned integer.
     * @tparam T The common type of the indices, as determined by `BS::common_index_type_t<T1, T2>`.
     * @tparam F The type of the sequence function.
     * @param first_index The first index in the sequence.
     * @param index_after_last The index after the last index in the sequence. The sequence will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted.
     * @param sequence A function that will be called once per index. Should take exactly one argument, the index. Must not return a value.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     */
    template <typename T1, typename T2, typename T = common_index_type_t<T1, T2>, typename F>
    void detach_sequence(const T1 first_index, const T2 index_after_last, F&& sequence, const priority_t priority = 0)
    {
        return enqueue_sequence<T, F, void, false>(static_cast<T>(first_index), static_cast<T>(index_after_last), std::forward<F>(sequence), priority);
    }

    /**
     * @brief Submit a function with no arguments and no return value into the task queue, with the specified priority. To submit a function with arguments, enclose it in a lambda expression. Does not return a future, so the user must use `wait()` or some other method to ensure that the task finishes executing, otherwise bad things will happen.
     *
     * @tparam F The type of the function.
     * @param task The function to submit.
     * @param priority The priority of the task. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     */
    template <typename F>
    void detach_task(F&& task, const priority_t priority = 0)
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            if constexpr (priority_enabled)
                tasks.emplace(std::forward<F>(task), priority);
            else
                tasks.emplace(std::forward<F>(task));
        }
        task_available_cv.notify_one();
    }

#ifdef BS_THREAD_POOL_NATIVE_EXTENSIONS
    /**
     * @brief Get a vector containing the underlying implementation-defined thread handles for each of the pool's threads, as obtained by `std::thread::native_handle()` (or `std::jthread::native_handle()` in C++20 and later).
     *
     * @return The native thread handles.
     */
    [[nodiscard]] std::vector<thread_t::native_handle_type> get_native_handles() const
    {
        std::vector<thread_t::native_handle_type> native_handles(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i)
            native_handles[i] = threads[i].native_handle();
        return native_handles;
    }
#endif

    /**
     * @brief Get the number of tasks currently waiting in the queue to be executed by the threads.
     *
     * @return The number of queued tasks.
     */
    [[nodiscard]] std::size_t get_tasks_queued() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks.size();
    }

    /**
     * @brief Get the number of tasks currently being executed by the threads.
     *
     * @return The number of running tasks.
     */
    [[nodiscard]] std::size_t get_tasks_running() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks_running;
    }

    /**
     * @brief Get the total number of unfinished tasks: either still waiting in the queue, or running in a thread. Note that `get_tasks_total() == get_tasks_queued() + get_tasks_running()`.
     *
     * @return The total number of tasks.
     */
    [[nodiscard]] std::size_t get_tasks_total() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return tasks_running + tasks.size();
    }

    /**
     * @brief Get the number of threads in the pool.
     *
     * @return The number of threads.
     */
    [[nodiscard]] std::size_t get_thread_count() const noexcept
    {
        return thread_count;
    }

    /**
     * @brief Get a vector containing the unique identifiers for each of the pool's threads, as obtained by `std::thread::get_id()` (or `std::jthread::get_id()` in C++20 and later).
     *
     * @return The unique thread identifiers.
     */
    [[nodiscard]] std::vector<thread_t::id> get_thread_ids() const
    {
        std::vector<thread_t::id> thread_ids(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i)
            thread_ids[i] = threads[i].get_id();
        return thread_ids;
    }

    /**
     * @brief Check whether the pool is currently paused. Only enabled if the flag `BS::tp::pause` is enabled in the template parameter.
     *
     * @return `true` if the pool is paused, `false` if it is not paused.
     */
    BS_THREAD_POOL_IF_PAUSE_ENABLED
    [[nodiscard]] bool is_paused() const
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        return paused;
    }

    /**
     * @brief Pause the pool. The workers will temporarily stop retrieving new tasks out of the queue, although any tasks already executing will keep running until they are finished. Only enabled if the flag `BS::tp::pause` is enabled in the template parameter.
     */
    BS_THREAD_POOL_IF_PAUSE_ENABLED
    void pause()
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        paused = true;
    }

    /**
     * @brief Purge all the tasks waiting in the queue. Tasks that are currently running will not be affected, but any tasks still waiting in the queue will be discarded, and will never be executed by the threads. Please note that there is no way to restore the purged tasks.
     */
    void purge()
    {
        const std::scoped_lock tasks_lock(tasks_mutex);
        tasks = {};
    }

    /**
     * @brief Reset the pool with the default number of threads (as if constructed with the default constructor). Waits for all tasks to be completed, both running and queued, then destroys the thread pool and creates a new one with an empty task queue. If pausing is enabled, only waits for tasks that are currently running before destroying the pool; once the pool is reset, it will then resume executing the tasks that remained in the queue and any newly submitted tasks. If the pool was paused before resetting it, the new pool will be paused as well.
     */
    void reset()
    {
        reset(0, [](std::size_t) {});
    }

    /**
     * @brief Reset the pool with a new number of threads. Waits for all tasks to be completed, both running and queued, then destroys the thread pool and creates a new one with an empty task queue. If pausing is enabled, only waits for tasks that are currently running before destroying the pool; once the pool is reset, it will then resume executing the tasks that remained in the queue and any newly submitted tasks. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param num_threads The number of threads to use.
     */
    void reset(const std::size_t num_threads)
    {
        reset(num_threads, [](std::size_t) {});
    }

    /**
     * @brief Reset the pool with the default number of threads and a new initialization function. Waits for all tasks to be completed, both running and queued, then destroys the thread pool and creates a new one with an empty task queue. If pausing is enabled, only waits for tasks that are currently running before destroying the pool; once the pool is reset, it will then resume executing the tasks that remained in the queue and any newly submitted tasks. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param init An initialization function to run in each thread before it starts executing any submitted tasks. The function must have no return value, and can either take one argument, the thread index of type `std::size_t`, or zero arguments. It will be executed exactly once per thread, when the thread is first constructed. The initialization function must not throw any exceptions, as that will result in program termination. Any exceptions must be handled explicitly within the function.
     */
    template <BS_THREAD_POOL_INIT_FUNC_CONCEPT(F)>
    void reset(F&& init)
    {
        reset(0, std::forward<F>(init));
    }

    /**
     * @brief Reset the pool with a new number of threads and a new initialization function. Waits for all tasks to be completed, both running and queued, then destroys the thread pool and creates a new one with an empty task queue. If pausing is enabled, only waits for tasks that are currently running before destroying the pool; once the pool is reset, it will then resume executing the tasks that remained in the queue and any newly submitted tasks. If the pool was paused before resetting it, the new pool will be paused as well.
     *
     * @param num_threads The number of threads to use.
     * @param init An initialization function to run in each thread before it starts executing any submitted tasks. The function must have no return value, and can either take one argument, the thread index of type `std::size_t`, or zero arguments. It will be executed exactly once per thread, when the thread is first constructed. The initialization function must not throw any exceptions, as that will result in program termination. Any exceptions must be handled explicitly within the function.
     */
    template <BS_THREAD_POOL_INIT_FUNC_CONCEPT(F)>
    void reset(const std::size_t num_threads, F&& init)
    {
        if constexpr (pause_enabled)
        {
            std::unique_lock tasks_lock(tasks_mutex);
            const bool was_paused = paused;
            paused = true;
            tasks_lock.unlock();
            reset_pool(num_threads, std::forward<F>(init));
            tasks_lock.lock();
            paused = was_paused;
            tasks_lock.unlock();
            if (!was_paused)
                task_available_cv.notify_all();
        }
        else
        {
            reset_pool(num_threads, std::forward<F>(init));
        }
    }

    /**
     * @brief Set the thread pool's cleanup function.
     *
     * @param cleanup A cleanup function to run in each thread right before it is destroyed, which will happen when the pool is destructed or reset. The function must have no return value, and can either take one argument, the thread index of type `std::size_t`, or zero arguments. The cleanup function must not throw any exceptions, as that will result in program termination. Any exceptions must be handled explicitly within the function.
     */
    template <BS_THREAD_POOL_INIT_FUNC_CONCEPT(F)>
    void set_cleanup_func(F&& cleanup)
    {
        if constexpr (std::is_invocable_v<F, std::size_t>)
        {
            cleanup_func = std::forward<F>(cleanup);
        }
        else
        {
            cleanup_func = [cleanup = std::forward<F>(cleanup)](std::size_t)
            {
                cleanup();
            };
        }
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The block function takes two arguments, the start and end of the block, so that it is only called once per block, but it is up to the user to make sure the block function correctly deals with all the indices in each block. If the block function has a return value, get a `BS::multi_future` for the eventual returned values. If the block function has no return value, get a `BS::multi_future<void>` which can be used to wait until all the tasks finish.
     *
     * @tparam T1 The type of the first index. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index. Should be a signed or unsigned integer.
     * @tparam T The common type of the indices, as determined by `BS::common_index_type_t<T1, T2>`.
     * @tparam F The type of the block function.
     * @tparam R The return type of the block function (can be `void`).
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted, and an empty `BS::multi_future` will be returned.
     * @param block A function that will be called once per block. Should take exactly two arguments: the first index in the block and the index after the last index in the block. `block(start, end)` should typically involve a loop of the form `for (T i = start; i < end; ++i)`. Can return a value.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     * @return A `BS::multi_future` that can be used to wait for all the tasks to finish. If the block function returns a value, the `BS::multi_future` can also be used to obtain the values returned by each block.
     */
    template <typename T1, typename T2, typename T = common_index_type_t<T1, T2>, typename F, typename R = std::invoke_result_t<std::decay_t<F>, T, T>>
    [[nodiscard]] multi_future<R> submit_blocks(const T1 first_index, const T2 index_after_last, F&& block, const std::size_t num_blocks = 0, const priority_t priority = 0)
    {
        return enqueue_blocks<T, F, R, true>(static_cast<T>(first_index), static_cast<T>(index_after_last), std::forward<F>(block), num_blocks, priority);
    }

    /**
     * @brief Submit an iterator range containing functions with no arguments into the task queue, with the specified priority. To submit functions with arguments, enclose them in lambda expressions. If the functions have return values, get a `BS::multi_future` for the eventual returned values. If the functions have no return values, get a `BS::multi_future<void>` which can be used to wait until all the tasks finish.
     *
     * @tparam I The type of the iterators.
     * @tparam F The type of the functions.
     * @tparam R The return type of the functions (can be `void`, but must be the same for all the functions).
     * @param first An iterator to the first function.
     * @param last An iterator to one past the last function.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     * @return A `BS::multi_future` that can be used to wait for all the tasks to finish. If the functions return values, the `BS::multi_future` can also be used to obtain the values returned by each task.
     */
    template <typename I, typename F = decltype(*std::declval<I>()), typename R = std::invoke_result_t<std::decay_t<F>>>
    [[nodiscard]] multi_future<R> submit_bulk(const I first, const I last, const priority_t priority = 0)
    {
        if (first != last)
        {
            const std::size_t num_tasks = static_cast<std::size_t>(std::distance(first, last));
            multi_future<R> all_futures;
            all_futures.reserve(num_tasks);
            std::vector<task_t> all_tasks;
            all_tasks.reserve(num_tasks);
            for (I it = first; it != last; ++it)
            {
                task_and_future<R> ft(std::move(*it));
                all_futures.emplace_back(std::move(ft.future));
                all_tasks.emplace_back(std::move(ft.task));
            }
            detach_bulk(all_tasks, priority);
            return all_futures;
        }
        return {};
    }

    /**
     * @brief Submit a container of functions with no arguments into the task queue, with the specified priority. To submit functions with arguments, enclose them in lambda expressions. If the functions have return values, get a `BS::multi_future` for the eventual returned values. If the functions have no return values, get a `BS::multi_future<void>` which can be used to wait until all the tasks finish.
     *
     * @tparam C The type of the container. Must either be an array or have `begin()` and `end()` member functions.
     * @tparam F The type of the functions.
     * @tparam R The return type of the functions (can be `void`, but must be the same for all the functions).
     * @param container The container.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     * @return A `BS::multi_future` that can be used to wait for all the tasks to finish. If the functions return values, the `BS::multi_future` can also be used to obtain the values returned by each task.
     */
    template <typename C, typename F = decltype(*std::declval<C&>().begin()), typename R = std::invoke_result_t<std::decay_t<F>>>
    [[nodiscard]] multi_future<R> submit_bulk(C& container, const priority_t priority = 0)
    {
        return submit_bulk(std::begin(container), std::end(container), priority);
    }

    /**
     * @brief Parallelize a loop by automatically splitting it into blocks and submitting each block separately to the queue, with the specified priority. The loop function takes one argument, the loop index, and it is called exactly once per index, but many times per block. Returns a `BS::multi_future` which can be used to wait until all the tasks finish.
     *
     * @tparam T1 The type of the first index. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index. Should be a signed or unsigned integer.
     * @tparam T The common type of the indices, as determined by `BS::common_index_type_t<T1, T2>`.
     * @tparam F The type of the loop function.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop. The loop will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted, and an empty `BS::multi_future` will be returned.
     * @param loop A function that will be called once per index, many times per block. Should take exactly one argument: the loop index. Must not return a value.
     * @param num_blocks The maximum number of blocks to split the loop into. The default is 0, which means the number of blocks will be equal to the number of threads in the pool.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     * @return A `BS::multi_future` that can be used to wait for all the tasks to finish.
     */
    template <typename T1, typename T2, typename T = common_index_type_t<T1, T2>, typename F>
    [[nodiscard]] multi_future<void> submit_loop(const T1 first_index, const T2 index_after_last, F&& loop, const std::size_t num_blocks = 0, const priority_t priority = 0)
    {
        return enqueue_loop<T, F, true>(static_cast<T>(first_index), static_cast<T>(index_after_last), std::forward<F>(loop), num_blocks, priority);
    }

    /**
     * @brief Submit a sequence of tasks enumerated by indices to the queue, with the specified priority. The sequence function takes one argument, the task index, and will be called once per index. If the sequence function has a return value, get a `BS::multi_future` for the eventual returned values. If the sequence function has no return value, get a `BS::multi_future<void>` which can be used to wait until all the tasks finish.
     *
     * @tparam T1 The type of the first index. Should be a signed or unsigned integer.
     * @tparam T2 The type of the index after the last index. Should be a signed or unsigned integer.
     * @tparam T The common type of the indices, as determined by `BS::common_index_type_t<T1, T2>`.
     * @tparam F The type of the sequence function.
     * @tparam R The return type of the sequence function (can be `void`).
     * @param first_index The first index in the sequence.
     * @param index_after_last The index after the last index in the sequence. The sequence will iterate from `first_index` to `(index_after_last - 1)` inclusive. In other words, it will be equivalent to `for (T i = first_index; i < index_after_last; ++i)`. Note that if `index_after_last <= first_index`, no tasks will be submitted, and an empty `BS::multi_future` will be returned.
     * @param sequence A function that will be called once per index. Should take exactly one argument, the index. Can return a value.
     * @param priority The priority of the tasks. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     * @return A `BS::multi_future` that can be used to wait for all the tasks to finish. If the sequence function returns a value, the `BS::multi_future` can also be used to obtain the values returned by each task.
     */
    template <typename T1, typename T2, typename T = common_index_type_t<T1, T2>, typename F, typename R = std::invoke_result_t<std::decay_t<F>, T>>
    [[nodiscard]] multi_future<R> submit_sequence(const T1 first_index, const T2 index_after_last, F&& sequence, const priority_t priority = 0)
    {
        return enqueue_sequence<T, F, R, true>(static_cast<T>(first_index), static_cast<T>(index_after_last), std::forward<F>(sequence), priority);
    }

    /**
     * @brief Submit a function with no arguments into the task queue, with the specified priority. To submit a function with arguments, enclose it in a lambda expression. If the function has a return value, get a future for the eventual returned value. If the function has no return value, get an `std::future<void>` which can be used to wait until the task finishes.
     *
     * @tparam F The type of the function.
     * @tparam R The return type of the function (can be `void`).
     * @param task The function to submit.
     * @param priority The priority of the task. Should be between -128 and +127 (a signed 8-bit integer). The default is 0. Only taken into account if the flag `BS::tp::priority` is enabled in the template parameter, otherwise has no effect.
     * @return A future to be used later to wait for the function to finish executing and/or obtain its returned value if it has one.
     */
    template <typename F, typename R = std::invoke_result_t<std::decay_t<F>>>
    [[nodiscard]] std::future<R> submit_task(F&& task, const priority_t priority = 0)
    {
        task_and_future<R> ft(std::forward<F>(task));
        detach_task(std::move(ft.task), priority);
        return std::move(ft.future);
    }

    /**
     * @brief Unpause the pool. The workers will resume retrieving new tasks out of the queue. Only enabled if the flag `BS::tp::pause` is enabled in the template parameter.
     */
    BS_THREAD_POOL_IF_PAUSE_ENABLED
    void unpause()
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            paused = false;
        }
        task_available_cv.notify_all();
    }

    /**
     * @brief Wait for tasks to be completed. Normally, this function waits for all tasks, both those that are currently running in the threads and those that are still waiting in the queue. However, if the pool is paused, this function only waits for the currently running tasks (otherwise it would wait forever). Note: To wait for just one specific task, use `submit_task()` instead, and call the `wait()` member function of the generated future.
     *
     * @throws `wait_deadlock` if called from within a thread of the same pool, which would result in a deadlock. Only enabled if the flag `BS::tp::wait_deadlock_checks` is enabled in the template parameter.
     */
    void wait()
    {
#ifdef __cpp_exceptions
        if constexpr (wait_deadlock_checks_enabled)
        {
            if (this_thread::get_pool() == this)
                throw wait_deadlock();
        }
#endif
        std::unique_lock tasks_lock(tasks_mutex);
        waiting = true;
        tasks_done_cv.wait(tasks_lock,
            [this]
            {
                if constexpr (pause_enabled)
                    return (tasks_running == 0) && (paused || tasks.empty());
                else
                    return (tasks_running == 0) && tasks.empty();
            });
        waiting = false;
    }

    /**
     * @brief Wait for tasks to be completed, but stop waiting after the specified duration has passed.
     *
     * @tparam R An arithmetic type representing the number of ticks to wait.
     * @tparam P An `std::ratio` representing the length of each tick in seconds.
     * @param duration The amount of time to wait.
     * @return `true` if all tasks finished running, `false` if the duration expired but some tasks are still running.
     * @throws `wait_deadlock` if called from within a thread of the same pool, which would result in a deadlock. Only enabled if the flag `BS::tp::wait_deadlock_checks` is enabled in the template parameter.
     */
    template <typename R, typename P>
    bool wait_for(const std::chrono::duration<R, P>& duration)
    {
#ifdef __cpp_exceptions
        if constexpr (wait_deadlock_checks_enabled)
        {
            if (this_thread::get_pool() == this)
                throw wait_deadlock();
        }
#endif
        std::unique_lock tasks_lock(tasks_mutex);
        waiting = true;
        const bool status = tasks_done_cv.wait_for(tasks_lock, duration,
            [this]
            {
                if constexpr (pause_enabled)
                    return (tasks_running == 0) && (paused || tasks.empty());
                else
                    return (tasks_running == 0) && tasks.empty();
            });
        waiting = false;
        return status;
    }

    /**
     * @brief Wait for tasks to be completed, but stop waiting after the specified time point has been reached.
     *
     * @tparam C The type of the clock used to measure time.
     * @tparam D An `std::chrono::duration` type used to indicate the time point.
     * @param timeout_time The time point at which to stop waiting.
     * @return `true` if all tasks finished running, `false` if the time point was reached but some tasks are still running.
     * @throws `wait_deadlock` if called from within a thread of the same pool, which would result in a deadlock. Only enabled if the flag `BS::tp::wait_deadlock_checks` is enabled in the template parameter.
     */
    template <typename C, typename D>
    bool wait_until(const std::chrono::time_point<C, D>& timeout_time)
    {
#ifdef __cpp_exceptions
        if constexpr (wait_deadlock_checks_enabled)
        {
            if (this_thread::get_pool() == this)
                throw wait_deadlock();
        }
#endif
        std::unique_lock tasks_lock(tasks_mutex);
        waiting = true;
        const bool status = tasks_done_cv.wait_until(tasks_lock, timeout_time,
            [this]
            {
                if constexpr (pause_enabled)
                    return (tasks_running == 0) && (paused || tasks.empty());
                else
                    return (tasks_running == 0) && tasks.empty();
            });
        waiting = false;
        return status;
    }

private:
    // ========================
    // Private member functions
    // ========================

    /**
     * @brief Create the threads in the pool and assign a worker to each thread.
     *
     * @param num_threads The number of threads to use.
     * @param init An initialization function to run in each thread before it starts executing any submitted tasks.
     */
    template <typename F>
    void create_threads(const std::size_t num_threads, F&& init)
    {
        if constexpr (std::is_invocable_v<F, std::size_t>)
        {
            init_func = std::forward<F>(init);
        }
        else
        {
            init_func = [init = std::forward<F>(init)](std::size_t)
            {
                init();
            };
        }
        thread_count = determine_thread_count(num_threads);
        threads = std::make_unique<thread_t[]>(thread_count);
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            tasks_running = thread_count;
#ifndef __cpp_lib_jthread
            workers_running = true;
#endif
        }
        for (std::size_t i = 0; i < thread_count; ++i)
        {
            threads[i] = thread_t(
                [this, i]
#ifdef __cpp_lib_jthread
                (const std::stop_token& stop_token)
                {
                    worker(stop_token, i);
                }
#else
                {
                    worker(i);
                }
#endif
            );
        }
    }

#ifndef __cpp_lib_jthread
    /**
     * @brief Destroy the threads in the pool.
     */
    void destroy_threads()
    {
        {
            const std::scoped_lock tasks_lock(tasks_mutex);
            workers_running = false;
        }
        task_available_cv.notify_all();
        for (std::size_t i = 0; i < thread_count; ++i)
            threads[i].join();
    }
#endif

    /**
     * @brief Determine how many threads the pool should have, based on the parameter passed to the constructor or reset().
     *
     * @param num_threads The parameter passed to the constructor or `reset()`. If the parameter is a positive number, then the pool will be created with this number of threads. If the parameter is zero, or a parameter was not supplied (in which case it will have the default value of 0), then the pool will be created with the total number of hardware threads available, as obtained from `thread_t::hardware_concurrency()`. If the latter returns zero for some reason, then the pool will be created with just one thread. If the native extensions are enabled, the pool will instead use the number of threads available to the process, as obtained from `BS::get_os_process_affinity()`, which can be less than the number of hardware threads.
     */
    [[nodiscard]] static std::size_t determine_thread_count(const std::size_t num_threads) noexcept(!thread_pool_native_extensions)
    {
        if (num_threads > 0)
            return num_threads;
#ifdef BS_THREAD_POOL_NATIVE_EXTENSIONS
        const std::optional<std::vector<bool>> affinity = BS::get_os_process_affinity();
        if (affinity.has_value())
        {
            const std::size_t affinity_thread_count = static_cast<std::size_t>(std::count(affinity->begin(), affinity->end(), true));
            return (affinity_thread_count > 0) ? affinity_thread_count : 1;
        }
#endif
        if (thread_t::hardware_concurrency() > 0)
            return thread_t::hardware_concurrency();
        return 1;
    }

    /**
     * @brief A helper function for `detach_blocks()` and `submit_blocks()`.
     *
     * @tparam T The type of the indices.
     * @tparam F The type of the block function.
     * @tparam R The return type of the block function (can be `void`).
     * @tparam submit `true` if called from `submit_blocks()`, `false` if called from `detach_blocks()`.
     * @tparam N The return type of this helper function.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop.
     * @param block A function that will be called once per block.
     * @param num_blocks The maximum number of blocks to split the loop into.
     * @param priority The priority of the tasks.
     * @return A `BS::multi_future` if `submit` is `true`, or `void` if `submit` is `false`.
     */
    template <typename T, typename F, typename R, bool submit, typename N = std::conditional_t<submit, multi_future<R>, void>>
    [[nodiscard]] N enqueue_blocks(const T first_index, const T index_after_last, F&& block, std::size_t num_blocks, const priority_t priority = 0)
    {
        if (index_after_last > first_index)
        {
            using block_task_t = block_task<T, F, R>;
            const std::shared_ptr<std::decay_t<F>> block_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(block));
            const blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            num_blocks = blks.get_num_blocks();
            std::vector<std::conditional_t<submit, block_task_t, task_t>> all_tasks;
            all_tasks.reserve(num_blocks);
            for (std::size_t i = 0; i < num_blocks; ++i)
                all_tasks.emplace_back(block_task_t{block_ptr, blks.start(i), blks.end(i)});
            if constexpr (submit)
                return submit_bulk(all_tasks, priority);
            else
                detach_bulk(all_tasks, priority);
        }
        return N();
    }

    /**
     * @brief A helper function for `detach_loop()` and `submit_loop()`.
     *
     * @tparam T The type of the indices.
     * @tparam F The type of the loop function.
     * @tparam submit `true` if called from `submit_loop()`, `false` if called from `detach_loop()`.
     * @tparam N The return type of this helper function.
     * @param first_index The first index in the loop.
     * @param index_after_last The index after the last index in the loop.
     * @param loop A function that will be called once per index, many times per block.
     * @param num_blocks The maximum number of blocks to split the loop into.
     * @param priority The priority of the tasks.
     * @return A `BS::multi_future` if `submit` is `true`, or `void` if `submit` is `false`.
     */
    template <typename T, typename F, bool submit, typename N = std::conditional_t<submit, multi_future<void>, void>>
    [[nodiscard]] N enqueue_loop(const T first_index, const T index_after_last, F&& loop, std::size_t num_blocks, const priority_t priority = 0)
    {
        if (index_after_last > first_index)
        {
            using loop_task_t = loop_task<T, F>;
            const std::shared_ptr<std::decay_t<F>> loop_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(loop));
            const blocks blks(first_index, index_after_last, num_blocks ? num_blocks : thread_count);
            num_blocks = blks.get_num_blocks();
            std::vector<std::conditional_t<submit, loop_task_t, task_t>> all_tasks;
            all_tasks.reserve(num_blocks);
            for (std::size_t i = 0; i < num_blocks; ++i)
                all_tasks.emplace_back(loop_task_t{loop_ptr, blks.start(i), blks.end(i)});
            if constexpr (submit)
                return submit_bulk(all_tasks, priority);
            else
                detach_bulk(all_tasks, priority);
        }
        return N();
    }

    /**
     * @brief A helper function for `detach_sequence()` and `submit_sequence()`.
     *
     * @tparam T The type of the indices.
     * @tparam F The type of the sequence function.
     * @tparam R The return type of the sequence function (can be `void`).
     * @tparam submit `true` if called from `submit_sequence()`, `false` if called from `detach_sequence()`.
     * @tparam N The return type of this helper function.
     * @param first_index The first index in the sequence.
     * @param index_after_last The index after the last index in the sequence.
     * @param sequence A function that will be called once per index.
     * @param priority The priority of the tasks.
     * @return A `BS::multi_future` if `submit` is `true`, or `void` if `submit` is `false`.
     */
    template <typename T, typename F, typename R, bool submit, typename N = std::conditional_t<submit, multi_future<R>, void>>
    [[nodiscard]] N enqueue_sequence(const T first_index, const T index_after_last, F&& sequence, const priority_t priority = 0)
    {
        if (index_after_last > first_index)
        {
            using sequence_task_t = sequence_task<T, F, R>;
            const std::shared_ptr<std::decay_t<F>> sequence_ptr = std::make_shared<std::decay_t<F>>(std::forward<F>(sequence));
            std::vector<std::conditional_t<submit, sequence_task_t, task_t>> all_tasks;
            all_tasks.reserve(static_cast<std::size_t>(index_after_last - first_index));
            for (T i = first_index; i < index_after_last; ++i)
                all_tasks.emplace_back(sequence_task_t{sequence_ptr, i});
            if constexpr (submit)
                return submit_bulk(all_tasks, priority);
            else
                detach_bulk(all_tasks, priority);
        }
        return N();
    }

    /**
     * @brief Pop a task from the queue.
     *
     * @return The task.
     */
    [[nodiscard]] task_t pop_task()
    {
        task_t task;
        if constexpr (priority_enabled)
            task = std::move(tasks.top().task);
        else
            task = std::move(tasks.front());
        tasks.pop();
        return task;
    }

    /**
     * @brief Reset the pool with a new number of threads and a new initialization function. This member function implements the actual reset, while the public member function `reset()` also handles the case where the pool is paused.
     *
     * @param num_threads The number of threads to use.
     * @param init An initialization function to run in each thread before it starts executing any submitted tasks.
     */
    template <typename F>
    void reset_pool(const std::size_t num_threads, F&& init)
    {
        wait();
#ifndef __cpp_lib_jthread
        destroy_threads();
#endif
        create_threads(num_threads, std::forward<F>(init));
    }

    /**
     * @brief A worker function to be assigned to each thread in the pool. Waits until it is notified by `detach_task()` that a task is available, and then retrieves the task from the queue and executes it. Once the task finishes, the worker notifies `wait()` in case it is waiting.
     *
     * @param idx The index of this thread.
     */
    void worker(BS_THREAD_POOL_WORKER_TOKEN const std::size_t idx)
    {
        this_thread::my_pool = this;
        this_thread::my_index = idx;
        init_func(idx);
        while (true)
        {
            std::unique_lock tasks_lock(tasks_mutex);
            --tasks_running;
            if constexpr (pause_enabled)
            {
                if (waiting && (tasks_running == 0) && (paused || tasks.empty()))
                    tasks_done_cv.notify_all();
            }
            else
            {
                if (waiting && (tasks_running == 0) && tasks.empty())
                    tasks_done_cv.notify_all();
            }
            task_available_cv.wait(tasks_lock BS_THREAD_POOL_WAIT_TOKEN,
                [this]
                {
                    if constexpr (pause_enabled)
                        return !(paused || tasks.empty()) BS_THREAD_POOL_OR_STOP_CONDITION;
                    else
                        return !tasks.empty() BS_THREAD_POOL_OR_STOP_CONDITION;
                });
            if (BS_THREAD_POOL_STOP_CONDITION)
                break;
            {
                task_t task = pop_task(); // NOLINT(misc-const-correctness) In C++23 this cannot be const since `std::move_only_function::operator()` is not a const member function.
                ++tasks_running;
                tasks_lock.unlock();
#ifdef __cpp_exceptions
                try
                {
#endif
                    task();
#ifdef __cpp_exceptions
                }
                catch (...)
                {
                }
#endif
            }
        }
        cleanup_func(idx);
        this_thread::my_index = std::nullopt;
        this_thread::my_pool = std::nullopt;
    }

    // ============
    // Private data
    // ============

    /**
     * @brief A mutex to synchronize access to the task queue by different threads.
     */
    mutable std::mutex tasks_mutex;

/**
 * @brief A condition variable to notify `worker()` that a new task has become available.
 */
#ifdef __cpp_lib_jthread
    std::condition_variable_any
#else
    std::condition_variable
#endif
        task_available_cv;

    /**
     * @brief A condition variable to notify `wait()` that the tasks are done.
     */
    std::condition_variable tasks_done_cv;

    /**
     * @brief A cleanup function to run in each thread right before it is destroyed, which will happen when the pool is destructed or reset. The function must have no return value, and can either take one argument, the thread index of type `std::size_t`, or zero arguments. The cleanup function must not throw any exceptions, as that will result in program termination. Any exceptions must be handled explicitly within the function. The default is an empty function, i.e., no cleanup will be performed.
     */
    move_only_function<void(std::size_t)> cleanup_func = [](std::size_t) {};

    /**
     * @brief An initialization function to run in each thread before it starts executing any submitted tasks. The function must have no return value, and can either take one argument, the thread index of type `std::size_t`, or zero arguments. It will be executed exactly once per thread, when the thread is first constructed. The initialization function must not throw any exceptions, as that will result in program termination. Any exceptions must be handled explicitly within the function. The default is an empty function, i.e., no initialization will be performed.
     */
    move_only_function<void(std::size_t)> init_func = [](std::size_t) {};

    /**
     * @brief A queue of tasks to be executed by the threads.
     */
    std::conditional_t<priority_enabled, std::priority_queue<pr_task>, std::queue<task_t>> tasks;

    /**
     * @brief A counter for the total number of currently running tasks.
     */
    std::size_t tasks_running = 0;

    /**
     * @brief The number of threads in the pool.
     */
    std::size_t thread_count = 0;

    /**
     * @brief A smart pointer to manage the memory allocated for the threads.
     */
    std::unique_ptr<thread_t[]> threads = nullptr;

    /**
     * @brief A flag indicating whether the workers should pause. When set to `true`, the workers temporarily stop retrieving new tasks out of the queue, although any tasks already executing will keep running until they are finished. When set to `false` again, the workers resume retrieving tasks. Only enabled if the flag `BS::tp::pause` is enabled in the template parameter.
     */
    std::conditional_t<pause_enabled, bool, std::monostate> paused = {};

    /**
     * @brief A flag indicating that `wait()` is active and expects to be notified whenever a task is done.
     */
    bool waiting = false;

#ifndef __cpp_lib_jthread
    /**
     * @brief A flag indicating to the workers to keep running. When set to `false`, the workers terminate permanently.
     */
    bool workers_running = false;
#endif
}; // class thread_pool

/**
 * @brief A utility class to synchronize printing to one or more output streams by different threads.
 */
class [[nodiscard]] synced_stream
{
public:
    /**
     * @brief Construct a new synced stream which prints to `std::cout`.
     */
    explicit synced_stream()
    {
        add_stream(std::cout);
    }

    /**
     * @brief Construct a new synced stream which prints to the given output stream(s).
     *
     * @tparam T The types of the output streams to print to.
     * @param streams The output streams to print to.
     */
    template <typename... T>
    explicit synced_stream(T&... streams)
    {
        (add_stream(streams), ...);
    }

    /**
     * @brief Add a stream to the list of output streams to print to.
     *
     * @param stream The stream.
     */
    void add_stream(std::ostream& stream)
    {
        out_streams.push_back(&stream);
    }

    /**
     * @brief Get a reference to a vector containing pointers to the output streams to print to.
     *
     * @return The output streams.
     */
    std::vector<std::ostream*>& get_streams() noexcept
    {
        return out_streams;
    }

    /**
     * @brief Print any number of items into each output stream. Ensures that no other threads print to the streams simultaneously, as long as they all exclusively use the same `BS::synced_stream` object to print.
     *
     * @tparam T The types of the items.
     * @param items The items to print.
     */
    template <typename... T>
    void print(const T&... items)
    {
        const std::scoped_lock stream_lock(stream_mutex);
        for (std::ostream* const stream : out_streams)
            (*stream << ... << items);
    }

    /**
     * @brief Print any number of items into each output stream, followed by a newline character. Ensures that no other threads print to the streams simultaneously, as long as they all exclusively use the same `BS::synced_stream` object to print.
     *
     * @tparam T The types of the items.
     * @param items The items to print.
     */
    template <typename... T>
    void println(T&&... items)
    {
        print(std::forward<T>(items)..., '\n');
    }

    /**
     * @brief Remove a stream from the list of output streams to print to.
     *
     * @param stream The stream.
     */
    void remove_stream(std::ostream& stream)
    {
        out_streams.erase(std::remove(out_streams.begin(), out_streams.end(), &stream), out_streams.end());
    }

    /**
     * @brief A stream manipulator to pass to a `BS::synced_stream` (an explicit cast of `std::endl`). Prints a newline character to the stream, and then flushes it. Should only be used if flushing is desired, otherwise a newline character should be used instead.
     */
    inline static std::ostream& (&endl)(std::ostream&) = static_cast<std::ostream& (&)(std::ostream&)>(std::endl);

    /**
     * @brief A stream manipulator to pass to a `BS::synced_stream` (an explicit cast of `std::flush`). Used to flush the stream.
     */
    inline static std::ostream& (&flush)(std::ostream&) = static_cast<std::ostream& (&)(std::ostream&)>(std::flush);

private:
    /**
     * @brief A mutex to synchronize printing.
     */
    mutable std::mutex stream_mutex;

    /**
     * @brief The output streams to print to.
     */
    std::vector<std::ostream*> out_streams;
}; // class synced_stream
} // namespace BS
#endif // BS_THREAD_POOL_HPP