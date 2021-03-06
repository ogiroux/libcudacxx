/*

Copyright (c) 2019, NVIDIA Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <mutex>
#include <thread>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <tuple>
#include <set>
#include <chrono>
#include <algorithm>

#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/climits>
#include <cuda/std/ratio>
#include <cuda/std/chrono>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/std/atomic>
#include <cuda/std/barrier>
#include <cuda/std/latch>
#include <cuda/std/semaphore>

#ifdef __CUDACC__
#  include <cuda_awbarrier.h>
#endif

#ifdef __CUDACC__
# define _ABI __host__ __device__
# define check(ans) { assert_((ans), __FILE__, __LINE__); }
inline void assert_(cudaError_t code, const char *file, int line) {
  if (code == cudaSuccess)
    return;
  std::cerr << "check failed: " << cudaGetErrorString(code) << ": " << file << ':' << line << std::endl;
  abort();
}
#else
# define _ABI
#endif

template <class T>
struct managed_allocator {
  typedef cuda::std::size_t size_type;
  typedef cuda::std::ptrdiff_t difference_type;

  typedef T value_type;
  typedef T* pointer;// (deprecated in C++17)(removed in C++20)    T*
  typedef const T* const_pointer;// (deprecated in C++17)(removed in C++20)    const T*
  typedef T& reference;// (deprecated in C++17)(removed in C++20)    T&
  typedef const T& const_reference;// (deprecated in C++17)(removed in C++20)    const T&

  template< class U > struct rebind { typedef managed_allocator<U> other; };
  managed_allocator() = default;
  template <class U> constexpr managed_allocator(const managed_allocator<U>&) noexcept {}
  T* allocate(std::size_t n) {
    void* out = nullptr;
#ifdef __CUDACC__
# ifdef __aarch64__
    check(cudaMallocHost(&out, n*sizeof(T), cudaHostAllocMapped));
    void* out2;
    check(cudaHostGetDevicePointer(&out2, out, 0));
    assert(out2==out); //< we can't handle non-uniform addressing
# else
    check(cudaMallocManaged(&out, n*sizeof(T)));
# endif
#else
    out = malloc(n*sizeof(T));
#endif
    return static_cast<T*>(out);
  }
  void deallocate(T* p, std::size_t) noexcept {
#ifdef __CUDACC__
# ifdef __aarch64__
    check(cudaFreeHost(p));
# else
    check(cudaFree(p));
# endif
#else
    free(p);
#endif
  }
};
template<class T, class... Args>
T* make_(Args &&... args) {
    managed_allocator<T> ma;
    auto n_ = new (ma.allocate(1)) T(std::forward<Args>(args)...);
#if defined(__CUDACC__) && !defined(__aarch64__) && !defined(_MSC_VER)
    check(cudaMemAdvise(n_, sizeof(T), cudaMemAdviseSetPreferredLocation, 0));
    check(cudaMemPrefetchAsync(n_, sizeof(T), 0));
#endif
    return n_;
}
template<class T>
void unmake_(T* ptr) {
    managed_allocator<T> ma;
    ptr->~T();
    ma.deallocate(ptr, sizeof(T));
}

struct null_mutex {
    _ABI void lock() noexcept { }
    _ABI void unlock() noexcept { }
};

struct mutex {
    _ABI void lock() noexcept {
        while (1 == l.exchange(1, cuda::std::memory_order_acquire))
#ifndef __NO_WAIT
            l.wait(1, cuda::std::memory_order_relaxed)
#endif
            ;
    }
    _ABI void unlock() noexcept {
        l.store(0, cuda::std::memory_order_release);
#ifndef __NO_WAIT
        l.notify_one();
#endif
    }
    alignas(64) cuda::atomic<int, cuda::thread_scope_device> l = ATOMIC_VAR_INIT(0);
};

struct ticket_mutex {
    _ABI void lock() noexcept {
        auto const my = in.fetch_add(1, cuda::std::memory_order_acquire);
        while(1) {
            auto const now = out.load(cuda::std::memory_order_acquire);
            if(now == my)
                return;
#ifndef __NO_WAIT
            out.wait(now, cuda::std::memory_order_relaxed);
#endif
        }
    }
    _ABI void unlock() noexcept {
        out.fetch_add(1, cuda::std::memory_order_release);
#ifndef __NO_WAIT
        out.notify_all();
#endif
    }
    alignas(64) cuda::atomic<int, cuda::thread_scope_device> in = ATOMIC_VAR_INIT(0);
    alignas(64) cuda::atomic<int, cuda::thread_scope_device> out = ATOMIC_VAR_INIT(0);
};

struct sem_mutex {
    _ABI void lock() noexcept {
        c.acquire();
    }
    _ABI void unlock() noexcept {
        c.release();
    }
    _ABI sem_mutex() : c(1) { }
    cuda::binary_semaphore<cuda::thread_scope_device> c;
};

using sum_mean_dev_t = std::tuple<double, double, double>;

template<class V>
sum_mean_dev_t sum_mean_dev(V && v) {
    assert(!v.empty());
    auto const sum = std::accumulate(v.begin(), v.end(), 0.0);
    assert(sum >= 0.0);
    auto const mean = sum / v.size();
    auto const sq_diff_sum = std::accumulate(v.begin(), v.end(), 0.0, [=](double left, double right) -> double {
        auto const delta = right - mean;
        return left + delta * delta;
    });
    auto const variance = sq_diff_sum / v.size();
    assert(variance >= 0.0);
    auto const stddev = std::sqrt(variance);
    return sum_mean_dev_t(sum, mean, stddev);
}

int get_max_threads(cuda::thread_scope scope) {

#ifndef __CUDACC__
    return std::thread::hardware_concurrency();
#else
    cudaDeviceProp deviceProp;
    check(cudaGetDeviceProperties(&deviceProp, 0));
    assert(deviceProp.major >= 7);
    return scope == cuda::thread_scope_block
        ? deviceProp.maxThreadsPerBlock
        : deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
#endif
}

#ifdef __CUDACC__

template<class F>
__launch_bounds__(1024, 2)
__global__ void launcher(F f, int t, int* p) {
    auto const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < t)
        p[tid] = (*f)(tid);
}
#endif

template <class F>
sum_mean_dev_t test_body(unsigned threads, F f, cuda::thread_scope scope) {

    std::vector<int, managed_allocator<int>> progress(threads, 0);

#ifdef __CUDACC__
    auto p_ = progress.data();
# if !defined(__aarch64__) && !defined(_MSC_VER)
    check(cudaMemAdvise(p_, threads * sizeof(int), cudaMemAdviseSetPreferredLocation, 0));
    check(cudaMemPrefetchAsync(p_, threads * sizeof(int), 0));
# endif
    auto f_ = make_<F>(f);
    cudaDeviceSynchronize();
    unsigned const max_blocks = scope == cuda::thread_scope_block ? 1 : get_max_threads(scope) / 512;
    unsigned const blocks = (std::min)(threads, max_blocks);
    unsigned const threads_per_block = (threads / blocks) + (threads % blocks ? 1 : 0);
    dim3 grid = {blocks, 1, 1};
    dim3 block = {threads_per_block, 1, 1};
    void* args[] = {&f_, &threads, &p_};
    cudaLaunchCooperativeKernel(&launcher<decltype(f_)>, grid, block, args, 0, 0);
    check(cudaDeviceSynchronize());
    check(cudaGetLastError());
    unmake_(f_);
#else
    std::vector<std::thread> ts(threads);
    for (int i = 0; i < threads; ++i)
        ts[i] = std::thread([&, i]() {
            progress[i] = f(i);
        });
    for (auto& t : ts)
        t.join();
#endif

    return sum_mean_dev(progress);
}

template <class F>
sum_mean_dev_t test_omp_body(int threads, F && f) {
#ifdef _OPENMP
    std::vector<int> progress(threads, 0);
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < threads; ++i)
        progress[i] = f(i);
    return sum_mean_dev(progress);
#else
    assert(0); // build with -fopenmp
    return sum_mean_dev_t();
#endif
}

template <class F>
void test(std::string const& name, unsigned threads, F && f, bool use_omp, bool rate_per_thread, cuda::thread_scope scope) {

    std::cout << name << ": " << std::flush;

    auto const t1 = std::chrono::steady_clock::now();
    auto const smd = use_omp ? test_omp_body(threads, f)
                             : test_body(threads, f, scope);
    auto const t2 = std::chrono::steady_clock::now();

    auto r = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / std::get<0>(smd);
    if(rate_per_thread)
        r *= threads;
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << r << "ns per step, fairness metric = "
              << 100 * (1.0 - std::min(1.0, std::get<2>(smd) / std::get<1>(smd))) << "%."
              << std::endl << std::flush;
//    std::cout << "smd = " << std::get<0>(smd) << " " << std::get<1>(smd) << " " << std::get<2>(smd) << std::endl;
}

template<class F>
void test_loop(cuda::thread_scope scope, F && f) {
    std::cout << "============================" << std::endl;
    static int const max = get_max_threads(scope);
    static std::vector<std::pair<int, std::string>> const counts =
        { { 1, "single-threaded" },
          { 2, "2 threads" },
//          { 3, "3 threads" },
          { 4, "4 threads" },
//          { 5, "5 threads" },
//          { 8, "8 threads" },
          { 16, "16 threads" },
//          { 32, "32 threads" },
//          { 64, "64 threads" },
          { 256, "256 threads" },
//          { 512, "512 threads" },
//          { 4096, "4096 threads" },
//          { max, "maximum occupancy" },
//#if !defined(__NO_SPIN) || !defined(__NO_WAIT)
//          { max * 2, "200% occupancy" }
//#endif
        };
    std::set<int> done{0};
    for(auto const& c : counts) {
        if(done.find(c.first) != done.end())
            continue;
        if(c.first <= max)
            f(c);
        done.insert(c.first);
    }
}

template<class M>
void test_mutex_contended(std::string const& name, bool use_omp = false) {
    test_loop(cuda::thread_scope_system, [&](std::pair<int, std::string> c) {
        M* m = make_<M>();
        auto f = [=] _ABI (int) -> int {
            int i = 0;
            auto const end = cuda::std::chrono::high_resolution_clock::now() + cuda::std::chrono::seconds(1);
            while(end > cuda::std::chrono::high_resolution_clock::now()) {
                m->lock();
                ++i;
                m->unlock();
            }
            return i;
        };
        test(name + ", " + c.second, c.first, f, use_omp, false, cuda::thread_scope_system);
        unmake_(m);
    });
};

template<class M>
void test_mutex_uncontended(std::string const& name, bool use_omp = false) {
    test_loop(cuda::thread_scope_system, [&](std::pair<int, std::string> c) {
        std::vector<M, managed_allocator<M>> ms(c.first);
        M* ms_ = &ms[0];
        auto f = [=] _ABI (int id) -> int {
            int i = 0;
            auto const end = cuda::std::chrono::high_resolution_clock::now() + cuda::std::chrono::seconds(1);
            while(end > cuda::std::chrono::high_resolution_clock::now()) {
                ms_[id].lock();
                ++i;
                ms_[id].unlock();
            }
            return i;
        };
        test(name + ": " + c.second, c.first, f, use_omp, true, cuda::thread_scope_system);
    });
};

template<class M>
void test_mutex(std::string const& name, bool use_omp = false) {
    test_mutex_uncontended<M>(name + " uncontended", use_omp);
    test_mutex_contended<M>(name + " contended", use_omp);
}

template<typename Barrier>
struct scope_of_barrier
{
    static const constexpr auto scope = cuda::thread_scope_system;
};

#ifdef __CUDACC__
template<>
struct scope_of_barrier<nvcuda::experimental::awbarrier>
{
    static const constexpr auto scope = cuda::thread_scope_block;
};
#endif

template<cuda::thread_scope Scope, typename F>
struct scope_of_barrier<cuda::barrier<Scope, F>>
{
    static const constexpr auto scope = Scope;
};

#ifdef __CUDACC__
template<class B>
void test_barrier_shared(std::string const& name) {

    constexpr auto scope = scope_of_barrier<B>::scope;
    assert(scope == cuda::thread_scope_block);
    (void)scope;

    test_loop(cuda::thread_scope_block, [&](std::pair<int, std::string> c) {
        int count = c.first;
        auto f = [=] __device__ (int)  -> int {
            __shared__ B b;
            if (threadIdx.x == 0) {
                init(&b, count);
            }
            __syncthreads();
            int i = 1;
            auto const end = cuda::std::chrono::high_resolution_clock::now() + cuda::std::chrono::seconds(1);
            while(end > cuda::std::chrono::high_resolution_clock::now()) {
                b.arrive_and_wait();
                ++i;
            }
            b.arrive_and_drop();
            return i;
        };
        test(name + ": " + c.second, c.first, f, false, true, cuda::thread_scope_block);
    });
};
#endif

template<class B>
void test_barrier(std::string const& name, bool use_omp = false) {

    constexpr auto scope = scope_of_barrier<B>::scope;
    (void)scope;

#ifdef __CUDACC__
    if (scope == cuda::thread_scope_block) {
        test_barrier_shared<B>(name + " __shared__");
    }
#endif

    test_loop(scope_of_barrier<B>::scope, [&](std::pair<int, std::string> c) {
        B* b = make_<B>(c.first);
        auto f = [=] _ABI (int)  -> int {
            int i = 0;
            auto const end = cuda::std::chrono::high_resolution_clock::now() + cuda::std::chrono::seconds(1);
            while(end > cuda::std::chrono::high_resolution_clock::now()) {
                b->arrive_and_wait();
//                (void)b->arrive();
  //              wait_for_parity(b, i & 1);
                ++i;
            }
            b->arrive_and_drop();
            ++i;
            return i;
        };
        test(name + ": " + c.second, c.first, f, use_omp, true, scope_of_barrier<B>::scope);
        unmake_(b);
    });
};

template<typename Latch>
struct scope_of_latch
{
    static const constexpr auto scope = cuda::thread_scope_system;
};

template<cuda::thread_scope Scope>
struct scope_of_latch<cuda::latch<Scope>>
{
    static const constexpr auto scope = Scope;
};

template<class L>
void test_latch(std::string const& name, bool use_omp = false) {

    constexpr auto scope = scope_of_latch<L>::scope;
    (void)scope;

    test_loop(scope, [&](std::pair<int, std::string> c) {

        managed_allocator<L> ma;

        size_t const n = 1024;
        auto* ls = ma.allocate(n);
        for(size_t i = 0; i < n; ++i)
            new (ls + i) L(c.first);

        auto f = [=] _ABI (int)  -> int {
            for (int i = 0; i < n; ++i)
                ls[i].arrive_and_wait();
            return n;
        };
        test(name + ": " + c.second, c.first, f, use_omp, true, scope);

        ma.deallocate(ls, n);
    });
};

#include <Windows.h>

int main() {

//    DebugBreak();

    int const max = get_max_threads(cuda::thread_scope_system);
    std::cout << "Will use up to " << max << " hardware threads." << std::endl;
#ifdef __CUDACC__
    int const block_max = get_max_threads(cuda::thread_scope_block);
    std::cout << "Will use up to " << block_max << " hardware threads in a single block." << std::endl;
#endif

#ifndef __NO_BARRIER
#ifdef __CUDACC__
    test_barrier<cuda::barrier<cuda::thread_scope_block>>("cuda::barrier<block>");
    test_barrier<cuda::barrier<cuda::thread_scope_device>>("cuda::barrier<device>");
#endif
    test_barrier<cuda::barrier<cuda::thread_scope_system>>("cuda::barrier<system>");
#ifdef __CUDACC__
    test_barrier_shared<nvcuda::experimental::awbarrier>("nvcuda::exp::awbarrier __shared__");
#endif
/*
#ifdef __CUDACC__
    test_latch<cuda::latch<cuda::thread_scope_block>>("cuda::latch<block>");
    test_latch<cuda::latch<cuda::thread_scope_device>>("cuda::latch<device>");
#endif
    test_latch<cuda::std::latch>("cuda::latch<system>");
*/
#endif
#ifndef __NO_MUTEX
    test_mutex<mutex>("spinlock_mutex");
    test_mutex<sem_mutex>("sem_mutex");
//  test_mutex<null_mutex>("Null");
    test_mutex<ticket_mutex>("ticket_mutex");
#ifndef __CUDACC__
    test_mutex<std::mutex>("std::mutex");
#endif
#endif

#ifdef _OPENMP
    struct omp_barrier {
        omp_barrier(ptrdiff_t) { }
        void arrive_and_wait() {
            #pragma omp barrier
        }
    };
    test_barrier<omp_barrier>("omp_barrier", true);
#endif

#if !defined(__CUDACC__) && defined(_POSIX_THREADS) && !defined(__APPLE__)
    struct posix_barrier {
        posix_barrier(ptrdiff_t count) {
            pthread_barrier_init(&pb, nullptr, count);
        }
        ~posix_barrier() {
            pthread_barrier_destroy(&pb);
        }
        void arrive_and_wait() {
            pthread_barrier_wait(&pb);
        }
        pthread_barrier_t pb;
    };
    test_barrier<posix_barrier>("pthread_barrier");
#endif

    return 0;
}
