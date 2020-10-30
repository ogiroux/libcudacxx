#include <cuda/std/barrier>

using barrier = cuda::std::barrier<>;
__managed__ cuda::std::aligned_storage<sizeof(barrier), alignof(barrier)>::type b_;

__global__ void test()
{
    auto& b = reinterpret_cast<barrier&>(b_);
    for(int i = 0;i < 1024; ++i)
        b.arrive_and_wait();
}

int main()
{
    new (&b_) cuda::std::barrier<>(256);

    test<<<32, 8>>>();
    cudaDeviceSynchronize();

    return 0;
}
