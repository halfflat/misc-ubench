#include <cassert>
#include <cstdint>
#include <random>
#include <vector>

#include "benchmark/benchmark.h"

using u32 = std::uint32_t;

// 32-bit unsigned square root implementations.

std::vector<u32> generate_test_set(unsigned count, bool uniform) {
    static std::minstd_rand R;
    std::uniform_int_distribution<u32> U1(0, (u32)-1);
    std::uniform_int_distribution<u32> U2(0, 31);

    std::vector<u32> g(count);
    if (uniform) std::generate(g.begin(), g.end(), [&] { return U1(R); });
    else std::generate(g.begin(), g.end(), [&] { return U1(R)>>U2(R); });
    return g;
}

u32 isqrt32_bsearch_iter16(u32 n) {
    u32 b = 1<<15;
    u32 r = 0;

    for (unsigned i = 0; i<16; ++i) {
        u32 t = r+b;
        if (t*t<=n) r = t;
        b >>= 1;
    }
    return r;
}

u32 isqrt32_bsearch(u32 n) {
    u32 i = 16;
    for (u32 k = 1u<<30; k>n; k>>=2) --i;

    u32 b = 1<<i;
    u32 r = 0;

    while (i-->0) {
        b >>= 1;
        u32 t = r+b;
        if (t*t<=n) r = t;
    }
    return r;
}

u32 isqrt32_digit_iter16(u32 n) {
    u32 b = 1u<<30;
    u32 r = 0;

    for (unsigned i = 16; i-->0;) {
        u32 t = r+b;
        u32 mask = -(t<=n);
        n -= t&mask;
        r = (r>>1)+(b&mask);
        b >>= 2;
    }
    return r;
}

u32 isqrt32_digit(u32 n) {
    u32 b = 1u<<30;
    u32 r = 0;

    while (b>n) b >>= 2;

    while (b) {
        u32 t = r+b;
        u32 mask = -(t<=n);
        n -= t&mask;
        r = (r>>1)+(b&mask);
        b >>= 2;
    }
    return r;
}

u32 isqrt32_reference(u32 n) {
    // Wikipedia implementation
    u32 b = 1u<<30;
    u32 r = 0;

    while (b>n) b >>= 2;

    while (b) {
        if (n>=r+b) {
            n -= r+b;
            r = (r>>1)+b;
        } else r >>= 1;
        b >>= 2;
    }
    return r;
}

template <u32 (*impl)(u32)>
void bench_isqrt(benchmark::State& state) {
    unsigned n = state.range(0);
    bool uniform = state.range(1);
    auto test_set = generate_test_set(n, uniform);

    for (u32 n: test_set) {
        auto r = impl(n);
        assert(r*r<=n);
        if (r+1<(1u<<16)) assert((r+1)*(r+1)>n);
    }

    for (auto _: state) {
        for (auto n: test_set) benchmark::DoNotOptimize(impl(n));
    }
}

#ifndef N
#define N 10000
#endif

BENCHMARK_TEMPLATE(bench_isqrt, isqrt32_reference)->ArgsProduct({{N}, {0, 1}});
BENCHMARK_TEMPLATE(bench_isqrt, isqrt32_bsearch_iter16)->ArgsProduct({{N}, {0, 1}});
BENCHMARK_TEMPLATE(bench_isqrt, isqrt32_bsearch)->ArgsProduct({{N}, {0, 1}});
BENCHMARK_TEMPLATE(bench_isqrt, isqrt32_digit_iter16)->ArgsProduct({{N}, {0, 1}});
BENCHMARK_TEMPLATE(bench_isqrt, isqrt32_digit)->ArgsProduct({{N}, {0, 1}});
BENCHMARK_MAIN();
