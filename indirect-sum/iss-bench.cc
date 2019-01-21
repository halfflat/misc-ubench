#include <cassert>
#include <functional>
#include <random>
#include <memory>
#include <system_error>
#include <vector>

#include <iostream>

#include <immintrin.h>

#include "benchmark/benchmark.h"

// Custom allocator for aligned and padded allocation for SIMD implementations.
// (Adapted from arbor source.)

template <typename T = void>
struct padded_allocator {
    static constexpr std::size_t alignment_ = 64;

    using value_type = T;
    using pointer = T*;

    padded_allocator() noexcept {}

    template <typename U>
    padded_allocator(const padded_allocator<U>& b) noexcept {}

    pointer allocate(std::size_t n) {
        if (n>std::size_t(-1)/sizeof(T)) {
            throw std::bad_alloc();
        }

        void* mem = nullptr;
        std::size_t size = round_up(n*sizeof(T), alignment_);
        std::size_t pm_align = std::max(alignment_, sizeof(void*));

        if (auto err = posix_memalign(&mem, pm_align, size)) {
            throw std::system_error(err, std::generic_category(), "posix_memalign");
        }
        return static_cast<pointer>(mem);
    }

    void deallocate(pointer p, std::size_t n) {
        std::free(p);
    }

    bool operator==(const padded_allocator& a) const { return true; }
    bool operator!=(const padded_allocator& a) const { return false; }

private:
    static std::size_t round_up(std::size_t v, std::size_t b) {
         std::size_t m = v%b;
         return v-m+(m? b: 0);
    }
};

template <typename T>
using padded_vector = std::vector<T, padded_allocator<T>>;

struct indirect_example {
    padded_vector<double> data;
    padded_vector<double> inc;
    padded_vector<int> offset;

    indirect_example(std::size_t datasz, std::size_t incsz):
        data(datasz), inc(incsz), offset(incsz) {}

    // checked default indirect addition:
    void run() {
        std::size_t datasz = data.size();
        std::size_t incsz = inc.size();
        assert(offset.size()>=incsz);

        for (std::size_t i = 0; i<incsz; ++i) {
            assert(offset[i]>=0 && offset[i]<datasz);
            data[offset[i]] += inc[i];
        }
    }
};

template <typename RNG>
indirect_example generate_example(std::size_t N, double sparsity, bool monotonic, RNG& R) {
    std::uniform_real_distribution<double> UD(-1., 1.);
    std::uniform_int_distribution<int> UI(0,N-1);

    indirect_example ex(N, N*sparsity);
    std::generate(ex.data.begin(), ex.data.end(), [&]() { return UD(R); });
    std::generate(ex.inc.begin(), ex.inc.end(), [&]() { return UD(R); });
    std::generate(ex.offset.begin(), ex.offset.end(), [&]() { return UI(R); });
    if (monotonic) std::sort(ex.offset.begin(), ex.offset.end());

    return ex;
}

using indirect_add_fn = std::function<void (indirect_example&)>;
void check_indirect_add(indirect_example ex, indirect_add_fn op) {
    // note: reordering of addition may make sum comparison inexact.
    double epsilon = ex.data.size()*1e-14;
    indirect_example ex_check = ex;

    ex_check.run();
    op(ex);

    assert(ex.data.size()==ex_check.data.size());
    for (std::size_t i = 0; i<ex.data.size(); ++i) {
        assert(std::abs(ex.data[i]-ex_check.data[i])<=epsilon);
    }
}

void run_benchmark(benchmark::State& state, indirect_add_fn op, std::size_t N, double sparsity, bool monotonic) {
    std::minstd_rand R;

    auto ex = generate_example(N, sparsity, monotonic, R);
    check_indirect_add(ex, op);

    for (auto _: state) {
        op(ex);
    }
}

void naive_impl(indirect_example& ex) {
    std::size_t incsz = ex.inc.size();

    for (std::size_t i = 0; i<incsz; ++i) {
        ex.data[ex.offset[i]] += ex.inc[i];
    }
}

void scalar_impl(indirect_example& ex) {
    std::size_t incsz = ex.inc.size();

    double* p = ex.data.data();
    const double* a = ex.inc.data();
    const int* o = ex.offset.data();

    double acc = 0;
    for (std::size_t i = 0; i<incsz-1; ++i) {
        acc += a[i];
        if (o[i]!=o[i+1]) {
            p[o[i]] += acc;
            acc = 0;
        }
    }
    acc += a[incsz-1];
    p[o[incsz-1]] += acc;
}

#if defined(__AVX512F__)
inline void addi_avx512(double* p, __m512i o, __m512d a) {
    __m512i confv = _mm512_conflict_epi32(o);

    int conf[8];
    _mm256_storeu_si256((__m256i*)conf, _mm512_castsi512_si256(confv));

    double aa[8];
    _mm512_storeu_pd((__m512d*)aa, a);

    __mmask16 wmask = _mm512_cmpeq_epi32_mask(confv, _mm512_setzero_epi32());

    __m512d x = _mm512_i32gather_pd(_mm512_castsi512_si256(o), (const void*)p, sizeof(double));

    __m512d p01 = _mm512_add_pd(
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[0]), _mm_set1_pd(aa[0])),
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[1]), _mm_set1_pd(aa[1])));

    __m512d p23 = _mm512_add_pd(
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[2]), _mm_set1_pd(aa[2])),
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[3]), _mm_set1_pd(aa[3])));

    __m512d p45 = _mm512_add_pd(
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[4]), _mm_set1_pd(aa[4])),
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[5]), _mm_set1_pd(aa[5])));

    __m512d p67 = _mm512_add_pd(
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[6]), _mm_set1_pd(aa[6])),
                        _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[7]), _mm_set1_pd(aa[7])));

    x = _mm512_add_pd(
            _mm512_add_pd(x, a),
            _mm512_add_pd(
                _mm512_add_pd(p01, p23),
                _mm512_add_pd(p45, p67)
            )
        );

    _mm512_mask_i32scatter_pd((void*)p, wmask, _mm512_castsi512_si256(o), x, sizeof(double));
}

void avx512_impl(indirect_example& ex) {
    std::size_t incsz = ex.inc.size();
    assert(incsz%8==0);

    double* p = ex.data.data();
    double* inc = ex.inc.data();
    int* off = ex.offset.data();

    __mmask16 lo = _cvtu32_mask16(0xffu);
    for (std::size_t i = 0; i<incsz; i+=8) {
        __m512d a = _mm512_load_pd((void*)inc);
        __m512i o = _mm512_maskz_loadu_epi32(lo, (void*)off);

        addi_avx512(p, o, a);

        inc += 8;
        off += 8;
    }
}
#endif

int main(int argc, char** argv) {
    std::vector<std::pair<std::string, indirect_add_fn>> impls = {
        {"naive", naive_impl},
        {"scalar", scalar_impl}
    };

#if defined(__AVX512F__)
    impls.push_back({"avx512", avx512_impl});
#endif

    std::size_t N = 10240;
    double sparse = 0.1, dense = 10, very_dense = 100;

    for (auto& impl: impls) {
        benchmark::RegisterBenchmark((impl.first+"/sparse").c_str(),
           [&](auto& st) { run_benchmark(st, impl.second, N, sparse, false); });
        benchmark::RegisterBenchmark((impl.first+"/dense").c_str(),
           [&](auto& st) { run_benchmark(st, impl.second, N, dense, false); });
        benchmark::RegisterBenchmark((impl.first+"/very_dense").c_str(),
           [&](auto& st) { run_benchmark(st, impl.second, N, very_dense, false); });
        benchmark::RegisterBenchmark((impl.first+"/dense_monotonic").c_str(),
           [&](auto& st) { run_benchmark(st, impl.second, N, dense, true); });
        benchmark::RegisterBenchmark((impl.first+"/very_dense_monotonic").c_str(),
           [&](auto& st) { run_benchmark(st, impl.second, N, very_dense, true); });
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}

