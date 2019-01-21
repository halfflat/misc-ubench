#include <cassert>
#include <functional>
#include <random>
#include <vector>

#include <immintrin.h>

#include "benchmark/benchmark.h"

struct indirect_example {
    std::vector<double> data;
    std::vector<double> inc;
    std::vector<int> offset;

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
    __m512i conf = _mm512_conflict_epi32(o);
    __mmask16 wmask = _mm512_cmpeq_epi32_mask(conf, _mm512_setzero_epi32());

    auto psum = [&](unsigned j) {
        return _mm512_maskz_broadcastsd_pd(_mm512_int2mask(conf[j]), _mm_set1_pd(a[j]));
    };

    __m512d x = _mm512_i32gather_pd(_mm512_castsi512_si256(o), (const void*)p, sizeof(double));

    x = _mm512_add_pd(
        _mm512_add_pd(x, a),
        _mm512_add_pd(
            _mm512_add_pd(
                _mm512_add_pd(psum(0), psum(1)),
                _mm512_add_pd(psum(2), psum(3))
            ),
            _mm512_add_pd(
                _mm512_add_pd(psum(4), psum(5)),
                _mm512_add_pd(psum(6), psum(7))
            )
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
        __m512i o = _mm512_maskz_load_epi32(lo, (void*)off);

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
        benchmark::RegisterBenchmark((impl.first+"/sparse").c_str(), run_benchmark, impl.second, N, sparse, false);
        benchmark::RegisterBenchmark((impl.first+"/dense").c_str(), run_benchmark, impl.second, N, dense, false);
        benchmark::RegisterBenchmark((impl.first+"/very_dense").c_str(), run_benchmark, impl.second, N, very_dense, false);
        benchmark::RegisterBenchmark((impl.first+"/dense_monotonic").c_str(), run_benchmark, impl.second, N, dense, true);
        benchmark::RegisterBenchmark((impl.first+"/very_dense_monotonic").c_str(), run_benchmark, impl.second, N, very_dense, true);
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}

