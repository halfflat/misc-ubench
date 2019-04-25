#include <algorithm>
#include <cassert>
#include <functional>
#include <random>
#include <vector>

#include <cstdio>
using std::printf;

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
indirect_example generate_example(std::size_t N, int min_width, int max_width, RNG& R) {
    std::uniform_real_distribution<double> UD(-1., 1.);

    if (N<1) N = 1;
    if (min_width<1) min_width=1;
    if (max_width>N) max_width=N;
    std::uniform_int_distribution<int> UI(min_width, max_width);

    indirect_example ex(N, N);
    std::generate(ex.data.begin(), ex.data.end(), [&]() { return UD(R); });
    std::generate(ex.inc.begin(), ex.inc.end(), [&]() { return UD(R); });

    for (int j = 0, k = 0; j<N; ) {
        int w = UI(R);
        if (w+j>N) w = N-j;
        std::fill(&ex.offset[j], &ex.offset[j]+w, k++);
        j += w;
    }

    return ex;
}

using indirect_add_fn = std::function<float (indirect_example&, int)>;

void check_indirect_add(indirect_example ex, indirect_add_fn op) {
    // note: reordering of addition may make sum comparison inexact.
    double epsilon = ex.data.size()*1e-14;
    indirect_example ex_check = ex;

    ex_check.run();
    op(ex, 1);

    assert(ex.data.size()==ex_check.data.size());
    for (std::size_t i = 0; i<ex.data.size(); ++i) {
        //printf("i: %02zu; check: %f; ex: %f\n", i, ex_check.data[i], ex.data[i]);
        assert(std::abs(ex.data[i]-ex_check.data[i])<=epsilon);
    }
}

void run_benchmark(benchmark::State& state, indirect_add_fn op, std::size_t N, int wl, int wh) {
    constexpr int reps = 5;
    std::minstd_rand R;

    auto ex = generate_example(N, wl, wh, R);
    check_indirect_add(ex, op);

    for (auto _: state) {
        float t = op(ex, reps);
        if (t) state.SetIterationTime(t);
    }
}

float naive_reduce(indirect_example& ex, int reps) {
    std::size_t incsz = ex.inc.size();

    for (int i = 0; i<reps; ++i) {
        for (std::size_t i = 0; i<incsz; ++i) {
            ex.data[ex.offset[i]] += ex.inc[i];
        }
    }
    return 0;
}

float scalar_reduce(indirect_example& ex, int reps) {
    std::size_t incsz = ex.inc.size();

    double* p = ex.data.data();
    const double* a = ex.inc.data();
    const int* o = ex.offset.data();

    for (int c = 0; c<reps; ++c) {
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
    return 0;
}

extern float arbor_cuda_reduce_impl(std::size_t N, double* p, const double* v, const int* index, int reps);

float arbor_cuda_reduce(indirect_example& ex, int reps) {
    return arbor_cuda_reduce_impl(ex.inc.size(), ex.data.data(), ex.inc.data(), ex.offset.data(), reps);
}

extern float expr1_cuda_reduce_impl(std::size_t N, double* p, const double* v, const int* index, int reps);

float expr1_cuda_reduce(indirect_example& ex, int reps) {
    return expr1_cuda_reduce_impl(ex.inc.size(), ex.data.data(), ex.inc.data(), ex.offset.data(), reps);
}

extern float expr2_cuda_reduce_impl(std::size_t N, double* p, const double* v, const int* index, int reps);

float expr2_cuda_reduce(indirect_example& ex, int reps) {
    return expr2_cuda_reduce_impl(ex.inc.size(), ex.data.data(), ex.inc.data(), ex.offset.data(), reps);
}

int main(int argc, char** argv) {
    struct impl {
        std::string name;
        indirect_add_fn fn;
        bool manual_timing;
    };

    impl impls[] = {
        {"naive", naive_reduce, false},
        {"scalar", scalar_reduce, false},
        {"arbor_cuda", arbor_cuda_reduce, true},
        {"expr1_cuda", expr1_cuda_reduce, true},
        {"expr2_cuda", expr2_cuda_reduce, true}
    };

    std::size_t N = 1024007;
    double sparse = 0.1, dense = 10, very_dense = 100;

    for (auto& impl: impls) {
        std::vector<benchmark::internal::Benchmark*> benches;

        benches.push_back(benchmark::RegisterBenchmark((impl.name+"/constant").c_str(),
           [&](auto& st) { run_benchmark(st, impl.fn, N, N, N); }));
        benches.push_back(benchmark::RegisterBenchmark((impl.name+"/distinct").c_str(),
           [&](auto& st) { run_benchmark(st, impl.fn, N, 1, 1); }));
        benches.push_back(benchmark::RegisterBenchmark((impl.name+"/w1_5").c_str(),
           [&](auto& st) { run_benchmark(st, impl.fn, N, 1, 5); }));
        benches.push_back(benchmark::RegisterBenchmark((impl.name+"/w15_60").c_str(),
           [&](auto& st) { run_benchmark(st, impl.fn, N, 15, 60); }));
        benches.push_back(benchmark::RegisterBenchmark((impl.name+"/w123").c_str(),
           [&](auto& st) { run_benchmark(st, impl.fn, N, 123, 123); }));

        for (auto& b: benches) {
            if (impl.manual_timing) b->UseManualTime();
            b->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                return *(std::min_element(std::begin(v), std::end(v)));
            });
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}

