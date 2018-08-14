#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>

#include <omp.h>

#include "benchmark/benchmark.h"

constexpr int N = 1000, M = 1000;

struct block {
    double *data=nullptr;
    int stride=1;

    double* operator[](int i) { return data+stride*i; }
};

enum { WRONG=0, PARAWRONG=1, SANE=2, PARASANE=3 };

#if defined(EXPENSIVE)
double expensive(double x) {
    double y = std::exp(x)-1;
    if (y>0.2) y=0.2;
    return std::pow(y, 1.1);
}
#else
inline double expensive(double x) { return x; }
#endif

void run(int which, int M, int N, block a, block b) {
    switch (which) {
    case WRONG:
        for (int j=1; j<N-1; ++j) {
            for (int i=1; i<M-1; ++i) {
                a[i][j] += expensive(0.5*(b[i+1][j]-b[i-1][j]) + 0.3*(b[i][j+1]-b[i][j-1]));
            }
        }
        break;
    case PARAWRONG:
        for (int j=1; j<N-1; ++j) {
            #pragma omp parallel for
            for (int i=1; i<M-1; ++i) {
                a[i][j] += expensive(0.5*(b[i+1][j]-b[i-1][j]) + 0.3*(b[i][j+1]-b[i][j-1]));
            }
        }
        break;
    case SANE:
        for (int i=1; i<M-1; ++i) {
            for (int j=1; j<N-1; ++j) {
                a[i][j] += expensive(0.5*(b[i+1][j]-b[i-1][j]) + 0.3*(b[i][j+1]-b[i][j-1]));
            }
        }
        break;
    case PARASANE:
        #pragma omp parallel for collapse(2)
        for (int i=1; i<M-1; ++i) {
            for (int j=1; j<N-1; ++j) {
                a[i][j] += expensive(0.5*(b[i+1][j]-b[i-1][j]) + 0.3*(b[i][j+1]-b[i][j-1]));
            }
        }
        break;
    }
}

void harness(benchmark::State& state, int dim, int which) {
#ifdef PAD
    int stride = dim+8;
#else
    int stride = dim;
#endif

    double* a_ = new double[dim*stride]();
    double* b_ = new double[dim*stride]();

    block a{a_, stride};
    block b{b_, stride};

    std::minstd_rand R;
    std::uniform_real_distribution<double> U(0,1e-3);

    std::fill(&a[0][0], &a[0][0]+dim*dim, 0.0);
    for (int i=0; i<dim; ++i)
        for (int j=0; j<dim; ++j)
            b[i][j] = U(R);

    for (auto _: state) {
        run(which, dim, dim, a, b);
        benchmark::ClobberMemory();
    }

    delete[] a_;
    delete[] b_;
}

std::function<void (benchmark::State&)> make_bench(int dim, int which) {
    return [=](benchmark::State& s) { harness(s, dim, which); };
}

int main(int argc, char** argv) {
    std::cout << "#thread: " << omp_get_max_threads() << "\n";
    for (int dim: {50, 100, 200, 400, 800, 1600}) {
        benchmark::RegisterBenchmark(("wrong/"+std::to_string(dim)).c_str(), make_bench(dim, WRONG))->UseRealTime();
        benchmark::RegisterBenchmark(("parawrong/"+std::to_string(dim)).c_str(), make_bench(dim, PARAWRONG))->UseRealTime();
        benchmark::RegisterBenchmark(("sane/"+std::to_string(dim)).c_str(), make_bench(dim, SANE))->UseRealTime();
        benchmark::RegisterBenchmark(("parasane/"+std::to_string(dim)).c_str(), make_bench(dim, PARASANE))->UseRealTime();
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}

