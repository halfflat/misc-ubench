#include "benchmark/benchmark.h"

void qappend_bench(benchmark::State& state) {
    volatile int x;
    for (auto _: state) {
        x = 1;
    }
}

BENCHMARK(qappend_bench);
BENCHMARK_MAIN();

