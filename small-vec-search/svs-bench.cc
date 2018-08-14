#include <iostream>
#include <string>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "benchmark/benchmark.h"

void make_random_cstr(char *b, size_t n) {
    static std::minstd_rand R;
    static std::uniform_int_distribution<char> C('A', 'z');
    std::normal_distribution<float> N(n/2.0, n/5.0);

    size_t len = (int)N(R);
    if (len<1) len=1;
    else if (len>=n) len=n-1;

    b[len]=0;
    for (size_t i=0; i<len; ++i) b[i]=C(R);
}

template <typename C, typename X>
bool find(const C& c, const X& x) {
    for (const auto& i: c) if (i==x) return true;
    return false; 
}

template <typename Y, typename X>
bool find(const std::set<Y>& c, const X& x) {
    return !!c.count(x);
}

template <typename Y, typename X>
bool find(const std::unordered_set<Y>& c, const X& x) {
    return !!c.count(x);
}

template <typename Container>
void bench_string_search(benchmark::State& state) {
    char buf[10];
    std::vector<std::string> keys(2*state.range(0));
    for (auto& k: keys) {
        make_random_cstr(buf, sizeof buf);
        k = buf;
    }

    Container set(keys.begin(), keys.begin()+keys.size()/2);

    while (state.KeepRunning()) {
        for (unsigned i = 0; i<keys.size(); ++i) {
            benchmark::DoNotOptimize(find(set, keys[i]));
        }
    }
}

BENCHMARK_TEMPLATE(bench_string_search,std::set<std::string>)->Arg(3)->Arg(6)->Arg(12)->Arg(20)->Arg(40);
BENCHMARK_TEMPLATE(bench_string_search,std::unordered_set<std::string>)->Arg(3)->Arg(6)->Arg(12)->Arg(20)->Arg(40);
BENCHMARK_TEMPLATE(bench_string_search,std::vector<std::string>)->Arg(3)->Arg(6)->Arg(12)->Arg(20)->Arg(40);

BENCHMARK_MAIN();

