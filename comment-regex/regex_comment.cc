#include <cassert>
#include <string>
#include <regex>

#include "benchmark/benchmark.h"

bool is_comment_regex(const std::string& line) {
    static std::regex re("\\s*(?:#.*)?");
    return std::regex_match(line, re);
}

bool is_comment_manual(const std::string& line) {
    auto i = line.find_first_not_of(" \r\n\t");
    return i==std::string::npos || line[i]=='#';
}

template <typename Fn>
void bench_is_comment(Fn fn, benchmark::State& state) {
    std::string pos_tests[] = {
        "",
        " \t# some commment",
        "# some commment",
        "  \t\r \r"
    };

    std::string neg_tests[] = {
        ".",
        "   \t .",
        "   \t x #foo",
    };

    while (state.KeepRunning()) {
        for (auto& s: pos_tests) {
            assert(fn(s));
        }
        for (auto& s: neg_tests) {
            assert(!fn(s));
        }
    }
}

void bench_is_comment_manual(benchmark::State& state) {
    bench_is_comment(is_comment_manual, state);
}

void bench_is_comment_regex(benchmark::State& state) {
    bench_is_comment(is_comment_regex, state);
}

BENCHMARK(bench_is_comment_regex);
BENCHMARK(bench_is_comment_manual);

BENCHMARK_MAIN();

