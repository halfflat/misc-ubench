#include <cassert>
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <type_traits>

#include "benchmark/benchmark.h"

template <typename T>
int signum(T x) {
    return (x>T(0)) - (x<T(0));
}

template <typename T, bool is_signed = std::is_signed<T>::value>
struct abs_ {
    T operator()(T x) const { return std::abs(x); }
};

template <typename T>
struct abs_<T, false> {
    T operator()(T x) const { return x; }
};

template <typename T>
T abs(T x) {
    return abs_<T>{}(x);
}

template <typename T>
T round_up1(T v, T base) {
    T r = base*(v/base);
    return r==v? v: v<0? r-abs(base): r+abs(base);
}

template <typename T>
T round_up2(T v, T base) {
    T r = base*(v/base);
    return r==v? v: r+signum(v)*signum(base)*base;
}

template <typename T>
T round_up3(T v, T base) {
    T m = v%base;
    return v-m+signum(m)*abs(base);
}

template <typename T>
T round_up4(T v, T base) {
    T m = v%base;
    return v-m+signum(m)*signum(base)*base;
}

unsigned round_up5(unsigned v, unsigned base) {
    unsigned m = v%base;
    return m? v+base-m: v;
}

template <typename T, typename U, typename C = typename std::common_type<T, U>::type>
C round_up_x(T v, U base) {
    C m = v%base;
    return v-m+signum(m)*abs(base);
}

template <typename T>
void generate(std::vector<T>& as, std::vector<T>& bs, std::size_t n) {
    static std::minstd_rand R;
    T minv = std::is_unsigned<T>::value? 0: -100;
    T maxv = 100;

    static std::uniform_int_distribution<T> A(3*minv, 3*maxv);
    static std::uniform_int_distribution<T> B(minv, maxv);

    as.resize(n);
    for (std::size_t i = 0; i<n; ++i) {
        as[i]=A(R);
    }

    bs.resize(n);
    for (std::size_t i = 0; i<n; ++i) {
        while (!(bs[i]=B(R))) ;
    }
}

template <typename T>
void bench_round_up(T (*fn)(T, T), benchmark::State& state) {
    constexpr std::size_t batch = 10000;
    std::vector<T> cs;
    std::vector<T> as, bs;

    for (auto _: state) {
        state.PauseTiming();
        generate(as, bs, batch);
        cs.resize(batch);
        state.ResumeTiming();

        for (std::size_t i=0; i<batch; ++i) {
            cs[i] = fn(as[i], bs[i]);
        }
    }

    for (std::size_t i = 0; i<cs.size(); ++i) {
        T a = as[i], b = bs[i], c = cs[i];
        assert(c%b==0);
        assert(a>=0 && c>=a || a<=0 && c<=a);

        a = abs(a);
        b = abs(b);
        c = abs(c);
        assert(a+b>c);
    }
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    benchmark::RegisterBenchmark("round_up1/int", [](benchmark::State& s) { bench_round_up(round_up1<int>, s); });
    benchmark::RegisterBenchmark("round_up2/int", [](benchmark::State& s) { bench_round_up(round_up2<int>, s); });
    benchmark::RegisterBenchmark("round_up3/int", [](benchmark::State& s) { bench_round_up(round_up3<int>, s); });
    benchmark::RegisterBenchmark("round_up4/int", [](benchmark::State& s) { bench_round_up(round_up4<int>, s); });

    benchmark::RegisterBenchmark("round_up1/unsigned", [](benchmark::State& s) { bench_round_up(round_up1<unsigned>, s); });
    benchmark::RegisterBenchmark("round_up2/unsigned", [](benchmark::State& s) { bench_round_up(round_up2<unsigned>, s); });
    benchmark::RegisterBenchmark("round_up3/unsigned", [](benchmark::State& s) { bench_round_up(round_up3<unsigned>, s); });
    benchmark::RegisterBenchmark("round_up4/unsigned", [](benchmark::State& s) { bench_round_up(round_up4<unsigned>, s); });
    benchmark::RegisterBenchmark("round_up5/unsigned", [](benchmark::State& s) { bench_round_up(round_up4<unsigned>, s); });

    benchmark::RegisterBenchmark("round_up_x/int", [](benchmark::State& s) { bench_round_up(round_up_x<int, int>, s); });
    benchmark::RegisterBenchmark("round_up_x/unsigned", [](benchmark::State& s) { bench_round_up(round_up_x<unsigned, unsigned>, s); });

    benchmark::RunSpecifiedBenchmarks();
}

