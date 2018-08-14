#include <cassert>
#include <random>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"

#include "augmaxheap.h"

// Problem: partially order a set S of non-empty half-open intervals by
// [a,b) < [c,d] iff b ≤ c. Find the minimal elements of S.

// Online algorithm: heap based.
template <typename T>
struct min_interval_heap {
    aug_max_heap<T> heap;

    void push_back(interval<T> ab) {
        if (!heap.empty()) {
            if (ab.first>=heap.min_second()) return;
            while (!heap.empty() && heap.top().first>=ab.second) heap.pop();
        }
        heap.push_back(std::move(ab));
    }

    using iterator = typename aug_max_heap<T>::const_iterator;
    iterator begin() const { return heap.begin(); }
    iterator end() const { return heap.end(); }

    std::size_t size() const { return heap.size(); }
};

// Online algorithm: vector based.
template <typename T>
struct min_interval_vector {
    std::vector<interval<T>> items, temp;

    void push_back(interval<T> ab) {
        if (items.empty()) items.push_back(std::move(ab));
        else {
            temp.clear();
            T min_second = items.front().second;
            for (auto p: items) {
                if (p.first >= ab.second) continue;
                if (p.second < min_second) min_second = p.second;
                temp.push_back(p);
            }
            if (ab.first<min_second) temp.push_back(std::move(ab));
            std::swap(items, temp);
        }
    }

    using iterator = typename std::vector<interval<T>>::const_iterator;
    iterator begin() const { return items.begin(); }
    iterator end() const { return items.end(); }

    std::size_t size() const { return items.size(); }
};

// Offline algorithm: apply global minimum upper bound.
template <typename T>
struct min_interval_offline {
    mutable std::vector<interval<T>> candidates;
    T upper;
    mutable bool stale = true;

    void push_back(interval<T> ab) {
        stale = true;
        if (candidates.empty()) {
            upper = ab.second;
            candidates.push_back(std::move(ab));
        }
        else {
            if (ab.second<upper) upper = ab.second;
            if (ab.first<upper) candidates.push_back(std::move(ab));
        }
    }

    using iterator = typename std::vector<interval<T>>::const_iterator;
    iterator begin() const { filter(); return candidates.begin(); }
    iterator end() const { filter(); return candidates.end(); }

    std::size_t size() const { filter(); return candidates.size(); }

    void filter() const {
        if (!stale) return;
        std::size_t n = candidates.size();
        for (std::size_t i = 0; i<n; ) {
            if (candidates[i].first>=upper) {
                std::swap(candidates[i], candidates[--n]);
            }
            else ++i;
        }
        candidates.resize(n);
        stale = false;
    }
};

template <typename Rng>
std::vector<interval<int>> generate_intervals(unsigned n, unsigned n_overlap, Rng& R) {
    std::vector<interval<int>> ivals;

    int width = 4*n_overlap;
    int c = width;

    std::uniform_int_distribution<int> U(0, width/2);

    for (unsigned i = 0; i<n_overlap; ++i) {
        ivals.push_back({c-U(R), c+1+U(R)});
    }

    while (ivals.size()<n) {
        interval<int> i = ivals[ivals.size()-n_overlap];
        i.first += width+2;
        i.second += width+2;
        ivals.push_back(i);
    }

    return ivals;
}

template <typename Impl>
void bench_min_interval(benchmark::State& state) {
    std::minstd_rand R;

    unsigned n = state.range(0);
    unsigned n_overlap = state.range(1);
    if (n_overlap<1u) n_overlap = 1u;
    if (n_overlap>n) n_overlap = n;

    std::vector<interval<int>> ivals = generate_intervals(n, n_overlap, R);

    for (auto _: state) {
        state.PauseTiming();
        std::shuffle(ivals.begin(), ivals.end(), R);
        state.ResumeTiming();

        Impl impl;
        for (const auto& i: ivals) impl.push_back(i);
        benchmark::DoNotOptimize(impl.size());
        assert(impl.size()==n_overlap);
    }
}

BENCHMARK_TEMPLATE(bench_min_interval, min_interval_heap<int>)
    ->Args({100, 1})
    ->Args({100, 3})
    ->Args({100, 30})
    ->Args({1000, 1})
    ->Args({1000, 30})
    ->Args({1000, 300})
    ->Args({10000, 1})
    ->Args({10000, 300})
    ->Args({10000, 3000});

BENCHMARK_TEMPLATE(bench_min_interval, min_interval_vector<int>)
    ->Args({100, 1})
    ->Args({100, 3})
    ->Args({100, 30})
    ->Args({1000, 1})
    ->Args({1000, 30})
    ->Args({1000, 300})
    ->Args({10000, 1})
    ->Args({10000, 300})
    ->Args({10000, 3000});

BENCHMARK_TEMPLATE(bench_min_interval, min_interval_offline<int>)
    ->Args({100, 1})
    ->Args({100, 3})
    ->Args({100, 30})
    ->Args({1000, 1})
    ->Args({1000, 30})
    ->Args({1000, 300})
    ->Args({10000, 1})
    ->Args({10000, 300})
    ->Args({10000, 3000});

BENCHMARK_MAIN();
