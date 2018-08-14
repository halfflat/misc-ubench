#pragma once

#include <cassert>
#include <iterator>
#include <numeric>
#include <utility>
#include <vector>

template <typename T>
using interval = std::pair<T, T>;

// Binary max heap of intervals by left hand value,
// where we also store at each node the subtree minimum
// of the right hand value.

template <typename T>
struct aug_max_heap {
    using value_type = interval<T>;
    using size_type = std::size_t;
    using const_reference = const value_type&;
    using reference = const value_type&;

    struct item: value_type {
        T min_second;

        item() {}
        item(value_type x): value_type(x), min_second(x.second) {}
    };

    std::vector<item> heap;

    size_type size() const { return heap.size(); }
    bool empty() const { return !heap.size(); }

    aug_max_heap() {}

    template <typename I>
    aug_max_heap(I b, I e): heap(b, e) {
        for (size_type k = size()/2; k-->0; ) down(k);
    }

    struct iterator {
        using inner = typename std::vector<item>::const_iterator;
        inner i;

        using value_type = aug_max_heap::value_type;
        using pointer_type = const value_type*;
        using reference = const value_type&;
        using difference_type = typename std::iterator_traits<inner>::difference_type;

        iterator() = default;
        iterator(inner i): i(i) {}

        iterator& operator++() { ++i; return *this; }
        iterator operator++(int) { return iterator(i++); }

        bool operator==(iterator j) const { return i==j.i; }
        bool operator!=(iterator j) const { return i!=j.i; }

        reference operator*() const { return *i; }
    };

    using const_iterator = iterator;

    iterator begin() const { return iterator(heap.begin()); }
    iterator end() const { return iterator(heap.end()); }

    T min_second() const { return heap[0].min_second; }
    const_reference top() const { return heap[0]; }

    void push_back(value_type p) {
        check_invariants();
        heap.push_back(std::move(p));
        up(heap.size()-1);
        check_invariants();
    }

    void pop() {
        if (empty()) return;

#ifndef NDEBUG
        auto keep = heap;
#endif
        check_invariants();
        std::swap(heap.front(), heap.back());

        heap.pop_back();
        if (empty()) return;

        unsigned k = (size()-1)/2;
        if (k>0) {
            heap[k].min_second = heap[k].second;
            update_min_second(k);
            for (k = (k-1)/2; k!=0; k = (k-1)/2) {
                heap[k].min_second = heap[k].second;
                update_min_second(k);
            }
        }
        down(0);

        check_invariants();
    }

    void check_invariants() {
#ifndef NDEBUG
        if (heap.empty()) return;
        for (size_type i = 0; i<size(); ++i) {
            auto& el = heap[i];
            size_type l = 2*i+1, r = 2*i+2;
            if (l>=size()) {
                assert(el.min_second==el.second);
            }
            else if (r>=size()) {
                assert(el.min_second==std::min(el.second, heap[l].second));
                assert(el.first>=heap[l].first);
            }
            else {
                assert(el.min_second==std::min(el.second, std::min(heap[l].min_second, heap[r].min_second)));
                assert(el.first>=heap[l].first && el.first>=heap[r].first);
            }
        }
#endif
    }

private:
    bool update_min_second(size_type p) {
        auto m = heap[p].min_second;

        auto c = 2*p+1;
        if (c<size()) {
            m = std::min(m, heap[c].min_second);
            if (++c<size()) m = std::min(m, heap[c].min_second);
        }

        if (m==heap[p].min_second) return false;
        heap[p].min_second = m;
        return true;
    }

    bool update_min_second(size_type p, size_type c) {
        if (c<size() && heap[c].min_second<heap[p].min_second) {
            heap[p].min_second = heap[c].min_second;
            return true;
        }
        return false;
    }

    void down(size_type k) {
        for (;;) {
            size_type l = 2*k+1;
            size_type r = l+1;

            size_type c = k;

            if (r<size()) {
                c = heap[l].first>heap[r].first? l: r;
            }
            else if (l<size()) {
                c = l;
            }

            if (c!=k && heap[k].first<heap[c].first) {
                std::swap(heap[k], heap[c]);
                update_min_second(k);
                k = c;
            }
            else {
                update_min_second(k);
                return;
            }
        }
        check_invariants();
    }

    void up(size_type k) {
        while (k!=0) {
            size_type p = (k-1)/2;
            if (heap[p].first>=heap[k].first) break;

            std::swap(heap[p], heap[k]);
            update_min_second(p, k);

            heap[k].min_second = heap[k].second;
            update_min_second(k);
            k = p;
        }
        while (k!=0) {
            size_type p = (k-1)/2;
            if (!update_min_second(p, k)) break;
            k = p;
        }
    }
};
