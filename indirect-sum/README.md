What's an efficient way of performing p[o[i]] += v[i] for SIMD value v and index o,
when the indices of o are not necesarily distinct?

Compare scalar approach with AVX512CD-based vector solution.

## Benchmark

* Run the indirect addition with a vector of data of size N and α*N indirect additions,
for some density factor α.

* For large α, also test the performance when the indirect indices are monotonic (i.e.
sorted.)

Currently N is 10240 and α is 0.1, 10, and 100.

## Notes

There are two non-vectorized implementations: the naive one simply applies `+=` for each
indirect addition; the 'scalar' test accumulates consecutive values with the same offset
to minimize the number of writes.

On an Ivy Bridge system with gcc 8.2.1, the naive implementation is about 2.3x slower
with α=100, but about 1.8x _quicker_ with α=10. I'm guessing this is the cost of missed
branch predictions, but would have to look more closely.


