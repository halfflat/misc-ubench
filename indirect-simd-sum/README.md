What's an efficient way of performing p[o[i]] += v[i] for SIMD value v and index o,
when the indices of o are not necesarily distinct?

Compare scalar approach with AVX512CD-based vector solution.

### Placeholder

Bench not made yet, but candidate code (for GCC and Clang, arch KNL or Skylake-X)
for the vector version would be along the lines of the following:

```
void addi_avx512(double* p, const __m512i& o, __m512d a) {
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
```
