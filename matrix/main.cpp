#include <cl_utils.h>
#include <common_miscs.h>
#include <vector>
#include <stdio.h>
#include <array>
#include <intrin.h>
#include <immintrin.h>
#include <type_traits>
#include <fstream>

#define __AVX2_ALIGNED   __declspec(align(32))

template <class REAL> static std::vector<REAL> gen_random_matrix(size_t cols, size_t rows) {

  std::uniform_real_distribution<REAL> rd{(REAL)-100.0, (REAL)100.0};

  size_t mat_len = cols * rows, i;
  std::vector<double> mat(mat_len);
  for (i = 0; i < mat_len; ++i) {
    mat[i] = rd(g_RandomEngine);
  }

  return mat;
}

template<class REAL>
void print_matrix(int indent, int cwidth, const std::vector<REAL> &mat, size_t cols, size_t rows) {

  ptrdiff_t i, j, k;

  k = -1;
  for(i = 0; i < rows; ++i) {
    printf("%*c", indent, ' ');
    for (j = 0; j < cols; ++j)
      printf("%*c%g", cwidth, ' ', mat[++k]);
    putc('\n', stdout);
  }
}

template <class REAL>
bool check_matrix_equiv(
    const std::vector<REAL> &mat1, const std::vector<REAL> &mat2, REAL eq_tol, size_t cols, size_t rows) {

  bool res = true;

  // Verify the results.
  if(mat1.size() != mat2.size()) {
    printf("matrix 1 and 2 have different dimensions\n");
    return false;
  }

  size_t count = mat1.size();
  size_t count_f4 = count & ~3;
  size_t count_r4 = count & 3;

  auto it1 = mat1.data(), it1end = it1 + count_f4;
  auto it2 = mat2.data();
  size_t i = 0;

  __m256d ymm0, ymm1;
  const __m256d ymm2 = _mm256_set1_pd(eq_tol);
  const __m256d ymm3 = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7fffffffffffffff));
  int cmp_masks;

  for(; it1 != it1end; it1 += 4, it2 += 4, i += 4) {
    ymm0 = _mm256_loadu_pd(it1);
    ymm1 = _mm256_loadu_pd(it2);
    ymm0 = _mm256_sub_pd(ymm0, ymm1);
    ymm0 = _mm256_and_pd(ymm0, ymm3);
    ymm1 = _mm256_cmp_pd(ymm0, ymm2, _CMP_GE_OQ);
    if ((cmp_masks = _mm256_movemask_pd(ymm1))) {

      if(cmp_masks & 1)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", i % cols, i / cols, *it1, *it2);
      if (cmp_masks & 2)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", (i+1) % cols, (i+1) / cols, *(it1+1), *(it2+1));
      if (cmp_masks & 4)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", (i+2) % cols, (i+2) / cols, *(it1+2), *(it2+2));
      if (cmp_masks & 8)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", (i+3) % cols, (i+3) / cols, *(it1+3), *(it2+3));
      res = false;
    }
  }

  if(count_r4) {
    ymm0 = _mm256_setzero_pd();
    ymm1 = _mm256_setzero_pd();
    memcpy(&ymm0, it1, count_r4 * sizeof(double));
    memcpy(&ymm1, it2, count_r4 * sizeof(double));
    ymm0 = _mm256_sub_pd(ymm0, ymm1);
    ymm0 = _mm256_and_pd(ymm0, ymm3);
    ymm1 = _mm256_cmp_pd(ymm0, ymm2, _CMP_GE_OQ);
    if ((cmp_masks = _mm256_movemask_pd(ymm1))) {

      if (cmp_masks & 1)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", i % cols, i / cols, *it1, *it2);
      if (cmp_masks & 2)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", (i + 1) % cols, (i + 1) / cols,
               *(it1 + 1), *(it2 + 1));
      if (cmp_masks & 4)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", (i + 2) % cols, (i + 2) / cols,
               *(it1 + 2), *(it2 + 2));
      if (cmp_masks & 8)
        printf("matrix 1 and 2 do not coincide in (%lld, %lld): %.3f, %.3f\n", (i + 3) % cols, (i + 3) / cols,
               *(it1 + 3), *(it2 + 3));
      res = false;
    }
  }

  return res;
}

#define __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)                                                        \
      ymm4 = _mm256_unpackhi_pd(ymm[0], ymm[1]);   /** ymm4: a1, b1, a3, b3 */                    \
      ymm[0] = _mm256_unpacklo_pd(ymm[0], ymm[1]); /** ymm0: a0, b0, a2, b2 */                    \
      ymm[1] = _mm256_unpacklo_pd(ymm[2], ymm[3]); /** ymm1: c0, d0, c2, d2 */                    \
      ymm[2] = _mm256_unpackhi_pd(ymm[2], ymm[3]); /** ymm2: c1, d1, c3, d3 */                    \
                                                                                                  \
      ymm[3] = _mm256_permute2f128_pd(ymm4, ymm[2], 0b00110001);   /** ymm3: a3, b3, c3, d3 */    \
      ymm4 = _mm256_permute2f128_pd(ymm4, ymm[2], 0b00100000);     /** ymm4: a1, b1, c1, d1 */    \
      ymm[2] = _mm256_permute2f128_pd(ymm[0], ymm[1], 0b00110001); /** ymm2: a2, b2, c2, d2 */    \
      ymm[0] = _mm256_permute2f128_pd(ymm[0], ymm[1], 0b00100000); /** ymm0: a0, b0, c0, d0 */    \
      ymm[1] = ymm4;                                               /** ymm1: a1, b1, c1, d1 */

static void __mat_transpose_avx2_4x8_impl(
  const double *mat,
  size_t nrows,
  size_t ncols,
  double *mat_res
) {
  const size_t nrows_f4 = nrows & ~3;
  const size_t nrows_r4 = nrows & 3;
  const size_t nrows_res_cb  = nrows_r4 * sizeof(double);
  const size_t ncols_f8 = ncols & ~7;
  const size_t ncols_r8 = ncols & 7;
  const size_t ncols_r4 = ncols & 3;
  const size_t ncols_res_cb2 = ncols_r4 * sizeof(double);
  ptrdiff_t i = 0;
  ptrdiff_t j;

  __m256d ymm[4], ymmN[4];
  __m256d ymm4;
  const __m256i bidx_base = _mm256_set_epi64x(3*nrows, 2*nrows, nrows, 0);
  const __m256i bidx_inc4 = _mm256_set1_epi64x(4 * nrows);
  const __m256i irows_inc4row = _mm256_set1_epi64x(3 * ncols + ncols_r8);
  __m256i bidx;
  __m256i irows = _mm256_set_epi64x(ncols*3, ncols*2, ncols*1, 0);
  __m256i irows2 = _mm256_add_epi64(irows, _mm256_set1_epi64x(4));
  __m256i irows_inc8 = _mm256_set1_epi64x(8);

  for (; i < nrows_f4; i += 4) {
    bidx = _mm256_add_epi64(bidx_base, _mm256_set1_epi64x(i));
    for(j = 0; j < ncols_f8; j += 8) {

      ymm[0]  = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
      ymmN[0] = _mm256_loadu_pd(mat + irows2.m256i_u64[0]);

      ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
      ymmN[1] = _mm256_loadu_pd(mat + irows2.m256i_u64[1]);

      ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
      ymmN[2] = _mm256_loadu_pd(mat + irows2.m256i_u64[2]);

      ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);
      ymmN[3] = _mm256_loadu_pd(mat + irows2.m256i_u64[3]);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)
      __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

      _mm256_storeu_pd(mat_res + bidx.m256i_u64[0], ymm[0]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[1], ymm[1]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[2], ymm[2]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[3], ymm[3]);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      _mm256_storeu_pd(mat_res + bidx.m256i_u64[0], ymmN[0]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[1], ymmN[1]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[2], ymmN[2]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[3], ymmN[3]);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      irows = _mm256_add_epi64(irows, irows_inc8);
      irows2 = _mm256_add_epi64(irows2, irows_inc8);
    }

    if (ncols_r8 > 4) {
      ymm[0] = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
      memcpy(&ymmN[0], mat + irows2.m256i_u64[0], ncols_res_cb2);
      ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
      memcpy(&ymmN[1], mat + irows2.m256i_u64[1], ncols_res_cb2);
      ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
      memcpy(&ymmN[2], mat + irows2.m256i_u64[2], ncols_res_cb2);
      ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);
      memcpy(&ymmN[3], mat + irows2.m256i_u64[3], ncols_res_cb2);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)
      __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

      _mm256_storeu_pd(mat_res + bidx.m256i_u64[0], ymm[0]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[1], ymm[1]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[2], ymm[2]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[3], ymm[3]);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      for(ptrdiff_t k = 0; k < ncols_r4; ++k) {
        _mm256_storeu_pd(mat_res + bidx.m256i_u64[k], ymmN[k]);
      }
    } else if(ncols_r8 == 4) {
      ymm[0] = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
      ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
      ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
      ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      _mm256_storeu_pd(mat_res + bidx.m256i_u64[0], ymm[0]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[1], ymm[1]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[2], ymm[2]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[3], ymm[3]);
    } else if (ncols_r8 > 0) {
      memcpy(&ymm[0], mat + irows.m256i_u64[0], ncols_res_cb2);
      memcpy(&ymm[1], mat + irows.m256i_u64[1], ncols_res_cb2);
      memcpy(&ymm[2], mat + irows.m256i_u64[2], ncols_res_cb2);
      memcpy(&ymm[3], mat + irows.m256i_u64[3], ncols_res_cb2);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        _mm256_storeu_pd(mat_res + bidx.m256i_u64[k], ymm[k]);
    }

    irows = _mm256_add_epi64(irows, irows_inc4row);
    irows2 = _mm256_add_epi64(irows2, irows_inc4row);
  }

  if(nrows_r4) {
    bidx = _mm256_add_epi64(bidx_base, _mm256_set1_epi64x(i));
    for (j = 0; j < ncols_f8; j += 8) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        ymm[k] = _mm256_loadu_pd(mat + irows.m256i_u64[k]);
        ymmN[k] = _mm256_loadu_pd(mat + irows2.m256i_u64[k]);
      }

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)
      __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

      memcpy(mat_res + bidx.m256i_u64[0], &ymm[0], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[1], &ymm[1], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[2], &ymm[2], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[3], &ymm[3], nrows_res_cb);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      memcpy(mat_res + bidx.m256i_u64[0], &ymmN[0], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[1], &ymmN[1], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[2], &ymmN[2], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[3], &ymmN[3], nrows_res_cb);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      irows = _mm256_add_epi64(irows, irows_inc8);
      irows2 = _mm256_add_epi64(irows2, irows_inc8);
    }

    if (ncols_r8 > 4) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        ymm[k] = _mm256_loadu_pd(mat + irows.m256i_u64[k]);
        memcpy(&ymmN[k], mat + irows2.m256i_u64[k], ncols_res_cb2);
      }

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)
      __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

      memcpy(mat_res + bidx.m256i_u64[0], &ymm[0], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[1], &ymm[1], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[2], &ymm[2], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[3], &ymm[3], nrows_res_cb);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        memcpy(mat_res + bidx.m256i_u64[k], &ymmN[k], nrows_res_cb);
    } else if (ncols_r8 == 4) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        ymm[k] = _mm256_loadu_pd(mat + irows.m256i_u64[k]);
      }

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      memcpy(mat_res + bidx.m256i_u64[0], &ymm[0], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[1], &ymm[1], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[2], &ymm[2], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[3], &ymm[3], nrows_res_cb);
    } else if (ncols_r8 > 0) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        memcpy(&ymm[k], mat + irows.m256i_u64[k], ncols_res_cb2);
      }

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        memcpy(mat_res + bidx.m256i_u64[k], &ymm[k], nrows_res_cb);
    }
  }
}

static void __mat_transpose_avx2_4x4_impl(
  const double *mat,
  size_t nrows,
  size_t ncols,
  double *mat_res
) {
  const size_t nrows_f4 = nrows & ~3;
  const size_t nrows_r4 = nrows & 3;
  const size_t nrows_res_cb = nrows_r4 * sizeof(double);
  const size_t ncols_f4 = ncols & ~3;
  const size_t ncols_r4 = ncols & 3;
  const size_t ncols_res_cb = ncols_r4 * sizeof(double);
  ptrdiff_t i, j;

  __m256d ymm[4];
  __m256d ymm4;
  const __m256i bidx_base = _mm256_set_epi64x(3 * nrows, 2 * nrows, nrows, 0);
  const __m256i bidx_inc4 = _mm256_set1_epi64x(4 * nrows);
  const __m256i irows_inc4row = _mm256_set1_epi64x(3 * ncols + ncols_r4);
  __m256i bidx;
  __m256i irows = _mm256_set_epi64x(ncols * 3, ncols * 2, ncols * 1, 0);
  __m256i irows_inc4 = _mm256_set1_epi64x(4);

  for (i = 0; i < nrows_f4; i += 4) {
    bidx = _mm256_add_epi64(bidx_base, _mm256_set1_epi64x(i));
    for (j = 0; j < ncols_f4; j += 4) {

      ymm[0] = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
      ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
      ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
      ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      _mm256_storeu_pd(mat_res + bidx.m256i_u64[0], ymm[0]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[1], ymm[1]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[2], ymm[2]);
      _mm256_storeu_pd(mat_res + bidx.m256i_u64[3], ymm[3]);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      irows = _mm256_add_epi64(irows, irows_inc4);
    }

    if (ncols_r4 > 0) {
      memcpy(&ymm[0], mat + irows.m256i_u64[0], ncols_res_cb);
      memcpy(&ymm[1], mat + irows.m256i_u64[1], ncols_res_cb);
      memcpy(&ymm[2], mat + irows.m256i_u64[2], ncols_res_cb);
      memcpy(&ymm[3], mat + irows.m256i_u64[3], ncols_res_cb);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        _mm256_storeu_pd(mat_res + bidx.m256i_u64[k], ymm[k]);
    }

    irows = _mm256_add_epi64(irows, irows_inc4row);
  }

  if (nrows_r4) {
    bidx = _mm256_add_epi64(bidx_base, _mm256_set1_epi64x(i));
    for (j = 0; j < ncols_f4; j += 4) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        ymm[k] = _mm256_loadu_pd(mat + irows.m256i_u64[k]);
      }

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      memcpy(mat_res + bidx.m256i_u64[0], &ymm[0], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[1], &ymm[1], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[2], &ymm[2], nrows_res_cb);
      memcpy(mat_res + bidx.m256i_u64[3], &ymm[3], nrows_res_cb);
      bidx = _mm256_add_epi64(bidx, bidx_inc4);

      irows = _mm256_add_epi64(irows, irows_inc4);
    }

    if (ncols_r4 > 0) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        memcpy(&ymm[k], mat + irows.m256i_u64[k], ncols_res_cb);
      }

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        memcpy(mat_res + bidx.m256i_u64[k], &ymm[k], nrows_res_cb);
    }
  }
}

static void __mat_transpose_avx2_4x4_impl2(
  const double *mat,
  size_t nrows,
  size_t ncols,
  double *mat_res
) {
  const size_t nrows_f4 = nrows & ~3;
  const size_t nrows_r4 = nrows & 3;
  const size_t nrows_res_cb = nrows_r4 * sizeof(double);
  const size_t ncols_f4 = ncols & ~3;
  const size_t ncols_r4 = ncols & 3;
  const size_t ncols_res_cb = ncols_r4 * sizeof(double);
  ptrdiff_t i, j;

  __m256d ymm[4];
  __m256d ymm4;

#define _LL_DOUBLE_PTR(m, index) (double *)m.m256i_u64[index]
#define _LL_DOUBLE_CPTR(m, index) (const double *)m.m256i_u64[index]

  __m256i mat_v = _mm256_set_epi64x((long long)(mat + 3*ncols), (long long)(mat + 2*ncols),
    (long long)(mat + ncols), (long long)mat);
  __m256i mat_res_v_base = _mm256_set_epi64x((long long)(mat_res + 3 * nrows), (long long)(mat_res + 2 * nrows),
                                             (long long)(mat_res + nrows), (long long)mat_res);
  __m256i mat_res_v;
  __m256i mat_v_offset4 = _mm256_set1_epi64x(32);
  __m256i mat_v_offset4row = _mm256_set1_epi64x(8*(3 * ncols + ncols_r4));
  __m256i mat_res_v_offset4row = _mm256_set1_epi64x(32 * nrows);

  for (i = 0; i < nrows_f4; i += 4) {
    mat_res_v = _mm256_add_epi64(mat_res_v_base, _mm256_set1_epi64x(i << 3));
    for (j = 0; j < ncols_f4; j += 4) {
      ymm[0] = _mm256_loadu_pd(_LL_DOUBLE_CPTR(mat_v, 0));
      ymm[1] = _mm256_loadu_pd(_LL_DOUBLE_CPTR(mat_v, 1));
      ymm[2] = _mm256_loadu_pd(_LL_DOUBLE_CPTR(mat_v, 2));
      ymm[3] = _mm256_loadu_pd(_LL_DOUBLE_CPTR(mat_v, 3));
      mat_v = _mm256_add_epi64(mat_v, mat_v_offset4);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      _mm256_storeu_pd(_LL_DOUBLE_PTR(mat_res_v, 0), ymm[0]);
      _mm256_storeu_pd(_LL_DOUBLE_PTR(mat_res_v, 1), ymm[1]);
      _mm256_storeu_pd(_LL_DOUBLE_PTR(mat_res_v, 2), ymm[2]);
      _mm256_storeu_pd(_LL_DOUBLE_PTR(mat_res_v, 3), ymm[3]);
      mat_res_v = _mm256_add_epi64(mat_res_v, mat_res_v_offset4row);
    }

    if (ncols_r4 > 0) {
      memcpy(&ymm[0], _LL_DOUBLE_CPTR(mat_v, 0), ncols_res_cb);
      memcpy(&ymm[1], _LL_DOUBLE_CPTR(mat_v, 1), ncols_res_cb);
      memcpy(&ymm[2], _LL_DOUBLE_CPTR(mat_v, 2), ncols_res_cb);
      memcpy(&ymm[3], _LL_DOUBLE_CPTR(mat_v, 3), ncols_res_cb);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        _mm256_storeu_pd(_LL_DOUBLE_PTR(mat_res_v, k), ymm[k]);
    }

    mat_v = _mm256_add_epi64(mat_v, mat_v_offset4row);
  }

  if (nrows_r4) {
    mat_res_v = _mm256_add_epi64(mat_res_v_base, _mm256_set1_epi64x(i << 3));
    for (j = 0; j < ncols_f4; j += 4) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        ymm[k] = _mm256_loadu_pd(_LL_DOUBLE_CPTR(mat_v, k));
      }
      mat_v = _mm256_add_epi64(mat_v, mat_v_offset4);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      memcpy(_LL_DOUBLE_PTR(mat_res_v, 0), &ymm[0], nrows_res_cb);
      memcpy(_LL_DOUBLE_PTR(mat_res_v, 1), &ymm[1], nrows_res_cb);
      memcpy(_LL_DOUBLE_PTR(mat_res_v, 2), &ymm[2], nrows_res_cb);
      memcpy(_LL_DOUBLE_PTR(mat_res_v, 3), &ymm[3], nrows_res_cb);
      mat_res_v = _mm256_add_epi64(mat_res_v, mat_res_v_offset4row);
    }

    if (ncols_r4 > 0) {
      for (ptrdiff_t k = 0; k < nrows_r4; ++k) {
        memcpy(&ymm[k], _LL_DOUBLE_CPTR(mat_v, k), ncols_res_cb);
      }

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        memcpy(_LL_DOUBLE_PTR(mat_res_v, k), &ymm[k], nrows_res_cb);
    }
  }

#undef _LL_DOUBLE_PTR
#undef _LL_DOUBLE_CPTR
}

template<size_t ...I, typename = std::enable_if_t<sizeof...(I) <= 16>>
static void __mxm_avx2_unroll_impl(
  const double *mat1,
  const double *mat2,
  size_t M,
  size_t K,
  size_t N,
  double *mat_res,
  const double *aux_res_row_buffers,
  std::index_sequence<I...>
) {
  constexpr size_t UnrollRowSize = sizeof...(I);
  size_t M_r4 = M % UnrollRowSize;
  size_t M_f4 = M - M_r4;
  size_t K_f4 = K & ~3;
  size_t K_r4 = K & 3;
  size_t K_r4_cb = K_r4 * sizeof(double);
  size_t N_f4 = N & ~3;
  size_t N_r4 = N & 3;
  size_t N_r4_cb = N_r4 * sizeof(double);
  size_t N_cpd4 = (N + 3) >> 2;
  ptrdiff_t i, j, jj, k, k_pd4;
  size_t mat1_v_inc = UnrollRowSize * K;
  size_t mat_res_v_inc = UnrollRowSize * N;

  const double *mat1_v[UnrollRowSize];
  const double *mat2_v;
  double *mat_res_v[UnrollRowSize];

  const __m256d ymm_zero = _mm256_setzero_pd();
  __m256d ymm1[UnrollRowSize];
  __m256d ymm2;

  __AVX2_ALIGNED double mat1_block[UnrollRowSize][4];

  __m256d *ymm_res[UnrollRowSize];

  if(aux_res_row_buffers)
    ymm_res[0] = (__m256d *)aux_res_row_buffers;
  else
    ymm_res[0] = (__m256d *)_aligned_malloc(N_cpd4 * 32 * UnrollRowSize, 32);

  ((ymm_res[I] = ymm_res[0] + I * N_cpd4),...);

  ((mat1_v[I] = mat1 + I*K),...);
  ((mat_res_v[I] = mat_res + I*N),...);

  for(i = 0; i < M_f4; i += UnrollRowSize) {
    for(j = 0; j < N_cpd4; ++j) {
      ((ymm_res[I][j] = ymm_zero),...);
    }

    mat2_v = mat2;
    for(j = 0; j < K_f4; j += 4) {
      ((*(__m256d *)(mat1_block[I]) = _mm256_loadu_pd(mat1_v[I] + j)),...);

      for(jj = 0; jj < 4; ++jj) {

        ((ymm1[I] = _mm256_broadcast_sd(mat1_block[I] + jj)),...);

        for(k = 0, k_pd4 = 0; k < N_f4; k += 4, k_pd4 += 1) {
          ymm2 = _mm256_loadu_pd(mat2_v + k);
          ((ymm_res[I][k_pd4] = _mm256_fmadd_pd(ymm1[I], ymm2, ymm_res[I][k_pd4])), ...);
        }

        if(N_r4) {
          memcpy(&ymm2, mat2_v + k, N_r4_cb);
          ((ymm_res[I][k_pd4] = _mm256_fmadd_pd(ymm1[I], ymm2, ymm_res[I][k_pd4])), ...);
        }

        mat2_v += N;
      }
    }

    if(K_r4) {
      ((memcpy(mat1_block[I], mat1_v[I] + j, K_r4_cb)),...);

      for (jj = 0; jj < K_r4; ++jj) {

        ((ymm1[I] = _mm256_broadcast_sd(mat1_block[I] + jj)),...);

        for(k = 0, k_pd4 = 0; k < N_f4; k += 4, k_pd4 += 1) {
          ymm2 = _mm256_loadu_pd(mat2_v + k);
          ((ymm_res[I][k_pd4] = _mm256_fmadd_pd(ymm1[I], ymm2, ymm_res[I][k_pd4])), ...);
        }

        if(N_r4) {
          memcpy(&ymm2, mat2_v + k, N_r4_cb);
          ((ymm_res[I][k_pd4] = _mm256_fmadd_pd(ymm1[I], ymm2, ymm_res[I][k_pd4])), ...);
        }

        mat2_v += N;
      }
    }

    for(k = 0, k_pd4 = 0; k < N_f4; k += 4, k_pd4 += 1) {
      ((_mm256_storeu_pd(mat_res_v[I] + k, ymm_res[I][k_pd4])), ...);
    }

    if(N_r4)
      ((memcpy(mat_res_v[I] + k, ymm_res[I] + k_pd4, N_r4_cb)), ...);

    ((mat1_v[I] += mat1_v_inc),...);
    ((mat_res_v[I] += mat_res_v_inc),...);
  }

  if(M_r4) {
    mat1 = mat1_v[0];
    mat_res = mat_res_v[0];
    M = M_r4;

    if(M_r4 >= 8)
      __mxm_avx2_unroll_impl(mat1, mat2, M, K, N, mat_res, (double *)ymm_res[0], std::make_index_sequence<8>());
    else if(M_r4 >= 4)
      __mxm_avx2_unroll_impl(mat1, mat2, M, K, N, mat_res, (double *)ymm_res[0], std::make_index_sequence<4>());
    else if(M_r4 >= 2)
      __mxm_avx2_unroll_impl(mat1, mat2, M, K, N, mat_res, (double *)ymm_res[0], std::make_index_sequence<2>());
    else if(M_r4)
      __mxm_avx2_unroll_impl(mat1, mat2, M, K, N, mat_res, (double *)ymm_res[0], std::make_index_sequence<1>());
  }

  if(!aux_res_row_buffers)
    _aligned_free(ymm_res[0]);
}

template<size_t ...I>
static void __mxv_avx2_fma_unroll_impl(
  const double *mat,
  const double *vec,
  size_t nrows,
  size_t ncols,
  size_t row_pitch,
  double *res,
  std::index_sequence<I...>
) {
  constexpr size_t BX = sizeof...(I);
  const size_t nrows_f = (nrows / BX) * BX;
  const size_t ncols_f = ncols & ~3;
  const size_t ncols_res_cb = (ncols & 3) * sizeof(double);
  const size_t mat_v_offset4row = BX * row_pitch;
  ptrdiff_t i, j;

  const double *mat_v[BX];
  const __m256d ymm_zero = _mm256_setzero_pd();
  __m256d re[BX], mat_res_buff, mv, vec_res_buff;

  ((mat_v[I] = mat + I * row_pitch),...);

  if(((uintptr_t)vec & 32)) {
    for (i = 0; i < nrows_f; i += BX) {

      ((re[I] = ymm_zero),...);

      for(j = 0; j < ncols_f; j += 4) {
        mv = _mm256_loadu_pd(vec + j);
        ((re[I] = _mm256_fmadd_pd(_mm256_loadu_pd(mat_v[I] + j), mv, re[I])), ...);
      }

      if(ncols_res_cb != 0) {
        vec_res_buff = ymm_zero;
        mat_res_buff = ymm_zero;
        memcpy(&vec_res_buff, vec + j, ncols_res_cb);
        ((memcpy(&mat_res_buff, mat_v[I] + j, ncols_res_cb),
          re[I] = _mm256_fmadd_pd(mat_res_buff, vec_res_buff, re[I])),
        ...);
      }

      ((re[I] = _mm256_hadd_pd(re[I], re[I]), re[I] = _mm256_permute4x64_pd(re[I], 0b00001000),
        re[I] = _mm256_hadd_pd(re[I], re[I]), res[i + I] = re[I].m256d_f64[0]),
      ...);

      ((mat_v[I] += mat_v_offset4row), ...);
    }

    for (; i < nrows; ++i) {

      re[0] = ymm_zero;

      for (j = 0; j < ncols_f; j += 4) {
        mv = _mm256_loadu_pd(vec + j);
        re[0] = _mm256_fmadd_pd(_mm256_loadu_pd(mat_v[0] + j), mv, re[0]);
      }
      if(ncols_res_cb != 0) {
        vec_res_buff = ymm_zero;
        mat_res_buff = ymm_zero;
        memcpy(&vec_res_buff, vec + j, ncols_res_cb);
        memcpy(&mat_res_buff, mat_v[0] + j, ncols_res_cb);
        re[0] = _mm256_fmadd_pd(mat_res_buff, vec_res_buff, re[0]);
      }

      re[0] = _mm256_hadd_pd(re[0], re[0]);
      re[0] = _mm256_permute4x64_pd(re[0], 0b00001000);
      re[0] = _mm256_hadd_pd(re[0], re[0]);
      res[i] = re[0].m256d_f64[0];

      mat_v[0] += row_pitch;
    }
  } else {
    for (i = 0; i < nrows_f; i += BX) {

      ((re[I] = ymm_zero), ...);

      for (j = 0; j < ncols_f; j += 4) {
        mv = _mm256_load_pd(vec + j);
        ((re[I] = _mm256_fmadd_pd(_mm256_loadu_pd(mat_v[I] + j), mv, re[I])), ...);
      }

      if (ncols_res_cb != 0) {
        vec_res_buff = ymm_zero;
        mat_res_buff = ymm_zero;
        memcpy(&vec_res_buff, vec + j, ncols_res_cb);
        ((memcpy(&mat_res_buff, mat_v[I] + j, ncols_res_cb),
          re[I] = _mm256_fmadd_pd(mat_res_buff, vec_res_buff, re[I])),
         ...);
      }

      ((re[I] = _mm256_hadd_pd(re[I], re[I]), re[I] = _mm256_permute4x64_pd(re[I], 0b00001000),
        re[I] = _mm256_hadd_pd(re[I], re[I]), res[i + I] = re[I].m256d_f64[0]),
       ...);

      ((mat_v[I] += mat_v_offset4row), ...);
    }

    for (; i < nrows; ++i) {

      re[0] = ymm_zero;

      for (j = 0; j < ncols_f; j += 4) {
        mv = _mm256_load_pd(vec + j);
        re[0] = _mm256_fmadd_pd(_mm256_loadu_pd(mat_v[0] + j), mv, re[0]);
      }
      if (ncols_res_cb != 0) {
        vec_res_buff = ymm_zero;
        mat_res_buff = ymm_zero;
        memcpy(&vec_res_buff, vec + j, ncols_res_cb);
        memcpy(&mat_res_buff, mat_v[0] + j, ncols_res_cb);
        re[0] = _mm256_fmadd_pd(mat_res_buff, vec_res_buff, re[0]);
      }

      re[0] = _mm256_hadd_pd(re[0], re[0]);
      re[0] = _mm256_permute4x64_pd(re[0], 0b00001000);
      re[0] = _mm256_hadd_pd(re[0], re[0]);
      res[i] = re[0].m256d_f64[0];

      mat_v[0] += row_pitch;
    }
  }
}

template<size_t BX>
void mxv_avx2_fma_unroll(
  const double *mat,
  const double *vec,
  size_t nrows,
  size_t ncols,
  size_t row_pitch,
  double *res
) {
  return __mxv_avx2_fma_unroll_impl(mat, vec, nrows, ncols, row_pitch, res, std::make_index_sequence<BX>{});
}

void mat_transpose_avx2_4x8_unroll(
  const double *mat,
  size_t nrows,
  size_t ncols,
  double *mat_res
) {
  return __mat_transpose_avx2_4x8_impl(mat,nrows, ncols, mat_res);
}

void mat_transpose_avx2_4x4_unroll(const double *mat, size_t nrows, size_t ncols, double *mat_res) {
  return __mat_transpose_avx2_4x4_impl(mat, nrows, ncols, mat_res);
}

void mat_transpose_avx2_4x4_unroll2(const double *mat, size_t nrows, size_t ncols, double *mat_res) {
  return __mat_transpose_avx2_4x4_impl2(mat, nrows, ncols, mat_res);
}

template<size_t UnrollRowSize>
void mxm_avx2_unroll(
  const double *mat1,
  const double *mat2,
  size_t M,
  size_t K,
  size_t N,
  double *mat_res
) {
  return __mxm_avx2_unroll_impl(mat1, mat2, M, K, N, mat_res, nullptr, std::make_index_sequence<UnrollRowSize>());
}

static ycl_program g_pMatrixProgram;
static ycl_program g_pMatMuplVecProgram;

CLHRESULT TestMatrixTransposeProfile(
    cl_context context, cl_device_id device, cl_command_queue cmd_queue, size_t ncols, size_t nrows) {

  CLHRESULT hr;

  // nrows = nrows & ~3;
  // ncols = ncols & ~7;

  const size_t mat_buff_size = ncols * nrows * sizeof(double);

  auto test_mat = gen_random_matrix<double>(ncols, nrows);
  printf("Input matrix dimensions: (rows: %lld, cols: %lld)\n", nrows, ncols);

  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  ycl_buffer input_mat_buff, output_mat_buff;

  V_RETURN((input_mat_buff <<=
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_buff_size, test_mat.data(), &hr),
            hr));
  V_RETURN((output_mat_buff <<= clCreateBuffer(context, CL_MEM_WRITE_ONLY, mat_buff_size, nullptr, &hr), hr));

  std::vector<double> test_mat2(test_mat.size());

  start = hp_timer::now();
  for(ptrdiff_t i = 0; i < nrows; ++i) {
    for(ptrdiff_t j = 0; j < ncols; ++j) {
      ptrdiff_t ij = i * ncols + j;
      ptrdiff_t ji = j * nrows + i;

      test_mat2[ji] = test_mat[ij];
    }
  }
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU seiralizing) elapsed:                         %.3fms\n", elapsed.count());

  std::vector<double> test_mat3(test_mat.size());
  start = hp_timer::now();
  mat_transpose_avx2_4x8_unroll(test_mat.data(), nrows, ncols, (double *)test_mat3.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU AVX2 blocking 4x8) elapsed:                   %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat3, test_mat2, 1.0E-6, nrows, ncols) ? "true" : "false");

  start = hp_timer::now();
  mat_transpose_avx2_4x4_unroll(test_mat.data(), nrows, ncols, (double *)test_mat3.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU AVX2 blocking 4x4 implementation 1) elapsed:  %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat3, test_mat2, 1.0E-6, nrows, ncols) ? "true" : "false");

  start = hp_timer::now();
  mat_transpose_avx2_4x4_unroll2(test_mat.data(), nrows, ncols, (double *)test_mat3.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU AVX2 blocking 4x4 implementation 2) elapsed:  %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat3, test_mat2, 1.0E-6, nrows, ncols) ? "true" : "false");

  ycl_kernel ker;
  V_RETURN((ker <<= clCreateKernel(g_pMatrixProgram, "mat_transpose", &hr), hr));

  int M = static_cast<int>(ncols), N = static_cast<int>(nrows);
  V_RETURN(SetKernelArguments(ker, &input_mat_buff, &output_mat_buff, &M, &N));

  size_t lworksize[3];
  V_RETURN(
      clGetKernelWorkGroupInfo(ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(lworksize), lworksize, nullptr));

  size_t gworksize[] = {RoundC(ncols, lworksize[0]), RoundC(nrows, lworksize[1])};
  ycl_event rd_done_ev;

  size_t max_work_size[3];
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_size), max_work_size, nullptr));

  gworksize[0] = std::min(gworksize[0], max_work_size[0]);
  gworksize[1] = std::min(gworksize[0], max_work_size[1]);

  start = hp_timer::now();
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker, 2, nullptr, gworksize, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, output_mat_buff, false, 0, mat_buff_size, (void *)test_mat.data(), 0, nullptr,
                               &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(Use only global storage) elapsed:                 %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n",  check_matrix_equiv(test_mat, test_mat2, 1.0E-6, nrows, ncols) ? "true" : "false");

  ycl_kernel ker_opt1;
  V_RETURN((ker_opt1 <<= clCreateKernel(g_pMatrixProgram, "mat_transpose_opt1", &hr), hr));

  V_RETURN(SetKernelArguments(ker_opt1, &input_mat_buff, &output_mat_buff, &M, &N));
  double data_pattern = 0.0;
  V_RETURN(clEnqueueFillBuffer(cmd_queue, output_mat_buff, &data_pattern, sizeof(data_pattern), 0,
                               mat_buff_size / sizeof(data_pattern), 0, nullptr, nullptr));
  start = hp_timer::now();
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker_opt1, 2, nullptr, gworksize, nullptr, 0, nullptr, nullptr));
  rd_done_ev = nullptr;
  V_RETURN(clEnqueueReadBuffer(cmd_queue, output_mat_buff, false, 0, mat_buff_size, (void *)test_mat.data(), 0, nullptr,
                               &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(Use shared local storage, 64bits bank) elapsed:   %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat, test_mat2, 1.0E-6, nrows, ncols) ? "true" : "false");

  ycl_kernel ker_opt2;
  V_RETURN((ker_opt2 <<= clCreateKernel(g_pMatrixProgram, "mat_transpose_opt2", &hr), hr));

  V_RETURN(SetKernelArguments(ker_opt2, &input_mat_buff, &output_mat_buff, &M, &N));
  V_RETURN(clEnqueueFillBuffer(cmd_queue, output_mat_buff, &data_pattern, sizeof(data_pattern), 0,
                               mat_buff_size / sizeof(data_pattern), 0, nullptr, nullptr));
  start = hp_timer::now();
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker_opt2, 2, nullptr, gworksize, nullptr, 0, nullptr, nullptr));
  rd_done_ev = nullptr;
  V_RETURN(clEnqueueReadBuffer(cmd_queue, output_mat_buff, false, 0, mat_buff_size, (void *)test_mat.data(), 0, nullptr,
                               &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(Eliminate bank conflict, 32bits bank) elapsed:    %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat, test_mat2, 1.0E-6, nrows, ncols) ? "true" : "false");

  return hr;
}

CLHRESULT TestMatrixMulitplicationProfile(
  cl_context context, cl_device_id device, cl_command_queue cmd_queue, size_t M, size_t K, size_t N) {

  CLHRESULT hr;
  ycl_kernel mul_ker;
  ycl_buffer a_buffer, b_buffer, c_buffer;

  printf("Input matrix A dimensions: (%llu, %llu)\n", M, K);
  printf("Input matrix B dimensions: (%llu, %llu)\n", K, N);

  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  const size_t a_data_size = M * K;
  const size_t b_data_size = K * N;
  const size_t c_data_size = M * N;
  const size_t a_buffer_size = a_data_size * sizeof(double);
  const size_t b_buffer_size = b_data_size * sizeof(double);
  const size_t c_buffer_size = c_data_size * sizeof(double);
  std::vector<double> a_data, b_data, c_data, c_data2;
  const double dbl_eq_tol = 1.0E-8;

  a_data = gen_random_matrix<double>(K, M);
  b_data = gen_random_matrix<double>(N, K);
  c_data2.resize(c_data_size);

  c_data.resize(c_data_size);
  start = hp_timer::now();
  for(ptrdiff_t i = 0; i < M; ++i) {
    ptrdiff_t ii = i*K;
    for(ptrdiff_t j = 0; j < N; ++j) {
      double c = 0.0;
      for(ptrdiff_t k = 0; k < K; ++k) {
        c += a_data[ii+k]*b_data[k*N+j];
      }
      c_data[i*N+j] = c;
    }
  }
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Matrix multiplication(CPU serializing) elapsed:                    %.3fms\n", elapsed.count());

  start = hp_timer::now();
  mxm_avx2_unroll<16>(a_data.data(), b_data.data(), M, K, N, (double *)c_data2.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Matrix multiplication(CPU AVX2+FMA Unroll 16 rows) elapsed:        %.3fms\n", elapsed.count());
  printf("Results coincedence: %s\n", check_matrix_equiv(c_data2, c_data, 1.0E-6, N, M) ? "true" : "false");

  start = hp_timer::now();
  mxm_avx2_unroll<8>(a_data.data(), b_data.data(), M, K, N, (double *)c_data2.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Matrix multiplication(CPU AVX2+FMA Unroll 8 rows) elapsed:         %.3fms\n", elapsed.count());
  printf("Results coincedence: %s\n", check_matrix_equiv(c_data2, c_data, 1.0E-6, N, M) ? "true" : "false");

  start = hp_timer::now();
  mxm_avx2_unroll<4>(a_data.data(), b_data.data(), M, K, N, (double *)c_data2.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Matrix multiplication(CPU AVX2+FMA Unroll 4 rows) elapsed:         %.3fms\n", elapsed.count());
  printf("Results coincedence: %s\n", check_matrix_equiv(c_data2, c_data, 1.0E-6, N, M) ? "true" : "false");

  V_RETURN((a_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        a_buffer_size, a_data.data(), &hr),
            hr));
  V_RETURN((b_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        b_buffer_size, b_data.data(), &hr),
            hr));
  V_RETURN(
      (c_buffer <<= clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_buffer_size, nullptr, &hr),
       hr));

  // Free some deprecated memory.
  a_data = std::vector<double>();
  b_data = a_data;

  V_RETURN((mul_ker <<= clCreateKernel(g_pMatrixProgram, "mat_mul", &hr), hr));

  size_t group_size[3];
  size_t global_size[2];
  const double dbl_zero = 0.0;

  V_RETURN(clGetKernelWorkGroupInfo(mul_ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(group_size), group_size, nullptr));
  global_size[0] = RoundC(N, group_size[0]);
  global_size[1] = RoundC(M, group_size[1]);

  std::array<uint32_t, 4> MKN{ (uint32_t)M, (uint32_t)K, (uint32_t)N };
  V_RETURN(SetKernelArguments(mul_ker, &a_buffer, &b_buffer, &c_buffer, &MKN));

  ycl_event rd_done_ev;

  start = hp_timer::now();

  V_RETURN(clEnqueueFillBuffer(cmd_queue, c_buffer, &dbl_zero, sizeof(dbl_zero), 0, c_data_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, mul_ker, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, c_buffer, false, 0, c_buffer_size, (void *)c_data2.data(), 0, nullptr, &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));

  V_RETURN(clWaitForEvents(1, &rd_done_ev));

  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Matrix multiplication(Use global storage only) elapsed:            %.3fms\n",elapsed.count());
  printf("Results coincedence: %s\n", check_matrix_equiv(c_data2, c_data, 1.0E-6, N, M) ? "true" : "false");

  V_RETURN((mul_ker <<= clCreateKernel(g_pMatrixProgram, "mat_mul_opt1", &hr), hr));
  V_RETURN(SetKernelArguments(mul_ker, &a_buffer, &b_buffer, &c_buffer, &MKN));

  start = hp_timer::now();

  V_RETURN(clEnqueueFillBuffer(cmd_queue, c_buffer, &dbl_zero, sizeof(dbl_zero), 0, c_data_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, mul_ker, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, c_buffer, false, 0, c_buffer_size, (void *)c_data2.data(), 0, nullptr,
                               rd_done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));

  V_RETURN(clWaitForEvents(1, &rd_done_ev));

  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Matrix multiplication(Use shared local storage) elapsed:           %.3fms\n", elapsed.count());
  printf("Results coincedence: %s\n", check_matrix_equiv(c_data2, c_data, 1.0E-6, N, M) ? "true" : "false");

  return hr;
}

CLHRESULT TestMatMulVecProfile(
    cl_context context, cl_device_id device, cl_command_queue cmd_queue, size_t mat_rows, size_t mat_cols, size_t mat_pitch) {

  CLHRESULT hr;

  std::vector<double> mat_data, vec_data;
  size_t mat_data_bsize = mat_pitch * mat_rows * sizeof(double);
  size_t vec_data_bsize = mat_cols * sizeof(double);
  size_t res_data_bsize = mat_rows * sizeof(double);
  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  mat_data = gen_random_matrix<double>(mat_pitch, mat_rows);
  vec_data = gen_random_matrix<double>(1, mat_cols);

  std::vector<double> res_data2;
  res_data2.resize(mat_rows);

  start = hp_timer::now();
  for (ptrdiff_t i = 0; i < mat_rows; ++i) {
    ptrdiff_t ii = i * mat_pitch;
    double temp = 0.0;
    for (ptrdiff_t j = 0; j < mat_cols; ++j) {
      temp += mat_data[ii + j] * vec_data[j];
    }
    res_data2[i] = temp;
  }
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("matrix [%lld x %lld] multipling vector [%lld x 1]  (CPU) elapsed:                            %.3fms\n",
         mat_rows, mat_cols, mat_cols, elapsed.count());

  std::vector<double> res_data(mat_rows);

  start = hp_timer::now();
  mxv_avx2_fma_unroll<3>(mat_data.data(), vec_data.data(), mat_rows, mat_cols, mat_pitch, (double *)res_data.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("matrix [%lld x %lld] multipling vector [%lld x 1]  (CPU, AVX2+FMA, unroll 3 rows) elapsed:   %.3fms\n",
         mat_rows, mat_cols, mat_cols, elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(res_data, res_data2, 1.0E-6, mat_rows, 1) ? "true" : "false");
  start = hp_timer::now();
  mxv_avx2_fma_unroll<4>(mat_data.data(), vec_data.data(), mat_rows, mat_cols, mat_pitch, (double *)res_data2.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("matrix [%lld x %lld] multipling vector [%lld x 1]  (CPU, AVX2+FMA, unroll 4 rows) elapsed:   %.3fms\n",
         mat_rows, mat_cols, mat_cols, elapsed.count());
  start = hp_timer::now();
  mxv_avx2_fma_unroll<5>(mat_data.data(), vec_data.data(), mat_rows, mat_cols, mat_pitch, (double *)res_data2.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("matrix [%lld x %lld] multipling vector [%lld x 1]  (CPU, AVX2+FMA, unroll 5 rows) elapsed:   %.3fms\n",
         mat_rows, mat_cols, mat_cols, elapsed.count());
  start = hp_timer::now();
  mxv_avx2_fma_unroll<7>(mat_data.data(), vec_data.data(), mat_rows, mat_cols, mat_pitch, (double *)res_data2.data());
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("matrix [%lld x %lld] multipling vector [%lld x 1]  (CPU, AVX2+FMA, unroll 7 rows) elapsed:   %.3fms\n", mat_rows,
         mat_cols, mat_cols, elapsed.count());

  ycl_buffer mat_buffer, vec_buffer, res_buffer;
  cl_uint row_size = (cl_uint)mat_rows,
          col_size = (cl_uint)mat_cols,
          pitch_size = (cl_uint)mat_pitch;

  V_RETURN2(mat_buffer <<=
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_data_bsize, mat_data.data(), &hr),
            hr);
  V_RETURN2(vec_buffer <<=
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, vec_data_bsize, vec_data.data(), &hr),
            hr);
  V_RETURN2(res_buffer <<=
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY, res_data_bsize, nullptr, &hr),
            hr);

  ycl_kernel kernel;
  size_t group_size[3];
  size_t max_work_item_size[3];
  size_t work_item_size[2];
  ycl_event done_ev;
  const double zero_pattern = 0.0;

  V_RETURN(
      clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_size), max_work_item_size, nullptr));

  V_RETURN2(kernel <<= clCreateKernel(g_pMatrixProgram, "mxv_block", &hr), hr);
  V_RETURN(SetKernelArguments(kernel, &mat_buffer, &vec_buffer, &row_size, &col_size, &pitch_size, &res_buffer));
  V_RETURN(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(group_size), group_size, nullptr));

  work_item_size[0] = RoundC(mat_cols, group_size[0]);
  work_item_size[0] = std::min(work_item_size[0], max_work_item_size[0]);
  work_item_size[0] = RoundF(work_item_size[0], group_size[0]);

  start = hp_timer::now();
  V_RETURN(clEnqueueFillBuffer(cmd_queue, res_buffer, &zero_pattern, sizeof(zero_pattern), 0, res_data_bsize, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, work_item_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, res_buffer, false, 0, res_data_bsize, (void *)res_data.data(), 0, nullptr,
                               done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("matrix [%lld x %lld] multipling vector [%lld x 1]  (One Row Per Block) elaped:               %.3fms\n", mat_rows, mat_cols, mat_cols,
    elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(res_data, res_data2, 1.0E-6, mat_rows, 1) ? "true" : "false");

  V_RETURN2(kernel <<= clCreateKernel(g_pMatMuplVecProgram, "mxv_warp", &hr),
            hr);
  V_RETURN(SetKernelArguments(kernel, &mat_buffer, &vec_buffer, &row_size, &col_size, &pitch_size, &res_buffer));
  V_RETURN(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(group_size), group_size,
                                    nullptr));

  work_item_size[0] = RoundC(mat_rows / group_size[1], group_size[0]);
  work_item_size[0] = std::min(work_item_size[0], max_work_item_size[0]);
  work_item_size[1] = RoundC(1, group_size[1]);

  start = hp_timer::now();
  V_RETURN(clEnqueueFillBuffer(cmd_queue, res_buffer, &zero_pattern, sizeof(zero_pattern), 0, res_data_bsize, 0,
                               nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, kernel, 2, nullptr, work_item_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, res_buffer, false, 0, res_data_bsize, (void *)res_data.data(), 0, nullptr,
                               done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("matrix [%lld x %lld] multipling vector [%lld x 1]  (One Raw Per Warp) elapsed:               %.3fms\n", mat_rows, mat_cols,
         mat_cols, elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(res_data, res_data2, 1.0E-6, mat_rows, 1) ? "true" : "false");

  return hr;
}

int main() {

  CLHRESULT hr;
  cl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;

  V_RETURN(FindOpenCLPlatform(CL_DEVICE_TYPE_GPU, {"NVIDIA CUDA", "AMD"}, {}, &platform, &device));
  V_RETURN(CreateDeviceContext(platform, device, &context));
  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

  V_RETURN(CreateProgramFromILFile(context, device, "OCL-SpirV/matrix.spv", &g_pMatrixProgram));
  g_pMatMuplVecProgram = g_pMatrixProgram;

  std::uniform_int_distribution<size_t> transpose_ncols_nrows_distr(16, 5000);

  for(ptrdiff_t i = 0; i < 100; ++i) {
    printf("Matrix Transpose Profile [%lld]:\n", i);
    TestMatrixTransposeProfile(context, device, cmd_queue, transpose_ncols_nrows_distr(g_RandomEngine),
                               transpose_ncols_nrows_distr(g_RandomEngine));
    printf("\n");
  }

  std::uniform_int_distribution<size_t> mul_ncols_nrows_distr(510, 2000);

  for(ptrdiff_t i = 0; i < 20; ++i) {
    printf("Matrix Multiplication Profile [%lld]:\n", i);
    TestMatrixMulitplicationProfile(context, device, cmd_queue, mul_ncols_nrows_distr(g_RandomEngine),
                                    mul_ncols_nrows_distr(g_RandomEngine), mul_ncols_nrows_distr(g_RandomEngine));
    printf("\n");
  }

  std::uniform_int_distribution<size_t> mxv_ncols_distr(510, 5000);

  for(ptrdiff_t i = 0; i < 40; ++i) {

    size_t mat_rows = mxv_ncols_distr(g_RandomEngine);
    size_t mat_cols = mxv_ncols_distr(g_RandomEngine);
    size_t mat_pitch = mxv_ncols_distr(g_RandomEngine);

    printf("Matrix-Vector Multiplication Profile [%lld]:\n", i);
    if(mat_cols > mat_pitch)
      std::swap(mat_cols, mat_pitch);

    TestMatMulVecProfile(context, device, cmd_queue, mat_rows, mat_cols, mat_pitch);
    printf("\n");
  }

  return hr;
}