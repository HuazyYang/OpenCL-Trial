#include <cl_utils.h>
#include <common_miscs.h>
#include <vector>
#include<intrin.h>

template <class REAL> static std::vector<REAL> gen_random_matrix(size_t cols, size_t rows) {

  std::uniform_real_distribution<REAL> rd{(REAL)-100.0, (REAL)100.0};

  size_t mat_len = cols * rows, i;
  std::vector<double> mat(mat_len);
  for (i = 0; i < mat_len; ++i) {
    mat[i] = rd(g_RandomEngine);
  }

  return mat;
}

bool check_matrix_equiv(
    const std::vector<double> &mat1, const std::vector<double> &mat2, double eq_tol, size_t cols, size_t rows) {

  bool res = true;

  // Verify the results.
  if (mat1.size() != mat2.size()) {
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

  for (; it1 != it1end; it1 += 4, it2 += 4, i += 4) {
    ymm0 = _mm256_loadu_pd(it1);
    ymm1 = _mm256_loadu_pd(it2);
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

  if (count_r4) {
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

void __transpose_square_matrix_avx2_4x4_impl(double *mat, size_t m) {

  if(m <= 1)
    return;

  const size_t ncols_f4 = m & ~3;
  const size_t ncols_r4 = m & 3;
  const size_t ncols_res_cb = ncols_r4 * sizeof(double);
  ptrdiff_t i, j;

  __m256d ymm[4], ymmN[4];
  __m256d ymm4;

  const __m256i inc_alias       = _mm256_set_epi64x(m * 3, m * 2, m, 0);
  const __m256i inc_4row        = _mm256_set1_epi64x(4 * m);
  const __m256i inc_next_4row   = _mm256_set1_epi64x(3 * m + ncols_r4);
  const __m256i inc_4sc         = _mm256_set1_epi64x(4);
  __m256i irows = inc_alias;
  __m256i bidx;

  for (i = 0; i < ncols_f4; i += 4) {

    irows = _mm256_add_epi64(irows, _mm256_set1_epi64x(i));

    ymm[0] = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
    ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
    ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
    ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);

    __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

    _mm256_storeu_pd(mat + irows.m256i_u64[0], ymm[0]);
    _mm256_storeu_pd(mat + irows.m256i_u64[1], ymm[1]);
    _mm256_storeu_pd(mat + irows.m256i_u64[2], ymm[2]);
    _mm256_storeu_pd(mat + irows.m256i_u64[3], ymm[3]);

    irows = _mm256_add_epi64(irows, inc_4sc);
    bidx = _mm256_add_epi64(inc_alias, _mm256_set1_epi64x((i + 4) * m + i));

    for (j = i + 4; j < ncols_f4; j += 4) {

      ymm[0] = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
      ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
      ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
      ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      ymmN[0] = _mm256_loadu_pd(mat + bidx.m256i_u64[0]);
      ymmN[1] = _mm256_loadu_pd(mat + bidx.m256i_u64[1]);
      ymmN[2] = _mm256_loadu_pd(mat + bidx.m256i_u64[2]);
      ymmN[3] = _mm256_loadu_pd(mat + bidx.m256i_u64[3]);

      __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

      _mm256_storeu_pd(mat + irows.m256i_u64[0], ymmN[0]);
      _mm256_storeu_pd(mat + irows.m256i_u64[1], ymmN[1]);
      _mm256_storeu_pd(mat + irows.m256i_u64[2], ymmN[2]);
      _mm256_storeu_pd(mat + irows.m256i_u64[3], ymmN[3]);

      _mm256_storeu_pd(mat + bidx.m256i_u64[0], ymm[0]);
      _mm256_storeu_pd(mat + bidx.m256i_u64[1], ymm[1]);
      _mm256_storeu_pd(mat + bidx.m256i_u64[2], ymm[2]);
      _mm256_storeu_pd(mat + bidx.m256i_u64[3], ymm[3]);

      bidx = _mm256_add_epi64(bidx, inc_4row);
      irows = _mm256_add_epi64(irows, inc_4sc);
    }

    if (ncols_r4 > 0) {
      memcpy(&ymm[0], mat + irows.m256i_u64[0], ncols_res_cb);
      memcpy(&ymm[1], mat + irows.m256i_u64[1], ncols_res_cb);
      memcpy(&ymm[2], mat + irows.m256i_u64[2], ncols_res_cb);
      memcpy(&ymm[3], mat + irows.m256i_u64[3], ncols_res_cb);

      __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

      for(ptrdiff_t k = 0; k < ncols_r4; ++k)
        ymmN[k] = _mm256_loadu_pd(mat + bidx.m256i_u64[k]);

      __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

      memcpy(mat + irows.m256i_u64[0], &ymmN[0], ncols_res_cb);
      memcpy(mat + irows.m256i_u64[1], &ymmN[1], ncols_res_cb);
      memcpy(mat + irows.m256i_u64[2], &ymmN[2], ncols_res_cb);
      memcpy(mat + irows.m256i_u64[3], &ymmN[3], ncols_res_cb);

      for (ptrdiff_t k = 0; k < ncols_r4; ++k)
        _mm256_storeu_pd(mat + bidx.m256i_u64[k], ymm[k]);
    }

    irows = _mm256_add_epi64(irows, inc_next_4row);
  }

  if (ncols_r4) {
    irows = _mm256_add_epi64(irows, _mm256_set1_epi64x(i));

    for (ptrdiff_t k = 0; k < ncols_r4; ++k) {
      memcpy(&ymm[k], mat + irows.m256i_u64[k], ncols_res_cb);
    }

    __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

    for (ptrdiff_t k = 0; k < ncols_r4; ++k)
      memcpy(mat + irows.m256i_u64[k], &ymm[k], ncols_res_cb);
  }
}

void __matrix_tranpose_inplace(double *mat, size_t m, size_t n) {

  if (m < 1 || n < 1)
    return;

  size_t count = m * n - 1;

  for (ptrdiff_t i = 1; i < count; ++i) {
    ptrdiff_t idx = i;
    do {
      idx = (idx * m) % count;
      // idx = (idx % n) * m + idx / n;
    } while (idx > i);

    if (i != idx)
      std::swap(mat[i], mat[idx]);
  }
}

CLHRESULT TestSquareMatrixTransposeInplace(
  size_t ncols
) {

  auto test_mat = gen_random_matrix<double>(ncols, ncols);
  printf("Input matrix dimensions: (rows: %lld, cols: %lld)\n", ncols, ncols);

  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  std::vector<double> test_mat2(test_mat);

  start = hp_timer::now();
  for (ptrdiff_t i = 0; i < ncols; ++i) {
    for (ptrdiff_t j = i + 1; j < ncols; ++j) {
      ptrdiff_t ij = i * ncols + j;
      ptrdiff_t ji = j * ncols + i;

      std::swap(test_mat2[ji], test_mat2[ij]);
    }
  }
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU seiralizing) elapsed:                         %.3fms\n", elapsed.count());

  start = hp_timer::now();
  __transpose_square_matrix_avx2_4x4_impl((double *)test_mat.data(), ncols);
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU AVX2 unroll 4x4) eplased:                     %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat, test_mat2, 1.0E-6, ncols, ncols) ? "true" : "false");

  return 0;
}

CLHRESULT TestMatrixTransposeInplaceProfile(
    cl_context context, cl_device_id device, cl_command_queue cmd_queue, size_t ncols, size_t nrows) {

  auto test_mat = gen_random_matrix<double>(ncols, nrows);
  printf("Input matrix dimensions: (rows: %lld, cols: %lld)\n", nrows, ncols);

  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  std::vector<double> test_mat2(test_mat.size());

  start = hp_timer::now();
  for (ptrdiff_t i = 0; i < nrows; ++i) {
    for (ptrdiff_t j = 0; j < ncols; ++j) {
      ptrdiff_t ij = i * ncols + j;
      ptrdiff_t ji = j * nrows + i;

      test_mat2[ji] = test_mat[ij];
    }
  }
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU seiralizing) elapsed:                         %.3fms\n", elapsed.count());

  __matrix_tranpose_inplace((double *)test_mat.data(), nrows, ncols);
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat, test_mat2, 1.0E-6, nrows, ncols) ? "true" : "false");

  return 0;
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

  std::uniform_int_distribution<size_t> transpose_ncols_nrows_distr(16, 5000);

  for (ptrdiff_t i = 0; i < 100; ++i) {
    printf("Matrix Transpose Profile [%lld]:\n", i);
    TestSquareMatrixTransposeInplace(transpose_ncols_nrows_distr(g_RandomEngine));

    // TestMatrixTransposeInplaceProfile(context, device, cmd_queue, transpose_ncols_nrows_distr(g_RandomEngine),
    //                                   transpose_ncols_nrows_distr(g_RandomEngine));
    printf("\n");
  }
}
