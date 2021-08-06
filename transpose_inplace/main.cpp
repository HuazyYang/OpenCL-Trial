#include <cl_utils.h>
#include <common_miscs.h>
#include <vector>
#include<intrin.h>
#include "native_transpose.h"
#include "openmp_transpose.h"
#include <omp.h>
#include "page_allocator.h"

#define MAX_MAT_ROW_COL_SIZE        10000

template <class REAL>
static void gen_random_matrix(size_t cols, size_t rows, REAL *buffer) {

  std::uniform_real_distribution<REAL> rd{(REAL)-100.0, (REAL)100.0};

  size_t mat_len = cols * rows, i;
  for (i = 0; i < mat_len; ++i) {
    buffer[i] = rd(g_RandomEngine);
  }
}

bool check_matrix_equiv(
    const double *mat1, const double *mat2, double eq_tol, size_t cols, size_t rows) {

  bool res = true;
  size_t count = cols * rows;

  size_t count_f4 = count & ~3;
  size_t count_r4 = count & 3;

  auto it1 = mat1, it1end = it1 + count_f4;
  auto it2 = mat2;
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


// #define __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)                                                        \
//       ymm4 = _mm256_unpackhi_pd(ymm[0], ymm[1]);   /** ymm4: a1, b1, a3, b3 */                    \
//       ymm[0] = _mm256_unpacklo_pd(ymm[0], ymm[1]); /** ymm0: a0, b0, a2, b2 */                    \
//       ymm[1] = _mm256_unpacklo_pd(ymm[2], ymm[3]); /** ymm1: c0, d0, c2, d2 */                    \
//       ymm[2] = _mm256_unpackhi_pd(ymm[2], ymm[3]); /** ymm2: c1, d1, c3, d3 */                    \
//                                                                                                   \
//       ymm[3] = _mm256_permute2f128_pd(ymm4, ymm[2], 0b00110001);   /** ymm3: a3, b3, c3, d3 */    \
//       ymm4 = _mm256_permute2f128_pd(ymm4, ymm[2], 0b00100000);     /** ymm4: a1, b1, c1, d1 */    \
//       ymm[2] = _mm256_permute2f128_pd(ymm[0], ymm[1], 0b00110001); /** ymm2: a2, b2, c2, d2 */    \
//       ymm[0] = _mm256_permute2f128_pd(ymm[0], ymm[1], 0b00100000); /** ymm0: a0, b0, c0, d0 */    \
//       ymm[1] = ymm4;                                               /** ymm1: a1, b1, c1, d1 */

// void __transpose_square_matrix_avx2_4x4_impl(double *mat, size_t m) {

//   if(m <= 1)
//     return;

//   const size_t ncols_f4 = m & ~3;
//   const size_t ncols_r4 = m & 3;
//   const size_t ncols_res_cb = ncols_r4 * sizeof(double);
//   ptrdiff_t i, j;

//   __m256d ymm[4], ymmN[4];
//   __m256d ymm4;

//   const __m256i inc_alias       = _mm256_set_epi64x(m * 3, m * 2, m, 0);
//   const __m256i inc_4row        = _mm256_set1_epi64x(4 * m);
//   const __m256i inc_next_4row   = _mm256_set1_epi64x(3 * m + ncols_r4);
//   const __m256i inc_4sc         = _mm256_set1_epi64x(4);
//   __m256i irows = inc_alias;
//   __m256i bidx;

//   for (i = 0; i < ncols_f4; i += 4) {

//     irows = _mm256_add_epi64(irows, _mm256_set1_epi64x(i));

//     ymm[0] = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
//     ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
//     ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
//     ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);

//     __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

//     _mm256_storeu_pd(mat + irows.m256i_u64[0], ymm[0]);
//     _mm256_storeu_pd(mat + irows.m256i_u64[1], ymm[1]);
//     _mm256_storeu_pd(mat + irows.m256i_u64[2], ymm[2]);
//     _mm256_storeu_pd(mat + irows.m256i_u64[3], ymm[3]);

//     irows = _mm256_add_epi64(irows, inc_4sc);
//     bidx = _mm256_add_epi64(inc_alias, _mm256_set1_epi64x((i + 4) * m + i));

//     for (j = i + 4; j < ncols_f4; j += 4) {

//       ymm[0] = _mm256_loadu_pd(mat + irows.m256i_u64[0]);
//       ymm[1] = _mm256_loadu_pd(mat + irows.m256i_u64[1]);
//       ymm[2] = _mm256_loadu_pd(mat + irows.m256i_u64[2]);
//       ymm[3] = _mm256_loadu_pd(mat + irows.m256i_u64[3]);

//       __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

//       ymmN[0] = _mm256_loadu_pd(mat + bidx.m256i_u64[0]);
//       ymmN[1] = _mm256_loadu_pd(mat + bidx.m256i_u64[1]);
//       ymmN[2] = _mm256_loadu_pd(mat + bidx.m256i_u64[2]);
//       ymmN[3] = _mm256_loadu_pd(mat + bidx.m256i_u64[3]);

//       __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

//       _mm256_storeu_pd(mat + irows.m256i_u64[0], ymmN[0]);
//       _mm256_storeu_pd(mat + irows.m256i_u64[1], ymmN[1]);
//       _mm256_storeu_pd(mat + irows.m256i_u64[2], ymmN[2]);
//       _mm256_storeu_pd(mat + irows.m256i_u64[3], ymmN[3]);

//       _mm256_storeu_pd(mat + bidx.m256i_u64[0], ymm[0]);
//       _mm256_storeu_pd(mat + bidx.m256i_u64[1], ymm[1]);
//       _mm256_storeu_pd(mat + bidx.m256i_u64[2], ymm[2]);
//       _mm256_storeu_pd(mat + bidx.m256i_u64[3], ymm[3]);

//       bidx = _mm256_add_epi64(bidx, inc_4row);
//       irows = _mm256_add_epi64(irows, inc_4sc);
//     }

//     if (ncols_r4 > 0) {
//       memcpy(&ymm[0], mat + irows.m256i_u64[0], ncols_res_cb);
//       memcpy(&ymm[1], mat + irows.m256i_u64[1], ncols_res_cb);
//       memcpy(&ymm[2], mat + irows.m256i_u64[2], ncols_res_cb);
//       memcpy(&ymm[3], mat + irows.m256i_u64[3], ncols_res_cb);

//       __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

//       for(ptrdiff_t k = 0; k < ncols_r4; ++k)
//         ymmN[k] = _mm256_loadu_pd(mat + bidx.m256i_u64[k]);

//       __AVX2_MAT4X4_TRANSPOSE(ymmN, ymm4)

//       memcpy(mat + irows.m256i_u64[0], &ymmN[0], ncols_res_cb);
//       memcpy(mat + irows.m256i_u64[1], &ymmN[1], ncols_res_cb);
//       memcpy(mat + irows.m256i_u64[2], &ymmN[2], ncols_res_cb);
//       memcpy(mat + irows.m256i_u64[3], &ymmN[3], ncols_res_cb);

//       for (ptrdiff_t k = 0; k < ncols_r4; ++k)
//         _mm256_storeu_pd(mat + bidx.m256i_u64[k], ymm[k]);
//     }

//     irows = _mm256_add_epi64(irows, inc_next_4row);
//   }

//   if (ncols_r4) {
//     irows = _mm256_add_epi64(irows, _mm256_set1_epi64x(i));

//     for (ptrdiff_t k = 0; k < ncols_r4; ++k) {
//       memcpy(&ymm[k], mat + irows.m256i_u64[k], ncols_res_cb);
//     }

//     __AVX2_MAT4X4_TRANSPOSE(ymm, ymm4)

//     for (ptrdiff_t k = 0; k < ncols_r4; ++k)
//       memcpy(mat + irows.m256i_u64[k], &ymm[k], ncols_res_cb);
//   }
// }

template<typename T>
void mat_tr_inplace_native(bool row_major, T *mat, size_t m, size_t n) {

  if (m < 1 || n < 1)
    return;

  if(!row_major)
    std::swap(m, n);

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
 

CLHRESULT TestMatrixTransposeInplaceProfile(
    cl_context context, cl_device_id device, cl_command_queue cmd_queue, double *mat_buffer, size_t ncols, size_t nrows) {

  double *test_mat, *test_mat2, *tmp_buffer;
  size_t mat_buffer_cb = nrows * ncols * sizeof(double);

  test_mat  =  mat_buffer;
  test_mat2  = mat_buffer + (MAX_MAT_ROW_COL_SIZE * MAX_MAT_ROW_COL_SIZE);
  tmp_buffer = test_mat2 + (MAX_MAT_ROW_COL_SIZE * MAX_MAT_ROW_COL_SIZE);

  gen_random_matrix<double>(ncols, nrows, mat_buffer);
  printf("Input matrix dimensions: (rows: %lld, cols: %lld)\n", nrows, ncols);

  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  memcpy(test_mat2, test_mat, mat_buffer_cb);
  start = hp_timer::now();
  mat_tr_inplace_native(true, test_mat2, nrows, ncols);
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("Transpose matrix(CPU seiralizing) elapsed:                         %.3fms\n", elapsed.count());

  start = hp_timer::now();
  tr_inplace::native::transpose(false, test_mat2, nrows, ncols, tmp_buffer);
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("C2R native implementation elapsed:                                 %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat2, test_mat, 1.0E-6, ncols, nrows) ? "true" : "false");

  mat_tr_inplace_native(true, test_mat2, nrows, ncols);

  start = hp_timer::now();
  tr_inplace::openmp::transpose(false, test_mat2, nrows, ncols, tmp_buffer);
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("C2R OpenMP implementation elapsed:                                 %.3fms\n", elapsed.count());
  printf("results coincidence: %s\n", check_matrix_equiv(test_mat2, test_mat, 1.0E-6, ncols, nrows) ? "true" : "false");

  return 0;
}

int main() {

  CLHRESULT hr;
  ycl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;

  V_RETURN(FindOpenCLPlatform(CL_DEVICE_TYPE_GPU, {"NVIDIA CUDA", "AMD"}, {}, &platform, &device));
  V_RETURN(CreateDeviceContext(platform, device, &context));
  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

  std::uniform_int_distribution<size_t> transpose_ncols_nrows_distr(
      16, MAX_MAT_ROW_COL_SIZE);

  double *mat_buffer = (double *)inplace::test::large_page_alloc(
      (MAX_MAT_ROW_COL_SIZE * MAX_MAT_ROW_COL_SIZE * 2 + MAX_MAT_ROW_COL_SIZE * omp_get_max_threads())*sizeof(double));
  if(!mat_buffer)
    V_RETURN2("Large page allocation failed, system error code(win32...): ", hr = inplace::test::get_last_error());

  for (ptrdiff_t i = 0; i < 100; ++i) {
    printf("Matrix Transpose In-place Profile [%lld]:\n", i);
    TestMatrixTransposeInplaceProfile(context, device, cmd_queue, mat_buffer, transpose_ncols_nrows_distr(g_RandomEngine),
                                      transpose_ncols_nrows_distr(g_RandomEngine));
    printf("\n");
  }

  inplace::test::large_page_dealloc(mat_buffer);
}
