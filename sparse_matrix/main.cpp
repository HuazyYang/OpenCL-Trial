#include <cl_utils.h>
#include <common_miscs.h>
#include <cstdint>
#include <algorithm>
#include <immintrin.h>

#define _DEFAULT_INIT(p) memset((p), 0, sizeof(*(p)))

#define _SAFE_DELETE_ARRAY(p) \
  do { if((p)) delete []p; p = nullptr; } while(0)

struct csr_mat {
  uint16_t rows;
  uint16_t cols;
  uint32_t *row_ptr;
  uint16_t *col_idx;
  double *vals;
};
#define CSR_MAT_INIT { 0, 0, nullptr, nullptr, nullptr }

void csr_mat_init(csr_mat *mat) { _DEFAULT_INIT(mat); }
void csr_mat_destroy(csr_mat *mat) {
  mat->rows = 0;
  mat->cols = 0;
  _SAFE_DELETE_ARRAY(mat->row_ptr);
  _SAFE_DELETE_ARRAY(mat->col_idx);
  _SAFE_DELETE_ARRAY(mat->vals);
}
void csr_mat_alloc(csr_mat *mat, uint16_t rows, uint16_t cols, uint32_t nnz) {
  csr_mat_destroy(mat);
  mat->rows = rows;
  mat->cols = cols;
  mat->row_ptr = new uint32_t[rows+1];
  mat->row_ptr[rows] = nnz;
  mat->col_idx = new uint16_t[nnz];
  mat->vals = new double[nnz];
}

struct raw_vector {
  uint16_t rows;
  double *vals;
};
#define RAW_VECTOR_INIT { 0, nullptr }

void raw_vector_init(raw_vector *vec) { _DEFAULT_INIT(vec);  }
void raw_vector_destroy(raw_vector *vec) {
  vec->rows = 0;
  _SAFE_DELETE_ARRAY(vec->vals);
}

void raw_vector_alloc(raw_vector *vec, uint16_t rows) {
  raw_vector_destroy(vec);

  vec->rows = rows;
  vec->vals = new double[rows];
}

void generate_random_csr_matrix(uint16_t rows, uint16_t cols, double fmin, double fmax, csr_mat *mat) {

  static std::minstd_rand zero_rd(g_RandomEngine());
  constexpr size_t zero_id_denom = 701;
  constexpr size_t zero_id_denom_half = zero_id_denom >> 1;

  uint32_t nnz = 0;
  std::uniform_int_distribution<uint16_t> cols_distr(1, cols);
  std::uniform_real_distribution<double> val_distr(fmin, fmax+1.0E6);
  uint16_t *col_range = new uint16_t[cols];
  uint32_t row_nnz;

  csr_mat_destroy(mat);

  mat->rows = rows;
  mat->cols = cols;

  mat->row_ptr = new uint32_t[rows+1];
  nnz = 0;
  mat->row_ptr[0] = 0;
  for(uint16_t i = 0; i < rows; ++i) {
    nnz += cols_distr(g_RandomEngine);
    mat->row_ptr[i+1] = nnz;
  }

  mat->col_idx = new uint16_t[nnz];
  mat->vals = new double[nnz];

  for(uint16_t i =0; i < cols; ++i)
    col_range[i] = i;

  std::shuffle(col_range, col_range + cols, zero_rd);
  for(uint16_t i = 0; i < rows; ++i) {
    row_nnz = mat->row_ptr[i+1] - mat->row_ptr[i];
    std::sort(col_range, col_range + row_nnz);
    for(uint32_t j = 0, k = mat->row_ptr[i]; j < row_nnz; ++j, ++k) {
      mat->col_idx[k] = col_range[j];
      mat->vals[k] = val_distr(g_RandomEngine);
    }
    std::shuffle(col_range, col_range+cols, zero_rd);
  }

  _SAFE_DELETE_ARRAY(col_range);
}

void generate_random_vector(uint16_t rows, double fmin, double fmax, raw_vector *vec) {

  std::uniform_real_distribution<double> fd(fmin, fmax+1.0E-6);

  raw_vector_alloc(vec, rows);

  for(uint16_t i=0; i < rows; ++i) {
    vec->vals[i] = fd(g_RandomEngine);
  }
}

int csr_mat_mul_vec(const csr_mat *mat, const raw_vector *vec, raw_vector *res) {

  if(mat->cols != vec->rows) {
    printf("csr_mat_mul_vec: Incompatible matrix and vector dimension.");
    return -1;
  }

  raw_vector_alloc(res, mat->rows);

  for(uint16_t i = 0; i < mat->rows; ++i) {
    uint32_t col_start = mat->row_ptr[i];
    uint32_t col_end = mat->row_ptr[i+1];
    double temp = 0.0;

    for(; col_start != col_end; ++col_start) {
      temp += mat->vals[col_start] * vec->vals[mat->col_idx[col_start]];
    }

    res->vals[i] = temp;
  }
  return 0;
}

bool check_matrix_equiv(const double *mat1, const double *mat2, size_t count, double eq_tol, size_t cols, size_t rows) {

  bool res = true;
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

static ycl_program g_pSparseMatrixProgram;

CLHRESULT TestCsrMatMulVec(cl_context context, cl_device_id device, cl_command_queue cmd_queue, uint16_t nrows, uint16_t ncols) {

  CLHRESULT hr;

  csr_mat mat = CSR_MAT_INIT;
  raw_vector vec = RAW_VECTOR_INIT;
  raw_vector res = RAW_VECTOR_INIT;

  printf("Input Matrix size: [%u X %u]\n", nrows, ncols);

  generate_random_csr_matrix(nrows, ncols, -10.0, 10.0, &mat);
  generate_random_vector(ncols, -10.0, 10.0, &vec);

  csr_mat_mul_vec(&mat, &vec, &res);

  size_t mat_row_ptr_buff_size = (mat.rows + 1)* sizeof(uint32_t);
  size_t mat_col_idx_buff_size = mat.row_ptr[mat.rows] * sizeof(uint16_t);
  size_t mat_vals_buff_size = mat.row_ptr[mat.rows] * sizeof(double);
  size_t vec_vals_buff_size = vec.rows * sizeof(double);
  size_t res_vals_buff_size = mat.rows * sizeof(double);

  ycl_buffer mat_row_ptr_buffer, mat_col_idx_buffer, mat_vals_buffer;
  ycl_buffer vec_vals_buffer;
  ycl_buffer res_vals_buffer;

  V_RETURN2(mat_row_ptr_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
    mat_row_ptr_buff_size, mat.row_ptr, &hr), hr);
  V_RETURN2(mat_col_idx_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
    mat_col_idx_buff_size, mat.col_idx, &hr), hr);
  V_RETURN2(mat_vals_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
    mat_vals_buff_size, mat.vals, &hr), hr);
  V_RETURN2(vec_vals_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
    vec_vals_buff_size, vec.vals, &hr), hr);
  V_RETURN2(res_vals_buffer <<= clCreateBuffer(context, CL_MEM_WRITE_ONLY|CL_MEM_HOST_READ_ONLY,
    res_vals_buff_size, nullptr, &hr), hr);

  raw_vector res2 = RAW_VECTOR_INIT;
  raw_vector_alloc(&res2, mat.rows);

  ycl_kernel kernel;
  ycl_event done_ev;
  cl_uint row_size = mat.rows;
  size_t max_work_item_size[3];
  size_t work_group_size[3];
  size_t work_item_size[2];
  const double zfpattern = 0.0;

  V_RETURN(
      clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_size), max_work_item_size, nullptr));

  V_RETURN2(kernel <<= clCreateKernel(g_pSparseMatrixProgram, "smm_native", &hr), hr);
  V_RETURN(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(work_group_size), work_group_size, nullptr));

  work_item_size[0] = std::min(max_work_item_size[0], (size_t)mat.rows);
  work_item_size[0] = RoundC(work_item_size[0], work_group_size[0]);

  V_RETURN(SetKernelArguments(kernel, &row_size, &mat_row_ptr_buffer, &mat_col_idx_buffer, &mat_vals_buffer,
                              &vec_vals_buffer, &res_vals_buffer));
  V_RETURN(clEnqueueFillBuffer(cmd_queue, res_vals_buffer, &zfpattern, sizeof(zfpattern), 0, res_vals_buff_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, work_item_size, work_group_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, res_vals_buffer, false, 0, res_vals_buff_size, res2.vals, 0, nullptr,
                               done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));

  printf("Results coincidence: %s\n", check_matrix_equiv(res2.vals, res.vals, res.rows, 1.0E-6, 1, res.rows) ? "true" : "false");

  V_RETURN2(kernel <<= clCreateKernel(g_pSparseMatrixProgram, "smm_warp_per_row", &hr), hr);
  V_RETURN(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(work_group_size),
                                    work_group_size, nullptr));

  work_item_size[0] = std::min(max_work_item_size[0], (size_t)mat.rows);
  work_item_size[1] = work_group_size[1];

  V_RETURN(SetKernelArguments(kernel, &row_size, &mat_row_ptr_buffer, &mat_col_idx_buffer, &mat_vals_buffer,
                              &vec_vals_buffer, &res_vals_buffer));
  V_RETURN(clEnqueueFillBuffer(cmd_queue, res_vals_buffer, &zfpattern, sizeof(zfpattern), 0, res_vals_buff_size, 0,
                               nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, kernel, 2, nullptr, work_item_size, work_group_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, res_vals_buffer, false, 0, res_vals_buff_size, res2.vals, 0, nullptr,
                               done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));

  printf("Results coincidence: %s\n",
         check_matrix_equiv(res2.vals, res.vals, res.rows, 1.0E-5, 1, res.rows) ? "true" : "false");

  csr_mat_destroy(&mat);
  raw_vector_destroy(&vec);
  raw_vector_destroy(&res);
  raw_vector_destroy(&res2);
  return hr;
}

int main() {

  CLHRESULT hr;
  const char *preferred_plats[] = {"NVIDIA", "AMD", nullptr};
  ycl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;

  V_RETURN(FindOpenCLPlatform(preferred_plats, CL_DEVICE_TYPE_GPU, &platform));
  V_RETURN(CreateDeviceContext(platform, CL_DEVICE_TYPE_GPU, &device, &context));
  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

  V_RETURN(CreateProgramFromFile(context, device, "#define _USE_DOUBLE_FP", "sparse_matrix.cl", &g_pSparseMatrixProgram));

  std::uniform_int_distribution<uint16_t> mat_nrows_distr(16, 10000), mat_ncols_distr(16, 10000);
  TestCsrMatMulVec(context, device, cmd_queue, mat_nrows_distr(g_RandomEngine), mat_ncols_distr(g_RandomEngine));
}