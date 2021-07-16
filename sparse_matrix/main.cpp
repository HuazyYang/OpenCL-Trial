#include <cl_utils.h>
#include <common_miscs.h>
#include <cstdint>
#include <algorithm>

#define _DEFAULT_INIT(p) memset((p), 0, sizeof(*(p)))

#define _SAFE_DELETE_ARRAY(p) \
  do { if((p)) delete []p; p = nullptr; } while(0)

struct csr_mat {
  uint16_t rows;
  uint16_t cols;
  uint16_t *row_ptr;
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
  mat->row_ptr = new uint16_t[rows+1];
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
  std::uniform_int_distribution<uint16_t> cols_distr(1, cols+1);
  std::uniform_real_distribution<double> val_distr(fmin, fmax+1.0E6);
  uint16_t *col_range = new uint16_t[cols];
  uint16_t row_nnz;

  csr_mat_destroy(mat);

  mat->rows = rows;
  mat->cols = cols;

  mat->row_ptr = new uint16_t[rows+1];
  nnz = 0;
  mat->row_ptr[0] = 0;
  for(uint16_t i = 1; i <= rows; ++i) {
    nnz += cols_distr(g_RandomEngine);
    mat->row_ptr[i] = nnz;
  }

  mat->col_idx = new uint16_t[nnz];
  mat->vals = new double[nnz];

  for(uint16_t i =0; i < cols; ++i)
    col_range[i] = i;

  std::shuffle(col_range, col_range + cols, zero_rd);
  for(uint16_t i = 0; i < rows; ++i) {
    row_nnz = mat->row_ptr[i+1] - mat->row_ptr[i];
    std::sort(col_range, col_range + row_nnz);
    for(uint16_t j = 0, k = mat->row_ptr[i]; j < row_nnz; ++j, ++k) {
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
    uint16_t col_start = mat->row_ptr[i];
    uint16_t col_end = mat->row_ptr[i+1];
    double temp = 0.0;

    for(; col_start != col_end; ++col_start) {
      temp += mat->vals[col_start] * vec->vals[mat->col_idx[col_start]];
    }

    res->vals[i] = temp;
  }
  return 0;
}

void test_csr_mat_mul_vec(uint16_t nrows, uint16_t ncols) {
  csr_mat mat = CSR_MAT_INIT;
  raw_vector vec = RAW_VECTOR_INIT;

  generate_random_csr_matrix(nrows, ncols, -10.0, 10.0, &mat);
  generate_random_vector(ncols, -10.0, 10.0, &vec);

  raw_vector res = RAW_VECTOR_INIT;

  csr_mat_mul_vec(&mat, &vec, &res);

  csr_mat_destroy(&mat);
  raw_vector_destroy(&vec);
  raw_vector_destroy(&res);
}

int main() {

  std::uniform_int_distribution<uint16_t> mat_nrows_distr(16, 5000), mat_ncols_distr(16, 5000);

  test_csr_mat_mul_vec(mat_nrows_distr(g_RandomEngine), mat_ncols_distr(g_RandomEngine));
}