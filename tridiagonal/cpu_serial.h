#include "tridiagonal_mat.h"

namespace cpu_solver {

size_t log2c(size_t n) {
  size_t i = 1, res = 0;
  for(; i < n; i <<= 1, ++res);

  return res;
}

template <typename T>
void thomas_serial(const tridiagonal_mat<T> *A, const column_vec<T> *d_, column_vec<T> *x_, T *c_ast, T *d_ast) {

  memcpy(c_ast, A->c, sizeof(T) * A->dim_x);
  memcpy(d_ast, d_->v, sizeof(T) * A->dim_x);

  size_t n = A->dim_x;
  const T *a = A->a;
  const T *b = A->b;
  T *c = c_ast;
  T *d = d_ast;
  T *x = x_->v;

  c[0] = c[0] / b[0];
  d[0] = d[0] / b[0];

  for(ptrdiff_t i = 1; i < n; ++i) {
    T temp = b[i] - c[i-1] * a[i];
    c[i] = c[i] / temp;
    d[i] = (d[i] - d[i-1]*a[i]) / temp;
  }

  x[n-1] = d[n-1];

  for(ptrdiff_t i = n-2; i >= 0; --i) {
    x[i] = d[i] - c[i] * x[i+1];
  }
}

template<typename T>
void cyclic_reduction(const tridiagonal_mat<T> *A, const column_vec<T> *d_, column_vec<T> *x_, T *a_ast, T *b_ast, T *c_ast, T *d_ast) {

  size_t n = A->dim_x;
  size_t buff_len = sizeof(T) * n;

  if(n < 2) {
    if(n == 1)
      x_->v[0] = d_->v[0] / A->b[0];
    return;
  }

  memcpy(a_ast, A->a, buff_len);
  memcpy(b_ast, A->b, buff_len);
  memcpy(c_ast, A->c, buff_len);
  memcpy(d_ast, d_->v, buff_len);

  T *a = a_ast;
  T *b = b_ast;
  T *c = c_ast;
  T *d = d_ast;
  T *x = x_->v;

  const size_t nlevel = log2c(n);
  size_t half_delta = 1, delta = 1;

  for(ptrdiff_t k = 1; k < nlevel; ++k) {
    half_delta = delta;
    delta <<= 1;
    ptrdiff_t i = delta - 1;
    ptrdiff_t j = i + half_delta;
    for(; j < n; i += delta, j += delta) {

      T k1 = a[i] / b[i - half_delta];
      T k2 = c[i] / b[j];

      a[i] = -a[i - half_delta] * k1;
      b[i] = b[i] - c[i - half_delta] * k1 - a[j] * k2;
      c[i] = -c[i + half_delta] * k2;
      d[i] = d[i] - d[i - half_delta] * k1 - d[j] * k2;
    }

    if(i < n && j >= n) {
      T k1 = a[i] / b[i - half_delta];

      a[i] = -a[i - half_delta] * k1;
      b[i] = b[i] - c[i - half_delta] * k1;
      c[i] = (T)0.0;
      d[i] = d[i] - d[i - half_delta] * k1;
    }
  }

  ptrdiff_t i = delta - 1, j = i + delta;
  if(j < n) {
    T tmp = b[i]*b[j] - a[j]*c[i];
    x[i] = (d[i]*b[j] - d[j]*c[i]) / tmp;
    x[j] = (b[i]*d[j] - a[j]*d[i]) / tmp;
  } else
    x[i] = d[i] / b[i];

  for(ptrdiff_t k = 1; k < nlevel; ++k) {
    i = half_delta - 1;
    j = i + half_delta;

    if(j < n)
      x[i] = (d[i] - c[i] * x[j]) / b[i];

    i += delta;
    j += delta;
    for(; j < n; i += delta, j += delta)
      x[i] = (d[i] - a[i]*x[i-half_delta] - c[i]*x[j]) / b[i];

    if(i < n && j >= n)
      x[i] = (d[i] - a[i]*x[i-half_delta]) / b[i];

    delta >>= 1;
    half_delta >>= 1;
  }
}

/**
 * @note: I do not want to implement parallel cyclic reduction on CPU side
 *        because it will need more auxillary diagonal matrix element buffer
 *        size of 4 *n, which is not much efficient any way.
 */

};