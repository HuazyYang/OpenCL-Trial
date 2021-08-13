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

/**
 * @param a_ast, @param b_ast, @param c_ast, @param d_ast
 *  must have a element count non-less than @param A->dim_x.
 */
template<typename T>
void cyclic_reduction(const tridiagonal_mat<T> *A, const column_vec<T> *d_, column_vec<T> *x_, T *a_ast, T *b_ast, T *c_ast, T *d_ast) {

  size_t n = A->dim_x;
  size_t buff_len = sizeof(T) * n;

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
    ptrdiff_t h = i - half_delta;
    for(; j < n; h = j, i += delta, j += delta) {

      T k1 = a[i] / b[h];
      T k2 = c[i] / b[j];

      a[i] = -a[h] * k1;
      b[i] = b[i] - c[h] * k1 - a[j] * k2;
      c[i] = -c[j] * k2;
      d[i] = d[i] - d[h] * k1 - d[j] * k2;
    }

    if(i < n) {
      T k1 = a[i] / b[h];

      a[i] = -a[h] * k1;
      b[i] = b[i] - c[h] * k1;
      c[i] = (T)0.0;
      d[i] = d[i] - d[h] * k1;
    }
  }

  ptrdiff_t i = delta - 1, j = i + delta, h;
  if(j < n) {
    T tmp = b[i]*b[j] - a[j]*c[i];
    x[i] = (d[i]*b[j] - d[j]*c[i]) / tmp;
    x[j] = (b[i]*d[j] - a[j]*d[i]) / tmp;
  } else  if(i < n)
    x[i] = d[i] / b[i];

  for(ptrdiff_t k = 1; k < nlevel; ++k) {
    i = half_delta - 1;
    j = i + half_delta;
    h = i - half_delta;

    if(j < n)
      x[i] = (d[i] - c[i] * x[j]) / b[i];

    h = j;
    i += delta;
    j += delta;
    for(; j < n; h = j, i += delta, j += delta)
      x[i] = (d[i] - a[i]*x[h] - c[i]*x[j]) / b[i];

    if(i < n)
      x[i] = (d[i] - a[i]*x[h]) / b[i];

    delta >>= 1;
    half_delta >>= 1;
  }
}

/**
 * @param a_ast, @param b_ast, @param c_ast, @param d_ast
 *  must have a element count non-less than @param A->dim_x + ( @param A->dim_x + 1 ) / 2.
 */
template <typename T>
void parallel_cyclic_reduction(const tridiagonal_mat<T> *A,
                               const column_vec<T> *d_, column_vec<T> *x_,
                               T *a_ast, T *b_ast, T *c_ast, T *d_ast) {
  size_t n = A->dim_x;
  size_t buff_len = sizeof(T) * n;

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

  for (ptrdiff_t k = 1; k < nlevel; ++k) {
    half_delta = delta;
    delta <<= 1;

    for(ptrdiff_t l = 0; l < half_delta; ++l) {

      ptrdiff_t h, i, j, ic;

      i = l;
      j = i + half_delta;
      h = i - half_delta;
      ic = n;

      T k2 = c[i] / b[j];
      a[ic] = (T)0.0;
      b[ic] = b[i] - a[j] * k2;
      c[ic] = -c[j] * k2;
      d[ic] = d[i] - d[j] * k2;

      h = j;
      i += delta;
      j += delta;
      ic += 1;
      for (; j < n; h = j, i += delta, j += delta, ++ic) {

        T k1 = a[i] / b[h];
        T k2 = c[i] / b[j];

        a[ic] = -a[h] * k1;
        b[ic] = b[i] - c[h] * k1 - a[j] * k2;
        c[ic] = -c[j] * k2;
        d[ic] = d[i] - d[h] * k1 - d[j] * k2;
      }

      if (i < n) {
        T k1 = a[i] / b[h];

        a[ic] = -a[h] * k1;
        b[ic] = b[i] - c[h] * k1;
        c[ic] = (T)0.0;
        d[ic] = d[i] - d[h] * k1;
      }

      i = l + half_delta;
      j = i + half_delta;
      h = l;
      for (; j < n; h = j, i += delta, j += delta) {

        T k1 = a[i] / b[h];
        T k2 = c[i] / b[j];

        a[i] = -a[h] * k1;
        b[i] = b[i] - c[h] * k1 - a[j] * k2;
        c[i] = -c[j] * k2;
        d[i] = d[i] - d[h] * k1 - d[j] * k2;
      }

      if (i < n) {
        T k1 = a[i] / b[h];

        a[i] = -a[h] * k1;
        b[i] = b[i] - c[h] * k1;
        c[i] = (T)0.0;
        d[i] = d[i] - d[h] * k1;
      }

      // Copy the back buffer back.
      i = l;
      ic = n;

      for(; i < n; i += delta, ++ic) {
        a[i] = a[ic];
        b[i] = b[ic];
        c[i] = c[ic];
        d[i] = d[ic];
      }
    }
  }

  ptrdiff_t i = 0, j = delta;
  for (; j < n; ++i, ++j) {

    T tmp = b[i] * b[j] - a[j] * c[i];
    x[i] = (d[i] * b[j] - d[j] * c[i]) / tmp;
    x[j] = (b[i] * d[j] - a[j] * d[i]) / tmp;
  }

  if(n > 0) {
    for (; i < delta; ++i)
      x[i] = d[i] / b[i];
  }
}

/**
 * @param mat_buffer must have a element count non-less than @param A->dim_x.
 */
template <typename T, typename C_NonZero_Predicate>
void __recursive_doubling_sub_eq(ptrdiff_t eq_start, ptrdiff_t eq_end, bool fill_recu_mat,
                                 const tridiagonal_mat<T> *A,
                                 const column_vec<T> *d_, column_vec<T> *x_,
                                 T (*mat_buffer_)[2][3],
                                 const C_NonZero_Predicate &fn_c_nz) {

  size_t n = static_cast<size_t>(eq_end - eq_start);
  size_t nupper_bound = n - 1;
  ptrdiff_t sub_eq_start = eq_start, sub_eq_end;
  const T *a = A->a + eq_start;
  const T *b = A->b + eq_start;
  const T *c = A->c + eq_start;
  const T *d = d_->v + eq_start;
  T *x = x_->v + eq_start;
  T (*mat_buffer)[2][3] = mat_buffer_ + eq_start;

  T tmp_mat[2][3];

  if(n < 1)
    return;

  if (n == 1) {
    T(*m)[3] = mat_buffer[0];
    m[0][0] = -b[0];
    m[0][1] = -(T)1.0;
    m[0][2] = d[0];
    m[1][0] = (T)1.0;
    m[1][1] = (T)0.0;
    m[1][2] = (T)0.0;
  } else if(!fill_recu_mat) {
    T(*m)[3] = mat_buffer[nupper_bound];
    m[0][0] = -b[nupper_bound];
    m[0][1] = -a[nupper_bound];
    m[0][2] = d[nupper_bound];
    m[1][0] = (T)1.0;
    m[1][1] = (T)0.0;
    m[1][2] = (T)0.0;
  } else {
    // Build recursive matrix
    if (fn_c_nz(c[0])) {
      T(*m)[3] = mat_buffer[0];
      m[0][0] = -b[0] / c[0];
      m[0][1] = -(T)1.0 / c[0];
      m[0][2] = d[0] / c[0];
      m[1][0] = (T)1.0;
      m[1][1] = (T)0.0;
      m[1][2] = (T)0.0;
    } else {
      sub_eq_end = eq_start + 1;
      printf("Forward recursive 1\n");
      __recursive_doubling_sub_eq(sub_eq_start, sub_eq_end, false, A, d_, x_, mat_buffer_, fn_c_nz);
      sub_eq_start = sub_eq_end;
    }

    for (ptrdiff_t i = 1; i < nupper_bound; ++i) {
      if(fn_c_nz(c[i])) {
        T(*m)[3] = mat_buffer[i];
        m[0][0] = -b[i] / c[i];
        m[0][1] = -a[i] / c[i];
        m[0][2] = d[i] / c[i];
        m[1][0] = (T)1.0;
        m[1][1] = (T)0.0;
        m[1][2] = (T)0.0;
      } else {
        sub_eq_end = eq_start + i + 1;
        printf("Forward recursive 2\n");
        __recursive_doubling_sub_eq(sub_eq_start, sub_eq_end, false, A, d_, x_, mat_buffer_, fn_c_nz);
        sub_eq_start = sub_eq_end;
      }
    }

    return __recursive_doubling_sub_eq(sub_eq_start, eq_end, false, A, d_, x_,
                                       mat_buffer_, fn_c_nz);
  }

  const size_t nlevel = log2c(n);
  size_t delta = 1, half_delta = 1;

  for(ptrdiff_t k = 0; k < nlevel; ++k) {

    half_delta = delta;
    delta <<= 1;

    ptrdiff_t i = (ptrdiff_t)n - 1 - half_delta;
    ptrdiff_t j = n - 1;
    for (; i >= 0; i -= 1, j -= 1) {

      T (*mi)[3] = mat_buffer[i];
      T (*mj)[3] = mat_buffer[j];

      tmp_mat[0][0] = mj[0][0] * mi[0][0] + mj[0][1] * mi[1][0];
      tmp_mat[0][1] = mj[0][0] * mi[0][1] + mj[0][1] * mi[1][1];
      tmp_mat[0][2] = mj[0][0] * mi[0][2] + mj[0][1] * mi[1][2] + mj[0][2];
      tmp_mat[1][0] = mj[1][0] * mi[0][0] + mj[1][1] * mi[1][0];
      tmp_mat[1][1] = mj[1][0] * mi[0][1] + mj[1][1] * mi[1][1];
      tmp_mat[1][2] = mj[1][0] * mi[0][2] + mj[1][1] * mi[1][2] + mj[1][2];

      memcpy(mj, tmp_mat, sizeof(tmp_mat));
    }
  }

  ptrdiff_t i = nupper_bound;
  T (*m)[3] = mat_buffer[i];
  x[0] = -m[0][2] / m[0][0];

  for(i = 0; i < nupper_bound; ++i) {
    m = mat_buffer[i];
    x[i+1] = m[0][0]*x[0] + m[0][2];
  }
}

template <typename T, typename C_NonZero_Predicate>
void recursive_doubling(const tridiagonal_mat<T> *A, const column_vec<T> *d,
                        column_vec<T> *x, T (*mat_buffer)[2][3],
                        const C_NonZero_Predicate &fn_c_nz) {

  return __recursive_doubling_sub_eq(0, static_cast<ptrdiff_t>(A->dim_x), true, A, d,
                                     x, mat_buffer, fn_c_nz);
}

};