#include "gcd.h"
#include <algorithm>
#include "reduced_math.h"
#include "index.h"
#include <omp.h>

namespace tr_inplace {
namespace openmp {
namespace details {

template <typename T, typename F> void col_shuffle(int m, int n, T *d, T *tmp, F fn) {
  tr_inplace::details::row_major_index rm(m, n);
  T *priv_tmp;
  F priv_fn;
  int tid;
  int i;
#pragma omp parallel private(tid, priv_tmp, priv_fn, i)
  {
    tid = omp_get_thread_num();
    priv_fn = fn;
    priv_tmp = tmp + m * tid;
#pragma omp for
    for (int j = 0; j < n; j++) {
      priv_fn.set_j(j);
      for (i = 0; i < m; i++) {
        priv_tmp[i] = d[rm(priv_fn(i), j)];
      }
      for (i = 0; i < m; i++) {
        d[rm(i, j)] = priv_tmp[i];
      }
    }
  }
}

template <typename T, typename F> void row_shuffle(int m, int n, T *d, T *tmp, F fn) {
  tr_inplace::details::row_major_index rm(m, n);
  T *priv_tmp;
  F priv_fn;
  int tid;
  int j;
#pragma omp parallel private(tid, priv_tmp, priv_fn, j)
  {
    tid = omp_get_thread_num();
    priv_fn = fn;
    priv_tmp = tmp + n * tid;
#pragma omp for
    for (int i = 0; i < m; i++) {
      priv_fn.set_i(i);
      for (j = 0; j < n; j++) {
        priv_tmp[j] = d[rm(i, priv_fn(j))];
      }
      for (j = 0; j < n; j++) {
        d[rm(i, j)] = priv_tmp[j];
      }
    }
  }
}

template <typename T> void transpose_fn(bool row_major, T *data, int m, int n, T *tmp) {

  if (m <= 1 || n <= 1)
    return;

  if (!row_major) {
    std::swap(m, n);
  }

  int c, t, k;
  tr_inplace::details::extended_gcd(m, n, c, t);
  if (c > 1) {
    tr_inplace::details::extended_gcd(m / c, n / c, t, k);
  } else {
    k = t;
  }
  if (c > 1) {
    col_shuffle(m, n, data, tmp, tr_inplace::details::prerotator(m, n / c));
  }
  row_shuffle(m, n, data, tmp, tr_inplace::details::shuffle(m, n, c, k));
  col_shuffle(m, n, data, tmp, tr_inplace::details::postpermuter(m, n, m / c));
}

} // namespace details

void transpose(bool row_major, float *data, int m, int n, float *tmp) {
  details::transpose_fn(row_major, data, m, n, tmp);
}

void transpose(bool row_major, double *data, int m, int n, double *tmp) {
  details::transpose_fn(row_major, data, m, n, tmp);
}

} // namespace openmp
} // namespace tr_inplace