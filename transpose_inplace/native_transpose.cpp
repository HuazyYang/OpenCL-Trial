#include "gcd.h"
#include <algorithm>
#include "reduced_math.h"
#include "index.h"

namespace tr_inplace {
namespace native {
namespace details {

template <typename T, typename F> void col_shuffle(int m, int n, T *d, T *tmp, F fn) {
  tr_inplace::details::row_major_index rm(m, n);
  int i;
  for (int j = 0; j < n; j++) {
    fn.set_j(j);
    for (int i = 0; i < m; i++) {
      tmp[i] = d[rm(fn(i), j)];
    }
    for (i = 0; i < m; i++) {
      d[rm(i, j)] = tmp[i];
    }
  }
}

template <typename T, typename F> void row_shuffle(int m, int n, T *d, T *tmp, F fn) {
  tr_inplace::details::row_major_index rm(m, n);
  int j;
  for (int i = 0; i < m; i++) {
    fn.set_i(i);
    for (j = 0; j < n; j++) {
      tmp[j] = d[rm(i, fn(j))];
    }
    for (j = 0; j < n; j++) {
      d[rm(i, j)] = tmp[j];
    }
  }
}

template <typename T> void transpose_fn(bool row_major, T *data, int m, int n, T *tmp) {

  if(m <= 1 || n <= 1)
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

} // namespace native
} // namespace tr_inplace