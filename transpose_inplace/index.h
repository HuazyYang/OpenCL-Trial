#pragma once
#include "reduced_math.h"

namespace tr_inplace {
namespace details {

struct column_major_index {
  const int m;
  const int n;

  column_major_index(const int &_m, const int &_n) : m(_m), n(_n) {}

  int operator()(const int &i, const int &j) const { return i + j * m; }
};

struct row_major_index {
  const int m;
  const int n;

  row_major_index(const int &_m, const int &_n) : m(_m), n(_n) {}

  row_major_index(const reduced_divisor &_m, const int &_n) : m(_m.get()), n(_n) {}

  int operator()(const int &i, const int &j) const { return j + i * n; }
};

struct prerotator {
  reduced_divisor m, b;
  prerotator() : m(1), b(1) {}
  prerotator(int _m, int _b) : m(_m), b(_b) {}
  int x;
  void set_j(const int &j) { x = b.div(j); }
  int operator()(const int &i) { return m.mod(i + x); }
};

struct postpermuter {
  reduced_divisor m;
  int n;
  reduced_divisor a;
  int j;
  postpermuter() : m(1), a(1) {}
  postpermuter(int _m, int _n, int _a) : m(_m), n(_n), a(_a) {}
  void set_j(const int &_j) { j = _j; }
  int operator()(const int &i) { return m.mod((i * n + j - a.div(i))); }
};

struct shuffle {
  int m, n, k;
  reduced_divisor b;
  reduced_divisor c;
  shuffle() : b(1), c(1) {}
  shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k), b(_n / _c), c(_c) {}
  int i;
  void set_i(const int &_i) { i = _i; }
  int f(const int &j) {
    int r = j + i * (n - 1);
    // The (int) casts here prevent unsigned promotion
    // and the subsequent underflow: c implicitly casts
    // int - unsigned int to
    // unsigned int - unsigned int
    // rather than to
    // int - int
    // Which leads to underflow if the result is negative.
    if (i - (int)c.mod(j) <= m - (int)c.get()) {
      return r;
    } else {
      return r + m;
    }
  }

  int operator()(const int &j) {
    int fij = f(j);
    unsigned int fijdivc, fijmodc;
    c.divmod(fij, fijdivc, fijmodc);
    // The extra mod in here prevents overflowing 32-bit int
    int term_1 = b.mod(k * b.mod(fijdivc));
    int term_2 = ((int)fijmodc) * (int)b.get();
    return term_1 + term_2;
  }
};


} // namespace details
} // namespace tr_inplace
