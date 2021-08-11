#pragma once

/**
 * @struct tridiagonal_mat
 * @description: describe a diagonal matrix which has the following representation:
 * | b1  c1  *   *   *   *        |
 * | a2  b2  c2  *   *   *        |
 * | *   .   .   .   *   *        |
 * | *   *   .   .   .   *        |
 * | *   *   *   .   .   c(n-1)   |
 * | *   *   *   *   an  bn       |
 */
template<typename T>
struct tridiagonal_mat {
  size_t dim_x;
  T *a;
  T *b;
  T *c;

  tridiagonal_mat() noexcept : dim_x(0), a(nullptr), b(nullptr), c(nullptr) {}
  ~tridiagonal_mat() { dealloc(); }

  void alloc(size_t dim) {

    dealloc();

    dim_x = dim;
    if (dim_x) {
      T *buffer = new T[dim_x * 3];
      a = buffer;
      b = a + dim_x;
      c = b + dim_x;
    }
  }

  void dealloc() {
    if (a) {
      delete[] a;
      a = nullptr;
      b = nullptr;
      c = nullptr;
      dim_x = 0;
    }
  }


};

template<typename T>
struct column_vec {
  size_t dim_y;
  T *v;

  column_vec(): dim_y(0), v(nullptr) {}
  ~column_vec() { dealloc(); }

  void alloc(size_t dim) {

    dealloc();

    dim_y = dim;
    if (dim_y) {
      v = new T[dim_y];
    }
  }

  void dealloc() {
    if (v)
      delete[] v;
      v = nullptr;
      dim_y = 0;
  }
};



