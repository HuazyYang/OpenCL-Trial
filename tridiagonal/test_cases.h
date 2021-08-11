#pragma once
#include "common_miscs.h"
#include <numeric>
#include <tuple>

template<typename T>
void test_gen_cyclic(T *a, T *b, T *c, T *d,
                     size_t system_size, int choice) {

  std::uniform_real_distribution<T> fd(static_cast<T>(0.0), static_cast<T>(1.0 + 1.0E-6));

  // fixed value, stable (no overflow, inf, nan etc)
  if (choice == 0) {
    for (ptrdiff_t j = 0; j < system_size; j++) {
      a[j] = (T)j;
      b[j] = (T)(j + 1);
      c[j] = (T)(j + 1);

      d[j] = (T)(j + 1);
    }
    a[0] = 0.0f;
    c[system_size - 1] = 0.0f;
  }

  // random
  if (choice == 1) {
    for (int j = 0; j < system_size; j++) {
      b[j] = fd(g_RandomEngine);
      a[j] = fd(g_RandomEngine);
      c[j] = fd(g_RandomEngine);
      d[j] = fd(g_RandomEngine);
    }
    a[0] = 0.0f;
    c[system_size - 1] = 0.0f;
  }

  // diagonally dominant
  if (choice == 2) {
    for (int j = 0; j < system_size; j++) {
      T ratio = fd(g_RandomEngine);
      b[j] = fd(g_RandomEngine);
      a[j] = b[j] * ratio * 0.5f;
      c[j] = b[j] * (1.0f - ratio) * 0.5f;
      d[j] = fd(g_RandomEngine);
    }
    a[0] = 0.0f;
    c[system_size - 1] = 0.0f;
  }

  // random not stable for cyclic reduction
  if (choice == 3) {
    for (int j = 0; j < system_size; j++) {
      b[j] = (T)fd(g_RandomEngine) + 3.0f;
      a[j] = (T)fd(g_RandomEngine) + 3.0f;
      c[j] = (T)fd(g_RandomEngine) + 3.0f;
      d[j] = (T)fd(g_RandomEngine) + 3.0f;
    }
    a[0] = 0.0f;
    c[system_size - 1] = 0.0f;
  }
}

template<typename T>
typename std::tuple<T, T, T> compare_var(const T *x1, const T *x2, size_t num_elements) {
  T mean = 0.0f; // mean error
  T root = 0.0f; // root mean square error
  T max = 0.0f; // max error
  for (ptrdiff_t i = 0; i < num_elements; i++) {
    T diff = std::abs(x1[i] - x2[i]);
    root += diff * diff;
    mean += diff;
    if (diff > max)
      max = diff;
  }
  mean /= (T)num_elements;
  root /= (T)num_elements;
  root = std::sqrt(root);
  return std::tuple<T, T, T>(max, mean, root);
}