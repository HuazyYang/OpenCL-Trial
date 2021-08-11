#include <common.cl.h>

// #define NATIVE_DIVIDE

#ifdef NATIVE_DIVIDE
#define DIV_IMPL(a, b) native_divide((a), (b))
#else
#define DIV_IMPL(a, b) (a) / (b)
#endif

__kernel void cr_kernel(
  __global REAL *a_d,
  __global REAL *b_d,
  __global REAL *c_d,
  __global REAL *d_d,
  __global REAL *x_d,
  uint dimx,
  uint iterations,
  __local REAL *tile
) {

  uint tid = get_local_id(0);

  uint half_dimx = dimx >> 1;

  __local REAL *a = tile;
  __local REAL *b = a + dimx;
  __local REAL *c = b + dimx;
  __local REAL *d = c + dimx;
  __local REAL *x = d + dimx;

  if(tid < dimx) {
    a[tid] = a_d[tid];
    b[tid] = b_d[tid];
    c[tid] = c_d[tid];
    d[tid] = d_d[tid];

    uint tid2 = tid + half_dimx;
    if(tid2 < dimx) {
    a[tid2] = a_d[tid2];
    b[tid2] = b_d[tid2];
    c[tid2] = c_d[tid2];
    d[tid2] = d_d[tid2];
    }
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint delta = 1, half_delta = 1;

  // Forward reduction
  for(uint k = 1; k < iterations; ++k) {

    half_delta = delta;
    delta <<= 1;

    uint i = delta * tid + delta - 1;
    uint h = i + half_delta;
    uint l = i - half_delta;

    if(i < dimx) {
      if(h < dimx) {
        REAL k1 = DIV_IMPL(a[i], b[l]);
        REAL k2 = DIV_IMPL(c[i], b[h]);

        a[i] = -a[l]*k1;
        b[i] = b[i] - c[l]*k1 - a[h]*k2;
        c[i] = -c[h]*k2;
        d[i] = d[i] - d[l]*k1 - d[h]*k2;
      } else {
        REAL k1 = DIV_IMPL(a[i], b[l]);

        a[i] = -a[l]*k1;
        b[i] = b[i] - c[l]*k1;
        c[i] = (REAL)0.0;
        d[i] = d[i] - d[l]*k1;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // Solve last equation.
  if(tid == 0) {
    uint i = delta - 1;
    uint j = i + delta;

    if(j < dimx) {
      REAL tmp = b[i]*b[j] - a[j]*c[i];
      x[i] = DIV_IMPL(d[i]*b[j] - d[j]*c[i], tmp);
      x[j] = DIV_IMPL(b[i]*d[j] - a[j]*d[i], tmp);
    } else
      x[i] = DIV_IMPL(d[i], b[i]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Backward substitution
  for(uint k = 1; k < iterations; ++k) {

    uint i = tid * delta + half_delta - 1;
    uint j = i + half_delta;

    if(tid == 0)
      x[i] = DIV_IMPL(d[i] - c[i] * x[j], b[i]);
    else if(j < dimx)
      x[i] = DIV_IMPL(d[i] - a[i] * x[i - half_delta] - c[i]*x[j], b[i]);
    else if(i < dimx)
      x[i] = DIV_IMPL(d[i] - a[i] * x[i - half_delta], b[i]);

    delta = half_delta;
    half_delta >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(tid < dimx) {
    x_d[tid] = x[tid];
    uint tid2 = tid + half_dimx;
    if(tid2 < dimx)
      x_d[tid2] = x[tid2];
  }
}