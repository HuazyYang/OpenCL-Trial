#include "config.cl.h"

__kernel void pcr_small_system(
  _In_ __global const REAL *a_d,
  _In_ __global const REAL *b_d,
  _In_ __global const REAL *c_d,
  _In_ __global const REAL *d_d,
  _Out_ __global REAL *x_d,
  _In_ uint dimx_eliminated,
  _In_ uint iterations,
  _In_ uint stride,
  _In_shared_(blockDim.x * 5) __local REAL *tile
) {

  const uint tile_stride = dimx_eliminated + 1;
  int tid = get_local_id(0);
  int delta = 1;
  int gi = tid * stride;

  __local REAL *a = tile;
  __local REAL *b = a + tile_stride;
  __local REAL *c = b + tile_stride;
  __local REAL *d = c + tile_stride;
  __local REAL *x = d + tile_stride;

  if(tid < dimx_eliminated) {
    a[tid] = a_d[gi];
    b[tid] = b_d[gi];
    c[tid] = c_d[gi];
    d[tid] = d_d[gi];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for(uint k = 1; k < iterations; ++k) {

    REAL a_new, b_new, c_new, d_new;

    int i = tid;
    int h = i - delta;
    int j = i + delta;
    delta <<= 1;

    if(i < dimx_eliminated) {
      if(h < 0) {
        REAL k2 = c[i] / b[j];
        a_new = (REAL)0.0;
        b_new = b[i] - a[j] * k2;
        c_new = -c[j] * k2;
        d_new = d[i] - d[j] * k2;
      } else if(j >= dimx_eliminated) {
        REAL k1 = a[i] / b[h];
        a_new = -a[h] * k1;
        b_new = b[i] - c[h] * k1;
        c_new = (REAL)0.0;
        d_new = d[i] - d[h] * k1;
      } else {
        REAL k1 = a[i] / b[h];
        REAL k2 = c[i] / b[j];
        a_new = -a[h] * k1;
        b_new = b[i] - c[h] * k1 - a[j] * k2;
        c_new = -c[j] * k2;
        d_new = d[i] - d[h] * k1 - d[j] * k2;
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    a[i] = a_new;
    b[i] = b_new;
    c[i] = c_new;
    d[i] = d_new;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  uint i = tid;
  uint j = i + delta;
  if(j < dimx_eliminated) {
    REAL tmp = b[i] * b[j] - a[j] * c[i];
    x[i] = (d[i] * b[j] - d[j] * c[i]) / tmp;
    x[j] = (b[i] * d[j] - a[j] * d[i]) / tmp;
  } else if(i < delta)
    x[i] = d[i] / b[i];
  barrier(CLK_LOCAL_MEM_FENCE);

  if(i < dimx_eliminated)
    x_d[gi] = x[i];
}