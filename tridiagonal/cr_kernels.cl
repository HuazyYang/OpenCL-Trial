#include "config.cl.h"

__kernel void cr_small_system(
  _In_ __global const REAL *a_d,
  _In_ __global const REAL *b_d,
  _In_ __global const REAL *c_d,
  _In_ __global const REAL *d_d,
  _Out_ __global REAL *x_d,
  _In_ uint dimx_eliminated,
  _In_ uint iterations,
  _In_ uint stride,
  _In_shared_(dimx_eliminated * 5) __local REAL *tile
) {

  uint tid = get_local_id(0);
  uint half_dimx_eliminated = dimx_eliminated >> 1;
  uint2 gi = (uint2)(tid, tid + half_dimx_eliminated) * stride;

  __local REAL *a = tile;
  __local REAL *b = a + dimx_eliminated;
  __local REAL *c = b + dimx_eliminated;
  __local REAL *d = c + dimx_eliminated;
  __local REAL *x = d + dimx_eliminated;

  if(tid < dimx_eliminated) {
    a[tid] = a_d[gi.x];
    b[tid] = b_d[gi.x];
    c[tid] = c_d[gi.x];
    d[tid] = d_d[gi.x];

    uint tid2 = tid + half_dimx_eliminated;
    if(tid2 < dimx_eliminated) {
    a[tid2] = a_d[gi.y];
    b[tid2] = b_d[gi.y];
    c[tid2] = c_d[gi.y];
    d[tid2] = d_d[gi.y];
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

    if(i < dimx_eliminated) {
      if(h < dimx_eliminated) {
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

    if(j < dimx_eliminated) {
      REAL tmp = b[i]*b[j] - a[j]*c[i];
      x[i] = DIV_IMPL(d[i]*b[j] - d[j]*c[i], tmp);
      x[j] = DIV_IMPL(b[i]*d[j] - a[j]*d[i], tmp);
    } else if(i < dimx_eliminated)
      x[i] = DIV_IMPL(d[i], b[i]);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Backward substitution
  for(uint k = 1; k < iterations; ++k) {

    uint i = tid * delta + half_delta - 1;
    uint j = i + half_delta;

    if(tid == 0)
      x[i] = DIV_IMPL(d[i] - c[i] * x[j], b[i]);
    else if(j < dimx_eliminated)
      x[i] = DIV_IMPL(d[i] - a[i] * x[i - half_delta] - c[i]*x[j], b[i]);
    else if(i < dimx_eliminated)
      x[i] = DIV_IMPL(d[i] - a[i] * x[i - half_delta], b[i]);

    delta = half_delta;
    half_delta >>= 1;
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(tid < dimx_eliminated) {
    x_d[gi.x] = x[tid];
    uint tid2 = tid + half_dimx_eliminated;
    if(tid2 < dimx_eliminated)
      x_d[gi.y] = x[tid2];
  }
}
