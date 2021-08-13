#include "config.cl.h"

/**
 * @description:
 *  CR-PCR Hybrid forward reduction kernel
 * @note:
*     gridDim.x = floor(dimx / (delta * blockDim.x))
 */
__kernel void cr_pcr_forward_reduction(
  _Inout_ __global REAL *a_d,
  _Inout_ __global REAL *b_d,
  _Inout_ __global REAL *c_d,
  _Inout_ __global REAL *d_d,
  _In_ uint dimx,
  _In_ uint delta,
  _In_shared_(blockDim.x * 4) __local REAL *tile
) {

  const uint bdim = get_local_size(0);
  const uint bid = get_group_id(0);
  const uint bdimc = min(bdim, (dimx - bid * bdim * delta)/delta);
  const uint gid = get_global_id(0);
  const uint tid = get_local_id(0);
  const uint tile_row_size = 2 * bdim + 1;

  __local REAL *a = tile;
  __local REAL *b = a + tile_row_size;
  __local REAL *c = b + tile_row_size;
  __local REAL *d = c + tile_row_size;

  uint half_delta = delta >> 1;
  uint i, l, h;
  uint tid2;

  i = bid * bdim * delta + tid * half_delta  + half_delta - 1;
  if(i < dimx) {
    a[tid] = a_d[i];
    b[tid] = b_d[i];
    c[tid] = c_d[i];
    d[tid] = d_d[i];
  }
  i += bdimc * half_delta;
  if(i < dimx) {
    tid2 = tid + bdimc;
    a[tid2] = a_d[i];
    b[tid2] = a_d[i];
    c[tid2] = a_d[i];
    d[tid2] = a_d[i];
  }

  if(tid == bdimc - 1) {
    i += half_delta;
    tid2 = bdimc << 1;
    if(i < dimx) {
      a[tid2] = a_d[i];
      b[tid2] = b_d[i];
      c[tid2] = c_d[i];
      d[tid2] = d_d[i];
    } else {
      a[tid2] = (REAL)0.0;
      b[tid2] = (REAL)1.0;
      c[tid2] = (REAL)0.0;
      d[tid2] = (REAL)0.0;
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  i = tid * 2 + 1;
  l = i - 1;
  h = i + 1;

  if(i < bdimc) {
    REAL k1 = DIV_IMPL(a[i], b[l]);
    REAL k2 = DIV_IMPL(c[i], b[h]);

    a[i] = -a[l]*k1;
    b[i] = b[i] - c[l]*k1 - a[h]*k2;
    c[i] = -c[h]*k2;
    d[i] = d[i] - d[l]*k1 - d[h]*k2;

    uint gi = gid * delta + delta - 1;
    a_d[gi] = a[i];
    b_d[gi] = b[i];
    c_d[gi] = c[i];
    d_d[gi] = d[i];
  }
}

/**
 * @description:
 *  CR-PCR Hybrid backward substitution kernel
 * @note:
*     gridDim.x = ceil(dimx / (delta * blockDim.x))
 */
__kernel void cr_pcr_backward_substitution(
  _In_ __global const REAL *a_d,
  _In_ __global const REAL *b_d,
  _In_ __global const REAL *c_d,
  _In_ __global const REAL *d_d,
  _Out_ __global REAL *x_d,
  _In_ uint dimx,
  _In_ uint delta,
  _In_shared_(blockDim + 1) __local REAL *tile
) {
  const uint bdim = get_local_size(0);
  const uint bid = get_group_id(0);
  const uint bdimc = min(bdim, (dimx - bid * bdim * delta + delta - 1)/delta);
  const uint gid = get_global_id(0);
  const uint tid = get_local_id(0);

  __local REAL *x = tile;

  uint half_delta = delta >> 1;
  uint i, l, h;
  uint tid2;

  i = gid * delta  + delta - 1;
  x[tid+1] = i < dimx ? x_d[i] : (REAL)0.0;
  i -= delta;
  if(tid == 0 && bid != 0)
    x[0] = x_d[i];
  barrier(CLK_LOCAL_MEM_FENCE);

  if(i < dimx)
    x_d[i] = (d_d[i] - a_d[i]*x[tid] - c_d[i]*x[tid+1]) / b_d[i];
}

