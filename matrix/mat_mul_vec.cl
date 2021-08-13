#include <opencl.h>
#include <common.cl.h>

#define BLOCKED_LOCAL_SIZE_X 256

#define WARP_LOCAL_SIZE_X 32
#define WARP_LOCAL_SIZE_Y 8

__attribute__((reqd_work_group_size(BLOCKED_LOCAL_SIZE_X, 1, 1)))
__kernel void mxv_block(
  __global const REAL *d_mat,
  __global const REAL *d_vec,
  uint row_size,
  uint col_size,
  uint mat_pitch,
  __global REAL * restrict d_r
) {

  __local REAL tile[BLOCKED_LOCAL_SIZE_X];

  const uint bid = get_group_id(0);
  const uint tid = get_local_id(0);
  const uint bsize = get_num_groups(0);

  for(uint i = bid; i < row_size; i += bsize) {

    uint irow = i * mat_pitch;
    REAL temp = (REAL)0.0;
    for(uint j = tid; j < col_size; j += BLOCKED_LOCAL_SIZE_X)
      temp += d_mat[irow + j] * d_vec[j];
    tile[tid] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for(uint j = (BLOCKED_LOCAL_SIZE_X >> 1); j > 16; j >>= 1) {
      if(tid < j)
        tile[tid] += tile[j + tid];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(tid < 16)
      tile[tid] += tile[tid + 16];
    if(tid < 8)
      tile[tid] += tile[tid + 8];
    if(tid < 4)
      tile[tid] += tile[tid + 4];
    if(tid < 2)
      tile[tid] += tile[tid + 2];
    if(tid < 1) {
      tile[0] += tile[1];
      d_r[i] = tile[0];
    }
  }
}

/**
 * One row per block(warp)
 */
__attribute__((reqd_work_group_size(WARP_LOCAL_SIZE_X, WARP_LOCAL_SIZE_Y, 1)))
__kernel void mxv_warp(
  __global const REAL *d_mat,
  __global const REAL *d_vec,
  uint row_size,
  uint col_size,
  uint mat_pitch,
  __global REAL * restrict d_r
  ) {

  __local REAL s_s[2*WARP_LOCAL_SIZE_Y][WARP_LOCAL_SIZE_X];

  const uint2 tid = (uint2)(get_local_id(0), get_local_id(1));
  const uint2 bcount = (uint2)(get_num_groups(0), get_num_groups(1));
  const uint2 bid = (uint2)(get_group_id(0), get_group_id(1));

  __local REAL * const s_v = (__local REAL *)s_s;
  __local REAL * const row_s_r = (__local REAL *)(s_s + WARP_LOCAL_SIZE_Y + tid.y);

  const uint bsize = WARP_LOCAL_SIZE_Y * WARP_LOCAL_SIZE_X;
  const uint tid_spaned = tid.x + tid.y*WARP_LOCAL_SIZE_X;
  const uint warpid = (bid.x + bid.y * bcount.x) * WARP_LOCAL_SIZE_Y + tid.y;
  const uint warp_count = WARP_LOCAL_SIZE_Y * bcount.x * bcount.y;
  const uint row_size_rc = ((row_size + warp_count - 1) / warp_count) * warp_count;
  const uint row_bound = row_size - 1;

  for(uint i = warpid; i < row_size_rc; i += warp_count) {

    uint irow = (i < row_size ? i : row_bound) * mat_pitch; // Protect index out of range.
    row_s_r[tid.x] = 0.0;

    for(uint j = 0; j < col_size; j += bsize) {
      uint j_tid = j + tid_spaned;

      s_v[tid_spaned] = j_tid < col_size ? d_vec[j_tid] : 0.0; // Protect index out of range.
      barrier(CLK_LOCAL_MEM_FENCE);

      uint bsize_spaned = min(bsize, col_size - j);
      REAL temp = (REAL)0.0;

      #pragma unroll (WARP_LOCAL_SIZE_Y)
      for(uint k = tid.x; k < bsize_spaned; k += WARP_LOCAL_SIZE_X)
        temp += d_mat[irow + j + k] * s_v[k];
      row_s_r[tid.x] += temp;
      // We will write to s_v laterly, so asychronize here.
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(tid.x < 16)
      row_s_r[tid.x] += row_s_r[tid.x + 16];
    if(tid.x < 8)
      row_s_r[tid.x] += row_s_r[tid.x + 8];
    if(tid.x < 4)
      row_s_r[tid.x] += row_s_r[tid.x + 4];
    if(tid.x < 2)
      row_s_r[tid.x] += row_s_r[tid.x + 2];
    if(tid.x < 1) {
      row_s_r[0] += row_s_r[1];
      d_r[i] = row_s_r[0];
    }
  }
}



