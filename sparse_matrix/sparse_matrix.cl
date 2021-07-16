
#define LOCAL_SIZE_X    64

#define WARP_LOCAL_SIZE_X   32
#define WARP_LOCAL_SIZE_Y   4

__attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1)))
__kernel void smm_native(
  uint row_size,
  __global const ushort *row_ptr,
  __global const ushort *col_idx,
  __global const REAL *mat_vals,
  __global const REAL *vec_vals,
  __global REAL * restrict res_vals
) {
  uint gid = get_global_id(0);
  uint gsize = get_global_size(0);

  for(uint i = gid; i < row_size; i += gsize) {
    REAL temp = 0.0;
    uint2 cpos = (uint2)(row_ptr[i], row_ptr[i+1]);
    for(; cpos.x < cpos.y; ++cpos.x) {
      temp += mat_vals[cpos.x] * vec_vals[col_idx[cpos.x]];
    }
    res_vals[i] = temp;
  }
}

__attribute__((reqd_work_group_size(WARP_LOCAL_SIZE_X, WARP_LOCAL_SIZE_Y, 1)))
__kernel void smm_warp_per_row(
  uint row_size,
  __global const ushort *row_ptr,
  __global const ushort *col_idx,
  __global const REAL *mat_vals,
  __global const REAL *vec_vals,
  __global REAL * restrict res_vals
) {
  __local REAL tile[WARP_LOCAL_SIZE_Y];

  const uint2 tid = (uint2)(get_local_id(0), get_local_id(1));
  const uint2 bcount = (uint2)(get_num_groups(0), get_num_groups(1));
  const uint2 bid = (uint2)(get_group_id(0), get_group_id(1));

  const uint bsize = WARP_LOCAL_SIZE_Y * WARP_LOCAL_SIZE_X;
  const uint warpid = (bid.x + bid.y * bcount.x) * WARP_LOCAL_SIZE_Y + tid.y;
  const uint warp_count = WARP_LOCAL_SIZE_Y * bcount.x * bcount.y;
  const uint row_size_rc = ((row_size + bsize - 1) / bsize) * bsize;
  const uint row_bound = row_size - 1;

  for(uint i = warpid; i < row_size_rc; i += warp_count) {

    tile[tid.y] = 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);

    uint ii = i < row_size ? i : row_bound;
    uint2 cpos = (uint2)(row_ptr[ii] + tid.x, row_ptr[ii+1]);
    REAL temp = 0.0;
    for(; cpos.x < cpos.y; cpos.x += WARP_LOCAL_SIZE_X) {
      temp += mat_vals[cpos.x] * vec_vals[col_idx[cpos.x]];
    }
    // For warp size >= 32, NO synchronizing point is needed.
    tile[tid.y] += temp;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(tid.x == 0 && i < row_size)
      res_vals[i] = tid[tid.y];
  }
}