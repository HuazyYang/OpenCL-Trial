#define LOCAL_SIZE_X      256
#define LOCAL_SIZE_X_BIT  8

/**
 * Compute vector dot product
 * @param c_temp must have a element count non-less than global block count.
 * @return c_temp[0] will hold the final result.
 */
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1)))
__kernel void vector_dist_sqr_reduced(__global const REAL *a,  __global const REAL *b, uint N, __global REAL *c_temp) {

  __local REAL tile[LOCAL_SIZE_X];
  const int tid = get_local_id(0);
  const int gid = get_global_id(0);
  const int gsize = get_global_size(0);
  const int bsize = LOCAL_SIZE_X;
  const int bid = get_group_id(0);
  const int gbsize = get_num_groups(0);
  REAL temp;

  tile[tid] = 0;

  for(int i = gid; i < N; i += gsize) {
    temp = a[i] - b[i];
    tile[tid] += temp * temp;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll (LOCAL_SIZE_X_BIT)
  for(int i = (bsize >> 1); i > 0; i >>= 1) {
    if(tid < i)
      tile[tid] += tile[i + tid];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  c_temp[bid] = tile[0];
  barrier(CLK_GLOBAL_MEM_FENCE);

  if(bid == 0) {
    temp = 0.0;
    for(int i = tid; i < gbsize; i += bsize)
      temp += c_temp[i];

    tile[tid] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll (LOCAL_SIZE_X_BIT)
    for(int i = (bsize >> 1); i > 0; i >>= 1) {
      if(i > tid)
        tile[tid] += tile[i + tid];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(tid == 0)
      c_temp[0] = tile[0];
  }
}
