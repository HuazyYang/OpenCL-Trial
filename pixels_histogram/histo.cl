#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable

#if __OPENCL_C_VERSION__ <= CL_VERSION_1_2
#define histo_inc_local(p)  atomic_inc(p)
#define histo_add_global(p, v) atomic_add(p, v)
#else
#define histo_inc_local(p) \
  atomic_fetch_add_explicit((atomic_uint *)(p), 1, memory_order_relaxed, memory_scope_work_group)
#define histo_add_global(p, v) \
  atomic_fetch_add_explicit((atomic_uint *)(p), v, memory_order_relaxed, memory_scope_device)
#endif

#define LOCAL_SIZE_X    64
#define LOCAL_SIZE_X_BIT    6
/** @note OPTD_LOCAL_SIZE_X must be power of 2 */
#define OPTD_LOCAL_SIZE_X 32
#define OPTD_LOCAL_SIZE_X_BIT 5

#define COALESCED_CACHE_LINE_SIZE       16
#define COALESCED_CACHE_LINE_SIZE_BIT   6
#define COALESCED_CACHE_INTN            int16

/**
 * @description:
 *    compute pixels histogram using global memory atomic increment.
 * @note:
 *    before kernel execution, histo must be initialized with zeros.
 */
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1)))
__kernel void histo_atomic(__global const uchar *pixels, uint num, __global uint * restrict histo) {

  __local uint local_h[256];
  const uint global_size = get_global_size(0);
  const uint local_size = LOCAL_SIZE_X;
  uint gid = get_global_id(0);
  uint tid = get_local_id(0);
  uint index;

  for(uint i = tid; i < 256; i += local_size)
    local_h[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(uint i = gid; i < num; i += global_size) {
    index = pixels[i];
    histo_inc_local(local_h + index);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll (256 >> LOCAL_SIZE_X_BIT)
  for(uint i = tid; i < 256; i += local_size)
    histo_add_global(histo + i, local_h[i]);
}

/**
 * @description: compute pixel histogram using coalesced access and global memory atomic increment
 * @note:
 *    - @param pixels_pui128 must be aligned at 16 bytes boundary, because we cast char* to uint4*.
 *    - global work group size must be muliple of 16.
 *    - before kernel can execute, histo must be initialized with zeros;
 */
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1)))
__kernel void hosto_atomic_coalesced(
  __global const uchar *pixels,
  uint num,
  __global uint * restrict histo
  ) {

  __local uint local_h[256];
  const uint global_size = get_global_size(0);
  const uint local_size = LOCAL_SIZE_X;
  uint tid = get_local_id(0);
  uint gid = get_global_id(0);
  COALESCED_CACHE_INTN pixel;

  __global const COALESCED_CACHE_INTN *pixels_pui = (__global const COALESCED_CACHE_INTN *)pixels;
  uint num_pui = num >> COALESCED_CACHE_LINE_SIZE_BIT;

  for(uint i = tid; i < 256; i += local_size)
    local_h[i] = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(uint i = gid; i < num_pui; i += global_size) {
    pixel = pixels_pui[i];

    #pragma unroll (COALESCED_CACHE_LINE_SIZE)
    for(uint j = 0; j < COALESCED_CACHE_LINE_SIZE; ++j) {
      uint pixel_j = pixel[j];
      uchar4 index = (uchar4)(pixel_j & 0xff, (pixel_j >> 8) & 0xff, (pixel_j >> 16) & 0xff,
      (pixel_j >> 24) & 0xff);
      histo_inc_local(local_h + index.s0);
      histo_inc_local(local_h + index.s1);
      histo_inc_local(local_h + index.s2);
      histo_inc_local(local_h + index.s3);
    }
  }

  #pragma unroll ((COALESCED_CACHE_LINE_SIZE << 4) - 1)
  for(uint j = (num_pui << COALESCED_CACHE_LINE_SIZE_BIT) + gid; j < num; j += global_size) {
    uchar index = pixels[j];
    histo_inc_local(local_h + index);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll (256 >> LOCAL_SIZE_X_BIT)
  for(uint i = tid; i < 256; i += local_size)
    histo_add_global(histo + i, local_h[i]);
}

/**
 * @description: compute pixel histogram using local shared memory only(optimized to ultimate version)
 */
__attribute__((reqd_work_group_size(OPTD_LOCAL_SIZE_X, 1, 1)))
__kernel void histo_optimized_ultimate(
  __global const uchar *pixels,
  uint num,
  __global uint * restrict histo
) {
  __local ushort2 local_h[128][OPTD_LOCAL_SIZE_X];
  const uint global_size = get_global_size(0);
  const uint local_size = OPTD_LOCAL_SIZE_X;
  const uint gid = get_global_id(0);
  const uint tid = get_local_id(0);

  for(uint i = 0; i < 128; ++i)
    local_h[i][tid] = (ushort2)(0, 0);
  barrier(CLK_LOCAL_MEM_FENCE);

  COALESCED_CACHE_INTN pixel;
  uint num_pui = num >> COALESCED_CACHE_LINE_SIZE_BIT;
  __global const COALESCED_CACHE_INTN *pixels_pui  =(__global const COALESCED_CACHE_INTN *)pixels;

  for(uint i = gid; i < num_pui; i += global_size) {
    pixel = pixels_pui[i];

    #pragma unroll (COALESCED_CACHE_LINE_SIZE)
    for(uint j = 0; j < COALESCED_CACHE_LINE_SIZE; ++j) {
      uint pixel_j = pixel[j];
      uchar4 index = (uchar4)(pixel_j & 0xff, (pixel_j >> 8) & 0xff, (pixel_j >> 16) & 0xff,
      (pixel_j >> 24) & 0xff);
      uchar4 row = index >> 1;
      uchar4 col = (index & (uchar4)(1, 1, 1, 1));
      local_h[row.s0][tid][col.s0]++;
      local_h[row.s1][tid][col.s1]++;
      local_h[row.s2][tid][col.s2]++;
      local_h[row.s3][tid][col.s3]++;
    }
  }

  #pragma unroll ((COALESCED_CACHE_LINE_SIZE << 4) - 1)
  for(uint i = (num_pui << COALESCED_CACHE_LINE_SIZE_BIT) + gid; i < num; i += global_size) {
    uchar index = pixels[i];
    uchar row = index >> 1;
    uchar col = index & 1;
    local_h[row][tid][col]++;
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  #pragma unroll (128 >> OPTD_LOCAL_SIZE_X_BIT)
  for(uint i = tid; i < 128; i += local_size) {
    uint2 sum = 0;
    #pragma unroll (OPTD_LOCAL_SIZE_X)
    for(uint j = 0; j < local_size; ++j) {
      ushort2 ih = local_h[i][(j + tid)&(local_size-1)];
      sum += (uint2)(ih.s0, ih.s1);
    }
    histo_add_global(histo + 2 * i, sum.s0);
    histo_add_global(histo + 2 * i + 1, sum.s1);
  }
}