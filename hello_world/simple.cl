
#if defined(cl_amd_fp64) || defined(cl_khr_fp64)
  #if defined(cl_amd_fp64)
      #pragma OPENCL EXTENSION cl_amd_fp64 : enable
  #elif defined(cl_khr_fp64)
      #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #endif
#endif

const sampler_t bilin_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__kernel void Add(read_only image2d_t imageA, image2d_t imageB, write_only image2d_t imageC) {

  int2 pos;
  pos.x = get_global_id(0);
  pos.y = get_global_id(1);

  float4 x = read_imagef(imageA, bilin_sampler, pos);
  float4 y = read_imagef(imageB, bilin_sampler, pos);

  write_imagef(imageC, pos, x + y);
}

__kernel void PI1(uint num, __local REAL *loc, __global REAL *r) {

  uint lid = get_local_id(0);
  uint did = get_global_id(0);
  uint gid = get_group_id(0);
  uint lsize = get_local_size(0);
  uint dsize = get_global_size(0);
  uint i;
  REAL x, temp, rnum = 1.0 / num;

  temp = 0.0f;
  for(i = did; i < num; i += dsize) {
    x = (i + 0.5) * rnum;
    temp += 4.0 / (1.0 + x * x);
  }

  loc[lid] = temp * rnum;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i = lsize/2; i > 0; i >>= 1) {
    if(lid < i)
      loc[lid] += loc[lid + i];

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
    r[gid] = loc[0];
}

/**
 * Execute on a single work group.
 */
__kernel void PI2(uint groupNum, __local REAL *loc, __global REAL *r) {

  uint lid = get_local_id(0);
  uint lsize = get_local_size(lsize);
  uint i;
  REAL temp = 0.0f;

  for(i=lid; i < groupNum; i += lsize) {
    temp += r[i];
  }

  loc[lid] = temp;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(i = lsize/2; i > 0; i >>= 1) {
    if(lid < i)
      loc[lid] += loc[lid+i];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if(lid == 0)
    r[0] = loc[0];
}


