
#define LOCAL_SIZE_X    64

#define WARP_LOCAL_SIZE_X   32
#define WARP_LOCAL_SIZE_Y   8

#define BLOCKED_LOCAL_SIZE_X  64

#ifndef BLOCKED_TILE_SIZE 
#define BLOCKED_TILE_SIZE     128
#endif

const sampler_t point_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__attribute__((reqd_work_group_size(LOCAL_SIZE_X, 1, 1)))
__kernel void smm_native(
  uint row_size,
  __global const uint *row_ptr,
  read_only image1d_buffer_t col_idx,
  read_only image1d_buffer_t mat_vals,
  read_only image1d_buffer_t vec_vals,
  write_only image1d_buffer_t res_vals
) {
  uint gid = get_global_id(0);
  uint gsize = get_global_size(0);

  for(uint i = gid; i < row_size; i += gsize) {
    REAL temp = 0.0;
    uint2 cpos = (uint2)(row_ptr[i], row_ptr[i+1]);
    for(; cpos.x != cpos.y; ++cpos.x) {
      int vidx = read_imagei(col_idx, (int)cpos.x).x;
      #ifdef _USE_DOUBLE_FP
      temp += as_double(read_imagef(mat_vals, (int)cpos.x).xy) *
          as_double(read_imagef(vec_vals, vidx).xy);
      #else
      temp += read_imagef(mat_vals, (int)cpos.x).x *
          read_imagef(vec_vals, vidx).x;
      #endif
    }
    #ifdef _USE_DOUBLE_FP
    write_imagef(res_vals, (int)i, (float4)(as_float2(temp), 0.0, 0.0));
    #else
    write_imagef(res_vals, (int)i, (float4)(temp));
    #endif
  }
}

__attribute__((reqd_work_group_size(WARP_LOCAL_SIZE_X, WARP_LOCAL_SIZE_Y, 1)))
__kernel void smm_warp_per_row(
  uint row_size,
  __global const uint *row_ptr,
  read_only image1d_buffer_t col_idx,
  read_only image1d_buffer_t mat_vals,
  read_only image1d_buffer_t vec_vals,
  write_only image1d_buffer_t res_vals
) {
  __local REAL tile[WARP_LOCAL_SIZE_Y][WARP_LOCAL_SIZE_X];

  const uint2 tid = (uint2)(get_local_id(0), get_local_id(1));
  const uint2 bcount = (uint2)(get_num_groups(0), get_num_groups(1));
  const uint2 bid = (uint2)(get_group_id(0), get_group_id(1));

  const uint bsize = WARP_LOCAL_SIZE_Y * WARP_LOCAL_SIZE_X;
  const uint warpid = (bid.x + bid.y * bcount.x) * WARP_LOCAL_SIZE_Y + tid.y;
  const uint warp_count = WARP_LOCAL_SIZE_Y * bcount.x * bcount.y;
  const uint row_size_rc = ((row_size + bsize - 1) / bsize) * bsize;
  const uint row_bound = row_size - 1;

  for(uint i = warpid; i < row_size_rc; i += warp_count) {

    uint ii = i < row_size ? i : row_bound;
    uint2 cpos = (uint2)(row_ptr[ii] + tid.x, row_ptr[ii+1]);
    REAL temp = 0.0;
    for(; cpos.x < cpos.y; cpos.x += WARP_LOCAL_SIZE_X) {
      int vidx = read_imagei(col_idx, (int)cpos.x).x;
      #ifdef _USE_DOUBLE_FP
      temp += as_double(read_imagef(mat_vals, (int)cpos.x).xy) *
          as_double(read_imagef(vec_vals, vidx).xy);
      #else
      temp += read_imagef(mat_vals, (int)cpos.x).x *
          read_imagef(vec_vals, vidx).x;
      #endif
    }
    // For warp size >= 32, NO synchronizing point is needed.
    tile[tid.y][tid.x] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);

#if (WARP_LOCAL_SIZE_X > 16)
    if(tid.x < 16)
      tile[tid.y][tid.x] += tile[tid.y][tid.x + 16];
#endif
    if(tid.x < 8)
      tile[tid.y][tid.x] += tile[tid.y][tid.x + 8];
    if(tid.x < 4)
      tile[tid.y][tid.x] += tile[tid.y][tid.x + 4];
    if(tid.x < 2)
      tile[tid.y][tid.x] += tile[tid.y][tid.x + 2];
    if(tid.x < 1 && i < row_size) {
      tile[tid.y][0] += tile[tid.y][1];
      #ifdef _USE_DOUBLE_FP
      write_imagef(res_vals, (int)i, (float4)(as_float2(tile[tid.y][0]), 0.0, 0.0));
      #else
      write_imagef(res_vals, (int)i, (float4)(tile[tid.y][0]));
      #endif
    }
  }
}

/*
__attribute__((reqd_work_group_size(BLOCKED_LOCAL_SIZE_X, 1, 1)))
__kernel void smm_block_multi_row(
  uint row_size,
  __global const ushort *row_heaps,
  __global const ushort *row_idx,
  __global read_only image1d_buffer row_ptr,
  __global read_only image1d_buffer col_idx,
  __global read_only image1d_buffer mat_vals,
  __global read_only image1d_buffer vec_vals,
  __global read_only image1d_buffer res_vals
) {

  __local REAL tile[BLOCKED_TILE_SIZE];

  const uint bid  = get_group_id(0);
  const uint tid = get_local_id(0);
  const uint2 row_range = (uint2)(row_heaps[bid], row_heaps[bid+1]);
  const uint heap_range_size = row_range.y - row_range.x;
  const uint2 col_range = (uint2)(
    read_imageui(row_ptr, point_sampler, row_range.x).x,
    read_imageui(row_ptr, point_sampler, row_range.y).x);

  if(heap_range_size == 0)
    return;
  else if(heap_range_size == 1) {

    REAL temp = 0.0;
    for(uint i = tid + col_range.x; i < col_range.y; i += BLOCKED_LOCAL_SIZE_X) {
      uint vidx = read_imageui(col_idx, point_sampler, i).x;
      #if _USE_DOUBLE_FP
        temp += as_double(read_imagef(mat_vals, point_sampler, i).xy) *
          as_double(read_imagef(vec_vals, vidx).xy);
      #else
        temp += read_imagef(mat_vals, point_sampler, i).x * read_imagef(vec_vals, vidx).x;
      #endif
    }
    tile[tid] = temp;
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint i = (BLOCKED_LOCAL_SIZE_X >> 1); i >= 32; i >>= 1) {
      if(tid < i)
        tile[tid] += tile[i+tid];
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
      #if _USE_DOUBLE_FP
        write_imagef(res_vals, row_idx[col_range.x], float4(as_float2(tile[0]), 0.0, 0.0));
      #else
        write_imagef(res_vals, row_idx[col_range.x], float4(tile[0]));
      #endif
    }
  } else {

   uint col_upper_rc = col_range.x + (col_range.y - col_range.x - BLOCKED_TILE_SIZE - 1) & ~(BLOCKED_TILE_SIZE - 1);

    for(uint i = col_range.x; i < col_upper_rc; i += BLOCKED_TILE_SIZE) {

      for(uint j = tid; j < BLOCKED_TILE_SIZE; j += BLOCKED_LOCAL_SIZE_X) {

        uint ii = j + i;
        uint vidx = read_imageui(col_idx, point_sampler, ii).x;

        #if _USE_DOUBLE_FP
        tile[j] = as_double(read_imagef(mat_vals, point_sampler, ii).xy) *
          as_double(read_imagef(vec_vals, vidx).xy);
        #else
        tile[j] = read_imagef(mat_vals, point_sampler, ii).x * read_imagef(vec_vals, vidx).x;
        #endif
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      REAL sum = 0.0;
      uint ii = i + tid;
      uint col_range2 = (uint2)(read_imageui(col_idx, point_sampler, ii).x,
          read_imageui(col_idx, point_sampler, ii+1).x) - (uint2)(i, i);
      for(uint k = col_range2.x; k < col_range2.y; ++k) {
        if(k < BLOCKED_LOCAL_SIZE_X)
          sum += tile[k];
      }

      #if _USE_DOUBLE_FP
      sum += read_imagef(res_vals, row_idx[ii])
      #else

      #endif
    }
  }
}
 */