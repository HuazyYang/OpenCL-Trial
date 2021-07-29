
/**
 * Implement the classical heat transfer equation:
 *  $$  \frac{\partial u}{\partial t} = \alpha \nabla u. $$
 *  where $ u = u(x, y, t) $ is 2D function in space dimensions
 *  and 1D function in time span.
 */

#define INIT_KERNEL_LOCAL_SIZE_X         64

typedef struct __attribute__((packed)) diff_params {
  int4 dims;
  float4 params;
} diff_params;

#define LOCAL_SIZE_X    16
#define LOCAL_SIZE_Y    16

__attribute__((reqd_work_group_size(INIT_KERNEL_LOCAL_SIZE_X, 1, 1)))
__kernel void heat_transfer_init(
  write_only image2d_t grid,
  __global const float *boundary_v
) {
  const uint2 dims = as_uint2(get_image_dim(grid));
  const uint2 gid = (uint2)(get_global_id(0), get_global_id(1));
  const uint gsize = get_global_size(0);
  const uint4 offsets;

  offsets.x = dims.y - 1;
  offsets.y = dims.x;
  for(uint i = gid.x; i < dims.x; i += gsize) {
    write_imagef(grid, (int2)(i, 0), (float4)(boundary_v[i]));
    write_imagef(grid, (int2)(i, offsets.x), (float4)(boundary_v[i + offsets.y]));
  }

  offsets.x = dims.x - 1;
  offsets.y = dims.x << 1;
  offsets.z = (dims.x <<  1) + dims.y;
  for(uint i = gid.x; i < dims.y; i += gsize) {
    write_imagef(grid, (int2)(0, i), (float4)(boundary_v[i + offsets.y]));
    write_imagef(grid, (int2)(offsets.x, i), (float4)(boundary_v[i + offsets.z]));
  }
}

const sampler_t bilin_mirror_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void heat_disturb(read_write image2d_t grid, int2 pos, float magnitude) {

  const int2 dims = as_int2(get_image_dim(grid));
  float val;
  int2 posoff;

  if(pos.x < dims.x && pos.y < dims.y) {
    val = read_imagef(grid, bilin_mirror_sampler, (float2)(pos.x, pos.y)).x;
    val += magnitude;
    write_imagef(grid, pos, (float4)(val));

    magnitude /= 2.0;
    posoff = pos + (int2)(-1, 0);
    val = read_imagef(grid, bilin_mirror_sampler, as_float2(posoff)).x;
    val += magnitude;
    write_imagef(grid, posoff, (float4)(val));

    posoff = pos + (int2)(1, 0);
    val = read_imagef(grid, bilin_mirror_sampler, as_float2(posoff)).x;
    val += magnitude;
    write_imagef(grid, posoff, (float4)(val));

    posoff = pos + (int2)(0, -1);
    val = read_imagef(grid, bilin_mirror_sampler, as_float2(posoff)).x;
    val += magnitude;
    write_imagef(grid, posoff, (float4)(val));

    posoff = pos + (int2)(0, 1);
    val = read_imagef(grid, bilin_mirror_sampler, as_float2(posoff)).x;
    val += magnitude;
    write_imagef(grid, posoff, (float4)(val));

    magnitude *= 0.89;
    #pragma unroll (2)
    for(int i = -1; i < 2; i += 2) {
      #pragma unroll (2)
      for(int j = -1; j < 2; j+= 2) {
        posoff =  pos + (int2)(i, j);
        if(posoff.x < dims.x && posoff.y < dims.y) {
          val = read_imagef(grid, bilin_mirror_sampler, as_float2(posoff)).x;
          val += magnitude;
          write_imagef(grid, posoff, (float4)(val));
        }
      }
    }
  }
}

__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void heat_transfer_step(
  read_only image2d_t curr_grid,
  write_only image2d_t next_grid,
  __constant diff_params *params
) {
  __local float __attribute__((numbanks(LOCAL_SIZE_X), bankwidth(4))) s_data[LOCAL_SIZE_Y+2][LOCAL_SIZE_X+2];
  const int2 tid = (int2)(get_local_id(0), get_local_id(1));
  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gsize = (int2)(get_global_size(0), get_global_size(1));
  const int2 dims_ru = (params->dims.xy + (int2)(LOCAL_SIZE_X - 1, LOCAL_SIZE_Y - 1)) &
            ~(int2)(LOCAL_SIZE_X-1,LOCAL_SIZE_Y-1);
  const int2 lpos = tid + (int2)(1, 1);
  float4 curr_val;

  for(int i = gid.x; i < dims_ru.x; i += gsize.x) {
    for(int j = gid.y; j < dims_ru.y; j += gsize.y) {

      s_data[lpos.y][lpos.x] = read_imagef(curr_grid, bilin_mirror_sampler, (float2)(i, j)).x;
      if(tid.y == 0)
        s_data[0][lpos.x] = read_imagef(curr_grid, bilin_mirror_sampler, (float2)(i, j-1)).x;
      if(lpos.y == LOCAL_SIZE_Y)
        s_data[LOCAL_SIZE_Y+1][lpos.x] = read_imagef(curr_grid, bilin_mirror_sampler, (float2)(i, j+1)).x;
      if(lpos.x == 1)
        s_data[lpos.y][0] = read_imagef(curr_grid, bilin_mirror_sampler, (float2)(i-1, j)).x;
      if(lpos.x == LOCAL_SIZE_X)
        s_data[lpos.y][LOCAL_SIZE_X+1] = read_imagef(curr_grid, bilin_mirror_sampler, (float2)(i+1, j)).x;

      barrier(CLK_LOCAL_MEM_FENCE);

      curr_val.x = params->params[0] * (
        s_data[lpos.y][lpos.x - 1] +
        s_data[lpos.y][lpos.x + 1] +
        s_data[lpos.y + 1][lpos.x] +
        s_data[lpos.y + 1][lpos.x]) + 
        params->params[1] * s_data[lpos.y][lpos.x];
      write_imagef(next_grid, (int2)(i, j), curr_val);

      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

