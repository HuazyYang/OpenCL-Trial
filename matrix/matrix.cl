
#if defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define LOCAL_SIZE_X    16
#define LOCAL_SIZE_Y    16

__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void mat_transpose(__global const REAL *in, __global REAL *out, int M, int N) {

  int2 global_id = (int2)(get_global_id(0), get_global_id(1));

  if(global_id.x < M && global_id.y < N)
    out[global_id.y + global_id.x*N] = in[global_id.x + global_id.y*M];
}

// Merge access optimization
// 'dims' have components "M, N, max(M, N)"
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void mat_transpose_opt1(__global const REAL *in, __global REAL *out, int M, int N) {

    __local REAL tile[LOCAL_SIZE_Y][LOCAL_SIZE_X];

  int2 blockIdx = (int2)(get_group_id(0), get_group_id(1));
  int2 threadIdx = (int2)(get_local_id(0), get_local_id(1));
  const int2 blockSize = (int2)(LOCAL_SIZE_X, LOCAL_SIZE_Y);
  int2 i = blockIdx * blockSize + threadIdx;

  if(i.x < M &&  i.y < N)
    tile[threadIdx.y][threadIdx.x] = in[i.x + i.y*M];
  barrier(CLK_LOCAL_MEM_FENCE);

  i = (int2)(blockIdx.y, blockIdx.x) * blockSize + threadIdx;
  if(i.x < N && i.y < M)
    out[i.x+i.y*N] = tile[threadIdx.x][threadIdx.y];
}

// Eliminate bank conflict.
// 'dims' have components "M, N, max(M, N)"
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void mat_transpose_opt2(__global const REAL *in, __global REAL *out, int M, int N) {

  __local REAL tile[LOCAL_SIZE_Y][LOCAL_SIZE_X+1];

  int2 blockIdx = (int2)(get_group_id(0), get_group_id(1));
  int2 threadIdx = (int2)(get_local_id(0), get_local_id(1));
  const int2 blockSize = (int2)(LOCAL_SIZE_X, LOCAL_SIZE_Y);
  int2 i = blockIdx * blockSize + threadIdx;

  if(i.x < M &&  i.y < N)
    tile[threadIdx.y][threadIdx.x] = in[i.x + i.y*M];
  barrier(CLK_LOCAL_MEM_FENCE);

  i = (int2)(blockIdx.y, blockIdx.x) * blockSize + threadIdx;
  if(i.x < N && i.y < M)
    out[i.x+i.y*N] = tile[threadIdx.x][threadIdx.y];
}

//
// Perform matrix multiplication: C = A * B, where A[M][K], B[K][N], C[M][N]
//
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_X, 1)))
__kernel void mat_mul(__global const REAL *A, __global const REAL *B, __global REAL *C, const uint4 MKN) {

  int2 i = (int2)(get_global_id(0), get_global_id(1));
  REAL a, b, c = 0.0;

  if(i.x < MKN.z && i.y < MKN.x) {
    #pragma unroll
    for(int k = 0; k < MKN.y; ++k) {
      a = A[MKN.y*i.y + k];
      b = B[MKN.z*k + i.x];

      c += a * b;
    }

    C[MKN.z*i.y + i.x] = c;
  }
}

__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_X, 1)))
__kernel void mat_mul_opt1(__global const REAL *A, __global const REAL *B, __global REAL *C, const uint4 MKN) {

  __local REAL tile[2*LOCAL_SIZE_X*LOCAL_SIZE_X];
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int2 tid = (int2)(get_local_id(0), get_local_id(1));
  const int2 mn_bsign = (int2)(gid.y < MKN.x, gid.x < MKN.z);

  __local REAL *tile_a = (__local REAL *)tile;
  __local REAL *tile_b = (__local REAL *)tile + LOCAL_SIZE_X*LOCAL_SIZE_X;

  REAL c = 0.0;
  int2 kk;
  int2 kk2;
  int tile_index;

  #pragma unroll
  for(int k = 0; k < MKN.y; k += LOCAL_SIZE_X) {

    kk = (int2)(k, k) + tid;
    kk2 = (int2)(kk.x < MKN.y, kk.y < MKN.y) & mn_bsign;

    tile_index = tid.y*LOCAL_SIZE_X+tid.x;

    barrier(CLK_LOCAL_MEM_FENCE);
    tile_a[tile_index] = kk2.x*A[gid.y*MKN.y + kk.x];
    tile_b[tile_index] = kk2.y*B[kk.y*MKN.z + gid.x];
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for(int i = 0; i < LOCAL_SIZE_X; ++i) {
      c += tile_a[tid.y*LOCAL_SIZE_X+i] * tile_b[i*LOCAL_SIZE_X + tid.x];
    }
  }

  if(mn_bsign.x & mn_bsign.y)
    C[gid.y*MKN.z+gid.x] = c;
}
