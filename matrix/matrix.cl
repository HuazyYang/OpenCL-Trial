
#if defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#elif defined(cl_khr_fp64)
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define LOCAL_SIZE_X    16
#define LOCAL_SIZE_Y    16

__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void mat_transpose(__global const REAL *in, __global REAL *out, int M, int N) {

  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gsize = (int2)(get_global_size(0), get_global_size(1));

  for(uint i = gid.y; i < N; i += gsize.y) {
    for(uint j = gid.x; j < M; j += gsize.x)
      out[i + j * N] = in[j + i*M];
  }
}

// Merge access optimization
// 'dims' have components "M, N, max(M, N)"
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void mat_transpose_opt1(__global const REAL *in, __global REAL *out, int M, int N) {

  __local REAL tile[LOCAL_SIZE_Y][LOCAL_SIZE_X];
  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gsize = (int2)(get_global_size(0), get_global_size(1));
  const int2 tid = (int2)(get_local_id(0), get_local_id(1));

  for(uint i = gid.y; i < N; i += gsize.y) {
    for(uint j = gid.x; j < M; j += gsize.x) {
      tile[tid.y][tid.x] = in[j + i*M];
      barrier(CLK_LOCAL_MEM_FENCE);

      int2 ii = (int2)(i - tid.y + tid.x, j - tid.x + tid.y);
      if(ii.x < N && ii.y < M)
        out[j + i*M] = tile[tid.x][tid.y];
    }
  }
}

// Eliminate bank conflict.
// 'dims' have components "M, N, max(M, N)"
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void mat_transpose_opt2(__global const REAL *in, __global REAL *out, int M, int N) {

  __local REAL tile[LOCAL_SIZE_Y][LOCAL_SIZE_X+1];
  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gsize = (int2)(get_global_size(0), get_global_size(1));
  const int2 tid = (int2)(get_local_id(0), get_local_id(1));

  for(uint i = gid.y; i < N; i += gsize.y) {
    for(uint j = gid.x; j < M; j += gsize.x) {
      tile[tid.y][tid.x] = in[j + i*M];
      barrier(CLK_LOCAL_MEM_FENCE);

      int2 ii = (int2)(i - tid.y + tid.x, j - tid.x + tid.y);
      if(ii.x < N && ii.y < M)
        out[j + i*M] = tile[tid.x][tid.y];
    }
  }
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
