#include <common.cl.h>

#define LOCAL_SIZE_X    16
#define LOCAL_SIZE_Y    16

#ifdef  _USE_DOUBLE_FP
#define _REAL_BANK_WITH   8
#else
#define _REAL_BANK_WITH   4
#endif

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

  __local REAL __attribute__((numbanks(LOCAL_SIZE_X), bankwith(_REAL_BANK_WITH))) tile[LOCAL_SIZE_Y][LOCAL_SIZE_X + 1];
  const int2 gid = (int2)(get_global_id(0), get_global_id(1));
  const int2 gsize = (int2)(get_global_size(0), get_global_size(1));
  const int2 tid = (int2)(get_local_id(0), get_local_id(1));
  const int M_ru = (M + LOCAL_SIZE_X - 1) & ~(LOCAL_SIZE_X - 1);
  const int N_ru = (N + LOCAL_SIZE_Y - 1) & ~(LOCAL_SIZE_Y - 1);

  for(uint i = gid.y; i < N_ru; i += gsize.y) {
    for(uint j = gid.x; j < M_ru; j += gsize.x) {
      if(j < M && i < N)
        tile[tid.y][tid.x] = in[j + i*M];
      barrier(CLK_LOCAL_MEM_FENCE);

      int2 ii = (int2)(i - tid.y, j - tid.x) + tid;
      if(ii.x < N && ii.y < M)
        out[ii.x + ii.y*N] = tile[tid.x][tid.y];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
}

// Eliminate bank conflict.
// 'dims' have components "M, N, max(M, N)"
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_Y, 1)))
__kernel void mat_transpose_opt2(__global const REAL *in, __global REAL *out, uint M, uint N) {

#ifdef _USE_DOUBLE_FP
  __local uint __attribute__((numbanks(LOCAL_SIZE_X), bankwith(4))) tile[2*LOCAL_SIZE_Y][LOCAL_SIZE_X + 1];
  const uint2 gid = (uint2)(get_global_id(0), get_global_id(1));
  const uint2 gsize = (uint2)(get_global_size(0), get_global_size(1));
  const uint2 tid = (uint2)(get_local_id(0), get_local_id(1));
  const uint M_ru = (M + LOCAL_SIZE_X - 1) & ~(LOCAL_SIZE_X - 1);
  const uint N_ru = (N + LOCAL_SIZE_Y - 1) & ~(LOCAL_SIZE_Y - 1);
  const uint2 tile_offset_tid = tid + (uint2)(LOCAL_SIZE_Y, LOCAL_SIZE_Y);

  for(uint i = gid.y; i < N_ru; i += gsize.y) {
    for(uint j = gid.x; j < M_ru; j += gsize.x) {
      uint2 tempu64;

      if(j < M && i < N) {
        tempu64 = as_uint2(in[j + i*M]);
        tile[tid.y][tid.x] = tempu64.x;
        tile[tile_offset_tid.y][tid.x] = tempu64.y;
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      uint2 ii = (uint2)(i - tid.y, j - tid.x) + tid;
      if(ii.x < N && ii.y < M) {
        tempu64 = (uint2)(tile[tid.x][tid.y], tile[tile_offset_tid.x][tid.y]);
        out[ii.x + ii.y*N] = as_double(tempu64);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }
  }
#else
  mat_transpose_opt1(in, out, M, N);
#endif
}

//
// Perform matrix multiplication: C = A * B, where A[M][K], B[K][N], C[M][N]
//
__attribute__((reqd_work_group_size(LOCAL_SIZE_X, LOCAL_SIZE_X, 1)))
__kernel void mat_mul(__global const REAL *A, __global const REAL *B, __global REAL *C, const uint4 MKN) {

  int2 i = (int2)(get_global_id(0), get_global_id(1));
  REAL a, b, c = 0.0;

  if(i.x < MKN.z && i.y < MKN.x) {
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

  __local REAL __attribute__((numbanks(LOCAL_SIZE_X), bankwith(_REAL_BANK_WITH))) tile[2*LOCAL_SIZE_X*LOCAL_SIZE_X];
  int2 gid = (int2)(get_global_id(0), get_global_id(1));
  int2 tid = (int2)(get_local_id(0), get_local_id(1));
  const int2 mn_bsign = (int2)(gid.y < MKN.x, gid.x < MKN.z);

  __local REAL *tile_a = (__local REAL *)tile;
  __local REAL *tile_b = (__local REAL *)tile + LOCAL_SIZE_X*LOCAL_SIZE_X;

  REAL c = 0.0;
  int2 kk;
  int2 kk2;
  int tile_index;

  for(int k = 0; k < MKN.y; k += LOCAL_SIZE_X) {

    kk = (int2)(k, k) + tid;
    kk2 = (int2)(kk.x < MKN.y, kk.y < MKN.y) & mn_bsign;

    tile_index = tid.y*LOCAL_SIZE_X+tid.x;

    barrier(CLK_LOCAL_MEM_FENCE);
    tile_a[tile_index] = kk2.x*A[gid.y*MKN.y + kk.x];
    tile_b[tile_index] = kk2.y*B[kk.y*MKN.z + gid.x];
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll (LOCAL_SIZE_X)
    for(int i = 0; i < LOCAL_SIZE_X; ++i) {
      c += tile_a[tid.y*LOCAL_SIZE_X+i] * tile_b[i*LOCAL_SIZE_X + tid.x];
    }
  }

  if(mn_bsign.x & mn_bsign.y)
    C[gid.y*MKN.z+gid.x] = c;
}
