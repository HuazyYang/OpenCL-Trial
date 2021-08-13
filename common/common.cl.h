#ifndef __COMMON_CL_H__
#define __COMMON_CL_H__

/** check fp arithmetic format */
#ifdef _USE_DOUBLE_FP
  #define REAL    double
  #define REAL2   double2
  #define REAL3   double3
  #define REAL4   double4 
  #define REAL16  double16
  #define REAL2x2 double2x2
  #define REAL3x3 double3x3
  #define REAL4x4 double4x4

  /** enable fp64 extension */
  #if defined(cl_amd_fp64)
      #pragma OPENCL EXTENSION cl_amd_fp64 : enable
  #elif defined(cl_khr_fp64)
      #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #endif /** enable fp64 extension */
#else /** _USE_DOUBLE_FP */
  #define REAL    float
  #define REAL2   float2
  #define REAL3   float3
  #define REAL4   float4 
  #define REAL16  float16
  #define REAL2x2 float2x2
  #define REAL3x3 float3x3
  #define REAL4x4 float4x4
#endif /** _USE_DOUBLE_FP */

#define _In_
#define _Inout_
#define _Out_
#define _In_shared_(a)

#endif /** __COMMON_CL_H__ */
