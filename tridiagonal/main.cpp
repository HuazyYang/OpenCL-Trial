#include <stdio.h>
#include "tridiagonal_mat.h"
#include "cpu_serial.h"
#include "test_cases.h"
#include <common_miscs.h>
#include <functional>
#include <cl_utils.h>
#include <memory>

#define SMALL_DIAGNAL_SYSTEM_MAX_DIM 256

static ycl_program g_pTridiagProgram;

void TestCPUSolvingDiagonalSystem(size_t dimx,
                                  const tridiagonal_mat<double> *A,
                                  const column_vec<double> *d,
                                  column_vec<double> *x0,
                                  column_vec<double> *x
                                  ) {

  std::unique_ptr<double[]> tmp_buffer;
  double *tmp[4];
  std::tuple<double, double, double> difference;

  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  size_t tmp_buffer_stride;
  tmp_buffer_stride = dimx + (dimx + 1) / 2;
  tmp_buffer.reset(new double[tmp_buffer_stride * 4]); // implies that 6 * n < 4 * (3 * n + 1) / 2
  tmp[0] = tmp_buffer.get();
  tmp[1] = tmp[0] + tmp_buffer_stride;
  tmp[2] = tmp[1] + tmp_buffer_stride;
  tmp[3] = tmp[2] + tmp_buffer_stride;

  start = hp_timer::now();
  cpu_solver::thomas_serial(A, d, x0, tmp[0], tmp[1]);
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("CPU Thomas Serializing elapsed:                                    "
         "%.3fms\n",
         elapsed.count());

  memset(x->v, -1, sizeof(double) * x->dim_y);
  start = hp_timer::now();
  cpu_solver::cyclic_reduction(A, d, x, tmp[0], tmp[1], tmp[2], tmp[3]);
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("CPU CR elapsed:                                                    "
         "%.3fms\n",
         elapsed.count());

  difference = compare_var(x->v, x0->v, x0->dim_y);
  printf("CPU CR difference: max: %.4f, mean: %.4f, sqrt_mean: %.4f\n", std::get<0>(difference),
         std::get<1>(difference), std::get<2>(difference));

  memset(x->v, -1, sizeof(double) * x->dim_y);
  start = hp_timer::now();
  cpu_solver::parallel_cyclic_reduction(A, d, x, tmp[0], tmp[1], tmp[2], tmp[3]);
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("CPU PCR elapsed:                                                   "
         "%.3fms\n",
         elapsed.count());

  difference = compare_var(x->v, x0->v, x0->dim_y);
  printf("CPU PCR difference: max: %.4f, mean: %.4f, sqrt_mean: %.4f\n", std::get<0>(difference),
         std::get<1>(difference), std::get<2>(difference));

  /*   memset(x.v, -1, sizeof(double) * x.dim_y);
    start = hp_timer::now();
    // This one tends to numeric instable in most cases.
    cpu_solver::recursive_doubling(&A, &d, &x, (double(*)[2][3])tmp[0], [](double c) { return std::abs(c) > 1.0e-5;
    }); fin = hp_timer::now(); elapsed = fmilliseconds_cast(fin - start); printf("CPU RD elapsed: "
           "%.3fms\n",
           elapsed.count());

    difference = compare_var(x.v, x0.v, x0.dim_y);
    printf("CPU RD difference: max: %.4f, mean: %.4f, sqrt_mean: %.4f\n",
           std::get<0>(difference), std::get<1>(difference),
           std::get<2>(difference)); */
}

CLHRESULT TestSolvingSmallDiagonalSystem(cl_command_queue cmd_queue, size_t dimx) {

  CLHRESULT hr = 0;
  cl_device_id device;
  cl_context context;
  ycl_kernel kernel;
  size_t buffer_len;
  cl_uint dimx32;
  cl_uint iterations32;
  cl_uint stride32;
  size_t local_mem_buffer_len;
  ycl_buffer a_d, b_d, c_d, d_d, x_d;
  uint64_t clr_pattern = -1ll;
  ycl_event done_ev;
  size_t local_size;

  std::uniform_int_distribution gen_pattern_distr(0, 3);
  tridiagonal_mat<double> A;
  column_vec<double> d;
  column_vec<double> x0, x;
  std::tuple<double, double, double> difference;

  hp_timer::time_point start, fin;
  fmilliseconds elapsed;

  A.alloc(dimx);
  d.alloc(dimx);
  x0.alloc(dimx);
  x.alloc(dimx);

  test_gen_cyclic(A.a, A.b, A.c, d.v, dimx, gen_pattern_distr(g_RandomEngine));

  printf("Input diagonal matrix dimension: %lld\n", dimx);

  TestCPUSolvingDiagonalSystem(dimx, &A, &d, &x0, &x);

  V_RETURN(clGetCommandQueueInfo(cmd_queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr));
  V_RETURN(clGetCommandQueueInfo(cmd_queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr));

  // OpenCL simulation
  buffer_len = A.dim_x * sizeof(double);
  V_RETURN2(a_d <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_len, A.a, &hr), hr);
  V_RETURN2(b_d <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_len, A.b, &hr), hr);
  V_RETURN2(c_d <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_len, A.c, &hr), hr);
  V_RETURN2(d_d <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_len, d.v, &hr), hr);
  V_RETURN2(x_d <<= clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, buffer_len, nullptr, &hr), hr);
  dimx32 = static_cast<cl_uint>(A.dim_x);
  local_mem_buffer_len = buffer_len * 5;
  iterations32 = cpu_solver::log2c(dimx);
  stride32 = 1;

  V_RETURN2(kernel <<= clCreateKernel(g_pTridiagProgram, "cr_small_system", &hr), hr);
  V_RETURN(SetKernelArguments(kernel, &a_d, &b_d, &c_d, &d_d, &x_d, &dimx32, &iterations32, &stride32,
                              local_mem_buffer_len));

  local_size = RoundC(dimx >> 1, 32);
  if(local_size == 0) local_size = 32;
  start = hp_timer::now();
  V_RETURN(clEnqueueFillBuffer(cmd_queue, x_d, &clr_pattern, sizeof(clr_pattern), 0, buffer_len, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, &local_size, &local_size, 0, nullptr, nullptr));
  V_RETURN(
      clEnqueueReadBuffer(cmd_queue, x_d, false, 0, buffer_len, x.v, 0, nullptr, done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("GPU CR(Small System) elapsed:                                      "
         "%.3fms\n",
         elapsed.count());

  difference = compare_var(x.v, x0.v, x0.dim_y);
  printf("GPU CR(Small System) difference: max: %.4f, mean: %.4f, sqrt_mean: "
         "%.4f\n",
         std::get<0>(difference), std::get<1>(difference), std::get<2>(difference));

  V_RETURN2(kernel <<= clCreateKernel(g_pTridiagProgram, "pcr_small_system", &hr), hr);
  local_mem_buffer_len = (buffer_len + sizeof(double)) * 4 + buffer_len;
  V_RETURN(SetKernelArguments(kernel, &a_d, &b_d, &c_d, &d_d, &x_d, &dimx32, &iterations32, &stride32,
                              local_mem_buffer_len));

  local_size = RoundC(dimx, 32);
  start = hp_timer::now();
  V_RETURN(clEnqueueFillBuffer(cmd_queue, x_d, &clr_pattern, sizeof(clr_pattern), 0, buffer_len, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, kernel, 1, nullptr, &local_size, &local_size, 0, nullptr, nullptr));
  V_RETURN(
      clEnqueueReadBuffer(cmd_queue, x_d, false, 0, buffer_len, x.v, 0, nullptr, done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));
  fin = hp_timer::now();
  elapsed = fmilliseconds_cast(fin - start);
  printf("GPU PCR(Small System) elapsed:                                     "
         "%.3fms\n",
         elapsed.count());

  difference = compare_var(x.v, x0.v, x0.dim_y);
  printf("GPU PCR(Small System) difference: max: %.4f, mean: %.4f, sqrt_mean: "
         "%.4f\n",
         std::get<0>(difference), std::get<1>(difference), std::get<2>(difference));

  return hr;
}

int main() {

  CLHRESULT hr;
  ycl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;

  V_RETURN(FindOpenCLPlatform(CL_DEVICE_TYPE_GPU, {"AMD", "NVIDIA"}, {}, &platform, &device));
  V_RETURN(CreateDeviceContext(platform, device, &context));
  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

  V_RETURN(CreateProgramFromILFile(context, device, "OCL-SpirV/tridiagonal.spv", &g_pTridiagProgram));

  std::uniform_int_distribution<size_t> sm_diag_dim_distr(1, 256);

  for (ptrdiff_t i = 0; i < 110; ++i) {
    printf("Test case[%lld] -- Small System\n", i + 1);
    TestSolvingSmallDiagonalSystem(cmd_queue, sm_diag_dim_distr(g_RandomEngine));
    printf("\n");
  }
}
