#include <cl_utils.h>
#include <vector>
#include <stdio.h>
#include <random>
#include <chrono>
#include <array>

std::default_random_engine g_RandomEngine{[]() -> std::random_device::result_type {
  std::random_device rdev;
  return rdev();
}()};

using hp_timer = std::chrono::high_resolution_clock;

template <class REAL> static std::vector<REAL> gen_random_matrix(size_t cols, size_t rows) {

  std::uniform_real_distribution<REAL> rd{(REAL)-100.0, (REAL)100.0};

  size_t mat_len = cols * rows, i;
  std::vector<double> mat(mat_len);
  for (i = 0; i < mat_len; ++i) {
    mat[i] = rd(g_RandomEngine);
  }

  return mat;
}

template<class REAL>
void print_matrix(int indent, int cwidth, const std::vector<REAL> &mat, size_t cols, size_t rows) {

  ptrdiff_t i, j, k;

  k = -1;
  for(i = 0; i < rows; ++i) {
    printf("%*c", indent, ' ');
    for (j = 0; j < cols; ++j)
      printf("%*c%g", cwidth, ' ', mat[++k]);
    putc('\n', stdout);
  }
}

static ycl_program g_pMatrixProgram;

CLHRESULT TestMatrixTransposeProfile(
    cl_context context, cl_device_id device, cl_command_queue cmd_queue, size_t ncols, size_t nrows) {

  CLHRESULT hr;

  const size_t mat_buff_size = ncols * nrows * sizeof(double);

  auto test_mat = gen_random_matrix<double>(ncols, nrows);
  // printf("Input matrix\n");
  // print_matrix(8, 12, test_mat, mat_cols, mat_rows);

  ycl_buffer input_mat_buff, output_mat_buff;

  V_RETURN((input_mat_buff <<=
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mat_buff_size, test_mat.data(), &hr),
            hr));
  V_RETURN((output_mat_buff <<= clCreateBuffer(context, CL_MEM_WRITE_ONLY, mat_buff_size, nullptr, &hr), hr));

  ycl_kernel ker;
  V_RETURN((ker <<= clCreateKernel(g_pMatrixProgram, "mat_transpose", &hr), hr));

  using hp_timer = std::chrono::high_resolution_clock;
  hp_timer::time_point start, fin2;
  std::chrono::microseconds elapsed;

  printf("Input matrix dimensions: (%llu, %llu)\n", nrows, ncols);

  int M = static_cast<int>(ncols), N = static_cast<int>(nrows);
  V_RETURN(SetKernelArguments(ker, &input_mat_buff, &output_mat_buff, &M, &N));

  size_t lworksize[3];
  V_RETURN(
      clGetKernelWorkGroupInfo(ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(lworksize), lworksize, nullptr));

  size_t gworksize[] = {RoundF(ncols, lworksize[0]), RoundF(nrows, lworksize[1])};
  ycl_event rd_done_ev;

  size_t max_work_size[3];
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_size), max_work_size, nullptr));

  gworksize[0] = std::min(gworksize[0], max_work_size[0]);
  gworksize[1] = std::min(gworksize[0], max_work_size[1]);

  start = hp_timer::now();
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker, 2, nullptr, gworksize, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, output_mat_buff, false, 0, mat_buff_size, (void *)test_mat.data(), 0, nullptr,
                               &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin2 = hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin2 - start);
  printf("Transpose matrix(Use only global storage) elapsed:     %12lld us\n", elapsed.count());

  // printf("Transposed matrix: (Use only global storage) ");
  // print_matrix(8, 12, test_mat, mat_rows, mat_cols);

  ycl_kernel ker_opt1;
  V_RETURN((ker_opt1 <<= clCreateKernel(g_pMatrixProgram, "mat_transpose_opt1", &hr), hr));

  V_RETURN(SetKernelArguments(ker_opt1, &input_mat_buff, &output_mat_buff, &M, &N));
  double data_pattern = 0.0;
  V_RETURN(clEnqueueFillBuffer(cmd_queue, output_mat_buff, &data_pattern, sizeof(data_pattern), 0,
                               mat_buff_size / sizeof(data_pattern), 0, nullptr, nullptr));
  start = hp_timer::now();
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker_opt1, 2, nullptr, gworksize, nullptr, 0, nullptr, nullptr));
  rd_done_ev = nullptr;
  V_RETURN(clEnqueueReadBuffer(cmd_queue, output_mat_buff, false, 0, mat_buff_size, (void *)test_mat.data(), 0, nullptr,
                               &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin2 = hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin2 - start);
  printf("Transpose matrix(Use shared local storage) elapsed:    %12lld us\n", elapsed.count());

  // printf("Transposed matrix: (Use shared local storage): ");
  // print_matrix(8, 12, test_mat, mat_rows, mat_cols);

  ycl_kernel ker_opt2;
  V_RETURN((ker_opt2 <<= clCreateKernel(g_pMatrixProgram, "mat_transpose_opt2", &hr), hr));

  V_RETURN(SetKernelArguments(ker_opt2, &input_mat_buff, &output_mat_buff, &M, &N));
  V_RETURN(clEnqueueFillBuffer(cmd_queue, output_mat_buff, &data_pattern, sizeof(data_pattern), 0,
                               mat_buff_size / sizeof(data_pattern), 0, nullptr, nullptr));
  start = hp_timer::now();
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker_opt2, 2, nullptr, gworksize, nullptr, 0, nullptr, nullptr));
  rd_done_ev = nullptr;
  V_RETURN(clEnqueueReadBuffer(cmd_queue, output_mat_buff, false, 0, mat_buff_size, (void *)test_mat.data(), 0, nullptr,
                               &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin2 = hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin2 - start);
  printf("Transpose matrix(Eliminate bank conflict) elapsed:     %12lld us\n", elapsed.count());
  // printf("Transposed matrix: (Eliminate bank conflict): ");
  // print_matrix(8, 12, test_mat, mat_rows, mat_cols);

  return hr;
}


CLHRESULT TestMatrixMulitplicationProfile(
  cl_context context, cl_device_id device, cl_command_queue cmd_queue, size_t M, size_t K, size_t N) {

  CLHRESULT hr;
  ycl_kernel mul_ker;
  ycl_buffer a_buffer, b_buffer, c_buffer;

  printf("Input matrix A dimensions: (%llu, %llu)\n", M, K);
  printf("Input matrix B dimensions: (%llu, %llu)\n", K, N);

  const size_t a_data_size = M * K;
  const size_t b_data_size = K * N;
  const size_t c_data_size = M * N;
  const size_t a_buffer_size = a_data_size * sizeof(double);
  const size_t b_buffer_size = b_data_size * sizeof(double);
  const size_t c_buffer_size = c_data_size * sizeof(double);
  std::vector<double> a_data, b_data, /*c_data,*/ c_data2;
  const double dbl_eq_tol = 1.0E-8;

  a_data = gen_random_matrix<double>(K, M);
  b_data = gen_random_matrix<double>(N, K);
  c_data2.resize(c_data_size);

  // c_data.resize(c_data_size);
  // for(ptrdiff_t i = 0; i < M; ++i) {
  //   ptrdiff_t ii = i*K;
  //   for(ptrdiff_t j = 0; j < N; ++j) {
  //     double c = 0.0;
  //     for(ptrdiff_t k = 0; k < K; ++k) {
  //       c += a_data[ii+k]*b_data[k*N+j];
  //     }
  //     c_data[i*N+j] = c;
  //   }
  // }

  V_RETURN((a_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        a_buffer_size, a_data.data(), &hr),
            hr));
  V_RETURN((b_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        b_buffer_size, b_data.data(), &hr),
            hr));
  V_RETURN(
      (c_buffer <<= clCreateBuffer(context, CL_MEM_WRITE_ONLY, c_buffer_size, nullptr, &hr),
       hr));
  
  // Free some deprecated memory.
  a_data = std::vector<double>();
  b_data = a_data;

  V_RETURN((mul_ker <<= clCreateKernel(g_pMatrixProgram, "mat_mul", &hr), hr));

  size_t group_size[3];
  size_t global_size[2];
  const double dbl_zero = 0.0;

  V_RETURN(clGetKernelWorkGroupInfo(mul_ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(group_size), group_size, nullptr));
  global_size[0] = RoundF(N, group_size[0]);
  global_size[1] = RoundF(M, group_size[1]);

  std::array<uint32_t, 4> MKN{ (uint32_t)M, (uint32_t)K, (uint32_t)N };
  V_RETURN(SetKernelArguments(mul_ker, &a_buffer, &b_buffer, &c_buffer, &MKN));

  hp_timer::time_point start, fin;
  std::chrono::microseconds elapsed;

  ycl_event rd_done_ev;

  start = hp_timer::now();

  V_RETURN(clEnqueueFillBuffer(cmd_queue, c_buffer, &dbl_zero, sizeof(dbl_zero), 0, c_data_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, mul_ker, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, c_buffer, false, 0, c_buffer_size, (void *)c_data2.data(), 0, nullptr, &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));

  V_RETURN(clWaitForEvents(1, &rd_done_ev));

  fin = hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin - start);
  printf("Matrix multiplication(Use global storage only) elapsed:    %12lld us\n",elapsed.count());

  // // Verify the results.
  // for(ptrdiff_t i = 0; i < M; ++i) {
  //   ptrdiff_t ii = i*N;
  //   for(ptrdiff_t j = 0; j < N; ++j) {
  //     if(abs(c_data[ii+j] - c_data2[ii+j]) >= dbl_eq_tol) {
  //       printf("matrix multiplication results do not coincide: (%lld, %lld)", i, j);
  //     }
  //   }
  // }

  V_RETURN((mul_ker <<= clCreateKernel(g_pMatrixProgram, "mat_mul_opt1", &hr), hr));
  V_RETURN(SetKernelArguments(mul_ker, &a_buffer, &b_buffer, &c_buffer, &MKN));

  start = hp_timer::now();

  V_RETURN(clEnqueueFillBuffer(cmd_queue, c_buffer, &dbl_zero, sizeof(dbl_zero), 0, c_data_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, mul_ker, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, c_buffer, false, 0, c_buffer_size, (void *)c_data2.data(), 0, nullptr,
                               rd_done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));

  V_RETURN(clWaitForEvents(1, &rd_done_ev));

  fin = hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin - start);
  printf("Matrix multiplication(Use shared local storage) elapsed:   %12lld us\n", elapsed.count());

  // // Verify the results.
  // for (ptrdiff_t i = 0; i < M; ++i) {
  //   ptrdiff_t ii = i * N;
  //   for (ptrdiff_t j = 0; j < N; ++j) {
  //     if (abs(c_data[ii + j] - c_data2[ii + j]) >= dbl_eq_tol) {
  //       printf("matrix multiplication results do not coincide: (%lld, %lld)", i, j);
  //     }
  //   }
  // }

  return hr;
}

int main() {

  CLHRESULT hr;
  cl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;

  V_RETURN(FindOpenCLPlatform(nullptr, CL_DEVICE_TYPE_GPU, &platform));
  V_RETURN(CreateDeviceContext(platform, CL_DEVICE_TYPE_GPU,  &device, &context));
  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

  V_RETURN(CreateProgramFromFile(context, device, "matrix.cl", &g_pMatrixProgram));

  std::uniform_int_distribution<size_t> transpose_ncols_nrows_distr(251, 10007);
  std::uniform_int_distribution<size_t> mul_ncols_nrows_distr(51, 5007);

  TestMatrixTransposeProfile(context, device, cmd_queue, transpose_ncols_nrows_distr(g_RandomEngine),
                             transpose_ncols_nrows_distr(g_RandomEngine));

  TestMatrixMulitplicationProfile(context, device, cmd_queue, mul_ncols_nrows_distr(g_RandomEngine),
                                  mul_ncols_nrows_distr(g_RandomEngine), mul_ncols_nrows_distr(g_RandomEngine));
}