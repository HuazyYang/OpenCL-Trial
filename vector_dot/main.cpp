#include <cl_utils.h>
#include <stdio.h>
#include <vector>
#include <random>

std::default_random_engine g_RandomEngine{[]() -> std::random_device::result_type {
  std::random_device rdev;
  return rdev();
}()};

template<typename T>
std::vector<T> generate_random_vector(uint32_t n) {

  std::uniform_real_distribution<T> fd((T)-1.0, (T)1.0);
  std::vector<T> v;
  v.resize(n);

  for(auto it = v.begin(); it != v.end(); ++it) {
    *it = fd(g_RandomEngine);
  }

  return v;
}

template<typename T, typename = std::enable_if_t<std::is_same_v<T, float>||std::is_same_v<T, double>>>
CLHRESULT TestVectorDot(cl_context context, cl_device_id device, cl_command_queue cmd_queue, uint32_t n) {

  CLHRESULT hr;
  constexpr size_t ElementSize = sizeof(T);
  const char *program_defs = ElementSize == 8 ? "#define _USE_DOUBLE_FP\n" : nullptr;
  ycl_program program;
  ycl_kernel ker;

  V_RETURN(CreateProgramFromFile(context, device, program_defs, "vector_dot.cl", program.ReleaseAndGetAddressOf()));
  V_RETURN((ker <<= clCreateKernel(program, "vector_dist_sqr_reduced", &hr), hr));

  std::vector<T> a_data, b_data;
  size_t a_buffer_size = n * ElementSize;
  size_t c_temp_buffer_size;

  a_data = generate_random_vector<T>(n);
  b_data = generate_random_vector<T>(n);

  ycl_buffer a_buffer, b_buffer, c_temp_buffer;

  V_RETURN(
      (a_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_buffer_size, a_data.data(), &hr),
       hr));
  V_RETURN(
      (b_buffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a_buffer_size, b_data.data(), &hr),
       hr));

  size_t max_work_size[3];
  size_t local_size[3];
  size_t group_count;
  size_t group_size;

  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_size), max_work_size, nullptr));
  V_RETURN(clGetKernelWorkGroupInfo(ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(local_size), local_size, nullptr));

  group_size = std::min((size_t)n, max_work_size[0]);
  group_count =  group_size / local_size[0] + ((group_size & (local_size[0]-1)) ? 1 : 0);
  group_size = group_count * local_size[0];
  c_temp_buffer_size = group_count * ElementSize;

  V_RETURN2(c_temp_buffer <<=
            clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY, c_temp_buffer_size, nullptr, &hr),
            hr);

  V_RETURN(SetKernelArguments(ker, &a_buffer, &b_buffer, &n, &c_temp_buffer));

  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker, 1, nullptr, &group_size, nullptr, 0, nullptr, nullptr));

  T dot_res;
  ycl_event rd_done_ev;
  V_RETURN(clEnqueueReadBuffer(cmd_queue, c_temp_buffer, false, 0, sizeof(dot_res), &dot_res, 0, nullptr, &rd_done_ev));
  V_RETURN(clFlush(cmd_queue));

  clWaitForEvents(1, &rd_done_ev);

  T dot_res2 = (T)0.0;
  const T dot_tol =  ElementSize == 4 ? (T)1.0E-3 : (T)1.0E-6;
  T temp;

  for(auto it_a = a_data.begin(), it_b = b_data.begin(); it_a != a_data.end(); ++it_a, ++it_b) {
    temp = *it_a - *it_b;
    dot_res2 += temp * temp;
  }

  printf("Dot result of vector dim(%d): %g, expect: %g, coincident: %s\n", n, dot_res, dot_res2,
         std::abs(dot_res - dot_res2) < dot_tol ? "true" : "false");

  return hr;
}


int main() {

  CLHRESULT hr;
  ycl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;

  V_RETURN(FindOpenCLPlatform(CL_DEVICE_TYPE_GPU, {"NVIDIA CUDA", "AMD"}, {}, &platform, &device));
  V_RETURN(CreateDeviceContext(platform, device, &context));
  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

  std::uniform_int_distribution<uint32_t> id(10, 100000);

  V_RETURN(TestVectorDot<float>(context, device, cmd_queue, id(g_RandomEngine)));

  V_RETURN(TestVectorDot<double>(context, device, cmd_queue, id(g_RandomEngine)));

  return hr;
}