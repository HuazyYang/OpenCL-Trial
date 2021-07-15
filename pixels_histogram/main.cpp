#include <cl_utils.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <chrono>

static ycl_program g_pHistoProgram;

std::default_random_engine g_RandomEngine{[]() -> std::random_device::result_type {
  std::random_device rdev;
  return rdev();
}()};

std::vector<uint8_t> CreateGrayscaleImageData(uint16_t width, uint16_t height) {

  uint32_t len = width * height;
  std::vector<uint8_t> pixel_buffer;
  std::uniform_int_distribution<uint16_t> cd(0, 0xffff);

  pixel_buffer.resize(len);
  for(int32_t i = 0; i < len; ++i) {
    pixel_buffer[i] = cd(g_RandomEngine) & 0xff;
  }

  return pixel_buffer;
}

void PrintHistogram(const uint32_t *histo) {

  printf("-------------------------------------------------------------------------\n");
  for(int32_t i = 0; i < 256; i += 16) {
    printf("    "
      "%4u, %4u, %4u, %4u, "
      "%4u, %4u, %4u, %4u, "
      "%4u, %4u, %4u, %4u, "
      "%4u, %4u, %4u, %4u, \n",
      histo[i], histo[i+1], histo[i+2], histo[i+3],
      histo[i+4], histo[i+5], histo[i+6], histo[i+7],
      histo[i+8], histo[i+9], histo[i+10], histo[i+11],
      histo[i+12], histo[i+13], histo[i+14], histo[i+15]);
  }
  printf("-------------------------------------------------------------------------\n");
}

int main() {

  CLHRESULT hr;
  const char *preferred_plats[] = {"NVIDIA CUDA", "AMD", nullptr };
  ycl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;

  V_RETURN(FindOpenCLPlatform(preferred_plats, CL_DEVICE_TYPE_GPU, &platform));

  V_RETURN(CreateDeviceContext(platform, CL_DEVICE_TYPE_GPU, &device, &context));

  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

  V_RETURN(CreateProgramFromFile(context, device, nullptr, "histo.cl", &g_pHistoProgram));

  std::uniform_int_distribution<uint16_t> dimid(1000, 10000);

  auto pixels_data = CreateGrayscaleImageData(dimid(g_RandomEngine), dimid(g_RandomEngine));
  uint32_t pixels_num = pixels_data.size();
  const size_t histo_num = 256;
  const size_t histo_buff_size = histo_num << 2;

  ycl_mem pixel_buff, histo_buff;

  V_RETURN((pixel_buff <<=
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, pixels_num, pixels_data.data(), &hr),
            hr));
  V_RETURN(
      (histo_buff <<= clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, histo_buff_size, nullptr, &hr),
       hr));

  ycl_kernel atomic_ker, atomic_coalesced_ker, optd_ker;
  V_RETURN((atomic_ker <<= clCreateKernel(g_pHistoProgram, "histo_atomic", &hr), hr));
  V_RETURN((atomic_coalesced_ker <<= clCreateKernel(g_pHistoProgram, "hosto_atomic_coalesced", &hr), hr));
  V_RETURN((optd_ker <<= clCreateKernel(g_pHistoProgram, "histo_optimized_ultimate", &hr), hr));

  cl_uint cu_cap;
  cl_ulong local_mem_cap;

  size_t group_size[3];
  uint32_t histo_data[256];
  uint32_t histo_init_data = 0;

  uint32_t histo_data2[256];
  memset(histo_data2, 0, sizeof(histo_data2));
  for (int32_t i = 0; i < pixels_num; ++i) {
    histo_data2[pixels_data[i]]++;
  }

  ycl_event rd_done_ev;

  using hp_timer = std::chrono::high_resolution_clock;
  hp_timer::time_point start, fin;
  std::chrono::microseconds elapsed;

  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu_cap), &cu_cap, nullptr));
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_cap), &local_mem_cap, nullptr));

  // Atomic Histogram
  V_RETURN(clGetKernelWorkGroupInfo(atomic_ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(group_size), group_size, nullptr));
  size_t work_item_size = group_size[0] * cu_cap * 2;

  V_RETURN(SetKernelArguments(atomic_ker, &pixel_buff, &pixels_num, &histo_buff));
  start = hp_timer::now();
  V_RETURN(clEnqueueFillBuffer(cmd_queue, histo_buff, &histo_init_data, sizeof(histo_init_data), 0, histo_buff_size, 0,
                               nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, atomic_ker, 1, nullptr, &work_item_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, histo_buff, false, 0, histo_buff_size, histo_data, 0, nullptr,
                               rd_done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin =  hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin - start);
  printf("Atomics Histogram elapsed: %lldus, coincident: %s\n", elapsed.count(),
         memcmp(histo_data, histo_data2, histo_buff_size) == 0 ? "true" : "false");

  // Amotics Coalesced Histogram
  V_RETURN(clGetKernelWorkGroupInfo(atomic_coalesced_ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(group_size),
                                    group_size, nullptr));
  work_item_size = group_size[0] * cu_cap * 2;

  V_RETURN(SetKernelArguments(atomic_coalesced_ker, &pixel_buff, &pixels_num, &histo_buff));
  start = hp_timer::now();
  V_RETURN(clEnqueueFillBuffer(cmd_queue, histo_buff, &histo_init_data, sizeof(histo_init_data), 0, histo_buff_size, 0,
                               nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, atomic_coalesced_ker, 1, nullptr, &work_item_size, nullptr, 0, nullptr,
                                  nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, histo_buff, false, 0, histo_buff_size, histo_data, 0, nullptr,
                               rd_done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin = hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin - start);
  printf("Atomics Coalesced Histogram elapsed: %lldus, coincident: %s\n", elapsed.count(),
         memcmp(histo_data, histo_data2, histo_buff_size) == 0 ? "true" : "false");

  // Optimize to Ultimate Histogram
  V_RETURN(clGetKernelWorkGroupInfo(optd_ker, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(group_size),
                                    group_size, nullptr));
  work_item_size = group_size[0] * cu_cap * 2;

  V_RETURN(SetKernelArguments(optd_ker, &pixel_buff, &pixels_num, &histo_buff));
  start = hp_timer::now();
  V_RETURN(clEnqueueFillBuffer(cmd_queue, histo_buff, &histo_init_data, sizeof(histo_init_data), 0, histo_buff_size, 0,
                               nullptr, nullptr));
  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, optd_ker, 1, nullptr, &work_item_size, nullptr, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, histo_buff, false, 0, histo_buff_size, histo_data, 0, nullptr,
                               rd_done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &rd_done_ev));
  fin = hp_timer::now();
  elapsed = std::chrono::duration_cast<std::chrono::microseconds>(fin - start);
  printf("Optimizely Ultimate Histogram elapsed: %lldus, coincident: %s\n", elapsed.count(),
         memcmp(histo_data, histo_data2, histo_buff_size) == 0 ? "true" : "false");

  return 0;
}