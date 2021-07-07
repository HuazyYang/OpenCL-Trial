#include <cl_utils.h>
#include <vector>
#include <random>
#include <fstream>

const size_t buff_size_x = 800;
const size_t buff_size_y = 600;

void CreateSourceBufferData(std::vector<float> &src1, std::vector<float> &src2) {

  std::random_device rdev;
  std::default_random_engine dre{rdev()};
  std::uniform_real_distribution<float> rd{-2.0f, 12.0f};
  ptrdiff_t i = 0;
  const size_t buff_size = buff_size_x * buff_size_y;

  src1.resize(buff_size);
  src2.resize(buff_size);

  for(i = 0; i < buff_size; ++i) {
    src1[i] = rd(dre);
    src2[i] = rd(dre);
  }
}

CLHRESULT TestAddSample(cl_context context, cl_command_queue cmd_queue, cl_program program) {

  CLHRESULT hr;
  ycl_kernel kernel;

  kernel <<= clCreateKernel(program, "Add", &hr);
  V_RETURN(hr);

  std::vector<float> data_src1, data_src2, data_dest;

  CreateSourceBufferData(data_src1, data_src2);

  cl_image_format img_format = {};
  cl_image_desc img_desc = {};
  img_format.image_channel_data_type = CL_FLOAT;
  img_format.image_channel_order = CL_R;

  img_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  img_desc.image_width = buff_size_x;
  img_desc.image_height = buff_size_y;
  img_desc.image_depth = 0;
  img_desc.image_array_size = 1;
  img_desc.image_row_pitch = 0;
  img_desc.image_slice_pitch = 0;
  img_desc.num_mip_levels = 0;
  img_desc.num_samples = 0;

  ycl_mem img_src1, img_src2, img_dest;

  V_RETURN((img_src1 <<= clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format, &img_desc,
                                     data_src1.data(), &hr),
            hr));
  V_RETURN((img_src2 <<= clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &img_format, &img_desc,
                                     data_src2.data(), &hr),
            hr));
  V_RETURN((img_dest <<=
                clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, &img_format, &img_desc, nullptr, &hr),
            hr));

  V_RETURN(clSetKernelArg(kernel, 0, sizeof(cl_mem), &img_src1));
  V_RETURN(clSetKernelArg(kernel, 1, sizeof(cl_mem), &img_src2));
  V_RETURN(clSetKernelArg(kernel, 2, sizeof(cl_mem), &img_dest));

  data_dest.resize(buff_size_x * buff_size_y);
  float *resptr = data_dest.data();
  size_t origin[] = {0, 0, 0};
  size_t region[] = {buff_size_x, buff_size_y, 1};
  ycl_event dest_event;

  size_t workSize[] = {buff_size_x, buff_size_y};

  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, kernel, 2, nullptr, workSize, nullptr, 0, nullptr, nullptr));

  V_RETURN(clEnqueueReadImage(cmd_queue, img_dest, false, origin, region, 0, 0, resptr, 0, nullptr, dest_event.ReleaseAndGetAddressOf()));

  V_RETURN(clFlush(cmd_queue));

  V_RETURN(clWaitForEvents(1, &dest_event));

  // Verify the result.
  size_t buff_size = buff_size_x * buff_size_y;

  for (ptrdiff_t i = 0; i < buff_size; ++i) {
    float res = data_src1[i] + data_src2[i];
    if (res != resptr[i]) {
      printf("(%lld, %lld) expect %g, but result is: %g\n", i / buff_size_y, i % buff_size_y, res, resptr[i]);
    }
  }

  return hr;
}

CLHRESULT TestComputePi(cl_context context, cl_command_queue cmd_queue, cl_program program) {

  CLHRESULT hr;
  ycl_kernel ker1, ker2;
  ycl_mem temp_buffer;

  const size_t local_size = 128;
  const size_t global_work_size = local_size * 128;
  const uint32_t num_division = 1 << 26;
  ycl_event done_ev;
  double pi_result;

  V_RETURN((ker1 <<= clCreateKernel(program, "PI1", &hr), hr));
  V_RETURN((ker2 <<= clCreateKernel(program, "PI2", &hr), hr));

  V_RETURN((temp_buffer <<= clCreateBuffer(context, CL_MEM_READ_WRITE, local_size * sizeof(double), nullptr, &hr), hr));

  V_RETURN(clSetKernelArg(ker1, 0, sizeof(uint32_t), &num_division));
  V_RETURN(clSetKernelArg(ker1, 1, sizeof(double) * local_size, nullptr));
  V_RETURN(clSetKernelArg(ker1, 2, sizeof(cl_mem), &temp_buffer));

  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker1, 1, nullptr, &global_work_size, &local_size, 0, nullptr, nullptr));

  clEnqueueBarrier(cmd_queue);

  uint32_t local_size32 = local_size;
  clSetKernelArg(ker2, 0, sizeof(uint32_t), &local_size32);
  clSetKernelArg(ker2, 1, sizeof(double) * local_size, nullptr);
  clSetKernelArg(ker2, 2, sizeof(cl_mem), &temp_buffer);

  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker2, 1, nullptr, &local_size, &local_size, 0, nullptr, nullptr));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, temp_buffer, false, 0, sizeof(pi_result), &pi_result, 0, nullptr, done_ev.ReleaseAndGetAddressOf()));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));

  printf("Compute PI result: %.18f\n", pi_result);

  return hr;
}

int main() {

  CLHRESULT hr;
  cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
  ycl_platform_id platform_id;
  ycl_context context;
  ycl_device_id device;
  ycl_command_queue cmd_queue;
  ycl_program program;

  V_RETURN(FindOpenCLPlatform(nullptr, dev_type, &platform_id));

  V_RETURN(CreateDeviceContext(platform_id, dev_type, device.ReleaseAndGetAddressOf(), context.ReleaseAndGetAddressOf()));

  V_RETURN(CreateCommandQueue(context, device, cmd_queue.ReleaseAndGetAddressOf()));

  std::ifstream fin("simple.cl", std::ios::binary);
  if(!fin) {
    CL_TRACE(-1, "Read CL program source file error!\n");
    return -1;
  }
  std::vector<char> src_buffer;
  size_t src_buffer_len;

  fin.seekg(0, std::ios::end);
  src_buffer_len = fin.tellg();
  fin.seekg(0, fin.beg);
  src_buffer.resize(src_buffer_len);
  fin.read(src_buffer.data(), src_buffer_len);
  fin.close();

  V_RETURN(CreateProgramFromSource(context, device, src_buffer.data(), src_buffer_len, program.ReleaseAndGetAddressOf()));
  src_buffer.clear();

  TestAddSample(context, cmd_queue, program);

  TestComputePi(context, cmd_queue, program);

  V_RETURN(clFinish(cmd_queue));
}