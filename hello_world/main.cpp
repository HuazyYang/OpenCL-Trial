#include <cl_utils.h>
#include <vector>
#include <random>
#include <fstream>

CLHRESULT FindOpenCLPlatform(const char* preferred_plat, cl_device_type dev_type, cl_platform_id *plat_id);
CLHRESULT CreateDeviceContext(cl_platform_id plat_id, cl_device_type dev_type, cl_device_id *dev, cl_context *dev_ctx);
CLHRESULT CreateCommandQueue(cl_device_id dev, cl_command_queue *cmd_queue);
CLHRESULT CreateProgramFromSource(
    cl_context context,
    cl_device_id device,
    const char *source,
    size_t src_len,
    cl_program *program
);

CLHRESULT FindOpenCLPlatform(const char *preferred_plat, cl_device_type dev_type, cl_platform_id *plat_id) {

  CLHRESULT hr;
  cl_uint numPlatform;
  cl_platform_id plat_id2 = nullptr;

  V_RETURN(clGetPlatformIDs(0, nullptr, &numPlatform));
  if(numPlatform == 0) {
    hr = -1;
    CL_TRACE(hr, "No Platform found!\n");
    return hr;
  }

  std::vector<cl_platform_id> platforms{numPlatform};

  V_RETURN(clGetPlatformIDs(numPlatform, &platforms[0], nullptr));

  if (preferred_plat != nullptr && preferred_plat[0]) {

    size_t nlength;
    std::vector<char> name;
    cl_uint numDevices;

    for(auto itPlat = platforms.begin(); itPlat != platforms.end(); ++itPlat) {

      V_RETURN(clGetPlatformInfo(*itPlat, CL_PLATFORM_NAME, 0, nullptr, &nlength));

      name.resize(nlength+1);
      name[nlength] = 0;

      V_RETURN(clGetPlatformInfo(*itPlat, CL_PLATFORM_NAME, nlength, &name[0], nullptr));

      if (_stricmp(name.data(), preferred_plat) == 0) {

        V_RETURN(clGetDeviceIDs(*itPlat, dev_type, 0, nullptr, &numDevices));
        if(numDevices == 0) {
          hr = -1;
          CL_TRACE(hr, "Error: Required device type does not exist on specified platform!\n");
          return hr;
        }

        plat_id2 = *itPlat;
        break;
      }
    }
  } else {
    plat_id2 = platforms[0];
  }

  *plat_id = plat_id2;
  return plat_id2 ? CL_SUCCESS : -1;
}

CLHRESULT CreateDeviceContext(cl_platform_id plat_id, cl_device_type dev_type, cl_device_id *dev, cl_context *dev_ctx) {
  CLHRESULT hr;

  cl_context_properties ctx_props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)plat_id, 0};
  *dev_ctx = clCreateContextFromType(ctx_props, dev_type, nullptr, nullptr, &hr);
  V_RETURN(hr);

  V_RETURN(clGetContextInfo(*dev_ctx, CL_CONTEXT_DEVICES, sizeof(*dev), dev, nullptr));

  return hr;
}

CLHRESULT CreateCommandQueue(cl_context dev_ctx, cl_device_id device, cl_command_queue *cmd_queue) {

  CLHRESULT hr;
  size_t nlength;
  std::vector<char> strver;
  char *p_ver, *token_ctx;
  float ver;

  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &nlength));
  if(nlength == 0) {
    hr = -1;
    CL_TRACE(hr, "Device version is not available!\n");
    return hr;
  }

  strver.resize(nlength + 1, 0);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_VERSION, nlength, &strver[0], nullptr));

  if ((p_ver = strtok_s(&strver[0], " ", &token_ctx)) != nullptr
    && (p_ver = strtok_s(nullptr, " ", &token_ctx)) != nullptr) {
    ver = strtof(p_ver, nullptr);
  } else {
    ver = 1.2f;
  }

  if(ver >= 2.0f) {
    const cl_command_queue_properties cmd_queue_props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    *cmd_queue = clCreateCommandQueueWithProperties(dev_ctx, device, cmd_queue_props, &hr);
    V_RETURN(hr);
  } else if(ver >= 1.2f) {

    const cl_command_queue_properties cmd_queue_props[] = {CL_QUEUE_PROFILING_ENABLE};
    *cmd_queue = clCreateCommandQueue(dev_ctx, device, cmd_queue_props[0], &hr);
    V_RETURN(hr);
  } else {
    hr = -1;
    CL_TRACE(hr, "OpenCL version is unknown!\n");
    return hr;
  }

  return hr;
}

CLHRESULT CreateProgramFromSource(
    cl_context context, cl_device_id device, const char *source, size_t src_len, cl_program *program) {

  CLHRESULT hr;
  char predefines[] = "#define REAL double\n";
  const char * sources[] = {predefines, source};
  size_t src_lens[] = { _countof(predefines)-1, src_len };

  *program = clCreateProgramWithSource(context, 2, sources, src_lens, &hr);
  V_RETURN(hr);

  hr = clBuildProgram(*program, 1, &device, "", nullptr, nullptr);
  if (hr == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size = 0;
    std::vector<char> build_log;

    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);

    build_log.resize(log_size);
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), nullptr);
    CL_TRACE(hr, "Build CL program erorr, log:\n%s\n", build_log.data());
  }

  if (CL_FAILED(hr)) {
    *program = nullptr;
    V_RETURN(hr);
  }

  return hr;
}

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

  V_RETURN(clEnqueueReadImage(cmd_queue, img_dest, false, origin, region, 0, 0, resptr, 0, nullptr, &dest_event));

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

  V_RETURN((temp_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, local_size * sizeof(double), nullptr, &hr), hr));

  V_RETURN(clSetKernelArg(ker1, 0, sizeof(uint32_t), &num_division));
  V_RETURN(clSetKernelArg(ker1, 1, sizeof(double) * local_size, nullptr));
  V_RETURN(clSetKernelArg(ker1, 2, sizeof(cl_mem), &temp_buffer));

  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker1, 1, nullptr, &global_work_size, &local_size, 0, nullptr, nullptr));

  clEnqueueBarrier(cmd_queue);

  uint32_t local_size32 = local_size;
  clSetKernelArg(ker2, 0, sizeof(uint32_t), &local_size32);
  clSetKernelArg(ker2, 1, sizeof(double) * local_size, nullptr);
  clSetKernelArg(ker2, 2, sizeof(cl_mem), &temp_buffer);

  V_RETURN(clEnqueueNDRangeKernel(cmd_queue, ker2, 1, nullptr, &local_size, &local_size, 0, nullptr, &done_ev));
  V_RETURN(clEnqueueReadBuffer(cmd_queue, temp_buffer, false, 0, sizeof(pi_result), &pi_result, 0, nullptr, &done_ev));
  V_RETURN(clFlush(cmd_queue));
  V_RETURN(clWaitForEvents(1, &done_ev));

  printf("Compute PI result: %.18f\n", pi_result);

  return hr;
}

int main() {

  CLHRESULT hr;
  cl_device_type dev_type = CL_DEVICE_TYPE_CPU;
  ycl_platform_id platform_id;
  ycl_context context;
  ycl_device_id device;
  ycl_command_queue cmd_queue;
  ycl_program program;

  V_RETURN(FindOpenCLPlatform(nullptr, dev_type, &platform_id));

  V_RETURN(CreateDeviceContext(platform_id, dev_type, &device, &context));

  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));

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

  V_RETURN(CreateProgramFromSource(context, device, src_buffer.data(), src_buffer_len, &program));
  src_buffer.clear();

  TestAddSample(context, cmd_queue, program);

  TestComputePi(context, cmd_queue, program);

  V_RETURN(clFinish(cmd_queue));
}