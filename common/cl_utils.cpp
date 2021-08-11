#include "cl_utils.h"
#include <CL/cl_gl_ext.h>
#include <stdio.h>
#include <stdarg.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <vector>
#include <fstream>

#pragma warning(disable: 4996)

const char* TranslateOpenCLError(cl_int errorCode)
{
    switch(errorCode)
    {
    case CL_SUCCESS:                            return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
//    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
//    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    

    default:
        return "UNKNOWN ERROR CODE";
    }
}

#ifdef _WIN32
void CLUtilsTrace(CLHRESULT hr, const char *fmt, ...) {
  char buff[1024];

  snprintf(buff, _countof(buff), "CL error: %s\n", TranslateOpenCLError(hr));
  OutputDebugStringA(buff);

  va_list vlist;
  va_start(vlist, fmt);
  vsnprintf(buff, _countof(buff), fmt, vlist);
  va_end(vlist);

  OutputDebugStringA(buff);
  OutputDebugStringA("\n");
}
#else
void CLUtilsTrace(CLHRESULT hr, const char *fmt, ...) {
  va_list vlist;
  va_start(vlist, fmt);
  vfprintf(stderr, fmt, vlist);
  fprintf(stderr, "\n");
}
#endif

CLHRESULT CheckCLExtensions(cl_device_id device, std::initializer_list<const char *> req_extensions) {
  CLHRESULT hr;
  size_t num_extensions;
  std::vector<char> extensions;
  size_t check_left = req_extensions.size();

  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, nullptr, &num_extensions));
  extensions.resize(num_extensions);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, num_extensions, (void *)extensions.data(), nullptr));

  char *token_ctx;
  const char *ext;

  ext = strtok_s((char *)extensions.data(), " ", &token_ctx);
  while(ext) {
    for(auto it_req = req_extensions.begin(); it_req != req_extensions.end(); ++it_req) {
      if(_stricmp(*it_req, ext) == 0) {
        --check_left;
        break;
      }
    }
    if(check_left == 0)
      break;
    ext = strtok_s(nullptr, " ", &token_ctx);
  }

  return check_left ? CL_INVALID_DEVICE : CL_SUCCESS;
}

static CLX_REFRET_T CL_API_CALL __clx_ref_no_op(void *) { return CL_SUCCESS; }

CLX_OBJECT_REFCOUNT_MGR_TABLE_ENTRY g_CLxObjectRefcountMgrTable[(int)CLX_OBJECT_TYPE::CLX_OBJECT_TYPE_MAX] = {
    {(CLX_OBJECT_ADDREF)__clx_ref_no_op, (CLX_OBJECT_RELEASE)__clx_ref_no_op },
    {(CLX_OBJECT_ADDREF)clRetainDevice, (CLX_OBJECT_RELEASE)clReleaseDevice},
    {(CLX_OBJECT_ADDREF)clRetainContext, (CLX_OBJECT_RELEASE)clReleaseContext},
    {(CLX_OBJECT_ADDREF)clRetainProgram, (CLX_OBJECT_RELEASE)clReleaseProgram},
    {(CLX_OBJECT_ADDREF)clRetainKernel, (CLX_OBJECT_RELEASE)clReleaseProgram},
    {(CLX_OBJECT_ADDREF)clRetainCommandQueue, (CLX_OBJECT_RELEASE)clReleaseCommandQueue},
    {(CLX_OBJECT_ADDREF)clRetainEvent, (CLX_OBJECT_RELEASE)clReleaseEvent},
    {(CLX_OBJECT_ADDREF)clRetainMemObject, (CLX_OBJECT_RELEASE)clReleaseMemObject},
    {(CLX_OBJECT_ADDREF)clRetainMemObject, (CLX_OBJECT_RELEASE)clReleaseMemObject},
    {(CLX_OBJECT_ADDREF)clRetainMemObject, (CLX_OBJECT_RELEASE)clReleaseMemObject},
    {(CLX_OBJECT_ADDREF)clRetainSampler, (CLX_OBJECT_RELEASE)clReleaseSampler},
};

CLHRESULT FindOpenCLPlatform(cl_device_type dev_type,
                             std::initializer_list<const char *> preferred_plats,
                             std::initializer_list<const char *> req_extensions,
                             cl_platform_id *plat_id,
                             cl_device_id *device) {
  return FindOpenCLPlatform2(dev_type, preferred_plats, req_extensions, nullptr, nullptr, plat_id, device);
}

extern CLHRESULT FindOpenCLPlatform2(cl_device_type dev_type,
                                     std::initializer_list<const char *> preferred_plats,
                                     std::initializer_list<const char *> req_extensions,
                                     void *glrc,      // Optional
                                     void *glsurface, // Optional
                                     cl_platform_id *plat_id,
                                     cl_device_id *device) {
  CLHRESULT hr;
  cl_uint num_plats;
  cl_platform_id sel_platform;
  cl_device_id sel_device;

  V_RETURN(clGetPlatformIDs(0, nullptr, &num_plats));
  if (num_plats == 0) {
    hr = -1;
    CL_TRACE(hr, "No Platform found!\n");
    return hr;
  }

  std::vector<cl_platform_id> platforms{num_plats};

  V_RETURN(clGetPlatformIDs(num_plats, &platforms[0], nullptr));

  size_t nlength;
  std::vector<char> name;
  size_t num_devs = 0;
  std::vector<cl_device_id> devs;
  bool preferred_name_matched;
  clGetGLContextInfoKHR_fn clGetGLContextInfoKHR;
  cl_device_type dev_type2;

  for (auto itPlat = platforms.begin(); itPlat != platforms.end(); ++itPlat) {

    if(preferred_plats.size() != 0) {

      V_RETURN(clGetPlatformInfo(*itPlat, CL_PLATFORM_NAME, 0, nullptr, &nlength));

      name.resize(nlength + 1);
      name[nlength] = 0;

      V_RETURN(clGetPlatformInfo(*itPlat, CL_PLATFORM_NAME, nlength, &name[0], nullptr));

      preferred_name_matched = false;
      for (auto it_name = preferred_plats.begin(); it_name != preferred_plats.end(); ++it_name) {
        if (_strnicmp(*it_name, name.data(), strlen(*it_name)) == 0) {
          preferred_name_matched = true;
          break;
        }
      }
      if(!preferred_name_matched)
        continue;
    }

    num_devs = 0;
    if(glrc) {
      clGetGLContextInfoKHR =
          (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddressForPlatform(*itPlat, "CLGetGLContextInfoKHR");
      if(clGetGLContextInfoKHR) {
        cl_context_properties props[] = {
          CL_GL_CONTEXT_KHR, (cl_context_properties)glrc,
          #ifdef __linux__
          CL_GLX_DISPLAY_KHR, (cl_context_properties)glsurface,
          #elif defined(_WIN32)
          CL_WGL_HDC_KHR, (cl_context_properties)glsurface,
          #else
          #error Unknown GL surface support!
          #endif
          CL_CONTEXT_PLATFORM, (cl_context_properties)*itPlat,
          0
        };

        V_RETURN(clGetGLContextInfoKHR(props, CL_DEVICES_FOR_GL_CONTEXT_KHR, 0, nullptr, &num_devs));
        if(num_devs == 0)
          continue;
        devs.resize(num_devs / sizeof(cl_device_id));
        V_RETURN(clGetGLContextInfoKHR(props, CL_DEVICES_FOR_GL_CONTEXT_KHR, num_devs, (void *)devs.data(), nullptr));
      }
    }

    if(num_devs == 0) {
      V_RETURN(clGetDeviceIDs(*itPlat, dev_type, 0, nullptr, (cl_uint *)&num_devs));
      if (num_devs == 0)
        continue;

      devs.resize(num_devs);
      V_RETURN(clGetDeviceIDs(*itPlat, dev_type, num_devs, (cl_device_id *)devs.data(), nullptr));
    }

    for(auto it_dev = devs.begin(); it_dev != devs.end(); ++it_dev) {

      V_RETURN(clGetDeviceInfo(*it_dev, CL_DEVICE_TYPE, sizeof(dev_type2), &dev_type2, nullptr));
      if(dev_type != dev_type2)
        continue;

      if(CheckCLExtensions(*it_dev, req_extensions) == CL_SUCCESS) {
        sel_platform = *itPlat;
        sel_device = *it_dev;
        break;
      }
    }
  }

  *plat_id = sel_platform;
  *device = sel_device;

  return sel_platform ? CL_SUCCESS : -1;
}

CLHRESULT CreateDeviceContext(cl_platform_id plat_id, cl_device_id dev, cl_context *dev_ctx) {
  CLHRESULT hr;

  cl_context_properties ctx_props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)plat_id, 0};
  V_RETURN2(*dev_ctx = clCreateContext(ctx_props, 1, &dev, nullptr, nullptr, &hr), hr);

  return hr;
}

CLHRESULT CreateCommandQueue(cl_context dev_ctx, cl_device_id device, cl_command_queue *cmd_queue) {

  CLHRESULT hr;
  size_t nlength;
  std::vector<char> strver;
  char *p_ver, *token_ctx;
  float ver;

  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &nlength));
  if (nlength == 0) {
    hr = -1;
    CL_TRACE(hr, "Device version is not available!\n");
    return hr;
  }

  strver.resize(nlength + 1, 0);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_VERSION, nlength, &strver[0], nullptr));

  if ((p_ver = strtok_s(&strver[0], " ", &token_ctx)) != nullptr &&
      (p_ver = strtok_s(nullptr, " ", &token_ctx)) != nullptr) {
    ver = strtof(p_ver, nullptr);
  } else {
    ver = 1.2f;
  }

  if (ver >= 2.0f) {
    const cl_command_queue_properties cmd_queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    *cmd_queue = clCreateCommandQueueWithProperties(dev_ctx, device, cmd_queue_props, &hr);
    V_RETURN(hr);
  } else if (ver >= 1.2f) {

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
    cl_context context, cl_device_id device, const char *defines, const char *source, size_t src_len, cl_program *program) {

  CLHRESULT hr;
  char internal_defs[] =
      R"(
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
)";

  if(!defines) defines = "";

  const char *sources[] = {defines, internal_defs, source};
  size_t src_lens[] = {strlen(defines), _countof(internal_defs) - 1, src_len};

  char tempbuff[256];
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(tempbuff), tempbuff, nullptr));
  char *token_ctx;
  char *pver = strtok_s(tempbuff, " ", &token_ctx);
  pver = strtok_s(nullptr, " ", &token_ctx);
  pver = strtok_s(nullptr, " ", &token_ctx);

  char cl_optbuff[128];
  sprintf_s(cl_optbuff, _countof(cl_optbuff), "-cl-std=CL%s", pver);

  V_RETURN2(*program = clCreateProgramWithSource(context, _countof(sources), sources, src_lens, &hr), hr);

  hr = clBuildProgram(*program, 1, &device, cl_optbuff, nullptr, nullptr);
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

CLHRESULT CreateProgramFromIL(cl_context context, cl_device_id device,
                              const void *il, size_t il_len,
                              cl_program *program) {

  CLHRESULT hr;
  char tempbuff[256];
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(tempbuff),
                           tempbuff, nullptr));
  char *token_ctx;
  char *pver = strtok_s(tempbuff, " ", &token_ctx);
  pver = strtok_s(nullptr, " ", &token_ctx);
  pver = strtok_s(nullptr, " ", &token_ctx);

  char cl_optbuff[128];
  sprintf_s(cl_optbuff, _countof(cl_optbuff), "-cl-std=CL%s", pver);

  V_RETURN2(*program = clCreateProgramWithIL(context, il, il_len, &hr), hr);

  hr = clBuildProgram(*program, 1, &device, cl_optbuff, nullptr, nullptr);
  if (hr == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size = 0;
    std::vector<char> build_log;

    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);

    build_log.resize(log_size);
    clGetProgramBuildInfo(*program, device, CL_PROGRAM_BUILD_LOG, log_size,
                          build_log.data(), nullptr);
    CL_TRACE(hr, "Build CL program erorr, log:\n%s\n", build_log.data());
    return hr;
  }

  if (CL_FAILED(hr)) {
    *program = nullptr;
    V_RETURN(hr);
  }

  return hr;
}

CLHRESULT CreateProgramFromFile(
    cl_context context, cl_device_id device, const char *defines, const char *fname, cl_program *program) {

  CLHRESULT hr;

  std::ifstream fin(fname, std::fstream::binary);
  if(!fin) {
    V_RETURN(("Invalid binary file", CL_INVALID_BINARY));
  }

  std::vector<char> source;
  std::streamsize srclen;
  fin.seekg(0, std::fstream::end);
  srclen = fin.tellg();
  fin.seekg(0, std::fstream::beg);
  source.resize(srclen);
  fin.read((char *)source.data(), srclen);
  fin.close();

  return CreateProgramFromSource(context, device, defines, source.data(), srclen, program);
}

CLHRESULT CreateProgramFromILFile(cl_context context, cl_device_id device,
                                  const char *fname, cl_program *program) {

  CLHRESULT hr;

  std::ifstream fin(fname, std::fstream::binary);
  if (!fin) {
    V_RETURN(("Invalid binary file", CL_INVALID_BINARY));
  }

  std::vector<char> source;
  std::streamsize srclen;
  fin.seekg(0, std::fstream::end);
  srclen = fin.tellg();
  fin.seekg(0, std::fstream::beg);
  source.resize(srclen);
  fin.read((char *)source.data(), srclen);
  fin.close();

  return CreateProgramFromIL(context, device, source.data(), srclen, program);
}