#include <cl_utils.h>
#include <stdio.h>
#include <vector>

int main() {

  CLHRESULT hr;
  cl_platform_id platfrom;
  ycl_context context;
  ycl_device_id device;
  char nbuff[256];
  cl_ulong mem_size;
  cl_uint mem_cacheline_size;

  V_RETURN(FindOpenCLPlatform(nullptr, CL_DEVICE_TYPE_GPU, &platfrom));

  V_RETURN(CreateDeviceContext(platfrom, CL_DEVICE_TYPE_GPU, device.ReleaseAndGetAddressOf(),
                               context.ReleaseAndGetAddressOf()));

  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(nbuff), nbuff, 0));
  printf("GPU device name: %s\n", nbuff);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(nbuff), nbuff, 0));
  printf("  device vendor: %s\n", nbuff);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(nbuff), nbuff, 0));
  printf("  device version: %s\n", nbuff);

  printf("  memory info:\n");
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(mem_size), &mem_size, 0));
  printf("    global memory cache size: %lluKB\n", mem_size >> 10);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(mem_cacheline_size), &mem_cacheline_size, 0));
  printf("    global memory cache line size: %uB\n", mem_cacheline_size);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, 0));
  printf("    global memory size: %lluMB\n", mem_size >> 20);
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(mem_size), &mem_size, 0));
  printf("    local memory size: %lluKB\n", mem_size >> 10);

  cl_device_local_mem_type type;
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(type), &type, nullptr));
  printf("    local memory implementation type: %s\n", type == CL_GLOBAL ? "Global Mem" : "SRAM");

  cl_command_queue_properties cq_props;
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cq_props), &cq_props, nullptr));
  printf("  command-queue executation mode supported:\n");
  if(cq_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
    printf("        out of order executation\n");
  else
    printf("        serialized executation\n");

  size_t max_work_group_size;
  V_RETURN(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr));
  printf("  maximum number of work-items in a work-group: %llu\n", max_work_group_size);

  return 0;
}