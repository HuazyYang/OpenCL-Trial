#include <cl_utils.h>
#include <common_miscs.h>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>
#include <glx_utils.hpp>
#include <glx_shader_program.hpp>
#include <GLFW/glfw3.h>

#if defined(_WIN32)
#define GLFW_EXPOSE_NATIVE_WIN32
#define GLFW_EXPOSE_NATIVE_WGL
#elif defined(__linux__)
#define GLFW_EXPOSE_NATIVE_X11
#define GLFW_EXPOSE_NATIVE_GLX
#else
#error "Unknown OpenGL/CL Interoperation target platform!"
#endif
#include <GLFW/glfw3native.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <array>

static const glm::vec2 GridDim(1000.0, 1000.0);
static const glm::ivec2 GridDifferentialCount(2048, 2048);
static const float DumpAttenuation = 0.1f;

static glm::mat4 ProjectionMat;
static GLx::GLTexture CurrValGridTexture;
static GLx::GLTexture NextValGridTexture;
static GLx::GLBuffer PersceneUBO;
static GLx::GLVAO GridVAO;
static GLx::GLBuffer QuadVBO;

static GLx::GLTexture LutColorTexture;

static GLx::ShaderProgram ColorShaderProgram;

// OpenCL/OpenGL Interop objects
static ycl_image CurrValGridImage;
static ycl_image NextValGridImage;
static ycl_buffer DiffParamsBuffer;
static ycl_buffer InitValBuffer;

static ycl_kernel InitKernel;
static ycl_kernel StepKernel;
static ycl_kernel DisturbKernel;
static ycl_event StepDoneEvent;

struct DifferentialParams {
  std::array<uint32_t, 4> dims;
  std::array<float, 4> params;
};

struct PersceneUBuffer {
  glm::mat4 ViewProj;
  glm::vec2 LutMinMax;
  float _padding[2];
};

static void render_view(GLFWwindow *window);
static void update(GLFWwindow *window, cl_context context, cl_device_id device, cl_command_queue cmd_queue, float dt, float elpased);

static int create_resources(cl_context context, cl_device_id device);
static void init_boundary_conditions(cl_command_queue cmd_queue);
static void advance_grid(cl_context context, cl_device_id device, cl_command_queue cmd_queue, float dt, float elapsed);

int main() {

  CLHRESULT hr;
  GLFWwindow *window;
  int windowWidth = 800, windowHeight = 600;
  auto resize_framebuffer_callback = [](GLFWwindow *, int width, int height) {
    glViewport(0, 0, width, height);

    float r = 1.0f * width / height;
    glm::vec2 half_vbb = 1.5f * GridDim / 2.0f;

    if(r < 1.0f)
      ProjectionMat = glm::orthoRH(-half_vbb.x * r, half_vbb.x * r, -half_vbb.y, half_vbb.y, -1.0f, 1.0f);
    else
      ProjectionMat = glm::orthoRH(-half_vbb.x, half_vbb.x, -half_vbb.y / r, half_vbb.y / r, -1.0f, 1.0f);
  };
  auto process_keystrokes_input = [](GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, 1);
  };

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // create main window
  window = glfwCreateWindow(windowWidth, windowHeight, "heat-transfer-simulation", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create the main window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, resize_framebuffer_callback);

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)&glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    return -1;
  }

  // resize the window for the first time
  resize_framebuffer_callback(window, windowWidth, windowHeight);

  // MSAA
  // glEnable(GL_MULTISAMPLE);

  // Create OpenCL device and context
  const char *preferred_plats[] = { "NVIDIA", "AMD", nullptr };
  ycl_platform_id platform;
  ycl_device_id device;
  ycl_context context;
  ycl_command_queue cmd_queue;
  ycl_program program;

  void *glrc;
  void *glsurface;

#ifdef __linux__
  glrc = (void *)glfwGetGLXContext(window);
  glsurface = (void *)glfwGetX11Display();
#elif defined(_WIN32)
  glrc = (void *)glfwGetWGLContext(window);
  glsurface = (void *)GetDC(glfwGetWin32Window(window));
#else
#error "Unknown GL surface!"
#endif

  V_RETURN(FindOpenCLPlatform2(CL_DEVICE_TYPE_GPU, {"NVIDIA", "AMD"}, {"cl_khr_gl_sharing"}, glrc, glsurface, &platform,
                               &device));

  cl_context_properties ctx_props[] = {CL_GL_CONTEXT_KHR,
                                   (cl_context_properties)glrc,
#ifdef __linux__
                                   CL_GLX_DISPLAY_KHR,
                                   (cl_context_properties)glsurface,
#elif defined(_WIN32)
                                   CL_WGL_HDC_KHR,
                                   (cl_context_properties)glsurface,
#else
#error Unknown GL surface support!
#endif
                                   CL_CONTEXT_PLATFORM,
                                   (cl_context_properties)(cl_platform_id)platform,
                                   0};

  V_RETURN2(context <<= clCreateContext(ctx_props, 1, &device, nullptr, nullptr, &hr), hr);
  V_RETURN(CreateCommandQueue(context, device, &cmd_queue));
  V_RETURN(CreateProgramFromFile(context, device, nullptr, "heat_transfer.cl", &program));
  V_RETURN2(InitKernel <<= clCreateKernel(program, "heat_transfer_init", &hr), hr);
  V_RETURN2(StepKernel <<= clCreateKernel(program, "heat_transfer_step", &hr), hr);
  V_RETURN2(DisturbKernel <<= clCreateKernel(program, "heat_disturb", &hr), hr);

  V_RETURN(create_resources(context, device));

  init_boundary_conditions(cmd_queue);

  if(ColorShaderProgram.Create(
    "./shaders/color.vs",
    "./shaders/color.fs"
  )) {
    return -1;
  }

  hp_timer::time_point start0, start, fin;
  fmilliseconds elapsed0, elapsed;

  start = start0 = hp_timer::now();

  while (!glfwWindowShouldClose(window)) {
    process_keystrokes_input(window);
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
      init_boundary_conditions(cmd_queue);

    fin = hp_timer::now();
    elapsed = fmilliseconds_cast(fin - start);
    start = fin;
    elapsed0 = fmilliseconds(fin - start0);

    update(window, context, device, cmd_queue, elapsed.count(), elapsed0.count());
    render_view(window);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}

int create_resources(cl_context context, cl_device_id device) {

  CLHRESULT hr;

  hr = 0;

  CurrValGridTexture = GLx::GLTexture::Create(GL_TEXTURE_2D);
  NextValGridTexture = GLx::GLTexture::Create(GL_TEXTURE_2D);

  glTextureStorage2D(CurrValGridTexture, 1, GL_R32F, (GLsizei)GridDifferentialCount.x,
                     (GLsizei)GridDifferentialCount.y);
  glTextureParameteri(CurrValGridTexture, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTextureParameteri(CurrValGridTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(CurrValGridTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(CurrValGridTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(CurrValGridTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTextureStorage2D(NextValGridTexture, 1, GL_R32F, (GLsizei)GridDifferentialCount.x,
                     (GLsizei)GridDifferentialCount.y);
  glTextureParameteri(NextValGridTexture, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTextureParameteri(NextValGridTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(NextValGridTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(NextValGridTexture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTextureParameteri(NextValGridTexture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  V_RETURN2(CurrValGridImage <<=
            clCreateFromGLTexture2D(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, CurrValGridTexture, &hr),
            hr);

  V_RETURN2(NextValGridImage <<=
            clCreateFromGLTexture2D(context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, NextValGridTexture, &hr),
            hr);

  V_RETURN2(DiffParamsBuffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                                sizeof(DifferentialParams), nullptr, &hr),
            hr);

  V_RETURN2(InitValBuffer <<= clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                                 (GridDifferentialCount.x + GridDifferentialCount.y) * 2 * sizeof(float), nullptr, &hr),
            hr);

#define RGB_CONV(r, g, b)                                                                                              \
  {static_cast<uint8_t>(r * 255.0f), static_cast<uint8_t>(g * 255.0f), static_cast<uint8_t>(b * 255.0f)}
  uint8_t LutColors[][3] = {
      RGB_CONV(0.0f, 0.0f, 1.0f), RGB_CONV(0.0f, 0.5f, 1.0f), RGB_CONV(0.0f, 1.0f, 1.0f), RGB_CONV(0.0f, 1.0f, 0.5f),
      RGB_CONV(0.0f, 1.0f, 0.0f), RGB_CONV(0.5f, 1.0f, 0.0f), RGB_CONV(1.0f, 1.0f, 0.0f), RGB_CONV(1.0f, 0.5f, 0.0f),
      RGB_CONV(1.0f, 0.0f, 0.0f), RGB_CONV(1.0f, 0.5f, 0.5f), RGB_CONV(1.0f, 1.0f, 1.0f),
  };
#undef RGB_CONV
  LutColorTexture = GLx::GLTexture::Create(GL_TEXTURE_1D);
  glTextureStorage1D(LutColorTexture, 1, GL_RGB8, _countof(LutColors));
  glTextureSubImage1D(LutColorTexture, 0, 0, _countof(LutColors), GL_RGB, GL_UNSIGNED_BYTE, LutColors);
  glTextureParameteri(LutColorTexture, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTextureParameteri(LutColorTexture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTextureParameteri(LutColorTexture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTextureParameteri(LutColorTexture, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTextureParameteri(LutColorTexture, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  PersceneUBO = GLx::GLBuffer::Create();
  glBindBuffer(GL_UNIFORM_BUFFER, PersceneUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(PersceneUBuffer), nullptr, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  GridVAO = GLx::GLVAO::Create();
  QuadVBO = GLx::GLBuffer::Create();
  glBindVertexArray(GridVAO);
  glBindBuffer(GL_ARRAY_BUFFER, QuadVBO);

  glm::vec2 half_dims = {GridDim.x / 2.0f, GridDim.y / 2.0f};
  glm::vec2 vertices[4][2] = {
      {{-half_dims.x, -half_dims.y}, { 0.0f, 0.0f }},
      {{+half_dims.x, -half_dims.y}, { 1.0f, 0.0f }},
      {{-half_dims.x, +half_dims.y}, { 0.0f, 1.0f }},
      {{+half_dims.x, +half_dims.y}, { 1.0f, 1.0f }}
  };
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2)*2, (void *)0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2)*2, (void *)8);
  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return hr;
}

void init_boundary_conditions(cl_command_queue cmd_queue) {

  CLHRESULT hr;
  cl_context context;
  cl_device_id device;
  float const_vals[4] = {0.f, 0.0f, 0.f, 0.f};
  size_t buffer_offsets[4] = {0, (size_t)GridDifferentialCount.x, 2 * (size_t)GridDifferentialCount.x,
                              2 * (size_t)GridDifferentialCount.x + (size_t)GridDifferentialCount.y};
  size_t buffer_sizes[4] = {(size_t)GridDifferentialCount.x, (size_t)GridDifferentialCount.x,
                            (size_t)GridDifferentialCount.y, (size_t)GridDifferentialCount.y};
  size_t buffer_size;
  size_t work_group_size[3];
  size_t max_work_item_size[3];
  size_t work_item_size[2];

  V(clGetCommandQueueInfo(cmd_queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr));
  V(clGetCommandQueueInfo(cmd_queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr));

  V(SetKernelArguments(InitKernel, &CurrValGridImage, &InitValBuffer));

  for (int i = 0; i < 4; ++i) {
    V(clEnqueueFillBuffer(cmd_queue, InitValBuffer, &const_vals[i], sizeof(const_vals[i]), buffer_offsets[i] * sizeof(float),
                          buffer_sizes[i] * sizeof(float), 0, nullptr, nullptr));
  }

  V(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_size), max_work_item_size, nullptr));
  V(clGetKernelWorkGroupInfo(InitKernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(work_group_size),
                             work_group_size, nullptr));

  work_item_size[0] = std::min((size_t)GridDifferentialCount.x, max_work_item_size[0]);
  work_item_size[0] = RoundF(work_item_size[0], work_group_size[0]);
  work_item_size[1] = work_group_size[1];

  cl_mem sync_objs[2] = {CurrValGridImage, NextValGridImage};
  float pattern[4] = { 0.0f, 0.0f, 0.0f, 0.0f};
  size_t origin[3] = {0, 0, 0};
  size_t region[3] = {(size_t)GridDifferentialCount.x, (size_t)GridDifferentialCount.y, 1};

  V(clEnqueueAcquireGLObjects(cmd_queue, 2, sync_objs, 0, nullptr, nullptr));
  V(clEnqueueFillImage(cmd_queue, CurrValGridImage, pattern, origin, region, 0, nullptr, nullptr));
  V(clEnqueueNDRangeKernel(cmd_queue, InitKernel, 2, nullptr, work_item_size, work_group_size, 0, nullptr, nullptr));
  V(clEnqueueReleaseGLObjects(cmd_queue, 2, sync_objs, 0, nullptr, nullptr));
}

void advance_grid(cl_context context, cl_device_id device, cl_command_queue cmd_queue, float dt, float elapsed) {

  CLHRESULT hr;
  static float UpdateInterval = .0f;

  cl_mem sync_objs[] = {CurrValGridImage, NextValGridImage};
  V(clEnqueueAcquireGLObjects(cmd_queue, 2, sync_objs, 0, nullptr, 0));

  UpdateInterval += dt;
  if(UpdateInterval > 100.0f) {
    std::uniform_int_distribution<int> posx_rd(0, GridDifferentialCount.x);
    std::uniform_int_distribution<int> posy_rd(0, GridDifferentialCount.y);
    std::array<uint32_t, 2> pos;
    std::uniform_real_distribution<float> mag_rd(0.0f, 600.0f);
    float magnitude;

    for(int i = 0; i < 6; ++i) {
      pos[0] = posx_rd(g_RandomEngine);
      pos[1] = posy_rd(g_RandomEngine);

      magnitude = mag_rd(g_RandomEngine);

      V(SetKernelArguments(DisturbKernel, &CurrValGridImage, &pos, &magnitude));

      size_t dual_work_item_size[1] = {1};
      size_t dual_work_group_size[3] = {1, 1, 1};
      V(clEnqueueNDRangeKernel(cmd_queue, DisturbKernel, 1, 0, dual_work_item_size, dual_work_group_size, 0, nullptr,
                              nullptr));
    }
    UpdateInterval = 0.0f;
  }

  DifferentialParams params;
  float dh = GridDim.x / (GridDifferentialCount.x - 1);

  params.dims[0] = (uint32_t)GridDifferentialCount.x;
  params.dims[1] = (uint32_t)GridDifferentialCount.y;
  params.params[0] = DumpAttenuation * dt / (dh * dh);
  params.params[0] = std::min(params.params[0], 0.2499f);
  params.params[1] = 1.0f - 4.0 * params.params[0];

  V(clEnqueueFillBuffer(cmd_queue, DiffParamsBuffer, &params, sizeof(params), 0, sizeof(params), 0, nullptr, nullptr));

  size_t work_group_size[3];
  size_t max_work_item_size[3];
  size_t work_item_size[2];

  V(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_size), max_work_item_size, nullptr));
  V(clGetKernelWorkGroupInfo(StepKernel, device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(work_group_size),
                             work_group_size, nullptr));

  work_item_size[0] = std::min((size_t)GridDifferentialCount.x, max_work_item_size[0]);
  work_item_size[1] = std::min((size_t)GridDifferentialCount.x, max_work_item_size[1]);
  work_item_size[0] = RoundF(work_item_size[0], work_group_size[0]);
  work_item_size[1] = RoundF(work_item_size[1], work_group_size[1]);

  V(SetKernelArguments(StepKernel, &CurrValGridImage, &NextValGridImage, &DiffParamsBuffer));
  V(clEnqueueNDRangeKernel(cmd_queue, StepKernel, 2, nullptr, work_item_size, work_group_size, 0, nullptr, nullptr));

  V(clEnqueueReleaseGLObjects(cmd_queue, 2, sync_objs, 0, nullptr, StepDoneEvent.ReleaseAndGetAddressOf()));

  std::swap(NextValGridImage, CurrValGridImage);
  std::swap(NextValGridTexture, CurrValGridTexture);
}

void update(
    GLFWwindow *windows, cl_context context, cl_device_id device, cl_command_queue cmd_queue, float dt, float elapsed) {

  advance_grid(context, device, cmd_queue, dt, elapsed);

  PersceneUBuffer scene_ubo;
  scene_ubo.ViewProj = ProjectionMat;
  scene_ubo.LutMinMax = {0.0f, 2.0f};

  glBindBuffer(GL_UNIFORM_BUFFER, PersceneUBO);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(scene_ubo), &scene_ubo);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

void render_view(GLFWwindow *window) {

  CLHRESULT hr;

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  ColorShaderProgram.Use();
  glBindVertexArray(GridVAO);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, PersceneUBO);
  glBindTextureUnit(0, CurrValGridTexture);
  glBindTextureUnit(1, LutColorTexture);

  V(clWaitForEvents(1, &StepDoneEvent));

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glBindVertexArray(0);
}
