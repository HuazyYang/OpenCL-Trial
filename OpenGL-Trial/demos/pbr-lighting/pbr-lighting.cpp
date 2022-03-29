#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <stb_image.h>
#include <stb_image_write.h>

#include <Camera.hpp>
#include <CommonUtils.hpp>
#include <GameTimer.hpp>
#include <GeometryGenerator.hpp>
#include <ShaderProgram.hpp>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>

#include "pbr/pbr_pre_compute.hpp"

int LoadTexture(const char *fname, GLx::GLTexture &texture);
int LoadHdrCubeTexture(const char *const fname[6], GLx::GLTexture &texture);
int LoadSpecularPrefilterredTexture(const char *directory, GLx::GLTexture &texture);
int LoadSpecularBRDFLUTTexture(const char *filePath, GLx::GLTexture &texture);

extern int ConvertEquivrectangleHdrImageToCubeImages(const char *hdrImagePath,
                                                     const char *cubeImagesPathPrefix);

struct LaunchConfigs {
  int ConvertEquivToCubeImage;
  int ComputePBRIBLDiffuseIrrImage;
  int ComputePBRIBLSpecularIrrImage;
  int ComputePBRIBLSpecularLUTImage;
} TheLaunchConfigs;

static char DemoTitle[] = "pbr-lighting";
static glm::mat4 ProjectionMat;
static ArcBallCamera Camera;
static glm::ivec2 LastMousePos;
static bool Captured;
static GameTimer Timer;

static ShaderProgram PBRSimgleProgram;
static GLx::GLBuffer PerFrameUBO, PerObjectUBO;

struct MaterialTextures {
  GLx::GLTexture AlbedoTexture;
  GLx::GLTexture NormalTexture;
  GLx::GLTexture MetallicTexture;
  GLx::GLTexture RoughnessTexture;
  GLx::GLTexture AoTexture;
};

static std::vector<MaterialTextures> MatTextures;
static std::vector<int> MatTexesIndices;

struct PointLight {
  glm::vec3 Pos;
  float padding0;
  glm::vec3 Radiance;
  float padding1;
};

struct PerFrameUBuffer {
  glm::mat4 ViewProj;
  glm::vec3 EyePosW;
  float padding0;
  PointLight PointLights[4];
};

struct PerObjectUBuffer {
  glm::mat4 World;
  glm::mat4 WorldInvTranspose;
  glm::mat4 TexTransform;
};

static GLx::GLVAO SphereVAO;
static GLx::GLBuffer SphereVBO, SphereEBO;
static GLsizei SphereIndexCount;

static PerFrameUBuffer PBRPerFrameUBuffer;
static PerObjectUBuffer PBRPerObjectUBuffer;

static GLx::GLTexture EnvCubeMapTexture;

static ShaderProgram SkyBoxShaderProgram;
GLx::GLVAO SkyBoxDummyVAO;
GLx::GLBuffer SkyBoxUBO;

struct SkyBoxUBuffer {
  glm::vec2 ProjectInvXY;
  float padding0[2];
  glm::mat4 ViewInverse;
};
static SkyBoxUBuffer EnvCubeBoxUBuffer;
static GLx::GLTexture EnvIrrCubeMapTexture;
static GLx::GLTexture EnvSpIrrCubeMapTexture;
static GLx::GLTexture SpecLUTTexture;

static void usage(const char *progname, const char *errs);
static int validate_args(int argc, char *argv[]);
static int create_shader_programs();
static int create_mesh_buffers();
static int create_textures();

static void render_view(GLFWwindow *window);
static void update(GLFWwindow *window);
static void destroy_resources();

void usage(const char *progname, const char *errs) {

  std::fprintf(stderr,
               "uage: %s  --[equiv-to-cube][compute-pbr-ibl-diffuse]\n"
               "       [compute-pbr-ibl-specular-irr][compute-pbr-ibl-specular-lut]",
               progname);
  if (errs) {
    std::fprintf(stderr, "Error: %s", errs);
  }
  abort();
}

int validate_args(int argc, char *argv[]) {
  int i;
  char errBuff[512];

  TheLaunchConfigs.ConvertEquivToCubeImage = 0;
  TheLaunchConfigs.ComputePBRIBLDiffuseIrrImage = 0;
  TheLaunchConfigs.ComputePBRIBLSpecularIrrImage = 0;
  TheLaunchConfigs.ComputePBRIBLSpecularLUTImage = 0;

  for (i = 1; i < argc; ++i) {
    if (strnicmp(argv[i], "--", 2) != 0) {
      snprintf(errBuff, _countof(errBuff), "Unkown command: %s", argv[i]);
      usage(argv[0], errBuff);
    } else if (stricmp(argv[i] + 2, "help") == 0) {
      usage(argv[0], nullptr);
    } else if (stricmp(argv[i] + 2, "equiv-to-cube") == 0) {
      TheLaunchConfigs.ConvertEquivToCubeImage = 1;
    } else if (stricmp(argv[i] + 2, "compute-pbr-ibl-diffuse") == 0) {
      TheLaunchConfigs.ComputePBRIBLDiffuseIrrImage = 1;
    } else if (stricmp(argv[i] + 2, "compute-pbr-ibl-specular-irr") == 0) {
      TheLaunchConfigs.ComputePBRIBLSpecularIrrImage = 1;
    } else if (stricmp(argv[i] + 2, "compute-pbr-ibl-specular-lut") == 0) {
      TheLaunchConfigs.ComputePBRIBLSpecularLUTImage = 1;
    } else {
      snprintf(errBuff, _countof(errBuff), "Unkown command: %s", argv[i]);
      usage(argv[0], errBuff);
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {

  validate_args(argc, argv);

  GLFWwindow *window;
  int windowWidth = 800, windowHeight = 600;
  auto resize_framebuffer_callback = [](GLFWwindow *, int width, int height) {
    glViewport(0, 0, width, height);
    ProjectionMat = glm::perspective(glm::radians(45.f), (float)width / (float)height, .1f, 1000.f);
  };
  auto process_keystrokes_input = [](GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, 1);
  };
  auto report_statistics_fps = [](GLFWwindow *window) {
    static double elapsed = 0.0;
    static long frame_conut = 0;

    elapsed += Timer.Delta();
    frame_conut += 1;

    if (elapsed >= 1.0) {
      char buff[128];
      std::sprintf(buff, "%s, FPS:%3.1f", DemoTitle, frame_conut / elapsed);
      elapsed = .0f;
      frame_conut = 0;
      glfwSetWindowTitle(window, buff);
    }
  };

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_VISIBLE, TheLaunchConfigs.ConvertEquivToCubeImage ||
                                       TheLaunchConfigs.ComputePBRIBLDiffuseIrrImage ||
                                       TheLaunchConfigs.ComputePBRIBLSpecularIrrImage ||
                                       TheLaunchConfigs.ComputePBRIBLSpecularLUTImage
                                   ? GLFW_FALSE
                                   : GLFW_TRUE);
  glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_FALSE);

  // create main window
  window = glfwCreateWindow(windowWidth, windowHeight, DemoTitle, nullptr, nullptr);
  if (!window) {
    std::cout << "Failed to create the main window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, resize_framebuffer_callback);
  glfwSetCursorPosCallback(window, [](GLFWwindow *window, double xpos, double ypos) {
    if (Captured) {
      float dx = (float)((int)xpos - LastMousePos.x);
      float dy = (float)((int)LastMousePos.y - ypos);
      LastMousePos = glm::ivec2(xpos, ypos);

      Camera.Rotate(dx, dy);
    }
  });
  glfwSetMouseButtonCallback(window, [](GLFWwindow *window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
      Captured = action == GLFW_PRESS;
      if (Captured) {
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        LastMousePos = glm::ivec2((int)xpos, (int)ypos);
      }
    }
  });
  glfwSetScrollCallback(window, [](GLFWwindow *window, double xoffset, double yoffset) {
    Camera.Zoom(-(float)yoffset, 1.0f, 100.0f);
  });

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)&glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    return -1;
  }

  if (TheLaunchConfigs.ConvertEquivToCubeImage) {
    std::filesystem::create_directory("runtime_asset");
    std::filesystem::create_directory("runtime_asset/textures");
    std::filesystem::create_directory("runtime_asset/textures/newport_loft");
    if (ConvertEquivrectangleHdrImageToCubeImages("../../../asset/textures/hdr/newport_loft.hdr",
                                                  "runtime_asset/textures/newport_loft/"))
      return -1;
  }

  if (LoadHdrCubeTexture(
          std::initializer_list<const char *>{"runtime_asset/textures/newport_loft/right.hdr",
                                              "runtime_asset/textures/newport_loft/left.hdr",
                                              "runtime_asset/textures/newport_loft/top.hdr",
                                              "runtime_asset/textures/newport_loft/bottom.hdr",
                                              "runtime_asset/textures/newport_loft/front.hdr",
                                              "runtime_asset/textures/newport_loft/back.hdr"}
              .begin(),
          EnvCubeMapTexture))
    return -1;

  GLint width, height;
  glBindTexture(GL_TEXTURE_CUBE_MAP, EnvCubeMapTexture);

  // Generate environment cube map mipmap levels
  // Used for generate specular irradiance cube map
  glBindTexture(GL_TEXTURE_CUBE_MAP, EnvCubeMapTexture);
  glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

  glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_TEXTURE_WIDTH, &width);
  glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, 0, GL_TEXTURE_HEIGHT, &height);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  if (TheLaunchConfigs.ComputePBRIBLDiffuseIrrImage) {
    std::filesystem::create_directory("runtime_asset/textures/newport_loft_irrmap");
    if (PBRUtils::GenerateDiffuseIrradianceCubeTextureFiles(
            GLx::GLTextureRef(EnvCubeMapTexture), width, height,
            "runtime_asset/textures/newport_loft_irrmap/")) {
      return -1;
    }
  }

  if (TheLaunchConfigs.ComputePBRIBLSpecularIrrImage) {
    std::filesystem::create_directory("runtime_asset/textures/newport_loft_spmap");
    if (PBRUtils::GenerateSpecularIrradianceMipTextureFiles(
            GLx::GLTextureRef(EnvCubeMapTexture), width, height, 8,
            "runtime_asset/textures/newport_loft_spmap/")) {
      return -1;
    }
  }

  if (TheLaunchConfigs.ComputePBRIBLSpecularLUTImage) {
    std::filesystem::create_directories("runtime_asset/textures/newport_loft_brdf");
    if (PBRUtils::GenerateSpecularBRDFTextureFile(
            GLx::GLTextureRef(EnvCubeMapTexture), width, height,
            "runtime_asset/textures/newport_loft_brdf/lut.hdr"))
      return -1;
  }

  if (TheLaunchConfigs.ConvertEquivToCubeImage || TheLaunchConfigs.ComputePBRIBLDiffuseIrrImage ||
      TheLaunchConfigs.ComputePBRIBLSpecularIrrImage ||
      TheLaunchConfigs.ComputePBRIBLSpecularLUTImage)
    return 0;

  if (LoadHdrCubeTexture(
          std::initializer_list<const char *>{
              "runtime_asset/textures/newport_loft_irrmap/right.hdr",
              "runtime_asset/textures/newport_loft_irrmap/left.hdr",
              "runtime_asset/textures/newport_loft_irrmap/top.hdr",
              "runtime_asset/textures/newport_loft_irrmap/bottom.hdr",
              "runtime_asset/textures/newport_loft_irrmap/front.hdr",
              "runtime_asset/textures/newport_loft_irrmap/back.hdr"}
              .begin(),
          EnvIrrCubeMapTexture))
    return -1;

  if(LoadSpecularPrefilterredTexture("runtime_asset/textures/newport_loft_spmap/", EnvSpIrrCubeMapTexture))
    return -1;
  if(LoadSpecularBRDFLUTTexture("runtime_asset/textures/newport_loft_brdf/lut.hdr", SpecLUTTexture))
    return -1;

  // resize the window for the first time
  resize_framebuffer_callback(window, windowWidth, windowHeight);

  Camera.SetPositions(glm::vec3{0.0f}, 10.0f, 0.0f, -90.f);

  if (create_shader_programs() || create_textures() || create_mesh_buffers())
    return -1;

  Timer.Reset();

  // MSAA
  glEnable(GL_MULTISAMPLE);
  // Depth test
  glEnable(GL_DEPTH_TEST);

  // Cull back face
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  // Enable interpolation between cube map faces
  // Used for sampler specular irradiance texture
  glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);

  while (!glfwWindowShouldClose(window)) {
    process_keystrokes_input(window);
    report_statistics_fps(window);

    Timer.Tick();

    update(window);
    render_view(window);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  destroy_resources();

  glfwTerminate();
  return 0;
}

void update(GLFWwindow *window) {

  glm::mat4 view = Camera.GetView();
  EnvCubeBoxUBuffer.ProjectInvXY =
      glm::vec2(1.0f / ProjectionMat[0][0], 1.0f / ProjectionMat[1][1]);
  EnvCubeBoxUBuffer.ViewInverse = glm::inverse(view);

  PBRPerFrameUBuffer.ViewProj = ProjectionMat * view;
  PBRPerFrameUBuffer.EyePosW = Camera.GetEyePosW();
  PBRPerFrameUBuffer.PointLights[0].Pos = {10.0f, 10.0f, 10.0f};
  PBRPerFrameUBuffer.PointLights[0].Radiance = {100.f, 100.f, 100.f};
  PBRPerFrameUBuffer.PointLights[1].Pos = {-10.0f, 10.0f, 10.0f};
  PBRPerFrameUBuffer.PointLights[1].Radiance = {100.f, 100.f, 100.f};
  PBRPerFrameUBuffer.PointLights[2].Pos = {-10.0f, -10.0f, 10.0f};
  PBRPerFrameUBuffer.PointLights[2].Radiance = {100.f, 100.f, 100.f};
  PBRPerFrameUBuffer.PointLights[3].Pos = {10.0f, -10.0f, 10.0f};
  PBRPerFrameUBuffer.PointLights[3].Radiance = {100.f, 100.f, 100.f};

  PBRPerObjectUBuffer.TexTransform = glm::mat4(1.0f);

  if (MatTexesIndices.empty()) {
    const int items = 7 * 7;
    std::random_device rdev;
    std::default_random_engine dre(rdev());
    std::uniform_int_distribution<int> id(0, (int)MatTextures.size() - 1);
    MatTexesIndices.resize(items);
    for (int i = 0; i < items; ++i) {
      MatTexesIndices[i] = id(dre);
    }
  }

  glNamedBufferSubData(SkyBoxUBO, 0, sizeof(EnvCubeBoxUBuffer), &EnvCubeBoxUBuffer);
  glNamedBufferSubData(PerFrameUBO, 0, sizeof(PerFrameUBuffer), &PBRPerFrameUBuffer);
}

void render_view(GLFWwindow *window) {

  glDepthFunc(GL_ALWAYS);

  // Render sky box first
  // We will write to defualt color buffer and depth buffer right now.
  SkyBoxShaderProgram.Use();
  glBindVertexArray(SkyBoxDummyVAO);
  glBindBufferBase(GL_UNIFORM_BUFFER, 0, SkyBoxUBO);
  glBindTextureUnit(0, EnvCubeMapTexture);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // Render scene.
  PBRSimgleProgram.Use();

  glDepthFunc(GL_LESS);

  glBindVertexArray(SphereVAO);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, PerFrameUBO);
  glBindBufferBase(GL_UNIFORM_BUFFER, 1, PerObjectUBO);
  glBindBuffer(GL_UNIFORM_BUFFER, PerObjectUBO);

  const int nrRows = 7, nrColumns = 7;
  const float spacing = 2.5f;
  glm::mat4 model;

  PBRPerObjectUBuffer.WorldInvTranspose = glm::mat4(1.0f);

  glBindTextureUnit(5, EnvIrrCubeMapTexture);
  glBindTextureUnit(6, EnvSpIrrCubeMapTexture);
  glBindTextureUnit(7, SpecLUTTexture);

  for (int i = 0; i < nrRows; ++i) {
    for (int j = 0; j < nrColumns; ++j) {
      model = glm::translate(glm::mat4(1.0f), glm::vec3((i - (nrRows >> 1)) * spacing,
                                                        (j - (nrColumns >> 1)) * spacing, 0.0));
      PBRPerObjectUBuffer.World = model;

      glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(PerObjectUBuffer), &PBRPerObjectUBuffer);

      auto &matTexes = MatTextures[MatTexesIndices[i * 7 + j]];

      // Bind textures and samplers
      glBindTextureUnit(0, matTexes.AlbedoTexture);
      glBindTextureUnit(1, matTexes.NormalTexture);
      glBindTextureUnit(2, matTexes.MetallicTexture);
      glBindTextureUnit(3, matTexes.RoughnessTexture);
      glBindTextureUnit(4, matTexes.AoTexture);

      glDrawElements(GL_TRIANGLES, SphereIndexCount, GL_UNSIGNED_SHORT, nullptr);
    }
  }

  return;
}

int create_shader_programs() {

  if (PBRSimgleProgram.Create("shaders/pbr-lighting.vs", "shaders/pbr-lighting.fs"))
    return -1;

  PerFrameUBO = GLx::GLBuffer::New();
  PerObjectUBO = GLx::GLBuffer::New();

  glBindBuffer(GL_UNIFORM_BUFFER, PerFrameUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(PerFrameUBuffer), NULL, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, PerObjectUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(PerObjectUBuffer), NULL, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, PerFrameUBO);
  glBindBufferBase(GL_UNIFORM_BUFFER, 1, PerObjectUBO);

  if (SkyBoxShaderProgram.Create("shaders/skybox.vs", "shaders/skybox.fs"))
    return -1;

  SkyBoxUBO = GLx::GLBuffer::New();
  glBindBuffer(GL_UNIFORM_BUFFER, SkyBoxUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(SkyBoxUBuffer), NULL, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  return 0;
}

int LoadTexture(const char *fname, GLx::GLTexture &texture) {
  texture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  int width, height, channels;
  GLuint format;
  unsigned char *data = stbi_load(fname, &width, &height, &channels, 0);
  if (!data) {
    std::cout << "ERROR:TEXTURE FILE:can not load file" << std::endl;
    return -1;
  }
  switch (channels) {
  default:
  case 1:
    format = GL_RED;
    break;
  case 2:
    format = GL_RG;
    break;
  case 3:
    format = GL_RGB;
    break;
  case 4:
    format = GL_RGBA;
    break;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
  stbi_image_free(data);

  glBindTexture(GL_TEXTURE_2D, 0);

  return 0;
}

int LoadHdrCubeTexture(const char *const fname[6], GLx::GLTexture &texture) {
  int i;
  int face;
  const float *data;
  int width, height;
  int channels;
  GLenum internalFormat;
  GLenum format;

  texture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

  for (i = 0; i < 6; ++i) {
    face = i + GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    data = stbi_loadf(fname[i], &width, &height, &channels, 0);
    if (!data) {
      std::cout << "ERROR: failed to load cube map face \"" << fname[i] << "\"" << std::endl;
      return -1;
    }
    switch (channels) {
    case 1:
    default:
      internalFormat = GL_R16F, format = GL_RED;
      break;
    case 2:
      internalFormat = GL_RG16F, format = GL_RG;
      break;
    case 3:
      internalFormat = GL_RGB16F, format = GL_RGB;
      break;
    case 4:
      internalFormat = GL_RGBA16F, format = GL_RGBA;
      break;
    }

    glTexImage2D(face, 0, internalFormat, width, height, 0, format, GL_FLOAT, data);
    stbi_image_free((void *)data);
  }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  return 0;
}

int LoadSpecularPrefilterredTexture(const char *directory, GLx::GLTexture &texture) {

  if (!std::filesystem::is_directory(directory))
    return -1;

  int levelIndex;
  wchar_t *end;
  int i;
  const wchar_t *facesFnames[6] = {L"right.hdr",  L"left.hdr",  L"top.hdr",
                                   L"bottom.hdr", L"front.hdr", L"back.hdr"};
  int width, height;
  int channels;
  GLenum face;
  float *data;
  GLenum internalFormat, format;

  texture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_CUBE_MAP, texture);

  for (auto &levelPath : std::filesystem::directory_iterator(directory)) {
    auto &path = levelPath.path();
    auto dirname = path.filename();
    if (std::wcsncmp(dirname.c_str(), L"level_", 6) != 0 ||
        (levelIndex = (int)std::wcstol(dirname.c_str() + 6, &end, 10),
         end == NULL || *end != 0 || levelIndex < 0)) {
      return -1;
    }

    std::filesystem::path facePath;
    for (i = 0; i < 6; ++i) {
      facePath = path;
      facePath.append(facesFnames[i]);

      face = i + GL_TEXTURE_CUBE_MAP_POSITIVE_X;
      data = stbi_loadf(facePath.string().c_str(), &width, &height, &channels, 0);
      if (!data) {
        std::cout << "ERROR: failed to load cube map face \"" << facePath << "\"" << std::endl;
        return -1;
      }
      switch (channels) {
      case 1:
      default:
        internalFormat = GL_R16F, format = GL_RED;
        break;
      case 2:
        internalFormat = GL_RG16F, format = GL_RG;
        break;
      case 3:
        internalFormat = GL_RGB16F, format = GL_RGB;
        break;
      case 4:
        internalFormat = GL_RGBA16F, format = GL_RGBA;
        break;
      }

      glTexImage2D(face, levelIndex, internalFormat, width, height, 0, format, GL_FLOAT, data);
      stbi_image_free((void *)data);
    }
  }

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
  return 0;
}

int LoadSpecularBRDFLUTTexture(const char *filePath, GLx::GLTexture &texture) {

  texture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_2D, texture);

  int width, height;
  int channels;
  float *data;
  GLenum internalFormat, format;

  data = stbi_loadf(filePath, &width, &height, &channels, 0);
  if (!data) {
    std::cout << "ERROR: failed to load specular brdf lut texture \"" << filePath << "\""
              << std::endl;
    return -1;
  }
  switch (channels) {
  case 1:
  default:
    internalFormat = GL_R16F, format = GL_RED;
    break;
  case 2:
    internalFormat = GL_RG16F, format = GL_RG;
    break;
  case 3:
    internalFormat = GL_RGB16F, format = GL_RGB;
    break;
  case 4:
    internalFormat = GL_RGBA16F, format = GL_RGBA;
    break;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, GL_FLOAT, data);
  stbi_image_free((void *)data);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  return 0;
}

int create_textures() {

  const char *dirs[] = {
      "gold",
      "grass",
      "plastic",
      "rusted_iron",
      "wall",
  };
  char buff[256];
  MaterialTextures matTextures;

  for (auto dir : dirs) {

    std::sprintf(buff, "../../../asset/textures/pbr/%s/albedo.png", dir);
    if (LoadTexture(buff, matTextures.AlbedoTexture))
      return -1;

    std::sprintf(buff, "../../../asset/textures/pbr/%s/normal.png", dir);
    if (LoadTexture(buff, matTextures.NormalTexture))
      return -1;

    std::sprintf(buff, "../../../asset/textures/pbr/%s/roughness.png", dir);
    if (LoadTexture(buff, matTextures.RoughnessTexture))
      return -1;

    std::sprintf(buff, "../../../asset/textures/pbr/%s/metallic.png", dir);
    if (LoadTexture(buff, matTextures.MetallicTexture))
      return -1;

    std::sprintf(buff, "../../../asset/textures/pbr/%s/ao.png", dir);
    if (LoadTexture(buff, matTextures.AoTexture))
      return -1;

    MatTextures.push_back(std::move(matTextures));
  }

  int width, height;
  int nrComponents;
  std::strcpy(buff, "../../../asset/textures/hdr/newport_loft.hdr");
  const float *data = stbi_loadf(buff, &width, &height, &nrComponents, 0);
  if (!data) {
    std::cout << "ERROR: can not load file \"" << buff << std::endl;
    return -1;
  }

  return 0;
}

int create_mesh_buffers() {

  auto buffer_data_and_set_vertex_layouts = [](GLx::GLVAO &VAO, GLx::GLBuffer &VBO,
                                               GLx::GLBuffer &EBO, GLsizei &indexCount,
                                               GeometryGenerator::MeshData &meshData) {
    VAO = GLx::GLVAO::New();
    VBO = GLx::GLBuffer::New();
    EBO = GLx::GLBuffer::New();

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    glBufferData(GL_ARRAY_BUFFER, meshData.Vertices.size() * sizeof(GeometryGenerator::Vertex),
                 meshData.Vertices.data(), GL_STATIC_DRAW);
    indexCount = (GLsizei)meshData.GetIndices16().size();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount * sizeof(uint16_t),
                 meshData.GetIndices16().data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
                          (void *)offsetof(GeometryGenerator::Vertex, Position));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
                          (void *)offsetof(GeometryGenerator::Vertex, Normal));
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
                          (void *)offsetof(GeometryGenerator::Vertex, TangentU));
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
                          (void *)offsetof(GeometryGenerator::Vertex, TexC));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  };

  auto meshData = GeometryGenerator::CreateSphere(1.0f, 64, 64);

  buffer_data_and_set_vertex_layouts(SphereVAO, SphereVBO, SphereEBO, SphereIndexCount, meshData);

  SkyBoxDummyVAO = GLx::GLVAO::New();
  return 0;
}

void destroy_resources() { return; }