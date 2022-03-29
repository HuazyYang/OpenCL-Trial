#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstring>

#include "CommonUtils.hpp"
#include "Camera.hpp"
#include "ShaderProgram.hpp"
#include "GeometryGenerator.hpp"
#include <algorithm>

static glm::mat4 ProjectionMat;
static ArcBallCamera Camera;
static glm::ivec2 LastMousePos;
static bool Captured;

static ShaderProgram ColorProgram;
static ShaderProgram StencilProgram;
static ShaderProgram BlurXProgram;
static ShaderProgram BlurYProgram;
static ShaderProgram MergeColorProgram;

static GLuint CubeVAO, CubeVBO, CubeEBO;
static GLsizei CubeIndexCount;
static GLuint GridVAO, GridVBO, GridEBO;
static GLsizei GridIndexCount;
static GLuint CubeDiffuseTexture;
static GLuint GridDiffuseTexture;

static GLuint BorderFBO;
static GLuint BorderTextures[2];
static GLuint BorderBlurFBOs[2];

static int create_shader_programs();
static int create_model_buffers();
static int create_textures();
static int create_frame_resources(int width, int height);
static void destroy_resources();
static void render_view(GLFWwindow *window);

#ifdef _WIN32
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE,
  PWSTR cmdLine, int showCmd) {
#else
int main(int, char**) {
#endif
  GLFWwindow *window;
  int windowWidth = 800, windowHeight = 600;
  auto resize_framebuffer_callback = [](GLFWwindow *, int width, int height) {
    glViewport(0, 0, width, height);
    create_frame_resources(width, height);
    ProjectionMat = glm::perspective(glm::radians(45.0f), (float)width/height, .1f, 1000.0f);
  };
  auto process_keystrokes_input = [](GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, 1);
  };

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // create main window
  window = glfwCreateWindow(windowWidth, windowHeight, "Stencil", nullptr, nullptr);
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

  // resize the window for the first time
  resize_framebuffer_callback(window, windowWidth, windowHeight);

  Camera.SetPositions(glm::vec3{0.0f}, 3.0f, -30.0f, 0.0f);

  if(
    create_shader_programs() ||
    create_model_buffers() ||
    create_textures()
  ) {
    return -1;
  }

  // MSAA
  glEnable(GL_MULTISAMPLE);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_STENCIL_TEST);
  glEnable(GL_CULL_FACE);

  while(!glfwWindowShouldClose(window)) {
    process_keystrokes_input(window);

    render_view(window);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  destroy_resources();

  glfwTerminate();
  return 0;
}

void render_view(GLFWwindow *window) {

  glm::mat3x2 texTransform;
  glm::mat4 world, mvp;
  const glm::vec4 borderColor{1.0f, 1.0f, 0.0f, 1.0f};
  int i;
  int width, height;

  glfwGetWindowSize(window, &width, &height);

  // bind default fbo
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glViewport(0, 0, width, height);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE); // enable write
  glEnable(GL_STENCIL_TEST);
  glStencilMask(0x1); // enable write

  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  glClearStencil(0);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);

  // draw the scene without stencil test and writing
  glBindVertexArray(GridVAO);
  ColorProgram.Use();
  ColorProgram.SetInt("DiffuseTexture", 0);

  mvp = ProjectionMat * Camera.GetView();
  ColorProgram.SetMat4("WorldViewProj", glm::value_ptr(mvp));
  texTransform = {
    5.0f, 0.0f,
    0.0f, 5.0f,
    0.0f, 0.0f
  };
  ColorProgram.SetMat3x2("TexTransform", glm::value_ptr(texTransform));
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, GridDiffuseTexture);

  // RS
  glDisable(GL_STENCIL_TEST);

  glDrawElements(GL_TRIANGLES, GridIndexCount, GL_UNSIGNED_SHORT, nullptr);

  // draw the scene need to write stencil
  glBindVertexArray(CubeVAO);

  // RS
  glEnable(GL_STENCIL_TEST);
  glStencilFunc(GL_ALWAYS, 1, 0x1);
  glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
  glStencilMask(0x1);

  glm::vec3 cubePositons[] = {
    {-.3f, .5001f, -.2f},
    {.2f, .5002f, .3f}
  };

  for(auto &pos : cubePositons) {
    world = glm::translate(glm::mat4(1.0f), pos);
    mvp =  ProjectionMat * Camera.GetView() * world;
    ColorProgram.SetMat4("WorldViewProj", glm::value_ptr(mvp));
    texTransform = glm::mat3(1.0f);
    ColorProgram.SetMat3x2("TexTransform", glm::value_ptr(texTransform));
    glBindTexture(GL_TEXTURE_2D, CubeDiffuseTexture);

    glDrawElements(GL_TRIANGLES, CubeIndexCount, GL_UNSIGNED_SHORT, nullptr);
  }

  // render into border gray scale texture
  glBindFramebuffer(GL_FRAMEBUFFER, BorderFBO);

  glViewport(0, 0, width >> 1, height >> 1);
  glClearColor(0.0f, .0f, .0f, .0f);
  glClear(GL_COLOR_BUFFER_BIT);

  glBindVertexArray(CubeVAO);
  StencilProgram.Use();

  // RS
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_STENCIL_TEST);

  StencilProgram.SetFloat("GrayScale", 0.8f);

  float cameraZoomIn = -0.04f * Camera.GetRadius();
  Camera.Zoom(cameraZoomIn, 0.0f, std::numeric_limits<float>::max());

  for(auto &pos : cubePositons) {
    world = glm::translate(glm::mat4(1.0f), pos);
    mvp = ProjectionMat * Camera.GetView() * world;
    StencilProgram.SetMat4("WorldViewProj", glm::value_ptr(mvp));

    glDrawElements(GL_TRIANGLES, CubeIndexCount, GL_UNSIGNED_SHORT, nullptr);
  }

  Camera.Zoom(-cameraZoomIn, 0.0f, std::numeric_limits<float>::max());

  // perfrom blur
  for(i=0; i < 4; ++i) {
    glBindFramebuffer(GL_FRAMEBUFFER, BorderBlurFBOs[i%2]);

    if((i%2) == 0)
      BlurXProgram.Use();
    else BlurYProgram.Use();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, BorderTextures[i%2]);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  glViewport(0, 0, width, height);

  MergeColorProgram.Use();
  // enable stencil test
  glEnable(GL_STENCIL_TEST);
  glStencilFunc(GL_NOTEQUAL, 1, 0x1);
  glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
  glStencilMask(0x0);
  // enable blend
  glEnable(GL_BLEND);
  glBlendEquation(GL_FUNC_ADD);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  MergeColorProgram.SetVec3("MaskBaseColor", glm::value_ptr(borderColor));
  MergeColorProgram.SetInt("GrayScaleTexture", 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, BorderTextures[0]);

  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  glDisable(GL_BLEND);

  return;
}

int create_shader_programs() {
  if(StencilProgram.Create("./shaders/draw_stencil.vs", "./shaders/draw_stencil.fs"))
    return -1;
  if(ColorProgram.Create("shaders/color.vs", "shaders/color.fs"))
    return -1;

  if(BlurXProgram.Create("shaders/gauss_blur.vs", "shaders/gauss_blur_x.fs"))
    return -1;
  if(BlurYProgram.Create("shaders/gauss_blur.vs", "shaders/gauss_blur_y.fs"))
    return -1;
  if(MergeColorProgram.Create("shaders/gauss_blur.vs", "shaders/merge_color.fs"))
    return -1;
  return 0;
}

static int create_model_buffers() {

  GeometryGenerator::MeshData meshData;

  auto buffer_data_and_set_vertex_layouts = [](
    GLuint &VAO, GLuint &VBO, GLuint &EBO, GLsizei &indexCount, GeometryGenerator::MeshData &meshData) {

    GLuint buffers[2];
    glGenVertexArrays(1, &VAO);
    glGenBuffers(2, buffers);
    VBO = buffers[0]; EBO = buffers[1];

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

    glBufferData(GL_ARRAY_BUFFER, meshData.Vertices.size()*sizeof(GeometryGenerator::Vertex),
      meshData.Vertices.data(), GL_STATIC_DRAW);
    indexCount = (GLsizei)meshData.GetIndices16().size();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexCount*sizeof(uint16_t),
        meshData.GetIndices16().data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
      (void *)offsetof(GeometryGenerator::Vertex, Position));
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
      (void *)offsetof(GeometryGenerator::Vertex, Normal));
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
      (void *)offsetof(GeometryGenerator::Vertex, TexC));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  };

  // cube
  meshData = GeometryGenerator::CreateBox(1.0f, 1.0f, 1.0f, 0);
  buffer_data_and_set_vertex_layouts(CubeVAO, CubeVBO, CubeEBO, CubeIndexCount, meshData);

  // sphere
  meshData = GeometryGenerator::CreateGrid(20.0f, 20.0f, 5, 5);
  buffer_data_and_set_vertex_layouts(GridVAO, GridVBO, GridEBO, GridIndexCount, meshData);

  return 0;
}

static int create_textures() {
  auto load_texture = [](const char *fname, GLuint *texture) {
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    int width, height, channels;
    GLuint format;
    unsigned char *data = stbi_load(fname,
                                    &width, &height, &channels, 0);
    if (!data) {
      std::cout << "ERROR:TEXTURE FILE:can not load file" << std::endl;
      return -1;
    }
    switch (channels) {
      default:
      case 1: format = GL_RED; break;
      case 2: format = GL_RG; break;
      case 3: format = GL_RGB; break;
      case 4: format = GL_RGBA; break;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format,
                 GL_UNSIGNED_BYTE, data);
    stbi_image_free(data);

    glBindTexture(GL_TEXTURE_2D, 0);

    return 0;
  };

  if(load_texture("../../../asset/textures/container2.png", &CubeDiffuseTexture))
    return -1;
  if(load_texture("../../../asset/textures/wall.jpg", &GridDiffuseTexture))
    return -1;
  return 0;
}

void destroy_resources() {
  GLuint buffers[2];

  glDeleteVertexArrays(1, &CubeVAO);
  buffers[0] = CubeVBO; buffers[1] = CubeEBO;
  glDeleteBuffers(2, buffers);
  glDeleteVertexArrays(1, &GridVAO);
  buffers[0] = GridVBO; buffers[1] = GridEBO;
  glDeleteBuffers(2, buffers);

  glDeleteTextures(1, &CubeDiffuseTexture);
  glDeleteTextures(1, &GridDiffuseTexture);

  // fbos, rbos, textures
  glDeleteFramebuffers(1, &BorderFBO);
  glDeleteTextures(2, BorderTextures);
  glDeleteFramebuffers(2, BorderBlurFBOs);
}

int create_frame_resources(int width, int height) {
  if(!BorderFBO) {
    GLuint fbos[3];
    glGenFramebuffers(_countof(fbos), fbos);
    BorderFBO = fbos[0];

    BorderBlurFBOs[0] = fbos[1];
    BorderBlurFBOs[1] = fbos[2];
  }

  // border grayscale frame buffer
  glBindFramebuffer(GL_FRAMEBUFFER, BorderFBO);

  if(BorderTextures[0]) glDeleteTextures(_countof(BorderTextures), BorderTextures);
  glGenTextures(_countof(BorderTextures), BorderTextures);

  glBindTexture(GL_TEXTURE_2D, BorderTextures[0]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width >> 1, height >> 1, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, BorderTextures[0], 0);

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "ERROR: Border frame buffer is not complete." << std::endl;
    return -1;
  }

  // blur along x frame buffer
  glBindFramebuffer(GL_FRAMEBUFFER, BorderBlurFBOs[0]);
  glBindTexture(GL_TEXTURE_2D, BorderTextures[1]);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, width >> 1, height >> 1, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, BorderTextures[1], 0);

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "ERROR: Border blur(X) frame buffer is not complete" << std::endl;
    return -1;
  }

  // bur along y frame buffer
  glBindFramebuffer(GL_FRAMEBUFFER, BorderBlurFBOs[1]);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, BorderTextures[0], 0);

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "ERROR: Border blur(y) frame buffer is not complete" << std::endl;
    return -1;
  }

  return 0;
}
