#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>

#include "Camera.hpp"
#include "CommonUtils.hpp"
#include "GeometryGenerator.hpp"
#include "ShaderProgram.hpp"
#include <algorithm>
#include <random>

int LoadTexture(const char *fname, GLuint *texture);

static glm::mat4 ProjectionMat;
static ArcBallCamera Camera;
static glm::ivec2 LastMousePos;
static bool Captured;

static ShaderProgram NormalMapProgram;
static GLuint PerframeUBO, PerObjectUBO;

struct BlinnUbPerFrame {
  glm::mat4 ViewProj;
  glm::vec3 EyePosW;
  float padding0;

  glm::vec2 MinTessDistAndFactor;
  glm::vec2 MaxTessDistAndFactor;
  float HeightMapScaleFactor;
  float padding1[3];

  struct SpotLight {
    glm::vec3 LightPosW;
    float padding1;
    glm::vec3 LightDirW;
    float padding2;
    glm::vec4 LightAttenuation;
  } Light0;
};

struct BlinnUbPerObject {
  glm::mat4 World;
  glm::mat4 WorldInvTranspose;
  glm::mat4 TexTransform;

  struct Material {
    glm::vec3 MatAmbient;
    float padding4;
    glm::vec3 MatDiffuse;
    float padding5;
    glm::vec4 MatSpecular;
  } Mat0;
};

static GLuint GridVAO, GridVBO, GridEBO;
static GLsizei GridIndexCount;
static GLuint AlbedoTexture;
static GLuint NormalTexture;
static GLuint HeightMapTexture;


BlinnUbPerFrame BlinnFrameBuffer;
BlinnUbPerObject BlinnObjectBuffer;

// UI
static bool EnableWireframe;

static int create_shader_programs();
static int create_mesh_buffers();
static int create_textures();

static void render_view(GLFWwindow *window);
static void update(GLFWwindow *window);
static void destroy_resources();

int main() {
  
  GLFWwindow *window;
  int windowWidth = 800, windowHeight = 600;
  auto resize_framebuffer_callback = [](GLFWwindow *, int width, int height) {
    glViewport(0, 0, width, height);
    ProjectionMat = glm::perspective(glm::radians(45.0f), (float)width / height, .1f, 1000.0f);
  };
  auto process_keystrokes_input = [](GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
      glfwSetWindowShouldClose(window, 1);
    if(glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
      EnableWireframe ^= true;
  };

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // create main window
  window = glfwCreateWindow(windowWidth, windowHeight, "parallax-map", nullptr, nullptr);
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
  glfwSetScrollCallback(
      window, [](GLFWwindow *window, double xoffset, double yoffset) { Camera.Zoom(-(float)yoffset, 1.0f, 100.0f); });

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)&glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    return -1;
  }

  // resize the window for the first time
  resize_framebuffer_callback(window, windowWidth, windowHeight);

  Camera.SetPositions(glm::vec3{0.0f}, 5.0f, -30.0f, -90.0f);

  if (create_shader_programs() || create_textures() || create_mesh_buffers())
    return -1;

  // MSAA
  glEnable(GL_MULTISAMPLE);

  // Cull back
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  while (!glfwWindowShouldClose(window)) {
    process_keystrokes_input(window);

    update(window);
    render_view(window);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  destroy_resources();

  glfwTerminate();
  return 0;
}

int create_shader_programs() {

  if(NormalMapProgram.Create(
    "shaders/parallax-map.vs",
    "shaders/parallax-map.tesc",
    "shaders/parallax-map.tese",
    nullptr,
    "shaders/parallax-map.fs"))
    return -1;

  glGenBuffers(1, &PerframeUBO);
  glGenBuffers(1, &PerObjectUBO);

  glBindBuffer(GL_UNIFORM_BUFFER, PerframeUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(BlinnUbPerFrame), NULL, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, PerObjectUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(BlinnUbPerObject), NULL, GL_STATIC_DRAW);

  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  glBindBufferBase(GL_UNIFORM_BUFFER, 0, PerframeUBO);
  glBindBufferBase(GL_UNIFORM_BUFFER, 1, PerObjectUBO);

  return 0;
}

int LoadTexture(const char *fname, GLuint *texture) {
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
}

int create_textures() {

  if(LoadTexture("../../../asset/textures/bricks2.jpg", &AlbedoTexture))
    return -1;

  glBindTexture(GL_TEXTURE_2D, AlbedoTexture);
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glBindTexture(GL_TEXTURE_2D, 0);

  if(LoadTexture("../../../asset/textures/bricks2_normal.jpg", &NormalTexture))
    return -1;

  if(LoadTexture("../../../asset/textures/bricks2_disp.jpg", &HeightMapTexture))
    return -1;

  return 0;
}

int create_mesh_buffers() {

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

  auto meshData = GeometryGenerator::CreateGrid(20.0f, 20.0f, 21, 21);

  buffer_data_and_set_vertex_layouts(GridVAO, GridVBO, GridEBO, GridIndexCount, meshData);

  return 0;
}

void update(GLFWwindow *window) {

  std::random_device rand_dev;
  static std::default_random_engine dre(rand_dev());
  static std::uniform_real_distribution<float> rd(-1.0f, 1.0f);

  float tm = (float)glfwGetTime();
  float theta = glm::radians(11.0f*tm);
  float theta_r = glm::radians(7.0f*tm);

  glm::vec3 spotLightPos = { 5.0*std::cos(theta), 1.0f, 4.0*std::sin(theta) };
  glm::vec3 spotLightDir = {  0.8 * std::cos(theta_r), -0.6, 0.8*std::cos(theta_r)  };

  BlinnFrameBuffer.ViewProj = ProjectionMat * Camera.GetView();
  BlinnFrameBuffer.EyePosW = Camera.GetEyePosW();

  BlinnFrameBuffer.MinTessDistAndFactor = glm::vec2(0.0f, 8.0f);
  BlinnFrameBuffer.MaxTessDistAndFactor = glm::vec2(100.0f, 2.0f);
  BlinnFrameBuffer.HeightMapScaleFactor = 0.1f;
  BlinnFrameBuffer.Light0.LightPosW = spotLightPos;
  BlinnFrameBuffer.Light0.LightDirW = spotLightDir;
  BlinnFrameBuffer.Light0.LightAttenuation = glm::vec4(1.0f, 0.01, 0.004, 12.0f * rd(dre) + 20.0f);

  BlinnObjectBuffer.World = glm::mat4(1.0f);
  BlinnObjectBuffer.WorldInvTranspose = glm::mat4(1.0f);
  BlinnObjectBuffer.TexTransform = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f, 5.0f, 0.0f));
  BlinnObjectBuffer.Mat0.MatAmbient = glm::vec3(0.05f);
  BlinnObjectBuffer.Mat0.MatDiffuse = glm::vec3(1.0f);
  BlinnObjectBuffer.Mat0.MatSpecular = glm::vec4(0.3f, 0.3f, 0.3f, 16.0f);

}

void render_view(GLFWwindow *window) {

  glClearColor(.0f, .0f, .0f, 1.0f);
  glClearDepth(1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBindVertexArray(GridVAO);

  NormalMapProgram.Use();
  glPatchParameteri(GL_PATCH_VERTICES, 3);

  if(EnableWireframe) {
    glPolygonMode(GL_FRONT_AND_BACK , GL_LINE);
  } else {
    glPolygonMode(GL_FRONT_AND_BACK , GL_FILL);
  }

  glBindBuffer(GL_UNIFORM_BUFFER, PerframeUBO);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(BlinnFrameBuffer), &BlinnFrameBuffer);

  glBindBuffer(GL_UNIFORM_BUFFER, PerObjectUBO);
  glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(BlinnObjectBuffer), &BlinnObjectBuffer);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, AlbedoTexture);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, NormalTexture);
  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, HeightMapTexture);

  glDrawElements(GL_PATCHES, GridIndexCount, GL_UNSIGNED_SHORT, nullptr);

  return;
}

void destroy_resources() {

  glDeleteBuffers(1, &PerframeUBO);
  glDeleteBuffers(1, &PerObjectUBO);

  glDeleteVertexArrays(1, &GridVAO);
  glDeleteBuffers(1, &GridVBO);
  glDeleteBuffers(1, &GridEBO);

  glDeleteTextures(1, &AlbedoTexture);
  glDeleteTextures(1, &NormalTexture);
  glDeleteTextures(1, &HeightMapTexture);

  return;
}