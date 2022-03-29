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

#ifdef _MSC_VER
#pragma warning(disable: 4996)
#endif

extern int LoadTexture(const char *fname, GLuint *texture);

struct DirectionalLight {
  glm::vec3 direction;
  glm::vec3 ambient;
  glm::vec3 diffuse;
  glm::vec3 specular;
};

struct PointLight {
  glm::vec3 posW;
  glm::vec3 ambient;
  glm::vec3 diffuse;
  glm::vec3 specular;
  glm::vec3 attenuation;
};

struct SpotLight {
  glm::vec3 posW;
  glm::vec4 direction_s;
  glm::vec3 ambient;
  glm::vec3 diffuse;
  glm::vec3 specular;
  glm::vec3 attenuation;
};

struct Material {
  glm::vec3 ambient;
  glm::vec3 diffuse;
  glm::vec4 specular;
};

static glm::mat4 ProjectionMat;
static bool Captured;
static ArcBallCamera Camera;
static glm::ivec2 LastMousePos;

static ShaderProgram LightingProgram;
static ShaderProgram LightSourceProgram;

static GLuint CubeVAO, CubeEBO, CubeVBO;
static GLsizei CubeIndexCount;
static GLuint SphereVAO, SphereEBO, SphereVBO;
static GLsizei SphereIndexCount;
static GLuint AlbedoTexture, SpecularTexure;

#define NUMBER_OF_POINT_LIGHTS  4
static DirectionalLight DirLight0;
static PointLight PointLights[NUMBER_OF_POINT_LIGHTS];
static SpotLight SpotLight0;
static Material Mat;

static void init_light_and_material();
static void create_buffers();
static int create_textures_and_samplers();
static void destroy_buffers();
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
    ProjectionMat = glm::perspective(glm::radians(30.0f), (float)width/height, .1f, 100.0f);
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
  window = glfwCreateWindow(windowWidth, windowHeight, "Lighting", nullptr, nullptr);
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
    Camera.Zoom(-(float)yoffset, 1.0f, 20.0f);
  });

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)&glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    return -1;
  }

  // Enable MSAA
  glEnable(GL_MULTISAMPLE);

  // resize window for the first time
  resize_framebuffer_callback(window, windowWidth, windowHeight);

  if(LightingProgram.Create("./shaders/lighting.vs", "./shaders/lighting.fs")) {
    std::cout << "GLSL:Failed to create lighting program." << std::endl;
    return -1;
  }

  if(LightSourceProgram.Create("./shaders/lighting.vs", "./shaders/light_source.fs")) {
    std::cout << "GLSL: Failed to create light source program" << std::endl;
    return -1;
  }

  // set a arc ball camera
  Camera.SetPositions(glm::vec3(0.0f, .0f, .0f), 3.0f, .0f, .0f);

  create_buffers();
  if(create_textures_and_samplers())
    return -1;
  init_light_and_material();

  LightingProgram.SetInt("DiffuseTexture", 0);
  LightingProgram.SetInt("SpecularTexture", 1);

  glEnable(GL_CULL_FACE);
  glFrontFace(GL_CCW);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);

  // render loop
  while (!glfwWindowShouldClose(window)) {

    // process input.
    process_keystrokes_input(window);
    // render viewer.
    render_view(window);

    // glfw: swap buffers and pull I/O events
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  destroy_buffers();

  glfwTerminate();
  return 0;
}

void render_view(GLFWwindow *window) {
  glClearColor(0.75f, 0.52f, 0.3f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  glm::vec3 tempLightColor{1.0f};
  glm::mat4 vp, world;
  float tm = (float)glfwGetTime();
  char nameBuff[64] = "PointLights[ ].";
  int i;

  LightingProgram.Use();

  vp = ProjectionMat * Camera.GetView();

  LightingProgram.SetMat4("ViewProj", glm::value_ptr(vp));

  LightingProgram.SetVec3("EyePosW", glm::value_ptr(Camera.GetEyePosW()));

  LightingProgram.SetVec3("DirLight0.direction", glm::value_ptr(DirLight0.direction));
  LightingProgram.SetVec3("DirLight0.ambient", glm::value_ptr(DirLight0.ambient));
  LightingProgram.SetVec3("DirLight0.diffuse", glm::value_ptr(DirLight0.diffuse));
  LightingProgram.SetVec3("DirLight0.specular", glm::value_ptr(DirLight0.specular));

  i = 0;
  for(auto &pl : PointLights) {
    nameBuff[12] = '0' + (char)i;
    std::strcpy(&nameBuff[15], "posW");
    LightingProgram.SetVec3(nameBuff, glm::value_ptr(PointLights[i].posW));
    std::strcpy(&nameBuff[15], "ambient");
    LightingProgram.SetVec3(nameBuff, glm::value_ptr(PointLights[i].ambient));
    std::strcpy(&nameBuff[15], "diffuse");
    LightingProgram.SetVec3(nameBuff, glm::value_ptr(PointLights[i].diffuse));
    std::strcpy(&nameBuff[15], "specular");
    LightingProgram.SetVec3(nameBuff, glm::value_ptr(PointLights[i].specular));
    std::strcpy(&nameBuff[15], "attenuation");
    LightingProgram.SetVec3(nameBuff, glm::value_ptr(PointLights[i].attenuation));
    ++i;
  }

  LightingProgram.SetVec3("SpotLight0.posW", glm::value_ptr(SpotLight0.posW));
  LightingProgram.SetVec4("SpotLight0.direction", glm::value_ptr(SpotLight0.direction_s));
  LightingProgram.SetVec3("SpotLight0.ambient", glm::value_ptr(SpotLight0.ambient));
  LightingProgram.SetVec3("SpotLight0.diffuse", glm::value_ptr(SpotLight0.diffuse));
  LightingProgram.SetVec3("SpotLight0.specular", glm::value_ptr(SpotLight0.specular));
  LightingProgram.SetVec3("SpotLight0.attenuation", glm::value_ptr(SpotLight0.attenuation));

  LightingProgram.SetVec3("Mat.ambient", glm::value_ptr(Mat.ambient));
  LightingProgram.SetVec3("Mat.diffuse", glm::value_ptr(Mat.diffuse));
  LightingProgram.SetVec4("Mat.specular", glm::value_ptr(Mat.specular));

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, AlbedoTexture);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, SpecularTexure);

  glBindVertexArray(CubeVAO);

  glm::vec3 cubePositions[] = {
      glm::vec3(0.0f, 0.0f, 0.0f),    glm::vec3(2.0f, 5.0f, -15.0f),
      glm::vec3(-1.5f, -2.2f, -2.5f), glm::vec3(-3.8f, -2.0f, -12.3f),
      glm::vec3(2.4f, -0.4f, -3.5f),  glm::vec3(-1.7f, 3.0f, -7.5f),
      glm::vec3(1.3f, -2.0f, -2.5f),  glm::vec3(1.5f, 2.0f, -2.5f),
      glm::vec3(1.5f, 0.2f, -1.5f),   glm::vec3(-1.3f, 1.0f, -1.5f)};
  float angle = .0f;
  for(auto &pos : cubePositions) {
    world = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
    world = glm::translate(world, pos);
    world =
        glm::rotate(world, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
    angle += 20.0f;

    LightingProgram.SetMat4("World", glm::value_ptr(world));
    world = glm::transpose(glm::inverse(world));
    LightingProgram.SetMat4("WorldInvTranspose", glm::value_ptr(world));

    glDrawElements(GL_TRIANGLES, CubeIndexCount, GL_UNSIGNED_SHORT, nullptr);
  }

  glBindTexture(GL_TEXTURE_2D, 0);

  LightSourceProgram.Use();
  glBindVertexArray(SphereVAO);

  LightSourceProgram.SetMat4("ViewProj", glm::value_ptr(vp));

  for(auto &pl : PointLights) {
    world = glm::scale(glm::translate(glm::mat4(1.0f), pl.posW), glm::vec3(0.1f));
    LightSourceProgram.SetMat4("World", glm::value_ptr(world));
    world = glm::transpose(glm::inverse(world));
    // LightSourceProgram.SetMat4("WorldInvTranspose", glm::value_ptr(world));
    LightSourceProgram.SetVec3("AmbientLight", glm::value_ptr(pl.diffuse));

    glDrawElements(GL_TRIANGLES, SphereIndexCount, GL_UNSIGNED_SHORT, nullptr);
  }
  world = glm::scale(glm::translate(glm::mat4(1.0f), SpotLight0.posW), glm::vec3(0.05f));
  LightSourceProgram.SetMat4("World", glm::value_ptr(world));
  world = glm::transpose(glm::inverse(world));
  // LightSourceProgram.SetMat4("WorldInvTranspose", glm::value_ptr(world));
  LightSourceProgram.SetVec3("AmbientLight", glm::value_ptr(SpotLight0.diffuse));
  glDrawElements(GL_TRIANGLES, SphereIndexCount, GL_UNSIGNED_SHORT, nullptr);

  glBindVertexArray(0);
}

void init_light_and_material() {
  DirLight0 = {
    glm::normalize(glm::vec3{-0.2f, -1.0f, -0.3f}),
    {0.3f, 0.24f, 0.24f},
    {0.7f, 0.42f, 0.26f},
    {0.5f, 0.5f, 0.5f}
  };
  glm::vec3 pointLightColors[] = {
      glm::vec3(1.0f, 0.6f, 0.0f),
      glm::vec3(1.0f, 0.0f, 0.0f),
      glm::vec3(1.0f, 1.0, 0.0),
      glm::vec3(0.2f, 0.2f, 1.0f)
  };

  PointLights[0] = { { 0.7f,  0.2f,  2.0f},  0.1f*pointLightColors[0], pointLightColors[0], pointLightColors[0], {1.0f, 0.09f, 0.032f} };
  PointLights[1] = { { 2.3f, -3.3f, -4.0f},  0.1f*pointLightColors[1], pointLightColors[1], pointLightColors[1], {1.0f, 0.09f, 0.032f} };
  PointLights[2] = { {-4.0f,  2.0f, -12.0f}, 0.1f*pointLightColors[2], pointLightColors[2], pointLightColors[2], {1.0f, 0.09f, 0.032f} };
  PointLights[3] = { { 0.0f,  0.0f, -3.0f} , 0.1f*pointLightColors[3], pointLightColors[3], pointLightColors[3], {1.0f, 0.09f, 0.032f} };

  glm::vec3 spotDirection = glm::normalize(glm::vec3{-1.0f, 1.0f, -2.0f});
  SpotLight0 = {
    {3.0f, 3.0f, -3.0f},
    {spotDirection, 12.0f},
    glm::vec3{0.0f},
    glm::vec3{0.8f, 0.8f, 0.0f},
    glm::vec3{0.8f, 0.8f, 0.0f},
    glm::vec3{1.0f, 0.09f, 0.032f}
  };

  Mat = {
    {1.0f, 1.0f, 1.0f},
    {1.0f, 1.0f, 1.0f},
    {0.5f, 0.5f, 0.5f, 32.0f}
  };
}

void create_buffers() {

  GLuint buffers[2];
  GeometryGenerator::MeshData meshData;

  glGenVertexArrays(2, buffers);
  CubeVAO = buffers[0]; SphereVAO = buffers[1];

  glBindVertexArray(CubeVAO);
  glGenBuffers(2, buffers);
  CubeVBO = buffers[0], CubeEBO = buffers[1];
  glBindBuffer(GL_ARRAY_BUFFER, CubeVBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, CubeEBO);

  meshData = GeometryGenerator::CreateBox(1.0f, 1.0f, 1.0f, 0);
  glBufferData(GL_ARRAY_BUFFER, meshData.Vertices.size()*sizeof(GeometryGenerator::Vertex),
    meshData.Vertices.data(), GL_STATIC_DRAW);
  CubeIndexCount = (GLsizei)meshData.GetIndices16().size();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, CubeIndexCount*sizeof(uint16_t),
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

  glBindVertexArray(SphereVAO);
  glGenBuffers(2, buffers);
  SphereVBO = buffers[0]; SphereEBO = buffers[1];
  glBindBuffer(GL_ARRAY_BUFFER, SphereVBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, SphereEBO);

  meshData = GeometryGenerator::CreateSphere(1.0f, 64, 32);
  glBufferData(GL_ARRAY_BUFFER, meshData.Vertices.size()*sizeof(GeometryGenerator::Vertex),
    meshData.Vertices.data(), GL_STATIC_DRAW);
  SphereIndexCount = (GLsizei)meshData.GetIndices16().size();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, SphereIndexCount*sizeof(uint16_t),
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

int create_textures_and_samplers() {

  if(LoadTexture("../../../asset/textures/container2.png", &AlbedoTexture))
    return -1;
  if(LoadTexture("../../../asset/textures/container2_specular.png", &SpecularTexure))
    return -1;

  return 0;
}

void destroy_buffers() {
  GLuint buffers[] = {CubeVBO, CubeEBO};
  glDeleteBuffers(2, buffers);
  glDeleteVertexArrays(1, &CubeVAO);

  buffers[0] = SphereVBO; buffers[1] = SphereEBO;
  glDeleteBuffers(2, buffers);
  glDeleteVertexArrays(1, &SphereVAO);

  glDeleteTextures(1, &AlbedoTexture);
  glDeleteTextures(1, &SpecularTexure);
}
