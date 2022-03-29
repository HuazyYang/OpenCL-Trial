#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>

#include "Camera.hpp"
#include "ShaderProgram.hpp"

static void framebuffer_size_callback(GLFWwindow *window, int width,
                                      int height);
static void mouse_callback(GLFWwindow *window, double xpos, double ypos);
static void mouse_button_callback(GLFWwindow *window, int button, int action,
                                  int mods);
static void mouse_scroll_callback(GLFWwindow *window, double xoffset,
                                  double yoffset);
static void processInput(GLFWwindow *window);
static void renderViewer(GLFWwindow *window);

static int createShaders();
static int createBuffers();
static int createTexturesAndSamplers();
static void cleanAllResources();

static ShaderProgram BasicShaderProgram;
static GLuint AlbedoTexture;
static GLuint AlbedoTexture2;
static GLuint VAO;
static GLuint EBO;
static GLuint VBO;
ArcBallCamera Camera;
glm::mat4 ProjectionMat;

glm::ivec2 LastMousePos;
bool Captured;

#ifdef _WIN32
int WINAPI wWinMain(HINSTANCE hInstance, HINSTANCE, PWSTR cmdLine,
                    int showCmd) {
#else
int main(int, char **) {
#endif

  int width, height;

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // create main window
  auto window = glfwCreateWindow(800, 600, "Hello OpenGL", nullptr, nullptr);
  if (!window) {
    std::cout << "Failed to create  the main window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, &framebuffer_size_callback);
  glfwSetCursorPosCallback(window, &mouse_callback);
  glfwSetMouseButtonCallback(window, &mouse_button_callback);
  glfwSetScrollCallback(window, &mouse_scroll_callback);

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)&glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // Enable MSAA
  glEnable(GL_MULTISAMPLE);

  // resize window for the first time
  glfwGetWindowSize(window, &width, &height);
  framebuffer_size_callback(window, width, height);

  if (createShaders())
    return -1;
  if (createBuffers())
    return -1;

  // set a arc ball camera
  Camera.SetPositions(glm::vec3(2.0f, .0f, .0f), 9.0f, .0f, .0f);

  if (createTexturesAndSamplers())
    return -1;

  // set texture sampler bound index in GLSL
  BasicShaderProgram.Use();
  BasicShaderProgram.SetInt("DiffuseSampler", 0);
  BasicShaderProgram.SetInt("DiffuseSampler2", 1);

  // render loop
  while (!glfwWindowShouldClose(window)) {

    // process input.
    processInput(window);
    // render viewer.
    renderViewer(window);
    // glfw: swap buffers and pull I/O events
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  cleanAllResources();

  glfwTerminate();
  return 0;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
  ProjectionMat = glm::perspective(glm::radians(45.0f), (float)width / height,
                                   .1f, 10000.0f);
}

void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, 1);
}

void renderViewer(GLFWwindow *window) {

  // clear back buffer(s) and depth-stencil buffer.
  glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  glClearDepth(1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glFrontFace(GL_CCW);

  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  BasicShaderProgram.Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, AlbedoTexture);
  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, AlbedoTexture2);

  glm::mat4 world{1.0f};
  glm::mat4 worldViewProj;
  glm::vec3 cubePositions[] = {
      glm::vec3(0.0f, 0.0f, 0.0f),    glm::vec3(2.0f, 5.0f, -15.0f),
      glm::vec3(-1.5f, -2.2f, -2.5f), glm::vec3(-3.8f, -2.0f, -12.3f),
      glm::vec3(2.4f, -0.4f, -3.5f),  glm::vec3(-1.7f, 3.0f, -7.5f),
      glm::vec3(1.3f, -2.0f, -2.5f),  glm::vec3(1.5f, 2.0f, -2.5f),
      glm::vec3(1.5f, 0.2f, -1.5f),   glm::vec3(-1.3f, 1.0f, -1.5f)};
  float angle = .0f;

  for (auto &pos : cubePositions) {
    world = glm::scale(glm::mat4(1.0f), glm::vec3(0.5f));
    world = glm::translate(world, pos);
    world =
        glm::rotate(world, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
    angle += 20.0f;
    worldViewProj = ProjectionMat * Camera.GetView() * world;

    BasicShaderProgram.SetMat4("WorldViewProj", glm::value_ptr(worldViewProj));

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, nullptr);
  }

  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
}

int createShaders() {
  return BasicShaderProgram.Create("./shaders/basic.vs", "./shaders/basic.fs");
}

int createBuffers() {

  struct Vertex {
    GLfloat v[3];
    GLfloat texcooord[2];
  } v[] = {
      // back
      {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
      {{+1.0f, -1.0f, -1.0f}, {1.0f, 0.0f}},
      {{+1.0f, +1.0f, -1.0f}, {1.0f, 1.0f}},
      {{-1.0f, +1.0f, -1.0f}, {0.0f, 1.0f}},
      // front
      {{-1.0f, -1.0f, +1.0f}, {0.0f, 0.0f}},
      {{+1.0f, -1.0f, +1.0f}, {1.0f, 0.0f}},
      {{+1.0f, +1.0f, +1.0f}, {1.0f, 1.0f}},
      {{-1.0f, +1.0f, +1.0f}, {0.0f, 1.0f}},
      // left
      {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
      {{-1.0f, +1.0f, -1.0f}, {1.0f, 0.0f}},
      {{-1.0f, +1.0f, +1.0f}, {1.0f, 1.0f}},
      {{-1.0f, -1.0f, +1.0f}, {0.0f, 1.0f}},
      // right
      {{+1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
      {{+1.0f, +1.0f, -1.0f}, {1.0f, 0.0f}},
      {{+1.0f, +1.0f, +1.0f}, {1.0f, 1.0f}},
      {{+1.0f, -1.0f, +1.0f}, {0.0f, 1.0f}},
      // bottom
      {{-1.0f, -1.0f, -1.0f}, {0.0f, 0.0f}},
      {{-1.0f, -1.0f, +1.0f}, {1.0f, 0.0f}},
      {{+1.0f, -1.0f, +1.0f}, {1.0f, 1.0f}},
      {{+1.0f, -1.0f, -1.0f}, {0.0f, 1.0f}},
      // top
      {{-1.0f, +1.0f, -1.0f}, {0.0f, 0.0f}},
      {{-1.0f, +1.0f, +1.0f}, {1.0f, 0.0f}},
      {{+1.0f, +1.0f, +1.0f}, {1.0f, 1.0f}},
      {{+1.0f, +1.0f, -1.0f}, {0.0f, 1.0f}},
  };
  GLushort indices[] = {
      0,  2,  1,  0,  3,  2,  4,  5,  6,  4,  6,  7,  8,  10, 9,  8,  11, 10,
      12, 13, 14, 12, 14, 15, 16, 18, 17, 16, 19, 18, 20, 21, 22, 20, 22, 23,
  };

  GLuint buffObjs[2];

  glGenBuffers(2, buffObjs);
  VBO = buffObjs[0], EBO = buffObjs[1];

  glGenVertexArrays(1, &VAO);
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(v), v, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                        (void *)(3 * sizeof(GLfloat)));
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // remember: do NOT unbind the EBO while a VAO is active as
  // the bound element buffer object IS stored in the VAO; keep the EBO bound.
  // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  return 0;
}

int createTexturesAndSamplers() {

  glGenTextures(1, &AlbedoTexture);
  glBindTexture(GL_TEXTURE_2D, AlbedoTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  int width, height, channels;
  unsigned char *data = stbi_load("../../../asset/textures/container.jpg",
                                  &width, &height, &channels, 0);
  if (!data) {
    std::cout << "ERROR:TEXTURE FILE:can not load file" << std::endl;
    return -1;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB,
               GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);
  stbi_image_free(data);

  glGenTextures(1, &AlbedoTexture2);
  glBindTexture(GL_TEXTURE_2D, AlbedoTexture2);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  data = stbi_load("../../../asset/textures/awesomeface.png", &width, &height,
                   &channels, 0);
  if (!data) {
    std::cout << "ERROR:TEXTURE FILE:can not load file" << std::endl;
    return -1;
  }
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);
  stbi_image_free(data);

  glBindTexture(GL_TEXTURE_2D, 0);

  return 0;
}

void cleanAllResources() {

  GLuint buffObjs[2] = {VBO, EBO};
  glDeleteBuffers(2, buffObjs);
  glDeleteVertexArrays(0, &VAO);

  glDeleteTextures(1, &AlbedoTexture);
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  if (Captured) {
    float dx = (float)((int)xpos - LastMousePos.x);
    float dy = (float)((int)LastMousePos.y - ypos);
    LastMousePos = glm::ivec2(xpos, ypos);

    Camera.Rotate(dx, dy);
  }
}

void mouse_button_callback(GLFWwindow *window, int button, int action,
                           int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    Captured = action == GLFW_PRESS;
    if (Captured) {
      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      LastMousePos = glm::ivec2((int)xpos, (int)ypos);
    }
  }
}

void mouse_scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  Camera.Zoom(-(float)yoffset, 1.0f, 20.0f);
}
