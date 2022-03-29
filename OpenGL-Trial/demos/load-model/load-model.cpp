#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
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
#include "model.hpp"

static glm::mat4 ProjectionMat;
static ArcBallCamera Camera;
static glm::ivec2 LastMousePos;
static bool Captured;

static ShaderProgram ColorProgram;
static Model TestModel;
GLx::GLBuffer FrameUBO;

struct FrameUniformBuffer {
  glm::mat4x4 ViewProj;
};

// UI
static bool EnableWireframe;

static int create_shader_programs();

static void render_view(GLFWwindow *window);
static void update(GLFWwindow *window);

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
  window = glfwCreateWindow(windowWidth, windowHeight, "load-model", nullptr, nullptr);
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

  if(TestModel.Load("../../../asset/objects/backpack/backpack.obj"))
    return -1;

  Camera.SetPositions(glm::vec3{0.0f}, 5.0f, -30.0f, -90.0f);

  if (create_shader_programs())
    return -1;

  // MSAA
  glEnable(GL_MULTISAMPLE);

  // Cull back
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);

  while (!glfwWindowShouldClose(window)) {
    process_keystrokes_input(window);

    update(window);
    render_view(window);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}

void update(GLFWwindow *window) {

  FrameUniformBuffer buffer;

  buffer.ViewProj = ProjectionMat *  Camera.GetView();

  glNamedBufferSubData(FrameUBO, 0, sizeof(buffer), &buffer);
}

void render_view(GLFWwindow *window) {

  glClearColor(.0f, .0f, .0f, 1.f);
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  ColorProgram.Use();
  glBindBufferBase(GL_UNIFORM_BUFFER, 0, FrameUBO);

  TestModel.Draw();
}

int create_shader_programs() {

  if(ColorProgram.Create(
    "shaders/color.vs",
    "shaders/color.fs"))
    return -1;

  FrameUBO = GLx::GLBuffer::New();
  glBindBuffer(GL_UNIFORM_BUFFER, FrameUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(FrameUniformBuffer), nullptr, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  return 0;
}
