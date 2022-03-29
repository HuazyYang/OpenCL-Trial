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
#include "skm_model.hpp"
#include "animator.h"
#include "animation.h"

static glm::mat4 ProjectionMat;
static ArcBallCamera Camera;
static glm::ivec2 LastMousePos;
static bool Captured;

static ShaderProgram ColorProgram;
GLx::GLBuffer FrameUBO;
GLx::GLBuffer FinalBoneMatricesSSBO;

struct FrameUniformBuffer {
  glm::mat4x4 ViewProj;
};

struct FinalBoneMatrixSSBHeader {
  int32_t FinalBoneMatrixCount;
  int32_t padding0[15];
};

// UI
static bool EnableWireframe;

static int create_shader_programs();

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
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
      EnableWireframe ^= true;
  };

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  // create main window
  window = glfwCreateWindow(windowWidth, windowHeight, "skeletal-animation", nullptr, nullptr);
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
    Camera.Zoom(-(float)yoffset, 1.0f, 10000.0f);
  });

  // glad: load all OpenGL function pointers
  if (!gladLoadGLLoader((GLADloadproc)&glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    return -1;
  }

  // resize the window for the first time
  resize_framebuffer_callback(window, windowWidth, windowHeight);

  Camera.SetPositions(glm::vec3{0.0f}, 5.0f, -30.0f, -90.0f);

  if (create_shader_programs())
    return -1;

  // load models
  // -----------
  Model ourModel;
  if (ourModel.Load("../../../asset/objects/vampire/dancing_vampire.dae"))
    return -1;
  Animation danceAnimation("../../../asset/objects/vampire/dancing_vampire.dae", &ourModel);
  Animator animator(&danceAnimation);

  float deltaFrame, lastFrame, currFrame;

  lastFrame = currFrame = glfwGetTime();

  // MSAA
  glEnable(GL_MULTISAMPLE);

  // Cull back
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);

  while (!glfwWindowShouldClose(window)) {
    process_keystrokes_input(window);

    // Update frame
    FrameUniformBuffer buffer;
    glm::mat4 model{1.0f};

    model = glm::translate(model, glm::vec3(.0f,  -.4f, .0f));
    model = glm::scale(model, glm::vec3(.5f, .5f, .5f));

    buffer.ViewProj = ProjectionMat * Camera.GetView() * model;

    glNamedBufferSubData(FrameUBO, 0, sizeof(buffer), &buffer);

    currFrame = glfwGetTime();
    deltaFrame = currFrame - lastFrame;
    lastFrame = currFrame;

    animator.UpdateAnimation(deltaFrame);

    auto &transforms = animator.GetPoseTransforms();
    GLint ssboByteSize;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, FinalBoneMatricesSSBO);
    glGetBufferParameteriv(GL_SHADER_STORAGE_BUFFER, GL_BUFFER_SIZE, &ssboByteSize);
    void *pBuffer = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
    std::memcpy(pBuffer, transforms.data(), std::min((size_t)ssboByteSize, transforms.size() * sizeof(glm::mat4)));
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    // Render frame
    glClearColor(.0f, .0f, .0f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    ColorProgram.Use();

    glBindBufferBase(GL_UNIFORM_BUFFER, 0, FrameUBO);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, FinalBoneMatricesSSBO);

    ourModel.Draw();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}

int create_shader_programs() {

  if (ColorProgram.Create("shaders/blend-animation.vs", "shaders/blend-animation.fs"))
    return -1;

  FrameUBO = GLx::GLBuffer::New();
  glBindBuffer(GL_UNIFORM_BUFFER, FrameUBO);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(FrameUniformBuffer), nullptr, GL_STATIC_DRAW);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);

  FinalBoneMatricesSSBO = GLx::GLBuffer::New();
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, FinalBoneMatricesSSBO);
  glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(glm::mat4) * 120, nullptr, GL_DYNAMIC_COPY);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

  return 0;
}
