
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

static GLuint SkyBoxVAO;
static GLuint CubeVAO, CubeVBO, CubeEBO;
static GLsizei CubeIndexCount;
static GLuint EnvMapTexture;

static ShaderProgram SkyBoxProgram;
static ShaderProgram EnvProgram;

static int create_shader_programs();
static int create_mesh_buffers();
static int create_textures();

static void render_view(GLFWwindow *window);

static void destroy_resources();

int main(int, char**) {
  
  GLFWwindow *window;
  int windowWidth = 800, windowHeight = 600;
  auto resize_framebuffer_callback = [](GLFWwindow *, int width, int height) {
    glViewport(0, 0, width, height);
    ProjectionMat = glm::perspective(glm::radians(45.0f), (float)width/height, .1f, 1000.0f);
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
  window = glfwCreateWindow(windowWidth, windowHeight, "Cube Environment Map", nullptr, nullptr);
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

  Camera.SetPositions(glm::vec3{0.0f}, 5.0f, 0.0f, -90.0f);

  if(create_shader_programs() ||
     create_textures() ||
     create_mesh_buffers())
    return -1;

  // MSAA
  glEnable(GL_MULTISAMPLE);

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

  glm::mat4 W, /*V,*/ P;
  glm::vec2 projectInvXY;
  glm::mat4 viewInverse;
  glClearDepth(1.0f);
  glClear(GL_DEPTH_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

  // For render skybox
  // RS, DDS
  glDisable(GL_DEPTH_TEST);
  glCullFace(GL_BACK);

  // bind a dummy vertex array, this is necessary athrough we do
  // not acturally need it.
  glBindVertexArray(SkyBoxVAO);

  SkyBoxProgram.Use();

  projectInvXY = glm::vec2(1.0f / ProjectionMat[0][0], 1.0f / ProjectionMat[1][1]);
  viewInverse = glm::inverse(Camera.GetView());

  SkyBoxProgram.SetVec2("ProjectInvXY", glm::value_ptr(projectInvXY));
  SkyBoxProgram.SetMat4("ViewInverse", glm::value_ptr(viewInverse));
  SkyBoxProgram.SetInt("EnvMapTexture", 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, EnvMapTexture);

  // Draw sky box screen quad
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

  // For render something else
  glEnable(GL_DEPTH_TEST);

  EnvProgram.Use();

  glBindVertexArray(CubeVAO);

  W = glm::mat4(1.0f);
  EnvProgram.SetMat4("World", glm::value_ptr(W));
  EnvProgram.SetMat4("WorldInvTranspose", glm::value_ptr(W));
  P =  ProjectionMat * Camera.GetView();
  EnvProgram.SetMat4("ViewProj", glm::value_ptr(P));
  EnvProgram.SetVec3("EyePosW", glm::value_ptr(Camera.GetEyePosW()));
  EnvProgram.SetInt("EnvMapTexture", 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, EnvMapTexture);

  glDrawElements(GL_TRIANGLES, CubeIndexCount, GL_UNSIGNED_SHORT, nullptr);
}

void destroy_resources() {
  glDeleteTextures(1, &EnvMapTexture);
  glDeleteVertexArrays(1, &SkyBoxVAO);

  glDeleteVertexArrays(1, &CubeVAO);
  glDeleteBuffers(1, &CubeVBO);
  glDeleteBuffers(1, &CubeEBO);
}

int create_shader_programs() {

  if(SkyBoxProgram.Create("shaders/skybox.vs", "shaders/skybox.fs"))
    return -1;
  if(EnvProgram.Create("shaders/env_reflect.vs", "shaders/env_reflect.fs"))
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
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(GeometryGenerator::Vertex),
      (void *)offsetof(GeometryGenerator::Vertex, TexC));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  };

  // create a dummy vertex array object for sky box
  glGenVertexArrays(1, &SkyBoxVAO);

  auto meshData = GeometryGenerator::CreateBox(1.0f, 1.0f, 1.0f, 1);

  buffer_data_and_set_vertex_layouts(CubeVAO, CubeVBO, CubeEBO, CubeIndexCount, meshData);

  return 0;
}

int create_textures() {

  const char * faces_fnames[] = {
    "../../../asset/textures/lake-mountain-skybox/right.jpg",
    "../../../asset/textures/lake-mountain-skybox/left.jpg",
    "../../../asset/textures/lake-mountain-skybox/top.jpg",
    "../../../asset/textures/lake-mountain-skybox/bottom.jpg",
    "../../../asset/textures/lake-mountain-skybox/front.jpg",
    "../../../asset/textures/lake-mountain-skybox/back.jpg",
  };
  int i;
  int face;
  const unsigned char *data;
  int width, height;
  int channels;
  int format;

  glGenTextures(1, &EnvMapTexture);
  glBindTexture(GL_TEXTURE_CUBE_MAP, EnvMapTexture);

  for(i = 0; i < 6; ++i) {
    face = i + GL_TEXTURE_CUBE_MAP_POSITIVE_X;
    data = stbi_load(faces_fnames[i], &width, &height, &channels, 0);
    if(!data) {
      std::cout << "ERROR: failed to load cube map face \"" << faces_fnames[i] << "\"" << std::endl;
      return -1;
    }

    switch (channels) {
      case 1: format = GL_RED; break; 
      case 2: format = GL_RG; break;
      case 3: format = GL_RGB; break;
      case 4: format = GL_RGBA; break;
    }

    glTexImage2D(face, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
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