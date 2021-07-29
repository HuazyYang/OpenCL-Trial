#include "glx_utils.hpp"
#include "glx_shader_program.hpp"
#include <cstdlib>
#include <fstream>
#include <glad/glad.h>
#include <iostream>
#include <string>

namespace GLx {

ShaderProgram::ShaderProgram() { programId_ = 0; }

ShaderProgram::~ShaderProgram() { Destory(); }

int ShaderProgram::Create(const char *vsFilePath, const char *fsFilePath) {
  return Create(vsFilePath, nullptr, nullptr, nullptr, fsFilePath);
}

int ShaderProgram::Create(const char *vsPath,  // VS
                         const char *tcsPath, // TCS
                         const char *tesPath, // TES
                         const char *gsPath,  // GS
                         const char *fsPath   // FS
) {

  std::ifstream fin;
  std::size_t srcLen;
  std::string srcBuffer;
  const char *psrcBuffer;
  char infoLog[512];

  auto is_path_valid = [](const char *filePath) {
    return filePath && filePath[0] != 0;
  };

  auto compile_shader = [&fin, &srcLen, &srcBuffer, &psrcBuffer,
                         &infoLog](const char *filePath, GLenum targetStage,
                                  GLuint *shaderObj) {
    int success;

    if (fin.open(filePath, std::ios_base::binary), !fin) {
      std::cout << "Error: Can not open file \"" << filePath << '\"' << std::endl;
      return -1;
    }

    fin.seekg(0, std::ios::end);
    srcLen = (std::size_t)fin.tellg();
    srcBuffer.resize(srcLen);
    fin.seekg(0, std::ios::beg);
    fin.read(&srcBuffer[0], srcLen);
    fin.close();

    *shaderObj = glCreateShader(targetStage);
    psrcBuffer = srcBuffer.c_str();
    glShaderSource(*shaderObj, 1, &psrcBuffer, (GLint *)&srcLen);
    glCompileShader(*shaderObj);

    glGetShaderiv(*shaderObj, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(*shaderObj, _countof(infoLog), nullptr, infoLog);
      std::cout << "Error: compile shader \"" << filePath << "\":\n"
                << infoLog << std::endl;
      return -1;
    }
    return 0;
  };

  const char *paths[] = {vsPath, tcsPath, tesPath, gsPath, fsPath};
  int valid[5];
  GLenum stages[] = {GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER,
                     GL_TESS_EVALUATION_SHADER, GL_GEOMETRY_SHADER,
                     GL_FRAGMENT_SHADER};
  GLuint shaderObjs[5];
  int i;
  int success;

  for (i = 0; i < 5; ++i) {
    if ((valid[i] = is_path_valid(paths[i])) &&
        compile_shader(paths[i], stages[i], &shaderObjs[i])) {
      for (--i; i >= 0; --i)
        if(valid[i])
          glDeleteShader(shaderObjs[i]);
      return -1;
    }
  }

  programId_ = glCreateProgram();
  for (i = 0; i < 5; ++i) {
    if(valid[i])
      glAttachShader(programId_, shaderObjs[i]);
  }
  glLinkProgram(programId_);

  for (i = 0; i < 5; ++i) {
    if(valid[i])
      glDeleteShader(shaderObjs[i]);
  }

  glGetProgramiv(programId_, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(programId_, _countof(infoLog), nullptr, infoLog);
    std::cout << "Error: link program:\n" << infoLog << std::endl;
    Destory();
    return -1;
  }

  return 0;
}

int ShaderProgram::Destory() {
  if (programId_) {
    glDeleteProgram(programId_);
    programId_ = 0;
  }
  return 0;
}

void ShaderProgram::Use() {
  if (programId_)
    glUseProgram(programId_);
}

ComputeProgram::ComputeProgram(): programId_(0) {}

ComputeProgram::~ComputeProgram() {
  Destroy();
}

void ComputeProgram::Destroy() {
  if(programId_) {
    glDeleteProgram(programId_);
    programId_ = 0;
  }
}

int ComputeProgram::Create(const char *filePath) {
    std::ifstream fin;
  std::size_t srcLen;
  std::string srcBuffer;
  const char *psrcBuffer;
  char infoLog[512];

  auto is_path_valid = [](const char *filePath) {
    return filePath && filePath[0] != 0;
  };

  auto compile_shader = [&fin, &srcLen, &srcBuffer, &psrcBuffer,
                         &infoLog](const char *filePath, GLenum targetStage,
                                  GLuint *shaderObj) {
    int success;

    if (fin.open(filePath, std::ios_base::binary), !fin) {
      std::cout << "Error: Can not open file \"" << filePath << '\"' << std::endl;
      return -1;
    }

    fin.seekg(0, std::ios::end);
    srcLen = (std::size_t)fin.tellg();
    srcBuffer.resize(srcLen);
    fin.seekg(0, std::ios::beg);
    fin.read(&srcBuffer[0], srcLen);
    fin.close();

    *shaderObj = glCreateShader(targetStage);
    psrcBuffer = srcBuffer.c_str();
    glShaderSource(*shaderObj, 1, &psrcBuffer, (GLint *)&srcLen);
    glCompileShader(*shaderObj);

    glGetShaderiv(*shaderObj, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(*shaderObj, _countof(infoLog), nullptr, infoLog);
      std::cout << "Error: compile shader \"" << filePath << "\":\n"
                << infoLog << std::endl;
      return -1;
    }
    return 0;
  };

  GLuint compShader;
  int success;

  if(compile_shader(filePath, GL_COMPUTE_SHADER, &compShader))
    return -1;

  Destroy();

  programId_ = glCreateProgram();
  glAttachShader(programId_, compShader);
  glLinkProgram(programId_);
  glDeleteShader(compShader);

  glGetProgramiv(programId_, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(programId_, _countof(infoLog), nullptr, infoLog);
    std::cout << "Error: link program:\n" << infoLog << std::endl;
    Destroy();
    return -1;
  }

  return 0;
}

void ComputeProgram::Use() {
  glUseProgram(programId_);
}

}