#include "ShaderProgram.hpp"
#include "CommonUtils.hpp"
#include <cstdlib>
#include <fstream>
#include <glad/glad.h>
#include <iostream>
#include <string>

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

void ShaderProgram::SetBool(const char *name, bool value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniform1i(cpos, value);
}
void ShaderProgram::SetInt(const char *name, int value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniform1i(cpos, value);
}
void ShaderProgram::SetFloat(const char *name, float value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniform1f(cpos, value);
}

void ShaderProgram::SetVec2(const char *name, const float *value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniform2fv(cpos, 1, value);
}

void ShaderProgram::SetVec3(const char *name, const float *value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniform3fv(cpos, 1, value);
}

void ShaderProgram::SetVec4(const char *name, const float *value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniform4fv(cpos, 1, value);
}

void ShaderProgram::SetMat2x2(const char *name, const float *value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniformMatrix2fv(cpos, 1, GL_FALSE, value);
}

void ShaderProgram::SetMat3x2(const char *name, const float *value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniformMatrix3x2fv(cpos, 1, GL_FALSE, value);
}

void ShaderProgram::SetMat3x3(const char *name, const float *value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniformMatrix3fv(cpos, 1, GL_FALSE, value);
}

void ShaderProgram::SetMat4(const char *name, const float *value) {
  GLint cpos = glGetUniformLocation(programId_, name);
  GTX_ASSERT(cpos >= 0 && "Name does not exist");
  if (cpos >= 0)
    glUniformMatrix4fv(cpos, 1, GL_FALSE, value);
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