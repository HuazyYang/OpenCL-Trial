#include "pbr_pre_compute.hpp"
#include <CommonUtils.hpp>
#include <ShaderProgram.hpp>
#include <cstdio>
#include <stb_image_write.h>
#include <memory>
#include <random>
#include <iostream>
#include <filesystem>

namespace PBRUtils {

int GenerateDiffuseIrradianceCubeTextureFiles(GLx::GLTextureRef inputEnvCubeMap, GLsizei width,
                                              GLsizei height, const char *filePathPrefix) {

  ComputeProgram intProgram;
  GLx::GLTexture diffuseIrrCubeTexture;

  if (intProgram.Create("pbr/shaders/diffuse_irrmap.comp"))
    return -1;

  // Create diffuse irradiance cube texture.
  diffuseIrrCubeTexture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_CUBE_MAP, diffuseIrrCubeTexture);
  glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGBA16F, width, height);

  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  // dispatch compute thread groups to execute shader.
  intProgram.Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, inputEnvCubeMap);
  glBindImageTexture(0, diffuseIrrCubeTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

  glDispatchCompute((GLuint)std::ceil(width / 32.0f), (GLuint)std::ceil(height / 32.0f), 6);
  glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);

  GLx::GLFrameBuffer rdFBO;
  rdFBO = GLx::GLFrameBuffer::New();

  const GLsizei buffSize = 3 * width * height * sizeof(float);
  std::unique_ptr<char[]> data{new char[buffSize]};

  const char *facesSuffices[6] = {"right", "left", "top", "bottom", "front", "back"};
  char path[256];

  glBindFramebuffer(GL_READ_FRAMEBUFFER, rdFBO);

  for (int i = 0; i < 6; ++i) {
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, diffuseIrrCubeTexture, 0);

    if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      std::cout << "FrameBuffer incomplete" << std::endl;
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      return -1;
    }

    glReadBuffer(GL_COLOR_ATTACHMENT0);

    std::snprintf(path, _countof(path), "%s%s.hdr", filePathPrefix, facesSuffices[i]);
    glReadnPixels(0, 0, width, height, GL_RGB, GL_FLOAT, buffSize, data.get());
    if (!stbi_write_hdr(path, width, height, 3, (const float *)data.get())) {
      std::cout << "Can not write to file" << std::endl;
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      return -1;
    }
  }

  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

  return 0;
}

extern int GenerateSpecularIrradianceMipTextureFiles(GLx::GLTextureRef inputEnvCubeMap,
                                                     GLsizei width, GLsizei height,
                                                     GLsizei mipLevels,
                                                     const char *filePathPrefix) {

  GLsizei maxMipLevel;
  GLsizei dim = (GLuint)std::min(width, height);
  for (maxMipLevel = 0; dim >= 1; dim >>= 1, maxMipLevel += 1)
    ;

  if (mipLevels > maxMipLevel) {
    std::cout << "Mip level exceeds maximum mip level: " << maxMipLevel << '.' << std::endl;
    return -1;
  }

  ComputeProgram accumProgram;
  GLx::GLTexture irrCubeTexture;
  std::vector<std::pair<int, int>> mipDimensions{(std::size_t)mipLevels};
  GLx::GLBuffer roughnessUB;
  float roughnessUBuffer[4];
  int layerWidth, layerHeight;

  if (accumProgram.Create("pbr/shaders/specular_irrmap.comp"))
    return -1;

  // Cube map texture with layers
  irrCubeTexture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_CUBE_MAP, irrCubeTexture);
  glTexStorage2D(GL_TEXTURE_CUBE_MAP, mipLevels, GL_RGBA16F, width, height);
  for (unsigned level = 0; level < (unsigned)mipLevels; ++level) {
    glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, level, GL_TEXTURE_WIDTH, &layerWidth);
    glGetTexLevelParameteriv(GL_TEXTURE_CUBE_MAP_POSITIVE_X, level, GL_TEXTURE_HEIGHT, &layerHeight);
    mipDimensions[level] = {layerWidth, layerHeight};
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  // Uniform buffer
  roughnessUB = GLx::GLBuffer::New();
  glBindBuffer(GL_UNIFORM_BUFFER, roughnessUB);
  glBufferData(GL_UNIFORM_BUFFER, sizeof(roughnessUBuffer), nullptr, GL_STATIC_DRAW);

  accumProgram.Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, inputEnvCubeMap);
  glBindBufferBase(GL_UNIFORM_BUFFER, 0, roughnessUB);

  for (unsigned level = 0; level < (unsigned)mipLevels; ++level) {
    roughnessUBuffer[0] = (float)level / (mipLevels + 1);
    glBindBuffer(GL_UNIFORM_BUFFER, roughnessUB);
    glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(roughnessUBuffer), roughnessUBuffer);

    glBindImageTexture(0, irrCubeTexture, level, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

    glDispatchCompute((GLuint)std::ceil(mipDimensions[level].first / 32.0f),
                      (GLuint)std::ceil(mipDimensions[level].second / 32.0f), 6);
  }

  glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);

  GLx::GLFrameBuffer rdFBO;
  int maxColorAttachments;
  int left;
  int index;
  int attachments;
  const char *facesSuffices[6] = {"right", "left", "top", "bottom", "front", "back"};
  char path[256];
  const int pixelBuffSize = width * height * sizeof(float) * 3;
  std::unique_ptr<char[]> pixelBuff{new char[pixelBuffSize]};

  rdFBO = GLx::GLFrameBuffer::New();
  glBindFramebuffer(GL_READ_FRAMEBUFFER, rdFBO);

  glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &maxColorAttachments);
  maxColorAttachments = std::min(maxColorAttachments, 6);

  // Store the results.
  for (unsigned level = 0; level < (unsigned)mipLevels; ++level) {

    snprintf(path, _countof(path), "%slevel_%d", filePathPrefix, level);
    std::filesystem::create_directory(path);

    left = 6;
    index = 0;
    while (left > 0) {
      int i;
      for (i = 0; i < maxColorAttachments && left > 0; ++i, ++index, --left) {
        glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
                               GL_TEXTURE_CUBE_MAP_POSITIVE_X + index, irrCubeTexture, level);
      }
      attachments = i;

      if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cout << "FrameBuffer incomplete" << std::endl;
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        return -1;
      }

      index -= attachments;
      for (i = 0; i < attachments; ++i, ++index) {
        glReadBuffer(GL_COLOR_ATTACHMENT0 + i);

        glReadnPixels(0, 0, mipDimensions[level].first, mipDimensions[level].second, GL_RGB,
                      GL_FLOAT, pixelBuffSize, pixelBuff.get());

        std::snprintf(path, _countof(path), "%slevel_%d/%s.hdr", filePathPrefix, level,
                      facesSuffices[index]);

        if (!stbi_write_hdr(path, mipDimensions[level].first, mipDimensions[level].second, 3,
                            (const float *)pixelBuff.get())) {
          std::cout << "Can not write to file" << std::endl;
          return -1;
        }
      }
    }
  }

  return 0;
}

int GenerateSpecularBRDFTextureFile(GLx::GLTextureRef inputEnvCubeMap, GLsizei width,
                                    GLsizei height, const char *filePath) {

  ComputeProgram intProgram;
  GLx::GLTexture lutImage;

  if (intProgram.Create("pbr/shaders/specular_brdf.comp"))
    return -1;

  lutImage = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_2D, lutImage);
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG16F, width, height);
  glBindTexture(GL_TEXTURE_2D, 0);

  intProgram.Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_CUBE_MAP, inputEnvCubeMap);
  glBindImageTexture(0, lutImage, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RG16F);

  glDispatchCompute((GLuint)std::ceil(width / 32.0f), (GLuint)std::ceil(height / 32.0f), 1);

  GLx::GLFrameBuffer rdFBO = GLx::GLFrameBuffer::New();
  const int pixelBuffSize = width * height * sizeof(float) * 3;
  std::unique_ptr<char[]> pixelBuff{new char[pixelBuffSize]};

  glBindFramebuffer(GL_READ_FRAMEBUFFER, rdFBO);
  glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, lutImage, 0);
  if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cout << "FrameBuffer incomplete" << std::endl;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    return -1;
  }

  glReadBuffer(GL_COLOR_ATTACHMENT0);

  glReadnPixels(0, 0, width, height, GL_RGB, GL_FLOAT, pixelBuffSize, pixelBuff.get());

  if (!stbi_write_hdr(filePath, width, height, 3, (const float *)pixelBuff.get())) {
    std::cout << "Can not write to file" << std::endl;
    return -1;
  }

  return 0;
}

} // namespace PBRUtils
