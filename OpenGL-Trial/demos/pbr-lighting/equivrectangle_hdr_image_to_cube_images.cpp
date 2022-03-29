#include <CommonUtils.hpp>
#include <ShaderProgram.hpp>
#include <glad/glad.h>
#include <iostream>
#include <memory>
#include <stb_image.h>
#include <stb_image_write.h>

int ConvertEquivrectangleHdrImageToCubeImages(const char *hdrImagePath,
                                              const char *cubeImagesPathPrefix) {
  int width, height;
  int nrComponents;
  GLx::GLTexture equivEnvMapTexture;
  GLx::GLTexture cubeMapTexture;
  ComputeProgram convProgram;

  if (convProgram.Create("shaders/equivrectangle_to_cubemap.comp"))
    return -1;

  // alloc resources
  const float *data = stbi_loadf(hdrImagePath, &width, &height, &nrComponents, 0);
  if (!data) {
    std::cout << "ERROR: can not load file \"" << hdrImagePath << std::endl;
    return -1;
  }

  equivEnvMapTexture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_2D, equivEnvMapTexture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGB, GL_FLOAT, data);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  stbi_image_free((void *)data);
  glBindTexture(GL_TEXTURE_2D, 0);

  cubeMapTexture = GLx::GLTexture::New();
  glBindTexture(GL_TEXTURE_CUBE_MAP, cubeMapTexture);
  height = (width >>= 2);
  glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGBA16F, width, height);
  for (int i = 0; i < 6; ++i) {
    glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, 0, 0, width, height, GL_RGBA16F, GL_FLOAT,
                    nullptr);
  }
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glBindTexture(GL_TEXTURE_CUBE_MAP, 0);

  // Dispatch compatation
  convProgram.Use();
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, equivEnvMapTexture);
  glBindImageTexture(0, cubeMapTexture, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_RGBA16F);

  glDispatchCompute((GLuint)std::ceil(width / 32.0f),
                    (GLuint)std::ceil(height / 32.0f), 6);
  glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

  GLx::GLFrameBuffer rdFBO = GLx::GLFrameBuffer::New();

  const GLsizei buffSize = 4 * width * height;
  std::unique_ptr<float[]> wdata {new float[buffSize]};

  const char *facesSuffices[6] = {"right", "left", "top", "bottom", "front", "back"};
  char path[256];

  glBindFramebuffer(GL_READ_FRAMEBUFFER, rdFBO);

  for (int i = 0; i < 6; ++i) {
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, cubeMapTexture, 0);

    if (glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      std::cout << "FrameBuffer incomplete" << std::endl;
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      return -1;
    }

    glReadBuffer(GL_COLOR_ATTACHMENT0);

    std::sprintf(path, "%s%s.hdr", cubeImagesPathPrefix, facesSuffices[i]);

    glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, &wdata[0]);
    if (!stbi_write_hdr(path, width, height, 3, &wdata[0])) {
      std::cout << "Can not write to file" << std::endl;
      glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
      return -1;
    }
  }

  glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

  return 0;
}