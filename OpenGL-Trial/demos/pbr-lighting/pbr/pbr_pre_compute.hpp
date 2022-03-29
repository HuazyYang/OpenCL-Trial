#pragma once
#include <CommonUtils.hpp>

namespace PBRUtils {

  extern int GenerateDiffuseIrradianceCubeTextureFiles(
    GLx::GLTextureRef inputEnvCubeMap,
    GLsizei width,
    GLsizei height,
    const char *filePathPrefix
  );

  extern int GenerateSpecularIrradianceMipTextureFiles(
    GLx::GLTextureRef inputEnvCubeMap,
    GLsizei width,
    GLsizei height,
    GLsizei mipLevels,
    const char *filePathPrefix
  );

  extern int GenerateSpecularBRDFTextureFile(
    GLx::GLTextureRef inputEnvCubeMap,
    GLsizei width,
    GLsizei height,
    const char *filePath
  );
};