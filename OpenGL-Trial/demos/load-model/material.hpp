#pragma once
#include <CommonUtils.hpp>

class Material {
public:
  enum {MAX_DIFFUSE_TEXTURE_COUNT = 4};
  enum {MAX_SPECULAR_TEXTURE_COUNT = 4};
  enum {MAX_NORMAL_TEXTURE_COUNT = 4};
  enum {MAX_HEIGHT_TEXTURE_COUNT = 4};

  Material(int index = -1);

  int GetIndex() const;

  GLx::GLTextureRef GetDiffuseTexture(int i) const;
  GLx::GLTextureRef GetSpecularTexture(int i) const;
  GLx::GLTextureRef GetNormalTexture(int i) const;
  GLx::GLTextureRef GetHeightTexture(int i) const;

  void SetDiffuseTextures(const GLx::GLTextureRef *pTextures, size_t count);
  void SetSpecularTextures(const GLx::GLTextureRef *pTextures, size_t count);
  void SetNormalTextures(const GLx::GLTextureRef *pTextures, size_t count);
  void SetHeightTextures(const GLx::GLTextureRef *pTextures, size_t count);

private:
  int index;
  GLx::GLTextureRef diffuse_textures[MAX_DIFFUSE_TEXTURE_COUNT];
  GLx::GLTextureRef specular_textures[MAX_SPECULAR_TEXTURE_COUNT];
  GLx::GLTextureRef normal_textures[MAX_NORMAL_TEXTURE_COUNT];
  GLx::GLTextureRef height_textures[MAX_HEIGHT_TEXTURE_COUNT];
};