#include "material.hpp"
#include <cstdlib>

Material::Material(int index) { this->index = index; }

int Material::GetIndex() const {
  return index;
}

GLx::GLTextureRef Material::GetDiffuseTexture(int i) const {
  GTX_ASSERT(i >= 0 && i < MAX_DIFFUSE_TEXTURE_COUNT);
  if (i >= 0 && i < MAX_DIFFUSE_TEXTURE_COUNT)
    return GLx::GLTextureRef(diffuse_textures[i]);
  return GLx::GLTextureRef(0);
}

GLx::GLTextureRef Material::GetSpecularTexture(int i) const {
  GTX_ASSERT(i >= 0 && i < MAX_SPECULAR_TEXTURE_COUNT);
  if (i >= 0 && i < MAX_SPECULAR_TEXTURE_COUNT)
    return GLx::GLTextureRef(specular_textures[i]);
  return GLx::GLTextureRef(0);
}

GLx::GLTextureRef Material::GetNormalTexture(int i) const {
  GTX_ASSERT(i >= 0 && i < MAX_NORMAL_TEXTURE_COUNT);
  if (i >= 0 && i < MAX_NORMAL_TEXTURE_COUNT)
    return GLx::GLTextureRef(normal_textures[i]);
  return GLx::GLTextureRef(0);
}

GLx::GLTextureRef Material::GetHeightTexture(int i) const {
  GTX_ASSERT(i >= 0 && i < MAX_HEIGHT_TEXTURE_COUNT);
  if (i >= 0 && i < MAX_HEIGHT_TEXTURE_COUNT)
    return GLx::GLTextureRef(diffuse_textures[i]);
  return GLx::GLTextureRef(0);
}

void Material::SetDiffuseTextures(const GLx::GLTextureRef *pTextures, size_t count) {
  GTX_ASSERT(count <= MAX_DIFFUSE_TEXTURE_COUNT);
  if (count <= MAX_DIFFUSE_TEXTURE_COUNT) {
    size_t i;
    for (i = 0; i < count; ++i) {
      diffuse_textures[i] = pTextures[i];
    }
    for(; i < MAX_DIFFUSE_TEXTURE_COUNT; ++i)
      diffuse_textures[i] = GLx::GLTextureRef(0);
  }
}
void Material::SetSpecularTextures(const GLx::GLTextureRef *pTextures, size_t count) {
  GTX_ASSERT(count <= MAX_SPECULAR_TEXTURE_COUNT);
  if (count <= MAX_SPECULAR_TEXTURE_COUNT) {
    size_t i;
    for (i = 0; i < count; ++i) {
      specular_textures[i] = pTextures[i];
    }
    for(; i < MAX_SPECULAR_TEXTURE_COUNT; ++i)
      specular_textures[i] = GLx::GLTextureRef(0);
  }
}
void Material::SetNormalTextures(const GLx::GLTextureRef *pTextures, size_t count) {
  GTX_ASSERT(count <= MAX_NORMAL_TEXTURE_COUNT);
  if (count <= MAX_NORMAL_TEXTURE_COUNT) {
    size_t i;
    for (i = 0; i < count; ++i) {
      normal_textures[i] = pTextures[i];
    }
    for(; i < MAX_NORMAL_TEXTURE_COUNT; ++i)
      normal_textures[i] = GLx::GLTextureRef(0);
  }
}
void Material::SetHeightTextures(const GLx::GLTextureRef *pTextures, size_t count) {
  GTX_ASSERT(count <= MAX_HEIGHT_TEXTURE_COUNT);
  if (count <= MAX_HEIGHT_TEXTURE_COUNT) {
    size_t i;
    for (i = 0; i < count; ++i) {
      height_textures[i] = pTextures[i];
    }
    for(; i < MAX_HEIGHT_TEXTURE_COUNT; ++i)
      height_textures[i] = GLx::GLTextureRef(0);
  }
}