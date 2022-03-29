#pragma once
#include <vector>
#include "mesh.hpp"
#include "material.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <string>
#include <glm/glm.hpp>

class Model {
public:
  Model();

  int Load(const char *path);
  void Draw();

private:
  void loadModel(const char *path);
  void processNode(aiNode *node, const aiScene *scene);
  Mesh processMesh(aiMesh *mesh, const aiScene *scene);
  Material loadMaterialTextures(aiMaterial *mat, int matIndex);

  std::string directory;
  std::vector<Material> material_dicts;
  std::vector<std::pair<std::string, GLx::GLTexture>> texture_dicts;
  std::vector<Mesh> meshes;
  bool gamma_correction;

  glm::vec3 aabb_min, aabb_max;
};