#pragma once
#include <vector>
#include "skm_mesh.hpp"
#include "material.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <string>
#include <glm/glm.hpp>
#include <map>
#include "animdata.h"

class Model {
public:
  Model();

  int Load(const char *path);
  void Draw();

  std::map<std::string, BoneInfo>&  GetOffsetMatMap();
  size_t GetBoneCount() const;

private:
  void loadModel(const char *path);
  void processNode(aiNode *node, const aiScene *scene);
  Mesh processMesh(aiMesh *mesh, const aiScene *scene);
  Material loadMaterialTextures(aiMaterial *mat, int matIndex);
  void SetVertexBoneData(Vertex *pVertex, int boneId, float weights);
  void ExtractBoneWeightsForVertices(std::vector<Vertex> &vertices, aiMesh *mesh, const aiScene *scene);

  std::string directory;
  std::vector<Material> material_dicts;
  std::vector<std::pair<std::string, GLx::GLTexture>> texture_dicts;
  std::vector<Mesh> meshes;
  std::map<std::string, BoneInfo> bone_infos;
  int next_bone_index;
  bool gamma_correction;

  glm::vec3 aabb_min, aabb_max;
};