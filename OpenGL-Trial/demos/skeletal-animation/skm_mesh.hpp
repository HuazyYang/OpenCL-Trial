#pragma once
#include <glm/glm.hpp>
#include <CommonUtils.hpp>
#include "material.hpp"

enum {MAX_BONE_WEIGHT_COUNT = 4};

struct Vertex {
  glm::vec3 Pos;
  glm::vec3 Normal;
  glm::vec3 Tangent;
  glm::vec2 TexCoord;

  int32_t BoneIds[MAX_BONE_WEIGHT_COUNT];
  float DeformWeights[MAX_BONE_WEIGHT_COUNT];

  Vertex();
};

class Mesh {
public:

  Mesh();

  int Create(const Vertex *pVertex, size_t vStride, size_t vCount, const unsigned int *pIndex,
             size_t iCount, GLx::PRIMITIVE_TOPOLOGY_TYPE primitiveType);

  void SetMaterial(const Material *mat);

  void Draw();

private:
  const Material *material;
  GLx::GLVAO VAO;
  GLx::GLBuffer VBO, EBO;
  size_t index_count;
  GLx::PRIMITIVE_TOPOLOGY_TYPE primitive_type;
};