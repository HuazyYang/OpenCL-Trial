#include "skm_mesh.hpp"

Vertex::Vertex() {
  for (int i = 0; i < MAX_BONE_WEIGHT_COUNT; ++i) {
    BoneIds[i] = -1;
    DeformWeights[i] = .0f;
  }
}

Mesh::Mesh() {
  primitive_type = GLx::PRIMITIVE_TOPOLOGY_UNKNOWN;
  index_count = 0;
}

int Mesh::Create(const Vertex *pVertex, size_t vStride, size_t vCount, const unsigned int *pIndex, size_t iCount,
                 GLx::PRIMITIVE_TOPOLOGY_TYPE primitiveType) {

  VAO = GLx::GLVAO::New();
  VBO = GLx::GLBuffer::New();
  EBO = GLx::GLBuffer::New();

  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);

  glEnableVertexAttribArray(0);
  glEnableVertexAttribArray(1);
  glEnableVertexAttribArray(2);
  glEnableVertexAttribArray(3);
  glEnableVertexAttribArray(4);
  glEnableVertexAttribArray(5);
  glBufferData(GL_ARRAY_BUFFER, vCount * vStride, pVertex, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vStride, (void *)offsetof(Vertex, Pos));
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vStride, (void *)offsetof(Vertex, Normal));
  glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, vStride, (void *)offsetof(Vertex, Tangent));
  glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, vStride, (void *)offsetof(Vertex, TexCoord));

  static_assert(_countof(((Vertex *)(0))->BoneIds) == 4, "Bone ids and weights must be 4!");

  glVertexAttribPointer(4, 4, GL_INT, GL_FALSE, vStride, (void *)offsetof(Vertex, BoneIds));
  glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, vStride, (void *)offsetof(Vertex, DeformWeights));

  if (primitiveType == GLx::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || primitiveType == GLx::PRIMITIVE_TOPOLOGY_LINE_LIST) {
    index_count = iCount;
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, iCount * sizeof(int), pIndex, GL_STATIC_DRAW);
  } else {
    index_count = vCount;
  }
  primitive_type = primitiveType;

  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  return 0;
}

void Mesh::SetMaterial(const Material *mat) { material = mat; }

void Mesh::Draw() {

  GLx::GLTextureRef texture;

  if (material) {
    texture = material->GetDiffuseTexture(0);
    if (texture)
      glBindTextureUnit(0, texture);
    texture = material->GetSpecularTexture(0);
    if (texture)
      glBindTextureUnit(1, texture);
    texture = material->GetNormalTexture(0);
    if (texture)
      glBindTextureUnit(2, texture);
    texture = material->GetHeightTexture(0);
    if (texture)
      glBindTextureUnit(3, texture);
  }

  glBindVertexArray(VAO);
  if (primitive_type == GLx::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST) {
    glDrawElements(GL_TRIANGLES, index_count, GL_UNSIGNED_INT, nullptr);
  } else if (primitive_type == GLx::PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP) {
    glDrawArrays(GL_TRIANGLES, 0, index_count);
  }
  glBindVertexArray(0);
}
