#include "skm_model.hpp"
#include "CommonUtils.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <iostream>
#include <filesystem>
#include "assimp_glm_helpers.hpp"

namespace fs = std::filesystem;

static int LoadTextureFromFile(const char *path, const std::string &directory, bool gamma,
                        GLx::GLTexture &texture);

Model::Model() {
  gamma_correction = false;
  next_bone_index = 0;
}

int Model::Load(const char *path) {
  loadModel(path);
  return 0;
}

void Model::Draw() {

  for(auto &mesh : meshes) {
    mesh.Draw();
  }
}

std::map<std::string, BoneInfo> &Model::GetOffsetMatMap() { return bone_infos; }
size_t Model::GetBoneCount() const { return bone_infos.size(); }

void Model::loadModel(const char *path) {
  // read file via ASSIMP
  Assimp::Importer importer;
  const aiScene *scene =
      importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals |
                                  aiProcess_FlipUVs | aiProcess_CalcTangentSpace|aiProcess_GenBoundingBoxes);
  // check for errors
  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) // if is Not Zero
  {
    std::cout << "ERROR::ASSIMP:: " << importer.GetErrorString() << std::endl;
    return;
  }
  // retrieve the directory path of the filepath
  fs::path path2(path);
  directory = path2.parent_path().string();

  material_dicts.resize(scene->mNumMaterials);

  // process ASSIMP's root node recursively
  processNode(scene->mRootNode, scene);
}

void Model::processNode(aiNode *node, const aiScene *scene) {
  // process each mesh located at the current node
  meshes.reserve(node->mNumMeshes);
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    // the node object only contains indices to index the actual objects in the scene.
    // the scene contains all the data, node is just to keep stuff organized (like relations between
    // nodes).
    aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    meshes.push_back(processMesh(mesh, scene));
  }
  // after we've processed all of the meshes (if any) we then recursively process each of the
  // children nodes
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    processNode(node->mChildren[i], scene);
  }
}

Mesh Model::processMesh(aiMesh *mesh, const aiScene *scene) {
  // data to fill
  std::vector<Vertex> vertices;
  std::vector<unsigned int> indices;

  // walk through each of the mesh's vertices
  vertices.reserve(mesh->mNumVertices);
  indices.reserve(size_t(mesh->mNumFaces * 1.5));
  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    Vertex vertex;
    glm::vec3 vector; // we declare a placeholder vector since assimp uses its own vector class that
                      // doesn't directly convert to glm's vec3 class so we transfer the data to
                      // this placeholder glm::vec3 first.
    // positions
    vector.x = mesh->mVertices[i].x;
    vector.y = mesh->mVertices[i].y;
    vector.z = mesh->mVertices[i].z;
    vertex.Pos = vector;
    // normals
    if (mesh->HasNormals()) {
      vector.x = mesh->mNormals[i].x;
      vector.y = mesh->mNormals[i].y;
      vector.z = mesh->mNormals[i].z;
      vertex.Normal = vector;
    }
    if(mesh->HasTangentsAndBitangents()) {
      // tangent
      vector.x = mesh->mTangents[i].x;
      vector.y = mesh->mTangents[i].y;
      vector.z = mesh->mTangents[i].z;
      vertex.Tangent = vector;
    }

    // texture coordinates
    if (mesh->mTextureCoords[0]) // does the mesh contain texture coordinates?
    {
      glm::vec2 vec;
      // a vertex can contain up to 8 different texture coordinates. We thus make the assumption
      // that we won't use models where a vertex can have multiple texture coordinates so we always
      // take the first set (0).
      vec.x = mesh->mTextureCoords[0][i].x;
      vec.y = mesh->mTextureCoords[0][i].y;
      vertex.TexCoord = vec;
    } else
      vertex.TexCoord = glm::vec2(0.0f, 0.0f);

    vertices.push_back(vertex);
  }

  if(mesh->HasBones()) {
    ExtractBoneWeightsForVertices(vertices, mesh, scene);
  }

  // now wak through each of the mesh's faces (a face is a mesh its triangle) and retrieve the
  // corresponding vertex indices.
  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    aiFace face = mesh->mFaces[i];
    // retrieve all indices of the face and store them in the indices vector
    for (unsigned int j = 0; j < face.mNumIndices; j++)
      indices.push_back(face.mIndices[j]);
  }
  // process materials
  aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];

  auto material2 = &material_dicts[mesh->mMaterialIndex];
  if(material2->GetIndex() < 0)
    *material2 = loadMaterialTextures(material, mesh->mMaterialIndex);

  // return a mesh object created from the extracted mesh data
  Mesh mesh2;
  mesh2.Create(vertices.data(), sizeof(Vertex), vertices.size(), indices.data(), indices.size(),
               GLx::PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  mesh2.SetMaterial(material2);

  return std::move(mesh2);
}

void Model::ExtractBoneWeightsForVertices(std::vector<Vertex> &vertices, aiMesh *mesh, const aiScene *scene) {

  for(unsigned boneIndex = 0; boneIndex < mesh->mNumBones; ++boneIndex) {

    auto boneInfoIter =  bone_infos.find(mesh->mBones[boneIndex]->mName.C_Str());
    int boneId = -1;
    if (boneInfoIter == bone_infos.end()) {
      BoneInfo boneInfo = {next_bone_index++, AssimpGLMHelpers::aiMatrix4x4ToGLMmat4(mesh->mBones[boneIndex]->mOffsetMatrix)};
      boneId = boneInfo.id;
      bone_infos.insert({std::string(mesh->mBones[boneIndex]->mName.C_Str()), std::move(boneInfo)});
    } else {
      boneId = boneInfoIter->second.id;
    }

    auto &weights = mesh->mBones[boneIndex]->mWeights;
    unsigned numWeights = mesh->mBones[boneIndex]->mNumWeights;
    for(unsigned weightIndex = 0; weightIndex < numWeights; ++weightIndex) {
      int vertIndex = weights[weightIndex].mVertexId;
      float weight = weights[weightIndex].mWeight;

      GTX_ASSERT(vertIndex < vertices.size() && "vertex index out of range");
      SetVertexBoneData(&vertices[vertIndex], boneId, weight);
    }
  }
}

void Model::SetVertexBoneData(Vertex *pVertex, int boneID, float weight) {
  for(int i = 0; i < MAX_BONE_WEIGHT_COUNT; ++i) {
    if(pVertex->BoneIds[i] < 0) {
      pVertex->BoneIds[i] = boneID;
      pVertex->DeformWeights[i] = weight;
      break;
    }
  }
}

Material Model::loadMaterialTextures(aiMaterial *mat, int matIndex) {
  Material mat2{matIndex};
  std::vector<GLx::GLTextureRef> textures;

  aiTextureType types[] = {aiTextureType_DIFFUSE, aiTextureType_SPECULAR,
                           aiTextureType_HEIGHT, aiTextureType_AMBIENT};

  for (auto type : types) {
    unsigned int texCount = mat->GetTextureCount(type);
    textures.clear();

    for (unsigned int i = 0; i < texCount; i++) {
      aiString str;
      mat->GetTexture(type, i, &str);
      // check if texture was loaded before and if so, continue to next iteration: skip loading a
      // new texture
      bool skip = false;

      for (unsigned int j = 0; j < texture_dicts.size(); ++j) {
        if (std::strcmp(texture_dicts[j].first.c_str(), str.C_Str()) == 0) {
          textures.push_back(texture_dicts[j].second);
          skip = true;
          break;
        }
      }

      if (!skip) {
        GLx::GLTexture texture;
        if (!LoadTextureFromFile(str.C_Str(), directory, gamma_correction, texture)) {
          textures.push_back(texture);
          texture_dicts.push_back({str.C_Str(), std::move(texture)});
        } else {
          textures.push_back({});
        }
      }
    }

    switch (type) {
    case aiTextureType_DIFFUSE:
      mat2.SetDiffuseTextures(textures.data(), textures.size());
      break;
    case aiTextureType_SPECULAR:
      mat2.SetSpecularTextures(textures.data(), textures.size());
      break;
    case aiTextureType_NORMALS:
      mat2.SetNormalTextures(textures.data(), textures.size());
      break;
    case aiTextureType_HEIGHT:
      mat2.SetHeightTextures(textures.data(), textures.size());
      break;
    }
  }

  return std::move(mat2);
}

int LoadTextureFromFile(const char *path, const std::string &directory, bool gamma,
                        GLx::GLTexture &texture) {
  std::string filename(path);
  filename = directory + '/' + filename;

  texture = GLx::GLTexture::New();

  int width, height, nrComponents;

  unsigned char *data = stbi_load(filename.c_str(), &width, &height, &nrComponents, 0);
  if (!data) {
    std::cout << "ERROR: can not load image file: " << filename << std::endl;
    return -1;
  }

  GLenum format;
  if (nrComponents == 1)
    format = GL_RED;
  else if (nrComponents == 3)
    format = GL_RGB;
  else if (nrComponents == 4)
    format = GL_RGBA;

  glBindTexture(GL_TEXTURE_2D, texture);
  glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
  stbi_image_free(data);
  glGenerateMipmap(GL_TEXTURE_2D);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glBindTexture(GL_TEXTURE_2D, 0);

  return 0;
}