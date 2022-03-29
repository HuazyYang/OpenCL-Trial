#version 460 core

#ifndef __MODEL_LOAD_UBO
#define __MODEL_LOAD_UBO
layout(std140, binding = 0) uniform ubPerframe {
  mat4x4 ViewProj;
};
#endif

layout(location = 0) in vec3 vPosL;
layout(location = 1) in vec3 vNormalL;
layout(location = 2) in vec3 vTangent;
layout(location = 3) in vec2 vTexCoord;

out VS_OUTPUT {
  vec3 vPosW;
  vec3 vNormalW;
  vec3 vTangentW;
  vec2 vTexCoord;
} vs_out;

void main() {

  vs_out.vPosW = vPosL;
  vs_out.vNormalW = vNormalL;
  vs_out.vTangentW = vTangent;
  vs_out.vTexCoord = vec2(vTexCoord.x, 1.0f - vTexCoord.y);

  gl_Position = ViewProj * vec4(vPosL, 1.0);
}