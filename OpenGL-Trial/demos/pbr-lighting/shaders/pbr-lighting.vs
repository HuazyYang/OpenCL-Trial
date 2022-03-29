#version 460 core

#ifndef __PBR_LIGHTING_UB__
#define __PBR_LIGHTING_UB__

struct PointLight {
  vec3 Pos;
  float padding0;
  vec3 Radiance;
  float padding1;
};

layout(std140, binding = 0) uniform cbPerFrame {
  mat4 ViewProj;
  vec3 EyePosW;
  float padding0;
  PointLight PointLights[4];
};

layout(std140, binding = 1) uniform cbPerObject {
  mat4 World;
  mat4 WorldInvTransform;
  mat4 TexTransform;
};

#endif /*__PBR_LIGHTING_UB__*/

layout(location = 0) in vec3 aPosL;
layout(location = 1) in vec3 aNormalL;
layout(location = 2) in vec3 aTangentU;
layout(location = 3) in vec2 aTexCoord;

out VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} vs_out;

void main() {

  vec4 posW = World * vec4(aPosL, 1.0f);

  vs_out.aPosW = posW.xyz;
  vs_out.aNormalW = (WorldInvTransform * vec4(aNormalL, 0.0)).xyz;
  vs_out.aTangentW = (World * vec4(aTangentU, 0.0)).xyz;
  vs_out.aTexCoord = (TexTransform * vec4(aTexCoord, 0.0, 1.0)).xy;

  gl_Position = ViewProj * posW;
}