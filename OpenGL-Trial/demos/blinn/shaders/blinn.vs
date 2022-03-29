#version 460 core

#ifndef __BLINN_UB__
#define __BLINN_UB__
layout(std140, binding = 0) uniform ubPerFrame {
  mat4 ViewProj;
  vec3 EyePosW;

  // Lights
  vec3 LightPosW;
  vec3 LightDirW;
  vec4 LightAttenuation;
};

layout(std140, binding = 1) uniform ubPerObject {
  mat4 World;
  mat4 WorldInvTranspose;
  mat4 TexTransform;

  // Material
  vec3 MatAmbient;
  vec3 MatDiffuse;
  vec4 MatSpecular;
};
#endif

layout(location = 0) in vec3 aPosL;
layout(location = 1) in vec3 aNormalL;
layout(location = 2) in vec2 aTexCoord;

out VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec2 aTexCoord;
} vs_out;

void main() {

  vec4 posW = World * vec4(aPosL, 1.0);

  vs_out.aPosW = posW.xyz;
  gl_Position = ViewProj * posW;
  vs_out.aNormalW = (WorldInvTranspose * vec4(aNormalL, 0.0)).xyz;
  vs_out.aTexCoord = (TexTransform * vec4(aTexCoord, 0.0, 1.0f)).xy;
}
