#version 460 core

#ifndef __NORMALMAP_UB__
#define __NORMALMAP_UB__
struct SpotLight {
  // Lights
  vec3 LightPosW;
  float padding0;
  vec3 LightDirW;
  float padding1;
  vec4 LightAttenuation;
};

struct Material {
  // Material
  vec3 MatAmbient;
  float padding0;
  vec3 MatDiffuse;
  float padding1;
  vec4 MatSpecular;
};

layout(std140, binding = 0) uniform ubPerFrame {
  mat4 ViewProj;
  vec3 EyePosW;

  vec2 MinTessDistAndFactor;
  vec2 MaxTessDistAndFactor;
  float HeightMapScaleFactor;
  float paddding0;

  SpotLight Light0;
};

layout(std140, binding = 1) uniform ubPerObject {
  mat4 World;
  mat4 WorldInvTranspose;
  mat4 TexTransform;

  Material Mat0;
};
#endif

layout(location = 0) in vec3 aPosL;
layout(location = 1) in vec3 aNormalL;
layout(location = 2) in vec3 aTangentL;
layout(location = 3) in vec2 aTexCoord;

out VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} vs_out;

void main() {

  vec4 posW = World * vec4(aPosL, 1.0);

  vs_out.aPosW = posW.xyz;
  // gl_Position = ViewProj * posW;
  vs_out.aNormalW = (WorldInvTranspose * vec4(aNormalL, 0.0)).xyz;
  vs_out.aTexCoord = (TexTransform * vec4(aTexCoord, 0.0, 1.0f)).xy;
  vs_out.aTangentW = (World * vec4(aTangentL, 0.0)).xyz;
}
