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

  SpotLight Light0;
};

layout(std140, binding = 1) uniform ubPerObject {
  mat4 World;
  mat4 WorldInvTranspose;
  mat4 TexTransform;

  Material Mat0;
};
#endif

in VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} fs_in;

layout(binding = 0) uniform sampler2D DiffuseTexture;
layout(binding = 1) uniform sampler2D NormalTexture;

vec3 ComputeWarppedNormal(vec3 N, in vec3 T, in vec3 normT) {

  T = normalize(T - N * dot(T, N));
  vec3 B = cross(N, T);

  return normT.x * T + normT.y * B + normT.z * N;
}

void main() {

  vec3 v = EyePosW - fs_in.aPosW;
  vec3 l = Light0.LightPosW - fs_in.aPosW;
  vec3 n;
  vec3 tangent;
  vec3 h;
  vec3 normT;

  // Normal mapping
  n = normalize(fs_in.aNormalW);
  tangent = normalize(fs_in.aTangentW);

  normT = texture(NormalTexture, fs_in.aTexCoord).xyz;
  normT = normT*2.0f - 1.0f;
  n = ComputeWarppedNormal(n, tangent, normT);
  n = normalize(n);

  // Blinn-Phony
  v = normalize(v);

  float d = length(l);

  l /= d;
  float strength = 1.0 / dot(Light0.LightAttenuation.xyz, vec3(1.0f, d, d*d))
    * pow(max(dot(-l, Light0.LightDirW), 0.0f), Light0.LightAttenuation.w);

  h = normalize(l + v);

  vec3 albedo =  texture(DiffuseTexture, fs_in.aTexCoord).xyz;
  vec3 A, D, S;

  A = Mat0.MatAmbient;
  D = Mat0.MatDiffuse * max(dot(n, l), 0.0);
  S = Mat0.MatSpecular.xyz * pow(max(dot(n, h), 0.0), Mat0.MatSpecular.w);

  gl_FragColor = vec4( (A+D+S)*albedo, 1.0f);
}