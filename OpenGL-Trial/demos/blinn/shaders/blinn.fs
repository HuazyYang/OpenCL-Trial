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

in VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec2 aTexCoord;
} fs_in;

uniform sampler2D DiffuseTexture;

void main() {

  vec3 v = EyePosW - fs_in.aPosW;
  vec3 l = LightPosW - fs_in.aPosW;
  vec3 n = fs_in.aNormalW;
  vec3 h;

  v = normalize(v);
  n = normalize(n);

  float d = length(l);

  l /= d;
  float strength = 1.0 / dot(LightAttenuation.xyz, vec3(1.0f, d, d*d))
    * pow(max(dot(-l, LightDirW), 0.0f), LightAttenuation.w);

  h = normalize(l + v);

  vec3 albedo =  texture(DiffuseTexture, fs_in.aTexCoord).xyz;
  vec3 A, D, S;

  A = MatAmbient;
  D = MatDiffuse * max(dot(n, l), 0.0);
  S = MatSpecular.xyz * pow(max(dot(n, h), 0.0), MatSpecular.w);

  gl_FragColor = vec4( (A+D+S)*albedo, 1.0f);
}