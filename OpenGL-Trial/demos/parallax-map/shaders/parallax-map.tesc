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

in VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} tesc_in[];

layout(vertices = 3) out;

out VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} tesc_out[];

float GetTessLevel(float d1, float d2) {

  float avgD = (d1 + d2) * 0.5;
  avgD = clamp(avgD, MinTessDistAndFactor.x, MaxTessDistAndFactor.x);

  return ceil(mix(MinTessDistAndFactor.y, MaxTessDistAndFactor.y,
    (avgD - MinTessDistAndFactor.x) / (MaxTessDistAndFactor.x - MinTessDistAndFactor.x)));
}

void ComputeTessFactors() {

  float d0 = distance(tesc_in[0].aPosW, EyePosW);
  float d1 = distance(tesc_in[1].aPosW, EyePosW);
  float d2 = distance(tesc_in[2].aPosW, EyePosW);

  gl_TessLevelOuter[0] = GetTessLevel(d1, d2);
  gl_TessLevelOuter[1] = GetTessLevel(d2, d0);
  gl_TessLevelOuter[2] = GetTessLevel(d0, d1);
  gl_TessLevelInner[0] = gl_TessLevelOuter[0];
}

void main() {

  // Constant
  ComputeTessFactors();

  // Control Point
  tesc_out[gl_InvocationID].aPosW = tesc_in[gl_InvocationID].aPosW;
  tesc_out[gl_InvocationID].aNormalW = tesc_in[gl_InvocationID].aNormalW;
  tesc_out[gl_InvocationID].aTangentW = tesc_in[gl_InvocationID].aTangentW;
  tesc_out[gl_InvocationID].aTexCoord = tesc_in[gl_InvocationID].aTexCoord;
}



