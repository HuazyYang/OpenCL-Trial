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

layout(triangles, fractional_even_spacing, ccw) in;

in VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} tese_in[];

out VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} tese_out;

layout(binding = 2) uniform sampler2D HeightMapTexture;

void main() {

  tese_out.aPosW = tese_in[0].aPosW * gl_TessCoord.x + tese_in[1].aPosW * gl_TessCoord.y + tese_in[2].aPosW * gl_TessCoord.z;
  tese_out.aNormalW = tese_in[0].aNormalW * gl_TessCoord.x + tese_in[1].aNormalW * gl_TessCoord.y + tese_in[2].aNormalW * gl_TessCoord.z;
  tese_out.aTangentW = tese_in[0].aTangentW * gl_TessCoord.x + tese_in[1].aTangentW * gl_TessCoord.y + tese_in[2].aTangentW * gl_TessCoord.z;
  tese_out.aTexCoord = tese_in[0].aTexCoord * gl_TessCoord.x + tese_in[1].aTexCoord * gl_TessCoord.y + tese_in[2].aTexCoord * gl_TessCoord.z;

  // may need to normalize T and N
  // tese_out.aTangentW = normalize(tese_out.aTangentW);
  // tese_out.aNormalW = normalize(tese_out.aNormalW);

  float h = (1.0f - texture(HeightMapTexture, tese_out.aTexCoord).x) * HeightMapScaleFactor;
  tese_out.aPosW = tese_out.aPosW + h * tese_out.aNormalW;

  gl_Position = ViewProj * vec4(tese_out.aPosW, 1.0f);
}

