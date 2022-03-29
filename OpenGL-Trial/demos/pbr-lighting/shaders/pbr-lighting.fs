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

layout(binding = 0) uniform sampler2D AlbedoMap;
layout(binding = 1) uniform sampler2D NormalMap;
layout(binding = 2) uniform sampler2D MetallicMap;
layout(binding = 3) uniform sampler2D RoughnessMap;
layout(binding = 4) uniform sampler2D AoMap;

layout(binding = 5) uniform samplerCube IBLDiffuseIrradianceMap;
layout(binding = 6) uniform samplerCube IBLSpecularPrefilterredMap;
layout(binding = 7) uniform sampler2D IBLSpecularLUTMap;

in VS_OUTPUT {
  vec3 aPosW;
  vec3 aNormalW;
  vec3 aTangentW;
  vec2 aTexCoord;
} fs_in;

const float PI = 3.14159265359;

// Tower-Bridge Reitz GGX
float DistributionGGX(vec3 N, vec3 H, float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = max(dot(N, H), 0.0);
  float NdotH2 = NdotH * NdotH;

  float nom = a2;
  float denom = (NdotH2 * (a2 - 1.0) + 1.0);
  denom = PI * denom * denom;

  return nom / max(denom, 1.e-7);
}

//  Smith Schlick GGX
float GeometrySchlickGGX(float NdotV, float roughness) {
  float r = roughness + 1.0;
  float k = r * r * 0.125;
  float nom = NdotV;
  float denom = NdotV * (1.0 - k) + k;
  return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
  float NdotV = max(dot(N, V), 0.0);
  float NdotL = max(dot(N, L), 0.0);
  float ggx2 = GeometrySchlickGGX(NdotV, roughness);
  float ggx1 = GeometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(max(1.0 - cosTheta, 0.0), 5.0);
}

vec3 FresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
  return F0 + (max(F0, vec3(1.0 - roughness)) - F0) * pow(max(1.0 - cosTheta, 0.0), 5);
}

vec3 ComputeNormal(vec3 T, vec3 N, vec3 normalT) {
  vec3 B;
  T = T - dot(N, T) * N;
  T = normalize(T);
  B = cross(N, T);

  return T * normalT.x + B * normalT.y + N * normalT.z;
}

void main() {

  vec3 N = normalize(fs_in.aNormalW);
  vec3 V = normalize(EyePosW - fs_in.aPosW);
  vec3 L;
  vec3 H;
  float d2, attenuation;

  vec3 albedo = pow(texture(AlbedoMap, fs_in.aTexCoord).rgb, vec3(2.2));
  vec3 normalT = texture(NormalMap, fs_in.aTexCoord).xyz * 2.0 - vec3(1.0);
  float metallic = texture(MetallicMap, fs_in.aTexCoord).r;
  float roughness = texture(RoughnessMap, fs_in.aTexCoord).r;
  float ao = texture(AoMap, fs_in.aTexCoord).r;

  N = ComputeNormal(normalize(fs_in.aTangentW), N, normalT);

  // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
  // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)
  vec3 F0 = vec3(0.04);
  F0 = mix(F0, albedo, metallic);

  // reflectance equation
  vec3 Lo = vec3(.0);
  float NDF, G;
  vec3 F;
  vec3 kS;
  vec3 kD;
  float NdotL;
  vec3 specular;

  for(int i = 0; i < 4; ++i) {

    L = PointLights[i].Pos - fs_in.aPosW;
    d2 = dot(L, L);

    attenuation = 1.0 / d2;
    L = L / sqrt(d2);
    H = normalize(L + V);

    NDF = DistributionGGX(N, H, roughness);
    G = GeometrySmith(N, V, L, roughness);
    F = FresnelSchlick(clamp(dot(H, V), 0.0, 1.0), F0);
    specular = (NDF * G * F) / max(4.0*max(dot(N, L), 0.0)*max(dot(N, V), 0.0), 1.e-2);

    // kS is equal to Fresnel
    kS = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0(unless the surface emits light). To prevent this
    // relationship the diffuse component (kD) should equal 1.0 - kS
    kD = vec3(1.0) - kS;
    // multiply kD by the inverse matalness such that only non-metals
    // have diffuse lighting, or a linear blend if partly metal(pure metals
    // have no diffuse light)
    kD *= (1.0 - metallic);

    // scale light by NdotL
    NdotL = max(dot(N, L), 0.0);
    Lo += (kD * albedo / PI + specular) * PointLights[i].Radiance * attenuation * NdotL;
  }

  F = FresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
  kD = 1.0 - F;
  kD *= 1.0 - metallic;

  vec3 iblDiffuseI;
  vec3 iblSpecularI;
  vec2 iblSpecularBRDF;
  int prefilterredMapMipLevels = textureQueryLevels(IBLSpecularPrefilterredMap);

  iblDiffuseI = texture(IBLDiffuseIrradianceMap, N).rgb;
  iblSpecularI = textureLod(IBLSpecularPrefilterredMap, N, roughness * prefilterredMapMipLevels).rgb;
  iblSpecularBRDF = texture(IBLSpecularLUTMap, vec2(max(dot(N, V), 0.0), roughness)).rg;
  specular = iblSpecularI * (iblSpecularBRDF.x * F + iblSpecularBRDF.y);

  vec3 ambient = (iblDiffuseI * albedo * kD + specular) * ao;
  vec3 color = ambient + Lo;

  // HDR tone mapping
  color = color / (color + vec3(1.0));

  // Gammar correction
  color = pow(color, vec3(1.0/2.2));

  gl_FragColor = vec4(color, 1.0);
}

