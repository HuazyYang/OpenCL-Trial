
#version 460 core

layout(binding = 0) uniform sampler2D DiffuseTex;
layout(binding = 2) uniform sampler2D NormalTex;
layout(binding = 3) uniform sampler2D HeightTex;

in VS_OUTPUT {
  vec3 vPosW;
  vec3 vNormalW;
  vec3 vTangentW;
  vec2 vTexCoord;
} fs_in;

void main() {

  vec3 n = normalize(fs_in.vNormalW);
  vec3 

  gl_FragColor = vec4(texture(DiffuseTex, fs_in.vTexCoord).xyz, 1.0);
}