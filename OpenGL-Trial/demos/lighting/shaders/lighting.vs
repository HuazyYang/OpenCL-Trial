#version 460 core

layout(location = 0) in vec3 aPosL;
layout(location = 1) in vec3 aNormalL;
layout(location = 2) in vec2 aTexCoord;

out vec3 aPosW;
out vec3 aNormalW;
out vec2 aAlbedoTextureCoord;

uniform mat4 ViewProj;
uniform mat4 World;
uniform mat4 WorldInvTranspose;

void main() {

  aPosW = (World * vec4(aPosL, 1.0f)).xyz;
  aNormalW = mat3(WorldInvTranspose) * aNormalL;
  aAlbedoTextureCoord = aTexCoord;

  gl_Position = ViewProj * vec4(aPosW, 1.0f);
}
