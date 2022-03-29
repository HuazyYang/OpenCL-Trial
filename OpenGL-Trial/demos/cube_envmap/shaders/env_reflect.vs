#version 460 core

layout(location = 0)in vec3 aPosL;
layout(location = 1)in vec3 aNormalL;

out vec3 aPosW;
out vec3 aNormalW;

uniform mat4 World;
uniform mat4 WorldInvTranspose;
uniform mat4 ViewProj;

void main() {
  vec4 posW = World * vec4(aPosL, 1.0f);
  gl_Position = ViewProj * posW;
  aPosW = posW.xyz;

  aNormalW = (WorldInvTranspose * vec4(aNormalL, 0.0f)).xyz;
}