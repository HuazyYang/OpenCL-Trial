#version 460 core

layout(location = 0) in vec3 aPosL;
layout(location = 2) in vec2 aTexCoord;

out vec2 aTexCoord0;

uniform mat4 WorldViewProj;
uniform mat3x2 TexTransform;

void main() {

  // copy vertex attribs
  // aTexCoord0 = aTexCoord;
  aTexCoord0 = TexTransform * vec3(aTexCoord, 1.0f);
  gl_Position = WorldViewProj * vec4(aPosL, 1.0f);
}