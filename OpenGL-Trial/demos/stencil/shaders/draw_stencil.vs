#version 460 core

layout(location=0) in vec3 aPosL;

uniform mat4 WorldViewProj;

void main() {
  gl_Position = WorldViewProj * vec4(aPosL, 1.0f);
}