#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

out vec3 aAlbedoColor0;
out vec2 aTexCoord0;

uniform mat4 WorldViewProj;

void main() {
  gl_Position = WorldViewProj * vec4(aPos, 1.0f);
  aTexCoord0 = aTexCoord;
}