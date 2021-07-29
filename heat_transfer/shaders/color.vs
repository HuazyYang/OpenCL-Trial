#version 460

layout(location = 0) in vec2 aPosL;
layout(location = 1) in vec2 aTexcoordIn;

out vec2 aTexcoord;

layout(std140, binding = 0) uniform PerSceneUB {
  mat4x4 ViewProj;
  vec2   LutMinMax;
};

void main() {
  aTexcoord = aTexcoordIn;
  gl_Position = ViewProj * vec4(aPosL, 0.0, 1.0);
}