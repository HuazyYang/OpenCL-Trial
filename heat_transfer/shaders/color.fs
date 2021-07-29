#version 460

layout(std140, binding = 0) uniform PerSceneUB {
  mat4x4 ViewProj;
  vec2   LutMinMax;
};

layout(binding = 0) uniform sampler2D ValueGridTexture;
layout(binding = 1) uniform sampler1D LookupTableTexture;

in vec2 aTexcoord;

void main() {

  float val = texture(ValueGridTexture, aTexcoord).x;
  val = (val - LutMinMax.x) / (LutMinMax.y - LutMinMax.x);
  val = clamp(val, 0.0, 1.0);

  gl_FragColor = vec4(texture(LookupTableTexture, val).xyz, 1);
}