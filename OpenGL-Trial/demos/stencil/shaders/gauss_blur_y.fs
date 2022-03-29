#version 460 core

in vec2 aTexCoord0;

uniform sampler2D AlbedoTexture;

uniform float Offsets9[3] = float[](0.0, 1.3846153846, 3.2307692308);
uniform float Weights9[3] = float[](0.2270270270, 0.3162162162, 0.0702702703);

uniform float Offsets13[4] = float[](0.0, 1.411764705882353, 3.2941176470588234, 5.176470588235294);
uniform float Weights13[4] = float[](0.1964825501511404, 0.2969069646728344, 0.09447039785044732, 0.010381362401148057);

void main() {

  float size = float(textureSize(AlbedoTexture, 0).y);

  vec4 color;
  vec2 off1 = vec2(Offsets13[1] / size, 0.0f);
  vec2 off2 = vec2(Offsets13[2] / size, 0.0f);
  vec2 off3 = vec2(Offsets13[3] / size, 0.0f);

  color = texture(AlbedoTexture, aTexCoord0) * Weights13[0];
  color += texture(AlbedoTexture, aTexCoord0 + off1) * Weights13[1];
  color += texture(AlbedoTexture, aTexCoord0 - off1) * Weights13[1];
  color += texture(AlbedoTexture, aTexCoord0 + off2) * Weights13[2];
  color += texture(AlbedoTexture, aTexCoord0 - off2) * Weights13[2];
  color += texture(AlbedoTexture, aTexCoord0 + off3) * Weights13[3];
  color += texture(AlbedoTexture, aTexCoord0 - off3) * Weights13[3];

  gl_FragColor = color;
}