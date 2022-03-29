#version 460 core

in vec2 aTexCoord0;

uniform sampler2D GrayScaleTexture;
uniform vec3 MaskBaseColor;

void main() {
  float scale = texture(GrayScaleTexture, aTexCoord0).r;
  if(scale < 0.001f)
    discard;
  gl_FragColor = vec4(MaskBaseColor * scale, scale);
}