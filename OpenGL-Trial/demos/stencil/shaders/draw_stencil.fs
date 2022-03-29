#version 460 core

uniform float GrayScale;

void main() {
  gl_FragColor = vec4(GrayScale);
}