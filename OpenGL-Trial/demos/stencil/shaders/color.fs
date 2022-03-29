#version 460 core

in vec2 aTexCoord0;

uniform sampler2D DiffuseTexture;

void main() {
  gl_FragColor = texture(DiffuseTexture, aTexCoord0);
}