#version 460 core

uniform vec3 AmbientLight;

void main() {
  gl_FragColor = vec4(AmbientLight, 1.0f);
}