#version 460 core

in vec3 aPosW;
in vec3 aNormalW;

uniform vec3 EyePosW;
uniform samplerCube EnvMapTexture;

void main() {

  vec3 v = normalize(aPosW - EyePosW);
  vec3 n = normalize(aNormalW);
  vec3 l = reflect(v, n);
  l.z *= -1.0f;
  gl_FragColor = vec4(texture(EnvMapTexture, l).xyz, 1.0f);
}