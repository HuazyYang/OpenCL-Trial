#version 460 core

layout(binding = 0) uniform sampler2D DiffuseTexture;

in VS_OUTPUT {
  vec2 aTexCoord;
} fs_in;

void main() {
  gl_FragColor = vec4(texture(DiffuseTexture, fs_in.aTexCoord).xyz, 1.0);
}