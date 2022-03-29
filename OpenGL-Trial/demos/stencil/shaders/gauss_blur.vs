#version 460 core

out vec2 aTexCoord0;

void main() {
  int id = gl_VertexID;
  int x = id % 2;
  int y = id / 2;

  aTexCoord0 = vec2(float(x), float(y));

  gl_Position = vec4(float((x << 1) - 1), float((y << 1) - 1), 0.0f, 1.0f); 
}