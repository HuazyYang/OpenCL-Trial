#version 460 core

out vec2 aNdcPos;

void main() {
  int x = ((gl_VertexID % 2) << 1) - 1;
  int y = ((gl_VertexID / 2) << 1) - 1;
  gl_Position = vec4(float(x), float(y), 1.0f, 1.0f);
  aNdcPos = gl_Position.xy;
}