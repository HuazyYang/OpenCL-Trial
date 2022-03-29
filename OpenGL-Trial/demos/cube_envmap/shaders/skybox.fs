#version 460 core

in vec2 aNdcPos;

uniform vec2 ProjectInvXY;
uniform mat4 ViewInverse;

uniform samplerCube EnvMapTexture;

void main() {

  vec4 rayDir = vec4(ProjectInvXY.x * aNdcPos.x,  ProjectInvXY.y * aNdcPos.y, -1.0f, .0f);
  rayDir = ViewInverse * rayDir;

  // Cube map texture space is left hand, so mirror the rayDir
  rayDir.z *= -1.0f;

  gl_FragColor = texture(EnvMapTexture, rayDir.xyz);
}