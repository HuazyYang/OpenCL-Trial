#version 460 core

in vec2 aNdcPos;

layout(std140, binding = 0) uniform cbPerFrame {
  vec2 ProjectInvXY;
  mat4 ViewInverse;
};

layout(binding = 0) uniform samplerCube EnvMapTexture;

void main() {

  vec4 rayDir = vec4(ProjectInvXY.x * aNdcPos.x,  ProjectInvXY.y * aNdcPos.y, -1.0f, .0f);
  rayDir = ViewInverse * rayDir;

  // Cube map texture space is left hand, so mirror the rayDir
  rayDir.z *= -1.0f;

  vec3 color = texture(EnvMapTexture, rayDir.xyz).xyz;
  color = color / (color + vec3(1.0));
  color = pow(color, vec3(1.0/2.2));

  gl_FragColor = vec4(color, 1.0);
}