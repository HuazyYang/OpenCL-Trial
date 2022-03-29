#version 460 core

#ifndef __BLEND_ANIMATION_UB
#define __BLEND_ANIMATION_UB
layout(std140, binding = 0) uniform cbPerFrame {
  mat4x4 ViewProj;
};

layout(std430, binding = 1) buffer BoneClipBuffer {
  mat4x4 FinalBoneMatrices[];
};
#endif /*__BLEND_ANIMATION_UB*/


layout(location = 0) in vec3 aPosL;
layout(location = 3) in vec2 aTexCoord;
layout(location = 4) in ivec4 aBoneIds;
layout(location = 5) in vec4 aDeformWeights;

out VS_OUTPUT {
  vec2 aTexCoord;
} vs_out;

void main() {
  vec4 posL = vec4(aPosL, 1.0);
  vec4 posW = vec4(0.0);
  int i = 0;
  int j;

  for(i = 0; i < 4; ++i) {
    if(aBoneIds[i] < 0)
      break;
    j = aBoneIds[i];
    if(j >= 20) {
      posW = posL;
      break;
    }
    posW += vec4(aDeformWeights[i]) *  (FinalBoneMatrices[j] * posL);
  }

  if(i == 0)
    posW = posL;

  vs_out.aTexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
  gl_Position = ViewProj * vec4(posW.xyz, 1.0);
}

