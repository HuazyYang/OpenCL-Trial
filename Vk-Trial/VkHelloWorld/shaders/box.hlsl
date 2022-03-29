
struct VSInput {
  [[vk::location(0)]]float3 posL : POSITION;
  [[vk::location(1)]]float2 texC : TEXCOORD;
};

struct VSOutput {
  float4 posH : SV_POSITION;
  [[vk::location(0)]]float2 texC : TEXCOORD;
};

cbuffer cbPerframe: register(b0,space0) {
  float4x4 g_matWorldViewProj;
  float4x4 g_matTexTransform;
};

VSOutput VSMain(VSInput vin) {
  VSOutput vout;
  vout.posH = mul(g_matWorldViewProj, float4(vin.posL, 1.0f));

  vout.texC = mul(g_matTexTransform, float4(vin.texC, 0.0, 1.0)).xy;

  return vout;
}



Texture2D g_txDiffuseMap: register(t0, space1);
Texture2D g_txMaskDiffuseMap: register(t1, space1);
SamplerState g_samLinear: register(s2, space1);
SamplerState g_samNearest: register(s3, space1);

interface IBaseColor {
  float4 getColor(float2 uv);
  float4 getMaskColor(float2 uv);
};

class UnderlineColor: IBaseColor {
  float4 getColor(float2 uv) {
    return g_txDiffuseMap.Sample(g_samLinear, uv);
  }

  float4 getMaskColor(float2 uv) {
    return 1.0.xxxx;
  }
  IBaseColor Next() {
    UnderlineColor color;
    return color;
  }
};

float4 PSMain(float2 texC: TEXCOORD): SV_TARGET {


  class UnderlineColorMask: UnderlineColor {
    float4 getMaskColor(float2 uv) {
      return g_txMaskDiffuseMap.Sample(g_samNearest, uv);
    }

    IBaseColor Next() {
      UnderlineColorMask color;
      return color;
    }
  } color_impl;

  float4 color = color_impl.getColor(texC) * color_impl.getMaskColor(texC);

  // float4 color = g_txDiffuseMap.Sample(g_samLinear, texC) * g_txMaskDiffuseMap.Sample(g_samNearest, texC);
  color.a = 1.0;

  return color;
}