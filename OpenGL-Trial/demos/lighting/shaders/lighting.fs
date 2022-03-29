#version 460 core

// light source models

// directional light
struct DirectionalLight {
  vec3 direction;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;
};

// point light
struct PointLight {
  vec3 posW;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;

  vec3 attenuation;
};

// spot light
struct SpotLight {
  vec3 posW;
  vec4 direction;
  vec3 ambient;
  vec3 diffuse;
  vec3 specular;

  vec3 attenuation;
};

struct Material {
  vec3 ambient;
  vec3 diffuse;
  vec4 specular;
};

in vec3 aPosW;
in vec3 aNormalW;
in vec2 aAlbedoTextureCoord;

uniform vec3 EyePosW;

#define NUMBER_OF_POINT_LIGHTS  4

uniform DirectionalLight DirLight0;
uniform PointLight PointLights[NUMBER_OF_POINT_LIGHTS];
uniform SpotLight SpotLight0;
uniform Material Mat;

uniform sampler2D DiffuseTexture;
uniform sampler2D SpecularTexture;

void computeDirectionalLight(
  in DirectionalLight L,
  in Material mat,
  in vec3 n,
  in vec3 v,
  out vec3 A, out vec3 D, out vec3 S) {
  A = L.ambient * mat.ambient;
  D = max(dot(-L.direction, n), 0.0f) * L.diffuse * mat.diffuse;
  S = L.specular * mat.specular.xyz * pow(max(dot(reflect(L.direction, n), v), 0.0f), mat.specular.w);
}

void computePointLight(
  in PointLight L,
  in Material mat,
  in vec3 n,
  in vec3 v,
  out vec3 A, out vec3 D, out vec3 S) {
  vec3 l = L.posW - aPosW;
  float d = length(l);
  float intensity = 1.0f / dot(L.attenuation, vec3(1.0f, d, d*d));

  if(intensity > 0.001f) {
    l /= d;
    A = L.ambient * mat.ambient;
    D = max(dot(l, n), 0.0f) * L.diffuse * mat.diffuse;
    S = L.specular * mat.specular.xyz * pow(max(dot(reflect(-l, n), v), 0.0f), mat.specular.w);
  } else
    A = D = S = vec3(0.0f);
}

void computeSpotLight(
  in SpotLight L,
  in Material mat,
  in vec3 n,
  in vec3 v,
  out vec3 A, out vec3 D, out vec3 S) {
  vec3 l = L.posW - aPosW;
  float d = length(l);
  float intensity = 1.0f / dot(L.attenuation, vec3(1.0f, d, d*d));

  l /= d;
  intensity *= pow(max(dot(-l, L.direction.xyz), 0.0f), L.direction.w);

  if(intensity > 0.001f) {
    A = L.ambient * mat.ambient * intensity;
    D = max(dot(l, n), 0.0f) * L.diffuse * mat.diffuse * intensity;
    S = L.specular * mat.specular.xyz * pow(max(dot(reflect(-l, n), v), 0.0f), mat.specular.w) * intensity;
  } else
    A = D = S = vec3(0.0f);
}

void main() {
  vec3 n, v;
  vec3 A, D, S;
  vec3 ambientSum = vec3(0.0f),
       diffuseSum = vec3(0.0f),
       specularSum = vec3(0.0f);
  vec3 diff, spec;

  n = normalize(aNormalW);
  v = normalize(EyePosW - aPosW);

  computeDirectionalLight(DirLight0, Mat, n, v, A, D, S);
  ambientSum += A;
  diffuseSum += D;
  specularSum += S;

  for(int i = 0; i < NUMBER_OF_POINT_LIGHTS; ++i) {
    computePointLight(PointLights[i], Mat, n, v, A, D, S);
    ambientSum += A;
    diffuseSum += D;
    specularSum += S;
  }

  computeSpotLight(SpotLight0, Mat, n, v, A, D, S);
  ambientSum += A;
  diffuseSum += D;
  specularSum += S;

  diff = texture(DiffuseTexture, aAlbedoTextureCoord).xyz;
  spec = texture(SpecularTexture, aAlbedoTextureCoord).xyz;

  gl_FragColor = vec4(diff * (ambientSum + diffuseSum) + spec * specularSum, 1.0f);
}