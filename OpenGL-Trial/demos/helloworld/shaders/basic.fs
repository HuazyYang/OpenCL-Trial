#version 460 core

in vec2 aTexCoord0;

uniform sampler2D DiffuseSampler;
uniform sampler2D DiffuseSampler2;

void main() {
    gl_FragColor = mix(texture(DiffuseSampler, aTexCoord0), texture(DiffuseSampler2, aTexCoord0), 0.2f);
}