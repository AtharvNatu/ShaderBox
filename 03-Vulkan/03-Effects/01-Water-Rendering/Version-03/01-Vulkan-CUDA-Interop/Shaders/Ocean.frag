#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 out_eyeSpacePosition;
layout(location = 1) in vec3 out_eyeSpaceNormal;
layout(location = 2) in vec3 out_worldSpaceNormal;

layout(location = 0) out vec4 FragColor;

layout(binding = 1) uniform OceanUBO
{
    vec4 deepColor;
    vec4 shallowColor;
    vec4 skyColor;
    vec4 lightDirection;

    float heightScale;
    float choppiness;
    vec2 size;
} oceanUbo;

void main()
{
    vec3 eyeVector = normalize(out_eyeSpacePosition);
    vec3 eyeSpaceNormalVector = normalize(out_eyeSpaceNormal);
    vec3 worldSpaceNormalVector = normalize(out_worldSpaceNormal);

    float facing = max(0.0, dot(eyeSpaceNormalVector, -eyeVector));
    float fresnel = pow(1.0 - facing, 5.0);
    float diffuse = max(0.0, dot(worldSpaceNormalVector, oceanUbo.lightDirection.xyz));

    vec4 waterColor = oceanUbo.deepColor;

    FragColor = waterColor * diffuse + oceanUbo.skyColor * fresnel;
}

