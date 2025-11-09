#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vPosition;
layout(location = 1) in float vHeight;
layout(location = 2) in vec2 vSlope;

layout(location = 0) out vec3 out_eyeSpacePosition;
layout(location = 1) out vec3 out_eyeSpaceNormal;
layout(location = 2) out vec3 out_worldSpaceNormal;

layout(binding = 0) uniform VertexUBO
{
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
} ubo;

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
    float height = vHeight;
    vec2 slope = vSlope;

    vec3 normal = normalize(
        cross(vec3(
            0.0, 
            slope.y * oceanUbo.heightScale, 
            2.0 / oceanUbo.size.x
        ), 
        vec3(
            2.0 / oceanUbo.size.y, 
            slope.x * oceanUbo.heightScale, 
            0.0
        )));
    
    out_worldSpaceNormal = normal;

    vec4 pos = vec4(vPosition.x, height * oceanUbo.heightScale, vPosition.z, 1.0);
    out_eyeSpacePosition = (ubo.viewMatrix * ubo.modelMatrix * pos).xyz;

    out_eyeSpaceNormal = (mat3(transpose(inverse(ubo.viewMatrix * ubo.modelMatrix))) * normal).xyz;

    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * pos;
    
}

