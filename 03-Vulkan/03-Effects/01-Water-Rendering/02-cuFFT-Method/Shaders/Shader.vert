#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec2 vTexcoord;

layout(location = 0) out vec4 out_position;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_texcoord;

layout(binding = 0) uniform VertexUBO 
{ 
    // Matrices
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;

    float heightAmp;
    float choppy;
    float scale;
} ubo;

layout(binding = 2) uniform sampler2D displacementMap;
layout(binding = 3) uniform sampler2D normalMap;

void main(void)
{
    // Code
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(vPosition, 1.0);
    
    vec4 displacement = texture(displacementMap, vTexcoord * ubo.scale);
    displacement.y *= ubo.heightAmp;

    out_position.xyz = vPosition + displacement.xyz;
    out_position.w = displacement.w;

    const vec4 slope = texture(normalMap, vTexcoord * ubo.scale);
    out_normal = normalize(vec3(
        - (slope.x / (1.0f + ubo.choppy * slope.z)),
        1.0f,
        - (slope.y / (1.0f + ubo.choppy * slope.w))
    ));
    
    out_texcoord = vTexcoord;
}
