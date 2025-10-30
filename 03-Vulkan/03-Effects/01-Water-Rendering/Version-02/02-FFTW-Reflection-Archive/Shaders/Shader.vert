#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vNormal;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;

layout(binding = 0) uniform VertexUBO
{
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
} ubo;

void main()
{
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(vPosition, 1.0f);
    
    out_position = vec3(ubo.modelMatrix * vec4(vPosition, 1.0));
    // out_normal = mat3(transpose(inverse(ubo.modelMatrix))) * vNormal;
    out_normal = normalize(mat3(transpose(inverse(ubo.modelMatrix))) * vNormal);
}


// #version 460
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 vPosition;
// layout(location = 1) in vec3 vNormal;

// layout(location = 0) out vec3 out_position_ws; // world space position
// layout(location = 1) out vec3 out_normal_ws;   // world space normal

// layout(binding = 0) uniform VertexUBO
// {
//     mat4 modelMatrix;
//     mat4 viewMatrix;
//     mat4 projectionMatrix;
// } ubo;

// void main()
// {
//     vec4 worldPos = ubo.modelMatrix * vec4(vPosition, 1.0);
//     gl_Position = ubo.projectionMatrix * ubo.viewMatrix * worldPos;

//     out_position_ws = worldPos.xyz;

//     // normal matrix = inverse(transpose(model))
//     out_normal_ws = normalize(mat3(transpose(inverse(ubo.modelMatrix))) * vNormal);
// }
