// #version 460 core
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 vPosition;
// layout(location = 1) in vec3 vColor;
// layout(location = 2) in vec3 vNormal;
// layout(location = 3) in vec2 vTexcoord;

// layout(location = 0) out vec3 out_color;
// layout(location = 1) out vec3 out_normal;
// layout(location = 2) out vec3 out_light_direction;
// layout(location = 3) out vec3 out_camera_direction;

// layout(binding = 0) uniform VertexUBO 
// { 
//     // Matrices
//     mat4 modelMatrix;
//     mat4 viewMatrix;
//     mat4 projectionMatrix;
//     vec4 cameraPosition;
// } ubo;

// void main(void)
// {
//     // Code
//     vec4 position = ubo.modelMatrix * vec4(vPosition, 1.0);

//     out_color = vColor;
//     out_normal = normalize(transpose(inverse(mat3(ubo.modelMatrix))) * vNormal);

//     vec3 lightPosition = vec3(20.0, 30.0, -30.0);
//     out_light_direction = normalize(lightPosition - position.xyz);
//     out_camera_direction = normalize(ubo.cameraPosition.xyz - position.xyz);

//     gl_Position = ubo.projectionMatrix * ubo.viewMatrix * position;
// }


// #version 460 core
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 vPosition;
// layout(location = 1) in vec3 vColor;
// layout(location = 2) in vec3 vNormal;
// layout(location = 3) in vec2 vTexcoord;

// layout(location = 0) out vec3 frag_position_ws;
// layout(location = 1) out vec3 frag_normal_ws;
// layout(location = 2) out vec3 frag_color;
// layout(location = 3) out vec3 frag_light_dir;
// layout(location = 4) out vec3 frag_view_dir;

// layout(binding = 0) uniform VertexUBO {
//     mat4 modelMatrix;
//     mat4 viewMatrix;
//     mat4 projectionMatrix;
//     vec4 cameraPosition;   // world-space
// } ubo;

// void main(void)
// {
//     vec3 lightPosition = vec3(20.0, 30.0, -30.0);

//     // World position
//     vec4 worldPos = ubo.modelMatrix * vec4(vPosition, 1.0);

//     frag_position_ws = worldPos.xyz;
//     frag_normal_ws   = normalize(transpose(inverse(mat3(ubo.modelMatrix))) * vNormal);
//     frag_color       = vColor;
//     frag_light_dir   = normalize(lightPosition - worldPos.xyz);
//     frag_view_dir    = normalize(ubo.cameraPosition.xyz - worldPos.xyz);

//     gl_Position = ubo.projectionMatrix * ubo.viewMatrix * worldPos;
// }

#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec2 vTexcoord;

layout(location = 0) out vec3 FragPos;
layout(location = 1) out vec3 Normal;
layout(location = 2) out vec2 TexCoords;

layout(binding = 0) uniform VertexUBO 
{
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 cameraPosition;
} ubo;

void main(void)
{
    // Transform vertex position into world space
    FragPos = vec3(ubo.modelMatrix * vec4(vPosition, 1.0));

    // Transform normal to world space
    Normal = mat3(transpose(inverse(ubo.modelMatrix))) * vNormal;

    TexCoords = vTexcoord;

    // Final clip-space position
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * vec4(FragPos, 1.0);
}
