#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec3 vNormal;
layout(location = 3) in vec2 vTexcoord;

layout(location = 0) out vec3 out_color;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec3 out_light_direction;
layout(location = 3) out vec3 out_camera_direction;

layout(binding = 0) uniform VertexUBO 
{ 
    // Matrices
    mat4 modelMatrix;
    mat4 viewProjectionMatrix;
    vec4 cameraPosition;
} ubo;

void main(void)
{
    // Code
    vec3 lightPosition = vec3(20.0, 30.0, -30.0);

    vec4 position = ubo.modelMatrix * vec4(vPosition, 1.0);

    out_color = vColor;
    out_normal = normalize(transpose(inverse(mat3(ubo.modelMatrix))) * vNormal);
    out_light_direction = normalize(lightPosition - position.xyz);
    out_camera_direction = normalize(vec3(ubo.cameraPosition) - position.xyz);

    gl_Position = ubo.viewProjectionMatrix * position;
}
