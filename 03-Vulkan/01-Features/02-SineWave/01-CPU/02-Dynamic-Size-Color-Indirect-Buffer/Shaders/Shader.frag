#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform mvpData 
{ 
    mat4 mvpMatrix;
    vec4 color;
} ubo;

void main(void)
{
    // Code
    FragColor = ubo.color;
}
