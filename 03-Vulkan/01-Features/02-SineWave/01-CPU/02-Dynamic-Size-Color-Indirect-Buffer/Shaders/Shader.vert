#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vPosition;

layout(binding = 0) uniform mvpData 
{ 
    mat4 mvpMatrix;
    vec4 color;
} ubo;

void main(void)
{
    // Code
    gl_Position = ubo.mvpMatrix * vPosition;
    gl_PointSize = 2.0;
}
