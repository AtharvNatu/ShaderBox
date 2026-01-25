#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vPosition_cpu;
layout(location = 1) in vec4 vPosition_gpu;

layout(binding = 0) uniform mvpData 
{ 
    mat4 mvpMatrix;
    vec4 color;
    int useGPU;
} ubo;

void main(void)
{
    // Code
    if (ubo.useGPU == 1)
    {
        gl_Position = ubo.mvpMatrix * vPosition_gpu;
        gl_PointSize = 2.0;
    }
    else
    {
        gl_Position = ubo.mvpMatrix * vPosition_cpu;
        gl_PointSize = 2.0;
    }
}
