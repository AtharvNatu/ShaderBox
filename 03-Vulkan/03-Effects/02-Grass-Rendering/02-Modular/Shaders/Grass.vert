#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 instancePosition;

void main(void)
{
    // Code
    gl_Position = vec4(instancePosition, 1.0);
}
