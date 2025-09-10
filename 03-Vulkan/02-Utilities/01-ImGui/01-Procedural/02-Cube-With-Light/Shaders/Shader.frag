#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 out_diffused_light_color;

layout(location = 0) out vec4 FragColor;

void main(void)
{
    // Code
    FragColor = vec4(out_diffused_light_color, 1.0);
}
