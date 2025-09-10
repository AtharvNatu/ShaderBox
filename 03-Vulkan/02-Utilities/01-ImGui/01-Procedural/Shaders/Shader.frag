#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 out_texcoord;

layout(binding = 1) uniform sampler2D utextureSampler;

layout(location = 0) out vec4 FragColor;

void main(void)
{
    // Code
    FragColor = texture(utextureSampler, out_texcoord);
}
