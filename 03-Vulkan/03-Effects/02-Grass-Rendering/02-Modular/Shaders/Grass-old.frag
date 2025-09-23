#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 out_texcoord;
layout(location = 1) in float colorVariation;

layout(set = 0, binding = 2) uniform sampler2D uGrassTextureSampler;

layout(location = 0) out vec4 FragColor;

void main(void)
{
    // Code
    vec4 color = texture(uGrassTextureSampler, out_texcoord);
    
    // Remove Transparent Areas
    if (color.a < 0.25) 
        discard;
    
    color.xyz = mix(color.xyz, 0.5 * color.xyz, colorVariation);
	
    FragColor = color;
}
