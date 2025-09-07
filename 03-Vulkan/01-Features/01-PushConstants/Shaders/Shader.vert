#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 out_color;

layout(push_constant) uniform PushModel
{
    mat4 modelMatrix;
} uModel;

layout(binding = 0) uniform vpMatrix 
{ 
    mat4 viewMatrix;
    mat4 projectionMatrix;
} uVP;

void main(void)
{
    // Code
    gl_Position = uVP.projectionMatrix * uVP.viewMatrix * uModel.modelMatrix * vPosition;
    out_color = vColor;
}
