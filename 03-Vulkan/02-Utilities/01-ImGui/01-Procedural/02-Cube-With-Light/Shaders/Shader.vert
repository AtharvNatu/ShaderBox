#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec3 vNormal;

layout(location = 0) out vec3 out_diffused_light_color;

layout(binding = 0) uniform ubo 
{
    // Matrices Related Uniforms
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;

    // Light Related Uniforms
    vec4 lightDiffuse;
    vec4 lightPosition;
    vec4 materialDiffuse;

    // Key Press Related Uniform
    uint keyPressed;
} uniformData;

void main(void)
{
    // Code
    gl_Position = uniformData.projectionMatrix * uniformData.viewMatrix * uniformData.modelMatrix * vPosition;

    if (uniformData.keyPressed == 1)
    {
        vec4 eyeCoordinates = uniformData.viewMatrix * uniformData.modelMatrix * vPosition;
        mat3 normalMatrix = mat3(transpose(inverse(uniformData.viewMatrix * uniformData.modelMatrix)));
        vec3 transformedNormals = normalize(normalMatrix * vNormal);
        vec3 lightDirection = normalize(vec3(uniformData.lightPosition - eyeCoordinates));
        out_diffused_light_color = vec3(uniformData.lightDiffuse) * vec3(uniformData.materialDiffuse) * max(dot(lightDirection, transformedNormals), 0.0);
    }
    else
    {
        out_diffused_light_color = vec3(1.0, 1.0, 1.0);
    }

}