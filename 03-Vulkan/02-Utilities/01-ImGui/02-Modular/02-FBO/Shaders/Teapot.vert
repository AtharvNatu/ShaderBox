#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 vPosition;
layout(location = 1) in vec3 vNormals;
layout(location = 2) in vec2 vTexcoords;

layout(location = 0) out vec2 out_texcoords;
layout(location = 1) out vec3 out_transformedNormals;
layout(location = 2) out vec3 out_lightDirection;
layout(location = 3) out vec3 out_viewerVector;

layout(binding = 0) uniform uniformData 
{
    // Matrices Related Uniforms
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;

    // Light Related Uniforms
    vec4 lightAmbient;
    vec4 lightDiffuse;
    vec4 lightSpecular;
    vec4 lightPosition;

    // Material Related Uniforms
    vec4 materialAmbient;
    vec4 materialDiffuse;
    vec4 materialSpecular;
    float materialShininess;

    // Texture and Light
    int bTextureEnabled;
    int bLightEnabled;
    
} ubo;

void main(void)
{
    // Code
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vPosition;

    vec4 eyeCoordinates = ubo.viewMatrix * ubo.modelMatrix * vPosition;
    mat3 normalMatrix = mat3(ubo.viewMatrix * ubo.modelMatrix);
    out_transformedNormals = normalMatrix * vNormals;
    out_lightDirection = vec3(ubo.lightPosition - eyeCoordinates);
    out_viewerVector = -eyeCoordinates.xyz;
    out_texcoords = vTexcoords;
}
