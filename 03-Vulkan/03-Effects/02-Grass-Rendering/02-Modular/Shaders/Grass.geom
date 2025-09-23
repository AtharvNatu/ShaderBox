#version 460 core
layout(points) in;
layout(triangle_strip, max_vertices = 12) out;

layout(location = 0) in vec3 vs_basePosition[];
layout(location = 1) in mat4 vs_modelWindMatrix[];
layout(location = 5) in float vs_randomYAngle[];
layout(location = 6) in float vs_colorVariation[];

layout(location = 0) out vec2 out_texcoord;
layout(location = 1) out float colorVariation;

layout(set = 0, binding = 0) uniform UBO 
{ 
    mat4 modelMatrix;
    mat4 projectionMatrix;
    float time;    
    float windStrength;
} ubo;

#define PI 3.14159265359

float grassSize = 0.5; // Or make it a UBO uniform

mat4 rotationY(float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
       -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

void createQuad(vec3 basePos, mat4 modelWind, float randomYAngle, mat4 crossModel, float size, float variation)
{
    vec4 vertices[4] = vec4[](
        vec4(-0.25, 0.0, 0.0, 1.0),
        vec4( 0.25, 0.0, 0.0, 1.0),
        vec4(-0.25, 0.5, 0.0, 1.0),
        vec4( 0.25, 0.5, 0.0, 1.0)
    );

    vec2 texcoords[4] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0)
    );

    mat4 randomYRot = rotationY(randomYAngle);

    for (int i = 0; i < 4; i++)
    {
        vec4 localOffset = vertices[i] * size;
        vec4 worldPos = ubo.modelMatrix * vec4(basePos, 1.0);  
        vec4 finalPos = worldPos + modelWind * randomYRot * crossModel * localOffset;

        gl_Position = ubo.projectionMatrix * finalPos;
        out_texcoord = texcoords[i];
        colorVariation = variation;

        EmitVertex();
    }
    EndPrimitive();
}

void main()
{
    mat4 model0   = mat4(1.0);
    mat4 model45  = rotationY(radians(45.0));
    mat4 modelm45 = rotationY(radians(-45.0));

    vec3 basePos          = vs_basePosition[0];
    mat4 modelWind        = vs_modelWindMatrix[0];
    float randomYAngle    = vs_randomYAngle[0];
    float variation       = vs_colorVariation[0];

    createQuad(basePos, modelWind, randomYAngle, model0,   grassSize, variation);
    createQuad(basePos, modelWind, randomYAngle, model45,  grassSize, variation);
    createQuad(basePos, modelWind, randomYAngle, modelm45, grassSize, variation);
}
