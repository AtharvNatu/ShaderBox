#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 instancePosition;

layout(location = 0) out vec3 vs_basePosition;
layout(location = 1) out mat4 vs_modelWindMatrix;
layout(location = 5) out float vs_randomYAngle;
layout(location = 6) out float vs_colorVariation;

layout(set = 0, binding = 0) uniform UBO 
{ 
    mat4 modelMatrix;
    mat4 projectionMatrix;
    float time;    
    float windStrength;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D uWindSampler;

// Define fbm and random here or inline it as simplified function

float random(vec2 st)
{
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float fbm(in vec2 st)
{
    // simplified or full implementation as in your GS, but lighter if possible
    float v = 0.0;
    float a = 0.5;
    vec2 shift = vec2(100.0);
    mat2 rotate = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));
    for (int i = 0; i < 5; i++)
    {
        v += a * random(st);
        st = rotate * st * 2.0 + shift;
        a *= 0.5;
    }
    return v;
}

mat4 rotationX(in float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0);
}

mat4 rotationZ(in float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0);
}

mat4 rotationY(in float angle)
{
    float c = cos(angle);
    float s = sin(angle);
    return mat4(
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0);
}

void main(void)
{
    vs_basePosition = instancePosition;

    // Compute wind UV coords & sample wind texture
    vec2 windDirection = vec2(1.0, 1.0);
    vec2 uv = (instancePosition.xz / 10.0) + windDirection * ubo.windStrength * ubo.time;
    uv = fract(uv);

    vec4 wind = texture(uWindSampler, uv);
    mat4 modelWind = rotationX(wind.x * 3.14159265 * 0.75 - 3.14159265 * 0.25) *
                     rotationZ(wind.y * 3.14159265 * 0.75 - 3.14159265 * 0.25);
    vs_modelWindMatrix = modelWind;

    // Compute random Y rotation angle
    vs_randomYAngle = random(instancePosition.xz) * 3.14159265;

    // Compute color variation using fbm noise
    vs_colorVariation = fbm(instancePosition.xz);

    // Pass gl_Position for Vulkan pipeline (required)
    gl_Position = vec4(instancePosition, 1.0);
}
