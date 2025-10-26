#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec2 vTexcoord;

layout(location = 0) out vec4 out_position; // xyz = displaced world-space pos, w = extra (jacobian/flag)
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_texcoord;

layout(binding = 0) uniform VertexUBO
{
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 waveParameters; // x=heightAmp, y=choppy, z=scale, w=time
} ubo;

layout(binding = 1) uniform sampler2D displacementMap;
layout(binding = 2) uniform sampler2D normalMap;

void main()
{
    // Clip-space from original position (model transform applied to original vertex)
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(vPosition, 1.0);

    // Sample displacement (use uv scaled by ubo.scale)
    vec2 uvScaled = vTexcoord * ubo.waveParameters.z;
    vec4 displacement = texture(displacementMap, uvScaled);

    // vertical amplification
    displacement.y *= ubo.waveParameters.x;

    // Optionally apply horizontal choppy displacement if available in disp.xz and choppy > 0:
    // The reference code uses displacement.x and displacement.z scaled by lambda/choppy. Keep it simple:
    vec3 displaced = vPosition + vec3(displacement.x, displacement.y, displacement.z);

    out_position.xyz = vec3(ubo.modelMatrix * vec4(displaced, 1.0)); // world-space displaced position
    out_position.w   = displacement.w; // propagate extra if you want (jacobian / flag)

    // Slope/normal map: convert slope -> surface normal and apply choppy
    vec4 slope = texture(normalMap, uvScaled);
    float choppy = ubo.waveParameters.y;
    out_normal = normalize(vec3(
        - (slope.x / (1.0 + choppy * slope.z)),
         1.0,
        - (slope.y / (1.0 + choppy * slope.w))
    ));

    out_texcoord = vTexcoord;
}

