// #version 450
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 vPosition;
// layout(location = 1) in vec2 vTexcoord;

// layout(location = 0) out vec4 out_position; // xyz = displaced world-space pos, w = extra (jacobian/flag)
// layout(location = 1) out vec3 out_normal;
// layout(location = 2) out vec2 out_texcoord;

// layout(std140, binding = 0) uniform VertexUBO
// {
//     mat4 modelMatrix;
//     mat4 viewMatrix;
//     mat4 projectionMatrix;
//     vec4 waveParameters; // x=heightAmp, y=choppy, z=scale, w=time
// } ubo;

// layout(binding = 1) uniform sampler2D displacementMap;
// layout(binding = 2) uniform sampler2D normalMap;

// void main()
// {
//     // Clip-space from original position (model transform applied to original vertex)
//     gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(vPosition, 1.0);

//     // Sample displacement (use uv scaled by ubo.scale)
//     vec2 uvScaled = vTexcoord * ubo.waveParameters.z;
//     vec4 displacement = texture(displacementMap, uvScaled);

//     // vertical amplification
//     displacement.y *= ubo.waveParameters.x;

//     // Optionally apply horizontal choppy displacement if available in disp.xz and choppy > 0:
//     // The reference code uses displacement.x and displacement.z scaled by lambda/choppy. Keep it simple:
//     vec3 displaced = vPosition + vec3(displacement.x, displacement.y, displacement.z);

//     out_position.xyz = vec3(ubo.modelMatrix * vec4(displaced, 1.0)); // world-space displaced position
//     out_position.w   = displacement.w; // propagate extra if you want (jacobian / flag)

//     // Slope/normal map: convert slope -> surface normal and apply choppy
//     vec4 slope = texture(normalMap, uvScaled);
//     float choppy = ubo.waveParameters.y;
//     out_normal = normalize(vec3(
//         - (slope.x / (1.0 + choppy * slope.z)),
//          1.0,
//         - (slope.y / (1.0 + choppy * slope.w))
//     ));

//     out_texcoord = vTexcoord;
// }



#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPos;   // mesh position
layout(location = 1) in vec2 inUV;    // UV coordinates

layout(binding = 0) uniform VertexUBO 
{
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 waveParameters; // x=HeightAmp, y=Choppy, z=Scale, w=Time
} ubo;

layout(binding = 1) uniform sampler2D dudvMap;
layout(binding = 2) uniform sampler2D normalMap;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragPos;

void main()
{
    float height = ubo.waveParameters.x;
    float choppy = ubo.waveParameters.y;
    float scale  = ubo.waveParameters.z;
    float time   = ubo.waveParameters.w;

    // Scale + animate UVs for displacement
    vec2 uvScaled   = inUV * scale;
    vec2 dudvShift  = vec2(time * 0.05, time * 0.03);
    vec2 uvTime     = uvScaled + dudvShift;

    // Sample dudv (DuDv) map and compute vertical displacement
    vec3 dudv       = texture(dudvMap, uvTime).rgb;
    float dispHeight= (dudv.r - 0.5) * height;

    // Apply vertical displacement
    vec3 displacedPos = inPos + vec3(0.0, dispHeight, 0.0);

    // Final clip-space position
    gl_Position = ubo.projectionMatrix *
                  ubo.viewMatrix *
                  ubo.modelMatrix *
                  vec4(displacedPos, 1.0);

    // Animated UVs for normal map sampling
    vec2 nUV = uvScaled + vec2(time * 0.02, time * 0.04);
    vec3 n   = texture(normalMap, nUV).rgb;

    // Transform [0,1] → [-1,1]
    fragNormal = normalize(n * 2.0 - 1.0);

    // Pass world-space position and UV to fragment shader
    fragPos = vec3(ubo.modelMatrix * vec4(displacedPos, 1.0));
    fragUV  = inUV;
}
