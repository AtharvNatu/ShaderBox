// #version 460 core
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 vPosition;
// layout(location = 1) in vec2 vTexcoord;

// layout(location = 0) out vec3 fragPosition;
// layout(location = 1) out vec3 fragNormal;
// layout(location = 2) out vec2 fragTexcoord;

// layout(binding = 0) uniform VertexUBO 
// { 
//     mat4 modelMatrix;
//     mat4 viewMatrix;
//     mat4 projectionMatrix;
//     vec4 waveParameters; // x = heightAmp, y = choppy, z = scale, w = padding
// } ubo;

// layout(binding = 2) uniform sampler2D displacementMap;
// layout(binding = 3) uniform sampler2D normalMap;

// void main()
// {
//     // Sample displacement map
//     vec4 disp = texture(displacementMap, vTexcoord * ubo.waveParameters.z);

//     // Apply vertical amplification
//     disp.y *= ubo.waveParameters.x;

//     // Move vertex
//     vec3 displacedPosition = vPosition + disp.xyz;

//     // Output final clip position
//     gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(displacedPosition, 1.0);

//     // Pass displaced world position to fragment shader
//     fragPosition = vec3(ubo.modelMatrix * vec4(displacedPosition, 1.0));

//     // Sample slope/normal data
//     vec2 slope = texture(normalMap, vTexcoord * ubo.waveParameters.z).xy;

//     // Construct approximate normal
//     fragNormal = normalize(vec3(-slope.x, 1.0, -slope.y));

//     fragTexcoord = vTexcoord;
// }


// #version 450
// #extension GL_ARB_separate_shader_objects : enable

// const vec2 positions[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));

// void main() 
// {
//     gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
// }



#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPos;   // from vertex buffer
layout(location = 1) in vec2 inUV;    // unused for now

layout(binding = 0) uniform VertexUBO 
{ 
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 waveParameters;
} ubo;

layout(location = 0) out vec3 fragColor;

void main() 
{
    // GOLDEN LINE
    // gl_Position = vec4(inPos.x / 10.0, inPos.z / 10.0, 0.0, 1.0);

    gl_Position = vec4(inPos, 1.0);
    // gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(inPos, 1.0);

    fragColor = (inPos + vec3(10.0)) / 20.0; // normalize into [0,1]
    // fragColor = vec3(ubo.modelMatrix.z / 10.0, 0.0, 0.0);
}
