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

// layout(location = 0) in vec3 inPos;   // from vertex buffer
// layout(location = 1) in vec2 inUV;    // unused for now

// layout(binding = 0) uniform VertexUBO 
// { 
//     mat4 modelMatrix;
//     mat4 viewMatrix;
//     mat4 projectionMatrix;
// } ubo;

// layout(binding = 1) uniform sampler2D displacementMap;
// layout(binding = 2) uniform sampler2D normalMap;

// layout(location = 0) out vec2 uv;
// layout(location = 1) out vec3 fragNormal;

// void main() 
// {
//     // GOLDEN LINE
//     // gl_Position = vec4(inPos.x / 10.0, inPos.z / 10.0, 0.0, 1.0);
//     // fragColor = (inPos + vec3(10.0)) / 20.0;

//     // works
//     // gl_Position = vec4(inPos, 1.0);
//     // uv = inUV;

//     // Works
//     // gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(inPos, 1.0);
//     // uv = inUV;

//     vec4 disp = texture(displacementMap, inUV);

//     // apply vertical displacement
//     vec3 displacedPos = inPos + vec3(disp.x, disp.y, disp.z);

//     gl_Position = ubo.projectionMatrix *
//                   ubo.viewMatrix *
//                   ubo.modelMatrix *
//                   vec4(displacedPos, 1.0);

//     vec2 slope = texture(normalMap, inUV).xy;
//     // fragNormal = normalize(vec3(-slope.x, 1.0, -slope.y));
//     fragNormal = normalize(vec3(0.0, 1.0, 0.0));

//     uv = inUV;
// }


// #version 450
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 inPos;   // from vertex buffer
// layout(location = 1) in vec2 inUV;    // grid UVs

// layout(std140, binding = 0) uniform VertexUBO 
// { 
//     mat4 modelMatrix;
//     mat4 viewMatrix;
//     mat4 projectionMatrix;
//     vec4 waveParameters; // (heightAmp, choppy, scale, padding)
// } ubo;

// layout(binding = 1) uniform sampler2D displacementMap;
// layout(binding = 2) uniform sampler2D normalMap;

// layout(location = 0) out vec2 uv;
// layout(location = 1) out vec3 fragNormal;
// layout(location = 2) out vec3 fragWorldPos;

// void main() 
// {
//     // Scale UVs for tiling
//     vec2 texUV = inUV * ubo.waveParameters.z;

//     // ---- DISPLACEMENT MAP ----
//     // displacementMap = vec4(dispX, height, dispZ, jacobian)
//     vec4 disp = texture(displacementMap, texUV);

//     // Apply vertical amplitude (scale Y only)
//     float displacedY = disp.g * ubo.waveParameters.x;

//     // Add horizontal choppiness (optional)
//     vec3 displacedPos = inPos + vec3(disp.r * ubo.waveParameters.y,
//                                      displacedY,
//                                      disp.b * ubo.waveParameters.y);

//     // ---- NORMAL MAP ----
//     // normalMap = vec4(slopeX, slopeZ, dDxdx, dDzdz)
//     vec4 slopeData = texture(normalMap, texUV);
//     vec2 slope = slopeData.rg;

//     fragNormal = normalize(vec3(-slope.x, 1.0, -slope.y));
//     fragWorldPos = vec3(ubo.modelMatrix * vec4(displacedPos, 1.0));

//     gl_Position = ubo.projectionMatrix * ubo.viewMatrix * vec4(fragWorldPos, 1.0);
//     uv = texUV;
// }


#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPos;   // from vertex buffer
layout(location = 1) in vec2 inUV;    // grid UVs

layout(binding = 0) uniform VertexUBO 
{ 
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 waveParameters; // (heightAmp, choppy, scale, padding)
} ubo;

layout(binding = 1) uniform sampler2D displacementMap;
layout(binding = 2) uniform sampler2D normalMap;


layout(location = 0) out vec4 dispColor;

void main() {
    vec4 disp = texture(displacementMap, inUV);
    dispColor = disp; // forward to frag
    gl_Position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vec4(inPos, 1.0);
}
