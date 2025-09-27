// #version 460 core
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 fragPosition;
// layout(location = 1) in vec3 fragNormal;
// layout(location = 2) in vec2 fragTexcoord;

// layout(location = 0) out vec4 FragColor;

// layout(binding = 1) uniform WaterSurfaceUBO
// {
//     vec4 cameraPosition;
//     vec4 absorptionCoefficient;
//     vec4 scatterCoefficient;
//     vec4 backScatterCoefficient;
//     vec4 terrainColor;
//     vec4 surfaceParameters; // x = Height, y = SkyIntensity, z = SpecularIntensity, w = SpecularHighlights
// } waterUbo;

// void main()
// {
//     // Light position in world space
//     // vec3 lightPos = vec3(20.0, 50.0, -30.0);

//     // // Lighting vectors
//     // vec3 N = normalize(fragNormal);
//     // vec3 L = normalize(lightPos - fragPosition);
//     // vec3 V = normalize(waterUbo.cameraPosition.xyz - fragPosition);
//     // vec3 H = normalize(L + V);

//     // // Diffuse term
//     // float diff = max(dot(N, L), 0.0);

//     // // Specular term (use w for shininess, z for intensity)
//     // float spec = pow(max(dot(N, H), 0.0), waterUbo.surfaceParameters.w) * waterUbo.surfaceParameters.z;

//     // // Simple bluish water base color
//     // vec3 baseColor = vec3(0.0, 0.3, 0.6);

//     // // Combine lighting
//     // vec3 color = baseColor * diff + spec * vec3(1.0);

//     // FragColor = vec4(color, 1.0);

//     // FragColor = vec4(fragNormal * 0.5 + 0.5, 1.0);
//     FragColor = vec4(1.0, 0.0, 0.0, 1.0);

// }



// #version 450
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec2 uv;
// layout(location = 1) in vec3 fragNormal;

// layout(location = 0) out vec4 outColor;

// // layout(binding = 3) uniform WaterSurfaceUBO
// // {
// //     vec4 cameraPosition;
// //     vec4 absorptionCoefficient;
// //     vec4 scatterCoefficient;
// //     vec4 backScatterCoefficient;
// //     vec4 terrainColor;
// //     vec4 surfaceParameters; // x = Height, y = SkyIntensity, z = SpecularIntensity, w = SpecularHighlights
// // } waterUbo;


// void main() 
// {
//     outColor = vec4(uv, 0.0, 1.0); // visible gradient
    
//     // vec3 N = normalize(fragNormal);
//     // vec3 L = normalize(vec3(0.0, 1.0, 1.0)); // test light
//     // float diff = max(dot(N, L), 0.0);

//     // outColor = vec4(vec3(diff), 1.0); // simple diffuse
// }


// #version 450
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec2 uv;
// layout(location = 1) in vec3 fragNormal;
// layout(location = 2) in vec3 fragWorldPos;

// layout(location = 0) out vec4 outColor;

// layout(std140, binding = 3) uniform WaterSurfaceUBO
// {
//     vec4 cameraPosition;
//     vec4 absorptionCoefficient;
//     vec4 scatterCoefficient;
//     vec4 backScatterCoefficient;
//     vec4 terrainColor;
//     vec4 surfaceParameters; // x=Height, y=SkyIntensity, z=SpecularIntensity, w=SpecularHighlights
// } waterUbo;

// void main() 
// {
//     vec3 N = normalize(fragNormal);
//     vec3 L = normalize(vec3(0.0, 1.0, 1.0)); // test light from above
//     vec3 V = normalize(waterUbo.cameraPosition.xyz - fragWorldPos);
//     vec3 H = normalize(L + V);

//     // Diffuse term
//     float diff = max(dot(N, L), 0.0);

//     // Specular term (Phong)
//     float spec = pow(max(dot(N, H), 0.0), waterUbo.surfaceParameters.w) 
//                  * waterUbo.surfaceParameters.z;

//     // Simple bluish base water color
//     vec3 baseColor = vec3(0.0, 0.3, 0.6);

//     vec3 color = baseColor * diff + spec * vec3(1.0);

//     outColor = vec4(color, 1.0);
// }

#version 450
#extension GL_ARB_separate_shader_objects : enable
layout(location = 0) in vec4 dispColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = dispColor;
}
