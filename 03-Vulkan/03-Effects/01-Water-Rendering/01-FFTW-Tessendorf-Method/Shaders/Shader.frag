// #version 450
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec4 out_position;
// layout(location = 1) in vec3 out_normal;
// layout(location = 2) in vec2 out_texcoord;

// layout(location = 0) out vec4 FragColor;

// layout(std140, binding = 3) uniform WaterSurfaceUBO
// {
//     vec4 cameraPosition;
//     vec4 absorptionCoefficient;
//     vec4 scatterCoefficient;
//     vec4 backScatterCoefficient;
//     vec4 terrainColor;
//     vec4 surfaceParameters; // x=Height, y=SkyIntensity, z=SpecularIntensity, w=SpecularHighlights
// } waterUbo;

// // (You can copy relevant functions from your repo here — Fresnel, Attenuate etc.)
// // For testing, we'll do a simple lighting + terrain look-up as in your repo:

// void main()
// {
//     vec3 pw = vec3(out_position.x, out_position.y + waterUbo.surfaceParameters.x, out_position.z);
//     vec3 N = normalize(out_normal);

//     // Simple test lighting
//     vec3 L = normalize(vec3(0.0, 1.0, 1.0));
//     float diff = max(dot(N, L), 0.0);
//     vec3 base = waterUbo.terrainColor.rgb; // base water color from UBO
//     vec3 color = base * diff;

//     FragColor = vec4(color, 1.0);

//     // optional tone map (repo):
//     FragColor = vec4(1.0) - exp(-FragColor * 2.0);

//     FragColor = vec4(vec3(diff), 1.0);
// }


#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

layout(std140, binding = 3) uniform WaterSurfaceUBO {
    vec4 cameraPosition;
    vec4 absorptionCoefficient;
    vec4 scatterCoefficient;
    vec4 backScatterCoefficient;
    vec4 terrainColor;
    vec4 surfaceParameters; // x=Height, y=SkyIntensity, z=SpecularIntensity, w=Shininess
} waterUbo;


void main() 
{
    // // Normalize vectors
    // vec3 N = normalize(fragNormal);
    // vec3 L = normalize(vec3(0.0, 1.0, 1.0));                // test light direction
    // vec3 V = normalize(waterUbo.cameraPosition.xyz - fragPos);
    // vec3 H = normalize(L + V);

    // // Diffuse lighting
    // float diff = max(dot(N, L), 0.0);

    // // Specular lighting
    // float spec = pow(max(dot(N, H), 0.0), waterUbo.surfaceParameters.w) 
    //              * waterUbo.surfaceParameters.z;

    // // Simple water base color
    // vec3 baseColor = vec3(0.0, 0.4, 0.7);

    // // Combine lighting
    // vec3 color = baseColor * diff + vec3(spec);

    // outColor = vec4(color, 1.0);

    vec3 camPos = waterUbo.cameraPosition.xyz;
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(camPos - fragPos);

    // Light direction (like sun)
    vec3 L = normalize(vec3(0.3, 1.0, 0.2));

    // Diffuse
    float diff = max(dot(N, L), 0.0);

    // Specular
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), waterUbo.surfaceParameters.w) *
                 waterUbo.surfaceParameters.z;

    // Base water tint (terrainColor * sky intensity)
    vec3 baseColor = waterUbo.terrainColor.rgb * waterUbo.surfaceParameters.y;

    vec3 color = baseColor * diff + vec3(spec);

    // Absorption with depth
    float dist = length(camPos - fragPos);
    color *= exp(-waterUbo.absorptionCoefficient.rgb * dist * 0.02);

    // Tone mapping
    outColor = vec4(1.0) - exp(-vec4(color, 1.0) * 2.0);
}
