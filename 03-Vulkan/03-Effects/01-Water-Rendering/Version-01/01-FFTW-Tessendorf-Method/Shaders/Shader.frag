#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 out_position;
layout(location = 1) in vec3 out_normal;
layout(location = 2) in vec2 out_texcoord;

layout(location = 0) out vec4 FragColor;

layout(binding = 3) uniform WaterSurfaceUBO
{
    vec4 cameraPosition;
    vec4 absorptionCoefficient;
    vec4 scatterCoefficient;
    vec4 backScatterCoefficient;
    vec4 terrainColor;
    vec4 surfaceParameters; // x=Height, y=SkyIntensity, z=SpecularIntensity, w=SpecularHighlights
} waterUbo;

// (You can copy relevant functions from your repo here — Fresnel, Attenuate etc.)
// For testing, we'll do a simple lighting + terrain look-up as in your repo:

void main()
{
    vec3 pw = vec3(out_position.x, out_position.y + waterUbo.surfaceParameters.x, out_position.z);
    vec3 N = normalize(out_normal);

    // Simple test lighting
    vec3 L = normalize(vec3(0.0, 1.0, 1.0));
    float diff = max(dot(N, L), 0.0);
    vec3 base = waterUbo.terrainColor.rgb; // base water color from UBO
    vec3 color = base * diff;

    FragColor = vec4(color, 1.0);

    // optional tone map (repo):
    FragColor = vec4(1.0) - exp(-FragColor * 2.0);

    // FragColor = vec4(vec3(diff), 1.0);
}

