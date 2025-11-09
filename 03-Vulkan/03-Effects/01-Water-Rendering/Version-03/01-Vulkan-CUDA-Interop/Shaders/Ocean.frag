#version 460
#extension GL_ARB_separate_shader_objects : enable

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------
layout(location = 0) in vec3 out_eyeSpacePosition;
layout(location = 1) in vec3 out_eyeSpaceNormal;
layout(location = 2) in vec3 out_worldSpaceNormal;

layout(location = 0) out vec4 FragColor;

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------
layout(binding = 1) uniform OceanUBO
{
    vec4 deepColor;
    vec4 shallowColor;
    vec4 skyColor;
    vec4 lightDirection;

    float heightScale;
    float choppiness;
    vec2 size;
} oceanUbo;

// Foam & Bubble textures
layout(binding = 2) uniform sampler2D texFoam;
layout(binding = 3) uniform sampler2D texBubbles;

// ---------------------------------------------------------------------------
// Surface parameters (for procedural foam calculation)
// ---------------------------------------------------------------------------
struct SurfaceParams {
    float foamEnergy;          // turbulence intensity
    float foamSurfaceFolding;  // curvature or steepness
    float foamWaveHats;        // crest factor
};

// ---------------------------------------------------------------------------
// Foam & Bubbles Calculation
// ---------------------------------------------------------------------------
float computeFoam(vec3 worldPos, SurfaceParams surf)
{
    // Sample multi-octave foam patterns
    float foamDensityLow       = texture(texFoam, worldPos.xy * 0.04).r - 1.0;
    float foamDensityHigh      = texture(texFoam, worldPos.xy * 0.15).r - 1.0;
    float foamDensityVeryHigh  = texture(texFoam, worldPos.xy * 0.30).r;
    float foamBubbles          = texture(texBubbles, worldPos.xy * 0.25).r;

    // Combine octaves
    float foamDensity = clamp(foamDensityHigh + min(3.5, surf.foamEnergy - 0.2), 0.0, 1.0);
    foamDensity += clamp(foamDensityLow + min(1.5, surf.foamEnergy), 0.0, 1.0);

    // Add folding enhancement
    foamDensity -= 0.1 * clamp(-surf.foamSurfaceFolding, 0.0, 1.0);
    foamDensity = max(0.0, foamDensity);
    foamDensity *= 1.0 + 0.8 * clamp(surf.foamSurfaceFolding, 0.0, 1.0);

    // Add crest foam (“wave hats”)
    foamDensity += max(0.0, foamDensityVeryHigh * 2.0 * surf.foamWaveHats);
    foamDensity = pow(foamDensity, 0.7);

    // Multiply by bubble pattern for microfoam detail
    foamBubbles = clamp(5.0 * (foamBubbles - 0.8), 0.0, 1.0);
    foamDensity = clamp(foamDensity * foamBubbles, 0.0, 1.0);

    return foamDensity;
}

// ---------------------------------------------------------------------------
// Foam Color Blending
// ---------------------------------------------------------------------------
vec3 applyFoamColor(vec3 baseColor, vec3 lightColor, float foamIntensity, SurfaceParams surf)
{
    const vec3 foamColor = vec3(0.9, 0.9, 0.9);
    const vec3 foamUnderwaterColor = vec3(0.6, 0.6, 0.6);

    // Bright foam diffuse
    vec3 foamDiffuse = foamColor * (0.3 + 0.7 * foamIntensity);

    // Subtle foam glow under surface
    vec3 foamUnderwater = foamUnderwaterColor * clamp(surf.foamEnergy * 0.05, 0.0, 1.0);

    // Blend foam into water
    vec3 color = mix(baseColor, foamDiffuse * lightColor, foamIntensity);
    color += foamUnderwater * lightColor * 0.5;

    return color;
}

// ---------------------------------------------------------------------------
// Main Shader
// ---------------------------------------------------------------------------
void main()
{
    vec3 eyeVector = normalize(out_eyeSpacePosition);
    vec3 eyeSpaceNormal = normalize(out_eyeSpaceNormal);
    vec3 worldNormal = normalize(out_worldSpaceNormal);

    float facing = max(0.0, dot(eyeSpaceNormal, -eyeVector));
    float fresnel = pow(1.0 - facing, 5.0);
    float diffuse = max(0.0, dot(worldNormal, normalize(oceanUbo.lightDirection.xyz)));

    // Base water shading
    vec3 deep = oceanUbo.deepColor.rgb;
    vec3 shallow = oceanUbo.shallowColor.rgb;
    vec3 sky = oceanUbo.skyColor.rgb;
    vec3 baseWaterColor = mix(deep, shallow, diffuse);
    vec3 reflection = sky * fresnel;
    vec3 lighting = baseWaterColor * diffuse + reflection;

    // Procedural foam setup (values can be animated over time)
    SurfaceParams surf;
    surf.foamEnergy = 0.8;
    surf.foamSurfaceFolding = abs(dot(worldNormal, normalize(oceanUbo.lightDirection.xyz))) * 1.2;
    surf.foamWaveHats = smoothstep(0.6, 1.0, fresnel); // more foam near glancing angles

    float foamIntensity = computeFoam(out_eyeSpacePosition, surf);

    // Apply foam on top of lighting
    vec3 finalColor = applyFoamColor(lighting, vec3(1.0), foamIntensity, surf);

    FragColor = vec4(finalColor, 1.0);
}
