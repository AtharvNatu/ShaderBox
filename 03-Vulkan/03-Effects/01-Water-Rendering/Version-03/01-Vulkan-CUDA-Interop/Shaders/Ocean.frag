#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 out_eyeSpacePosition;
layout(location = 1) in vec3 out_eyeSpaceNormal;
layout(location = 2) in vec3 out_worldSpaceNormal;

layout(location = 0) out vec4 FragColor;

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

// ---- USER-TUNABLE PARAMETERS ----
float streakWidth     = 0.25;  // bigger = wider (0.1–2.0)
float streakLength    = 40.0;  // bigger = longer (10–200)
float streakSoftness  = 1.5;   // bigger = softer (0.5–4.0)
float streakStrength  = 5.0;   // brightness (1–20)


//------------------------------------------------------------
// Physically-based cinematic sunset streak
// Spread in *reflection direction* + *sideways width*
//------------------------------------------------------------
float computeSunStreak(vec3 N, vec3 V, vec3 L)
{
    // Maximum streak when sun near horizon
    float horizon = clamp(1.0 - abs(L.y), 0.0, 1.0);
    if (horizon < 0.01)
        return 0.0;

    // Ideal reflection direction (what water “would” reflect)
    vec3 R = normalize(reflect(-L, N));

    //----------------------------------------------------------------
    // 1) ALIGNMENT LENGTH (how much camera looks into reflection ray)
    //----------------------------------------------------------------
    float alignment = max(dot(R, V), 0.0);    // 1.0 = perfect mirror alignment

    // Exponential control: higher = longer / thinner
    float lengthTerm = exp(-pow(1.0 - alignment, streakLength));

    //----------------------------------------------------------------
    // 2) WIDTH CONTROL (how far camera is off sideways)
    //----------------------------------------------------------------
    // Compute sideways angle between V and the reflection plane
    float sideways = length(cross(R, V));      // 0 = perfect alignment

    // Smooth Gaussian width attenuation
    float widthTerm = exp(-(sideways * sideways) / (streakWidth * streakWidth));

    //----------------------------------------------------------------
    // 3) SOFT GLOWING HALO
    //----------------------------------------------------------------
    float glow = pow(alignment, streakSoftness);

    //----------------------------------------------------------------
    // Final streak intensity
    //----------------------------------------------------------------
    float streak = lengthTerm * widthTerm * glow * horizon;

    return streak;
}


void main()
{
    vec3 N_eye   = normalize(out_eyeSpaceNormal);
    vec3 V_eye   = normalize(-out_eyeSpacePosition);
    vec3 N_world = normalize(out_worldSpaceNormal);
    vec3 L       = normalize(oceanUbo.lightDirection.xyz);

    //-------------------------------------------------------
    // Base lighting
    //-------------------------------------------------------
    float facing  = max(0.0, dot(N_eye, -normalize(out_eyeSpacePosition)));
    float fresnel = pow(1.0 - facing, 5.0);
    float diffuse = max(dot(N_world, L), 0.0);

    //-------------------------------------------------------
    // Height-based deep vs shallow
    //-------------------------------------------------------
    float height = out_eyeSpacePosition.y * oceanUbo.heightScale;
    float relativeHeight = clamp(height * 0.5 + 0.5, 0.0, 1.0);

    vec3 heightColor = mix(
        oceanUbo.deepColor.rgb,
        oceanUbo.shallowColor.rgb,
        relativeHeight
    );

    //-------------------------------------------------------
    // Fresnel sky mix
    //-------------------------------------------------------
    vec3 reflectionColor = mix(
        heightColor,
        oceanUbo.skyColor.rgb,
        fresnel
    );

    //-------------------------------------------------------
    // Specular
    //-------------------------------------------------------
    vec3 H = normalize(L + V_eye);
    float spec  = pow(max(dot(N_world, H), 0.0), 64.0) * 0.5;

    //-------------------------------------------------------
    // Base shading
    //-------------------------------------------------------
    vec3 finalColor =
        heightColor * diffuse +
        reflectionColor * fresnel +
        spec * oceanUbo.skyColor.rgb;

    //-------------------------------------------------------
    // SUN STREAK (big, adjustable, cinematic)
    //-------------------------------------------------------
    float streak = computeSunStreak(N_world, V_eye, L) * streakStrength;

    vec3 streakColor = vec3(1.0, 0.40, 0.08); // realistic glowing sunset orange

    finalColor += streak * streakColor;

    FragColor = vec4(finalColor, 1.0);
}
