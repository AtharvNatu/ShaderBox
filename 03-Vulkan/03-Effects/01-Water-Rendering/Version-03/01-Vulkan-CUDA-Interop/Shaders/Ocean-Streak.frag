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

//-------------------------------------------------------
// Compute narrow sunset streak (very directional)
//-------------------------------------------------------
float computeSunStreak(vec3 N, vec3 V, vec3 L)
{
    // Sun at horizon → strong streak
    float horizon = clamp(1.0 - abs(L.y), 0.0, 1.0);

    // Reflection vector
    vec3 R = reflect(-L, N);

    // Alignment between camera direction and perfect reflection
    float alignment = max(dot(R, V), 0.0);

    // Very thin streak
    float streak = pow(alignment, 200.0) * horizon;

    return streak;
}


void main()
{
    vec3 eyeVector     = normalize(out_eyeSpacePosition);
    vec3 eyeNormal     = normalize(out_eyeSpaceNormal);
    vec3 worldNormal   = normalize(out_worldSpaceNormal);
    vec3 L             = normalize(oceanUbo.lightDirection.xyz);
    vec3 V             = normalize(-out_eyeSpacePosition);

    //-------------------------------------------------------
    // Base lighting
    //-------------------------------------------------------
    float facing  = max(0.0, dot(eyeNormal, -eyeVector));
    float fresnel = pow(1.0 - facing, 5.0);
    float diffuse = max(0.0, dot(worldNormal, L));

    //-------------------------------------------------------
    // Height-based shallow/deep blend
    //-------------------------------------------------------
    float height = out_eyeSpacePosition.y * oceanUbo.heightScale;
    float relativeHeight = clamp(height * 0.5 + 0.5, 0.0, 1.0);

    vec3 heightColor = mix(
        oceanUbo.deepColor.rgb,
        oceanUbo.shallowColor.rgb,
        relativeHeight
    );

    //-------------------------------------------------------
    // Fresnel sky reflection
    //-------------------------------------------------------
    vec3 reflectionColor = mix(
        heightColor,
        oceanUbo.skyColor.rgb,
        fresnel
    );

    //-------------------------------------------------------
    // Blinn–Phong specular highlight
    //-------------------------------------------------------
    vec3 H = normalize(L + V);
    float spec  = pow(max(dot(worldNormal, H), 0.0), 64.0) * 0.5;

    //-------------------------------------------------------
    // Combine (without streak)
    //-------------------------------------------------------
    vec3 finalColor =
        heightColor * diffuse +
        reflectionColor * fresnel +
        spec * oceanUbo.skyColor.rgb;

    //-------------------------------------------------------
    // Sunset sun streak — only appears when flag enabled
    //-------------------------------------------------------
    float streak = computeSunStreak(worldNormal, V, L)
                       * 5.0f;

    vec3 streakColor = vec3(1.0, 0.45, 0.1);  // warm sunset tint

    finalColor += streak * streakColor;

    FragColor = vec4(finalColor, 1.0);
}

