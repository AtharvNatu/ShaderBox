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

void main()
{
    vec3 eyeVector = normalize(out_eyeSpacePosition);
    vec3 eyeSpaceNormal = normalize(out_eyeSpaceNormal);
    vec3 worldSpaceNormal = normalize(out_worldSpaceNormal);
    vec3 lightDir = normalize(oceanUbo.lightDirection.xyz);

    // Fresnel and diffuse terms
    float facing = max(0.0, dot(eyeSpaceNormal, -eyeVector));
    float fresnel = pow(1.0 - facing, 5.0);
    float diffuse = max(0.0, dot(worldSpaceNormal, lightDir));

    // Height-based blending (similar to shallow/deep blending)
    float height = out_eyeSpacePosition.y * oceanUbo.heightScale;
    float relativeHeight = clamp(height * 0.5 + 0.5, 0.0, 1.0);
    vec3 heightColor = mix(oceanUbo.deepColor.rgb, oceanUbo.shallowColor.rgb, relativeHeight);

    // Reflection / sky color blend (approximate)
    vec3 reflectionColor = mix(heightColor, oceanUbo.skyColor.rgb, fresnel);

    // Simple specular highlight using Blinn–Phong
    vec3 viewDir = normalize(-out_eyeSpacePosition);
    vec3 halfVec = normalize(lightDir + viewDir);
    float spec = pow(max(dot(worldSpaceNormal, halfVec), 0.0), 64.0) * 0.5;

    // Combine diffuse + reflection + specular
    vec3 finalColor = heightColor * diffuse + reflectionColor * fresnel + spec * oceanUbo.skyColor.rgb;

    FragColor = vec4(finalColor, 1.0);

}

