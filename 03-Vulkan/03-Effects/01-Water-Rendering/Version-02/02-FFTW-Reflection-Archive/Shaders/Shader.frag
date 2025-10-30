#version 460
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 out_position;
layout(location = 1) in vec3 out_normal;

layout(location = 0) out vec4 FragColor;

layout(binding = 1) uniform WaterUBO
{
    //* Light Attributes
    vec4 lightPosition;
    vec4 lightAmbient;
    vec4 lightDiffuse;
    vec4 lightSpecular;

    //* Misc
    vec4 viewPosition;
    vec4 heightVector;  // 0 -> Height Min | 1 -> Height Max | 2, 3 -> Padding

} waterUbo;

void main()
{
    vec3 normalized_normals = normalize(out_normal);
    vec3 light_direction = normalize(waterUbo.lightPosition.xyz - out_position);
    vec3 view_direction = normalize(waterUbo.viewPosition.xyz - out_position.xyz);

    vec3 ambientFactor = vec3(0.0);
    vec3 diffuseFactor = vec3(1.0);
    vec3 skyColor = vec3(0.65, 0.80, 0.95);

    if (dot(normalized_normals, view_direction) < 0)
        normalized_normals = -normalized_normals;

    // Ambient
    vec3 ambient = waterUbo.lightAmbient.rgb * ambientFactor;

    vec3 shallowColor = vec3(0.0, 0.64, 0.68);
    vec3 deepColor = vec3(0.02, 0.05, 0.10);
    // float relativeHeight = (out_position.y - waterUbo.heightVector.x) / (waterUbo.heightVector.y - waterUbo.heightVector.x);
    float relativeHeight = clamp(
    (out_position.y - waterUbo.heightVector.x) / max((waterUbo.heightVector.y - waterUbo.heightVector.x), 0.001),
    0.0, 1.0);
    vec3 heightColor = relativeHeight * shallowColor + (1 - relativeHeight) * deepColor;

    // Spray
    float sprayUpperThreshold = 1.0;
    float sprayLowerThreshold = 0.9;
    float sprayRatio = 0;
    if (relativeHeight > sprayLowerThreshold)
        sprayRatio = (relativeHeight - sprayLowerThreshold) / (sprayUpperThreshold - sprayLowerThreshold);
    vec3 sprayBaseColor = vec3(1.0);
    vec3 sprayColor = sprayRatio * sprayBaseColor;

    // Pseudo Reflect -> Smaller power will have more concentrated reflect
    float reflectionCoefficient = pow(max(dot(normalized_normals, view_direction), 0.0), 0.3);
    vec3 reflectColor = (1 - reflectionCoefficient) * skyColor;

    // Specular
    vec3 reflectionDirection = reflect(-light_direction, normalized_normals);
    float specularCoefficient = pow(max(dot(view_direction, reflectionDirection), 0.0), 64) * 3;
    vec3 specular = waterUbo.lightSpecular.rgb * specularCoefficient;

    vec3 combinedColor = ambient + heightColor + reflectColor;

    specularCoefficient = clamp(specularCoefficient, 0, 1);
    combinedColor *= (1 - specularCoefficient);
    combinedColor += specular;
    combinedColor += sprayColor;

    FragColor = vec4(combinedColor, 1.0);
}

// #version 460
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 out_position;
// layout(location = 1) in vec3 out_normal;

// layout(location = 0) out vec4 FragColor;

// layout(binding = 1) uniform WaterUBO
// {
//     // Light Attributes
//     vec4 lightPosition;
//     vec4 lightAmbient;
//     vec4 lightDiffuse;
//     vec4 lightSpecular;

//     // Misc
//     vec4 viewPosition;
//     vec4 heightVector; // x = minHeight, y = maxHeight
// } waterUbo;

// void main()
// {
//     // Normalize inputs
//     vec3 N = normalize(out_normal);
//     vec3 L = normalize(waterUbo.lightPosition.xyz - out_position);
//     vec3 V = normalize(waterUbo.viewPosition.xyz - out_position);
//     vec3 H = normalize(L + V);

//     if (dot(N, V) < 0.0)
//         N = -N;

//     // --- Base colors
//     vec3 shallowColor = vec3(0.0, 0.64, 0.68);
//     vec3 deepColor    = vec3(0.02, 0.05, 0.10);
//     vec3 skyColor     = vec3(0.65, 0.80, 0.95);

//     // --- Relative height normalization
//     float relativeHeight = clamp(
//         (out_position.y - waterUbo.heightVector.x) /
//         max((waterUbo.heightVector.y - waterUbo.heightVector.x), 0.001),
//         0.0, 1.0);

//     vec3 heightColor = mix(deepColor, shallowColor, relativeHeight);

//     // --- Foam / Spray at crests
//     float foamThreshold = 0.85;
//     float foamRange = 0.15;
//     float foamMask = smoothstep(foamThreshold, foamThreshold + foamRange, relativeHeight);
//     vec3 foamColor = mix(vec3(0.0), vec3(1.0), foamMask);

//     // --- Diffuse Lighting
//     float NdotL = max(dot(N, L), 0.0);
//     vec3 diffuse = waterUbo.lightDiffuse.rgb * NdotL * 0.7; // softer diffuse

//     // --- Specular Highlight
//     float specPower = 64.0;
//     float specularStrength = 0.3;
//     float NdotH = max(dot(N, H), 0.0);
//     vec3 specular = waterUbo.lightSpecular.rgb *
//                     pow(NdotH, specPower) * specularStrength;

//     // --- Fresnel (Schlick's approximation)
//     float F0 = 0.02;
//     float fresnel = F0 + (1.0 - F0) * pow(1.0 - max(dot(N, V), 0.0), 5.0);
//     fresnel = clamp(fresnel, 0.0, 0.5); // limit reflectivity
//     vec3 reflection = skyColor * fresnel;

//     // --- Combine Components
//     vec3 baseColor = mix(heightColor, reflection, fresnel);
//     baseColor += diffuse + specular;
//     baseColor = mix(baseColor, foamColor, foamMask * 0.5);

//     // --- Ambient
//     baseColor += waterUbo.lightAmbient.rgb * 0.15;

//     // --- Simple Tone Mapping (ACES-like)
//     baseColor = baseColor / (baseColor + vec3(1.0));
//     baseColor = pow(baseColor, vec3(1.0 / 2.2)); // gamma correction

//     FragColor = vec4(clamp(baseColor, 0.0, 1.0), 1.0);
// }

