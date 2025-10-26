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
    float relativeHeight = (out_position.y - waterUbo.heightVector.x) / (waterUbo.heightVector.y - waterUbo.heightVector.x);
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
