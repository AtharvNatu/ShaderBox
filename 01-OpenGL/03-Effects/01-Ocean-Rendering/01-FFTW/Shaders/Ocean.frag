#version 460 core

in vec3 out_position;
in vec3 out_normal;

uniform vec3 u_lightPosition;
uniform vec3 u_lightAmbient;
uniform vec3 u_lightDiffuse;
uniform vec3 u_lightSpecular;

uniform vec3 u_viewPosition;
uniform float u_heightMax;
uniform float u_heightMin;

out vec4 FragColor;

void main(void)
{ 
    vec3 normalized_normals = normalize(out_normal);
    vec3 light_direction = normalize(u_lightPosition - out_position);
    vec3 view_direction = normalize(u_viewPosition - out_position);

    vec3 ambientFactor = vec3(0.0);
    vec3 diffuseFactor = vec3(1.0);
    vec3 skyColor = vec3(0.65, 0.80, 0.95);

    if (dot(normalized_normals, view_direction) < 0)
        normalized_normals = -normalized_normals;

    // Ambient
    vec3 ambient = u_lightAmbient * ambientFactor;

    vec3 shallowColor = vec3(0.0, 0.64, 0.68);
    vec3 deepColor = vec3(0.02, 0.05, 0.10);
    float relativeHeight = (out_position.y - u_heightMin) / (u_heightMax - u_heightMin);
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
    vec3 specular = u_lightSpecular * specularCoefficient;

    vec3 combinedColor = ambient + heightColor + reflectColor;

    specularCoefficient = clamp(specularCoefficient, 0, 1);
    combinedColor *= (1 - specularCoefficient);
    combinedColor += specular;

    FragColor = vec4(combinedColor, 1.0);

}
