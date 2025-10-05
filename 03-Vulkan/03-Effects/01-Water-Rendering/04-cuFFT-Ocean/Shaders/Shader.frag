// #version 460 core
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 out_color;
// layout(location = 1) in vec3 out_normal;
// layout(location = 2) in vec3 out_light_direction;
// layout(location = 3) in vec3 out_camera_direction;

// layout(location = 0) out vec4 FragColor;


// void main(void)
// {
//     // Code
//     vec3 halfDir = normalize(out_light_direction + out_camera_direction);
//     float specular = pow(max(dot(out_normal, halfDir), 0.0), 10.0);
//     float diffuse = dot(out_light_direction, out_normal);

//     const vec3 lightColor = 0.4 * normalize(vec3(253, 251, 211));

//     FragColor = vec4(out_color * diffuse + lightColor * specular, 1.0);
// }

// #version 460 core
// #extension GL_ARB_separate_shader_objects : enable

// layout(location = 0) in vec3 frag_position_ws;
// layout(location = 1) in vec3 frag_normal_ws;
// layout(location = 2) in vec3 frag_color;
// layout(location = 3) in vec3 frag_light_dir;
// layout(location = 4) in vec3 frag_view_dir;

// layout(location = 0) out vec4 FragColor;

// // Fresnel approximation (Schlick)
// float fresnelSchlick(float cosTheta, float F0)
// {
//     return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
// }

// void main(void)
// {
//     vec3 N = normalize(frag_normal_ws);
//     vec3 L = normalize(frag_light_dir);   // sun direction
//     vec3 V = normalize(frag_view_dir);
//     vec3 H = normalize(L + V);

//     // --- Lighting ---
//     float NdotL = max(dot(N, L), 0.0);
//     float specular = pow(max(dot(N, H), 0.0), 64.0);

//     vec3 sunColor = vec3(1.0, 0.95, 0.8);

//     // --- Fresnel Reflection ---
//     float cosTheta = clamp(dot(N, V), 0.0, 1.0);
//     float fresnel = fresnelSchlick(cosTheta, 0.02);

//     // --- Horizon Gradient Reflection ---
//     vec3 skyTop    = vec3(0.2, 0.5, 0.9);  // blue sky
//     vec3 skyHorizon= vec3(0.8, 0.9, 1.0);  // pale near horizon
//     float horizonFactor = clamp(N.y * 0.5 + 0.5, 0.0, 1.0);
//     vec3 horizonColor = mix(skyHorizon, skyTop, horizonFactor);

//     // --- Water absorption / deep tint ---
//     vec3 deepColor = vec3(0.0, 0.1, 0.2);

//     // Reflection vs. refraction
//     vec3 reflection = horizonColor;
//     vec3 refraction = deepColor * frag_color;

//     vec3 baseWater = mix(refraction, reflection, fresnel);

//     // --- Sun disk reflection (glint) ---
//     vec3 R = reflect(-V, N);                 // reflection vector
//     float sunGlint = max(dot(R, L), 0.0);    // alignment with sun
//     sunGlint = pow(sunGlint, 200.0);         // control sharpness
//     vec3 sunDisk = sunColor * sunGlint * 2.0; // intensity

//     // Add sunlight (diffuse + specular + sun disk)
//     vec3 sunLight = sunColor * (NdotL + 0.5 * specular);
//     vec3 color = baseWater + sunLight + sunDisk;

//     // Gamma correction
//     color = pow(color, vec3(1.0/2.2));

//     FragColor = vec4(color, 1.0);
// }


#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 FragPos;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 TexCoords;

layout(location = 0) out vec4 FragColor;

layout(binding = 0) uniform VertexUBO 
{
    mat4 modelMatrix;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    vec4 cameraPosition;
} ubo;

layout(binding = 2) uniform LightingUBO 
{
    vec4 lightPosition;
    vec4 cameraPosition;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    float heightMin;
    float heightMax;
} lightingUBO;

void main()
{
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(vec3(lightingUBO.lightPosition) - FragPos); 
    vec3 viewDir  = normalize(vec3(lightingUBO.cameraPosition) - FragPos);

    vec3 ambientFactor = vec3(0.0);
	vec3 diffuseFactor = vec3(1.0);

    vec3 skyColor = vec3(0.65, 0.80, 0.95);

    if (dot(norm, viewDir) < 0) 
        norm = -norm;

    // Ambient
    vec3 ambient = vec3(lightingUBO.ambient) * ambientFactor;

    // Height-based water color
    vec3 shallowColor = vec3(0.0, 0.64, 0.68);
    vec3 deepColor    = vec3(0.02, 0.05, 0.10);
    float relativeHeight = clamp((FragPos.y - lightingUBO.heightMin) / (lightingUBO.heightMax - lightingUBO.heightMin), 0.0, 1.0);
    vec3 heightColor = mix(deepColor, shallowColor, relativeHeight);

    // Spray
	float sprayThresholdUpper = 1.0;
	float sprayThresholdLower = 0.9;
	float sprayRatio = 0;
	if (relativeHeight > sprayThresholdLower) sprayRatio = (relativeHeight - sprayThresholdLower) / (sprayThresholdUpper - sprayThresholdLower);
	vec3 sprayBaseColor = vec3(1.0);
	vec3 sprayColor = sprayRatio * sprayBaseColor;	
	
    // Diffuse  	
	float diff = max(dot(norm, lightDir), 0.0);
	vec3 diffuse = diffuseFactor * vec3(lightingUBO.diffuse) * diff;

    // Pseudo reflection
    float refCoeff = pow(max(dot(norm, viewDir), 0.0), 0.3);
    vec3 reflectColor = (1.0 - refCoeff) * skyColor;

    // Specular
    vec3 reflectDir = reflect(-lightDir, norm);
    float specCoeff = pow(max(dot(viewDir, reflectDir), 0.0), 64.0) * 3.0;
    vec3 specular = vec3(lightingUBO.specular) * specCoeff;

    // Combine
    vec3 combinedColor = ambient + diffuse + heightColor + reflectColor;

    specCoeff = clamp(specCoeff, 0.0, 1.0);
    combinedColor *= (1.0 - specCoeff);
    combinedColor += specular;

    FragColor = vec4(combinedColor, 1.0);
}
