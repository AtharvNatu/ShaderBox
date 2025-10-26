#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 out_color;
layout(location = 1) in vec3 out_normal;
layout(location = 2) in vec3 out_light_direction;
layout(location = 3) in vec3 out_camera_direction;

layout(location = 0) out vec4 FragColor;


void main(void)
{
    // Code
    vec3 halfDir = normalize(out_light_direction + out_camera_direction);
    float specular = pow(max(dot(out_normal, halfDir), 0.0), 10.0);
    float diffuse = dot(out_light_direction, out_normal);

    const vec3 lightColor = 0.4 * normalize(vec3(253, 251, 211));

    FragColor = vec4(out_color * diffuse + lightColor * specular, 1.0);
}
