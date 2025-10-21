#version 460 core

in vec3 a_position;
in vec3 a_normal;
in vec2 a_texcoord;

uniform mat4 u_modelMatrix;
uniform mat4 u_viewMatrix;
uniform mat4 u_projectionMatrix;

out vec3 out_normal;
out vec3 out_position;
out vec2 out_texcoord;

void main(void)
{ 
    gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * vec4(a_position, 1.0f);
    
    out_position = vec3(u_modelMatrix * vec4(a_position, 1.0f));
    out_normal = mat3(transpose(inverse(u_modelMatrix))) * a_normal;
    out_texcoord = a_texcoord * 2.0;
}

