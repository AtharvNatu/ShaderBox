#version 460 core

layout (points) in;
layout (triangle_strip, max_vertices = 36) out;

//! OUT
layout(location = 0) out vec2 out_texcoord;
layout(location = 1) out float colorVariation;

layout(set = 0, binding = 0) uniform UBO 
{ 
    mat4 modelMatrix;
    mat4 projectionMatrix;
    float time;	
	float windStrength;
} ubo;

layout(set = 0, binding = 1) uniform sampler2D uWindSampler;

// Variable Declarations
float grassSize;
int drawRandomGrass = 0;

// Constants
#define c_min_size 0.4f
#define PI 3.141592653589793
#define NUM_OCTAVES 5

// Utility Functions
//-----------------------------------------------------------------------

mat4 rotationX(in float angle)
{
	return mat4(	1.0,  0,           0,             0,
					0,	  cos(angle),  -sin(angle),   0,
					0,    sin(angle),  cos(angle),    0,
					0,    0,           0,             1);
}

mat4 rotationY(in float angle)
{
	return mat4(	cos(angle),  	0,          sin(angle),     0,
					0,	  			1.0,  		0,   			0,
					-sin(angle),    0,  		cos(angle),    	0,
					0,    			0,          0,             	1);
}

mat4 rotationZ(in float angle)
{
	return mat4(	cos(angle),  	-sin(angle),	0,	 0,
					sin(angle),	  	cos(angle),  	0,   0,
					0,    			0,  			1,   0,
					0,    			0,          	0,   1);
}

float random(vec2 st)
{
	return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
}

float noise(in vec2 st)
{
	vec2 i = floor(st);
	vec2 f = fract(st);

	// 4 corners in 2D of a tile
	float a = random(i);
	float b = random(i + vec2(1.0, 0.0));
	float c = random(i + vec2(0.0, 1.0));
	float d = random(i + vec2(1.0, 1.0));

	// Cubic Hermine Curve
	vec2 u = f * f * (3.0 - 2.0 * f);

	// Mix 4 corners' Percentages
	return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float fbm(in vec2 st)
{
	float v = 0.0;
	float a = 0.5;

	vec2 shift = vec2(100.0);

	// Rotate To Reduce Axial Bias
	mat2 rotate = mat2(cos(0.5), sin(0.5), -sin(0.5), cos(0.5));

	for (int i = 0; i < NUM_OCTAVES; i++)
	{
		v = v + a * noise(st);
		st = rotate * st * 2.0 + shift;
		a = a * 0.5;
	}

	return v;
}

//-----------------------------------------------------------------------
void createQuad(vec3 basePosition, mat4 crossModel)
{
	// Variable Declarations
	vec4 vertices[4];
	vec2 texcoords[4];
	vec4 vertexOffset = vec4(0.0);

	// Code
	vertices[0] = vec4(-0.25, 0.0, 0.0, 0.0);	// Bottom Left
	vertices[1] = vec4(0.25, 0.0, 0.0, 0.0);	// Bottom Right
	vertices[2] = vec4(-0.25, 0.5, 0.0, 0.0);	// Top Left
	vertices[3] = vec4(0.25, 0.5, 0.0, 0.0);	// Top Right

	texcoords[0] = vec2(0.0, 0.0);	// Bottom Left
	texcoords[1] = vec2(1.0, 0.0);	// Bottom Right
	texcoords[2] = vec2(0.0, 1.0);	// Top Left
	texcoords[3] = vec2(1.0, 1.0);	// Top Right

	vec4 worldSpacePosition = ubo.modelMatrix * vec4(basePosition, 1.0);

	// Wind
	vec2 windDirection = vec2(1.0, 1.0);
	float windStrength = ubo.windStrength;
	vec2 uv = (basePosition.xz / 10.0) + windDirection * windStrength * ubo.time;
	uv.x = mod(uv.x, 1.0);
	uv.y = mod(uv.y, 1.0);

	vec4 wind = texture(uWindSampler, uv);
	mat4 modelWind = rotationX(wind.x * PI * 0.75f - PI * 0.25f) * rotationZ(wind.y * PI * 0.75f - PI * 0.25f);
	
	// Random Y-Axis Rotation
	mat4 modelRandomYRotation = rotationY(random(basePosition.xz) * PI);

	mat4 modelWindMatrix = modelWind;

	// Billboard Creation Loop
	for (int i = 0; i < 4; i++)
	{
		if (drawRandomGrass == 1)
		{
			vertexOffset = modelWindMatrix * modelRandomYRotation * crossModel * (vertices[i] * grassSize);
		}
		else if (drawRandomGrass == 0)
		{
			vertexOffset = modelWindMatrix * modelRandomYRotation * crossModel * vertices[i];
		}
		
		// Apply model matrix after all local transformations
    	vec4 worldPosition = ubo.modelMatrix * (vec4(basePosition, 1.0) + vertexOffset);
		gl_Position = ubo.projectionMatrix * worldPosition;
		out_texcoord = texcoords[i];
        colorVariation = fbm(gl_in[0].gl_Position.xz);
		EmitVertex();
	}

	EndPrimitive();
}

void createGrass(int numberOfQuads)
{
	// Variable Declarations
	mat4 model0 = mat4(1.0);
	mat4 model45 = rotationY(radians(45));
	mat4 modelm45 = rotationY(-radians(45));

	createQuad(gl_in[0].gl_Position.xyz, model0);
	createQuad(gl_in[0].gl_Position.xyz, model45);
	createQuad(gl_in[0].gl_Position.xyz, modelm45);
}

void main(void)
{
	// Variable Declarations
	int details = 3;

	// Random Grass Size
	if (drawRandomGrass == 1)
		grassSize = random(gl_in[0].gl_Position.xz) * (1.0 - c_min_size) + c_min_size;
	
	// Grass Creation
	createGrass(details);
} 



