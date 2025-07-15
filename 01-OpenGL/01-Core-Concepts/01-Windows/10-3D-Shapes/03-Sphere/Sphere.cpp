#include "Sphere.hpp"

Sphere::Sphere(double radius, int slices, int stacks)
{
    // Code
    generate_sphere_data(radius, slices, stacks);
}

void Sphere::add_texcoord(float a, float b)
{
    // Code
    texcoords[texcoord_pointer++] = a;
    texcoords[texcoord_pointer++] = b;
}

void Sphere::add_vertex(float x, float y, float z)
{
    // Code
    vertices[vertex_pointer++] = x;
    vertices[vertex_pointer++] = y;
    vertices[vertex_pointer++] = z;
}

void Sphere::add_normal(float p, float q, float r)
{
    normals[normal_pointer++] = p;
    normals[normal_pointer++] = q;
    normals[normal_pointer++] = r;
}

int Sphere::get_number_of_vertices()
{
    return vertex_pointer / 3;
}

int Sphere::get_number_of_texcoords()
{
    return texcoord_pointer;
}

int Sphere::get_number_of_normals()
{
    return normal_pointer;
}

float* Sphere::get_vertices()
{
    return vertices;
}

float* Sphere::get_texcoords()
{
    return texcoords;
}

float* Sphere::get_normals()
{
    return normals;
}

void Sphere::generate_sphere_data(double radius, int slices, int stacks)
{
    // Variable Declarations
    float sin_cache_1a[CACHE_SIZE], sin_cache_2a[CACHE_SIZE], sin_cache_3a[CACHE_SIZE];
    float sin_cache_1b[CACHE_SIZE], sin_cache_2b[CACHE_SIZE], sin_cache_3b[CACHE_SIZE];
    
    float cos_cache_1a[CACHE_SIZE], cos_cache_2a[CACHE_SIZE], cos_cache_3a[CACHE_SIZE];
    float cos_cache_1b[CACHE_SIZE], cos_cache_2b[CACHE_SIZE], cos_cache_3b[CACHE_SIZE];

    float z_low, z_high;
    float sin_tmp_1 = 0.0f, sin_tmp_2 = 0.0f, sin_tmp_3 = 0.0f, sin_tmp_4 = 0.0f;
    float cos_tmp_1 = 0.0f, cos_tmp_2 = 0.0f;

    int start, finish;

    // Code
    if (slices >= CACHE_SIZE)
        slices = CACHE_SIZE - 1;

    if (stacks >= CACHE_SIZE)
        stacks = CACHE_SIZE - 1;
    
    if (slices < 2 || stacks < 1 || radius < 0.0)
        return;

    for (int i = 0; i < slices; i++)
    {
        float angle = 2 * (float)M_PI * i / slices;
        sin_cache_1a[i] = (float)sin(angle);
        cos_cache_1a[i] = (float)cos(angle);
        sin_cache_2a[i] = sin_cache_1a[i];
        cos_cache_2a[i] = cos_cache_1a[i];
    }

    for (int j = 0; j <= stacks; j++)
    {
        float angle = (float)M_PI * j / stacks;
        sin_cache_2b[j] = (float)sin(angle);
        cos_cache_2b[j] = (float)cos(angle);
        sin_cache_1b[j] = radius * (float)sin(angle);
        cos_cache_1b[j] = radius * (float)cos(angle);
    }

    sin_cache_1b[0] = 0;
    sin_cache_1b[stacks] = 0;

    sin_cache_1a[slices] = sin_cache_1a[0];
    cos_cache_1a[slices] = cos_cache_1a[0];
    sin_cache_2a[slices] = sin_cache_2a[0];
    cos_cache_1a[slices] = cos_cache_2a[0];

    start = 0;
    finish = stacks;

    for (int i = start; i < finish; i++)
    {
        z_low = cos_cache_1b[i];
        z_high = cos_cache_1b[i + 1];

        sin_tmp_1 = sin_cache_1b[i];
        sin_tmp_2 = sin_cache_1b[i + 1];
        sin_tmp_3 = sin_cache_2b[i + 1];
        sin_tmp_4 = sin_cache_2b[i];

        cos_tmp_1 = cos_cache_2b[i + 1];
        cos_tmp_2 = cos_cache_2b[i];

        for (int j = 0; j <= slices; j++)
        {
            add_vertex(
                sin_tmp_2 * sin_cache_1a[j], 
                sin_tmp_2 * cos_cache_1a[j], 
                z_high
            );

            add_vertex(
                sin_tmp_1 * sin_cache_1a[j], 
                sin_tmp_1 * cos_cache_1a[j], 
                z_low
            );

            add_normal(
                sin_cache_2a[j] * sin_tmp_3,
                cos_cache_2a[j] * sin_tmp_3,
                cos_tmp_1
            );

            add_normal(
                sin_cache_2a[j] * sin_tmp_4,
                cos_cache_2a[j] * sin_tmp_4,
                cos_tmp_2
            );

            add_texcoord(
                1 - ((float)j / slices),
                1 - ((float)(i + 1) / stacks)
            );

            add_texcoord(
                1 - ((float)j / slices),
                1 - ((float)i / stacks)
            );
        }
    }
}

Sphere::~Sphere()
{
    // Code
}
