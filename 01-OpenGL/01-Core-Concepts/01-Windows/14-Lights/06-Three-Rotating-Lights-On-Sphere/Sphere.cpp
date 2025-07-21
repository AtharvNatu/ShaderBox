#include "Sphere.hpp"

Sphere::Sphere(double radius, int slices, int stacks)
{
    // Code
    generate_sphere_data(radius, slices, stacks);
}

void Sphere::add_vertex(float x, float y, float z)
{
    // Code
   vertices.push_back(x);
   vertices.push_back(y);
   vertices.push_back(z);
}

void Sphere::add_texcoord(float u, float v)
{
    // Code
    texcoords.push_back(u);
    texcoords.push_back(v);
}

void Sphere::add_normal(float nx, float ny, float nz)
{
    normals.push_back(nx);
    normals.push_back(ny);
    normals.push_back(nz);
}

size_t Sphere::get_number_of_vertices() const
{
    return vertices.size();
}

size_t Sphere::get_number_of_texcoords() const
{
    return texcoords.size();
}

size_t Sphere::get_number_of_normals() const
{
    return normals.size();
}

size_t Sphere::get_number_of_indices() const
{
    return indices.size();
}

const std::vector<float>& Sphere::get_vertices() const
{
    return vertices;
}

const std::vector<float>& Sphere::get_texcoords() const
{
    return texcoords;
}

const std::vector<float>& Sphere::get_normals() const
{
    return normals;
}

const std::vector<unsigned int>& Sphere::get_indices() const
{
    return indices;
}

void Sphere::generate_sphere_data(double radius, int slices, int stacks)
{
    // Variable Declarations
    float sin_cache[CACHE_SIZE + 1];
    float cos_cache[CACHE_SIZE + 1];

    // Code

    //* Clamp
    if (slices >= CACHE_SIZE)
        slices = CACHE_SIZE - 1;

    if (stacks >= CACHE_SIZE)
        stacks = CACHE_SIZE - 1;

    if (slices < 2 || stacks < 1 || radius <= 0.0)
        return;

    //* Cache sin/cos for slices
    for (int i = 0; i <= slices; i++)
    {
        float angle = 2.0f * static_cast<float>(M_PI) * i / slices;
        sin_cache[i] = sinf(angle);
        cos_cache[i] = cosf(angle);
    }

    //* Generate Vertices, Texcoords and Normals
    for (int stack = 0; stack <= stacks; stack++)
    {
        float phi = static_cast<float>(M_PI) * stack / stacks;
        float sin_phi = sinf(phi);
        float cos_phi = cosf(phi);

        for (int slice = 0; slice <= slices; slice++)
        {
            float x = sin_phi * cos_cache[slice];
            float y = sin_phi * sin_cache[slice];
            float z = cos_phi;

            add_vertex(
                radius * x, 
                radius * y, 
                radius * z
            );

            add_texcoord(
                (float)slice / slices,
                1.0f - (float)stack / stacks
            );

            add_normal(x, y ,z);
        }
    }

    //* Generate Indices
    for (int stack = 0; stack < stacks; stack++)
    {
        for (int slice = 0; slice < slices; slice++)
        {
            int first = (stack) * (slices + 1) + slice;
            int second = (stack + 1) * (slices + 1) + slice;

            //* Two triangles per quad
            indices.push_back(first);
            indices.push_back(second);
            indices.push_back(first + 1);

            indices.push_back(second);
            indices.push_back(second + 1);
            indices.push_back(first + 1);
            
        }
    }
}
