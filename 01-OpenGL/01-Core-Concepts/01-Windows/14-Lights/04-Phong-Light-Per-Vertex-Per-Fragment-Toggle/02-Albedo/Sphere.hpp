#ifndef SPHERE_HPP
#define SPHERE_HPP

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #ifndef _USE_MATH_DEFINES
        #define _USE_MATH_DEFINES
    #endif
#endif

#include <cmath>

#include <iostream>
#include <vector>

const int CACHE_SIZE = 240;

class Sphere
{
    private:
        // Data Members
        std::vector<float> vertices;
        std::vector<float> texcoords;
        std::vector<float> normals;
        std::vector<unsigned int> indices;

        // Internal Member Functions
        void add_vertex(float x, float y, float z);
        void add_texcoord(float u, float v);
        void add_normal(float nx, float ny, float nz);
        void generate_sphere_data(double radius, int slices, int stacks);
        
    public:
        Sphere(double radius, int slices, int stacks);
        ~Sphere() = default;

        size_t get_number_of_vertices() const;
        size_t get_number_of_texcoords() const;
        size_t get_number_of_normals() const;
        size_t get_number_of_indices() const;

        const std::vector<float>& get_vertices() const;
        const std::vector<float>& get_texcoords() const;
        const std::vector<float>& get_normals() const;
        const std::vector<unsigned int>& get_indices() const;
};

#endif