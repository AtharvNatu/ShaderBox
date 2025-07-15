#ifndef SPHERE_HPP
#define SPHERE_HPP

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #ifndef _USE_MATH_DEFINES
        #define _USE_MATH_DEFINES
    #endif
#endif

#include <cmath>

const int MAX_ARRAY_SIZE = 100000;
const int CACHE_SIZE = 240;

class Sphere
{
    private:
        // Data Members
        float vertices[MAX_ARRAY_SIZE];
        float texcoords[MAX_ARRAY_SIZE];
        float normals[MAX_ARRAY_SIZE];

        int vertex_pointer;
        int texcoord_pointer;
        int normal_pointer;

        // Internal Member Functions
        void add_vertex(float x, float y, float z);
        void add_texcoord(float a, float b);
        void add_normal(float p, float q, float r);
        void generate_sphere_data(double radius, int slices, int stacks);
        
    public:
        Sphere(double radius, int slices, int stacks);
        ~Sphere();

        int get_number_of_vertices(void);
        int get_number_of_texcoords(void);
        int get_number_of_normals(void);

        float* get_vertices(void);
        float* get_texcoords(void);
        float* get_normals(void);
};

#endif
