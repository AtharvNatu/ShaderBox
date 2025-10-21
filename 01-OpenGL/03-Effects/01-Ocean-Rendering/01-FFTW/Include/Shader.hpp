#ifndef SHADER_HPP
#define SHADER_HPP

#include <cstdio>
#include <cstdlib>
#include <string>

#include "Logger.hpp"
#include "Common.hpp"

enum SHADER_TYPE
{
    VERTEX,
    GEOMETRY,
    TESSELLATION_CONTROL,
    TESSELLATION_EVALUATION,
    FRAGMENT,
    COMPUTE
};

class Shader
{
    private:
        GLuint shaderObject;
        const GLchar* getShaderSource(std::string filePath);
        Logger* logger = nullptr;

    public:
        Shader();
        ~Shader();
        GLuint createShaderObject(SHADER_TYPE shaderType, std::string filePath);
};



#endif // SHADER_HPP