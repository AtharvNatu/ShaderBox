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
        Logger* logger = nullptr;
        GLuint shaderObject;
        
        const GLchar* getShaderSource(std::string filePath);

    public:
        Shader();
        ~Shader();

        GLuint createShaderObject(SHADER_TYPE shaderType, std::string filePath);
        bool linkShaderProgramObject(GLuint shaderProgramObject);
        void uninitializeShaders(GLuint shaderProgramObject);
};



#endif // SHADER_HPP