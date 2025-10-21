#include "Shader.hpp"

Shader::Shader()
{
    logger = Logger::getInstance("Ocean.log");
}

GLuint Shader::createShaderObject(SHADER_TYPE shaderType, std::string filePath)
{
    std::string strShaderType;

    switch(shaderType)
    {
        case VERTEX:
            shaderObject = glCreateShader(GL_VERTEX_SHADER);
            strShaderType = "Vertex";
        break;
        case GEOMETRY:
            shaderObject = glCreateShader(GL_GEOMETRY_SHADER);
            strShaderType = "Geometry";
        break;
        case TESSELLATION_CONTROL:
            shaderObject = glCreateShader(GL_TESS_CONTROL_SHADER);
            strShaderType = "Tessellation Control";
        break;
        case TESSELLATION_EVALUATION:
            shaderObject = glCreateShader(GL_TESS_EVALUATION_SHADER);
            strShaderType = "Tessellation Evaluation";
        break;
        case FRAGMENT:
            shaderObject = glCreateShader(GL_FRAGMENT_SHADER);
            strShaderType = "Fragment";
        break;
        case COMPUTE:
            shaderObject = glCreateShader(GL_COMPUTE_SHADER);
            strShaderType = "Compute";
        break;

        default:
        break;
    }

    const GLchar* shaderSourceCode = getShaderSource(filePath);
    glShaderSource(shaderObject, 1, (const GLchar**)&shaderSourceCode, NULL);
    glCompileShader(shaderObject);

    GLint status = 0;
    GLint infoLogLength = 0;
    GLchar* szLog = NULL;

    glGetShaderiv(shaderObject, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(shaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength * sizeof(GLchar));
            if (szLog == NULL)
            {
                logger->printLog("ERROR : %s() => Failed to allocate memory to szLog for %s Shader Log !!!\n", __func__, strShaderType.c_str());
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetShaderInfoLog(shaderObject, infoLogLength, &logSize, szLog);
                logger->printLog("ERROR : %s Shader Compilation Log : %s\n", strShaderType.c_str(), szLog);
                free(szLog);
                szLog = NULL;
                return FS_COMPILE_ERROR;
            }
        }
    }

    free((void*)shaderSourceCode);
    shaderSourceCode = NULL;

    return shaderObject;
}

const GLchar* Shader::getShaderSource(std::string filePath)
{
    // Code
    FILE* shaderFile = fopen(filePath.c_str(), "rb");
    if (shaderFile == NULL)
    {
        logger->printLog("ERROR : Failed To Open Shader File :  %s !!!\n", filePath.c_str());
        return NULL;
    }

    fseek(shaderFile, 0L, SEEK_END);
    long size = ftell(shaderFile);
    if (size == 0)
    {
        logger->printLog("ERROR : Empty Shader File :  %s !!!\n", filePath.c_str());
        if (shaderFile)
        {
            fclose(shaderFile);
            shaderFile = NULL;
        }

        return NULL;
    }
    fseek(shaderFile, 0L, SEEK_SET);

    GLchar* shaderData = (char*)malloc(size + 1);
    if (shaderData == NULL)
    {
        logger->printLog("ERROR : Failed To Allocate Memory To Shader Data !!!\n");

        if (shaderFile)
        {
            fclose(shaderFile);
            shaderFile = NULL;
        }

        return NULL;
    }

    size_t retVal = fread(shaderData, 1, size, shaderFile);
    if (retVal != (size_t)size)
    {
        logger->printLog("ERROR : Failed To Read Data From Shader Source File :  %s !!!\n", filePath.c_str());
        free(shaderData);
        shaderData = NULL;

        if (shaderFile)
        {
            fclose(shaderFile);
            shaderFile = NULL;
        }
        
        return NULL;
    }

    if (shaderFile)
    {
        fclose(shaderFile);
        shaderFile = NULL;
    }

    shaderData[size] = '\0';

    return (const GLchar*)shaderData;
}

Shader::~Shader()
{
    if (logger)
    {
        logger->deleteInstance();
        logger = nullptr;
    }

}
