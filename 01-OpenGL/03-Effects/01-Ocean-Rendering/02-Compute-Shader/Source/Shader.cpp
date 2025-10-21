#include "Shader.hpp"

Shader::Shader()
{
    logger = Logger::getInstance("Ocean.log");
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

bool Shader::linkShaderProgramObject(GLuint shaderProgramObject)
{
    glLinkProgram(shaderProgramObject);

    GLint status = 0;
    GLint infoLogLength = 0;
    GLchar* szLog = NULL;

    glGetProgramiv(shaderProgramObject, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramiv(shaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength);
            if (szLog == NULL)
            {
                logger->printLog("ERROR : %s() => Failed to allocate memory to szLog for Shader Program Log !!!\n", __func__);
                return false;
            }
            else
            {
                GLsizei logSize;
                glGetProgramInfoLog(shaderProgramObject, GL_INFO_LOG_LENGTH, &logSize, szLog);
                logger->printLog("ERROR : Shader Program Link Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return false;
            }
        }
    }

    return true;
}

void Shader::uninitializeShaders(GLuint shaderProgramObject)
{
    // Code
    if (shaderProgramObject)
    {
        glUseProgram(shaderProgramObject);
        {
            GLsizei numAttachedShaders;
            glGetProgramiv(shaderProgramObject, GL_ATTACHED_SHADERS, &numAttachedShaders);
            
            GLuint* shaderObjects = NULL;
            shaderObjects = (GLuint*)malloc(numAttachedShaders * sizeof(GLuint));
            if (shaderObjects == NULL)
            {
                logger->printLog("ERROR : %s() => Failed to allocate memory to shaderObjects for Shader Program Log !!!\n", __func__);
                return;
            }

            glGetAttachedShaders(shaderProgramObject, numAttachedShaders, &numAttachedShaders, shaderObjects);

            for (GLsizei i = 0; i < numAttachedShaders; i++)
            {
                glDetachShader(shaderProgramObject, shaderObjects[i]);
                glDeleteShader(shaderObjects[i]);
                shaderObjects[i] = 0;
            }
            free(shaderObjects);
            shaderObjects = NULL;
        }
        glUseProgram(0);
        glDeleteProgram(shaderProgramObject);
        shaderProgramObject = 0;
    }
}

Shader::~Shader()
{
    if (logger)
    {
        logger->deleteInstance();
        logger = nullptr;
    }

}
