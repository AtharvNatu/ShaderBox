#ifndef OGL_HPP
#define OGL_HPP

//! OpenGL Header Files
#include <GL/glew.h>
#include <GL/gl.h>

enum INIT_ERRORS
{
    CPF_ERROR = -1,
    SPF_ERROR = -2,
    WGL_CC_ERROR = -3,
    WGL_MC_ERROR = -4,
    GLEW_INIT_ERROR = -5,
    VS_COMPILE_ERROR = -6,
    TES_COMPILE_ERROR = -7,
    TCS_COMPILE_ERROR = -8,
    GS_COMPILE_ERROR = -9,
    FS_COMPILE_ERROR = -10,
    PROGRAM_LINK_ERROR = -11,
    MEM_ALLOC_FAILED = -12
};

enum ATTRIBUTES
{
    ATTRIBUTE_POSITION = 0,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEXTURE0
};

#endif