#ifndef TEXTURE_HPP
#define TEXTURE_HPP

#include "Common.hpp"

class Texture
{
    private:
        GLuint id;
        unsigned int width, height;
        GLint internalFormat;

    public:
        Texture(
            unsigned int _width, 
            unsigned int _height,
            GLint _internalFormat,
            GLenum format,
            GLenum type,
            GLint minFilter = GL_NEAREST,
            GLint magFilter = GL_NEAREST,
            GLint wrapR = GL_CLAMP_TO_BORDER,
            GLint wrapS = GL_CLAMP_TO_BORDER,
            const GLvoid* data = 0
        );
        ~Texture();

        bool loadPNG(GLuint* texture, const char* imageFile);
        void setWrappingParameters(GLint wrapR, GLint wrapS);
        void bindImage(GLuint unit, GLenum access, GLenum format) const;

};


#endif  // TEXTURE_HPP

