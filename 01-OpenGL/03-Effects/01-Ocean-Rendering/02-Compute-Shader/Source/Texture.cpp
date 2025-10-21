#include "Texture.hpp"

//! STB Header For PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Texture::Texture(unsigned int _width, 
    unsigned int _height,
    GLint _internalFormat,
    GLenum format,
    GLenum type,
    GLint minFilter,
    GLint magFilter,
    GLint wrapR,
    GLint wrapS,
    const GLvoid* data
) : width(_width), height(_height), internalFormat(_internalFormat)
{
    // Code
    glGenTextures(1, &id);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, id);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, minFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, magFilter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, wrapR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        internalFormat,
        width,
        height,
        0,
        format,
        type,
        data
    );

    glBindTexture(GL_TEXTURE_2D, 0);

}

bool Texture::loadPNG(GLuint* texture, const char* imageFile)
{
    // Variable Declarations
    int width, height;
    int num_channels;
    unsigned char* image = NULL;

    // Code
    image = stbi_load(
        imageFile,
        &width,
        &height,
        &num_channels,
        STBI_rgb_alpha
    );
    if (image == NULL)
        return false;
    
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glGenTextures(1, texture);
    glBindTexture(GL_TEXTURE_2D, *texture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        if (num_channels == 3)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGB,
                width,
                height,
                0,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                image
            );
        else if (num_channels == 4)
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA,
                width,
                height,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                image
            );
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(image);
    image = NULL;

    return false;
}

void Texture::setWrappingParameters(GLint wrapR, GLint wrapS)
{
    // Code
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, wrapR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrapS);
}

void Texture::bindImage(GLuint unit, GLenum access, GLenum format) const
{
    glBindImageTexture(unit, id, 0, GL_FALSE, 0, access, format);
}

Texture::~Texture()
{
    glDeleteTextures(1, &id);
    id = 0;
}
