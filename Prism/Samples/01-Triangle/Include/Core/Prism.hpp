#ifndef PRISM_HPP
#define PRISM_HPP

#include "Platform.hpp"

#if PLATFORM_WINDOWS
    #include "Win32-Window.hpp"
#endif

//! API Headers
#include "OGL.hpp"

#include <iostream>
#include <cstdlib>

class Prism
{
    private:
        Win32Window* window = nullptr;
        std::string windowTitle;
        int windowWidth, windowHeight;

    public:
        Prism();
        Prism(API api, std::string _windowTitle, int _windowWidth, int _windowHeight);
        ~Prism();

        void setWindowTitle(std::string _windowTitle);
        void setWindowWidth(int _windowWidth);
        void setWindowHeight(int _windowHeight);

        int exec();

};

#endif  // PRISM_HPP
