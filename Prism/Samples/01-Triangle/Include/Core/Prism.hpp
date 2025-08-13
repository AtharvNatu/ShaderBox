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
        #if PLATFORM_WINDOWS
            Win32Window* window = nullptr;
        #endif
        
        API api;

    public:
        Prism();
        Prism(API _api, std::string _windowTitle, int _windowWidth, int _windowHeight);
        ~Prism();

        API getAPI() const;
        void setAPI(API api);
        
        std::string getWindowTitle() const;
        void setWindowTitle(std::string _windowTitle);

        int getWindowWidth() const;
        void setWindowWidth(int _windowWidth);

        int getWindowHeight() const;
        void setWindowHeight(int _windowHeight);

        INIT_STATUS initialize();
        void display();
        void update();
        void uninitialize();

        int exec();

};

#endif  // PRISM_HPP
