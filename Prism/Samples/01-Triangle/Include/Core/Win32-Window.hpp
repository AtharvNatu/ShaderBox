#ifndef WIN32_WINDOW_HPP
#define WIN32_WINDOW_HPP

#include <Windows.h>
#include <shellapi.h>
#include <iostream>

#include "Win32-Resource.hpp"
#include "Logger.hpp"

class Win32Window
{
    private:
        HWND hwnd = NULL;
        HDC hdc = NULL;
        HGLRC hrc = NULL;
        HINSTANCE hInstance = NULL;

        LPCWSTR lpwstrTitle;
        std::string strTitle;
        int windowWidth;
        int windowHeight;
    
    public:
        Win32Window(std::string _windowTitle, int _windowWidth, int _windowHeight);
        Win32Window();
        ~Win32Window();

        std::string getWindowTitle() const;
        void setWindowTitle(std::string _windowTitle);

        int getWindowWidth() const;
        void setWindowWidth(int _windowWidth);

        int getWindowHeight() const;
        void setWindowHeight(int _windowHeight);

        void initialize();
        void uninitialize();

        int render();

        void toggleFullScreen();

        static LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
};

#endif  // WIN32_WINDOW_HPP
