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
    
    public:
        Win32Window(const char* windowTitle, int windowWidth, int windowHeight);
        ~Win32Window();

        BOOL initialize(int windowWidth, int windowHeight);
        void uninitialize();

        int render();

        void toggleFullScreen();

        static LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
};

#endif  // WIN32_WINDOW_HPP
