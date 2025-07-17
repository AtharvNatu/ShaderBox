#ifndef PLATFORM_HPP
#define PLATFORM_HPP

enum class Platform
{
    Windows = 1,
    Linux = 2,
    macOS = 3
};

enum class API
{
    OpenGL = 1,
    Vulkan = 2,
    DX11 = 3,
    DX12 = 4,
    Metal = 5,
    WebGPU = 6
};

enum class SDK
{
    Win32 = 1,
    X11 = 2,
    Cocoa = 3
};

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    #define PLATFORM_WINDOWS    1
#elif defined(__linux)
    #define PLATFORM_LINUX      1
#elif defined(__APPLE__)
    #define PLATFORM_MACOS      1
#endif


#endif  // PLATFORM_HPP