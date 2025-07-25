# ShaderBox
A repository having OpenGL, DirectX11/12, Vulkan, Metal samples with Prism rendering engine which abstracts all given APIs

## Source Code Organization ##

1. OpenGL - Contains OpenGL samples sub-divided into :
        a) Core concepts - Projection, 2D, 3D, Texture, Lights, Framebuffer Object etc.
        b) Utilities - Model loading, font rendering, Dear ImGUI
        c) Effects - Grass rendering, water rendering, fire effect etc
    All samples are programmed to run natively on specific operating systems.

2. D3D11 - TODO
3. Vulkan - TODO
4. D3D12 - TODO
5. Metal - TODO
6. Prism - In Progress

### Notes Regarding Samples in Core Concepts ###
 - For core concepts, every sample contains a Build file (.bat or .sh) with compiler driver commands. Under Prism engine, this will all be unified using CMake.
 - Along with this, all samples within core concepts have .cpp extension mainly to incorporate vmath or glm, but mostly use C-style procedural code.
   
### Currently Implemented Platforms for OpenGL ###

- [x] Windows 11/10 64-bit
- [ ] Linux
- [ ] macOS

### Requirements ###
 - OpenGL Extension Wrangler (GLEW) 2.1
 - vmath (vmath.h) in OpenGL (Referred from OpenGL RedBook)
 - FreeType (For font rendering)
 - Dear ImGUI
 - For Windows
    1) MSVC Toolchain
    2) Win32 SDK
 - For Linux
    1) GCC Toolchain
    2) X11 API (Based on Xorg Server)


### References ###
 - AstroMediComp Real-Time Rendering (RTR) Course
 - AstroMediComp Advanced Real-Time Rendering (ARTR) Course
 - OpenGL Programming Guide (Red Book), 8th Edition - Dave Shreiner, Graham Sellers, John Kessenich, Bill Licea-Kane
 - OpenGL Shading Language (Orange Book), 3rd Edition - Randi Rost, Bill Licea-Kane
 - OpenGL Superbible (Blue Book), 7th Edition - Graham Sellers, Richard Wright, Nicholas Haemel
 