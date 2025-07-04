# ShaderBox
A repository having OpenGL, DirectX11/12, Vulkan, Metal samples with Prism rendering engine which abstracts all given APIs

Notes regarding core concepts
 - For core concepts, every sample contains a Build file (.bat or .sh) with compiler driver commands. Under Prism engine, this will all be unified using CMake.
 - Along with this, all samples within core concepts have .cpp extension just for accomodation of vmath or glm

Requirements
 - OpenGL Extension Wrangler (GLEW) 2.1
 - vmath (vmath.h) in OpenGL (Referred from OpenGL RedBook)
 - For Windows - MSVC Toolchain, Win32 SDK
 - For Linux - GCC Toolchainl, X11 API
