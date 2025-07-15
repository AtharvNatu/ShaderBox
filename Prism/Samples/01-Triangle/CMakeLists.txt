cmake_minimum_required(VERSION 3.15)
project(Prism_Triangle)

# C++ Version Requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Resource File (For Windows)
if (WIN32)
    enable_language(RC)
    set(SOURCES OGL.cpp)
else()
    set(SOURCES OGL.cpp)
endif()

# GLEW
set(GLEW_DIR "C:/glew")
include_directories(${GLEW_DIR}/include)
link_directories(${GLEW_DIR}/lib/Release/x64)

# Add Executable
add_executable(OGL ${SOURCES})

set_target_properties(OGL PROPERTIES WIN32_EXECUTABLE TRUE)

# Link Libraries
target_link_libraries(OGL
    glew32
    user32
    gdi32
)