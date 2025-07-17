cmake_minimum_required(VERSION 3.15)
project(Prism_Triangle)

# C++ Version Requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



# Resource File (For Windows)
if (WIN32)
    enable_language(RC)
    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
else()
    file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
endif()

# GLEW
set(GLEW_DIR "C:/glew")
include_directories(${GLEW_DIR}/include)
link_directories(${GLEW_DIR}/lib/Release/x64)

# Include
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# Add Executable
add_executable(OGL ${SOURCES})

if (WIN32)
    set_target_properties(OGL PROPERTIES WIN32_EXECUTABLE TRUE)
    set_target_properties(${OGL} PROPERTIES LINK_FLAGS "/ENTRY:mainCRTStartup")
endif()

# Link Libraries
target_link_libraries(OGL
    glew32
    user32
    gdi32
)