
@echo off

set API=Vulkan

set VULKAN_INCLUDE_PATH="C:\\VulkanSDK\\Vulkan\\Include"
set VULKAN_LIB_PATH="C:\\VulkanSDK\Vulkan\\Lib"
set VULKAN_BIN_PATH="C:\\VulkanSDK\\Vulkan\\Bin"

set CUDA_INCLUDE_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include"
set CUDA_LIB_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\lib\\x64"

set SOURCE_PATH=Source
set INCLUDE_PATH=Include
set IMAGES_PATH=Assets\Images

set BIN_DIR=Bin

set SPV=1
set DEBUG=1

if not exist %BIN_DIR% mkdir %BIN_DIR%

cls

if exist %BIN_DIR%\*.obj del /q %BIN_DIR%\*.obj >nul 2>&1
if exist %BIN_DIR%\*.exe del /q %BIN_DIR%\*.res >nul 2>&1
if exist %BIN_DIR%\*.res del /q %BIN_DIR%\*.exe >nul 2>&1

if %DEBUG% == 1 (

echo ***** DEBUG MODE *****
echo ----------------------------------------------------------------------------------------------------------------
echo Compiling %API%, CUDA and Win32 Source Code ...
echo ----------------------------------------------------------------------------------------------------------------
        nvcc ^
        -std=c++17 ^
        -c ^
        -w ^
        -I%CUDA_INCLUDE_PATH% ^
        -I%VULKAN_INCLUDE_PATH% ^
        -I%VULKAN_INCLUDE_PATH%\glm ^
        -I%INCLUDE_PATH% ^
        -Xcompiler="/EHsc" ^
        -Wno-deprecated-gpu-targets ^
        %SOURCE_PATH%\Ocean.cu ^
        %SOURCE_PATH%\Camera.cpp ^
        %SOURCE_PATH%\Vk.cpp

if errorlevel 1 (
        @echo:
        echo Compilation Failed !!!
        exit /b 1
)

move *.obj %BIN_DIR% >nul 2>&1

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Compiling Resource Files ...
echo ----------------------------------------------------------------------------------------------------------------
rc.exe ^
        /I %INCLUDE_PATH% ^
        /I %IMAGES_PATH%^
        /fo %BIN_DIR%\Vk.res ^
        Assets\Vk.rc

if errorlevel 1 (
        @echo:
        echo Resource Compilation Failed !!!
        exit /b 1
)

@echo:
if %SPV% == 1 (
    echo ----------------------------------------------------------------------------------------------------------------
    echo Compiling Shader Files To SPIR-V Binaries ...
    echo ----------------------------------------------------------------------------------------------------------------
    cd Shaders
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Ocean.vert.spv Ocean.vert
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Ocean.frag.spv Ocean.frag
    move Ocean.vert.spv ../Bin
    move Ocean.frag.spv ../Bin
    cd ..
    if errorlevel 1 (
        @echo:
        echo Shader Compilation Failed !!!
        exit /b 1
    )
)

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Linking Libraries and Resources...
echo Creating Executable...
echo ----------------------------------------------------------------------------------------------------------------
link.exe ^
        /OUT:%BIN_DIR%\Vk.exe ^
        %BIN_DIR%\*.obj ^
        %BIN_DIR%\Vk.res ^
        /LIBPATH:%VULKAN_LIB_PATH% ^
        /LIBPATH:%CUDA_LIB_PATH% ^
        user32.lib gdi32.lib cudart.lib cufft.lib ^
        /SUBSYSTEM:WINDOWS

if errorlevel 1 (
        @echo:
        echo Linking Failed !!!
        exit /b 1
)

move /Y %BIN_DIR%\Vk.exe . >nul 2>&1

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Launching Application ...
echo ----------------------------------------------------------------------------------------------------------------
Vk.exe

)

