
@echo off

set API=Vulkan

set VULKAN_INCLUDE_PATH="C:\\VulkanSDK\\Vulkan\\Include"
set VULKAN_LIB_PATH="C:\\VulkanSDK\Vulkan\\Lib"

set CUDA_INCLUDE_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\include"
set CUDA_LIB_PATH="C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\lib\\x64"

set SPV=1

cls

if exist *.obj del *.obj
if exist *.exe del *.exe
if exist *.res del *.res

echo ----------------------------------------------------------------------------------------------------------------
echo Compiling %API% and Win32 Source Code ...
echo ----------------------------------------------------------------------------------------------------------------
nvcc.exe ^
        -c ^
        -w ^
        -I%CUDA_INCLUDE_PATH% ^
        -I%VULKAN_INCLUDE_PATH% ^
        -I%VULKAN_INCLUDE_PATH%\glm ^
        -Xcompiler="/EHsc" ^
        -Wno-deprecated-gpu-targets ^
        Vk.cu

if errorlevel 1 (
        @echo:
        echo Compilation Failed !!!
        exit /b 1
)

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Compiling Resource Files ...
echo ----------------------------------------------------------------------------------------------------------------
rc.exe Vk.rc

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
    C:\VulkanSDK\Vulkan\Bin\glslangValidator.exe -V -H -o Shader.vert.spv Shader.vert
    C:\VulkanSDK\Vulkan\Bin\glslangValidator.exe -V -H -o Shader.frag.spv Shader.frag
    move Shader.vert.spv ../
    move Shader.frag.spv ../
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
        Vk.obj ^
        Vk.res ^
        /LIBPATH:%VULKAN_LIB_PATH% ^
        /LIBPATH:%CUDA_LIB_PATH% ^
        user32.lib gdi32.lib cudart.lib ^
        /SUBSYSTEM:WINDOWS

if errorlevel 1 (
        @echo:
        echo Linking Failed !!!
        exit /b 1
)

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Launching Application ...
echo ----------------------------------------------------------------------------------------------------------------
Vk.exe


