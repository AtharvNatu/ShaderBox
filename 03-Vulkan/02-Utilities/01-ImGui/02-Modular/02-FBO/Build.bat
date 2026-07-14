
@echo off

set API=Vulkan

set VULKAN_INCLUDE_PATH="C:\\VulkanSDK\\Vulkan\\Include"
set VULKAN_LIB_PATH="C:\\VulkanSDK\Vulkan\\Lib"
set VULKAN_BIN_PATH="C:\\VulkanSDK\\Vulkan\\Bin"

set SOURCE_PATH=Source
set INCLUDE_PATH=Include
set IMAGES_PATH=Assets\Images

set BIN_DIR=Bin

@REM SHADER RELATED
set SPV=0

set VERT_SHDR_1="Cube.vert"
set FRAG_SHDR_1="Cube.frag"

set VERT_SHDR_2="Teapot.vert"
set FRAG_SHDR_2="Teapot.frag"

set VERT_SHDR_3="Overlay.vert"
set FRAG_SHDR_3="Overlay.frag"

if not exist %BIN_DIR% mkdir %BIN_DIR%

cls

if exist %BIN_DIR%\*.obj del /q %BIN_DIR%\*.obj >nul 2>&1
if exist %BIN_DIR%\*.exe del /q %BIN_DIR%\*.res >nul 2>&1
if exist %BIN_DIR%\*.res del /q %BIN_DIR%\*.exe >nul 2>&1

echo ----------------------------------------------------------------------------------------------------------------
echo Compiling %API% and Win32 Source Code ...
echo ----------------------------------------------------------------------------------------------------------------
        
cl.exe  /c ^
        /EHsc ^
        /std:c++20 ^
        /I %VULKAN_INCLUDE_PATH% ^
        /I %VULKAN_INCLUDE_PATH%\glm ^
        /I %INCLUDE_PATH% ^
        %SOURCE_PATH%\Overlay\*.cpp ^
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
if %SPV%==1 (
    echo ----------------------------------------------------------------------------------------------------------------
    echo Compiling Shader Files To SPIR-V Binaries ...
    echo ----------------------------------------------------------------------------------------------------------------
    cd Shaders
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o %VERT_SHDR_1%.spv %VERT_SHDR_1%
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o %FRAG_SHDR_1%.spv %FRAG_SHDR_1%
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o %VERT_SHDR_2%.spv %VERT_SHDR_2%
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o %FRAG_SHDR_2%.spv %FRAG_SHDR_2%
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o %VERT_SHDR_3%.spv %VERT_SHDR_3%
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o %FRAG_SHDR_3%.spv %FRAG_SHDR_3%
    move %VERT_SHDR_1%.spv ../%BIN_DIR%
    move %FRAG_SHDR_1%.spv ../%BIN_DIR%
    move %VERT_SHDR_2%.spv ../%BIN_DIR%
    move %FRAG_SHDR_2%.spv ../%BIN_DIR%
    move %VERT_SHDR_3%.spv ../%BIN_DIR%
    move %FRAG_SHDR_3%.spv ../%BIN_DIR%
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
        user32.lib gdi32.lib^
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
cd %BIN_DIR%
Vk.exe
cd ..
