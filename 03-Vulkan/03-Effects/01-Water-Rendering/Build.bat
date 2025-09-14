
@echo off

set API=Vulkan

set VULKAN_INCLUDE_PATH=C:\VulkanSDK\Vulkan\Include
set VULKAN_LIB_PATH=C:\VulkanSDK\Vulkan\Lib
set VULKAN_BIN_PATH=C:\VulkanSDK\Vulkan\Bin

set IMGUI_PATH=ImGui
set IMGUI_BACKENDS=%IMGUI_PATH%\backends

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
echo Compiling %API% and Win32 Source Code ...
echo ----------------------------------------------------------------------------------------------------------------
cl.exe  /c ^
        /EHsc ^
        /std:c++17 ^
        /Fo%BIN_DIR%\ ^
        /I %VULKAN_INCLUDE_PATH% ^
        /I %VULKAN_INCLUDE_PATH%\glm ^
        /I %IMGUI_PATH% ^
        /I %IMGUI_BACKENDS% ^
        /I %INCLUDE_PATH% ^
        %SOURCE_PATH%\*.cpp ^
        %IMGUI_PATH%\imgui.cpp ^
        %IMGUI_PATH%\imgui_draw.cpp ^
        %IMGUI_PATH%\imgui_widgets.cpp ^
        %IMGUI_PATH%\imgui_tables.cpp ^
        %IMGUI_BACKENDS%\imgui_impl_win32.cpp ^
        %IMGUI_BACKENDS%\imgui_impl_vulkan.cpp

if errorlevel 1 (
        @echo:
        echo Compilation Failed !!!
        exit /b 1
)
)

@REM @echo:
@REM echo ----------------------------------------------------------------------------------------------------------------
@REM echo Compiling Resource Files ...
@REM echo ----------------------------------------------------------------------------------------------------------------
@REM rc.exe ^
@REM         /I %INCLUDE_PATH% ^
@REM         /I %IMAGES_PATH%^
@REM         /fo %BIN_DIR%\Vk.res ^
@REM         Assets\Vk.rc

@REM if errorlevel 1 (
@REM         @echo:
@REM         echo Resource Compilation Failed !!!
@REM         exit /b 1
@REM )

@REM @echo:
@REM if %SPV% == 1 (
@REM     echo ----------------------------------------------------------------------------------------------------------------
@REM     echo Compiling Shader Files To SPIR-V Binaries ...
@REM     echo ----------------------------------------------------------------------------------------------------------------
@REM     cd Shaders
@REM     %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Shader.vert.spv Shader.vert
@REM     %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Shader.frag.spv Shader.frag
@REM     move Shader.vert.spv ../Bin
@REM     move Shader.frag.spv ../Bin
@REM     cd ..
@REM     if errorlevel 1 (
@REM         @echo:
@REM         echo Shader Compilation Failed !!!
@REM         exit /b 1
@REM     )
@REM )

@REM @echo:
@REM echo ----------------------------------------------------------------------------------------------------------------
@REM echo Linking Libraries and Resources...
@REM echo Creating Executable...
@REM echo ----------------------------------------------------------------------------------------------------------------
@REM link.exe ^
@REM         /OUT:%BIN_DIR%\Vk.exe ^
@REM         %BIN_DIR%\*.obj ^
@REM         %BIN_DIR%\Vk.res ^
@REM         /LIBPATH:%VULKAN_LIB_PATH% user32.lib gdi32.lib libfftw3f-3.lib /SUBSYSTEM:WINDOWS

@REM if errorlevel 1 (
@REM         @echo:
@REM         echo Linking Failed !!!
@REM         exit /b 1
@REM )

@REM move /Y %BIN_DIR%\Vk.exe . >nul 2>&1

@REM @echo:
@REM echo ----------------------------------------------------------------------------------------------------------------
@REM echo Launching Application ...
@REM echo ----------------------------------------------------------------------------------------------------------------
@REM Vk.exe

@REM )

