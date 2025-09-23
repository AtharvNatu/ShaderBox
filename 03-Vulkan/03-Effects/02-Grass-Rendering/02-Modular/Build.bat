
@echo off

set API=Vulkan

set VULKAN_INCLUDE_PATH=C:\VulkanSDK\Vulkan\Include
set VULKAN_LIB_PATH=C:\VulkanSDK\Vulkan\Lib
set VULKAN_BIN_PATH=C:\VulkanSDK\Vulkan\Bin

set SOURCE_PATH=Source
set INCLUDE_PATH=Include
set IMAGES_PATH=Assets\Images

set BIN_DIR=Bin

set SPV=0
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
        /fsanitize=address ^
        /MDd ^
        /Zi ^
        /Od ^
        /Fo%BIN_DIR%\ ^
        /I %VULKAN_INCLUDE_PATH% ^
        /I %VULKAN_INCLUDE_PATH%\glm ^
        /I %IMGUI_PATH% ^
        /I %IMGUI_BACKENDS% ^
        /I %INCLUDE_PATH% ^
        %SOURCE_PATH%\*.cpp

if errorlevel 1 (
        @echo:
        echo Compilation Failed !!!
        exit /b 1
)

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
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Grass.vert.spv Grass.vert
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Grass.geom.spv Grass.geom
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Grass.frag.spv Grass.frag
    move Grass.vert.spv ../Bin
    move Grass.geom.spv ../Bin
    move Grass.frag.spv ../Bin
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
        /INCREMENTAL:NO ^
        /DEBUG ^
        /OUT:%BIN_DIR%\Vk.exe ^
        %BIN_DIR%\*.obj ^
        %BIN_DIR%\Vk.res ^
        /LIBPATH:%VULKAN_LIB_PATH% ^
        user32.lib ^
        gdi32.lib ^
        "clang_rt.asan_dynamic-x86_64.lib" ^
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

) else (


echo ***** RELEASE MODE *****
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
        %SOURCE_PATH%\*.cpp

if errorlevel 1 (
        @echo:
        echo Compilation Failed !!!
        exit /b 1
)

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
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Grass.vert.spv Grass.vert
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Grass.geom.spv Grass.geom
    %VULKAN_BIN_PATH%\glslangValidator.exe -V -H -o Grass.frag.spv Grass.frag
    move Grass.vert.spv ../Bin
    move Grass.geom.spv ../Bin
    move Grass.frag.spv ../Bin
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
        /LIBPATH:%VULKAN_LIB_PATH% user32.lib gdi32.lib /SUBSYSTEM:WINDOWS

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

