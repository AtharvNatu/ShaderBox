@echo off

set API=OpenGL
set GLEW_INCLUDE_PATH=C:\glew\include
set GLEW_LIB_PATH=C:\glew\lib\Release\x64
set IMGUI_PATH=ImGui
set IMGUI_BACKENDS=%IMGUI_PATH%\backends
set BIN_DIR=Bin

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
        /std:c++17 ^
        /Fo%BIN_DIR%\ ^
        /I %GLEW_INCLUDE_PATH% ^
        /I %IMGUI_PATH% ^
        /I %IMGUI_BACKENDS% ^
        *.cpp ^
        %IMGUI_PATH%\imgui.cpp ^
        %IMGUI_PATH%\imgui_draw.cpp ^
        %IMGUI_PATH%\imgui_widgets.cpp ^
        %IMGUI_PATH%\imgui_tables.cpp ^
        %IMGUI_BACKENDS%\imgui_impl_win32.cpp ^
        %IMGUI_BACKENDS%\imgui_impl_opengl3.cpp

if errorlevel 1 (
        @echo:
        echo Compilation Failed !!!
        exit /b 1
)


@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Compiling Resource Files ...
echo ----------------------------------------------------------------------------------------------------------------
rc.exe /fo %BIN_DIR%\OGL.res OGL.rc

if errorlevel 1 (
        @echo:
        echo Resource Compilation Failed !!!
        exit /b 1
)


@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Linking Libraries and Resources...
echo Creating Executable...
echo ----------------------------------------------------------------------------------------------------------------
link.exe ^
        /OUT:%BIN_DIR%\OGL.exe ^
        %BIN_DIR%\*.obj ^
        %BIN_DIR%\OGL.res ^
        /LIBPATH:%GLEW_LIB_PATH% ^
        user32.lib gdi32.lib libfftw3f-3.lib

if errorlevel 1 (
        @echo:
        echo Linking Failed !!!
        exit /b 1
)

move /Y %BIN_DIR%\OGL.exe . >nul 2>&1

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Launching Application ...
echo ----------------------------------------------------------------------------------------------------------------
OGL.exe
