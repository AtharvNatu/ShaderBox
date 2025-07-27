@echo off

set API=OpenGL
set GLEW_PATH=C:\glew
set IMGUI_PATH=ImGui
set IMGUI_BACKENDS=%IMGUI_PATH%\backends

cls

if exist *.obj del *.obj
if exist *.exe del *.exe
if exist *.res del *.res

echo ----------------------------------------------------------------------------------------------------------------
echo Compiling %API% and Win32 Source Code ...
echo ----------------------------------------------------------------------------------------------------------------
cl.exe  /c ^
        /EHsc ^
        /std:c++17 ^
        /I %GLEW_PATH%\include ^
        /I %IMGUI_PATH% ^
        /I %IMGUI_BACKENDS% ^
        OGL.cpp ^
        %IMGUI_PATH%\imgui.cpp ^
        %IMGUI_PATH%\imgui_draw.cpp ^
        %IMGUI_PATH%\imgui_widgets.cpp ^
        %IMGUI_PATH%\imgui_tables.cpp ^
        %IMGUI_BACKENDS%\imgui_impl_win32.cpp ^
        %IMGUI_BACKENDS%\imgui_impl_opengl3.cpp

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Compiling Resource Files ...
echo ----------------------------------------------------------------------------------------------------------------
rc.exe OGL.rc

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Linking Libraries and Resources...
echo Creating Executable...
echo ----------------------------------------------------------------------------------------------------------------
link.exe /OUT:OGL.exe *.obj OGL.res /LIBPATH:C:\glew\lib\Release\x64 user32.lib gdi32.lib

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Launching Application ...
echo ----------------------------------------------------------------------------------------------------------------
OGL.exe
