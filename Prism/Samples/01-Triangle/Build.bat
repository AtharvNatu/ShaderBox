@echo off

set API=OpenGL

cls

if exist *.obj del *.obj
if exist *.exe del *.exe
if exist *.res del *.res

echo ----------------------------------------------------------------------------------------------------------------
echo Compiling %API% and Win32 Source Code ...
echo ----------------------------------------------------------------------------------------------------------------
@REM cl.exe  /c ^
@REM         /EHsc ^
@REM         /std:c++17 ^
@REM         /I C:\glew\include ^
@REM         /I "Include\Core"
@REM         Source\*.cpp ^
@REM         Test\*.cpp

cl.exe  /c  /EHsc /std:c++17 /I C:\glew\include /I "Include\Core" Source/Core/*.cpp Test/*.cpp /DUNICODE

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Compiling Resource Files ...
echo ----------------------------------------------------------------------------------------------------------------
rc.exe /I "Include\Core" Resources/App.rc

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Linking Libraries and Resources...
echo Creating Executable...
echo ----------------------------------------------------------------------------------------------------------------
link.exe *.obj App.res /LIBPATH:C:\glew\lib\Release\x64 user32.lib gdi32.lib

@echo:
echo ----------------------------------------------------------------------------------------------------------------
echo Launching Application ...
echo ----------------------------------------------------------------------------------------------------------------
OGL.exe
