#include "Win32-Window.hpp"

int main(int argc, char** argv)
{
    Win32Window window("Prism Window", 800, 600);
    
    return window.render();
}
