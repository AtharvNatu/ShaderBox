#include "Prism.hpp"

Prism::Prism()
{
    return;
}

Prism::Prism(API _api, std::string _windowTitle, int _windowWidth, int _windowHeight)
{
    window = new Win32Window(_windowTitle.c_str(), _windowWidth, _windowHeight);
}

void Prism::setWindowTitle(std::string _windowTitle)
{
    windowTitle = _windowTitle;
}

void Prism::setWindowWidth(int _windowWidth)
{
    windowWidth = _windowWidth;
}

int Prism::exec()
{
   return window->render();
}

Prism::~Prism()
{
    if (window)
    {
        delete window;
        window = nullptr;
    }
}





