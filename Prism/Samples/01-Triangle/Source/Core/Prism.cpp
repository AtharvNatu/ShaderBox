#include "Prism.hpp"

Prism::Prism()
{
    #if PLATFORM_WINDOWS
        window = new Win32Window;
    #endif
}

Prism::Prism(API _api, std::string _windowTitle, int _windowWidth, int _windowHeight)
{
    api = _api;

    #if PLATFORM_WINDOWS
        window = new Win32Window(_windowTitle, _windowWidth, _windowHeight);
    #endif
}

API Prism::getAPI() const
{
    return api;
}

void Prism::setAPI(API _api)
{
    _api = api;
}

std::string Prism::getWindowTitle() const
{
    return window->getWindowTitle();
}

void Prism::setWindowTitle(std::string _windowTitle)
{
    window->setWindowTitle(_windowTitle);
}

int Prism::getWindowWidth() const
{
    return window->getWindowWidth();
}

void Prism::setWindowWidth(int _windowWidth)
{
    window->setWindowWidth(_windowWidth);
}

int Prism::getWindowHeight() const
{
    return window->getWindowHeight();
}

void Prism::setWindowHeight(int _windowHeight)
{
    window->setWindowHeight(_windowHeight);
}

INIT_STATUS Prism::initialize()
{
    window->initialize();

    return INIT_STATUS::SUCCESS;
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





