#include "Win32-Window.hpp"

//! Global Variable Declarations
BOOL bFullScreen = FALSE;
BOOL bActiveWindow = FALSE;
BOOL bDone = FALSE;

WINDOWPLACEMENT wpPrev;
DWORD dwStyle;

Win32Window::Win32Window(const char* windowTitle, int windowWidth, int windowHeight)
{
    // Code
    hInstance = GetModuleHandle(NULL);

    std::string sTitle(windowTitle);
    std::wstring wTitle(sTitle.begin(), sTitle.end());
    lpwstrTitle = wTitle.c_str();

    initialize(windowWidth, windowHeight);
}

BOOL Win32Window::initialize(int windowWidth, int windowHeight)
{
    // Variable Declarations
    WNDCLASSEX wndclass;
    HWND hwnd;
    TCHAR szAppName[] = TEXT("Prism");

    // Code

    // Initialization of WNDCLASSEX Structure
    wndclass.cbSize = sizeof(WNDCLASSEX);
    wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wndclass.hInstance = hInstance;
    wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(APP_ICON));
    wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(APP_ICON));
    wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wndclass.cbClsExtra = 0;
    wndclass.cbWndExtra = 0;
    wndclass.lpfnWndProc = WndProc;
    wndclass.lpszClassName = szAppName;
    wndclass.lpszMenuName = NULL;
    
    // Register the class
    RegisterClassEx(&wndclass);

    //* Window Centering
    int screenX = GetSystemMetrics(SM_CXSCREEN);
    int centerX = (screenX / 2) - (windowWidth / 2);

    int screenY = GetSystemMetrics(SM_CYSCREEN);
    int centerY = (screenY / 2) - (windowHeight / 2);

    // Create Window
    hwnd = CreateWindowExW(
        WS_EX_APPWINDOW,
        szAppName,
        lpwstrTitle,
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
        centerX,
        centerY,
        windowWidth,
        windowHeight,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    // Show and Update Window
    ShowWindow(hwnd, 0);
    UpdateWindow(hwnd);

    // Bring the window to foreground and set focus
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    return TRUE;
}

int Win32Window::render()
{
    // Variable Declarations
    MSG msg;

    // while (bDone == FALSE)
    // {
    //     if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
    //     {
    //         if (msg.message == WM_QUIT)
    //             bDone = TRUE;
    //         else
    //         {
    //             TranslateMessage(&msg);
    //             DispatchMessage(&msg);
    //         }
    //     }
    //     // else
    //     // {
    //     //     if (bActiveWindow)
    //     //     {
    //     //         //! Render the scene
    //     //         display();

    //     //         //! Update the scene
    //     //         update();
    //     //     }
    //     // }
    // }

    while (GetMessage(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return (int)msg.wParam;
}

void Win32Window::toggleFullScreen()
{
    // Variable Declarations
    MONITORINFO mi;

    // Code
    if (bFullScreen == FALSE)
    {
        dwStyle = GetWindowLong(hwnd, GWL_STYLE);

        if (dwStyle & WS_OVERLAPPEDWINDOW)
        {
            mi.cbSize = sizeof(MONITORINFO);

            if (GetWindowPlacement(hwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(hwnd, MONITORINFOF_PRIMARY), &mi))
            {
                SetWindowLong(hwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
                SetWindowPos(
                    hwnd, 
                    HWND_TOP, 
                    mi.rcMonitor.left, 
                    mi.rcMonitor.top, 
                    mi.rcMonitor.right - mi.rcMonitor.left,
                    mi.rcMonitor.bottom - mi.rcMonitor.top,
                    SWP_NOZORDER | SWP_FRAMECHANGED
                );
            }

            ShowCursor(FALSE);
            bFullScreen = TRUE;
        }
    }
    else
    {
        SetWindowLong(hwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &wpPrev);
        SetWindowPos(
            hwnd, 
            HWND_TOP,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOZORDER
        );

        ShowCursor(TRUE);
        bFullScreen = FALSE;
    }
}

void Win32Window::uninitialize()
{
    if (hwnd)
    {
        DestroyWindow(hwnd);
        hwnd = NULL;
    }
}

Win32Window::~Win32Window()
{
    uninitialize();
}

static void ToggleFullScreen()
{

}



// Callback Function
LRESULT CALLBACK Win32Window::WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    // Code
    switch(iMsg)
    {
        case WM_CREATE:
            memset(&wpPrev, 0, sizeof(WINDOWPLACEMENT));
            wpPrev.length = sizeof(WINDOWPLACEMENT);
        break;

        case WM_SETFOCUS:
            bActiveWindow = TRUE;
        break;

        case WM_KILLFOCUS:
            bActiveWindow = FALSE;
        break;

        // case WM_SIZE:
        //     resize(LOWORD(lParam), HIWORD(lParam));
        // break;

        case WM_KEYDOWN:
            switch(wParam)
            {
                case 27:
                    DestroyWindow(hwnd);
                break;

                default:
                break;
            }
        break;

        case WM_CHAR:
            switch(wParam)
            {
                case 'F':
                case 'f':
                    ToggleFullScreen();
                break;

                default:
                break;
            }
        break;

        case WM_CLOSE:
            DestroyWindow(hwnd);
        break;

        case WM_DESTROY:
            PostQuitMessage(0);
        break;

        default:
        break;
    }

    return DefWindowProc(hwnd, iMsg, wParam, lParam);
}

int main(int, char **);

static inline char* wideToMulti(unsigned int codePage, const wchar_t* aw)
{
    int required = WideCharToMultiByte(codePage, 0, aw, -1, nullptr, 0, nullptr, nullptr);
    char* result = new char[required];
    WideCharToMultiByte(codePage, 0, aw, -1, result, required, nullptr, nullptr);
    return result;
}

static inline int myEntryPoint()
{
    int argc = __argc;
    char** argv = __argv;

    if (argv)
        return main(argc, argv);

    wchar_t** argvW = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argvW)
        return -1;

    argv = new char*[argc + 1];
    for (int i = 0; i < argc; ++i)
        argv[i] = wideToMulti(CP_ACP, argvW[i]);
    argv[argc] = nullptr;

    LocalFree(argvW);

    int result = main(argc, argv);

    for (int i = 0; i < argc; ++i)
        delete[] argv[i];
    delete[] argv;

    return result;
}

extern "C" int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int)
{
    return myEntryPoint();
}
