//* Header Files
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "OGL.h"

#define WIN_WIDTH   800
#define WIN_HEIGHT  600

//* Global Function Declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//* Global Variable Declarations
HWND ghwnd = FALSE;
BOOL gbFullScreen = FALSE;
BOOL gbActiveWindow = FALSE;
FILE *gpFile = NULL;
WINDOWPLACEMENT wpPrev;
DWORD dwStyle;

// Entry Point Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    // Function Declarations
    int initialize(void);
    void display(void);
    void update(void);
    void uninitialize(void);

    // Variable Declarations
    WNDCLASSEX wndclass;
    HWND hwnd;
    MSG msg;
    TCHAR szAppName[] = TEXT("OpenGL-Window");
    BOOL bDone = FALSE;
    int iRetVal = 0;

    // Code
    gpFile = fopen("OGL.log", "w");
    if (gpFile == NULL)
    {
        MessageBox(NULL, TEXT("Failed To Create Log File ... Exiting !!!"), TEXT("File I/O Error"), MB_OK | MB_ICONERROR);
        exit(EXIT_FAILURE);
    }
    else
        fprintf(gpFile, "%s() => Program Started Successfully\n", __func__);
    
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
    int centerX = (screenX / 2) - (WIN_WIDTH / 2);

    int screenY = GetSystemMetrics(SM_CYSCREEN);
    int centerY = (screenY / 2) - (WIN_HEIGHT / 2);

    // Create Window
    hwnd = CreateWindowEx(
        WS_EX_APPWINDOW,
        szAppName,
        TEXT("Skeleton Window"),
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
        centerX,
        centerY,
        WIN_WIDTH,
        WIN_HEIGHT,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    ghwnd = hwnd;

    //! Initialize
    iRetVal = initialize();

    // Show and Update Window
    ShowWindow(hwnd, iCmdShow);
    UpdateWindow(hwnd);

    // Bring the window to foreground and set focus
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    //! Game Loop
    while (bDone == FALSE)
    {
        if (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
                bDone = TRUE;
            else
            {
                TranslateMessage(&msg);
                DispatchMessage(&msg);
            }
        }
        else
        {
            if (gbActiveWindow)
            {
                //! Render the scene
                display();

                //! Update the scene
                update();
            }
        }
    }

    uninitialize();

    return (int)msg.wParam;

}

// Callback Function
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    // Function Declarations
    void ToggleFullScreen(void);
    void resize(int, int);
    void uninitialize(void);

    // Code
    switch(iMsg)
    {
        case WM_CREATE:
            memset(&wpPrev, 0, sizeof(WINDOWPLACEMENT));
            wpPrev.length = sizeof(WINDOWPLACEMENT);
        break;

        case WM_SETFOCUS:
            gbActiveWindow = TRUE;
        break;

        case WM_KILLFOCUS:
            gbActiveWindow = FALSE;
        break;

        case WM_SIZE:
            resize(LOWORD(lParam), HIWORD(lParam));
        break;

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

void ToggleFullScreen(void)
{
    // Variable Declarations
    MONITORINFO mi;

    // Code
    if (gbFullScreen == FALSE)
    {
        dwStyle = GetWindowLong(ghwnd, GWL_STYLE);

        if (dwStyle & WS_OVERLAPPEDWINDOW)
        {
            mi.cbSize = sizeof(MONITORINFO);

            if (GetWindowPlacement(ghwnd, &wpPrev) && GetMonitorInfo(MonitorFromWindow(ghwnd, MONITORINFOF_PRIMARY), &mi))
            {
                SetWindowLong(ghwnd, GWL_STYLE, dwStyle & ~WS_OVERLAPPEDWINDOW);
                SetWindowPos(
                    ghwnd, 
                    HWND_TOP, 
                    mi.rcMonitor.left, 
                    mi.rcMonitor.top, 
                    mi.rcMonitor.right - mi.rcMonitor.left,
                    mi.rcMonitor.bottom - mi.rcMonitor.top,
                    SWP_NOZORDER | SWP_FRAMECHANGED
                );
            }

            ShowCursor(FALSE);
            gbFullScreen = TRUE;
        }
    }
    else
    {
        SetWindowLong(ghwnd, GWL_STYLE, dwStyle | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(ghwnd, &wpPrev);
        SetWindowPos(
            ghwnd, 
            HWND_TOP,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_NOZORDER
        );

        ShowCursor(TRUE);
        gbFullScreen = FALSE;
    }
}

int initialize(void)
{
    // Code
    return 0;
}

void resize(int width, int height)
{
    // Code
    if (height <= 0)
        height = 1;
}

void display(void)
{
    // Code
}

void update(void)
{
    // Code
}

void uninitialize(void)
{
    // Function Declarations
    void ToggleFullScreen(void);

    // Code
    if (gbFullScreen)
        ToggleFullScreen();
    
    if (ghwnd)
    {
        DestroyWindow(ghwnd);
        ghwnd = NULL;
    }

    if (gpFile)
    {
        fprintf(gpFile, "%s() => Program Terminated Successfully\n", __func__);
        fclose(gpFile);
        gpFile = NULL;
    }
}
