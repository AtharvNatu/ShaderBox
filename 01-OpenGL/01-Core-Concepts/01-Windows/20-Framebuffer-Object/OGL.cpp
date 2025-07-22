//* Header Files
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#include "OGL.h"
#include "vmath.h"
#include "Sphere.hpp"

//! OpenGL Header Files
#include <GL/glew.h>
#include <GL/gl.h>

#define WIN_WIDTH   800
#define WIN_HEIGHT  600

#define FBO_WIDTH   512
#define FBO_HEIGHT  512

//! OpenGL Libraries
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")

//* Global Function Declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

//* Global Variable Declarations
HWND ghwnd = NULL;
HDC ghdc = NULL;
HGLRC ghrc = NULL;

BOOL gbFullScreen = FALSE;
BOOL gbActiveWindow = FALSE;

FILE *gpFile = NULL;

WINDOWPLACEMENT wpPrev;
DWORD dwStyle;

int winWidth;
int winHeight;

enum INIT_ERRORS
{
    CPF_ERROR = -1,
    SPF_ERROR = -2,
    WGL_CC_ERROR = -3,
    WGL_MC_ERROR = -4,
    GLEW_INIT_ERROR = -5,
    VS_COMPILE_ERROR = -6,
    TES_COMPILE_ERROR = -7,
    TCS_COMPILE_ERROR = -8,
    GS_COMPILE_ERROR = -9,
    FS_COMPILE_ERROR = -10,
    PROGRAM_LINK_ERROR = -11,
    MEM_ALLOC_FAILED = -12,
    FBO_ERROR = -13,
    SPHERE_INIT_ERROR = -14
};

enum ATTRIBUTES
{
    ATTRIBUTE_POSITION = 0,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEXTURE0
};

//! Cube Related
GLuint shaderProgramObject;

GLuint vao_cube;
GLuint vbo_cube_position;
GLuint vbo_cube_texcoord;

GLuint mvpMatrixUniform;
GLuint textureSamplerUniform;

vmath::mat4 perspectiveProjectionMatrix;

float angleCube = 0.0f;

//* FBO Related Variables
//* -----------------------------------------------------------------------
GLuint fbo;
GLuint rbo;
GLuint fbo_texture;
bool bFBOResult = false;
//* -----------------------------------------------------------------------

// Texture Scene Related Variables
// ------------------------------------------------------------------------------------
GLuint shaderProgramObject_pv = 0;
GLuint shaderProgramObject_pf = 0;

GLuint vao_sphere = 0;
GLuint vbo_sphere_position = 0;
GLuint vbo_sphere_normal = 0;
GLuint ebo_sphere_indices = 0;

//! Per-Vertex
//! -----------------------------------------------------------------------
GLuint laUniform_pv[3];   //? Light Ambient
GLuint ldUniform_pv[3];   //? Light Diffuse
GLuint lsUniform_pv[3];   //? Light Specular
GLuint lightPositionUniform_pv[3];

GLuint kaUniform_pv;   //? Material Ambient
GLuint kdUniform_pv;   //? Material Diffuse
GLuint ksUniform_pv;   //? Material Specular
GLuint materialShininessUniform_pv;

GLuint lightEnabledUniform_pv;

GLuint modelMatrixUniform_pv = 0;
GLuint viewMatrixUniform_pv = 0;
GLuint projectionMatrixUniform_pv = 0;
//! -----------------------------------------------------------------------

//! Per-Fragment
//! -----------------------------------------------------------------------
GLuint laUniform_pf[3];   //? Light Ambient
GLuint ldUniform_pf[3];   //? Light Diffuse
GLuint lsUniform_pf[3];   //? Light Specular
GLuint lightPositionUniform_pf[3];

GLuint kaUniform_pf;   //? Material Ambient
GLuint kdUniform_pf;   //? Material Diffuse
GLuint ksUniform_pf;   //? Material Specular
GLuint materialShininessUniform_pf;

GLuint lightEnabledUniform_pf;

GLuint modelMatrixUniform_pf = 0;
GLuint viewMatrixUniform_pf = 0;
GLuint projectionMatrixUniform_pf = 0;
//! -----------------------------------------------------------------------

BOOL bLight = FALSE;

struct Light
{
    vmath::vec4 lightAmbient;
    vmath::vec4 lightDiffuse;
    vmath::vec4 lightSpecular;
    vmath::vec4 lightPosition;
};

Light lights[3];

GLfloat materialAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat materialDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat materialShininess = 50.0f;

GLfloat lightAngleZero = 0.0f;
GLfloat lightAngleOne = 0.0f;
GLfloat lightAngleTwo = 0.0f;

GLfloat radius = 5.0f;
// ------------------------------------------------------------------------------------

//! Create Sphere Object
Sphere* sphere = nullptr;
GLuint gNumIndices = 0;

vmath::mat4 perspectiveProjectionMatrix_sphere;

GLchar chosenShader = 'v';
const GLfloat animationSpeed = 0.75f;

// Entry Point Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    // Function Declarations
    int initialize(void);
    void display(void);
    void update(void);
    void uninitialize(void);
    void uninitializeSphere(void);

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
        TEXT("OpenGL : Render-To-Texture => Framebuffer Object"),
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
    switch(iRetVal)
    {
        case CPF_ERROR:
            fprintf(gpFile, "ERROR : %s() => ChoosePixelFormat() Failed !!!\n", __func__);
            uninitialize();
        break;

        case SPF_ERROR:
            fprintf(gpFile, "ERROR : %s() => SetPixelFormat() Failed !!!\n", __func__);
            uninitialize();
        break;

        case WGL_CC_ERROR:
            fprintf(gpFile, "ERROR : %s() => Failed to create OpenGL context !!!\n", __func__);
            uninitialize();
        break;

        case WGL_MC_ERROR:
            fprintf(gpFile, "ERROR : %s() => Failed to make rendering context as current context !!!\n", __func__);
            uninitialize();
        break;

        case GLEW_INIT_ERROR:
            fprintf(gpFile, "ERROR : %s() => GLEW Initialization Failed !!!\n", __func__);
            uninitialize();
        break;

        case VS_COMPILE_ERROR:
        case TES_COMPILE_ERROR:
        case TCS_COMPILE_ERROR:
        case GS_COMPILE_ERROR:
        case FS_COMPILE_ERROR:
            uninitialize();
        break;

        case PROGRAM_LINK_ERROR:
            uninitialize();
        break;

        case MEM_ALLOC_FAILED:
            uninitialize();
        break;

        case FBO_ERROR:
            uninitialize();
        break;

        case SPHERE_INIT_ERROR:
            uninitializeSphere();
        break;

        default:
            fprintf(gpFile, "%s() => Initialization Completed Successfully\n", __func__);
        break;
    }

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
                    ToggleFullScreen();
                break;

                default:
                break;
            }
        break;

        case WM_CHAR:
            switch(wParam)
            {
                case 'L':
                case 'l':
                    bLight = !bLight;
                break;

                case 'F':
                case 'f':
                    chosenShader = 'f';
                break;

                case 'V':
                case 'v':
                    chosenShader = 'v';
                break;

                case 'Q':
                case 'q':
                    DestroyWindow(hwnd);
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
    // Function Declarations
    void resize(int, int);
    void printGLInfo(void);
    int initializeSphere(int, int);
    bool createFBO(GLint, GLint);

    // Variable Declarations
    PIXELFORMATDESCRIPTOR pfd;
    int iPixelFormatIndex;
    GLenum initStatus;

    // Code
    ZeroMemory(&pfd, sizeof(PIXELFORMATDESCRIPTOR));

    // Initialization of PIXELFORMATDESCRIPTOR Structure
    pfd.nSize = sizeof(PIXELFORMATDESCRIPTOR);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cRedBits = 8;
    pfd.cBlueBits = 8;
    pfd.cGreenBits = 8;
    pfd.cAlphaBits = 8;
    pfd.cDepthBits = 32;

    // Get DC
    ghdc = GetDC(ghwnd);

    // Choose Pixel Format
    iPixelFormatIndex = ChoosePixelFormat(ghdc, &pfd);
    if (iPixelFormatIndex == 0)
        return CPF_ERROR;
    
    // Set chosen Pixel Format
    if (SetPixelFormat(ghdc, iPixelFormatIndex, &pfd) == FALSE)
        return SPF_ERROR;
    
    // Create OpenGL Rendering Context
    ghrc = wglCreateContext(ghdc);
    if (ghrc == NULL)
        return WGL_CC_ERROR;
    
    // Make the rendering context as the current context
    if (wglMakeCurrent(ghdc, ghrc) == FALSE)
        return WGL_MC_ERROR;
    
    // GLEW Initialization
    initStatus = glewInit();
    if (initStatus != GLEW_OK)
        return GLEW_INIT_ERROR;

    printGLInfo();

    //! Vertex Shader
    //! ----------------------------------------------------------------------------
    const GLchar* vertexShaderSourceCode = 
        "#version 460 core" \
        "\n" \
        "in vec4 a_position;" \
        "in vec2 a_texcoord;" \

        "uniform mat4 u_mvpMatrix;" \

        "out vec2 a_texcoord_out;" \

        "void main(void)" \
        "{" \
            "gl_Position = u_mvpMatrix * a_position;" \
            "a_texcoord_out = a_texcoord;" \
        "}";

    GLuint vertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderObject, 1, (const GLchar**)&vertexShaderSourceCode, NULL);
    glCompileShader(vertexShaderObject);

    GLint status = 0;
    GLint infoLogLength = 0;
    GLchar* szLog = NULL;

    glGetShaderiv(vertexShaderObject, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength * sizeof(GLchar));
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Vertex Shader Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetShaderInfoLog(vertexShaderObject, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Vertex Shader Compilation Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return VS_COMPILE_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------

    //! Fragment Shader
    //! ----------------------------------------------------------------------------
    const GLchar* fragmentShaderSourceCode = 
        "#version 460 core" \
        "\n" \
        "in vec2 a_texcoord_out;" \
        "uniform sampler2D u_textureSampler;" \
        "out vec4 FragColor;" \
        "void main(void)" \
        "{" \
            "FragColor = texture(u_textureSampler, a_texcoord_out);" \
        "}";

    GLuint fragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderObject, 1, (const GLchar**)&fragmentShaderSourceCode, NULL);
    glCompileShader(fragmentShaderObject);

    status = 0;
    infoLogLength = 0;
    szLog = NULL;

    glGetShaderiv(fragmentShaderObject, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength * sizeof(GLchar));
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Fragment Shader Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetShaderInfoLog(fragmentShaderObject, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Fragment Shader Compilation Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return FS_COMPILE_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------
    
    //! Shader Program Object
    //! ----------------------------------------------------------------------------
    shaderProgramObject = glCreateProgram();

    glAttachShader(shaderProgramObject, vertexShaderObject);
    glAttachShader(shaderProgramObject, fragmentShaderObject);

    //! Bind Attribute
    glBindAttribLocation(shaderProgramObject, ATTRIBUTE_POSITION, "a_position");
    glBindAttribLocation(shaderProgramObject, ATTRIBUTE_TEXTURE0, "a_texcoord");

    glLinkProgram(shaderProgramObject);

    status = 0;
    infoLogLength = 0;
    szLog = NULL;

    glGetProgramiv(shaderProgramObject, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramiv(shaderProgramObject, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength);
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Shader Program Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetProgramInfoLog(shaderProgramObject, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Shader Program Link Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return PROGRAM_LINK_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------

    //! OpenGL Code

    //! Get Uniform Location
    mvpMatrixUniform = glGetUniformLocation(shaderProgramObject, "u_mvpMatrix");
    textureSamplerUniform = glGetUniformLocation(shaderProgramObject, "u_textureSampler");

    const GLfloat cube_position[] =
    {
        // Top
        1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, 1.0f, 1.0f,
        1.0f, 1.0f, 1.0f,

        // Bottom
        1.0f, -1.0f, -1.0f,
       -1.0f, -1.0f, -1.0f,
       -1.0f, -1.0f,  1.0f,
        1.0f, -1.0f,  1.0f,

        // Front
        1.0f, 1.0f, 1.0f,
       -1.0f, 1.0f, 1.0f,
       -1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,

        // Back
        1.0f, 1.0f, -1.0f,
       -1.0f, 1.0f, -1.0f,
       -1.0f, -1.0f, -1.0f,
        1.0f, -1.0f, -1.0f,

        // Right
        1.0f, 1.0f, -1.0f,
        1.0f, 1.0f, 1.0f,
        1.0f, -1.0f, 1.0f,
        1.0f, -1.0f, -1.0f,

        // Left
        -1.0f, 1.0f, 1.0f,
        -1.0f, 1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f, 1.0f
    };

    const GLfloat cube_texcoords[] =
    {
        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,

        0.0f, 0.0f,
        1.0f, 0.0f,
        1.0f, 1.0f,
        0.0f, 1.0f,
    };

    //! VAO and VBO Related Code

    // VAO For Cube
    glGenVertexArrays(1, &vao_cube);
    glBindVertexArray(vao_cube);
    {
        //* Cube Position
        glGenBuffers(1, &vbo_cube_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_position);
        {
            glBufferData(GL_ARRAY_BUFFER, sizeof(cube_position), cube_position, GL_STATIC_DRAW);
            glVertexAttribPointer(ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(ATTRIBUTE_POSITION);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //* Cube Texture
        glGenBuffers(1, &vbo_cube_texcoord);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_texcoord);
        {
            glBufferData(GL_ARRAY_BUFFER, sizeof(cube_texcoords), cube_texcoords, GL_STATIC_DRAW);
            glVertexAttribPointer(ATTRIBUTE_TEXTURE0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(ATTRIBUTE_TEXTURE0);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);

    // Depth Related Code
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Clear the screen using white color
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // Enable texture
    glEnable(GL_TEXTURE_2D);
    
    perspectiveProjectionMatrix = vmath::mat4::identity();

    // Warmup resize call
    resize(WIN_WIDTH, WIN_HEIGHT);

    // FBO Code
    bFBOResult = createFBO(FBO_WIDTH, FBO_HEIGHT);
    if (bFBOResult)
    {
        int ret = initializeSphere(FBO_WIDTH, FBO_HEIGHT);
        if (ret != 0)
            return SPHERE_INIT_ERROR;
    } 

    return 0;
}

void printGLInfo(void)
{
    // Variable Declarations
    GLint numExtensions = 0;

    // Code
    fprintf(gpFile, "\nOpenGL Information\n");
    fprintf(gpFile, "------------------------------------------------------\n");
    
    fprintf(gpFile, "OpenGL Vendor : %s\n", glGetString(GL_VENDOR));
    fprintf(gpFile, "OpenGL Renderer : %s\n", glGetString(GL_RENDERER));
    fprintf(gpFile, "OpenGL Version : %s\n", glGetString(GL_VERSION));
    fprintf(gpFile, "GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    fprintf(gpFile, "------------------------------------------------------\n");

    glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
    fprintf(gpFile, "\nNumber of Supported Extensions : %d\n", numExtensions);
    fprintf(gpFile, "------------------------------------------------------\n");
    for (GLint i = 0; i < numExtensions; i++)
        fprintf(gpFile, "%s\n", glGetStringi(GL_EXTENSIONS, i));
    fprintf(gpFile, "------------------------------------------------------\n"); 
}

bool createFBO(GLint textureWidth, GLint textureHeight)
{
    // Code

    //* Step - 1 : Check available renderbuffer size
    int maxRenderbufferSize;
    glGetIntegerv(GL_MAX_RENDERBUFFER_SIZE, &maxRenderbufferSize);
    if (maxRenderbufferSize < textureWidth || maxRenderbufferSize < textureHeight)
    {
        fprintf(gpFile, "Insufficient Renderbuffer Size !!!\n");
        return false;
    }

    //* Step - 2 : Create Framebuffer Object
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    {
        //* Step - 3 : Create Renderbuffer Object
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
       
        //* Step - 4 : Storage and format of Renderbuffer
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT16, textureWidth, textureHeight);

        //* Step - 5 : Create empty texture for upcoming target scene
        glGenTextures(1, &fbo_texture);
        glBindTexture(GL_TEXTURE_2D, fbo_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureWidth, textureHeight, 0, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, NULL);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbo_texture, 0);

        //* Step - 6 : Give Renderbuffer to FBO
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);
    
        //* Step - 7 : Check successful creation of Framebuffer
        GLenum result = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if (result != GL_FRAMEBUFFER_COMPLETE)
        {
            fprintf(gpFile, "Incomplete Framebuffer !!!\n");
            return false;
        }
    }
    //! Automatically unbinds RBO and Texture
    glBindFramebuffer(GL_FRAMEBUFFER, 0); 

    return true;
}

int initializeSphere(int width, int height)
{
    // Function Declarations
    void resizeSphere(int, int);

    // Code

    //? Per-Vertex
    //? -----------------------------------------------------------------------------------------------------------

    //! Vertex Shader
    //! ----------------------------------------------------------------------------
    const GLchar* vertexShaderSourceCode_pv = 
        "#version 460 core" \
        "\n" \
        
        "in vec4 a_position;" \
        "in vec3 a_normal;" \

        "uniform mat4 u_modelMatrix;" \
        "uniform mat4 u_viewMatrix;" \
        "uniform mat4 u_projectionMatrix;" \

        "uniform vec3 u_la[3];" \
        "uniform vec3 u_ld[3];" \
        "uniform vec3 u_ls[3];" \
        "uniform vec4 u_lightPosition[3];" \

        "uniform vec3 u_ka;" \
        "uniform vec3 u_kd;" \
        "uniform vec3 u_ks;" \
        "uniform float u_materialShininess;" \
        
        "uniform int u_lightEnabled;" \

        "out vec3 phong_ads_light;" \

        "void main(void)" \
        "{" \
            "if (u_lightEnabled == 1)" \
            "{" \
                "vec3 ambient[3];" \
                "vec3 lightDirection[3];" \
                "vec3 diffuse[3];" \
                "vec3 reflectionVector[3];" \
                "vec3 specular[3];" \

                "vec4 eyeCoordinates = u_viewMatrix * u_modelMatrix * a_position;" \
                "mat3 normalMatrix = mat3(u_viewMatrix * u_modelMatrix);" \
                "vec3 transformedNormals = normalize(normalMatrix * a_normal);" \
                "vec3 viewerVector = normalize(-eyeCoordinates.xyz);" \

                "for (int i = 0; i < 3; i++)" \
                "{" \
                    "ambient[i] = u_la[i] * u_ka;" \
                    "lightDirection[i] = normalize(vec3(u_lightPosition[i]) - eyeCoordinates.xyz);" \
                    "diffuse[i] = u_ld[i] * u_kd * max(dot(lightDirection[i], transformedNormals), 0.0);" \
                    "reflectionVector[i] = reflect(-lightDirection[i], transformedNormals);" \
                    "specular[i] = u_ls[i] * u_ks * pow(max(dot(reflectionVector[i], viewerVector), 0.0), u_materialShininess);" \
                    "phong_ads_light += ambient[i] + diffuse[i] + specular[i];" \
                "}" \
            "}" \
            "else" \
            "{" \
                "phong_ads_light = vec3(1.0, 1.0, 1.0);" \
            "}" \

            "gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * a_position;" \
        "}";

    GLuint vertexShaderObject_pv = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderObject_pv, 1, (const GLchar**)&vertexShaderSourceCode_pv, NULL);
    glCompileShader(vertexShaderObject_pv);

    GLint status = 0;
    GLint infoLogLength = 0;
    GLchar* szLog = NULL;

    glGetShaderiv(vertexShaderObject_pv, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObject_pv, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength * sizeof(GLchar));
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Per-Vertex Vertex Shader Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetShaderInfoLog(vertexShaderObject_pv, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Per-Vertex Vertex Shader Compilation Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return VS_COMPILE_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------

    //! Fragment Shader
    //! ----------------------------------------------------------------------------
    const GLchar* fragmentShaderSourceCode_pv = 
        "#version 460 core" \
        "\n" \

        "in vec3 phong_ads_light;" \

        "out vec4 FragColor;" \

        "void main(void)" \
        "{" \
            "FragColor = vec4(phong_ads_light, 1.0);" \
        "}";

    GLuint fragmentShaderObject_pv = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderObject_pv, 1, (const GLchar**)&fragmentShaderSourceCode_pv, NULL);
    glCompileShader(fragmentShaderObject_pv);

    status = 0;
    infoLogLength = 0;
    szLog = NULL;

    glGetShaderiv(fragmentShaderObject_pv, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObject_pv, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength * sizeof(GLchar));
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Per-Vertex Fragment Shader Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetShaderInfoLog(fragmentShaderObject_pv, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Per-Vertex Fragment Shader Compilation Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return FS_COMPILE_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------
    
    //! Shader Program Object
    //! ----------------------------------------------------------------------------
    shaderProgramObject_pv = glCreateProgram();

    glAttachShader(shaderProgramObject_pv, vertexShaderObject_pv);
    glAttachShader(shaderProgramObject_pv, fragmentShaderObject_pv);

    //! Bind Attribute
    glBindAttribLocation(shaderProgramObject_pv, ATTRIBUTE_POSITION, "a_position");
    glBindAttribLocation(shaderProgramObject_pv, ATTRIBUTE_NORMAL, "a_normal");

    glLinkProgram(shaderProgramObject_pv);

    status = 0;
    infoLogLength = 0;
    szLog = NULL;

    glGetProgramiv(shaderProgramObject_pv, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramiv(shaderProgramObject_pv, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength);
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Per-Vertex Shader Program Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetProgramInfoLog(shaderProgramObject_pv, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Per-Vertex Shader Program Link Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return PROGRAM_LINK_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------

    //! OpenGL Code

    //! Get Uniform Location
    modelMatrixUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_modelMatrix");
    viewMatrixUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_viewMatrix");
    projectionMatrixUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_projectionMatrix");

    laUniform_pv[0] = glGetUniformLocation(shaderProgramObject_pv, "u_la[0]");
    ldUniform_pv[0] = glGetUniformLocation(shaderProgramObject_pv, "u_ld[0]");
    lsUniform_pv[0] = glGetUniformLocation(shaderProgramObject_pv, "u_ls[0]");
    lightPositionUniform_pv[0] = glGetUniformLocation(shaderProgramObject_pv, "u_lightPosition[0]");

    laUniform_pv[1] = glGetUniformLocation(shaderProgramObject_pv, "u_la[1]");
    ldUniform_pv[1] = glGetUniformLocation(shaderProgramObject_pv, "u_ld[1]");
    lsUniform_pv[1] = glGetUniformLocation(shaderProgramObject_pv, "u_ls[1]");
    lightPositionUniform_pv[1] = glGetUniformLocation(shaderProgramObject_pv, "u_lightPosition[1]");

    laUniform_pv[2] = glGetUniformLocation(shaderProgramObject_pv, "u_la[2]");
    ldUniform_pv[2] = glGetUniformLocation(shaderProgramObject_pv, "u_ld[2]");
    lsUniform_pv[2] = glGetUniformLocation(shaderProgramObject_pv, "u_ls[2]");
    lightPositionUniform_pv[2] = glGetUniformLocation(shaderProgramObject_pv, "u_lightPosition[2]");

    kaUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_ka");
    kdUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_kd");
    ksUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_ks");
    materialShininessUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_materialShininess");
    
    lightEnabledUniform_pv = glGetUniformLocation(shaderProgramObject_pv, "u_lightEnabled");
    //? -----------------------------------------------------------------------------------------------------------

    //? Per-Fragment
    //? -----------------------------------------------------------------------------------------------------------

    //! Vertex Shader
    //! ----------------------------------------------------------------------------
    const GLchar* vertexShaderSourceCode_pf = 
        "#version 460 core" \
        "\n" \
        
        "in vec4 a_position;" \
        "in vec3 a_normal;" \

        "uniform mat4 u_modelMatrix;" \
        "uniform mat4 u_viewMatrix;" \
        "uniform mat4 u_projectionMatrix;" \

        "uniform vec4 u_lightPosition[3];" \
        "uniform int u_lightEnabled;" \

        "out vec3 transformedNormals;" \
        "out vec3 lightDirection[3];" \
        "out vec3 viewerVector;" \

        "void main(void)" \
        "{" \
            "if (u_lightEnabled == 1)" \
            "{" \
                "vec4 eyeCoordinates = u_viewMatrix * u_modelMatrix * a_position;" \
                "mat3 normalMatrix = mat3(u_viewMatrix * u_modelMatrix);" \
                "transformedNormals = normalMatrix * a_normal;" \
                "viewerVector = -eyeCoordinates.xyz;" \
                "for (int i = 0; i < 3; i++)" \
                "{" \
                    "lightDirection[i] = vec3(u_lightPosition[i]) - eyeCoordinates.xyz;" \
                "}" \
            "}" \

            "gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * a_position;" \
        "}";

    GLuint vertexShaderObject_pf = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderObject_pf, 1, (const GLchar**)&vertexShaderSourceCode_pf, NULL);
    glCompileShader(vertexShaderObject_pf);

    status = 0;
    infoLogLength = 0;
    szLog = NULL;

    glGetShaderiv(vertexShaderObject_pf, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(vertexShaderObject_pf, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength * sizeof(GLchar));
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Per-Fragment Vertex Shader Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetShaderInfoLog(vertexShaderObject_pf, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Per-Fragment Vertex Shader Compilation Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return VS_COMPILE_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------

    //! Fragment Shader
    //! ----------------------------------------------------------------------------
    const GLchar* fragmentShaderSourceCode_pf = 
        "#version 460 core" \
        "\n" \

        "in vec3 transformedNormals;" \
        "in vec3 lightDirection[3];" \
        "in vec3 viewerVector;" \

        "uniform vec3 u_la[3];" \
        "uniform vec3 u_ld[3];" \
        "uniform vec3 u_ls[3];" \

        "uniform vec3 u_ka;" \
        "uniform vec3 u_kd;" \
        "uniform vec3 u_ks;" \
        "uniform float u_materialShininess;" \
        
        "uniform int u_lightEnabled;" \

        "out vec4 FragColor;" \

        "void main(void)" \
        "{" \
            "vec3 phong_ads_light;" \

            "if (u_lightEnabled == 1)" \
            "{" \
                "vec3 ambient[3];" \
                "vec3 diffuse[3];" \
                "vec3 reflectionVector[3];" \
                "vec3 specular[3];" \
                "vec3 normalized_light_direction[3];" \

                "vec3 normalized_transformed_normals = normalize(transformedNormals);" \
                "vec3 normalized_viewer_vector = normalize(viewerVector);" \

                "for (int i = 0; i < 3; i++)" \
                "{" \
                    "ambient[i] = u_la[i] * u_ka;" \
                    "normalized_light_direction[i] = normalize(lightDirection[i]);" \
                    "diffuse[i] = u_ld[i] * u_kd * max(dot(normalized_light_direction[i], normalized_transformed_normals), 0.0);" \
                    "reflectionVector[i] = reflect(-normalized_light_direction[i], normalized_transformed_normals);" \
                    "specular[i] = u_ls[i] * u_ks * pow(max(dot(reflectionVector[i], normalized_viewer_vector), 0.0), u_materialShininess);" \
                    "phong_ads_light += ambient[i] + diffuse[i] + specular[i];" \
                "}" \
            "}" \
            
            "else" \
            "{" \
                "phong_ads_light = vec3(1.0, 1.0, 1.0);" \
            "}" \

            "FragColor = vec4(phong_ads_light, 1.0);" \
        "}";

    GLuint fragmentShaderObject_pf = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderObject_pf, 1, (const GLchar**)&fragmentShaderSourceCode_pf, NULL);
    glCompileShader(fragmentShaderObject_pf);

    status = 0;
    infoLogLength = 0;
    szLog = NULL;

    glGetShaderiv(fragmentShaderObject_pf, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetShaderiv(fragmentShaderObject_pf, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength * sizeof(GLchar));
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Per-Fragment Fragment Shader Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetShaderInfoLog(fragmentShaderObject_pf, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Per-Fragment Fragment Shader Compilation Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return FS_COMPILE_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------
    
    //! Shader Program Object
    //! ----------------------------------------------------------------------------
    shaderProgramObject_pf = glCreateProgram();

    glAttachShader(shaderProgramObject_pf, vertexShaderObject_pf);
    glAttachShader(shaderProgramObject_pf, fragmentShaderObject_pf);

    //! Bind Attribute
    glBindAttribLocation(shaderProgramObject_pf, ATTRIBUTE_POSITION, "a_position");
    glBindAttribLocation(shaderProgramObject_pf, ATTRIBUTE_NORMAL, "a_normal");

    glLinkProgram(shaderProgramObject_pf);

    status = 0;
    infoLogLength = 0;
    szLog = NULL;

    glGetProgramiv(shaderProgramObject_pf, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        glGetProgramiv(shaderProgramObject_pf, GL_INFO_LOG_LENGTH, &infoLogLength);
        if (infoLogLength > 0)
        {
            szLog = (GLchar*)malloc(infoLogLength);
            if (szLog == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to szLog for Per-Fragment Shader Program Log !!!\n", __func__);
                return MEM_ALLOC_FAILED;
            }
            else
            {
                GLsizei logSize;
                glGetProgramInfoLog(shaderProgramObject_pf, GL_INFO_LOG_LENGTH, &logSize, szLog);
                fprintf(gpFile, "ERROR : Per-Fragment Shader Program Link Log : %s\n", szLog);
                free(szLog);
                szLog = NULL;
                return PROGRAM_LINK_ERROR;
            }
        }
    }
    //! ----------------------------------------------------------------------------

    //! OpenGL Code

    //! Get Uniform Location
    modelMatrixUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_modelMatrix");
    viewMatrixUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_viewMatrix");
    projectionMatrixUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_projectionMatrix");

    laUniform_pf[0] = glGetUniformLocation(shaderProgramObject_pf, "u_la[0]");
    ldUniform_pf[0] = glGetUniformLocation(shaderProgramObject_pf, "u_ld[0]");
    lsUniform_pf[0] = glGetUniformLocation(shaderProgramObject_pf, "u_ls[0]");
    lightPositionUniform_pf[0] = glGetUniformLocation(shaderProgramObject_pf, "u_lightPosition[0]");

    laUniform_pf[1] = glGetUniformLocation(shaderProgramObject_pf, "u_la[1]");
    ldUniform_pf[1] = glGetUniformLocation(shaderProgramObject_pf, "u_ld[1]");
    lsUniform_pf[1] = glGetUniformLocation(shaderProgramObject_pf, "u_ls[1]");
    lightPositionUniform_pf[1] = glGetUniformLocation(shaderProgramObject_pf, "u_lightPosition[1]");

    laUniform_pf[2] = glGetUniformLocation(shaderProgramObject_pf, "u_la[2]");
    ldUniform_pf[2] = glGetUniformLocation(shaderProgramObject_pf, "u_ld[2]");
    lsUniform_pf[2] = glGetUniformLocation(shaderProgramObject_pf, "u_ls[2]");
    lightPositionUniform_pf[2] = glGetUniformLocation(shaderProgramObject_pf, "u_lightPosition[2]");

    kaUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_ka");
    kdUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_kd");
    ksUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_ks");
    materialShininessUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_materialShininess");
    
    lightEnabledUniform_pf = glGetUniformLocation(shaderProgramObject_pf, "u_lightEnabled");
    //? -----------------------------------------------------------------------------------------------------------

    sphere = new Sphere(1.5f, 50, 16);
    gNumIndices = sphere->get_number_of_indices();

    //! VAO and VBO Related Code

    // VAO For Sphere
    glGenVertexArrays(1, &vao_sphere);
    glBindVertexArray(vao_sphere);
    {
        //* Sphere Position
        glGenBuffers(1, &vbo_sphere_position);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_position);
        {
            glBufferData(
                GL_ARRAY_BUFFER, 
                sphere->get_number_of_vertices() * sizeof(float), 
                sphere->get_vertices().data(), 
                GL_STATIC_DRAW
            );
            glVertexAttribPointer(ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(ATTRIBUTE_POSITION);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //* Sphere Normal
        glGenBuffers(1, &vbo_sphere_normal);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_sphere_normal);
        {
            glBufferData(
                GL_ARRAY_BUFFER, 
                sphere->get_number_of_normals() * sizeof(float), 
                sphere->get_normals().data(), 
                GL_STATIC_DRAW
            );
            glVertexAttribPointer(ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, NULL);
            glEnableVertexAttribArray(ATTRIBUTE_NORMAL);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //* Sphere EBO
        glGenBuffers(1, &ebo_sphere_indices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_sphere_indices);
        {
            glBufferData(
                GL_ELEMENT_ARRAY_BUFFER, 
                gNumIndices * sizeof(GLuint), 
                sphere->get_indices().data(), 
                GL_STATIC_DRAW
            );
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);

    lights[0].lightAmbient = vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    lights[0].lightDiffuse = vmath::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    lights[0].lightSpecular = vmath::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    lights[0].lightPosition = vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    lights[1].lightAmbient = vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    lights[1].lightDiffuse = vmath::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    lights[1].lightSpecular = vmath::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    lights[1].lightPosition = vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    lights[2].lightAmbient = vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    lights[2].lightDiffuse = vmath::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    lights[2].lightSpecular = vmath::vec4(0.0f, 0.0f, 1.0f, 1.0f);
    lights[2].lightPosition = vmath::vec4(0.0f, 0.0f, 0.0f, 1.0f);

    perspectiveProjectionMatrix_sphere = vmath::mat4::identity();

    resizeSphere(FBO_WIDTH, FBO_HEIGHT);

    return 0;
}

void resize(int width, int height)
{
    // Code
    if (height <= 0)
        height = 1;
    
    winWidth = width;
    winHeight = height;
    
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void resizeSphere(int width, int height)
{
    if (height <= 0)
        height = 1;
    
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    perspectiveProjectionMatrix_sphere = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void)
{
    // Code
    void displaySphere(GLint, GLint);
    void updateSphere(void);

    // Code
    if (bFBOResult)
    {
        displaySphere(FBO_WIDTH, FBO_HEIGHT);
        updateSphere();
    }

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    resize(winWidth, winHeight);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgramObject);
    {
        // Transformations
        vmath::mat4 translationMatrix = vmath::mat4::identity();
        vmath::mat4 rotationMatrix = vmath::mat4::identity();
        vmath::mat4 rotationMatrix_x = vmath::mat4::identity();
        vmath::mat4 rotationMatrix_y = vmath::mat4::identity();
        vmath::mat4 rotationMatrix_z = vmath::mat4::identity();
        vmath::mat4 scaleMatrix = vmath::mat4::identity();
        vmath::mat4 modelViewMatrix = vmath::mat4::identity();
        vmath::mat4 modelViewProjectionMatrix = vmath::mat4::identity();

        translationMatrix = vmath::translate(0.0f, 0.0f, -5.0f);
        scaleMatrix = vmath::scale(0.75f, 0.75f, 0.75f);
        rotationMatrix_x = vmath::rotate(angleCube, 1.0f, 0.0f, 0.0f);
        rotationMatrix_y = vmath::rotate(angleCube, 0.0f, 1.0f, 0.0f);
        rotationMatrix_z = vmath::rotate(angleCube, 0.0f, 0.0f, 1.0f);
        rotationMatrix = rotationMatrix_x * rotationMatrix_y * rotationMatrix_z;
        modelViewMatrix = translationMatrix * scaleMatrix * rotationMatrix;
        modelViewProjectionMatrix = perspectiveProjectionMatrix * modelViewMatrix;

        glUniformMatrix4fv(mvpMatrixUniform, 1, GL_FALSE, modelViewProjectionMatrix);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, fbo_texture);
        {
            glUniform1i(textureSamplerUniform, 0);
            glBindVertexArray(vao_cube);
            {
                glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
                glDrawArrays(GL_TRIANGLE_FAN, 4, 4);
                glDrawArrays(GL_TRIANGLE_FAN, 8, 4);
                glDrawArrays(GL_TRIANGLE_FAN, 12, 4);
                glDrawArrays(GL_TRIANGLE_FAN, 16, 4);
                glDrawArrays(GL_TRIANGLE_FAN, 20, 4);
                glDrawArrays(GL_TRIANGLE_FAN, 24, 4);
            }
            glBindVertexArray(0);
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);

    SwapBuffers(ghdc);
}

void displaySphere(GLint textureWidth, GLint textureHeight)
{
    // Code
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    resizeSphere(textureWidth, textureHeight);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (chosenShader == 'v')
        glUseProgram(shaderProgramObject_pv);
    else if (chosenShader == 'f')
        glUseProgram(shaderProgramObject_pf);
    {
        // Transformations
        vmath::mat4 translationMatrix = vmath::mat4::identity();
        vmath::mat4 modelMatrix = vmath::mat4::identity();
        vmath::mat4 viewMatrix = vmath::mat4::identity();

        translationMatrix = vmath::translate(0.0f, 0.0f, -5.0f);
        modelMatrix = translationMatrix;

        // Light 0
        lights[0].lightPosition[2] = radius * cos(vmath::radians(lightAngleZero));
        lights[0].lightPosition[1] = radius * sin(vmath::radians(lightAngleZero));

        // Light 1
        lights[1].lightPosition[0] = radius * cos(vmath::radians(lightAngleOne));
        lights[1].lightPosition[2] = radius * sin(vmath::radians(lightAngleOne));

        // Light 2
        lights[2].lightPosition[1] = radius * sin(vmath::radians(lightAngleTwo));
        lights[2].lightPosition[0] = radius * cos(vmath::radians(lightAngleTwo));


        if (chosenShader == 'v')
        {       
            glUniformMatrix4fv(modelMatrixUniform_pv, 1, GL_FALSE, modelMatrix);
            glUniformMatrix4fv(viewMatrixUniform_pv, 1, GL_FALSE, viewMatrix);
            glUniformMatrix4fv(projectionMatrixUniform_pv, 1, GL_FALSE, perspectiveProjectionMatrix_sphere);

            if (bLight)
            {
                glUniform1i(lightEnabledUniform_pv, 1);

                for (int i = 0; i < 3; i++)
                {
                    glUniform3fv(laUniform_pv[i], 1, lights[i].lightAmbient);
                    glUniform3fv(ldUniform_pv[i], 1, lights[i].lightDiffuse);
                    glUniform3fv(lsUniform_pv[i], 1, lights[i].lightSpecular);
                    glUniform4fv(lightPositionUniform_pv[i], 1, lights[i].lightPosition);
                }
                
                glUniform3fv(kaUniform_pv, 1, materialAmbient);
                glUniform3fv(kdUniform_pv, 1, materialDiffuse);
                glUniform3fv(ksUniform_pv, 1, materialSpecular);
                glUniform1f(materialShininessUniform_pv, materialShininess);
            }
            else
                glUniform1i(lightEnabledUniform_pv, 0);
        }
        else if (chosenShader == 'f')
        {       
            glUniformMatrix4fv(modelMatrixUniform_pf, 1, GL_FALSE, modelMatrix);
            glUniformMatrix4fv(viewMatrixUniform_pf, 1, GL_FALSE, viewMatrix);
            glUniformMatrix4fv(projectionMatrixUniform_pf, 1, GL_FALSE, perspectiveProjectionMatrix_sphere);

            if (bLight)
            {
                glUniform1i(lightEnabledUniform_pf, 1);

                for (int i = 0; i < 3; i++)
                {
                    glUniform3fv(laUniform_pf[i], 1, lights[i].lightAmbient);
                    glUniform3fv(ldUniform_pf[i], 1, lights[i].lightDiffuse);
                    glUniform3fv(lsUniform_pf[i], 1, lights[i].lightSpecular);
                    glUniform4fv(lightPositionUniform_pf[i], 1, lights[i].lightPosition);
                }

                glUniform3fv(kaUniform_pf, 1, materialAmbient);
                glUniform3fv(kdUniform_pf, 1, materialDiffuse);
                glUniform3fv(ksUniform_pf, 1, materialSpecular);
                glUniform1f(materialShininessUniform_pf, materialShininess);
            }
            else
                glUniform1i(lightEnabledUniform_pf, 0);
        }

        glBindVertexArray(vao_sphere);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_sphere_indices);
        {
            glDrawElements(
                GL_TRIANGLES, 
                gNumIndices, 
                GL_UNSIGNED_INT,
                0
            );
        }
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
    glUseProgram(0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void update(void)
{
    // Code
    angleCube += animationSpeed;
    if (angleCube >= 360.0f)
        angleCube = 0.0f;
}

void updateSphere(void)
{
    // Code
    lightAngleZero += animationSpeed;
    if (lightAngleZero >= 360.0f)
        lightAngleZero = 0.0f;

    lightAngleOne += animationSpeed;
    if (lightAngleOne >= 360.0f)
        lightAngleOne = 0.0f;

    lightAngleTwo += animationSpeed;
    if (lightAngleTwo >= 360.0f)
        lightAngleTwo = 0.0f;
}

void uninitialize(void)
{
    // Function Declarations
    void ToggleFullScreen(void);
    void uninitializeSphere(void);

    // Code
    if (gbFullScreen)
        ToggleFullScreen();

    uninitializeSphere();

    if (vbo_cube_texcoord)
    {
        glDeleteBuffers(1, &vbo_cube_texcoord);
        vbo_cube_texcoord = 0;
    }

    if (vbo_cube_position)
    {
        glDeleteBuffers(1, &vbo_cube_position);
        vbo_cube_position = 0;
    }

    if (vao_cube)
    {
        glDeleteVertexArrays(1, &vao_cube);
        vao_cube = 0;
    }

     if (shaderProgramObject)
    {
        glUseProgram(shaderProgramObject);
        {
            GLsizei numAttachedShaders;
            glGetProgramiv(shaderProgramObject, GL_ATTACHED_SHADERS, &numAttachedShaders);
            
            GLuint* shaderObjects = NULL;
            shaderObjects = (GLuint*)malloc(numAttachedShaders * sizeof(GLuint));
            if (shaderObjects == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to shaderObjects for Shader Program Log !!!\n", __func__);
                uninitialize();
            }

            glGetAttachedShaders(shaderProgramObject, numAttachedShaders, &numAttachedShaders, shaderObjects);

            for (GLsizei i = 0; i < numAttachedShaders; i++)
            {
                glDetachShader(shaderProgramObject, shaderObjects[i]);
                glDeleteShader(shaderObjects[i]);
                shaderObjects[i] = 0;
            }
            free(shaderObjects);
            shaderObjects = NULL;
        }
        glUseProgram(0);
        glDeleteProgram(shaderProgramObject);
        shaderProgramObject = 0;
    }
    
    if (wglGetCurrentContext() == ghrc)
        wglMakeCurrent(NULL, NULL);

    if (ghrc)
    {
        wglDeleteContext(ghrc);
        ghrc = NULL;
    }

    if (ghdc)
    {
        ReleaseDC(ghwnd, ghdc);
        ghdc = NULL;
    }
    
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

void uninitializeSphere(void)
{
    // Code
    if (fbo_texture)
    {
        glDeleteBuffers(1, &fbo_texture);
        fbo_texture = 0;
    }

    if (rbo)
    {
        glDeleteRenderbuffers(1, &rbo);
        rbo = 0;
    }

    if (fbo)
    {
        glDeleteFramebuffers(1, &fbo);
        fbo = 0;
    }

    if (sphere)
    {
        delete sphere;
        sphere = nullptr;
    }

    if (ebo_sphere_indices)
    {
        glDeleteBuffers(1, &ebo_sphere_indices);
        ebo_sphere_indices = 0;
    }

    if (vbo_sphere_normal)
    {
        glDeleteBuffers(1, &vbo_sphere_normal);
        vbo_sphere_normal = 0;
    }

    if (vbo_sphere_position)
    {
        glDeleteBuffers(1, &vbo_sphere_position);
        vbo_sphere_position = 0;
    }

    if (vao_sphere)
    {
        glDeleteVertexArrays(1, &vao_sphere);
        vao_sphere = 0;
    }

    if (shaderProgramObject_pf)
    {
        glUseProgram(shaderProgramObject_pf);
        {
            GLsizei numAttachedShaders;
            glGetProgramiv(shaderProgramObject_pf, GL_ATTACHED_SHADERS, &numAttachedShaders);
            
            GLuint* shaderObjects = NULL;
            shaderObjects = (GLuint*)malloc(numAttachedShaders * sizeof(GLuint));
            if (shaderObjects == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to shaderObjects for Per-Fragment Shader Program Log !!!\n", __func__);
                uninitialize();
            }

            glGetAttachedShaders(shaderProgramObject_pf, numAttachedShaders, &numAttachedShaders, shaderObjects);

            for (GLsizei i = 0; i < numAttachedShaders; i++)
            {
                glDetachShader(shaderProgramObject_pf, shaderObjects[i]);
                glDeleteShader(shaderObjects[i]);
                shaderObjects[i] = 0;
            }
            free(shaderObjects);
            shaderObjects = NULL;
        }
        glUseProgram(0);
        glDeleteProgram(shaderProgramObject_pf);
        shaderProgramObject_pf = 0;
    }

    if (shaderProgramObject_pv)
    {
        glUseProgram(shaderProgramObject_pv);
        {
            GLsizei numAttachedShaders;
            glGetProgramiv(shaderProgramObject_pv, GL_ATTACHED_SHADERS, &numAttachedShaders);
            
            GLuint* shaderObjects = NULL;
            shaderObjects = (GLuint*)malloc(numAttachedShaders * sizeof(GLuint));
            if (shaderObjects == NULL)
            {
                fprintf(gpFile, "ERROR : %s() => Failed to allocate memory to shaderObjects for Per-Vertex Shader Program Log !!!\n", __func__);
                uninitialize();
            }

            glGetAttachedShaders(shaderProgramObject_pv, numAttachedShaders, &numAttachedShaders, shaderObjects);

            for (GLsizei i = 0; i < numAttachedShaders; i++)
            {
                glDetachShader(shaderProgramObject_pv, shaderObjects[i]);
                glDeleteShader(shaderObjects[i]);
                shaderObjects[i] = 0;
            }
            free(shaderObjects);
            shaderObjects = NULL;
        }
        glUseProgram(0);
        glDeleteProgram(shaderProgramObject_pv);
        shaderProgramObject_pv = 0;
    }
}
