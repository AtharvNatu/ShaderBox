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
    MEM_ALLOC_FAILED = -12
};

enum ATTRIBUTES
{
    ATTRIBUTE_POSITION = 0,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEXTURE0
};

//* Shaders, VAO & VBO
//* -----------------------------------------------------------------------
GLuint shaderProgramObject = 0;

GLuint vao_sphere = 0;
GLuint vbo_sphere_position = 0;
GLuint vbo_sphere_normal = 0;
GLuint ebo_sphere_indices = 0;
//* -----------------------------------------------------------------------

// Uniforms
// -----------------------------------------------------------------------
GLuint modelMatrixUniform = 0;
GLuint viewMatrixUniform = 0;
GLuint projectionMatrixUniform = 0;
// -----------------------------------------------------------------------

// Light Related
// -----------------------------------------------------------------------
GLuint laUniform;   //? Light Ambient
GLuint ldUniform;   //? Light Diffuse
GLuint lsUniform;   //? Light Specular
GLuint lightPositionUniform;

GLuint kaUniform;   //? Material Ambient
GLuint kdUniform;   //? Material Diffuse
GLuint ksUniform;   //? Material Specular
GLuint materialShininessUniform;

GLuint lightEnabledUniform;

BOOL bLight = FALSE;

GLfloat lightAmbient[] = { 0.0f, 0.0f, 0.0f, 1.0f };
GLfloat lightDiffuse[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightSpecular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
GLfloat lightPosition[] = { 0.0f, 0.0f, 0.0f, 1.0f };

struct Material
{
    GLfloat materialAmbient[4];
    GLfloat materialDiffuse[4];
    GLfloat materialSpecular[4];
    GLfloat materialShininess;
};

Material materials[6][4];

GLint giWindowWidth = 0, giWindowHeight = 0;

GLint keyPressed = 0;

GLfloat angleForXRotation = 0.0f;
GLfloat angleForYRotation = 0.0f;
GLfloat angleForZRotation = 0.0f;

const GLfloat rotationRadius = 40.0f;
const GLfloat animationSpeed = 0.75f;
// -----------------------------------------------------------------------

//! Create Sphere Object
Sphere* sphere = nullptr;

vmath::mat4 perspectiveProjectionMatrix;

GLuint gNumIndices = 0;

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
        TEXT("OpenGL : 24 Spheres"),
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

                case 'L':
                case 'l':
                    bLight = !bLight;
                break;

                case 'X':
                case 'x':
                    keyPressed = 1;
                    angleForXRotation = 0.0f;
                break;

                case 'Y':
                case 'y':
                    keyPressed = 2;
                    angleForYRotation = 0.0f;
                break;

                case 'Z':
                case 'z':
                    keyPressed = 3;
                    angleForZRotation = 0.0f;
                break;

                default:
                    keyPressed = 0;
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
    void initializeMaterials(void);

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
        "in vec3 a_normal;" \

        "uniform mat4 u_modelMatrix;" \
        "uniform mat4 u_viewMatrix;" \
        "uniform mat4 u_projectionMatrix;" \

        "uniform vec4 u_lightPosition;" \
        "uniform int u_lightEnabled;" \

        "out vec3 transformedNormals;" \
        "out vec3 lightDirection;" \
        "out vec3 viewerVector;" \

        "void main(void)" \
        "{" \
            "if (u_lightEnabled == 1)" \
            "{" \
                "vec4 eyeCoordinates = u_viewMatrix * u_modelMatrix * a_position;" \
                "mat3 normalMatrix = mat3(u_viewMatrix * u_modelMatrix);" \
                "transformedNormals = normalMatrix * a_normal;" \
                "lightDirection = vec3(u_lightPosition) - eyeCoordinates.xyz;" \
                "viewerVector = -eyeCoordinates.xyz;" \
            "}" \

            "gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * a_position;" \
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

        "in vec3 transformedNormals;" \
        "in vec3 lightDirection;" \
        "in vec3 viewerVector;" \

        "uniform vec3 u_la;" \
        "uniform vec3 u_ld;" \
        "uniform vec3 u_ls;" \

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
                "vec3 normalized_transformed_normals = normalize(transformedNormals);" \
                "vec3 normalized_light_direction = normalize(lightDirection);" \
                "vec3 normalized_viewer_vector = normalize(viewerVector);" \

                "vec3 ambient = u_la * u_ka;" \
                "vec3 diffuse = u_ld * u_kd * max(dot(normalized_light_direction, normalized_transformed_normals), 0.0);" \
                "vec3 reflectionVector = reflect(-normalized_light_direction, normalized_transformed_normals);" \
                "vec3 specular = u_ls * u_ks * pow(max(dot(reflectionVector, normalized_viewer_vector), 0.0), u_materialShininess);" \
                "phong_ads_light = ambient + diffuse + specular;" \
            "}" \
            "else" \
            "{" \
                "phong_ads_light = vec3(1.0, 1.0, 1.0);" \
            "}" \

            "FragColor = vec4(phong_ads_light, 1.0);" \
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
    glBindAttribLocation(shaderProgramObject, ATTRIBUTE_NORMAL, "a_normal");

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
    modelMatrixUniform = glGetUniformLocation(shaderProgramObject, "u_modelMatrix");
    viewMatrixUniform = glGetUniformLocation(shaderProgramObject, "u_viewMatrix");
    projectionMatrixUniform = glGetUniformLocation(shaderProgramObject, "u_projectionMatrix");

    laUniform = glGetUniformLocation(shaderProgramObject, "u_la");
    ldUniform = glGetUniformLocation(shaderProgramObject, "u_ld");
    lsUniform = glGetUniformLocation(shaderProgramObject, "u_ls");
    lightPositionUniform = glGetUniformLocation(shaderProgramObject, "u_lightPosition");

    kaUniform = glGetUniformLocation(shaderProgramObject, "u_ka");
    kdUniform = glGetUniformLocation(shaderProgramObject, "u_kd");
    ksUniform = glGetUniformLocation(shaderProgramObject, "u_ks");
    materialShininessUniform = glGetUniformLocation(shaderProgramObject, "u_materialShininess");
    
    lightEnabledUniform = glGetUniformLocation(shaderProgramObject, "u_lightEnabled");

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

    // Depth Related Code
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Clear the screen using gray color
    glClearColor(0.3f, 0.3f, 0.3f, 1.0f);

    initializeMaterials();

    perspectiveProjectionMatrix = vmath::mat4::identity();

    // Warmup resize call
    resize(WIN_WIDTH, WIN_HEIGHT);

    return 0;
}

void initializeMaterials(void)
{
    // Gems
    // ***** 1st sphere on 1st column, Emerald *****
    // ambient material
    materials[5][0].materialAmbient[0] = 0.0215; // r
    materials[5][0].materialAmbient[1] = 0.1745; // g
    materials[5][0].materialAmbient[2] = 0.0215; // b
    materials[5][0].materialAmbient[3] = 1.0;   // a

    // diffuse material
    materials[5][0].materialDiffuse[0] = 0.07568; // r
    materials[5][0].materialDiffuse[1] = 0.61424; // g
    materials[5][0].materialDiffuse[2] = 0.07568; // b
    materials[5][0].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[5][0].materialSpecular[0] = 0.633;    // r
    materials[5][0].materialSpecular[1] = 0.727811; // g
    materials[5][0].materialSpecular[2] = 0.633;    // b
    materials[5][0].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[5][0].materialShininess = 0.6 * 128;

    // ***** 2nd sphere on 1st column, jade *****
    // ambient material
    materials[4][0].materialAmbient[0] = 0.135;  // r
    materials[4][0].materialAmbient[1] = 0.2225; // g
    materials[4][0].materialAmbient[2] = 0.1575; // b
    materials[4][0].materialAmbient[3] = 1.0;   // a

    // diffuse material
    materials[4][0].materialDiffuse[0] = 0.54; // r
    materials[4][0].materialDiffuse[1] = 0.89; // g
    materials[4][0].materialDiffuse[2] = 0.63; // b
    materials[4][0].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[4][0].materialSpecular[0] = 0.316228; // r
    materials[4][0].materialSpecular[1] = 0.316228; // g
    materials[4][0].materialSpecular[2] = 0.316228; // b
    materials[4][0].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[4][0].materialShininess = 0.1 * 128.0;

    // ***** 3rd sphere on 1st column, obsidian *****
    // ambient material
    materials[3][0].materialAmbient[0] = 0.05375; // r
    materials[3][0].materialAmbient[1] = 0.05;    // g
    materials[3][0].materialAmbient[2] = 0.06625; // b
    materials[3][0].materialAmbient[3] = 1.0;    // a

    // diffuse material
    materials[3][0].materialDiffuse[0] = 0.18275; // r
    materials[3][0].materialDiffuse[1] = 0.17;    // g
    materials[3][0].materialDiffuse[2] = 0.22525; // b
    materials[3][0].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[3][0].materialSpecular[0] = 0.332741; // r
    materials[3][0].materialSpecular[1] = 0.328634; // g
    materials[3][0].materialSpecular[2] = 0.346435; // b
    materials[3][0].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[3][0].materialShininess = 0.3 * 128.0;

    // ***** 4th sphere on 1st column, pearl *****
    // ambient material
    materials[2][0].materialAmbient[0] = 0.25;    // r
    materials[2][0].materialAmbient[1] = 0.20725; // g
    materials[2][0].materialAmbient[2] = 0.20725; // b
    materials[2][0].materialAmbient[3] = 1.0;    // a

    // diffuse material
    materials[2][0].materialDiffuse[0] = 1.0;   // r
    materials[2][0].materialDiffuse[1] = 0.829; // g
    materials[2][0].materialDiffuse[2] = 0.829; // b
    materials[2][0].materialDiffuse[3] = 1.0;  // a

    // specular material
    materials[2][0].materialSpecular[0] = 0.296648; // r
    materials[2][0].materialSpecular[1] = 0.296648; // g
    materials[2][0].materialSpecular[2] = 0.296648; // b
    materials[2][0].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[2][0].materialShininess = 0.088 * 128.0;

    // ***** 5th sphere on 1st column, ruby *****
    // ambient material
    materials[1][0].materialAmbient[0] = 0.1745;  // r
    materials[1][0].materialAmbient[1] = 0.01175; // g
    materials[1][0].materialAmbient[2] = 0.01175; // b
    materials[1][0].materialAmbient[3] = 1.0;    // a

    // diffuse material
    materials[1][0].materialDiffuse[0] = 0.61424; // r
    materials[1][0].materialDiffuse[1] = 0.04136; // g
    materials[1][0].materialDiffuse[2] = 0.04136; // b
    materials[1][0].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[1][0].materialSpecular[0] = 0.727811; // r
    materials[1][0].materialSpecular[1] = 0.626959; // g
    materials[1][0].materialSpecular[2] = 0.626959; // b
    materials[1][0].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[1][0].materialShininess = 0.6 * 128.0;

    // ***** 6th sphere on 1st column, turquoise *****
    // ambient material
    materials[0][0].materialAmbient[0] = 0.1;     // r
    materials[0][0].materialAmbient[1] = 0.18725; // g
    materials[0][0].materialAmbient[2] = 0.1745;  // b
    materials[0][0].materialAmbient[3] = 1.0;    // a

    // diffuse material
    materials[0][0].materialDiffuse[0] = 0.396;   // r
    materials[0][0].materialDiffuse[1] = 0.74151; // g
    materials[0][0].materialDiffuse[2] = 0.69102; // b
    materials[0][0].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[0][0].materialSpecular[0] = 0.297254; // r
    materials[0][0].materialSpecular[1] = 0.30829;  // g
    materials[0][0].materialSpecular[2] = 0.306678; // b
    materials[0][0].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[0][0].materialShininess = 0.1 * 128.0;

    // ***** 1st sphere on 2nd column, brass *****
    // ambient material
    materials[5][1].materialAmbient[0] = 0.329412; // r
    materials[5][1].materialAmbient[1] = 0.223529; // g
    materials[5][1].materialAmbient[2] = 0.027451; // b
    materials[5][1].materialAmbient[3] = 1.0;     // a

    // diffuse material
    materials[5][1].materialDiffuse[0] = 0.780392; // r
    materials[5][1].materialDiffuse[1] = 0.568627; // g
    materials[5][1].materialDiffuse[2] = 0.113725; // b
    materials[5][1].materialDiffuse[3] = 1.0;     // a

    // specular material
    materials[5][1].materialSpecular[0] = 0.992157; // r
    materials[5][1].materialSpecular[1] = 0.941176; // g
    materials[5][1].materialSpecular[2] = 0.807843; // b
    materials[5][1].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[5][1].materialShininess = 0.21794872 * 128.0;

    // ***** 2nd sphere on 2nd column, bronze *****
    // ambient material
    materials[4][1].materialAmbient[0] = 0.2125; // r
    materials[4][1].materialAmbient[1] = 0.1275; // g
    materials[4][1].materialAmbient[2] = 0.054;  // b
    materials[4][1].materialAmbient[3] = 1.0;   // a

    // diffuse material
    materials[4][1].materialDiffuse[0] = 0.714;   // r
    materials[4][1].materialDiffuse[1] = 0.4284;  // g
    materials[4][1].materialDiffuse[2] = 0.18144; // b
    materials[4][1].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[4][1].materialSpecular[0] = 0.393548; // r
    materials[4][1].materialSpecular[1] = 0.271906; // g
    materials[4][1].materialSpecular[2] = 0.166721; // b
    materials[4][1].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[4][1].materialShininess = 0.2 * 128.0;

    // ***** 3rd sphere on 2nd column, chrome *****
    // ambient material
    materials[3][1].materialAmbient[0] = 0.25; // r
    materials[3][1].materialAmbient[1] = 0.25; // g
    materials[3][1].materialAmbient[2] = 0.25; // b
    materials[3][1].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[3][1].materialDiffuse[0] = 0.4;  // r
    materials[3][1].materialDiffuse[1] = 0.4;  // g
    materials[3][1].materialDiffuse[2] = 0.4;  // b
    materials[3][1].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[3][1].materialSpecular[0] = 0.774597; // r
    materials[3][1].materialSpecular[1] = 0.774597; // g
    materials[3][1].materialSpecular[2] = 0.774597; // b
    materials[3][1].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[3][1].materialShininess = 0.6 * 128.0;

    // ***** 4th sphere on 2nd column, copper *****
    // ambient material
    materials[2][1].materialAmbient[0] = 0.19125; // r
    materials[2][1].materialAmbient[1] = 0.0735;  // g
    materials[2][1].materialAmbient[2] = 0.0225;  // b
    materials[2][1].materialAmbient[3] = 1.0;    // a

    // diffuse material
    materials[2][1].materialDiffuse[0] = 0.7038;  // r
    materials[2][1].materialDiffuse[1] = 0.27048; // g
    materials[2][1].materialDiffuse[2] = 0.0828;  // b
    materials[2][1].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[2][1].materialSpecular[0] = 0.256777; // r
    materials[2][1].materialSpecular[1] = 0.137622; // g
    materials[2][1].materialSpecular[2] = 0.086014; // b
    materials[2][1].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[2][1].materialShininess = 0.1 * 128.0;

    // ***** 5th sphere on 2nd column, gold *****
    // ambient material
    materials[1][1].materialAmbient[0] = 0.24725; // r
    materials[1][1].materialAmbient[1] = 0.1995;  // g
    materials[1][1].materialAmbient[2] = 0.0745;  // b
    materials[1][1].materialAmbient[3] = 1.0;    // a

    // diffuse material
    materials[1][1].materialDiffuse[0] = 0.75164; // r
    materials[1][1].materialDiffuse[1] = 0.60648; // g
    materials[1][1].materialDiffuse[2] = 0.22648; // b
    materials[1][1].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[1][1].materialSpecular[0] = 0.628281; // r
    materials[1][1].materialSpecular[1] = 0.555802; // g
    materials[1][1].materialSpecular[2] = 0.366065; // b
    materials[1][1].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[1][1].materialShininess = 0.4 * 128.0;


    // ***** 6th sphere on 2nd column, silver *****
    // ambient material
    materials[0][1].materialAmbient[0] = 0.19225; // r
    materials[0][1].materialAmbient[1] = 0.19225; // g
    materials[0][1].materialAmbient[2] = 0.19225; // b
    materials[0][1].materialAmbient[3] = 1.0;    // a

    // diffuse material
    materials[0][1].materialDiffuse[0] = 0.50754; // r
    materials[0][1].materialDiffuse[1] = 0.50754; // g
    materials[0][1].materialDiffuse[2] = 0.50754; // b
    materials[0][1].materialDiffuse[3] = 1.0;    // a

    // specular material
    materials[0][1].materialSpecular[0] = 0.508273; // r
    materials[0][1].materialSpecular[1] = 0.508273; // g
    materials[0][1].materialSpecular[2] = 0.508273; // b
    materials[0][1].materialSpecular[3] = 1.0;     // a

    // shininess
    materials[0][1].materialShininess = 0.4 * 128.0;

    // ***** 1st sphere on 3rd column, black *****
    // ambient material
    materials[5][2].materialAmbient[0] = 0.0;  // r
    materials[5][2].materialAmbient[1] = 0.0;  // g
    materials[5][2].materialAmbient[2] = 0.0;  // b
    materials[5][2].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[5][2].materialDiffuse[0] = 0.01; // r
    materials[5][2].materialDiffuse[1] = 0.01; // g
    materials[5][2].materialDiffuse[2] = 0.01; // b
    materials[5][2].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[5][2].materialSpecular[0] = 0.50; // r
    materials[5][2].materialSpecular[1] = 0.50; // g
    materials[5][2].materialSpecular[2] = 0.50; // b
    materials[5][2].materialSpecular[3] = 1.0; // a

    // shininess
    materials[5][2].materialShininess = 0.25 * 128.0;


    // ***** 2nd sphere on 3rd column, cyan *****
    // ambient material
    materials[4][2].materialAmbient[0] = 0.0;  // r
    materials[4][2].materialAmbient[1] = 0.1;  // g
    materials[4][2].materialAmbient[2] = 0.06; // b
    materials[4][2].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[4][2].materialDiffuse[0] = 0.0;        // r
    materials[4][2].materialDiffuse[1] = 0.50980392; // g
    materials[4][2].materialDiffuse[2] = 0.50980392; // b
    materials[4][2].materialDiffuse[3] = 1.0;       // a

    // specular material
    materials[4][2].materialSpecular[0] = 0.50196078; // r
    materials[4][2].materialSpecular[1] = 0.50196078; // g
    materials[4][2].materialSpecular[2] = 0.50196078; // b
    materials[4][2].materialSpecular[3] = 1.0;       // a

    // shininess
    materials[4][2].materialShininess = 0.25 * 128.0;

    // ***** 3rd sphere on 2nd column, green *****
    // ambient material
    materials[3][2].materialAmbient[0] = 0.0;  // r
    materials[3][2].materialAmbient[1] = 0.0;  // g
    materials[3][2].materialAmbient[2] = 0.0;  // b
    materials[3][2].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[3][2].materialDiffuse[0] = 0.1;  // r
    materials[3][2].materialDiffuse[1] = 0.35; // g
    materials[3][2].materialDiffuse[2] = 0.1;  // b
    materials[3][2].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[3][2].materialSpecular[0] = 0.45; // r
    materials[3][2].materialSpecular[1] = 0.55; // g
    materials[3][2].materialSpecular[2] = 0.45; // b
    materials[3][2].materialSpecular[3] = 1.0; // a

    // shininess
    materials[3][2].materialShininess = 0.25 * 128.0;

    // ***** 4th sphere on 3rd column, red *****
    // ambient material
    materials[2][2].materialAmbient[0] = 0.0;  // r
    materials[2][2].materialAmbient[1] = 0.0;  // g
    materials[2][2].materialAmbient[2] = 0.0;  // b
    materials[2][2].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[2][2].materialDiffuse[0] = 0.5;  // r
    materials[2][2].materialDiffuse[1] = 0.0;  // g
    materials[2][2].materialDiffuse[2] = 0.0;  // b
    materials[2][2].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[2][2].materialSpecular[0] = 0.7;  // r
    materials[2][2].materialSpecular[1] = 0.6;  // g
    materials[2][2].materialSpecular[2] = 0.6;  // b
    materials[2][2].materialSpecular[3] = 1.0; // a

    // shininess
    materials[2][2].materialShininess = 0.25 * 128.0;

    // ***** 5th sphere on 3rd column, white *****
    // ambient material
    materials[1][2].materialAmbient[0] = 0.0;  // r
    materials[1][2].materialAmbient[1] = 0.0;  // g
    materials[1][2].materialAmbient[2] = 0.0;  // b
    materials[1][2].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[1][2].materialDiffuse[0] = 0.55; // r
    materials[1][2].materialDiffuse[1] = 0.55; // g
    materials[1][2].materialDiffuse[2] = 0.55; // b
    materials[1][2].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[1][2].materialSpecular[0] = 0.70; // r
    materials[1][2].materialSpecular[1] = 0.70; // g
    materials[1][2].materialSpecular[2] = 0.70; // b
    materials[1][2].materialSpecular[3] = 1.0; // a

    // shininess
    materials[1][2].materialShininess = 0.25 * 128.0;

    // ***** 6th sphere on 3rd column, yellow plastic *****
    // ambient material
    materials[0][2].materialAmbient[0] = 0.0;  // r
    materials[0][2].materialAmbient[1] = 0.0;  // g
    materials[0][2].materialAmbient[2] = 0.0;  // b
    materials[0][2].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[0][2].materialDiffuse[0] = 0.5;  // r
    materials[0][2].materialDiffuse[1] = 0.5;  // g
    materials[0][2].materialDiffuse[2] = 0.0;  // b
    materials[0][2].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[0][2].materialSpecular[0] = 0.60; // r
    materials[0][2].materialSpecular[1] = 0.60; // g
    materials[0][2].materialSpecular[2] = 0.50; // b
    materials[0][2].materialSpecular[3] = 1.0; // a

    // shininess
    materials[0][2].materialShininess = 0.25 * 128.0;

    // ***** 1st sphere on 4th column, black *****
    // ambient material
    materials[5][3].materialAmbient[0] = 0.02; // r
    materials[5][3].materialAmbient[1] = 0.02; // g
    materials[5][3].materialAmbient[2] = 0.02; // b
    materials[5][3].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[5][3].materialDiffuse[0] = 0.01; // r
    materials[5][3].materialDiffuse[1] = 0.01; // g
    materials[5][3].materialDiffuse[2] = 0.01; // b
    materials[5][3].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[5][3].materialSpecular[0] = 0.4;  // r
    materials[5][3].materialSpecular[1] = 0.4;  // g
    materials[5][3].materialSpecular[2] = 0.4;  // b
    materials[5][3].materialSpecular[3] = 1.0; // a

    // shininess
    materials[5][3].materialShininess = 0.078125 * 128.0;

    // ***** 2nd sphere on 4th column, cyan *****
    // ambient material
    materials[4][3].materialAmbient[0] = 0.0;  // r
    materials[4][3].materialAmbient[1] = 0.05; // g
    materials[4][3].materialAmbient[2] = 0.05; // b
    materials[4][3].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[4][3].materialDiffuse[0] = 0.4;  // r
    materials[4][3].materialDiffuse[1] = 0.5;  // g
    materials[4][3].materialDiffuse[2] = 0.5;  // b
    materials[4][3].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[4][3].materialSpecular[0] = 0.04; // r
    materials[4][3].materialSpecular[1] = 0.7;  // g
    materials[4][3].materialSpecular[2] = 0.7;  // b
    materials[4][3].materialSpecular[3] = 1.0; // a

    // shininess
    materials[4][3].materialShininess = 0.078125 * 128.0;


    // ***** 3rd sphere on 4th column, green *****
    // ambient material
    materials[3][3].materialAmbient[0] = 0.0;  // r
    materials[3][3].materialAmbient[1] = 0.05; // g
    materials[3][3].materialAmbient[2] = 0.0;  // b
    materials[3][3].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[3][3].materialDiffuse[0] = 0.4;  // r
    materials[3][3].materialDiffuse[1] = 0.5;  // g
    materials[3][3].materialDiffuse[2] = 0.4;  // b
    materials[3][3].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[3][3].materialSpecular[0] = 0.04; // r
    materials[3][3].materialSpecular[1] = 0.7;  // g
    materials[3][3].materialSpecular[2] = 0.04; // b
    materials[3][3].materialSpecular[3] = 1.0; // a

    // shininess
    materials[3][3].materialShininess = 0.078125 * 128.0;

    // ***** 4th sphere on 4th column, red *****
    // ambient material
    materials[2][3].materialAmbient[0] = 0.05; // r
    materials[2][3].materialAmbient[1] = 0.0;  // g
    materials[2][3].materialAmbient[2] = 0.0;  // b
    materials[2][3].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[2][3].materialDiffuse[0] = 0.5;  // r
    materials[2][3].materialDiffuse[1] = 0.4;  // g
    materials[2][3].materialDiffuse[2] = 0.4;  // b
    materials[2][3].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[2][3].materialSpecular[0] = 0.7;  // r
    materials[2][3].materialSpecular[1] = 0.04; // g
    materials[2][3].materialSpecular[2] = 0.04; // b
    materials[2][3].materialSpecular[3] = 1.0; // a

    // shininess
    materials[2][3].materialShininess = 0.078125 * 128.0;


    // ***** 5th sphere on 4th column, white *****
    // ambient material
    materials[1][3].materialAmbient[0] = 0.05; // r
    materials[1][3].materialAmbient[1] = 0.05; // g
    materials[1][3].materialAmbient[2] = 0.05; // b
    materials[1][3].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[1][3].materialDiffuse[0] = 0.5;  // r
    materials[1][3].materialDiffuse[1] = 0.5;  // g
    materials[1][3].materialDiffuse[2] = 0.5;  // b
    materials[1][3].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[1][3].materialSpecular[0] = 0.7;  // r
    materials[1][3].materialSpecular[1] = 0.7;  // g
    materials[1][3].materialSpecular[2] = 0.7;  // b
    materials[1][3].materialSpecular[3] = 1.0; // a

    // shininess
    materials[1][3].materialShininess = 0.078125 * 128.0;

    // ***** 6th sphere on 4th column, yellow rubber *****
    // ambient material
    materials[0][3].materialAmbient[0] = 0.05; // r
    materials[0][3].materialAmbient[1] = 0.05; // g
    materials[0][3].materialAmbient[2] = 0.0;  // b
    materials[0][3].materialAmbient[3] = 1.0; // a

    // diffuse material
    materials[0][3].materialDiffuse[0] = 0.5;  // r
    materials[0][3].materialDiffuse[1] = 0.5;  // g
    materials[0][3].materialDiffuse[2] = 0.4;  // b
    materials[0][3].materialDiffuse[3] = 1.0; // a

    // specular material
    materials[0][3].materialSpecular[0] = 0.7;  // r
    materials[0][3].materialSpecular[1] = 0.7;  // g
    materials[0][3].materialSpecular[2] = 0.04; // b
    materials[0][3].materialSpecular[3] = 1.0; // a

    // shininess
    materials[0][3].materialShininess = 0.078125 * 128.0;
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

void resize(int width, int height)
{
    // Code
    if (height <= 0)
        height = 1;

    giWindowHeight = (GLint)height;
    giWindowWidth = (GLint)width;
    
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void)
{
    // Variable Declarations
    GLfloat viewportWidth = 0.0f;
    GLfloat viewportHeight = 0.0f;

    // Code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgramObject);
    {
        // Transformations
        vmath::mat4 translationMatrix = vmath::mat4::identity();
        vmath::mat4 modelMatrix = vmath::mat4::identity();
        vmath::mat4 viewMatrix = vmath::mat4::identity();

        translationMatrix = vmath::translate(0.0f, 0.0f, -6.0f);

        modelMatrix = translationMatrix;

        if (keyPressed == 1)
        {
            lightPosition[1] = rotationRadius * sin(vmath::radians(angleForXRotation));
            lightPosition[2] = rotationRadius * cos(vmath::radians(angleForXRotation));
        }
        else if (keyPressed == 2)
        {
            lightPosition[0] = rotationRadius * cos(vmath::radians(angleForYRotation));
            lightPosition[2] = rotationRadius * sin(vmath::radians(angleForYRotation));
        }
        else if (keyPressed == 3)
        {
            lightPosition[0] = rotationRadius * cos(vmath::radians(angleForZRotation));
            lightPosition[1] = rotationRadius * sin(vmath::radians(angleForZRotation));
        }
        else
        {
            lightPosition[0] = 0.0f;
        }

        glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
        glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, viewMatrix);
        glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix);

        if (bLight)
        {
            glUniform1i(lightEnabledUniform, 1);

            glUniform3fv(laUniform, 1, lightAmbient);
            glUniform3fv(ldUniform, 1, lightDiffuse);
            glUniform3fv(lsUniform, 1, lightSpecular);
            glUniform4fv(lightPositionUniform, 1, lightPosition);
        }
        else
            glUniform1i(lightEnabledUniform, 0);

        glBindVertexArray(vao_sphere);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_sphere_indices);

        viewportWidth = (GLfloat)giWindowWidth / 5.5f;
        viewportHeight = (GLfloat)giWindowHeight / 6.0f;
        
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                glViewport(j * viewportWidth, i * viewportHeight, viewportWidth, viewportHeight);

                glUniform3fv(kaUniform, 1, materials[i][j].materialAmbient);
                glUniform3fv(kdUniform, 1, materials[i][j].materialDiffuse);
                glUniform3fv(ksUniform, 1, materials[i][j].materialSpecular);
                glUniform1f(materialShininessUniform, materials[i][j].materialShininess);

                glDrawElements(GL_TRIANGLES, gNumIndices, GL_UNSIGNED_INT, 0);
            }
        }
        
        glBindVertexArray(0);
    }
    glUseProgram(0);

    SwapBuffers(ghdc);
}

void update(void)
{
    // Code
    if (keyPressed == 1)
    {
        angleForXRotation += animationSpeed;
        if (angleForXRotation >= 360.0f)
            angleForXRotation = 0.0f;
    }

    if (keyPressed == 2)
    {
        angleForYRotation += animationSpeed;
        if (angleForYRotation >= 360.0f)
            angleForYRotation = 0.0f;
    }

    if (keyPressed == 3)
    {
        angleForZRotation += animationSpeed;
        if (angleForZRotation >= 360.0f)
            angleForZRotation = 0.0f;
    }
}

void uninitialize(void)
{
    // Function Declarations
    void ToggleFullScreen(void);

    // Code
    if (gbFullScreen)
        ToggleFullScreen();

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
