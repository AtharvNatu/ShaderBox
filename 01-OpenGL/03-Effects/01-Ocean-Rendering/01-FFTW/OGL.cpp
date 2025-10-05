//* Header Files
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

#include "OGL.h"
#include "vmath.h"
#include "Ocean.hpp"

//! ImGui Related
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"

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

//? Shaders, VAO & VBO
//? -----------------------------------------------------------------------
GLuint shaderProgramObject = 0;

GLuint vao_ocean = 0;
GLuint vbo_ocean = 0;
GLuint ebo_ocean = 0;
//? -----------------------------------------------------------------------

//? Uniforms
//? -----------------------------------------------------------------------
GLuint modelMatrixUniform = 0;
GLuint viewMatrixUniform = 0;
GLuint projectionMatrixUniform = 0;

GLuint lightPositionUniform = 0;
GLuint lightAmbientUniform = 0;
GLuint lightDiffuseUniform = 0;
GLuint lightSpecularUniform = 0;

GLuint viewPositionUniform = 0;
GLuint heightMinUniform = 0;
GLuint heightMaxUniform = 0;

//? -----------------------------------------------------------------------

vmath::mat4 perspectiveProjectionMatrix;


//? Ocean Related
//? -----------------------------------------------------------------------
#define MESH_SIZE   128

int N = MESH_SIZE;
int M = MESH_SIZE;

int x_length = 1000;
int z_length = 1000;

float A = 3e-7f;
float V = 30;                // Wind Speed
vmath::vec2 omega(1, 1);     // Wind Direction
float fTime = 0.0f;
float heightMin = 0, heightMax = 0;

Ocean* ocean = nullptr;
GLuint numElements = 0;
GLuint* indices = nullptr;

vmath::vec3 lightPosition(0.0f, 50, 0.0);
vmath::vec3 lightDirection(vmath::normalize(vmath::vec3(0, 1, -2)));

bool bWireFrame = FALSE;

//? -----------------------------------------------------------------------

ImFont* font;

// Entry Point Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    // Function Declarations
    int initialize(void);
    void display(void);
    void update(void);
    void ToggleFullScreen(void);
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
        TEXT("ShaderBox-OpenGL : Ocean using FFTW"),
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

    ToggleFullScreen();

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

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// Callback Function
LRESULT CALLBACK WndProc(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
    // Function Declarations
    void ToggleFullScreen(void);
    void resize(int, int);
    void uninitialize(void);

    // Code
    if (ImGui_ImplWin32_WndProcHandler(hwnd, iMsg, wParam, lParam))
        return true;

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

        gbFullScreen = FALSE;
    }
}

int initialize(void)
{
    // Function Declarations
    void initializeImGui(const char*, float);
    void generateIndices(void);
    void resize(int, int);
    void printGLInfo(void);

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

    initializeImGui("ImGui\\Poppins-Regular.ttf", 20.0f);

    //! Vertex Shader
    //! ----------------------------------------------------------------------------
    const GLchar* vertexShaderSourceCode = 
        "#version 460 core" \
        "\n" \
        
        "in vec3 a_position;" \
        "in vec3 a_normal;" \

        "uniform mat4 u_modelMatrix;" \
        "uniform mat4 u_viewMatrix;" \
        "uniform mat4 u_projectionMatrix;" \

        "out vec3 out_normal;" \
        "out vec3 out_position;" \

        "void main(void)" \
        "{" \
            "gl_Position = u_projectionMatrix * u_viewMatrix * u_modelMatrix * vec4(a_position, 1.0f);" \
            
            "out_position = vec3(u_modelMatrix * vec4(a_position, 1.0f));" \
            "out_normal = mat3(transpose(inverse(u_modelMatrix))) * a_normal;" \
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

        "in vec3 out_position;" \
        "in vec3 out_normal;" \

        "uniform vec3 u_lightPosition;" \
        "uniform vec3 u_lightAmbient;" \
        "uniform vec3 u_lightDiffuse;" \
        "uniform vec3 u_lightSpecular;" \

        "uniform vec3 u_viewPosition;" \
        "uniform float u_heightMax = 0;" \
        "uniform float u_heightMin = 0;" \

        "out vec4 FragColor;" \

        "void main(void)" \
        "{" \
            "vec3 normalized_normals = normalize(out_normal);" \
            "vec3 light_direction = normalize(u_lightPosition - out_position);" \
            "vec3 view_direction = normalize(u_viewPosition - out_position);" \

            "vec3 ambientFactor = vec3(0.0);" \
            "vec3 diffuseFactor = vec3(1.0);" \
            "vec3 skyColor = vec3(0.65, 0.80, 0.95);" \

            "if (dot(normalized_normals, view_direction) < 0)" \
                "normalized_normals = -normalized_normals;" \
            
            // Ambient
            "vec3 ambient = u_lightAmbient * ambientFactor;" \

            "vec3 shallowColor = vec3(0.0, 0.64, 0.68);" \
            "vec3 deepColor = vec3(0.02, 0.05, 0.10);" \

            "float relativeHeight = (out_position.y - u_heightMin) / (u_heightMax - u_heightMin);" \
            "vec3 heightColor = relativeHeight * shallowColor + (1 - relativeHeight) * deepColor;" \

            // Spray
            "float sprayUpperThreshold = 1.0;" \
            "float sprayLowerThreshold = 0.9;" \
            "float sprayRatio = 0;" \
            "if (relativeHeight > sprayLowerThreshold)" \
                "sprayRatio = (relativeHeight - sprayLowerThreshold) / (sprayUpperThreshold - sprayLowerThreshold);" \
            "vec3 sprayBaseColor = vec3(1.0);" \
            "vec3 sprayColor = sprayRatio * sprayBaseColor;" \

            // Diffuse
            "float diff = max(dot(normalized_normals, view_direction), 0.0);" \
            "vec3 diffuse = diffuseFactor * u_lightDiffuse * diff;" \
            "diffuse = vec3(0.0);" \

            // Pseudo Reflect -> Smaller power will have more concentrated reflect.
            "float reflectionCoefficient = pow(max(dot(normalized_normals, view_direction), 0.0), 0.3);" \
            "vec3 reflectColor = (1 - reflectionCoefficient) * skyColor;" \

            // Specular
            "vec3 reflectionDirection = reflect(-light_direction, normalized_normals);" \
            "float specularCoefficient = pow(max(dot(view_direction, reflectionDirection), 0.0), 64) * 3;" \
            "vec3 specular = u_lightSpecular * specularCoefficient;" \

            "vec3 combinedColor = ambient + diffuse + heightColor + reflectColor;" \

            "specularCoefficient = clamp(specularCoefficient, 0, 1);" \
            "combinedColor *= (1 - specularCoefficient);" \
            "combinedColor += specular;" \

            "FragColor = vec4(combinedColor, 1.0);" \
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

    lightPositionUniform = glGetUniformLocation(shaderProgramObject, "u_lightPosition");
    viewPositionUniform = glGetUniformLocation(shaderProgramObject, "u_viewPosition");

    lightAmbientUniform = glGetUniformLocation(shaderProgramObject, "u_lightAmbient");
    lightDiffuseUniform = glGetUniformLocation(shaderProgramObject, "u_lightDiffuse");
    lightSpecularUniform = glGetUniformLocation(shaderProgramObject, "u_lightSpecular");

    heightMinUniform = glGetUniformLocation(shaderProgramObject, "u_heightMin");
    heightMaxUniform = glGetUniformLocation(shaderProgramObject, "u_heightMax");

    lightPosition = lightDirection * 50.0f;

    generateIndices();

    //! VAO and VBO Related Code

    // VAO For Ocean Surface
    glGenVertexArrays(1, &vao_ocean);
    glBindVertexArray(vao_ocean);
    {
        //* VBO
        glGenBuffers(1, &vbo_ocean);

        //* EBO
        glGenBuffers(1, &ebo_ocean);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_ocean);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, numElements * sizeof(GLuint), indices, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        delete[] indices;
        indices = nullptr;
    }
    glBindVertexArray(0);

    //! Ocean
    ocean = new Ocean(N, M, x_length, z_length, omega, V, A, 1);

    // Depth Related Code
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);


    // Clear the screen using black color
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

    perspectiveProjectionMatrix = vmath::mat4::identity();

    // Warmup resize call
    resize(WIN_WIDTH, WIN_HEIGHT);

    return 0;
}

void generateIndices()
{
    // Code
    int p = 0;

    numElements = (N - 1) * (M - 1) * 6;
    indices = new GLuint[numElements];

    for (int j = 0; j < N - 1; j++)
    {
        for (int i = 0; i < M - 1; i++)
        {
            indices[p++] = i + j * N;
            indices[p++] = (i + 1) + j * N;
            indices[p++] = i + (j + 1) * N;

            indices[p++] = (i + 1) + j * N;
            indices[p++] = (i + 1) + (j + 1) * N;
            indices[p++] = i + (j + 1) * N;
        }
    }
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
    
    glViewport(0, 0, (GLsizei)width, (GLsizei)height);

    perspectiveProjectionMatrix = vmath::perspective(45.0f, (GLfloat)width / (GLfloat)height, 0.1f, 100.0f);
}

void display(void)
{
    // Function Declarations;
    void renderImGui();

    // Code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(shaderProgramObject);
    {
        // Transformations
        vmath::mat4 translationMatrix = vmath::mat4::identity();
        vmath::mat4 scaleMatrix = vmath::mat4::identity();

        vmath::mat4 modelMatrix = vmath::mat4::identity();
        vmath::mat4 viewMatrix = vmath::mat4::identity();

        translationMatrix = vmath::translate(0.0f, -10.0f, -50.0f);
        scaleMatrix = vmath::scale(0.05f, 0.05f, 0.05f);
        modelMatrix = translationMatrix * scaleMatrix;

        glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
        glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, viewMatrix);
        glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix);
        
        glUniform3f(lightPositionUniform, lightPosition[0], lightPosition[1], lightPosition[2]);
        glUniform3f(viewPositionUniform, 30.0f, 30.0f, 60.0f);

        glUniform3f(lightAmbientUniform, 1.0f, 1.0f, 1.0f);
        glUniform3f(lightDiffuseUniform, 1.0f, 1.0f, 1.0f);
        glUniform3f(lightSpecularUniform, 1.0f, 0.9f, 0.7f);

        glUniform1f(heightMinUniform, heightMin * 0.1);
        glUniform1f(heightMaxUniform, heightMax * 0.1);

        glBindVertexArray(vao_ocean);
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_ocean);
            glDrawElements(GL_TRIANGLES, numElements, GL_UNSIGNED_INT, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        glBindVertexArray(0);
        
    }
    glUseProgram(0);

    //! Render ImGui Window on Top
    renderImGui();

    SwapBuffers(ghdc);
}

void update(void)
{
    // Function Declarations
    void buildTessendorfMesh(Ocean*);

    // Code
    fTime += 0.05f;

    buildTessendorfMesh(ocean);
}

void buildTessendorfMesh(Ocean* _ocean)
{
    // Code
    _ocean->generate_fft_data(fTime);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            int index = j * N + i;

            if (_ocean->displacement_map[index][1] > heightMax)
                heightMax = _ocean->displacement_map[index][1];
            else if (_ocean->displacement_map[index][1] < heightMin)
                heightMin = _ocean->displacement_map[index][1];
        }
    }

    glBindVertexArray(vao_ocean);
    {   
        //* Interleaved Vertex Buffer
        glBindBuffer(GL_ARRAY_BUFFER, vbo_ocean);
        {
            int meshSize = sizeof(vmath::vec3) * N * M;

            glBufferData(GL_ARRAY_BUFFER, meshSize * 2, NULL, GL_STATIC_DRAW);

            // Copy Displacement Data
            glBufferSubData(GL_ARRAY_BUFFER, 0, meshSize, _ocean->displacement_map);
            glVertexAttribPointer(ATTRIBUTE_POSITION, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
            glEnableVertexAttribArray(ATTRIBUTE_POSITION);

            // Copy Normals Data
            glBufferSubData(GL_ARRAY_BUFFER, meshSize, meshSize, _ocean->normal_map);
            glVertexAttribPointer(ATTRIBUTE_NORMAL, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)(uintptr_t)meshSize);
            glEnableVertexAttribArray(ATTRIBUTE_NORMAL);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);

}

void uninitialize(void)
{
    // Function Declarations
    void ToggleFullScreen(void);
    void uninitializeImGui(void);

    // Code
    if (gbFullScreen)
        ToggleFullScreen();

    if (ocean)
    {
        delete ocean;
        ocean = nullptr;
    }

    uninitializeImGui();

    if (ebo_ocean)
    {
        glDeleteBuffers(1, &ebo_ocean);
        ebo_ocean = 0;
    }

    if (vbo_ocean)
    {
        glDeleteBuffers(1, &vbo_ocean);
        vbo_ocean = 0;
    }

    if (vao_ocean)
    {
        glDeleteVertexArrays(1, &vao_ocean);
        vao_ocean = 0;
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



//! ImGui Related Functions
void initializeImGui(const char* fontFile, float fontSize)
{
    //! Setup ImGui Context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    //! Setup ImGui Style
    ImGui::StyleColorsDark();

    //! Setup Platform / Renderer Backends
    ImGui_ImplWin32_InitForOpenGL(ghwnd);
    ImGui_ImplOpenGL3_Init();

    io.Fonts->AddFontFromFileTTF(fontFile, fontSize, NULL, io.Fonts->GetGlyphRangesDefault());
}


void renderImGui(void)
{
    //! Start ImGui Frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    ImGui::SetWindowSize(ImVec2(486, 150));
    ImGui::Begin("ImGui");
    ImGui::PushFont(font);
    {
        if (ImGui::Checkbox("Wireframe", &bWireFrame))
            glPolygonMode(GL_FRONT_AND_BACK, bWireFrame ? GL_LINE : GL_FILL);
        // ImGui::SliderFloat("Cube Rotation Speed", (float*)&animationSpeed, 0.05f, 3.0f);
        ImGui::Text("FPS : %.1f", ImGui::GetIO().Framerate);
        ImGui::Text("GPU : %s", glGetString(GL_RENDERER));
    }
    ImGui::PopFont();
    ImGui::End();

    ImGui::Render();

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void uninitializeImGui(void)
{
    // Code
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}

