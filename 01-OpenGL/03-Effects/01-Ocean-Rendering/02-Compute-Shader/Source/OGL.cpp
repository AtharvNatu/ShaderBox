//* Header Files
#include <windows.h>
#include <windowsx.h>

//! ImGui Related
#include "imgui.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_win32.h"

#include "OGL.h"
#include "Common.hpp"
#include "Logger.hpp"
#include "Shader.hpp"
#include "Ocean.hpp"
#include "Texture.hpp"
#include "Camera.hpp"

#include "helper_timer.h"

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

WINDOWPLACEMENT wpPrev;
DWORD dwStyle;

Logger* logger = nullptr;

//? Shaders, VAO & VBO
//? -----------------------------------------------------------------------
Shader shader;

GLuint graphicsShaderProgramObject = 0, computeShaderProgramObject = 0;

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

GLuint foamIntensityUniform = 0;
GLuint subSurfaceScatteringStrengthUniform = 0;
GLuint fogDensityUniform = 0;
GLuint timeUniform = 0;
GLuint causticSamplerUniform = 0;

//? -----------------------------------------------------------------------

vmath::mat4 perspectiveProjectionMatrix;


//? Ocean Related
//? -----------------------------------------------------------------------
struct GridVertex
{
    vmath::vec3 position;
    vmath::vec2 texcoords;
};

const int GRID_SIZE = 1024;
const int RESOLUTION = 512;
const int WORK_GROUP_DIMENSION = 32;
const int OCEAN_SIZE = 1024;

float windMagnitude = 14.142135f;
float windAngle = 45.f;
float choppiness = 1.5f;
int sunElevation = 0;
int sunAzimuth = 90;

vmath::vec3 lightPosition(0.0f, 50, 0.0);
vmath::vec3 lightDirection(vmath::normalize(vmath::vec3(0, 1, -2)));

bool bWireFrame = false;
bool reloadSettings = false;

//! Camera
Camera camera(vmath::vec3(0.0f, 0.0f, 3.0f), vmath::vec3(0.0f, 1.0f, 0.0f));
float lastX = WIN_WIDTH / 2.0f;
float lastY = WIN_HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;
bool mouseInput = false;

//! Textures
Texture *initialSpectrumTexture = nullptr;
Texture *pingPhaseTexture = nullptr;
Texture *pongPhaseTexture = nullptr;
Texture *spectrumTexture = nullptr;
Texture *tempTexture = nullptr;
Texture *normalMap = nullptr;

StopWatchInterface* timer = nullptr;


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
    logger = Logger::getInstance("Ocean.log");
    
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
        TEXT("ShaderBox-OpenGL : Ocean using Compute Shader"),
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
            logger->printLog("ERROR : %s() => ChoosePixelFormat() Failed !!!\n", __func__);
            uninitialize();
        break;

        case SPF_ERROR:
            logger->printLog("ERROR : %s() => SetPixelFormat() Failed !!!\n", __func__);
            uninitialize();
        break;

        case WGL_CC_ERROR:
            logger->printLog("ERROR : %s() => Failed to create OpenGL context !!!\n", __func__);
            uninitialize();
        break;

        case WGL_MC_ERROR:
            logger->printLog("ERROR : %s() => Failed to make rendering context as current context !!!\n", __func__);
            uninitialize();
        break;

        case GLEW_INIT_ERROR:
            logger->printLog("ERROR : %s() => GLEW Initialization Failed !!!\n", __func__);
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
            logger->printLog("%s() => Initialization Completed Successfully\n", __func__);
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
                case 'F':
                case 'f':
                    ToggleFullScreen();
                break;
                
                case 'W':
                case 'w':
                    camera.processKeyboard(FORWARD, deltaTime);
                break;
                
                case 'S':
                case 's':
                    camera.processKeyboard(BACKWARD, deltaTime);
                break;
                
                case 'A':
                case 'a':
                    camera.processKeyboard(LEFT, deltaTime);
                break;
                
                case 'D':
                case 'd':
                    camera.processKeyboard(RIGHT, deltaTime);
                break;

                case VK_UP:
                    camera.processKeyboard(UP, deltaTime);
                break;
                case VK_DOWN:
                    camera.processKeyboard(DOWN, deltaTime);
                break;

                case VK_SPACE:
                    mouseInput = !mouseInput;
                break;

                case 27:
                    DestroyWindow(hwnd);
                break;

                default:
                break;
            }
        break;
        
        case WM_MOUSEWHEEL:
        {
            if (mouseInput)
            {
                float yOffset = GET_WHEEL_DELTA_WPARAM(wParam) / static_cast<float>(WHEEL_DELTA);
                camera.processMouseScroll(yOffset);
            }
        }  
        break;

        case WM_MOUSEMOVE:
        {
            if (mouseInput)
            {
                float xPosition = static_cast<float>(GET_X_LPARAM(lParam));
                float yPosition = static_cast<float>(GET_Y_LPARAM(lParam));

                if (firstMouse)
                {
                    lastX = xPosition;
                    lastY = yPosition;
                    firstMouse = false;
                }

                float xOffset = xPosition - lastX;
                float yOffset = lastY - yPosition;

                lastX = xPosition;
                lastY = yPosition;

                camera.processMouseMovement(xOffset, yOffset);
            }
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
    BOOL LoadPNGTexture(GLuint* texture, const char* imageFile);
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

    // printGLInfo();

    initializeImGui("ImGui\\Poppins-Regular.ttf", 20.0f);

    //! Generate Grid
    const int vertexCount = GRID_SIZE + 1;
    float texcoordScale = 2.0f;

    std::vector<GridVertex> vertices(vertexCount * vertexCount);
    std::vector<unsigned int> indices(GRID_SIZE * GRID_SIZE * 2 * 3);

    //* Vertex Data
    unsigned int idx = 0;
    for (int z = -GRID_SIZE / 2; z <= GRID_SIZE / 2; ++z)
    {
        for (int x = -GRID_SIZE / 2; x <= GRID_SIZE / 2; ++x)
        {
            vertices[idx].position = vmath::vec3(float(x), 0.0f, float(z));

            float u = ((float)x / GRID_SIZE) + 0.5f;
            float v = ((float)z / GRID_SIZE) + 0.5f;
            vertices[idx++].texcoords = vmath::vec2(u, v) * texcoordScale;
        }
    }
    assert(idx == vertices.size());

    //* Index Data (Clockwise Winding)
    idx = 0;
    for (unsigned int y = 0; y < GRID_SIZE; y++)
    {
        for (unsigned int x = 0; x < GRID_SIZE; x++)
        {
            indices[idx++] = (vertexCount * y) + x;
            indices[idx++] = (vertexCount * (y + 1)) + x;
            indices[idx++] = (vertexCount * y) + x + 1;

            indices[idx++] = (vertexCount * y) + x + 1;
            indices[idx++] = (vertexCount * (y + 1)) + x;
            indices[idx++] = (vertexCount * (y + 1)) + x + 1;
        }
    }
    assert(idx == indices.size());

    //! Shader Programs
    //! ----------------------------------------------------------------------------
    GLuint initialSpectrumComputeShaderObject = shader.createShaderObject(COMPUTE, "Shaders\\Initial-Spectrum.comp");
    GLuint phaseComputeShaderObject = shader.createShaderObject(COMPUTE, "Shaders\\Phase.comp");
    GLuint spectrumComputeShaderObject = shader.createShaderObject(COMPUTE, "Shaders\\Spectrum.comp");
    GLuint fftHorizontalComputeShaderObject = shader.createShaderObject(COMPUTE, "Shaders\\FFT-Horizontal.comp");
    GLuint fftVerticalComputeShaderObject = shader.createShaderObject(COMPUTE, "Shaders\\FFT-Vertical.comp");
    GLuint normalMapComputeShaderObject = shader.createShaderObject(COMPUTE, "Shaders\\Normal-Map.comp");

    GLuint vertexShaderObject = shader.createShaderObject(VERTEX, "Shaders\\Ocean.vert");
    GLuint fragmentShaderObject = shader.createShaderObject(FRAGMENT, "Shaders\\Ocean.frag");
    
    graphicsShaderProgramObject = glCreateProgram();
    computeShaderProgramObject = glCreateProgram();

    glAttachShader(computeShaderProgramObject, initialSpectrumComputeShaderObject);
    glAttachShader(computeShaderProgramObject, phaseComputeShaderObject);
    glAttachShader(computeShaderProgramObject, spectrumComputeShaderObject);
    glAttachShader(computeShaderProgramObject, fftHorizontalComputeShaderObject);
    glAttachShader(computeShaderProgramObject, fftVerticalComputeShaderObject);
    glAttachShader(computeShaderProgramObject, normalMapComputeShaderObject);

    glAttachShader(graphicsShaderProgramObject, vertexShaderObject);
    glAttachShader(graphicsShaderProgramObject, fragmentShaderObject);

    //! Bind Attribute
    glBindAttribLocation(graphicsShaderProgramObject, ATTRIBUTE_POSITION, "a_position");
    glBindAttribLocation(graphicsShaderProgramObject, ATTRIBUTE_TEXTURE0, "a_texcoord");
    glBindAttribLocation(graphicsShaderProgramObject, ATTRIBUTE_NORMAL, "a_normal");

    //! OpenGL Code

    //! Get Uniform Location
    modelMatrixUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_modelMatrix");
    viewMatrixUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_viewMatrix");
    projectionMatrixUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_projectionMatrix");

    lightPositionUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_lightPosition");
    viewPositionUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_viewPosition");

    lightAmbientUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_lightAmbient");
    lightDiffuseUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_lightDiffuse");
    lightSpecularUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_lightSpecular");

    heightMinUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_heightMin");
    heightMaxUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_heightMax");

    timeUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_time");
    foamIntensityUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_foamIntensity");
    subSurfaceScatteringStrengthUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_subSurfaceScatteringStrength");
    fogDensityUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_underWaterFogDensity");
    causticSamplerUniform = glGetUniformLocation(graphicsShaderProgramObject, "u_causticMap");

    lightPosition = lightDirection * 50.0f;

    // generateIndices();

    //! VAO and VBO Related Code

    // VAO For Ocean Surface
    glGenVertexArrays(1, &vao_ocean);
    glBindVertexArray(vao_ocean);
    {
        //* VBO
        glGenBuffers(1, &vbo_ocean);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_ocean);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(GridVertex), vertices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        //* EBO
        glGenBuffers(1, &ebo_ocean);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_ocean);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);

    //! Ocean Compute

    //* Initial Spectrum
    initialSpectrumTexture = new Texture(RESOLUTION, RESOLUTION, GL_R32F, GL_RED, GL_FLOAT);

    //* Phase
    std::vector<float> pingPhaseVector(RESOLUTION * RESOLUTION);
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(0.f, 1.f);

    for (size_t i = 0; i < RESOLUTION * RESOLUTION; i++)
        pingPhaseVector[i] = dist(rng) * 2.0f * M_PI;

    pingPhaseTexture = new Texture(
        RESOLUTION,
        RESOLUTION,
        GL_R32F,
        GL_RED,
        GL_FLOAT,
        GL_NEAREST,
        GL_NEAREST,
        GL_CLAMP_TO_BORDER,
        GL_CLAMP_TO_BORDER,
        pingPhaseVector.data()
    );
    pongPhaseTexture = new Texture(RESOLUTION, RESOLUTION, GL_R32F, GL_RED, GL_FLOAT);

    //* Time-varying Spectrum
    spectrumTexture = new Texture(
        RESOLUTION,
        RESOLUTION,
        GL_RGBA32F,
        GL_RGBA,
        GL_FLOAT,
        GL_LINEAR,
        GL_LINEAR,
        GL_REPEAT,
        GL_REPEAT
    );
    tempTexture = new Texture(RESOLUTION, RESOLUTION, GL_RGBA32F, GL_RGBA, GL_FLOAT);

    //* Normal Map
    normalMap = new Texture(
        RESOLUTION,
        RESOLUTION,
        GL_RGBA32F,
        GL_RGBA,
        GL_FLOAT,
        GL_LINEAR,
        GL_LINEAR,
        GL_REPEAT,
        GL_REPEAT
    );

    // Depth Related Code
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Clear the screen using black color
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);

    perspectiveProjectionMatrix = vmath::mat4::identity();

    // Warmup resize call
    resize(WIN_WIDTH, WIN_HEIGHT);

    return 0;
}


void printGLInfo(void)
{
    // Variable Declarations
    GLint numExtensions = 0;

    // Code
    logger->printLog("\nOpenGL Information\n");
    logger->printLog("------------------------------------------------------\n");
    
    logger->printLog("OpenGL Vendor : %s\n", glGetString(GL_VENDOR));
    logger->printLog("OpenGL Renderer : %s\n", glGetString(GL_RENDERER));
    logger->printLog("OpenGL Version : %s\n", glGetString(GL_VERSION));
    logger->printLog("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    logger->printLog("------------------------------------------------------\n");

    glGetIntegerv(GL_NUM_EXTENSIONS, &numExtensions);
    logger->printLog("\nNumber of Supported Extensions : %d\n", numExtensions);
    logger->printLog("------------------------------------------------------\n");
    for (GLint i = 0; i < numExtensions; i++)
        logger->printLog("%s\n", glGetStringi(GL_EXTENSIONS, i));
    logger->printLog("------------------------------------------------------\n"); 
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

    glUseProgram(graphicsShaderProgramObject);
    {
        // Transformations
        vmath::mat4 translationMatrix = vmath::mat4::identity();
        vmath::mat4 scaleMatrix = vmath::mat4::identity();

        vmath::mat4 modelMatrix = vmath::mat4::identity();
        vmath::mat4 viewMatrix = vmath::mat4::identity();

        translationMatrix = vmath::translate(0.0f, -2.0f, -50.0f);
        scaleMatrix = vmath::scale(0.05f, 0.05f, 0.05f);
        modelMatrix = translationMatrix * scaleMatrix;

        glUniformMatrix4fv(modelMatrixUniform, 1, GL_FALSE, modelMatrix);
        glUniformMatrix4fv(viewMatrixUniform, 1, GL_FALSE, camera.getViewMatrix());
        glUniformMatrix4fv(projectionMatrixUniform, 1, GL_FALSE, perspectiveProjectionMatrix);
        
        glUniform3f(lightPositionUniform, lightPosition[0], lightPosition[1], lightPosition[2]);
        glUniform3f(viewPositionUniform, 30.0f, 30.0f, 60.0f);

        // glUniform3f(lightAmbientUniform, 1.0f, 1.0f, 1.0f);
        // glUniform3f(lightDiffuseUniform, 1.0f, 1.0f, 1.0f);
        // glUniform3f(lightSpecularUniform, 1.0f, 0.9f, 0.7f);

        // glUniform1f(heightMinUniform, heightMin * 0.1);
        // glUniform1f(heightMaxUniform, heightMax * 0.1);

        // glUniform1f(timeUniform, fTime);
        // glUniform1f(foamIntensityUniform, 1.5f);
        // glUniform1f(subSurfaceScatteringStrengthUniform, 0.6f);
        // glUniform1f(fogDensityUniform, 0.3f);

        glBindVertexArray(vao_ocean);
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_ocean);
            glDrawElements(GL_TRIANGLES, GRID_SIZE * GRID_SIZE * 2 * 3, GL_UNSIGNED_INT, 0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }
        glBindVertexArray(0);

        glBindTexture(GL_TEXTURE_2D, 0);
        
    }
    glUseProgram(0);

    //! Render ImGui Window on Top
    renderImGui();

    SwapBuffers(ghdc);
}

void update(void)
{
    // Code
    deltaTime = sdkGetTimerValue(&timer) / 1000.0f;
}

void uninitialize(void)
{
    // Function Declarations
    void ToggleFullScreen(void);
    void uninitializeImGui(void);

    // Code
    if (gbFullScreen)
        ToggleFullScreen();

    if (timer)
    {
        sdkStopTimer(&timer);
        sdkDeleteTimer(&timer);
        timer = nullptr;
    }

    if (normalMap)
    {
        delete normalMap;
        normalMap = nullptr;
    }
    if (tempTexture)
    {
        delete tempTexture;
        tempTexture = nullptr;
    }
    if (spectrumTexture)
    {
        delete spectrumTexture;
        spectrumTexture = nullptr;
    }
    if (pongPhaseTexture)
    {
        delete pongPhaseTexture;
        pongPhaseTexture = nullptr;
    }
    if (pingPhaseTexture)
    {
        delete pingPhaseTexture;
        pingPhaseTexture = nullptr;
    }
    if (initialSpectrumTexture)
    {
        delete initialSpectrumTexture;
        initialSpectrumTexture = nullptr;
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

    shader.uninitializeShaders(computeShaderProgramObject);
    shader.uninitializeShaders(graphicsShaderProgramObject);
    
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

    if (logger)
    {
        logger->deleteInstance();
        logger = nullptr;
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
    static float foamIntensity = 1.5f;
    static float subSurfaceScatteringStrength = 0.6f;
    static float fogDensity = 0.3f;

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

        ImGui::SliderFloat("Foam Intensity", &foamIntensity, 0.0f, 3.0f);
        ImGui::SliderFloat("SSS Strength", &subSurfaceScatteringStrength, 0.0f, 2.0f);
        ImGui::SliderFloat("Fog Density", &fogDensity, 0.0f, 1.0f);

        glUniform1f(foamIntensityUniform, foamIntensity);
        glUniform1f(subSurfaceScatteringStrengthUniform, subSurfaceScatteringStrength);
        glUniform1f(fogDensityUniform, fogDensity);

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

