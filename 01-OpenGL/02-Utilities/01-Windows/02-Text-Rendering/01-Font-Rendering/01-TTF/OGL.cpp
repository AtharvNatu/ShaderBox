//* Header Files
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

//! C++ Headers
#include <map>
#include <string>

#include "OGL.h"
#include "vmath.h"

//! OpenGL Header Files
#include <GL/glew.h>
#include <GL/gl.h>

//! FreeType
#include <ft2build.h>
#include FT_FREETYPE_H

#define WIN_WIDTH   800
#define WIN_HEIGHT  600

//! OpenGL Libraries
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "OpenGL32.lib")
#pragma comment(lib, "freetype.lib")

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
    MEM_ALLOC_FAILED = -12,
    FT_INIT_FAILED = -13,
    FT_LOAD_FACE_FAILED = -14,
    LOAD_FONT_FAILED = -15
};

enum ATTRIBUTES
{
    ATTRIBUTE_POSITION = 0,
    ATTRIBUTE_COLOR,
    ATTRIBUTE_NORMAL,
    ATTRIBUTE_TEXTURE0
};

GLuint shaderProgramObject = 0;

vmath::mat4 perspectiveProjectionMatrix;

//! Font Related
FT_Library ftLibrary;
FT_Face ftFace;
const int PIXEL_HEIGHT = 48;

struct Character
{
    GLuint textureID;       // ID of glyph
    vmath::vec2 size;       // Size of glyph
    vmath::vec2 bearing;    // Offset from baseline to left/top of glyph
    GLuint advance;         // Offset to advance to next glyph  
};

std::map<GLchar, Character> characterMap;

GLuint vao_font = 0;
GLuint vbo_font = 0;
GLuint mvpMatrixUniform = 0;
GLuint textureSamplerUniform = 0;
GLuint textColorUniform = 0;

std::string text = "OpenGL TTF";
float scale = 0.05f;
float x = -3.5f;
float y = 0.0f;

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
        TEXT("OpenGL : TTF Rendering"),
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

        case FT_INIT_FAILED:
        case FT_LOAD_FACE_FAILED:
        case LOAD_FONT_FAILED:
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
    int loadFont(const char*);

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
        
        "in vec4 vertex;" \
        "uniform mat4 u_mvpMatrix;" \
        "out vec2 a_texcoord_out;" \

        "void main(void)" \
        "{" \
            "gl_Position = u_mvpMatrix * vec4(vertex.xy, 0.0, 1.0);" \
            "a_texcoord_out = vertex.zw;" \
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
        "out vec4 FragColor;" \

        "uniform sampler2D u_textureSampler;" \
        "uniform vec3 u_textColor;" \

        "void main(void)" \
        "{" \
            "vec4 sampled = vec4(1.0, 1.0, 1.0, texture(u_textureSampler, a_texcoord_out).r);" \
            "FragColor = vec4(u_textColor, 1.0) * sampled;" \
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
    glBindAttribLocation(shaderProgramObject, ATTRIBUTE_POSITION, "vertex");

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
    textColorUniform = glGetUniformLocation(shaderProgramObject, "u_textColor");

    //! VAO and VBO Related Code
    glGenVertexArrays(1, &vao_font);
    glBindVertexArray(vao_font);
    {
        glGenBuffers(1, &vbo_font);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_font);
        {
            //* 2D Quad Requires 6 vertices of 4 floats := 6 * 4 * sizeof(float)
            glBufferData(GL_ARRAY_BUFFER, 6 * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
            glVertexAttribPointer(ATTRIBUTE_POSITION, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
            glEnableVertexAttribArray(ATTRIBUTE_POSITION);
        }
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
    glBindVertexArray(0);

    // Depth Related Code
    glClearDepth(1.0f);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);

    // Clear the screen using black color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    perspectiveProjectionMatrix = vmath::mat4::identity();

    //! Enable Blending 
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    //! Font Loading
    if (loadFont("Poppins-Regular.ttf") != 0)
        return LOAD_FONT_FAILED;

    // Warmup resize call
    resize(WIN_WIDTH, WIN_HEIGHT);

    return 0;
}

int loadFont(const char* fontFile)
{
    // Code
    if (FT_Init_FreeType(&ftLibrary) != 0)
    {
        fprintf(gpFile, "ERROR : Failed To Initialize FreeType Library !!!\n");
        return FT_INIT_FAILED;
    }

    if (FT_New_Face(ftLibrary, fontFile, 0, &ftFace) != 0)
    {
        fprintf(gpFile, "ERROR : Failed To Load Font !!!\n");
        return FT_LOAD_FACE_FAILED;
    }

    //* Setting the width as 0, lets the face dynamically calculate the width based on the given height
    FT_Set_Pixel_Sizes(ftFace, 0, PIXEL_HEIGHT);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    for (unsigned char ch = 0; ch < 128; ch++)
    {
        //* Load character glyph
        if (FT_Load_Char(ftFace, ch, FT_LOAD_RENDER) != 0)
        {
            fprintf(gpFile, "ERROR : Failed To Load Glyph !!!\n");
            continue;
        }

        //* Generate Texture
        unsigned int texture;

        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        {
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RED,
                ftFace->glyph->bitmap.width,
                ftFace->glyph->bitmap.rows,
                0,
                GL_RED,
                GL_UNSIGNED_BYTE,
                ftFace->glyph->bitmap.buffer
            );

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            Character character;
            character.textureID = texture;
            character.size = vmath::vec2(ftFace->glyph->bitmap.width, ftFace->glyph->bitmap.rows);
            character.bearing = vmath::vec2(ftFace->glyph->bitmap_left, ftFace->glyph->bitmap_top);
            character.advance = ftFace->glyph->advance.x;

            characterMap.insert(std::pair<GLchar, Character>(ch, character));
        }
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    FT_Done_Face(ftFace);
    FT_Done_FreeType(ftLibrary);

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
    // Code
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Transformations
    vmath::mat4 translationMatrix = vmath::mat4::identity();
    vmath::mat4 fontMatrix = vmath::mat4::identity();

    glUseProgram(shaderProgramObject);
    {
        glUniform3fv(textColorUniform, 1, vmath::vec3(0.0f, 1.0f, 0.0f));
        glUniform1i(textureSamplerUniform, 0);

        glActiveTexture(GL_TEXTURE0);

        glBindVertexArray(vao_font);
        {
            float xCursor = 0.0f;
            std::string::const_iterator iter;
            for (iter = text.begin(); iter != text.end(); iter++)
            {
                Character ch = characterMap[*iter];

                float xPosition = x + ch.bearing[0] * scale;
                float yPosition = y - (ch.size[1] - ch.bearing[1]) * scale;

                float width = ch.size[0] * scale;
                float height = ch.size[1] * scale;

                float vertices[6][4] = 
                {
                    { xPosition,          yPosition + height,  0.0f,   0.0f },
                    { xPosition,          yPosition,           0.0f,   1.0f },
                    { xPosition + width,  yPosition,           1.0f,   1.0f },

                    { xPosition,          yPosition + height,  0.0f,   0.0f },
                    { xPosition + width,  yPosition,           1.0f,   1.0f },
                    { xPosition + width,  yPosition + height,  1.0f,   0.0f },
                };

                translationMatrix = vmath::translate(x + xCursor, y, -20.0f);
                fontMatrix = perspectiveProjectionMatrix * translationMatrix;
                glUniformMatrix4fv(mvpMatrixUniform, 1, NULL, fontMatrix);

                // Render Glyph Texture On Quad
                glBindBuffer(GL_ARRAY_BUFFER, vbo_font);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
                glBindBuffer(GL_ARRAY_BUFFER, 0); 
                
                glBindTexture(GL_TEXTURE_2D, ch.textureID);
                glDrawArrays(GL_TRIANGLES, 0, 6);

                xCursor += ((float)(ch.advance >> 6)) * scale;
            }
        }
        glBindVertexArray(0);
        glBindTexture(GL_TEXTURE_2D, 0);

    }
    glUseProgram(0);

    SwapBuffers(ghdc);
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
    
    if (vbo_font)
    {
        glDeleteBuffers(1, &vbo_font);
        vbo_font = 0;
    }

    if (vao_font)
    {
        glDeleteVertexArrays(1, &vao_font);
        vao_font = 0;
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
