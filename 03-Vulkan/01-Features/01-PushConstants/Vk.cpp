#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>

//! Vulkan Related Header Files
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Vk.h"

//! Vulkan Related Libraries
#pragma comment(lib, "vulkan-1.lib")

#define WIN_WIDTH   800
#define WIN_HEIGHT  600

// Global Function Declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// Global Variable Declarations
HWND ghwnd = NULL;
BOOL gbFullScreen = FALSE;
BOOL gbWindowMinimized = FALSE;
BOOL gbActiveWindow = FALSE;
FILE *gpFile = NULL;
WINDOWPLACEMENT wpPrev;
DWORD dwStyle;
const char *gpSzAppName = "ARTR";

//! Vulkan Related Global Variables

//? Instance Extension Related Variables
uint32_t enabledInstanceExtensionCount = 0;

//* VK_KHR_SURFACE_EXTENSION_NAME,
//* VK_KHR_WIN32_SURFACE_EXTENSION_NAME,
//* VK_EXT_DEBUG_REPORT_EXTENSION_NAME
const char *enabledInstanceExtensionNames_array[3];

//? Vulkan Instance
VkInstance vkInstance = VK_NULL_HANDLE;

//? Vulkan Presentation Surface
VkSurfaceKHR vkSurfaceKHR = VK_NULL_HANDLE;

//? Vulkan Physical Device Related
VkPhysicalDevice vkPhysicalDevice_selected = VK_NULL_HANDLE;
uint32_t graphicsQueueFamilyIndex_selected = UINT32_MAX;
VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;

uint32_t physicalDeviceCount = 0;
VkPhysicalDevice *vkPhysicalDevice_array = NULL;

//? Device Extensions Related Variables
uint32_t enabledDeviceExtensionCount = 0;
const char *enabledDeviceExtensionNames_array[1]; //* -> VK_KHR_SWAPCHAIN_EXTENSTION_NAME

//? Vulkan Device Creation Related Variables
VkDevice vkDevice = VK_NULL_HANDLE;

//? Vulkan Device Queue Related Variables
VkQueue vkQueue = VK_NULL_HANDLE;

//? Color Format and Color Space
VkFormat vkFormat_color = VK_FORMAT_UNDEFINED;
VkColorSpaceKHR vkColorSpaceKHR = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

//? Presentation Mode
VkPresentModeKHR vkPresentModeKHR = VK_PRESENT_MODE_FIFO_KHR;

//? Swapchain
VkSwapchainKHR vkSwapchainKHR = VK_NULL_HANDLE;
VkExtent2D vkExtent2D_swapchain;
int winWidth = WIN_WIDTH;
int winHeight = WIN_HEIGHT;

//? Swapchain Images and Image Views
uint32_t swapchainImageCount = UINT32_MAX;
VkImage *swapchainImage_array = NULL;
VkImageView *swapchainImageView_array = NULL;

//? Command Pool
VkCommandPool vkCommandPool = VK_NULL_HANDLE;

//? Command Buffer
VkCommandBuffer *vkCommandBuffer_array = NULL;

//? Render Pass
VkRenderPass vkRenderPass = VK_NULL_HANDLE;

//? Frame Buffer
VkFramebuffer *vkFramebuffer_array = NULL;

//? Fences and Semaphores
VkSemaphore vkSemaphore_backBuffer = VK_NULL_HANDLE;
VkSemaphore vkSemaphore_renderComplete = VK_NULL_HANDLE;
VkFence *vkFence_array = NULL;

//? Clear Color Values
VkClearColorValue vkClearColorValue;

//? Render
BOOL bInitialized = FALSE;
uint32_t currentImageIndex = UINT32_MAX;

//? Validation
BOOL bValidation = TRUE;
uint32_t enabledValidationLayerCount = 0;
const char *enabledValidationLayerNames_array[1];   //* For VK_LAYER_KHRONOS_validation
VkDebugReportCallbackEXT vkDebugReportCallbackEXT = VK_NULL_HANDLE;
PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallbackEXT_fnptr = NULL;

//? Vertex Buffer Related Variables
typedef struct
{
    VkBuffer vkBuffer;
    VkDeviceMemory vkDeviceMemory;
} VertexData;

//? Position Related Variables
VertexData vertexData_position_triangle;
VertexData vertexData_color_triangle;

VertexData vertexData_position_rectangle;
VertexData vertexData_color_rectangle;

//? Uniform Related Variables
typedef struct
{
    glm::mat4 viewMatrix;
    glm::mat4 projectionMatix;
} VP_UniformData;

typedef struct
{
    glm::mat4 modelMatrix;
} PushConstant;

typedef struct
{
    VkBuffer vkBuffer;
    VkDeviceMemory vkDeviceMemory;
} UniformData;

UniformData uniformData;
PushConstant model_uniformData[6];

//? Shader Related Variables
VkShaderModule vkShaderModule_vertex_shader = VK_NULL_HANDLE;
VkShaderModule vkShaderModule_fragment_shader = VK_NULL_HANDLE;

//? DescriptorSetLayout Related Variables
VkDescriptorSetLayout vkDescriptorSetLayout = VK_NULL_HANDLE;

//? PipelineLayout Related Variables
VkPipelineLayout vkPipelineLayout = VK_NULL_HANDLE;

//? Descriptor Pool
VkDescriptorPool vkDescriptorPool = VK_NULL_HANDLE;

//? Descriptor Set
VkDescriptorSet vkDescriptorSet = VK_NULL_HANDLE;

//? Pipeline Related Variables
VkViewport vkViewport;
VkRect2D vkRect2D_scissor;
VkPipeline vkPipeline = VK_NULL_HANDLE;

// Entry Point Function
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int iCmdShow)
{
    // Function Declarations
    VkResult initialize(void);
    VkResult display(void);
    void update(void);
    void uninitialize(void);

    // Variable Declarations
    WNDCLASSEX wndclass;
    HWND hwnd;
    MSG msg;
    TCHAR szAppName[255];
    BOOL bDone = FALSE;
    VkResult vkResult = VK_SUCCESS;

    // Code
    gpFile = fopen("Log.txt", "w");
    if (gpFile == NULL)
    {
        MessageBox(NULL, TEXT("Failed To Create Log File ... Exiting !!!"), TEXT("File I/O Error"), MB_OK | MB_ICONERROR);
        exit(EXIT_FAILURE);
    }
    else
        fprintf(gpFile, "%s() => Program Started Successfully\n", __func__);

    wsprintf(szAppName, TEXT("%s"), gpSzAppName);

    // Initialization of WNDCLASSEX Structure
    wndclass.cbSize = sizeof(WNDCLASSEX);
    wndclass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    wndclass.cbWndExtra = 0;
    wndclass.cbClsExtra = 0;
    wndclass.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wndclass.lpfnWndProc = WndProc;
    wndclass.hInstance = hInstance;
    wndclass.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(ADN_ICON));
    wndclass.hIconSm = LoadIcon(hInstance, MAKEINTRESOURCE(ADN_ICON));
    wndclass.hCursor = LoadCursor(NULL, IDC_ARROW);
    wndclass.lpszClassName = szAppName;
    wndclass.lpszMenuName = NULL;

    // Register the class
    RegisterClassEx(&wndclass);

    // Get Screen Co-ordinates
    int screenX = GetSystemMetrics(SM_CXSCREEN);
    int screenY = GetSystemMetrics(SM_CYSCREEN);

    // Create Window
    hwnd = CreateWindowEx(
        WS_EX_APPWINDOW,
        szAppName,
        TEXT("Atharv Natu : Vulkan Push Constants"),
        WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS | WS_VISIBLE,
        (screenX / 2) - (WIN_WIDTH / 2),
        (screenY / 2) - (WIN_HEIGHT / 2),
        WIN_WIDTH,
        WIN_HEIGHT,
        NULL,
        NULL,
        hInstance,
        NULL
    );

    ghwnd = hwnd;

    //* Initialize
    vkResult = initialize();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => initialize() Failed : %d !!!\n", __func__, vkResult);
        DestroyWindow(hwnd);
        hwnd = NULL;
    }
    else
        fprintf(gpFile, "%s() => initialize() Succeeded\n", __func__);

    // Show and Update Window
    ShowWindow(hwnd, iCmdShow);
    UpdateWindow(hwnd);

    // Bring the window to foreground and set focus
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    //* Game Loop
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
            if (gbActiveWindow == TRUE && gbWindowMinimized == FALSE)
            {
                //* Render the scene
                vkResult = display();
                if (vkResult != VK_FALSE && vkResult != VK_SUCCESS && vkResult != VK_ERROR_OUT_OF_DATE_KHR && vkResult != VK_SUBOPTIMAL_KHR)
                {
                    fprintf(gpFile, "%s() => Call To Display Failed !!!\n", __func__);
                    bDone = TRUE;
                }

                //* Update the scene
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
    VkResult resize(int, int);
    void uninitialize(void);

    // Code
    switch(iMsg)
    {
        case WM_CREATE:
            memset((void*)&wpPrev, 0, sizeof(WINDOWPLACEMENT));
            wpPrev.length = sizeof(WINDOWPLACEMENT);
        break;

        case WM_SETFOCUS:
            gbActiveWindow = TRUE;
        break;

        case WM_KILLFOCUS:
            gbActiveWindow = FALSE;
        break;

        case WM_SIZE:
            if (wParam == SIZE_MINIMIZED)
                gbWindowMinimized = TRUE;
            else
            {
                gbWindowMinimized = FALSE;
                resize(LOWORD(lParam), HIWORD(lParam));
            }
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

VkResult initialize(void)
{
    // Function Declarations
    VkResult createVulkanInstance(void);
    VkResult getSupportedSurface(void);
    VkResult getPhysicalDevice(void);
    VkResult printVkInfo(void);
    VkResult createVulkanDevice(void);
    void getDeviceQueue(void);
    VkResult createSwapchain(VkBool32);
    VkResult createImagesAndImageViews(void);
    VkResult createCommandPool(void);
    VkResult createCommandBuffers(void);
    VkResult createVertexBuffer(void);
    VkResult createUniformBuffer(void);
    VkResult createShaders(void);
    VkResult createDescriptorSetLayout(void);
    VkResult createPipelineLayout(void);
    VkResult createDescriptorPool(void);
    VkResult createDescriptorSet(void);
    VkResult createRenderPass(void);
    VkResult createPipeline(void);
    VkResult createFramebuffers(void);
    VkResult createSemaphores(void);
    VkResult createFences(void);
    VkResult buildCommandBuffers(void);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    vkResult = createVulkanInstance();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createVulkanInstance() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createVulkanInstance() Succeeded\n", __func__);

    //! Create Vulkan Presentation Surface
    vkResult = getSupportedSurface();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => getSupportedSurface() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => getSupportedSurface() Succeeded\n", __func__);

    //! Enumerate and Selected Required Physical Device and its Queue Family Index
    vkResult = getPhysicalDevice();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => getPhysicalDevice() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => getPhysicalDevice() Succeeded\n", __func__);

    //! Print Vulkan Info
    vkResult = printVkInfo();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => printVkInfo() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => printVkInfo() Succeeded\n", __func__);

    //! Create Vulkan Device
    vkResult = createVulkanDevice();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createVulkanDevice() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createVulkanDevice() Succeeded\n", __func__);

    //! Get Device Queue
    getDeviceQueue();

    //! Create Swapchain
    vkResult = createSwapchain(VK_FALSE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createSwapchain() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createSwapchain() Succeeded\n", __func__);

    //! Create Swapchain Image and Image Views
    vkResult = createImagesAndImageViews();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createImagesAndImageViews() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createImagesAndImageViews() Succeeded\n", __func__);
    
    //! Create Command Pool
    vkResult = createCommandPool();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createCommandPool() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createCommandPool() Succeeded\n", __func__);

    //! Create Command Buffers
    vkResult = createCommandBuffers();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createCommandBuffers() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createCommandBuffers() Succeeded\n", __func__);

    //! Create Vertex Buffer
    vkResult = createVertexBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createVertexBuffer() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createVertexBuffer() Succeeded\n", __func__);

    //! Create Uniform Buffer
    vkResult = createUniformBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createUniformBuffer() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createUniformBuffer() Succeeded\n", __func__);

    //! Create Shaders
    vkResult = createShaders();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createShaders() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createShaders() Succeeded\n", __func__);

    //! Create DescriptorSetLayout
    vkResult = createDescriptorSetLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Succeeded\n", __func__);

    //! Create Pipeline Layout
    vkResult = createPipelineLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createPipelineLayout() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createPipelineLayout() Succeeded\n", __func__);

    //! Create Descriptor Pool
    vkResult = createDescriptorPool();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createDescriptorPool() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createDescriptorPool() Succeeded\n", __func__);

    //! Create Descriptor Set
    vkResult = createDescriptorSet();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createDescriptorSet() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createDescriptorSet() Succeeded\n", __func__);

    //! Create Render Pass
    vkResult = createRenderPass();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createRenderPass() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createRenderPass() Succeeded\n", __func__);

    //! Create Pipeline
    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createPipeline() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createPipeline() Succeeded\n", __func__);

    //! Create Framebuffers
    vkResult = createFramebuffers();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createFramebuffers() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createFramebuffers() Succeeded\n", __func__);

    //! Create Semaphores
    vkResult = createSemaphores();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createSemaphores() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createSemaphores() Succeeded\n", __func__);

    //! Create Fences
    vkResult = createFences();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createFences() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createFences() Succeeded\n", __func__);

    //! Initialize Clear Color Values (Analogous to glClearColor())
    memset((void*)&vkClearColorValue, 0, sizeof(VkClearColorValue));
    vkClearColorValue.float32[0] = 0.0f;    //* R
    vkClearColorValue.float32[1] = 0.0f;    //* G
    vkClearColorValue.float32[2] = 0.0f;    //* B
    vkClearColorValue.float32[3] = 1.0f;    //* A

    vkResult = buildCommandBuffers();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => buildCommandBuffers() Failed\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => buildCommandBuffers() Succeeded\n", __func__);

    //! Initialization Completed
    bInitialized = TRUE;
    fprintf(gpFile, "%s() => Initialization Completed Successfully\n", __func__);
    
    return vkResult;
}

VkResult resize(int width, int height)
{
    // Function Declarations
    VkResult createSwapchain(VkBool32);
    VkResult createImagesAndImageViews(void);
    VkResult createCommandBuffers(void);
    VkResult createPipelineLayout(void);
    VkResult createRenderPass(void);
    VkResult createPipeline(void);
    VkResult createFramebuffers(void);
    VkResult buildCommandBuffers(void);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    if (height <= 0)
        height = 1;

    //* Check the bInitialized Variable
    if (bInitialized == FALSE)
    {
        fprintf(gpFile, "%s() => Initialization Not Yet Completed or Failed !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    //* As recreation of swapchain is needed, we are going to repeat many steps of initialize() again. Hence, set bInitialize = FALSE again
    bInitialized = FALSE;
    {
        //* Set Global winWidth and winHeight variables
        winWidth = width;
        winHeight = height;

        //? DESTROY
        //?--------------------------------------------------------------------------------------------------
        //* Wait for device to complete in-hand tasks
        if (vkDevice)
            vkDeviceWaitIdle(vkDevice);
        
        //* Check presence of swapchain
        if (vkSwapchainKHR == VK_NULL_HANDLE)
        {
            fprintf(gpFile, "%s() => Swapchain is already NULL ... cannot proceed !!!\n", __func__);
            vkResult = VK_ERROR_INITIALIZATION_FAILED;
            return vkResult;
        }

        //* Destroy Framebuffer
        for (uint32_t i = 0; i < swapchainImageCount; i++)
            vkDestroyFramebuffer(vkDevice, vkFramebuffer_array[i], NULL);
        if (vkFramebuffer_array)
        {
            free(vkFramebuffer_array);
            vkFramebuffer_array = NULL;
        }

        //* Destroy Command Buffer
        for (uint32_t i = 0; i < swapchainImageCount; i++)
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_array[i]);
        if (vkCommandBuffer_array)
        {
            free(vkCommandBuffer_array);
            vkCommandBuffer_array = NULL;
        }

        //* Destroy PipelineLayout
        if (vkDescriptorSetLayout)
        {
            vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, NULL);
            vkPipelineLayout = VK_NULL_HANDLE;
        }

        //* Destroy Pipeline
        if (vkPipeline)
        {
            vkDestroyPipeline(vkDevice, vkPipeline, NULL);
            vkPipeline = VK_NULL_HANDLE;
        }

        //* Destroy Render Pass
        if (vkRenderPass)
        {
            vkDestroyRenderPass(vkDevice, vkRenderPass, NULL);
            vkRenderPass = VK_NULL_HANDLE;
        }
        
        //* Destroy Swapchain Image and Image Views
        for (uint32_t i = 0; i < swapchainImageCount; i++)
            vkDestroyImageView(vkDevice, swapchainImageView_array[i], NULL);

        if (swapchainImageView_array)
        {
            free(swapchainImageView_array);
            swapchainImageView_array = NULL;
        }

        //! No need to free swapchain images -> Uncommenting causes the code to crash
        // for (uint32_t i = 0; i < swapchainImageCount; i++)
        // {
        //     vkDestroyImage(vkDevice, swapchainImage_array[i], NULL);
        //     fprintf(gpFile, "%s() => vkDestroyImage() Succeeded\n", __func__);
        // } 

        if (swapchainImage_array)
        {
            free(swapchainImage_array);
            swapchainImage_array = NULL;
        }

        //* Destroy Swapchain
        if (vkSwapchainKHR)
        {
            vkDestroySwapchainKHR(vkDevice, vkSwapchainKHR, NULL);
            vkSwapchainKHR = VK_NULL_HANDLE;
        }
        //?--------------------------------------------------------------------------------------------------
        
        //? RECREATE FOR RESIZE
        //?--------------------------------------------------------------------------------------------------
        //* Create Swapchain
        vkResult = createSwapchain(VK_FALSE);
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createSwapchain() Failed : %d !!!\n", __func__, vkResult);
            return vkResult;
        }

        //* Create Swapchain Image and Image Views
        vkResult = createImagesAndImageViews();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createImagesAndImageViews() Failed : %d !!!\n", __func__, vkResult);
            return vkResult;
        }

        //* Create Render Pass
        vkResult = createRenderPass();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createRenderPass() Failed : %d !!!\n", __func__, vkResult);
            return vkResult;
        }

        //* Create Pipeline Layout
        vkResult = createPipelineLayout();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createPipelineLayout() Failed : %d !!!\n", __func__, vkResult);
            return vkResult;
        }

        //* Create Pipeline
        vkResult = createPipeline();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createPipeline() Failed : %d !!!\n", __func__, vkResult);
            return vkResult;
        }

        //* Create Command Buffers
        vkResult = createCommandBuffers();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createCommandBuffers() Failed : %d !!!\n", __func__, vkResult);
            return vkResult;
        }

        //* Create Framebuffers
        vkResult = createFramebuffers();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createFramebuffers() Failed : %d !!!\n", __func__, vkResult);
            return vkResult;
        }

        //* Build Command Buffers
        vkResult = buildCommandBuffers();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => buildCommandBuffers() Failed\n", __func__);
            return vkResult;
        }
        //?--------------------------------------------------------------------------------------------------
    }
    bInitialized = TRUE;

    return vkResult;
}

VkResult display(void)
{
    // Function Declarations
    VkResult resize(int, int);
    VkResult updateUniformBuffer(void);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    if (bInitialized == FALSE)
    {
        fprintf(gpFile, "%s() => Initialization Not Yet Completed !!!\n", __func__);
        return (VkResult)VK_FALSE;
    }

    //! Acquire next image index
    vkResult = vkAcquireNextImageKHR(vkDevice, vkSwapchainKHR, UINT64_MAX, vkSemaphore_backBuffer, VK_NULL_HANDLE, &currentImageIndex);
    if (vkResult != VK_SUCCESS)
    {
        if (vkResult == VK_ERROR_OUT_OF_DATE_KHR || vkResult == VK_SUBOPTIMAL_KHR)
            resize(winWidth, winHeight);
        else
        {
            fprintf(gpFile, "%s() => vkAcquireNextImageKHR() Failed : %d\n", __func__, vkResult);
            return vkResult;
        }
    }

    //! Use fence to allow host to wait for completion of execution of previous command buffer
    vkResult = vkWaitForFences(vkDevice, 1, &vkFence_array[currentImageIndex], VK_TRUE, UINT64_MAX);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkWaitForFences() Failed : %d\n", __func__, vkResult);
        return vkResult;
    }

    //! Make sure fences are ready for execution of next command buffer
    vkResult = vkResetFences(vkDevice, 1, &vkFence_array[currentImageIndex]);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkResetFences() Failed : %d\n", __func__, vkResult);
        return vkResult;
    }

    //! Render color attachment
    const VkPipelineStageFlags waitDstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    //! Declare and initialize VkSubmitInfo stucture
    VkSubmitInfo vkSubmitInfo;
    memset((void*)&vkSubmitInfo, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.pNext = NULL;
    vkSubmitInfo.pWaitDstStageMask = &waitDstStageMask;
    vkSubmitInfo.waitSemaphoreCount = 1;
    vkSubmitInfo.pWaitSemaphores = &vkSemaphore_backBuffer;
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &vkCommandBuffer_array[currentImageIndex];
    vkSubmitInfo.signalSemaphoreCount = 1;
    vkSubmitInfo.pSignalSemaphores = &vkSemaphore_renderComplete;

    //! Submit above work to the queue
    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo, vkFence_array[currentImageIndex]);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkQueueSubmit() Failed : %d\n", __func__, vkResult);
        return vkResult;
    }

    //! Present Rendered Image
    VkPresentInfoKHR vkPresentInfoKHR;
    memset((void*)&vkPresentInfoKHR, 0, sizeof(VkPresentInfoKHR));
    vkPresentInfoKHR.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    vkPresentInfoKHR.pNext = NULL;
    vkPresentInfoKHR.swapchainCount = 1;
    vkPresentInfoKHR.pSwapchains = &vkSwapchainKHR;
    vkPresentInfoKHR.pImageIndices = &currentImageIndex;
    vkPresentInfoKHR.waitSemaphoreCount = 1;
    vkPresentInfoKHR.pWaitSemaphores = &vkSemaphore_renderComplete;

    //! Present the queue
    vkResult = vkQueuePresentKHR(vkQueue, &vkPresentInfoKHR);
    if (vkResult != VK_SUCCESS)
    {
        if (vkResult == VK_ERROR_OUT_OF_DATE_KHR || vkResult == VK_SUBOPTIMAL_KHR)
            resize(winWidth, winHeight);
        else
        {
            fprintf(gpFile, "%s() => vkQueuePresentKHR() Failed : %d\n", __func__, vkResult);
            return vkResult;
        }
    }

    vkResult = updateUniformBuffer();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => updateUniformBuffer() Failed : %d\n", __func__, vkResult);

    vkDeviceWaitIdle(vkDevice);

    return vkResult;
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

    //* Step - 5 of Device Creation (Destroy Vulkan Device)
    //! vkDeviceWaitIdle(vkDevice) should be the 1st API to maintain synchronization
    if (vkDevice)
    {
        vkDeviceWaitIdle(vkDevice);
        fprintf(gpFile, "%s() => vkDeviceWaitIdle() Succeeded\n", __func__);
    }

    //* Step - 7 of Fences and Semaphores
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        vkDestroyFence(vkDevice, vkFence_array[i], NULL);
        fprintf(gpFile, "%s() => vkDestroyFence() Succeeded For Index : %d\n", __func__, i);
    }
    if (vkFence_array)
    {
        free(vkFence_array);
        vkFence_array = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For vkFence_array\n", __func__);
    }

    if (vkSemaphore_renderComplete)
    {
        vkDestroySemaphore(vkDevice, vkSemaphore_renderComplete, NULL);
        fprintf(gpFile, "%s() => vkDestroySemaphore() Succeeded For vkSemaphore_renderComplete\n", __func__);
        vkSemaphore_renderComplete = VK_NULL_HANDLE;
    }

    if (vkSemaphore_backBuffer)
    {
        vkDestroySemaphore(vkDevice, vkSemaphore_backBuffer, NULL);
        fprintf(gpFile, "%s() => vkDestroySemaphore() Succeeded For vkSemaphore_backBuffer\n", __func__);
        vkSemaphore_backBuffer = VK_NULL_HANDLE;
    }

    //* Step - 5 of Frame Buffer
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        vkDestroyFramebuffer(vkDevice, vkFramebuffer_array[i], NULL);
        fprintf(gpFile, "%s() => vkDestroyFramebuffer() Succeeded For Index : %d\n", __func__, i);
    }
    if (vkFramebuffer_array)
    {
        free(vkFramebuffer_array);
        vkFramebuffer_array = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For vkFramebuffer_array\n", __func__);
    }

    if (vkPipeline)
    {
        vkDestroyPipeline(vkDevice, vkPipeline, NULL);
        vkPipeline = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyPipeline() Succeeded\n", __func__);
    }

    //* Step - 6 of Render Pass
    if (vkRenderPass)
    {
        vkDestroyRenderPass(vkDevice, vkRenderPass, NULL);
        vkRenderPass = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyRenderPass() Succeeded\n", __func__);
    }

    //* Destroy Descriptor Pool (Destroys Descriptor Set with it)
    if (vkDescriptorPool)
    {
        vkDestroyDescriptorPool(vkDevice, vkDescriptorPool, NULL);
        vkDescriptorPool = VK_NULL_HANDLE;
        vkDescriptorSet = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyDescriptorPool() => Destroyed vkDescriptorPool and vkDescriptorSet Successfully\n", __func__);
    }

    //* Step - 5 of PipelineLayout
    if (vkPipelineLayout)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout, NULL);
        vkPipelineLayout = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyPipelineLayout() Succeeded\n", __func__);
    }

    //* Step - 5 of DescriptorSetLayout
    if (vkDescriptorSetLayout)
    {
        vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout, NULL);
        vkDescriptorSetLayout = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyDescriptorSetLayout() Succeeded\n", __func__);
    }

    //* Step - 11 of Shaders
    if (vkShaderModule_fragment_shader)
    {
        vkDestroyShaderModule(vkDevice, vkShaderModule_fragment_shader, NULL);
        vkShaderModule_fragment_shader = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyShaderModule() Succeeded For Fragment Shader\n", __func__);
    }

    if (vkShaderModule_vertex_shader)
    {
        vkDestroyShaderModule(vkDevice, vkShaderModule_vertex_shader, NULL);
        vkShaderModule_vertex_shader = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyShaderModule() Succeeded For Vertex Shader\n", __func__);
    }

    //* Destroy Uniform Buffer
    if (uniformData.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData.vkDeviceMemory, NULL);
        uniformData.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For uniformData.vkDeviceMemory\n", __func__);
    }

    if (uniformData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData.vkBuffer, NULL);
        uniformData.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succedded For uniformData.vkBuffer\n", __func__);
    }

    //* Destroy Color Buffer
    if (vertexData_color_rectangle.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_color_rectangle.vkDeviceMemory, NULL);
        vertexData_color_rectangle.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_color_rectangle.vkDeviceMemory\n", __func__);
    }

    if (vertexData_color_rectangle.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_color_rectangle.vkBuffer, NULL);
        vertexData_color_rectangle.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_color_rectangle.vkBuffer\n", __func__);
    }

    //* Step - 14 of Vertex Buffer
    if (vertexData_position_rectangle.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_position_rectangle.vkDeviceMemory, NULL);
        vertexData_position_rectangle.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_position_rectangle.vkDeviceMemory\n", __func__);
    }

    if (vertexData_position_rectangle.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_position_rectangle.vkBuffer, NULL);
        vertexData_position_rectangle.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_position_rectangle.vkBuffer\n", __func__);
    }

    //* Destroy Color Buffer
    if (vertexData_color_triangle.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_color_triangle.vkDeviceMemory, NULL);
        vertexData_color_triangle.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_color_triangle.vkDeviceMemory\n", __func__);
    }

    if (vertexData_color_triangle.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_color_triangle.vkBuffer, NULL);
        vertexData_color_triangle.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_color_triangle.vkBuffer\n", __func__);
    }

    //* Step - 14 of Vertex Buffer
    if (vertexData_position_triangle.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_position_triangle.vkDeviceMemory, NULL);
        vertexData_position_triangle.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_position_triangle.vkDeviceMemory\n", __func__);
    }

    if (vertexData_position_triangle.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_position_triangle.vkBuffer, NULL);
        vertexData_position_triangle.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_position_triangle.vkBuffer\n", __func__);
    }

    //* Step - 5 of Command Buffer
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_array[i]);
        fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For Index : %d\n", __func__, i);
    }

    if (vkCommandBuffer_array)
    {
        free(vkCommandBuffer_array);
        vkCommandBuffer_array = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For vkCommandBuffer_array\n", __func__);
    }

    //* Step - 4 of Command Pool (Destroy Command Pool)
    if (vkCommandPool)
    {
        vkDestroyCommandPool(vkDevice, vkCommandPool, NULL);
        vkCommandPool = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyCommandPool() Succeeded\n", __func__);
    }

    //* Step - 7 of Swapchain Image and Image Views
    for (uint32_t i = 0; i < swapchainImageCount; i++)
        vkDestroyImageView(vkDevice, swapchainImageView_array[i], NULL);
    fprintf(gpFile, "%s() => vkDestroyImageView() Succeeded\n", __func__);

    //* Step - 8 of Swapchain Image and Image Views
    if (swapchainImageView_array)
    {
        free(swapchainImageView_array);
        swapchainImageView_array = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For swapchainImageView_array\n", __func__);
    }

    //! No need to free swapchain images ->  Uncommenting causes the code to crash
    // for (uint32_t i = 0; i < swapchainImageCount; i++)
    // {
    //     vkDestroyImage(vkDevice, swapchainImage_array[i], NULL);
    //     fprintf(gpFile, "%s() => vkDestroyImage() Succeeded\n", __func__);
    // } 

    if (swapchainImage_array)
    {
        free(swapchainImage_array);
        swapchainImage_array = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For swapchainImage_array\n", __func__);
    }

    //* Step - 10 of Swapchain (Destroy Swapchain)
    if (vkSwapchainKHR)
    {
        vkDestroySwapchainKHR(vkDevice, vkSwapchainKHR, NULL);
        vkSwapchainKHR = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroySwapchainKHR() Succeeded\n", __func__);
    }

    if (vkDevice)
    {
        vkDestroyDevice(vkDevice, NULL);
        vkDevice = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyDevice() Succeeded\n", __func__);
    }

    //* No need to destroy device queue

    //* No need to destroy selected physical device

    //* Step - 5 of Presentation Surface
    if (vkSurfaceKHR)
    {
        vkDestroySurfaceKHR(vkInstance, vkSurfaceKHR, NULL);
        vkSurfaceKHR = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroySurfaceKHR() Succeeded\n", __func__);
    }

    if (vkDebugReportCallbackEXT && vkDestroyDebugReportCallbackEXT_fnptr)
    {
        vkDestroyDebugReportCallbackEXT_fnptr(vkInstance, vkDebugReportCallbackEXT, NULL);
        vkDebugReportCallbackEXT = VK_NULL_HANDLE;
        vkDestroyDebugReportCallbackEXT_fnptr = NULL;
    }

    //* Step - 5 of Instance Creation
    if (vkInstance)
    {
        vkDestroyInstance(vkInstance, NULL);
        vkInstance = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyInstance() Succeeded\n", __func__);
    }
    
    if (gpFile)
    {
        fprintf(gpFile, "%s() => Program Terminated Successfully\n", __func__);
        fclose(gpFile);
        gpFile = NULL;
    }
}

//! Definition of Vulkan Related Functions
VkResult createVulkanInstance(void)
{
    // Function Declarations
    VkResult fillInstanceExtensionNames(void);
    VkResult fillValidationLayerNames(void);
    VkResult createValidationCallbackFunction(void);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Step - 1
    vkResult = fillInstanceExtensionNames();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => fillInstanceExtensionNames() Failed : %d !!!\n", __func__, vkResult);
        return VK_ERROR_INITIALIZATION_FAILED;
    }      
    else
        fprintf(gpFile, "%s() => fillInstanceExtensionNames() Succeeded\n", __func__);

    //! Fill Validation Layers
    if (bValidation == TRUE)
    {
        vkResult = fillValidationLayerNames();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => fillValidationLayerNames() Failed : %d !!!\n", __func__, vkResult);
            return VK_ERROR_INITIALIZATION_FAILED;
        }      
        else
            fprintf(gpFile, "%s() => fillValidationLayerNames() Succeeded\n", __func__);
    }

    //* Step - 2
    VkApplicationInfo vkApplicationInfo;
    memset((void*)&vkApplicationInfo, 0, sizeof(VkApplicationInfo));
    vkApplicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    vkApplicationInfo.pNext = NULL;
    vkApplicationInfo.pApplicationName = gpSzAppName;
    vkApplicationInfo.applicationVersion = 1;
    vkApplicationInfo.pEngineName = gpSzAppName;
    vkApplicationInfo.engineVersion = 1;
    vkApplicationInfo.apiVersion = VK_API_VERSION_1_4;

    //* Step - 3
    VkInstanceCreateInfo vkInstanceCreateInfo;
    memset((void*)&vkInstanceCreateInfo, 0, sizeof(VkInstanceCreateInfo));
    vkInstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    vkInstanceCreateInfo.pNext = NULL;
    vkInstanceCreateInfo.pApplicationInfo = &vkApplicationInfo;
    vkInstanceCreateInfo.enabledExtensionCount = enabledInstanceExtensionCount;
    vkInstanceCreateInfo.ppEnabledExtensionNames = enabledInstanceExtensionNames_array;

    if (bValidation == TRUE)
    {
        vkInstanceCreateInfo.enabledLayerCount = enabledValidationLayerCount;
        vkInstanceCreateInfo.ppEnabledLayerNames = enabledValidationLayerNames_array;
    }
    else
    {
        vkInstanceCreateInfo.enabledLayerCount = 0;
        vkInstanceCreateInfo.ppEnabledLayerNames = NULL;
    }
        
    //* Step - 4
    vkResult = vkCreateInstance(&vkInstanceCreateInfo, NULL, &vkInstance);
    if (vkResult == VK_ERROR_INCOMPATIBLE_DRIVER)
    {
        fprintf(gpFile, "%s() => vkCreateInstance() Failed Due To Incompatible Driver : %d!!!\n", __func__, vkResult);
        return vkResult;
    } 
    else if (vkResult == VK_ERROR_EXTENSION_NOT_PRESENT)
    {
        fprintf(gpFile, "%s() => vkCreateInstance() Failed Because Required Extension Is Not Present : %d!!!\n", __func__, vkResult);
        return vkResult;
    }
    else if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateInstance() Failed : %d!!!\n", __func__, vkResult);
        return vkResult;
    }
    else 
        fprintf(gpFile, "%s() => vkCreateInstance() Succeeded\n", __func__);

    //! Handling Validation Callbacks
    if (bValidation == TRUE)
    {
        vkResult = createValidationCallbackFunction();
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => createValidationCallbackFunction() Failed : %d !!!\n", __func__, vkResult);
            return VK_ERROR_INITIALIZATION_FAILED;
        }      
        else
            fprintf(gpFile, "%s() => createValidationCallbackFunction() Succeeded\n", __func__);
    }

    return vkResult;
}

VkResult fillInstanceExtensionNames(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Step - 1
    uint32_t instanceExtensionCount = 0;
    vkResult = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionCount, NULL);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 1 : vkEnumerateInstanceExtensionProperties() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => Call 1 : vkEnumerateInstanceExtensionProperties() Succeeded\n", __func__);

    //* Step - 2
    VkExtensionProperties *vkExtensionProperties_array = NULL;
    vkExtensionProperties_array = (VkExtensionProperties*)malloc(instanceExtensionCount * sizeof(VkExtensionProperties));
    if (vkExtensionProperties_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkExtensionProperties_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    vkResult = vkEnumerateInstanceExtensionProperties(NULL, &instanceExtensionCount, vkExtensionProperties_array);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 2 : vkEnumerateInstanceExtensionProperties() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => Call 2 : vkEnumerateInstanceExtensionProperties() Succeeded\n", __func__);

    //* Step - 3
    char **instanceExtensionNames_array = NULL;
    instanceExtensionNames_array = (char**)malloc(sizeof(char*) * instanceExtensionCount);
    if (instanceExtensionNames_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For instanceExtensionNames_array !!!\n", __func__);
        if (vkExtensionProperties_array)
        {
            free(vkExtensionProperties_array);
            vkExtensionProperties_array = NULL;
        }
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    for (uint32_t i = 0; i < instanceExtensionCount; i++)
    {
        instanceExtensionNames_array[i] = (char*)malloc(sizeof(char) * (strlen(vkExtensionProperties_array[i].extensionName) + 1));
        if (instanceExtensionNames_array[i] == NULL)
        {
            fprintf(gpFile, "%s() => malloc() Failed For instanceExtensionNames_array[%d] !!!\n", __func__, i);
            if (instanceExtensionNames_array)
            {
                free(instanceExtensionNames_array);
                instanceExtensionNames_array = NULL;
            }
            if (vkExtensionProperties_array)
            {
                free(vkExtensionProperties_array);
                vkExtensionProperties_array = NULL;
            }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        memcpy(instanceExtensionNames_array[i], vkExtensionProperties_array[i].extensionName, strlen(vkExtensionProperties_array[i].extensionName) + 1);

        fprintf(gpFile, "%s() => Vulkan Instance Extension Name : %s\n", __func__, instanceExtensionNames_array[i]);
    }

    //* Step - 4
    if (vkExtensionProperties_array)
    {
        free(vkExtensionProperties_array);
        vkExtensionProperties_array = NULL;
    }

    //* Step - 5
    VkBool32 vulkanSurfaceExtensionFound = VK_FALSE;
    VkBool32 win32SurfaceExtensionFound = VK_FALSE;
    VkBool32 debugReportExtensionFound = VK_FALSE;

    for (uint32_t i = 0; i < instanceExtensionCount; i++)
    {
        if (strcmp(instanceExtensionNames_array[i], VK_KHR_SURFACE_EXTENSION_NAME) == 0)
        {
            vulkanSurfaceExtensionFound = VK_TRUE;
            enabledInstanceExtensionNames_array[enabledInstanceExtensionCount++] = VK_KHR_SURFACE_EXTENSION_NAME;
        }
           
        if (strcmp(instanceExtensionNames_array[i], VK_KHR_WIN32_SURFACE_EXTENSION_NAME) == 0)
        {
            win32SurfaceExtensionFound = VK_TRUE;
            enabledInstanceExtensionNames_array[enabledInstanceExtensionCount++] = VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
        } 

        if (strcmp(instanceExtensionNames_array[i], VK_EXT_DEBUG_REPORT_EXTENSION_NAME) == 0)
        {
            debugReportExtensionFound = VK_TRUE;
            if (bValidation == TRUE)
                enabledInstanceExtensionNames_array[enabledInstanceExtensionCount++] = VK_EXT_DEBUG_REPORT_EXTENSION_NAME;
            else
            {
                // Array will not have entry of VK_EXT_DEBUG_REPORT_EXTENSION_NAME
            }            
        }
    }

    //* Step - 6
    if (instanceExtensionNames_array)
    {
        for (uint32_t i = 0; i < instanceExtensionCount; i++)
        {
            free(instanceExtensionNames_array[i]);
            instanceExtensionNames_array[i] = NULL;
        }
        free(instanceExtensionNames_array);
        instanceExtensionNames_array = NULL;
    }
    
    //* Step - 7
    if (vulkanSurfaceExtensionFound == VK_FALSE)
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => VK_KHR_SURFACE_EXTENSION_NAME Extension Not Found !!!\n", __func__);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => VK_KHR_SURFACE_EXTENSION_NAME Extension Found\n", __func__);

    if (win32SurfaceExtensionFound == VK_FALSE)
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => VK_KHR_WIN32_SURFACE_EXTENSION_NAME Extension Not Found !!!\n", __func__);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => VK_KHR_WIN32_SURFACE_EXTENSION_NAME Extension Found\n", __func__);

    if (debugReportExtensionFound == VK_FALSE)
    {
        if (bValidation == TRUE)
        {
            vkResult = VK_ERROR_INITIALIZATION_FAILED;
            fprintf(gpFile, "%s() => VALIDATION ON : VK_EXT_DEBUG_REPORT_EXTENSION_NAME Extension Not Supported !!!\n", __func__);
            return vkResult;
        }
        else
            fprintf(gpFile, "%s() => VALIDATION OFF : VK_EXT_DEBUG_REPORT_EXTENSION_NAME Extension Not Supported !!!\n", __func__);
    }
    else
    {
        if (bValidation == TRUE)
            fprintf(gpFile, "%s() => VALIDATION ON : VK_EXT_DEBUG_REPORT_EXTENSION_NAME Extension Supported\n", __func__);
        else
            fprintf(gpFile, "%s() => VALIDATION OFF : VK_EXT_DEBUG_REPORT_EXTENSION_NAME Extension Supported\n", __func__);
    }

    //* Step - 8
    for (uint32_t i = 0; i < enabledInstanceExtensionCount; i++)
        fprintf(gpFile, "%s() => Enabled Vulkan Instance Extension Name : %s\n", __func__, enabledInstanceExtensionNames_array[i]);

    return vkResult;
}

VkResult fillValidationLayerNames(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    uint32_t validationLayerCount = 0;
    vkResult = vkEnumerateInstanceLayerProperties(&validationLayerCount, NULL);
    if (vkResult != VK_SUCCESS)  
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => Call 1 : vkEnumerateInstanceLayerProperties() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Call 1 : vkEnumerateInstanceLayerProperties() Succeeded\n", __func__);

    VkLayerProperties *vkLayerProperties_array = NULL;
    vkLayerProperties_array = (VkLayerProperties*)malloc(validationLayerCount * sizeof(VkLayerProperties));
    if (vkLayerProperties_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkLayerProperties_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    vkResult = vkEnumerateInstanceLayerProperties(&validationLayerCount, vkLayerProperties_array);
    if (vkResult != VK_SUCCESS)  
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => Call 2 : vkEnumerateInstanceLayerProperties() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Call 2 : vkEnumerateInstanceLayerProperties() Succeeded\n", __func__);

    char **validationLayerNames_array = NULL;
    validationLayerNames_array = (char**)malloc(sizeof(char*) * validationLayerCount);
    if (validationLayerNames_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For validationLayerNames_array !!!\n", __func__);
        if (vkLayerProperties_array)
        {
            free(vkLayerProperties_array);
            vkLayerProperties_array = NULL;
        }
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    for (uint32_t i = 0; i < validationLayerCount; i++)
    {
        validationLayerNames_array[i] = (char*)malloc(sizeof(char) * (strlen(vkLayerProperties_array[i].layerName) + 1));
        if (validationLayerNames_array[i] == NULL)
        {
            fprintf(gpFile, "%s() => malloc() Failed For validationLayerNames_array[%d] !!!\n", __func__, i);
            if (validationLayerNames_array)
            {
                free(validationLayerNames_array);
                validationLayerNames_array = NULL;
            }
            if (vkLayerProperties_array)
            {
                free(vkLayerProperties_array);
                vkLayerProperties_array = NULL;
            }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        memcpy(validationLayerNames_array[i], vkLayerProperties_array[i].layerName, strlen(vkLayerProperties_array[i].layerName) + 1);

        fprintf(gpFile, "%s() => Vulkan Instance Layer Name : %s\n", __func__, validationLayerNames_array[i]);
    }

    if (vkLayerProperties_array)
    {
        free(vkLayerProperties_array);
        vkLayerProperties_array = NULL;
    }

    VkBool32 validationLayerFound = VK_FALSE;
    for (uint32_t i = 0; i < validationLayerCount; i++)
    {
        if (strcmp(validationLayerNames_array[i], "VK_LAYER_KHRONOS_validation") == 0)
        {
            validationLayerFound = VK_TRUE;
            enabledValidationLayerNames_array[enabledValidationLayerCount++] = "VK_LAYER_KHRONOS_validation";
        }
    }

    if (validationLayerNames_array)
    {
        for (uint32_t i = 0; i < validationLayerCount; i++)
        {
            free(validationLayerNames_array[i]);
            validationLayerNames_array[i] = NULL;
        }
        free(validationLayerNames_array);
        validationLayerNames_array = NULL;
    }

    if (validationLayerFound == VK_FALSE)
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => VK_LAYER_KHRONOS_validation Not Supported !!!\n", __func__);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => VK_LAYER_KHRONOS_validation Supported\n", __func__);

    for (uint32_t i = 0; i < enabledValidationLayerCount; i++)
        fprintf(gpFile, "%s() => Enabled Vulkan Validation Layer Name : %s\n", __func__, enabledValidationLayerNames_array[i]);

    return vkResult;
}

VkResult createValidationCallbackFunction(void)
{
    // Callback Declaration
    VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
        VkDebugReportFlagsEXT,
        VkDebugReportObjectTypeEXT,
        uint64_t,
        size_t,
        int32_t,
        const char*,
        const char*,
        void*
    );

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;
    PFN_vkCreateDebugReportCallbackEXT vkCreateDebugReportCallbackEXT_fnptr = NULL;

    // Code
    
    //* Get the required function pointers
    vkCreateDebugReportCallbackEXT_fnptr = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(vkInstance, "vkCreateDebugReportCallbackEXT");
    if (vkCreateDebugReportCallbackEXT_fnptr == NULL)
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => vkGetInstanceProcAddr() Failed To Get Function Pointer For vkCreateDebugReportCallbackEXT !!!\n", __func__);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkGetInstanceProcAddr() Succeeded To Get Function Pointer For vkCreateDebugReportCallbackEXT\n", __func__);

    vkDestroyDebugReportCallbackEXT_fnptr = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(vkInstance, "vkDestroyDebugReportCallbackEXT");
    if (vkDestroyDebugReportCallbackEXT_fnptr == NULL)
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => vkGetInstanceProcAddr() Failed To Get Function Pointer For vkDestroyDebugReportCallbackEXT !!!\n", __func__);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkGetInstanceProcAddr() Succeeded To Get Function Pointer For vkDestroyDebugReportCallbackEXT\n", __func__);

    //* Get the Vulkan Debug Report Callback Object
    VkDebugReportCallbackCreateInfoEXT vkDebugReportCallbackCreateInfoEXT;
    memset((void*)&vkDebugReportCallbackCreateInfoEXT, 0, sizeof(VkDebugReportCallbackCreateInfoEXT));
    vkDebugReportCallbackCreateInfoEXT.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CREATE_INFO_EXT;
    vkDebugReportCallbackCreateInfoEXT.pNext = NULL;
    vkDebugReportCallbackCreateInfoEXT.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
    vkDebugReportCallbackCreateInfoEXT.pUserData = NULL;
    vkDebugReportCallbackCreateInfoEXT.pfnCallback = debugReportCallback;

    vkResult = vkCreateDebugReportCallbackEXT_fnptr(vkInstance, &vkDebugReportCallbackCreateInfoEXT, NULL, &vkDebugReportCallbackEXT);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateDebugReportCallbackEXT_fnptr() Failed : %d !!!\n", __func__, vkResult);
        return VK_ERROR_INITIALIZATION_FAILED;
    }      
    else
        fprintf(gpFile, "%s() => vkCreateDebugReportCallbackEXT_fnptr() Succeeded\n", __func__);

    return vkResult;
}

VkResult getSupportedSurface(void)
{
    // Code

    //* Step - 1
    VkWin32SurfaceCreateInfoKHR vkWin32SurfaceCreateInfoKHR;
    VkResult vkResult = VK_SUCCESS;

    //* Step - 2
    memset((void*)&vkWin32SurfaceCreateInfoKHR, 0, sizeof(VkWin32SurfaceCreateInfoKHR));

    //* Step - 3
    vkWin32SurfaceCreateInfoKHR.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    vkWin32SurfaceCreateInfoKHR.pNext = NULL;
    vkWin32SurfaceCreateInfoKHR.flags = 0;
    vkWin32SurfaceCreateInfoKHR.hinstance = (HINSTANCE)GetWindowLongPtr(ghwnd, GWLP_HINSTANCE);
    vkWin32SurfaceCreateInfoKHR.hwnd = ghwnd;

    //* Step - 4
    vkResult = vkCreateWin32SurfaceKHR(vkInstance, &vkWin32SurfaceCreateInfoKHR, NULL, &vkSurfaceKHR);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateWin32SurfaceKHR() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateWin32SurfaceKHR() Succeeded\n", __func__);

    return vkResult;
}

VkResult getPhysicalDevice(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;
    
    // Code

    //* Step - 2
    vkResult = vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceCount, NULL);
    if (vkResult == VK_SUCCESS)
        fprintf(gpFile, "%s() Call 1 => vkEnumeratePhysicalDevices() Succeeded\n", __func__); 
    else if (physicalDeviceCount == 0)
    {
        fprintf(gpFile, "%s() => vkEnumeratePhysicalDevices() Returned 0 Devices !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
    {
        fprintf(gpFile, "%s() Call 1 => vkEnumeratePhysicalDevices() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //* Step - 3
    vkPhysicalDevice_array = (VkPhysicalDevice*)malloc(physicalDeviceCount * sizeof(VkPhysicalDevice));
    if (vkPhysicalDevice_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkPhysicalDevice_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 4
    vkResult = vkEnumeratePhysicalDevices(vkInstance, &physicalDeviceCount, vkPhysicalDevice_array);
    if (vkResult != VK_SUCCESS)
    {   
        fprintf(gpFile, "%s() Call 2 => vkEnumeratePhysicalDevices() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
       fprintf(gpFile, "%s() Call 2 => vkEnumeratePhysicalDevices() Succeeded\n", __func__);

    //* Step - 5
    VkBool32 bFound = VK_FALSE;
    for (uint32_t i = 0; i < physicalDeviceCount; i++)
    {
        //* Step - 5.1
        uint32_t queueCount = UINT32_MAX;

        //* Step - 5.2
        vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &queueCount, NULL);

        //* Step - 5.3
        VkQueueFamilyProperties *vkQueueFamilyProperties_array = NULL;
        vkQueueFamilyProperties_array = (VkQueueFamilyProperties*)malloc(queueCount * sizeof(VkQueueFamilyProperties));
        if (vkQueueFamilyProperties_array == NULL)
        {
            fprintf(gpFile, "%s() => malloc() Failed For vkQueueFamilyProperties_array !!!\n", __func__);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        //* Step - 5.4
        vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &queueCount, vkQueueFamilyProperties_array);

        //* Step - 5.5
        VkBool32 *isQueueSurfaceSupported_array = NULL;
        isQueueSurfaceSupported_array = (VkBool32*)malloc(queueCount * sizeof(VkBool32));
        if (isQueueSurfaceSupported_array == NULL)
        {
            fprintf(gpFile, "%s() => malloc() Failed For isQueueSurfaceSupported_array\n", __func__);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        //* Step - 5.6
        for (uint32_t j = 0; j < queueCount; j++)
            vkGetPhysicalDeviceSurfaceSupportKHR(vkPhysicalDevice_array[i], j, vkSurfaceKHR, &isQueueSurfaceSupported_array[j]);

        //* Step - 5.7
        for (uint32_t j = 0; j < queueCount; j++)
        {
            if (vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                if (isQueueSurfaceSupported_array[j] == VK_TRUE)
                {
                    vkPhysicalDevice_selected = vkPhysicalDevice_array[i];
                    graphicsQueueFamilyIndex_selected = j;
                    bFound = VK_TRUE;
                    break;
                }
            }
        }

        //* Step - 5.8
        if (isQueueSurfaceSupported_array)
        {
            free(isQueueSurfaceSupported_array);
            fprintf(gpFile, "%s() => free() Succeeded For isQueueSurfaceSupported_array\n", __func__);
            isQueueSurfaceSupported_array = NULL;
        }

        if (vkQueueFamilyProperties_array)
        {
            free(vkQueueFamilyProperties_array);
            fprintf(gpFile, "%s() => free() Succeeded For vkQueueFamilyProperties_array\n", __func__);
            vkQueueFamilyProperties_array = NULL;
        }

        //* Step - 5.9
        if (bFound == VK_TRUE)
            break;
        
    }

    //* Step - 5.10
    if (bFound == VK_TRUE)
        fprintf(gpFile, "%s() => Succeeded To Obtain Graphics Supported Physical Device\n", __func__);

    //* Step - 6
    else
    {
        fprintf(gpFile, "%s() => Failed To Obtain Graphics Supported Physical Device !!!\n", __func__);
        if (vkPhysicalDevice_array)
        {
            free(vkPhysicalDevice_array);
            fprintf(gpFile, "%s() => free() Succeeded For vkPhysicalDevice_array\n", __func__);
            vkPhysicalDevice_array = NULL;
        }
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    
    //* Step - 7
    memset((void*)&vkPhysicalDeviceMemoryProperties, 0, sizeof(VkPhysicalDeviceMemoryProperties));

    //* Step - 8
    vkGetPhysicalDeviceMemoryProperties(vkPhysicalDevice_selected, &vkPhysicalDeviceMemoryProperties);

    //* Step - 9
    VkPhysicalDeviceFeatures vkPhysicalDeviceFeatures_array;
    memset((void*)&vkPhysicalDeviceFeatures_array, 0, sizeof(VkPhysicalDeviceFeatures));
    vkGetPhysicalDeviceFeatures(vkPhysicalDevice_selected, &vkPhysicalDeviceFeatures_array);

    //* Step - 10
    if (vkPhysicalDeviceFeatures_array.tessellationShader == VK_TRUE)
        fprintf(gpFile, "%s() => Selected Physical Device Supports Tessellation Shader\n", __func__);
    else
        fprintf(gpFile, "%s() => Selected Physical Device Does Not Support Tessellation Shader !!!\n", __func__);

    if (vkPhysicalDeviceFeatures_array.geometryShader == VK_TRUE)
        fprintf(gpFile, "%s() => Selected Physical Device Supports Geometry Shader\n", __func__);
    else
        fprintf(gpFile, "%s() => Selected Physical Device Does Not Support Geometry Shader !!!\n", __func__);

    return vkResult;
}

VkResult printVkInfo(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    fprintf(gpFile, "\nVULKAN INFORMATION\n");
    fprintf(gpFile, "------------------------------------------------------------------------------------------------\n");
    
    //* Step - 3.1
    for (uint32_t i = 0; i < physicalDeviceCount; i++)
    {
        //* Step - 3.2
        VkPhysicalDeviceProperties vkPhysicalDeviceProperties;
        memset((void*)&vkPhysicalDeviceProperties, 0, sizeof(VkPhysicalDeviceProperties));
        vkGetPhysicalDeviceProperties(vkPhysicalDevice_array[i], &vkPhysicalDeviceProperties);

        //* Step - 3.3
        uint32_t majorVersion = VK_API_VERSION_MAJOR(vkPhysicalDeviceProperties.apiVersion);
        uint32_t minorVersion = VK_API_VERSION_MINOR(vkPhysicalDeviceProperties.apiVersion);
        uint32_t patchVersion = VK_API_VERSION_PATCH(vkPhysicalDeviceProperties.apiVersion);
        fprintf(gpFile, "Vulkan API Version : %u.%u.%u\n", majorVersion, minorVersion, patchVersion);

        //* Step - 3.4
        fprintf(gpFile, "Device Name : %s\n", vkPhysicalDeviceProperties.deviceName);

        //* Step - 3.5
        switch(vkPhysicalDeviceProperties.deviceType)
        {
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                fprintf(gpFile, "Device Type : Integrated GPU (iGPU)\n");
            break;

            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                fprintf(gpFile, "Device Type : Discrete GPU (dGPU)\n");
            break;

            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                fprintf(gpFile, "Device Type : Virtual GPU (vGPU)\n");
            break;

            case VK_PHYSICAL_DEVICE_TYPE_CPU:
                fprintf(gpFile, "Device Type : CPU\n");
            break;

            case VK_PHYSICAL_DEVICE_TYPE_OTHER:
                fprintf(gpFile, "Device Type : Other\n");
            break;

            default:
                fprintf(gpFile, "Device Type : UNKNOWN\n");
            break;
        }

        //* Step - 3.6
        fprintf(gpFile, "Vendor ID : 0x%4x\n", vkPhysicalDeviceProperties.vendorID);

        //* Step - 3.7
        fprintf(gpFile, "Device ID : 0x%4x\n", vkPhysicalDeviceProperties.deviceID);
    }

    fprintf(gpFile, "------------------------------------------------------------------------------------------------\n\n");

    //* Step - 3.8
    if (vkPhysicalDevice_array)
    {
        free(vkPhysicalDevice_array);
        fprintf(gpFile, "%s() => free() Succeeded For vkPhysicalDevice_array\n", __func__);
        vkPhysicalDevice_array = NULL;
    }

    return vkResult;
}

VkResult fillDeviceExtensionNames(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Step - 1
    uint32_t deviceExtensionCount = 0;
    vkResult = vkEnumerateDeviceExtensionProperties(vkPhysicalDevice_selected, NULL, &deviceExtensionCount, NULL);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 1 : vkEnumerateDeviceExtensionProperties() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => Call 1 : vkEnumerateDeviceExtensionProperties() Succeeded\n", __func__);

    //* Step - 2
    VkExtensionProperties *vkExtensionProperties_array = NULL;
    vkExtensionProperties_array = (VkExtensionProperties*)malloc(deviceExtensionCount * sizeof(VkExtensionProperties));
    if (vkExtensionProperties_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkExtensionProperties_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    vkResult = vkEnumerateDeviceExtensionProperties(vkPhysicalDevice_selected, NULL, &deviceExtensionCount, vkExtensionProperties_array);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 2 : vkEnumerateDeviceExtensionProperties() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => Call 2 : vkEnumerateDeviceExtensionProperties() Succeeded\n", __func__);

    //* Step - 3
    char **deviceExtensionNames_array = NULL;
    deviceExtensionNames_array = (char**)malloc(sizeof(char*) * deviceExtensionCount);
    if (deviceExtensionNames_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For deviceExtensionNames_array !!!\n", __func__);
        if (vkExtensionProperties_array)
        {
            free(vkExtensionProperties_array);
            vkExtensionProperties_array = NULL;
        }
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    for (uint32_t i = 0; i < deviceExtensionCount; i++)
    {
        deviceExtensionNames_array[i] = (char*)malloc(sizeof(char) * (strlen(vkExtensionProperties_array[i].extensionName) + 1));
        if (deviceExtensionNames_array[i] == NULL)
        {
            fprintf(gpFile, "%s() => malloc() Failed For deviceExtensionNames_array[%d] !!!\n", __func__, i);
            if (deviceExtensionNames_array)
            {
                free(deviceExtensionNames_array);
                deviceExtensionNames_array = NULL;
            }
            if (vkExtensionProperties_array)
            {
                free(vkExtensionProperties_array);
                vkExtensionProperties_array = NULL;
            }
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        memcpy(deviceExtensionNames_array[i], vkExtensionProperties_array[i].extensionName, strlen(vkExtensionProperties_array[i].extensionName) + 1);

        fprintf(gpFile, "%s() => Vulkan Device Extension Name : %s\n", __func__, deviceExtensionNames_array[i]);
    }

    fprintf(gpFile, "\n------------------------------------------------------------------------------------------------\n");
    fprintf(gpFile, "%s() => Vulkan Device Extension Count : %d\n", __func__, deviceExtensionCount);
    fprintf(gpFile, "------------------------------------------------------------------------------------------------\n\n");

    //* Step - 4
    if (vkExtensionProperties_array)
    {
        free(vkExtensionProperties_array);
        vkExtensionProperties_array = NULL;
    }

    //* Step - 5
    VkBool32 vulkanSwapchainExtensionFound = VK_FALSE;
    for (uint32_t i = 0; i < deviceExtensionCount; i++)
    {
        if (strcmp(deviceExtensionNames_array[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
        {
            vulkanSwapchainExtensionFound = VK_TRUE;
            enabledDeviceExtensionNames_array[enabledDeviceExtensionCount++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        }
    }

    //* Step - 6
    if (deviceExtensionNames_array)
    {
        for (uint32_t i = 0; i < deviceExtensionCount; i++)
        {
            free(deviceExtensionNames_array[i]);
            deviceExtensionNames_array[i] = NULL;
        }
        free(deviceExtensionNames_array);
        deviceExtensionNames_array = NULL;
    }
    
    //* Step - 7
    if (vulkanSwapchainExtensionFound == VK_FALSE)
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => VK_KHR_SWAPCHAIN_EXTENSION_NAME Extension Not Found !!!\n", __func__);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => VK_KHR_SWAPCHAIN_EXTENSION_NAME Extension Found\n", __func__);

    //* Step - 8
    for (uint32_t i = 0; i < enabledDeviceExtensionCount; i++)
        fprintf(gpFile, "%s() => Enabled Vulkan Device Extension Name : %s\n", __func__, enabledDeviceExtensionNames_array[i]);

    return vkResult;

}

VkResult createVulkanDevice(void)
{
    // Function Declaration
    VkResult fillDeviceExtensionNames(void);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;
    float queuePriorities[1] = { 1.0f };

    // Code

    //* Step - 1
    vkResult = fillDeviceExtensionNames();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => fillDeviceExtensionNames() Failed : %d !!!\n", __func__, vkResult);
        return VK_ERROR_INITIALIZATION_FAILED;
    }      
    else
        fprintf(gpFile, "%s() => fillDeviceExtensionNames() Succeeded\n", __func__);

    
    //* Step - 2
    //! Newly Added Code
    VkDeviceQueueCreateInfo vkDeviceQueueCreateInfo;
    memset((void*)&vkDeviceQueueCreateInfo, 0, sizeof(VkDeviceQueueCreateInfo));
    vkDeviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    vkDeviceQueueCreateInfo.pNext = NULL;
    vkDeviceQueueCreateInfo.flags = 0;
    vkDeviceQueueCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex_selected;
    vkDeviceQueueCreateInfo.queueCount = 1;
    vkDeviceQueueCreateInfo.pQueuePriorities = queuePriorities;

    VkDeviceCreateInfo vkDeviceCreateInfo;
    memset((void*)&vkDeviceCreateInfo, 0, sizeof(VkDeviceCreateInfo));
    vkDeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    vkDeviceCreateInfo.pNext = NULL;
    vkDeviceCreateInfo.flags = 0;
    vkDeviceCreateInfo.enabledExtensionCount = enabledDeviceExtensionCount;
    vkDeviceCreateInfo.ppEnabledExtensionNames = enabledDeviceExtensionNames_array;
    vkDeviceCreateInfo.queueCreateInfoCount = 1;
    vkDeviceCreateInfo.pQueueCreateInfos = &vkDeviceQueueCreateInfo;
    vkDeviceCreateInfo.pEnabledFeatures = NULL;
    //* Deprecated in Vulkan Spec
    vkDeviceCreateInfo.enabledLayerCount = 0;
    vkDeviceCreateInfo.ppEnabledLayerNames = NULL;

    //* Step - 3
    vkResult = vkCreateDevice(vkPhysicalDevice_selected, &vkDeviceCreateInfo, NULL, &vkDevice);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateDevice() Failed : %d !!!\n", __func__, vkResult);
        return VK_ERROR_INITIALIZATION_FAILED;
    }      
    else
        fprintf(gpFile, "%s() => vkCreateDevice() Succeeded\n", __func__);

    return vkResult;
}

void getDeviceQueue(void)
{
    // Code
    vkGetDeviceQueue(vkDevice, graphicsQueueFamilyIndex_selected, 0, &vkQueue);

    if (vkQueue == VK_NULL_HANDLE)
    {
        fprintf(gpFile, "%s() => vkGetDeviceQueue() returned NULL for vkQueue !!!\n", __func__);
        return;
    }
    else
        fprintf(gpFile, "%s() => vkGetDeviceQueue() Succeeded ...\n", __func__);
}

VkResult getPhysicalDeviceSurfaceFormatAndColorSpace(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Step - 1
    uint32_t formatCount = 0;
    vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &formatCount, NULL);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 1 : vkGetPhysicalDeviceSurfaceFormatsKHR() Failed : %d !!!\n", __func__, vkResult);
    else if (formatCount == 0)
    {
        fprintf(gpFile, "%s() => Call 1 : vkGetPhysicalDeviceSurfaceFormatsKHR() Returned 0 Devices !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Call 1 : vkGetPhysicalDeviceSurfaceFormatsKHR() Succeeded\n", __func__);

    //* Step - 2
    VkSurfaceFormatKHR *vkSurfaceFormatKHR_array = (VkSurfaceFormatKHR*)malloc(formatCount * sizeof(VkSurfaceFormatKHR));
    if (vkSurfaceFormatKHR_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkSurfaceFormatKHR_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 3
    vkResult = vkGetPhysicalDeviceSurfaceFormatsKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &formatCount, vkSurfaceFormatKHR_array);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 2 : vkGetPhysicalDeviceSurfaceFormatsKHR() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => Call 2 : vkGetPhysicalDeviceSurfaceFormatsKHR() Succeeded\n", __func__);

    //* Step - 4
    if (formatCount == 1 && vkSurfaceFormatKHR_array[0].format == VK_FORMAT_UNDEFINED)
        vkFormat_color = VK_FORMAT_B8G8R8A8_UNORM;
    else
    {
        vkFormat_color = vkSurfaceFormatKHR_array[0].format;
        vkColorSpaceKHR = vkSurfaceFormatKHR_array[0].colorSpace;
    }

    //* Step - 5
    if (vkSurfaceFormatKHR_array)
    {
        free(vkSurfaceFormatKHR_array);
        vkSurfaceFormatKHR_array = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For vkSurfaceFormatKHR_array\n", __func__);
    }

    return vkResult;
}

VkResult getPhysicalDevicePresentMode(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Step - 1
    uint32_t presentModeCount = 0;
    vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &presentModeCount, NULL);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 1 : vkGetPhysicalDeviceSurfacePresentModesKHR() Failed : %d !!!\n", __func__, vkResult);
    else if (presentModeCount == 0)
    {
        fprintf(gpFile, "%s() => Call 1 : vkGetPhysicalDeviceSurfacePresentModesKHR() Returned 0 Devices !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Call 1 : vkGetPhysicalDeviceSurfacePresentModesKHR() Succeeded\n", __func__);

    //* Step - 2
    VkPresentModeKHR *vkPresentModeKHR_array = (VkPresentModeKHR*)malloc(presentModeCount * sizeof(VkPresentModeKHR));
    if (vkPresentModeKHR_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkPresentModeKHR_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 3
    vkResult = vkGetPhysicalDeviceSurfacePresentModesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &presentModeCount, vkPresentModeKHR_array);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 2 : vkGetPhysicalDeviceSurfacePresentModesKHR() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => Call 2 : vkGetPhysicalDeviceSurfacePresentModesKHR() Succeeded\n", __func__);


    //* Step - 4
    for (uint32_t i = 0; i < presentModeCount; i++)
    {
        if (vkPresentModeKHR_array[i] == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            vkPresentModeKHR = VK_PRESENT_MODE_MAILBOX_KHR;
            fprintf(gpFile, "\n------------------------------------------------------------------------------------------------\n");
            fprintf(gpFile, "Vulkan Physical Device Present Mode : VK_PRESENT_MODE_MAILBOX_KHR");
            fprintf(gpFile, "\n------------------------------------------------------------------------------------------------\n\n");
            break;
        }
    }

    if (vkPresentModeKHR != VK_PRESENT_MODE_MAILBOX_KHR)
    {
        vkPresentModeKHR = VK_PRESENT_MODE_FIFO_KHR;
        fprintf(gpFile, "\n------------------------------------------------------------------------------------------------\n");
        fprintf(gpFile, "Vulkan Physical Device Present Mode : VK_PRESENT_MODE_FIFO_KHR");
        fprintf(gpFile, "\n------------------------------------------------------------------------------------------------\n\n");
    }
        

    //* Step - 5
    if (vkPresentModeKHR_array)
    {
        free(vkPresentModeKHR_array);
        vkPresentModeKHR_array = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For vkPresentModeKHR_array\n", __func__);
    }

    return vkResult;
}

VkResult createSwapchain(VkBool32 vsync)
{
    // Function Declarations
    VkResult getPhysicalDeviceSurfaceFormatAndColorSpace(void);
    VkResult getPhysicalDevicePresentMode(void);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Step - 1
    vkResult = getPhysicalDeviceSurfaceFormatAndColorSpace();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => getPhysicalDeviceSurfaceFormatAndColorSpace() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => getPhysicalDeviceSurfaceFormatAndColorSpace() Succeeded\n", __func__);

    //* Step - 2
    VkSurfaceCapabilitiesKHR vkSurfaceCapabilitiesKHR;
    memset((void*)&vkSurfaceCapabilitiesKHR, 0, sizeof(VkSurfaceCapabilitiesKHR));
    vkResult = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkPhysicalDevice_selected, vkSurfaceKHR, &vkSurfaceCapabilitiesKHR);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkGetPhysicalDeviceSurfaceCapabilitiesKHR() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkGetPhysicalDeviceSurfaceCapabilitiesKHR() Succeeded\n", __func__);

    //* Step - 3
    uint32_t testingNumberOfSwapchainImages = vkSurfaceCapabilitiesKHR.minImageCount + 1;
    uint32_t desiredNumberOfSwapchainImages = 0;

    if (vkSurfaceCapabilitiesKHR.maxImageCount > 0 && vkSurfaceCapabilitiesKHR.maxImageCount < testingNumberOfSwapchainImages)
        desiredNumberOfSwapchainImages = vkSurfaceCapabilitiesKHR.maxImageCount;
    else
        desiredNumberOfSwapchainImages = vkSurfaceCapabilitiesKHR.minImageCount;

    //* Step - 4
    memset((void*)&vkExtent2D_swapchain, 0, sizeof(VkExtent2D));
    if (vkSurfaceCapabilitiesKHR.currentExtent.width != UINT32_MAX)
    {
        vkExtent2D_swapchain.width = vkSurfaceCapabilitiesKHR.currentExtent.width;
        vkExtent2D_swapchain.height = vkSurfaceCapabilitiesKHR.currentExtent.height;
        fprintf(gpFile, "%s() => [If Block] => Swapchain Image Width x Swapchain Image Height = %d x %d\n", __func__, vkExtent2D_swapchain.width, vkExtent2D_swapchain.height);
    }
    else
    {
        // If surface size is already defined, then swapchain image size must match with it
        VkExtent2D vkExtent2D;
        memset((void*)&vkExtent2D, 0, sizeof(VkExtent2D));

        vkExtent2D.width = (uint32_t)winWidth;
        vkExtent2D.height = (uint32_t)winHeight;

        vkExtent2D_swapchain.width = glm::max(
            vkSurfaceCapabilitiesKHR.minImageExtent.width, 
            glm::min(vkSurfaceCapabilitiesKHR.maxImageExtent.width, vkExtent2D.width)
        );

        vkExtent2D_swapchain.height = glm::max(
            vkSurfaceCapabilitiesKHR.minImageExtent.height,
            glm::min(vkSurfaceCapabilitiesKHR.maxImageExtent.height, vkExtent2D.height)
        );

        fprintf(gpFile, "%s() => [Else Block] => Swapchain Image Width x Swapchain Image Height = %d x %d\n", __func__, vkExtent2D_swapchain.width, vkExtent2D_swapchain.height);
    }

    //* Step - 5
    //! VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT => Mandatory
    //! VK_IMAGE_USAGE_TRANSFER_SRC_BIT => Optional (Useful for Texture, FBO, Compute)
    VkImageUsageFlags vkImageUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    //* Step - 6
    VkSurfaceTransformFlagBitsKHR vkSurfaceTransformFlagBitsKHR;
    if (vkSurfaceCapabilitiesKHR.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) //* Check For Identity Matrix
        vkSurfaceTransformFlagBitsKHR = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    else
        vkSurfaceTransformFlagBitsKHR = vkSurfaceCapabilitiesKHR.currentTransform;

    //* Step - 7
    vkResult = getPhysicalDevicePresentMode();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => getPhysicalDevicePresentMode() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => getPhysicalDevicePresentMode() Succeeded\n", __func__);

    //* Step - 8
    VkSwapchainCreateInfoKHR vkSwapchainCreateInfoKHR;
    memset((void*)&vkSwapchainCreateInfoKHR, 0, sizeof(VkSwapchainCreateInfoKHR));
    vkSwapchainCreateInfoKHR.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    vkSwapchainCreateInfoKHR.pNext = NULL;
    vkSwapchainCreateInfoKHR.flags = 0;
    vkSwapchainCreateInfoKHR.surface = vkSurfaceKHR;
    vkSwapchainCreateInfoKHR.minImageCount = desiredNumberOfSwapchainImages;
    vkSwapchainCreateInfoKHR.imageFormat = vkFormat_color;
    vkSwapchainCreateInfoKHR.imageColorSpace = vkColorSpaceKHR;
    vkSwapchainCreateInfoKHR.imageExtent.width = vkExtent2D_swapchain.width;
    vkSwapchainCreateInfoKHR.imageExtent.height = vkExtent2D_swapchain.height;
    vkSwapchainCreateInfoKHR.imageUsage = vkImageUsageFlags;
    vkSwapchainCreateInfoKHR.preTransform = vkSurfaceTransformFlagBitsKHR;
    vkSwapchainCreateInfoKHR.imageArrayLayers = 1;
    vkSwapchainCreateInfoKHR.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkSwapchainCreateInfoKHR.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    vkSwapchainCreateInfoKHR.presentMode = vkPresentModeKHR;
    vkSwapchainCreateInfoKHR.clipped = VK_TRUE;

    vkResult = vkCreateSwapchainKHR(vkDevice, &vkSwapchainCreateInfoKHR, NULL, &vkSwapchainKHR);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateSwapchainKHR() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateSwapchainKHR() Succeeded\n", __func__);


    return VK_SUCCESS;
}

VkResult createImagesAndImageViews(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Step - 1
    vkResult = vkGetSwapchainImagesKHR(vkDevice, vkSwapchainKHR, &swapchainImageCount, NULL);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 1 : vkGetSwapchainImagesKHR() Failed : %d !!!\n", __func__, vkResult);
    else if (swapchainImageCount == 0)
    {
        fprintf(gpFile, "%s() => Call 1 : vkGetSwapchainImagesKHR() Returned 0 Images !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Call 1 : vkGetSwapchainImagesKHR() => Swapchain Image Count = %d\n", __func__, swapchainImageCount);


    //* Step - 2
    swapchainImage_array = (VkImage*)malloc(swapchainImageCount * sizeof(VkImage));
    if (swapchainImage_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For swapchainImage_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 3
    vkResult = vkGetSwapchainImagesKHR(vkDevice, vkSwapchainKHR, &swapchainImageCount, swapchainImage_array);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => Call 2 : vkGetSwapchainImagesKHR() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => Call 2 : vkGetSwapchainImagesKHR() Succeeded\n", __func__);


    //* Step - 4
    swapchainImageView_array = (VkImageView*)malloc(swapchainImageCount * sizeof(VkImageView));
    if (swapchainImageView_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For swapchainImageView_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 5
    VkImageViewCreateInfo vkImageViewCreateInfo;
    memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
    vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vkImageViewCreateInfo.pNext = NULL;
    vkImageViewCreateInfo.flags = 0;
    vkImageViewCreateInfo.format = vkFormat_color;
    vkImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    vkImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    vkImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    vkImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    vkImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    vkImageViewCreateInfo.subresourceRange.levelCount = 1;
    vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    vkImageViewCreateInfo.subresourceRange.layerCount = 1;
    vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

    //* Step - 6
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        vkImageViewCreateInfo.image = swapchainImage_array[i];
        vkResult = vkCreateImageView(vkDevice, &vkImageViewCreateInfo, NULL, &swapchainImageView_array[i]);
        if (vkResult != VK_SUCCESS)
            fprintf(gpFile, "%s() => vkCreateImageView() Failed For Index : %d, Error Code : %d !!!\n", __func__, i, vkResult);
        else
            fprintf(gpFile, "%s() => vkCreateImageView() Succeeded For Index : %d\n", __func__, i);
    }

    return vkResult;
}

VkResult createCommandPool(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    VkCommandPoolCreateInfo vkCommandPoolCreateInfo;
    memset((void*)&vkCommandPoolCreateInfo, 0, sizeof(VkCommandPoolCreateInfo));
    vkCommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    vkCommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCommandPoolCreateInfo.pNext = NULL;
    vkCommandPoolCreateInfo.queueFamilyIndex = graphicsQueueFamilyIndex_selected;

    vkResult = vkCreateCommandPool(vkDevice, &vkCommandPoolCreateInfo, NULL, &vkCommandPool);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateCommandPool() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateCommandPool() Succeeded\n", __func__);

    return vkResult;
}

VkResult createCommandBuffers(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Step - 1
    VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo;
    memset((void*)&vkCommandBufferAllocateInfo, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; 
    vkCommandBufferAllocateInfo.pNext = NULL;
    vkCommandBufferAllocateInfo.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkCommandBufferAllocateInfo.commandBufferCount = 1;

    //* Step - 2
    vkCommandBuffer_array = (VkCommandBuffer*)malloc(swapchainImageCount * sizeof(VkCommandBuffer));
    if (vkCommandBuffer_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkCommandBuffer_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 3
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo, &vkCommandBuffer_array[i]);
        if (vkResult != VK_SUCCESS)
            fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For Index : %d, Error Code : %d !!!\n", __func__, i, vkResult);
        else
            fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Succeeded For Index : %d\n", __func__, i);
    }

    return vkResult;
}

VkResult createVertexBuffer(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Step - 3
    float triangle_position[] = 
    {
        0.0f,   1.0f,   0.0f,
        -1.0f,  -1.0f,  0.0f,
        1.0f,   -1.0f,  0.0f  
    };

    float triangle_color[] = 
    {
        0.498f,   0.450f,   0.443f,
        0.498f,   0.450f,   0.443f,
        0.498f,   0.450f,   0.443f
    };

    float rectangle_position[] = 
    {
        // Triangle 1
        1.0f,   1.0f,   0.0f,   // Top Right
        -1.0f,  1.0f,   0.0f,   // Top Left
        -1.0f,  -1.0f,  0.0f,   // Bottom Left

        // Triangle 2
        -1.0f,  -1.0f,  0.0f,   // Bottom Left
        1.0f,   -1.0f,  0.0f,   // Bottom Right
        1.0f,   1.0f,   0.0f,   // Top Right
    };

    float rectangle_color[] = 
    {
        0.443f,   0.450f,   0.494f,
        0.443f,   0.450f,   0.494f,
        0.443f,   0.450f,   0.494f,
        0.443f,   0.450f,   0.494f,
        0.443f,   0.450f,   0.494f,
        0.443f,   0.450f,   0.494f
    };

    // Code
    
    //! Triangle
    //! -------------------------------------------------------------------------------------------------------------------------------------
    
    //? Position
    //? -------------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_position_triangle, 0, sizeof(VertexData));

    //* Step - 5
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(triangle_position);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    
    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_position_triangle.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Position Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Position Buffer Used For Triangle\n", __func__);
    
    //* Step - 7
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_position_triangle.vkBuffer, &vkMemoryRequirements);

    //* Step - 8
    VkMemoryAllocateInfo vkMemoryAllocateInfo;
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;
    
    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_position_triangle.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Position Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Position Buffer Used For Triangle\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_position_triangle.vkBuffer, vertexData_position_triangle.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Position Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Position Buffer Used For Triangle \n", __func__);

    //* Step - 11
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, vertexData_position_triangle.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Position Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Position Buffer Used For Triangle \n", __func__);

    //* Step - 12
    memcpy(data, triangle_position, sizeof(triangle_position));

    //* Step - 13
    vkUnmapMemory(vkDevice, vertexData_position_triangle.vkDeviceMemory);
    //? -------------------------------------------------------------------------------------------------------------------------------------
    
    //? Color
    //? -------------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_color_triangle, 0, sizeof(VertexData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(triangle_color);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    
    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_color_triangle.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Color Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Color Buffer Used For Triangle\n", __func__);
    
    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_position_triangle.vkBuffer, &vkMemoryRequirements);

    //* Step - 8
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;
    
    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_color_triangle.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Color Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Color Buffer Used For Triangle\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_color_triangle.vkBuffer, vertexData_color_triangle.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Color Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Color Buffer Used For Triangle \n", __func__);

    //* Step - 11
    data = NULL;
    vkResult = vkMapMemory(vkDevice, vertexData_color_triangle.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Color Buffer Used For Triangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Color Buffer Used For Triangle \n", __func__);

    //* Step - 12
    memcpy(data, triangle_color, sizeof(triangle_color));

    //* Step - 13
    vkUnmapMemory(vkDevice, vertexData_color_triangle.vkDeviceMemory);
    //? -------------------------------------------------------------------------------------------------------------------------------------
    //! -------------------------------------------------------------------------------------------------------------------------------------
    
    //! Rectangle
    //! -------------------------------------------------------------------------------------------------------------------------------------

    //? Position
    //? -------------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_position_rectangle, 0, sizeof(VertexData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(rectangle_position);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    
    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_position_rectangle.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Position Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Position Buffer Used For Rectangle\n", __func__);
    
    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_position_rectangle.vkBuffer, &vkMemoryRequirements);

    //* Step - 8
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;
    
    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_position_rectangle.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Position Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Position Buffer Used For Rectangle\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_position_rectangle.vkBuffer, vertexData_position_rectangle.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Position Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Position Buffer Used For Rectangle \n", __func__);

    //* Step - 11
    data = NULL;
    vkResult = vkMapMemory(vkDevice, vertexData_position_rectangle.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Position Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Position Buffer Used For Rectangle \n", __func__);

    //* Step - 12
    memcpy(data, rectangle_position, sizeof(rectangle_position));

    //* Step - 13
    vkUnmapMemory(vkDevice, vertexData_position_rectangle.vkDeviceMemory);
    //? -------------------------------------------------------------------------------------------------------------------------------------

    //? Color
    //? -------------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_color_rectangle, 0, sizeof(VertexData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(rectangle_color);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    
    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_color_rectangle.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Color Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Color Buffer Used For Rectangle\n", __func__);
    
    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_color_rectangle.vkBuffer, &vkMemoryRequirements);

    //* Step - 8
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;
    
    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_color_rectangle.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Color Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Color Buffer Used For Rectangle\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_color_rectangle.vkBuffer, vertexData_color_rectangle.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Color Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Color Buffer Used For Rectangle \n", __func__);

    //* Step - 11
    data = NULL;
    vkResult = vkMapMemory(vkDevice, vertexData_color_rectangle.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Color Buffer Used For Rectangle : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Color Buffer Used For Rectangle \n", __func__);

    //* Step - 12
    memcpy(data, rectangle_color, sizeof(rectangle_color));

    //* Step - 13
    vkUnmapMemory(vkDevice, vertexData_color_rectangle.vkDeviceMemory);
    //? -------------------------------------------------------------------------------------------------------------------------------------
    //! -------------------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult createUniformBuffer(void)
{
    // Function Declarations
    VkResult updateUniformBuffer(void);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(VP_UniformData);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Uniform Data\n", __func__);
    
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData.vkBuffer, &vkMemoryRequirements);

    VkMemoryAllocateInfo vkMemoryAllocateInfo;
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }     
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Uniform Data\n", __func__);

    vkResult = vkBindBufferMemory(vkDevice, uniformData.vkBuffer, uniformData.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Uniform Data\n", __func__);

    vkResult = updateUniformBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => updateUniformBuffer() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => updateUniformBuffer() Succeeded\n", __func__);


    return vkResult;
}

VkResult updateUniformBuffer(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    VP_UniformData mvp_UniformData;
    memset((void*)&mvp_UniformData, 0, sizeof(VP_UniformData));

    //! Update Matrices
    model_uniformData[0].modelMatrix = glm::mat4(1.0f);
    model_uniformData[0].modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(-1.5f, 0.0f, -6.0f));

    model_uniformData[1].modelMatrix = glm::mat4(1.0f);
    model_uniformData[1].modelMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(1.5f, 0.0f, -6.0f));

    mvp_UniformData.viewMatrix = glm::mat4(1.0f);
    
    glm::mat4 perspectiveProjectionMatrix = glm::mat4(1.0f);
    perspectiveProjectionMatrix = glm::perspective(
        glm::radians(45.0f),
        (float)winWidth / (float)winHeight,
        0.1f,
        100.0f
    );
    //! 2D Matrix with Column Major (Like OpenGL)
    perspectiveProjectionMatrix[1][1] = perspectiveProjectionMatrix[1][1] * (-1.0f);
    mvp_UniformData.projectionMatix = perspectiveProjectionMatrix;

    //! Map Uniform Buffer
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, uniformData.vkDeviceMemory, 0, sizeof(VP_UniformData), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &mvp_UniformData, sizeof(VP_UniformData));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData.vkDeviceMemory);

    return vkResult;
}

VkResult createShaders(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Vertex Shader
    //! ---------------------------------------------------------------------------------------------------------------------------
    //* Step - 6
    const char* szFileName = "Shader.vert.spv";
    FILE *fp = NULL;
    size_t size;

    fp = fopen(szFileName, "rb");
    if (fp == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Open SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Succeeded In Opening SPIR-V Shader File : %s\n", __func__, szFileName);

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    if (size == 0)
    {
        fprintf(gpFile, "%s() => Empty SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    fseek(fp, 0L, SEEK_SET);

    char* shaderData = (char*)malloc(size * sizeof(char));
    if (shaderData == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For shaderData !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    size_t retVal = fread(shaderData, size, 1, fp);
    if (retVal != 1)
    {
        fprintf(gpFile, "%s() => Failed To Read From SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Successfully Read Shader From SPIR-V Shader File : %s\n", __func__, szFileName);
    
    if (fp)
    {
        fclose(fp);
        fp = NULL;
        fprintf(gpFile, "%s() => Closed SPIR-V File : %s\n", __func__, szFileName);
    }

    //* Step - 7
    VkShaderModuleCreateInfo vkShaderModuleCreateInfo;
    memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
    vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0; //! Reserved, must be 0
    vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
    vkShaderModuleCreateInfo.codeSize = size;

    //* Step - 8
    vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_vertex_shader);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateShaderModule() Failed For Vertex Shader : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateShaderModule() Succeeded For Vertex Shader\n", __func__);

    //* Step - 9
    if (shaderData)
    {
        free(shaderData);
        shaderData = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For shaderData\n", __func__);
    }

    fprintf(gpFile, "%s() => Vertex Shader Module Successfully Created\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------

    //! Fragment Shader
    //! ---------------------------------------------------------------------------------------------------------------------------
    szFileName = "Shader.frag.spv";

    fp = fopen(szFileName, "rb");
    if (fp == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Open SPIR-V Shader File :  %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Succeeded In Opening SPIR-V Shader File : %s\n", __func__, szFileName);

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    if (size == 0)
    {
        fprintf(gpFile, "%s() => Empty SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    fseek(fp, 0L, SEEK_SET);

    shaderData = (char*)malloc(size * sizeof(char));
    if (shaderData == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For shaderData !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    retVal = fread(shaderData, size, 1, fp);
    if (retVal != 1)
    {
        fprintf(gpFile, "%s() => Failed To Read From SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Successfully Read Shader From SPIR-V Shader File : %s\n", __func__, szFileName);
    
    if (fp)
    {
        fclose(fp);
        fp = NULL;
        fprintf(gpFile, "%s() => Closed SPIR-V File : %s\n", __func__, szFileName);
    }

    //* Step - 7
    memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
    vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0; //! Reserved, must be 0
    vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
    vkShaderModuleCreateInfo.codeSize = size;

    //* Step - 8
    vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_fragment_shader);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateShaderModule() Failed For Fragment Shader : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateShaderModule() Succeeded For Fragment Shader\n", __func__);

    //* Step - 9
    if (shaderData)
    {
        free(shaderData);
        shaderData = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For shaderData\n", __func__);
    }

    fprintf(gpFile, "%s() => Fragment Shader Module Successfully Created\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult createDescriptorSetLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Initialize VkDescriptorSetLayoutBinding
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding;
    memset((void*)&vkDescriptorSetLayoutBinding, 0, sizeof(VkDescriptorSetLayoutBinding));
    vkDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding.binding = 0;   //! Mapped with layout(binding = 0) in vertex shader
    vkDescriptorSetLayoutBinding.descriptorCount = 1;
    vkDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    vkDescriptorSetLayoutBinding.pImmutableSamplers = NULL;

    //* Step - 3
    VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo;
    memset((void*)&vkDescriptorSetLayoutCreateInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
    vkDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vkDescriptorSetLayoutCreateInfo.pNext = NULL;
    vkDescriptorSetLayoutCreateInfo.flags = 0;
    vkDescriptorSetLayoutCreateInfo.bindingCount = 1;   //! An integer value where you want to bind descriptor set
    vkDescriptorSetLayoutCreateInfo.pBindings = &vkDescriptorSetLayoutBinding;

    //* Step - 4
    vkResult = vkCreateDescriptorSetLayout(vkDevice, &vkDescriptorSetLayoutCreateInfo, NULL, &vkDescriptorSetLayout);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Succeeded\n", __func__);

    return vkResult;
}

VkResult createPipelineLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    VkPushConstantRange vkPushConstantRange;
    memset((void*)&vkPushConstantRange, 0, sizeof(VkPushConstantRange));
    vkPushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    vkPushConstantRange.offset = 0;
    vkPushConstantRange.size = sizeof(glm::mat4);

    //* Step - 3
    VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo;
    memset((void*)&vkPipelineLayoutCreateInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
    vkPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkPipelineLayoutCreateInfo.pNext = NULL;
    vkPipelineLayoutCreateInfo.flags = 0;
    vkPipelineLayoutCreateInfo.setLayoutCount = 1;
    vkPipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout;
    vkPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    vkPipelineLayoutCreateInfo.pPushConstantRanges = &vkPushConstantRange;

    //* Step - 4
    vkResult = vkCreatePipelineLayout(vkDevice, &vkPipelineLayoutCreateInfo, NULL, &vkPipelineLayout);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Succeeded\n", __func__);

    return vkResult;
}

VkResult createDescriptorPool(void)
{
    // Variable Declarations
    VkResult vkResult;

    // Code

    //* Vulkan expects decriptor pool size before creating actual descriptor pool
    VkDescriptorPoolSize vkDescriptorPoolSize;
    memset((void*)&vkDescriptorPoolSize, 0, sizeof(VkDescriptorPoolSize));
    vkDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorPoolSize.descriptorCount = 1;
   
    //* Create the pool
    VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
    memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkDescriptorPoolCreateInfo.pNext = NULL;
    vkDescriptorPoolCreateInfo.flags = 0;
    vkDescriptorPoolCreateInfo.poolSizeCount = 1;
    vkDescriptorPoolCreateInfo.pPoolSizes = &vkDescriptorPoolSize;
    vkDescriptorPoolCreateInfo.maxSets = 1;

    vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Failed : %d !!!\n", __func__, vkResult);  
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Succeeded\n", __func__);

    return vkResult;
}

VkResult createDescriptorSet(void)
{
    // Variable Declarations
    VkResult vkResult;

    // Code

    //* Initialize DescriptorSetAllocationInfo
    VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo;
    memset((void*)&vkDescriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
    vkDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vkDescriptorSetAllocateInfo.pNext = NULL;
    vkDescriptorSetAllocateInfo.descriptorPool = vkDescriptorPool;
    vkDescriptorSetAllocateInfo.descriptorSetCount = 1;
    vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout;

    vkResult = vkAllocateDescriptorSets(vkDevice, &vkDescriptorSetAllocateInfo, &vkDescriptorSet);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }  
    else
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Succeeded\n", __func__);
    
    //* Describe whether we want buffer as uniform or image as uniform
    VkDescriptorBufferInfo vkDescriptorBufferInfo;
    memset((void*)&vkDescriptorBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
    vkDescriptorBufferInfo.buffer = uniformData.vkBuffer;
    vkDescriptorBufferInfo.offset = 0;
    vkDescriptorBufferInfo.range = sizeof(VP_UniformData);

    /* Update above descriptor set directly to the shader
    There are 2 ways :-
        1) Writing directly to the shader
        2) Copying from one shader to another shader
    */
    VkWriteDescriptorSet vkWriteDescriptorSet;
    memset((void*)&vkWriteDescriptorSet, 0, sizeof(VkWriteDescriptorSet));
    vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet.pNext = NULL;
    vkWriteDescriptorSet.dstSet = vkDescriptorSet;
    vkWriteDescriptorSet.dstArrayElement = 0;
    vkWriteDescriptorSet.descriptorCount = 1;
    vkWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkWriteDescriptorSet.pBufferInfo = &vkDescriptorBufferInfo;
    vkWriteDescriptorSet.pImageInfo = NULL;
    vkWriteDescriptorSet.pTexelBufferView = NULL;
    vkWriteDescriptorSet.dstBinding = 0;

    vkUpdateDescriptorSets(vkDevice, 1, &vkWriteDescriptorSet, 0, NULL);

    return vkResult;
}

VkResult createRenderPass(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Step - 1

    //! Color Attachment (Graphics Pipeline)
    VkAttachmentDescription vkAttachmentDescription_array[1];
    memset((void*)vkAttachmentDescription_array, 0, sizeof(VkAttachmentDescription) * _ARRAYSIZE(vkAttachmentDescription_array));
    vkAttachmentDescription_array[0].flags = 0;
    vkAttachmentDescription_array[0].format = vkFormat_color;
    vkAttachmentDescription_array[0].samples = VK_SAMPLE_COUNT_1_BIT; //* No MSAA
    vkAttachmentDescription_array[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    vkAttachmentDescription_array[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    vkAttachmentDescription_array[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    vkAttachmentDescription_array[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    vkAttachmentDescription_array[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkAttachmentDescription_array[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    //* Step - 2
    VkAttachmentReference vkAttachmentReference;
    memset((void*)&vkAttachmentReference, 0, sizeof(VkAttachmentReference));
    vkAttachmentReference.attachment = 0;   //* 0 specifies 0th index in above array
    vkAttachmentReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    //* Step - 3
    VkSubpassDescription vkSubpassDescription;
    memset((void*)&vkSubpassDescription, 0, sizeof(VkSubpassDescription));
    vkSubpassDescription.flags = 0;
    vkSubpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    vkSubpassDescription.inputAttachmentCount = 0;
    vkSubpassDescription.pInputAttachments = NULL;
    vkSubpassDescription.colorAttachmentCount = _ARRAYSIZE(vkAttachmentDescription_array);
    vkSubpassDescription.pColorAttachments = &vkAttachmentReference;
    vkSubpassDescription.pDepthStencilAttachment = NULL;
    vkSubpassDescription.pPreserveAttachments = NULL;
    vkSubpassDescription.pResolveAttachments = NULL;

    //* Step - 4
    VkRenderPassCreateInfo vkRenderPassCreateInfo;
    memset((void*)&vkRenderPassCreateInfo, 0, sizeof(VkRenderPassCreateInfo));
    vkRenderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    vkRenderPassCreateInfo.pNext = NULL;
    vkRenderPassCreateInfo.flags = 0;
    vkRenderPassCreateInfo.attachmentCount = _ARRAYSIZE(vkAttachmentDescription_array);
    vkRenderPassCreateInfo.pAttachments = vkAttachmentDescription_array;
    vkRenderPassCreateInfo.subpassCount = 1;
    vkRenderPassCreateInfo.pSubpasses = &vkSubpassDescription;
    vkRenderPassCreateInfo.dependencyCount = 0;
    vkRenderPassCreateInfo.pDependencies = NULL;

    //* Step - 5
    vkResult = vkCreateRenderPass(vkDevice, &vkRenderPassCreateInfo, NULL, &vkRenderPass);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateRenderPass() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateRenderPass() Succeeded\n", __func__);

    return vkResult;
}

VkResult createPipeline(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Code

    //! Vertex Input State
    VkVertexInputBindingDescription vkVertexInputBindingDescription_array[2];
    memset((void*)vkVertexInputBindingDescription_array, 0, sizeof(VkVertexInputBindingDescription) * _ARRAYSIZE(vkVertexInputBindingDescription_array));
    
    //! Position
    vkVertexInputBindingDescription_array[0].binding = 0;   //! Corresponds to layout(location = 0) in Vertex Shader
    vkVertexInputBindingDescription_array[0].stride = sizeof(float) * 3; 
    vkVertexInputBindingDescription_array[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    //! Color
    vkVertexInputBindingDescription_array[1].binding = 1; //! Corresponds to layout(location = 1) in Vertex Shader
    vkVertexInputBindingDescription_array[1].stride = sizeof(float) * 3; 
    vkVertexInputBindingDescription_array[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[2];
    memset((void*)vkVertexInputAttributeDescription_array, 0, sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));
    
    //! Position
    vkVertexInputAttributeDescription_array[0].binding = 0;
    vkVertexInputAttributeDescription_array[0].location = 0;
    vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[0].offset = 0;
    
    //! Color
    vkVertexInputAttributeDescription_array[1].binding = 1;
    vkVertexInputAttributeDescription_array[1].location = 1;
    vkVertexInputAttributeDescription_array[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[1].offset = 0;

    VkPipelineVertexInputStateCreateInfo vkPipelineVertexInputStateCreateInfo;
    memset((void*)&vkPipelineVertexInputStateCreateInfo, 0, sizeof(VkPipelineVertexInputStateCreateInfo));
    vkPipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vkPipelineVertexInputStateCreateInfo.pNext = NULL;
    vkPipelineVertexInputStateCreateInfo.flags = 0;
    vkPipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = _ARRAYSIZE(vkVertexInputBindingDescription_array);
    vkPipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = vkVertexInputBindingDescription_array;
    vkPipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = _ARRAYSIZE(vkVertexInputAttributeDescription_array);
    vkPipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = vkVertexInputAttributeDescription_array;
    
    //! Input Assembly State
    VkPipelineInputAssemblyStateCreateInfo vkPipelineInputAssemblyStateCreateInfo;
    memset((void*)&vkPipelineInputAssemblyStateCreateInfo, 0, sizeof(VkPipelineInputAssemblyStateCreateInfo));
    vkPipelineInputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    vkPipelineInputAssemblyStateCreateInfo.pNext = NULL;
    vkPipelineInputAssemblyStateCreateInfo.flags = 0;
    vkPipelineInputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    //! Rasterization State
    VkPipelineRasterizationStateCreateInfo vkPipelineRasterizationStateCreateInfo;
    memset((void*)&vkPipelineRasterizationStateCreateInfo, 0, sizeof(VkPipelineRasterizationStateCreateInfo));
    vkPipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    vkPipelineRasterizationStateCreateInfo.pNext = NULL;
    vkPipelineRasterizationStateCreateInfo.flags = 0;
    vkPipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    vkPipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
    vkPipelineRasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    vkPipelineRasterizationStateCreateInfo.lineWidth = 1.0f;

    //! Color Blend State
    VkPipelineColorBlendAttachmentState vkPipelineColorBlendAttachmentState_array[1];
    memset((void*)vkPipelineColorBlendAttachmentState_array, 0, sizeof(VkPipelineColorBlendAttachmentState) * _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array));
    vkPipelineColorBlendAttachmentState_array[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    vkPipelineColorBlendAttachmentState_array[0].blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo vkPipelineColorBlendStateCreateInfo;
    memset((void*)&vkPipelineColorBlendStateCreateInfo, 0, sizeof(VkPipelineColorBlendStateCreateInfo));
    vkPipelineColorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    vkPipelineColorBlendStateCreateInfo.pNext = NULL;
    vkPipelineColorBlendStateCreateInfo.flags = 0;
    vkPipelineColorBlendStateCreateInfo.attachmentCount = _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array);
    vkPipelineColorBlendStateCreateInfo.pAttachments = vkPipelineColorBlendAttachmentState_array;

    //! Viewport Scissor State
    VkPipelineViewportStateCreateInfo vkPipelineViewportStateCreateInfo;
    memset((void*)&vkPipelineViewportStateCreateInfo, 0, sizeof(VkPipelineViewportStateCreateInfo));
    vkPipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vkPipelineViewportStateCreateInfo.pNext = NULL;
    vkPipelineViewportStateCreateInfo.flags = 0;
    vkPipelineViewportStateCreateInfo.viewportCount = 1;    //* We can specify multiple viewports here
    vkPipelineViewportStateCreateInfo.scissorCount = 1;

    //! Viewport Info     
    memset((void*)&vkViewport, 0, sizeof(VkViewport));
    vkViewport.x = 0;
    vkViewport.y = 0;
    vkViewport.width = (float)vkExtent2D_swapchain.width;
    vkViewport.height = (float)vkExtent2D_swapchain.height;
    vkViewport.minDepth = 0.0f;
    vkViewport.maxDepth = 1.0f;

    vkPipelineViewportStateCreateInfo.pViewports = &vkViewport;
    
    //! Scissor Info
    memset((void*)&vkRect2D_scissor, 0, sizeof(VkRect2D));
    vkRect2D_scissor.offset.x = 0;
    vkRect2D_scissor.offset.y = 0;
    vkRect2D_scissor.extent.width = vkExtent2D_swapchain.width;
    vkRect2D_scissor.extent.height = vkExtent2D_swapchain.height;
   
    vkPipelineViewportStateCreateInfo.pScissors = &vkRect2D_scissor;

    //! Depth Stencil State !//

    //! Dynamic State !//

    //! Multi-Sample State
    VkPipelineMultisampleStateCreateInfo vkPipelineMultisampleStateCreateInfo;
    memset((void*)&vkPipelineMultisampleStateCreateInfo, 0, sizeof(VkPipelineMultisampleStateCreateInfo));
    vkPipelineMultisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    vkPipelineMultisampleStateCreateInfo.pNext = NULL;
    vkPipelineMultisampleStateCreateInfo.flags = 0;
    vkPipelineMultisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    //! Shader Stage State
    VkPipelineShaderStageCreateInfo vkPipelineShaderStageCreateInfo_array[2];
    memset((void*)vkPipelineShaderStageCreateInfo_array, 0, sizeof(VkPipelineShaderStageCreateInfo) * _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array));
    
    //* Vertex Shader
    vkPipelineShaderStageCreateInfo_array[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[0].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[0].flags = 0;
    vkPipelineShaderStageCreateInfo_array[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    vkPipelineShaderStageCreateInfo_array[0].module = vkShaderModule_vertex_shader;
    vkPipelineShaderStageCreateInfo_array[0].pName = "main";
    vkPipelineShaderStageCreateInfo_array[0].pSpecializationInfo = NULL;

    //* Fragment Shader
    vkPipelineShaderStageCreateInfo_array[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[1].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[1].flags = 0;
    vkPipelineShaderStageCreateInfo_array[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkPipelineShaderStageCreateInfo_array[1].module = vkShaderModule_fragment_shader;
    vkPipelineShaderStageCreateInfo_array[1].pName = "main";
    vkPipelineShaderStageCreateInfo_array[1].pSpecializationInfo = NULL;

    //! Tessellation State !//

    //! As pipelines are created from pipeline caches, we will create VkPipelineCache Object
    VkPipelineCacheCreateInfo vkPipelineCacheCreateInfo;
    memset((void*)&vkPipelineCacheCreateInfo, 0, sizeof(VkPipelineCacheCreateInfo));
    vkPipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkPipelineCacheCreateInfo.pNext = NULL;
    vkPipelineCacheCreateInfo.flags = 0;
    
    VkPipelineCache vkPipelineCache = VK_NULL_HANDLE;
    vkResult = vkCreatePipelineCache(vkDevice, &vkPipelineCacheCreateInfo, NULL, &vkPipelineCache);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Succeeded\n", __func__);

    //! Create actual Graphics Pipeline
    VkGraphicsPipelineCreateInfo vkGraphicsPipelineCreateInfo;
    memset((void*)&vkGraphicsPipelineCreateInfo, 0, sizeof(VkGraphicsPipelineCreateInfo));
    vkGraphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    vkGraphicsPipelineCreateInfo.pNext = NULL;
    vkGraphicsPipelineCreateInfo.flags = 0;
    vkGraphicsPipelineCreateInfo.pVertexInputState = &vkPipelineVertexInputStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pInputAssemblyState = &vkPipelineInputAssemblyStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pRasterizationState = &vkPipelineRasterizationStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pColorBlendState = &vkPipelineColorBlendStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pViewportState = &vkPipelineViewportStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pDepthStencilState = NULL;
    vkGraphicsPipelineCreateInfo.pDynamicState = NULL;
    vkGraphicsPipelineCreateInfo.pMultisampleState = &vkPipelineMultisampleStateCreateInfo;
    vkGraphicsPipelineCreateInfo.stageCount = _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array);
    vkGraphicsPipelineCreateInfo.pStages = vkPipelineShaderStageCreateInfo_array;
    vkGraphicsPipelineCreateInfo.pTessellationState = NULL;
    vkGraphicsPipelineCreateInfo.layout = vkPipelineLayout;
    vkGraphicsPipelineCreateInfo.renderPass = vkRenderPass;
    vkGraphicsPipelineCreateInfo.subpass = 0;
    vkGraphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    vkGraphicsPipelineCreateInfo.basePipelineIndex = 0;

    vkResult = vkCreateGraphicsPipelines(vkDevice, vkPipelineCache, 1, &vkGraphicsPipelineCreateInfo, NULL, &vkPipeline);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateGraphicsPipelines() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateGraphicsPipelines() Succeeded\n", __func__);

    //* Destroy Pipeline Cache
    if (vkPipelineCache)
    {
        vkDestroyPipelineCache(vkDevice, vkPipelineCache, NULL);
        vkPipelineCache = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyPipelineCache() Succeeded\n", __func__);
    }

    return vkResult;
}

VkResult createFramebuffers(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Step - 1
    VkImageView vkImageView_attachments_array[1];
    memset((void*)vkImageView_attachments_array, 0, sizeof(VkImageView) * _ARRAYSIZE(vkImageView_attachments_array));

    //* Step - 2
    VkFramebufferCreateInfo vkFramebufferCreateInfo;
    memset((void*)&vkFramebufferCreateInfo, 0, sizeof(VkFramebufferCreateInfo));
    vkFramebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    vkFramebufferCreateInfo.flags = 0;
    vkFramebufferCreateInfo.pNext = NULL;
    vkFramebufferCreateInfo.attachmentCount = _ARRAYSIZE(vkImageView_attachments_array);
    vkFramebufferCreateInfo.pAttachments = vkImageView_attachments_array;
    vkFramebufferCreateInfo.renderPass = vkRenderPass;
    vkFramebufferCreateInfo.width = vkExtent2D_swapchain.width;
    vkFramebufferCreateInfo.height = vkExtent2D_swapchain.height;
    vkFramebufferCreateInfo.layers = 1;

    //* Step - 3
    vkFramebuffer_array = (VkFramebuffer*)malloc(sizeof(VkFramebuffer) * swapchainImageCount);
    if (vkFramebuffer_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkFramebuffer_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 4
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        vkImageView_attachments_array[0] = swapchainImageView_array[i];

        vkResult = vkCreateFramebuffer(vkDevice, &vkFramebufferCreateInfo, NULL, &vkFramebuffer_array[i]);
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => vkCreateFramebuffer() Failed For Index : %d, Reason : %d !!!\n", __func__, i, vkResult);
            vkResult = VK_ERROR_INITIALIZATION_FAILED;
            return vkResult;
        }
    }

    return vkResult;
}

VkResult createSemaphores(void)
{
    // Code
    VkResult vkResult = VK_SUCCESS;

    //* Step - 2
    VkSemaphoreCreateInfo vkSemaphoreCreateInfo;
    memset((void*)&vkSemaphoreCreateInfo, 0, sizeof(VkSemaphoreCreateInfo));
    vkSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    vkSemaphoreCreateInfo.flags = 0;    //! Must Be 0 (Reserved)
    vkSemaphoreCreateInfo.pNext = NULL;

    //* Step - 3
    vkResult = vkCreateSemaphore(vkDevice, &vkSemaphoreCreateInfo, NULL, &vkSemaphore_backBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateSemaphore() Failed For vkSemaphore_backBuffer : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    vkResult = vkCreateSemaphore(vkDevice, &vkSemaphoreCreateInfo, NULL, &vkSemaphore_renderComplete);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateSemaphore() Failed For vkSemaphore_renderComplete : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    return vkResult;
}

VkResult createFences(void)
{
    // Code
    VkResult vkResult = VK_SUCCESS;

    //* Step - 4
    VkFenceCreateInfo vkFenceCreateInfo;
    memset((void*)&vkFenceCreateInfo, 0, sizeof(VkFenceCreateInfo));
    vkFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    vkFenceCreateInfo.pNext = NULL;
    vkFenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    //* Step - 5
    vkFence_array = (VkFence*)malloc(sizeof(VkFence) * swapchainImageCount);
    if (vkFence_array == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For vkFence_array !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    //* Step - 6
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        vkResult = vkCreateFence(vkDevice, &vkFenceCreateInfo, NULL, &vkFence_array[i]);
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => vkCreateFence() Failed For Index : %d, Reason : %d\n", __func__, i, vkResult);
            vkResult = VK_ERROR_INITIALIZATION_FAILED;
            return vkResult;
        }
        else
            fprintf(gpFile, "%s() => vkCreateFence() Succeeded For Index : %d\n", __func__, i);
    }

    return vkResult;
}

VkResult buildCommandBuffers(void)
{
    // Code
    VkResult vkResult = VK_SUCCESS;

    //! Loop per swapchain image
    for (uint32_t i = 0; i < swapchainImageCount; i++)
    {
        //* Step - 1 => Reset Command Buffer
        vkResult = vkResetCommandBuffer(vkCommandBuffer_array[i], 0);   //! 0 specifies not to release the resources
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => vkResetCommandBuffer() Failed For Index : %d, Reason : %d\n", __func__, i, vkResult);
            vkResult = VK_ERROR_INITIALIZATION_FAILED;
            return vkResult;
        }
        else
            fprintf(gpFile, "%s() => vkResetCommandBuffer() Succeeded For Index : %d\n", __func__, i);

        //* Step - 2
        VkCommandBufferBeginInfo vkCommandBufferBeginInfo;
        memset((void*)&vkCommandBufferBeginInfo, 0, sizeof(VkCommandBufferBeginInfo));
        vkCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkCommandBufferBeginInfo.pNext = NULL;
        vkCommandBufferBeginInfo.flags = 0;     //! 0 specifies that we will use only the primary command buffer, and not going to use this command buffer simultaneously between multiple threads

        //* Step - 3
        vkResult = vkBeginCommandBuffer(vkCommandBuffer_array[i], &vkCommandBufferBeginInfo);
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For Index : %d, Reason : %d\n", __func__, i, vkResult);
            vkResult = VK_ERROR_INITIALIZATION_FAILED;
            return vkResult;
        }
        else
            fprintf(gpFile, "%s() => vkBeginCommandBuffer() Succeeded For Index : %d\n", __func__, i);

        //* Step - 4 => Set Clear Value
        VkClearValue vkClearValue_array[1];
        memset((void*)vkClearValue_array, 0, sizeof(VkClearValue) * _ARRAYSIZE(vkClearValue_array));
        vkClearValue_array[0].color = vkClearColorValue;

        //* Step - 5
        VkRenderPassBeginInfo vkRenderPassBeginInfo;
        memset((void*)&vkRenderPassBeginInfo, 0, sizeof(VkRenderPassBeginInfo));
        vkRenderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        vkRenderPassBeginInfo.pNext = NULL;
        vkRenderPassBeginInfo.renderPass = vkRenderPass;
        vkRenderPassBeginInfo.renderArea.offset.x = 0;
        vkRenderPassBeginInfo.renderArea.offset.y = 0;
        vkRenderPassBeginInfo.renderArea.extent.width = vkExtent2D_swapchain.width;
        vkRenderPassBeginInfo.renderArea.extent.height = vkExtent2D_swapchain.height;
        vkRenderPassBeginInfo.clearValueCount = _ARRAYSIZE(vkClearValue_array);
        vkRenderPassBeginInfo.pClearValues = vkClearValue_array;
        vkRenderPassBeginInfo.framebuffer = vkFramebuffer_array[i];
        
        //* Step - 6
        vkCmdBeginRenderPass(vkCommandBuffer_array[i], &vkRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
        {
            //! Bind with Pipeline
            vkCmdBindPipeline(vkCommandBuffer_array[i], VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline);

            //! Bind the Descriptor Set to the Pipeline
            vkCmdBindDescriptorSets(
                vkCommandBuffer_array[i],
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                vkPipelineLayout,
                0,
                1,
                &vkDescriptorSet,
                0,
                NULL
            );

            vkCmdPushConstants(
                vkCommandBuffer_array[i],
                vkPipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(glm::mat4),
                &model_uniformData[0]
            );

            //! Bind with Vertex Position Buffer
            VkDeviceSize vkDeviceSize_offset_position[1];
            memset((void*)vkDeviceSize_offset_position, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_position));
            vkCmdBindVertexBuffers(
                vkCommandBuffer_array[i], 
                0, 
                1, 
                &vertexData_position_triangle.vkBuffer, 
                vkDeviceSize_offset_position
            );

            //! Bind with Vertex Color Buffer
            VkDeviceSize vkDeviceSize_offset_color[1];
            memset((void*)vkDeviceSize_offset_color, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_color));
            vkCmdBindVertexBuffers(
                vkCommandBuffer_array[i], 
                1, 
                1, 
                &vertexData_color_triangle.vkBuffer, 
                vkDeviceSize_offset_color
            );

            //! Vulkan Drawing Function
            vkCmdDraw(vkCommandBuffer_array[i], 3, 1, 0, 0);

            vkCmdPushConstants(
                vkCommandBuffer_array[i],
                vkPipelineLayout,
                VK_SHADER_STAGE_VERTEX_BIT,
                0,
                sizeof(glm::mat4),
                &model_uniformData[1]
            );

            //! Bind with Vertex Position Buffer
            memset((void*)vkDeviceSize_offset_position, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_position));
            vkCmdBindVertexBuffers(
                vkCommandBuffer_array[i], 
                0, 
                1, 
                &vertexData_position_rectangle.vkBuffer, 
                vkDeviceSize_offset_position
            );

            //! Bind with Vertex Color Buffer
            memset((void*)vkDeviceSize_offset_color, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_color));
            vkCmdBindVertexBuffers(
                vkCommandBuffer_array[i], 
                1, 
                1, 
                &vertexData_color_rectangle.vkBuffer, 
                vkDeviceSize_offset_color
            );

            //! Vulkan Drawing Function
            vkCmdDraw(vkCommandBuffer_array[i], 6, 1, 0, 0);
        }
        //* Step - 7
        vkCmdEndRenderPass(vkCommandBuffer_array[i]);

        //* Step - 8
        vkResult = vkEndCommandBuffer(vkCommandBuffer_array[i]);
        if (vkResult != VK_SUCCESS)
        {
            fprintf(gpFile, "%s() => vkEndCommandBuffer() Failed For Index : %d, Reason : %d\n", __func__, i, vkResult);
            vkResult = VK_ERROR_INITIALIZATION_FAILED;
            return vkResult;
        }
        else
            fprintf(gpFile, "%s() => vkEndCommandBuffer() Succeeded For Index : %d\n", __func__, i);
    }

    return vkResult;
}


VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallback(
    VkDebugReportFlagsEXT vkDebugReportFlagsEXT,
    VkDebugReportObjectTypeEXT vkDebugReportObjectTypeEXT,
    uint64_t object,
    size_t location,
    int32_t messageCode,
    const char* pLayerPrefix,
    const char* pMessage,
    void* pUserData
)
{
    // Code
    fprintf(gpFile, "ADN_VALIDATION : debugReportCallback() => %s(%d) = %s\n", pLayerPrefix, messageCode, pMessage);
    return VK_FALSE;
}


