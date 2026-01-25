#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#include "Vk.h"

//! Vulkan Related Header Files
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

//! OpenCL Headers
#define CL_TARGET_OPENCL_VERSION    300
#include <CL/opencl.h>

//* CL_MEM_DEVICE_HANDLE_LIST_KHR
//* CL_MEM_DEVICE_HANDLE_LIST_END_KHR
#include <CL/cl_ext.h>

//! ImGui Related
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "imgui_impl_win32.h"

//! Vulkan Related Libraries
#pragma comment(lib, "vulkan-1.lib")

//! OpenCL Library
#pragma comment(lib, "OpenCL.lib")

#define WIN_WIDTH   800
#define WIN_HEIGHT  600

// Global Function Declarations
LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// Global Variable Declarations
HWND ghwnd = NULL;
BOOL gbFullScreen = FALSE;
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

//! Device Selection
struct PhysicalDeviceInfo
{
    VkPhysicalDevice vkPhysicalDevice;
    VkPhysicalDeviceProperties vkPhysicalDeviceProperties;
};

PhysicalDeviceInfo *physicalDeviceInfo_array = NULL;
int selectedDevice = -1;
BOOL bShowPopup = TRUE;
BOOL bReinitialize = FALSE;

//* OpenCL Related Variables
cl_int oclResult;
cl_platform_id oclPlatformId;
cl_device_id oclDeviceId;

//! ImGui Related
ImFont* font;

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
        TEXT("Atharv Natu : Vulkan Device Selection"),
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
            if (gbActiveWindow)
            {
                //* Render the scene
                display();

                //* Update the scene
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
            resize(LOWORD(wParam), HIWORD(wParam));
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
    VkResult createRenderPass(void);
    VkResult createFramebuffers(void);
    VkResult createSemaphores(void);
    VkResult createFences(void);
    VkResult buildCommandBuffers(void);
    void initializeImGui(const char* fontFile, float fontSize);

    //! OpenCL Related Function Declarations
    cl_int initializeOpenCL(void);

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

    //! Initialize OpenCL
    // oclResult = initializeOpenCL();
    // if (oclResult != CL_SUCCESS)
    // {
    //     fprintf(gpFile, "%s() => initializeOpenCL() Failed : %d !!!\n", __func__, vkResult);
    //     vkResult = VK_ERROR_INITIALIZATION_FAILED;
    //     return vkResult;
    // } 
    // else
    //     fprintf(gpFile, "%s() => initializeOpenCL() Succeeded\n", __func__);

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

    //! Initialize ImGui
    initializeImGui("../ImGui/Poppins-Regular.ttf", 24.0f);

    // vkResult = buildCommandBuffers();
    // if (vkResult != VK_SUCCESS)
    // {
    //     fprintf(gpFile, "%s() => buildCommandBuffers() Failed\n", __func__);
    //     vkResult = VK_ERROR_INITIALIZATION_FAILED;
    //     return vkResult;
    // }
    // else
    //     fprintf(gpFile, "%s() => buildCommandBuffers() Succeeded\n", __func__);

    //! Initialization Completed
    bInitialized = TRUE;
    
    return vkResult;
}

cl_int initializeOpenCL(void)
{
    // Function Declarations
    BOOL openCLPlatformSupportsRequiredExtensions(cl_platform_id);

    // Variable Declarations
    cl_uint platformCount;
    cl_platform_id* oclPlatformIDs = NULL;
    cl_device_id* oclDeviceIDs = NULL;
    cl_uint deviceCount;
    
    //* Code

    // Step - 1 : Get Number of OpenCL Supported Platforms
    oclResult = clGetPlatformIDs(0, NULL, &platformCount);
    if (oclResult != CL_SUCCESS)
    {
        fprintf(gpFile, "%s() => OpenCL Error : Call 1 : clGetPlatformIDs() Failed : %d !!!\n", __func__, oclResult);
        return oclResult;
    }
    else if (platformCount == 0)
    {
        fprintf(gpFile, "%s() => clGetPlatformIDs() Returned 0 OpenCL Supported Plaforms !!!\n", __func__);
        return oclResult;
    }
    
    //* We have 1 or more platforms, fetch them into an array
    oclPlatformIDs = (cl_platform_id*)malloc(platformCount * sizeof(cl_platform_id));
    if (oclPlatformIDs == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Allocate Memory To oclPlatformIds !!!\n", __func__);
        return oclResult;
    }

    oclResult = clGetPlatformIDs(platformCount, oclPlatformIDs, NULL);
    if (oclResult != CL_SUCCESS)
    {
        fprintf(gpFile, "%s() => OpenCL Error : Call 2 : clGetPlatformIDs() Failed : %d !!!\n", __func__, oclResult);
        return oclResult;
    }

    //* Iterate over array to check for device
    int interopPlatformFound = -1;
    for (cl_uint i = 0; i < platformCount; i++)
    {
        //* Get number of devices for platform
        oclResult = clGetDeviceIDs(oclPlatformIDs[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
        if (oclResult != CL_SUCCESS)
            continue;
        else if (deviceCount == 0)
            continue;

        //* Check whether the GPU found at ith platform supports required extensions
        //*     1) cl_khr_device_uuid
        //*     2) cl_khr_external_memory
        if (openCLPlatformSupportsRequiredExtensions(oclPlatformIDs[i]) == FALSE)
            continue;

        //* Required Platform Found
        //* --------------------------------------------------------------------------------------------------------
        oclPlatformId = oclPlatformIDs[i];

        //* Print properties of selected platform
        size_t infoSize;
        char* oclPlatformInfo = NULL;
        clGetPlatformInfo(oclPlatformId, CL_PLATFORM_NAME, 0, NULL, &infoSize);

        oclPlatformInfo = (char*)malloc(infoSize * sizeof(char));
        if (oclPlatformInfo == NULL)
        {
            fprintf(gpFile, "%s() => Failed To Allocate Memory To oclPlatformInfo !!!\n", __func__);
            return oclResult;
        }

        clGetPlatformInfo(oclPlatformId, CL_PLATFORM_NAME, infoSize, oclPlatformInfo, NULL);
        fprintf(gpFile, "Selected OpenCL Platform : %s\n", oclPlatformInfo);

        free(oclPlatformInfo);
        oclPlatformInfo = NULL;

        interopPlatformFound = 1;
        break;
        //* --------------------------------------------------------------------------------------------------------
    } 
    
    free(oclPlatformIDs);
    oclPlatformIDs = NULL;

    if (interopPlatformFound == -1)
    {
        fprintf(gpFile, "%s() => No OpenCL Supported Platform with GPU Found !!!\n", __func__);
        return -32; //* Value for invalid OpenCL platform
    }

    //* Allocate memory for 1 or more GPU devices in found supported platform
    oclDeviceIDs = (cl_device_id*)malloc(deviceCount * sizeof(cl_device_id));
    if (oclDeviceIDs == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Allocate Memory To oclDeviceIDs !!!\n", __func__);
        return oclResult;
    }

    //* Get IDs Into Allocated Buffer
    oclResult = clGetDeviceIDs(oclPlatformId, CL_DEVICE_TYPE_GPU, deviceCount, oclDeviceIDs, NULL);
    if (oclResult != CL_SUCCESS)
    {
        fprintf(gpFile, "%s() => OpenCL Error : clGetDeviceIDs() Failed : %d !!!\n", __func__, oclResult);
        return oclResult;
    }

    //* Check UUID between Vulkan and OpenCL
    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties;
    memset((void*)&vkPhysicalDeviceIDProperties, 0, sizeof(VkPhysicalDeviceIDProperties));
    vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2;
    memset((void*)&vkPhysicalDeviceProperties2, 0, sizeof(VkPhysicalDeviceProperties2));
    vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    vkGetPhysicalDeviceProperties2(vkPhysicalDevice_selected, &vkPhysicalDeviceProperties2);

    cl_uchar cl_uuid[CL_UUID_SIZE_KHR];
    int interopDeviceFound = -1;

    //* Iterate Over All IDs and Check For Vulkan Matching UUID
    for (cl_uint i = 0; i < deviceCount; i++)
    {
        //* Get OpenCL Capable Device's UUID
        oclResult = clGetDeviceInfo(oclDeviceIDs[i], CL_DEVICE_UUID_KHR, CL_UUID_SIZE_KHR, &cl_uuid, NULL);
        if (oclResult != CL_SUCCESS)
        {
            fprintf(gpFile, "%s() => OpenCL Error : clGetDeviceInfo() Failed For Index : %d, Reason : %d !!!\n", __func__, i, oclResult);
            return oclResult;
        }

        //* Compare OpenCL Device UUID with Vulkan Device UUID
        BOOL uuidMatch = TRUE;
        for (uint32_t j = 0; j < CL_UUID_SIZE_KHR; j++)
        {
            if (cl_uuid[j] != vkPhysicalDeviceIDProperties.deviceUUID[j])
            {
                uuidMatch = FALSE;
                break;
            }
        }

        if (uuidMatch == FALSE)
            continue;
        
        //* Required Device Found
        //* --------------------------------------------------------------------------------------------------------
        oclDeviceId = oclDeviceIDs[i];

        //* Print properties of selected device
        size_t infoSize;
        char* oclDeviceInfo = NULL;

        clGetDeviceInfo(oclDeviceId, CL_DEVICE_NAME, 0, NULL, &infoSize);

        oclDeviceInfo = (char*)malloc(infoSize * sizeof(char));
        if (oclDeviceInfo == NULL)
        {
            fprintf(gpFile, "%s() => Failed To Allocate Memory To oclDeviceInfo !!!\n", __func__);
            return oclResult;
        }

        clGetDeviceInfo(oclDeviceId, CL_DEVICE_NAME, infoSize, oclDeviceInfo, NULL);
        fprintf(gpFile, "Selected OpenCL Device : %s\n", oclDeviceInfo);

        free(oclDeviceInfo);
        oclDeviceInfo = NULL;

        interopDeviceFound = 1;
        break;
        //* --------------------------------------------------------------------------------------------------------
    }

    free(oclDeviceIDs);
    oclDeviceIDs = NULL;

    if (interopDeviceFound == -1)
    {
        fprintf(gpFile, "%s() => No OpenCL Supported Device Found !!!\n", __func__);
        return -2; //* Value for unavailable device
    }

    return CL_SUCCESS;
}

BOOL openCLPlatformSupportsRequiredExtensions(cl_platform_id ocl_platform_id)
{
    // Variable Declarations
    size_t extensionSize;
    char* oclPlatformExtensions = NULL;

    // Code

    //* List Current Platform's Extensions
    clGetPlatformInfo(ocl_platform_id, CL_PLATFORM_EXTENSIONS, 0, NULL, &extensionSize);

    oclPlatformExtensions = (char*)malloc(extensionSize * sizeof(char));
    if (oclPlatformExtensions == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Allocate Memory To oclPlatformExtensions !!!\n", __func__);
        return FALSE;
    }

    clGetPlatformInfo(ocl_platform_id, CL_PLATFORM_EXTENSIONS, extensionSize, oclPlatformExtensions, NULL);

    char* oclPlatformExtensions_copy_for_strtok = NULL;
    oclPlatformExtensions_copy_for_strtok = (char*)malloc(extensionSize * sizeof(char));
    if (oclPlatformExtensions_copy_for_strtok == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Allocate Memory To oclPlatformExtensions_copy_for_strtok !!!\n", __func__);
        return FALSE;
    }

    strcpy(oclPlatformExtensions_copy_for_strtok, oclPlatformExtensions);

    //* Check No. Of Extensions Found
    char* token = strtok(oclPlatformExtensions_copy_for_strtok, " ");
    int i = 0;
    while (token != NULL)
    {
        i++;
        token = strtok(NULL, " ");
    }
    int extensionCount = i;
    fprintf(gpFile, "\nNo. Of OpenCL Extensions Found = %d\n", extensionCount);

    //* Create array of lengths of names of each found extension
    int* extensionLengths_array = NULL;
    extensionLengths_array = (int*)malloc(extensionCount * sizeof(int));
    if (extensionLengths_array == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Allocate Memory To extensionLengths_array !!!\n", __func__);
        return FALSE;
    }

    strcpy(oclPlatformExtensions_copy_for_strtok, oclPlatformExtensions);

    token = strtok(oclPlatformExtensions_copy_for_strtok, " ");
    i = 0;
    while (token != NULL)
    {
        extensionLengths_array[i] = strlen(token) + 1;
        token = strtok(NULL, " ");
        i++;
    }

    //* Accordingly allocate a string array, such that it will hold strings equal to extension count and each string will be of required length
    //* and each string will be of required length, taken from extensionLengths_array
    char** clExtensions_array = NULL;
    clExtensions_array = (char**)malloc(extensionCount * sizeof(char*));
    if (clExtensions_array == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Allocate Memory To clExtensions_array !!!\n", __func__);
        return FALSE;
    }

    for (i = 0; i < extensionCount; i++)
    {
        clExtensions_array[i] = (char*)malloc(extensionLengths_array[i] * sizeof(char));
        if (clExtensions_array[i] == NULL)
        {
            fprintf(gpFile, "%s() => Failed To Allocate Memory To clExtensions_array[i], Index = %d !!!\n", __func__, i);
            return FALSE;
        }
    }

    //* Populate extensions into the above 2D array
    strcpy(oclPlatformExtensions_copy_for_strtok, oclPlatformExtensions);

    token = strtok(oclPlatformExtensions_copy_for_strtok, " ");
    i = 0;
    while (token != NULL)
    {
        memcpy(clExtensions_array[i], token, extensionLengths_array[i]);
        token = strtok(NULL, " ");
        i++;
    }

    free(oclPlatformExtensions_copy_for_strtok);
    oclPlatformExtensions_copy_for_strtok = NULL;

    //* Print all supported OpenCL Extensions
    fprintf(gpFile, "-------------------------------------------------------------------------------\n");
    for (i = 0; i < extensionCount; i++)
        fprintf(gpFile, "%s\n", clExtensions_array[i]);
    fprintf(gpFile, "-------------------------------------------------------------------------------\n");

    fflush(gpFile);

    //* Check whether the following extensions are present
        //*     1) cl_khr_device_uuid
        //*     2) cl_khr_external_memory
    BOOL bDeviceUuidExtensionFound = FALSE;
    BOOL bExternalMemoryExtensionFound = FALSE;

    for (i = 0; i < extensionCount; i++)
    {   
        //* For cl_khr_device_uuid Extension
        if (strcmp(clExtensions_array[i], "cl_khr_device_uuid") == 0)
            bDeviceUuidExtensionFound = TRUE;
        else
            bDeviceUuidExtensionFound = FALSE;

        //* For cl_khr_external_memory Extension
        if (strcmp(clExtensions_array[i], "cl_khr_external_memory") == 0)
            bExternalMemoryExtensionFound = TRUE;
        else
            bExternalMemoryExtensionFound = FALSE;
    }

    //! Free Memory
    for (int i = 0; i < extensionCount; i++)
    {
        free(clExtensions_array[i]);
        clExtensions_array[i] = NULL;
    }

    free(clExtensions_array);
    clExtensions_array = NULL;

    free(token);
    token = NULL;

    free(extensionLengths_array);
    extensionLengths_array = NULL;

    free(oclPlatformExtensions);
    oclPlatformExtensions = NULL;

    if (bDeviceUuidExtensionFound == TRUE && bExternalMemoryExtensionFound == TRUE)
    {
        fprintf(gpFile, "\nSelected OpenCL Platform Supports Both Extensions\n");
        
        return TRUE;
    }
    
    return FALSE;
}

void resize(int width, int height)
{
    // Code
    if (height <= 0)
        height = 1;
}

VkResult display(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Function Declarations
    void renderImGui(void);
    VkResult recordCommandBufferForImage(uint32_t imageIndex);
    VkResult initialize(void);
    void uninitialize(void);

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
        fprintf(gpFile, "%s() => vkAcquireNextImageKHR() Failed : %d\n", __func__, vkResult);
        return vkResult;
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

    //!!! ImGui Render
    renderImGui();

    if (bReinitialize)
    {
        uninitialize();
        bReinitialize = FALSE;
        initialize();
    }

    //!!! RECORD COMMANDS FOR CURRENT IMAGE
    vkResult = recordCommandBufferForImage(currentImageIndex);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => recordCommandBufferForImage() Failed : %d\n", __func__, vkResult);
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
        fprintf(gpFile, "%s() => vkQueuePresentKHR() Failed : %d\n", __func__, vkResult);
        return vkResult;
    }

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
    void uninitializeImGui(void);

    // Code
    if (bReinitialize == FALSE)
    {
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

        uninitializeImGui();

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

        //* Step - 6 of Render Pass
        if (vkRenderPass)
        {
            vkDestroyRenderPass(vkDevice, vkRenderPass, NULL);
            vkRenderPass = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyRenderPass() Succeeded\n", __func__);
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

        //! No need to free swapchain images
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
    else
    {
        //* Step - 5 of Device Creation (Destroy Vulkan Device)
        //! vkDeviceWaitIdle(vkDevice) should be the 1st API to maintain synchronization
        if (vkDevice)
        {
            vkDeviceWaitIdle(vkDevice);
            fprintf(gpFile, "%s() => vkDeviceWaitIdle() Succeeded\n", __func__);
        }

        uninitializeImGui();

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

        //* Step - 6 of Render Pass
        if (vkRenderPass)
        {
            vkDestroyRenderPass(vkDevice, vkRenderPass, NULL);
            vkRenderPass = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyRenderPass() Succeeded\n", __func__);
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

        //! No need to free swapchain images
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
        fprintf(gpFile, "%s() => Call 1 : vkEnumeratePhysicalDevices() Succeeded : Physical Device Count = %d\n", __func__, physicalDeviceCount); 
    else if (physicalDeviceCount == 0)
    {
        fprintf(gpFile, "%s() => Call 1 : vkEnumeratePhysicalDevices() Returned 0 Devices !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
    {
        fprintf(gpFile, "%s() => Call 1 : vkEnumeratePhysicalDevices() Failed : %d !!!\n", __func__, vkResult);
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
        fprintf(gpFile, "%s() => Call 2 : vkEnumeratePhysicalDevices() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
       fprintf(gpFile, "%s() => Call 2 : vkEnumeratePhysicalDevices() Succeeded\n", __func__);

    //* Step - 5
    VkBool32 bFound = VK_FALSE;
    for (uint32_t i = 0; i < physicalDeviceCount; i++)
    {
        //* Step - 5.1
        uint32_t queueCount = UINT32_MAX;

        //* Step - 5.2
        vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &queueCount, NULL);
        fprintf(gpFile, "%s() => vkGetPhysicalDeviceQueueFamilyProperties() Succeeded : Queue Count = %d\n", __func__, queueCount);

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
    VkPhysicalDeviceFeatures vkPhysicalDeviceFeatures;
    memset((void*)&vkPhysicalDeviceFeatures, 0, sizeof(VkPhysicalDeviceFeatures));
    vkGetPhysicalDeviceFeatures(vkPhysicalDevice_selected, &vkPhysicalDeviceFeatures);

    //* Step - 10
    if (vkPhysicalDeviceFeatures.tessellationShader == VK_TRUE)
        fprintf(gpFile, "%s() => Selected Physical Device Supports Tessellation Shader\n", __func__);
    else
        fprintf(gpFile, "%s() => Selected Physical Device Does Not Support Tessellation Shader !!!\n", __func__);

    if (vkPhysicalDeviceFeatures.geometryShader == VK_TRUE)
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
    fprintf(gpFile, "------------------------------------------------------------------------------------------------");

    physicalDeviceInfo_array = (PhysicalDeviceInfo*)malloc(physicalDeviceCount * sizeof(PhysicalDeviceInfo));
    
    //* Step - 3.1
    for (uint32_t i = 0; i < physicalDeviceCount; i++)
    {
        fprintf(gpFile, "\nDevice Number : %d\n", i);
        fprintf(gpFile, "*******************************************************\n");
        
        //* Step - 3.2
        VkPhysicalDeviceProperties vkPhysicalDeviceProperties;
        memset((void*)&vkPhysicalDeviceProperties, 0, sizeof(VkPhysicalDeviceProperties));
        vkGetPhysicalDeviceProperties(vkPhysicalDevice_array[i], &vkPhysicalDeviceProperties);

        physicalDeviceInfo_array[i].vkPhysicalDevice = vkPhysicalDevice_array[i];
        physicalDeviceInfo_array[i].vkPhysicalDeviceProperties = vkPhysicalDeviceProperties;

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
        fprintf(gpFile, "Device ID : 0x%4x\n", vkPhysicalDeviceProperties.deviceID);

        //* Step - 3.7
        fprintf(gpFile, "Vendor ID : 0x%4x\n", vkPhysicalDeviceProperties.vendorID);

        switch(vkPhysicalDeviceProperties.vendorID)
        {
            case 0x10DE: fprintf(gpFile, "Vendor Name : NVIDIA\n"); break;
            case 0x1002: fprintf(gpFile, "Vendor Name : AMD\n"); break;
            case 0x8086: fprintf(gpFile, "Vendor Name : Intel\n"); break;
            default: fprintf(gpFile, "Vendor Name : Unknown (0x%4x)\n", vkPhysicalDeviceProperties.vendorID);
        }

        //* Additional Properties
        uint32_t queueCount = UINT32_MAX;
        vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &queueCount, NULL);
        fprintf(gpFile, "\nNo. of Queue Families: %d\n", queueCount);

        VkQueueFamilyProperties* vkQueueFamilyProperties_array = NULL;
        vkQueueFamilyProperties_array = (VkQueueFamilyProperties*)malloc(queueCount * sizeof(VkQueueFamilyProperties));
        if (vkQueueFamilyProperties_array == NULL)
        {
            fprintf(gpFile, "%s() => malloc() Failed For vkQueueFamilyProperties_array !!!\n", __func__);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        vkGetPhysicalDeviceQueueFamilyProperties(vkPhysicalDevice_array[i], &queueCount, vkQueueFamilyProperties_array);

        VkBool32* isQueueSurfaceSupported_array = NULL;
        isQueueSurfaceSupported_array = (VkBool32*)malloc(queueCount * sizeof(VkBool32));
        if (isQueueSurfaceSupported_array == NULL)
        {
            fprintf(gpFile, "%s() => malloc() Failed For isQueueSurfaceSupported_array\n", __func__);
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }

        for (uint32_t j = 0; j < queueCount; j++)
            vkGetPhysicalDeviceSurfaceSupportKHR(vkPhysicalDevice_array[i], j, vkSurfaceKHR, &isQueueSurfaceSupported_array[j]);

        for (uint32_t j = 0; j < queueCount; j++)
        {
            fprintf(gpFile, "\nQueue Family : %d\n", j);
            fprintf(gpFile, "****************************************\n");

            if (vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_GRAPHICS_BIT)
                fprintf(gpFile, "Supports Graphics : Yes\n");
            else
                fprintf(gpFile, "Supports Graphics : No\n");

            if (vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_COMPUTE_BIT)
                fprintf(gpFile, "Supports Compute : Yes\n");
            else
                fprintf(gpFile, "Supports Compute : No\n");

            if (vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_TRANSFER_BIT)
                fprintf(gpFile, "Supports Transfer Oeprations : Yes\n");
            else
                fprintf(gpFile, "Supports Transfer Oeprations : No\n");

            if (vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR)
                fprintf(gpFile, "Supports Video Encoding : Yes\n");
            else
                fprintf(gpFile, "Supports Video Encoding : No\n");

            if (vkQueueFamilyProperties_array[j].queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR)
                fprintf(gpFile, "Supports Video Decoding : Yes\n");
            else
                fprintf(gpFile, "Supports Video Decoding : No\n");

            if (isQueueSurfaceSupported_array[j] == VK_TRUE)
                fprintf(gpFile, "Supports Presentation : Yes\n");
            else
                fprintf(gpFile, "Supports Presentation : No\n");
            fprintf(gpFile, "****************************************\n\n");
        }

        if (isQueueSurfaceSupported_array)
        {
            free(isQueueSurfaceSupported_array);
            isQueueSurfaceSupported_array = NULL;
        }

        if (vkQueueFamilyProperties_array)
        {
            free(vkQueueFamilyProperties_array);
            vkQueueFamilyProperties_array = NULL;
        }

        fprintf(gpFile, "*******************************************************\n");
    }

    fprintf(gpFile, "------------------------------------------------------------------------------------------------\n\n");

    //* Step - 3.8
    if (vkPhysicalDevice_array)
    {
        free(vkPhysicalDevice_array);
        fprintf(gpFile, "%s() => free() Succeeded For vkPhysicalDevice_array\n", __func__);
        vkPhysicalDevice_array = NULL;
    }

    // if (physicalDeviceCount >= 1)
    // {
    //     MessageBox(ghwnd, TEXT(""), TEXT("Physical Device Selection"), MB_OK);
    // }

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

        vkExtent2D_swapchain.width = max(
            vkSurfaceCapabilitiesKHR.minImageExtent.width, 
            min(vkSurfaceCapabilitiesKHR.maxImageExtent.width, vkExtent2D.width)
        );

        vkExtent2D_swapchain.height = max(
            vkSurfaceCapabilitiesKHR.minImageExtent.height,
            min(vkSurfaceCapabilitiesKHR.maxImageExtent.height, vkExtent2D.height)
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


VkResult recordCommandBufferForImage(uint32_t imageIndex)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    VkCommandBuffer commandBuffer = vkCommandBuffer_array[imageIndex];

    //* Step - 1 => Reset Command Buffer
    vkResult = vkResetCommandBuffer(commandBuffer, 0);   //! 0 specifies not to release the resources
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkResetCommandBuffer() Failed : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    //* Step - 2
    VkCommandBufferBeginInfo vkCommandBufferBeginInfo;
    memset((void*)&vkCommandBufferBeginInfo, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo.pNext = NULL;
    vkCommandBufferBeginInfo.flags = 0;     //! 0 specifies that we will use only the primary command buffer, and not going to use this command buffer simultaneously between multiple threads

    //* Step - 3
    vkResult = vkBeginCommandBuffer(commandBuffer, &vkCommandBufferBeginInfo);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

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
    vkRenderPassBeginInfo.framebuffer = vkFramebuffer_array[imageIndex];
    
    //* Step - 6
    vkCmdBeginRenderPass(commandBuffer, &vkRenderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
    {
        ImDrawData* imDrawData = ImGui::GetDrawData();
        if (imDrawData != nullptr && imDrawData->TotalVtxCount > 0)
            ImGui_ImplVulkan_RenderDrawData(imDrawData, commandBuffer);  
    }
    //* Step - 7
    vkCmdEndRenderPass(commandBuffer);

    //* Step - 8
    vkResult = vkEndCommandBuffer(commandBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkEndCommandBuffer() Failed : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
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
    ImGui_ImplWin32_Init(ghwnd);
    ImGui_ImplVulkan_InitInfo imgui_vulkan_init_info;
    memset((void*)&imgui_vulkan_init_info, 0, sizeof(ImGui_ImplVulkan_InitInfo));
    imgui_vulkan_init_info.Instance = vkInstance;
    imgui_vulkan_init_info.Device = vkDevice;
    imgui_vulkan_init_info.PhysicalDevice = vkPhysicalDevice_selected;
    imgui_vulkan_init_info.QueueFamily = graphicsQueueFamilyIndex_selected;
    imgui_vulkan_init_info.Queue = vkQueue;
    imgui_vulkan_init_info.PipelineCache = VK_NULL_HANDLE;
    imgui_vulkan_init_info.DescriptorPoolSize = 8;
    imgui_vulkan_init_info.RenderPass = vkRenderPass;
    imgui_vulkan_init_info.Subpass = 0;
    imgui_vulkan_init_info.MinImageCount = 2;
    imgui_vulkan_init_info.ImageCount = swapchainImageCount;
    imgui_vulkan_init_info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    imgui_vulkan_init_info.Allocator = NULL;
    imgui_vulkan_init_info.CheckVkResultFn = NULL;

    bool imguiStatus = ImGui_ImplVulkan_Init(&imgui_vulkan_init_info);
    if (imguiStatus == false)
    {
        fprintf(gpFile, "%s() => ImGui_ImplVulkan_Init() Failed !!!\n", __func__);
        return;
    }

    io.Fonts->AddFontFromFileTTF(fontFile, fontSize, NULL, io.Fonts->GetGlyphRangesDefault());
}

void renderImGui(void)
{
    // Code
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();

    ImGui::PushFont(font);
    {
        if (bShowPopup)
            ImGui::OpenPopup("Select Vulkan Device");

        if (ImGui::BeginPopupModal("Select Vulkan Device", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::Text("Available Vulkan Physical Devices");
            ImGui::Separator();

            for (int i = 0; i < physicalDeviceCount; i++)
            {
                VkPhysicalDeviceProperties properties = physicalDeviceInfo_array[i].vkPhysicalDeviceProperties;

                char label[256];
                sprintf(label,
                    "%d: %s (%s)",
                    i,
                    properties.deviceName,
                    properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? "dGPU" :
                    properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? "iGPU" :
                    "Other");

                if (ImGui::IsItemHovered())
                {
                    ImGui::BeginTooltip();
                    ImGui::Text("API Version : %u.%u.%u",
                        VK_API_VERSION_MAJOR(properties.apiVersion),
                        VK_API_VERSION_MINOR(properties.apiVersion),
                        VK_API_VERSION_PATCH(properties.apiVersion));
                    ImGui::Text("Vendor ID   : 0x%04x", properties.vendorID);
                    ImGui::EndTooltip();
                }

                if (ImGui::Selectable(label, selectedDevice == i))
                    selectedDevice = i;
                
            }

            ImGui::Spacing();
            ImGui::Separator();

            ImGui::BeginDisabled(selectedDevice < 0);
            if (ImGui::Button("Ok", ImVec2(120, 0)))
            {
                bReinitialize = TRUE;
                vkPhysicalDevice_selected = physicalDeviceInfo_array[selectedDevice].vkPhysicalDevice;
                bShowPopup = false;
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndDisabled();

            ImGui::SameLine();

            if (ImGui::Button("Cancel", ImVec2(120, 0)))
            {
                selectedDevice = -1;
                bShowPopup = false;
                ImGui::CloseCurrentPopup();
                uninitialize();
            }

            ImGui::EndPopup();
        }
    }
    ImGui::PopFont();

    ImGui::Render();
}

void uninitializeImGui(void)
{
    // Code
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
}