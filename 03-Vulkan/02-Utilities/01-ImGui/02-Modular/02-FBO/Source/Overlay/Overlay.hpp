#ifndef IMGUI_HPP
#define IMGUI_HPP

#define NOMINMAX
#include <algorithm>
#include <vector>
#include <memory>
#include <format>

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <windowsx.h>

#include "PropertyMetaData.hpp"
#include "PerformanceStats.hpp"

namespace Overlay
{
    VkResult Init(
        float width, 
        float height, 
        float fontSize,
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VkCommandPool commandPool,
        VkQueue queue,
        VkRenderPass renderPass,
        VkPhysicalDeviceMemoryProperties memoryProperties,
        uint32_t imageCount,
        FILE** pLogFile
    );

    void Cleanup();

    //* Win32 Message Handler
    void RegisterWin32MsgHandler(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
    
    //* Overlay Functions
    void Render(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void NewFrame(uint32_t imageIndex, int width, int height);

    //* Performance Stats Related
    void ShowPerformanceStats();
    void UpdatePerformanceStats();

    //! Overlay UI
    void AddText(
        const std::string& categoryName,
        const char* value,
        glm::vec4 color = glm::vec4(1.0f),
        int column = 0
    );

    void AddDynamicText(
        const std::string& categoryName,
        std::function<std::string()> callback,
        const glm::vec4& color = glm::vec4(1.0f),
        int column = 0
    );

    void AddCheckBox(
        const std::string& categoryName, 
        const std::string& label,
        bool* value,
        bool readOnly = false,
        std::function<void()> callback = nullptr
    );

    void AddSliderInt(
        const std::string& categoryName, 
        const std::string& label,
        int* value,
        int min,
        int max,
        bool readOnly = false,
        std::function<void()> callback = nullptr
    );

    void AddSliderInt2(
        const std::string& categoryName, 
        const std::string& label,
        glm::vec2* value,
        int min,
        int max,
        bool readOnly = false,
        std::function<void()> callback = nullptr
    );

    void AddSliderInt3(
        const std::string& categoryName, 
        const std::string& label,
        glm::vec3* value,
        int min,
        int max,
        bool readOnly = false,
        std::function<void()> callback = nullptr
    );

    void AddSliderInt4(
        const std::string& categoryName, 
        const std::string& label,
        glm::vec4* value,
        int min,
        int max,
        bool readOnly = false,
        std::function<void()> callback = nullptr
    );

    void AddSliderFloat(
        const std::string& categoryName, 
        const std::string& label,
        float* value,
        float min,
        float max,
        bool readOnly = false,
        std::function<void()> callback = nullptr
    );

    void AddPlotLines(
        const std::string& categoryName,
        const std::string& label,
        const std::vector<float>* buffer,
        float scaleMin = FLT_MAX,
        float scaleMax = FLT_MAX,
        ImVec2 graphSize = ImVec2(100.0f, 200.0f),
        int column = 0
    );

    namespace Detail
    {
        UICategory* GetCategory(const std::string& name);
    }
}

#endif

