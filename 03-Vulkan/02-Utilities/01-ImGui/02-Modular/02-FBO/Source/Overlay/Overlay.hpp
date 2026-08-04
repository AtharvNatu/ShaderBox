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
        std::string categoryName,
        std::string value,
        const glm::vec4& color = glm::vec4(1.0f),
        int column = 0
    );

    void AddDynamicText(
        std::string categoryName,
        std::function<std::string()> callback,
        const glm::vec4& color = glm::vec4(1.0f),
        int column = 0
    );

    void AddButton(
        std::string categoryName,
        std::string label,
        std::function<void()> callback,
        float width = 90.0f,
        float height = 30.0f
    );

    void AddCheckBox(
        std::string categoryName, 
        std::string label,
        bool* value,
        std::function<void()> callback = nullptr
    );

    void AddRadioButton(
        std::string categoryName, 
        std::string label,
        int* value,
        int data,
        std::function<void()> callback = nullptr,
        bool sameLine = true
    );

    void AddSliderInt(
        std::string categoryName, 
        std::string label,
        int* value,
        int min,
        int max,
        std::function<void()> callback = nullptr
    );

    void AddSliderInt2(
        std::string categoryName, 
        std::string label,
        glm::vec2& value,
        int min,
        int max,
        std::function<void()> callback = nullptr
    );

    void AddSliderInt3(
        std::string categoryName, 
        std::string label,
        glm::vec3& value,
        int min,
        int max,
        std::function<void()> callback = nullptr
    );

    void AddSliderInt4(
        std::string categoryName, 
        std::string label,
        glm::vec4& value,
        int min,
        int max,
        std::function<void()> callback = nullptr
    );

    void AddSliderFloat(
        std::string categoryName, 
        std::string label,
        float* value,
        float min,
        float max,
        std::function<void()> callback = nullptr
    );

     void AddSliderFloat2(
        std::string categoryName, 
        std::string label,
        glm::vec2& value,
        float min,
        float max,
        std::function<void()> callback = nullptr
    );

    void AddSliderFloat3(
        std::string categoryName, 
        std::string label,
        glm::vec3& value,
        float min,
        float max,
        std::function<void()> callback = nullptr
    );

    void AddSliderFloat4(
        std::string categoryName, 
        std::string label,
        glm::vec4& value,
        float min,
        float max,
        std::function<void()> callback = nullptr
    );

    void AddPlotLines(
        std::string categoryName,
        std::string label,
        const std::vector<float>* buffer,
        float scaleMin = FLT_MAX,
        float scaleMax = FLT_MAX,
        ImVec2 graphSize = ImVec2(100.0f, 200.0f),
        int column = 0
    );

    namespace Category
    {
        UICategory* GetCategory(const std::string& name);
    }
}

#endif

