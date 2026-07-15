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

extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;
extern VkCommandPool vkCommandPool;
extern VkQueue vkQueue;
extern VkRenderPass vkRenderPass;
extern VkExtent2D vkExtent2D_swapchain;
extern FILE* gpFile;
extern int winWidth, winHeight;

class Overlay
{   
    private:

        //* Overlay Structure
        std::vector<UICategory> categories;

        //* Vulkan Resources for rendering the UI
        struct BufferData
        {
            VkBuffer vkBuffer = VK_NULL_HANDLE;
            VkDeviceMemory vkDeviceMemory = VK_NULL_HANDLE;
            void *mapped = nullptr;
        };

        struct PushConstants
        {
            glm::vec2 scale;
            glm::vec2 translate;

        } pushData;

        BufferData vertexBuffer;
        BufferData indexBuffer;
        int32_t vertexCount = 0;
        int32_t indexCount = 0;

        VkSampler fontSampler = VK_NULL_HANDLE;
        VkImage fontImage = VK_NULL_HANDLE;
        VkImageView fontImageView = VK_NULL_HANDLE;
        VkDeviceMemory fontMemory = VK_NULL_HANDLE;

        VkShaderModule vkShaderModule_vertex_shader_imgui = VK_NULL_HANDLE;
        VkShaderModule vkShaderModule_fragment_shader_imgui = VK_NULL_HANDLE;

        VkDescriptorSetLayout vkDescriptorSetLayout_imgui = VK_NULL_HANDLE;
        VkDescriptorSet vkDescriptorSet_imgui = VK_NULL_HANDLE;
        VkDescriptorPool vkDescriptorPool_imgui = VK_NULL_HANDLE;
        VkPipeline vkPipeline_imgui = VK_NULL_HANDLE;
        VkPipelineLayout vkPipelineLayout_imgui = VK_NULL_HANDLE;

        float overlayWidth = 0.0f;
        float overlayHeight = 0.0f;
        bool visible = true;

        VkResult createBuffer(BufferData* bufferData, VkBufferUsageFlagBits bufferUsageFlagBits, VkDeviceSize bufferSize);
        VkResult mapBufferMemory(BufferData* bufferData);
        void unmapBufferMemory(BufferData* bufferData);
        void destroyBuffer(BufferData* bufferData);
        
        VkResult createFontTexture();
        VkResult createShaders();
        VkResult createDescriptorPool();
        VkResult createDescriptorSetLayout();
        VkResult createDescriptorSet();
        VkResult createPipelineLayout();
        VkResult createPipeline();

        void updateBuffers();
        void drawProperties();

        //* Win32 Integration
        void addMouseMoveHandler(LPARAM lParam);
        void addMouseButtonHandler(int buttonIndex, bool status);
        void addMouseWheelHandler(WPARAM wParam);
        void addKeyboardHandler(WPARAM wParam);
        void toggle();

        UICategory* findCategory(const std::string& name);
        UICategory* getCategory(const std::string& name);
    
    public:
    
        Overlay(float width, float height, float fontSize);
        ~Overlay();

        PerformanceStats performanceStats;

        //* Win32 Message Handler
        void registerWin32MsgHandler(HWND hwnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
        
        //* Overlay Functions
        void render(VkCommandBuffer commandBuffer);
        void newFrame();

    public:

        template<typename T>
        void addSlider(
            const std::string& categoryName, 
            const std::string& label, 
            T* value,
            T min,
            T max, 
            bool readOnly = false, 
            std::function<void()> callback = nullptr
        )
        {
            // Code
            UICategory* category = getCategory(categoryName);
            category->properties.emplace_back(
                std::make_unique<UISlider<T>>(
                    categoryName,
                    label,
                    value,
                    min,
                    max,
                    readOnly,
                    callback
                )
            );
        }

        void addCheckBox(
            const std::string& categoryName, 
            const std::string& label,
            bool* value,
            bool readOnly = false,
            std::function<void()> callback = nullptr
        )
        {
            // Code
            UICategory* category = getCategory(categoryName);
            category->properties.emplace_back(
                std::make_unique<UICheckBox>(
                    categoryName,
                    label,
                    value,
                    readOnly,
                    callback
                )
            );
        }

        void addText(
            const std::string& categoryName,
            const char* value,
            glm::vec4 color
        )
        {
            // Code
            UICategory* category = getCategory(categoryName);
            category->properties.emplace_back(
                std::make_unique<UIText>(
                    categoryName,
                    value,
                    color
                )
            );
        }

        void addDynamicText(
            const std::string& categoryName,
            std::function<std::string()> callback,
            const glm::vec4& color = glm::vec4(1.0f)
        )
        {
            // Code
            UICategory* category = getCategory(categoryName);
            category->properties.emplace_back(
                std::make_unique<UIDynamicText>(
                    categoryName,
                    std::move(callback),
                    color
                )
            );
        }


        
};


#endif

