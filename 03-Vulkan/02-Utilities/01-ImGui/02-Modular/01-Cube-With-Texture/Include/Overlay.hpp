#ifndef IMGUI_HPP
#define IMGUI_HPP

#include "imgui.h"

#define NOMINMAX
#include <cstdio>
#include <algorithm>
#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <map>

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <windowsx.h>

extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;
extern VkCommandPool vkCommandPool;
extern VkQueue vkQueue;
extern VkRenderPass vkRenderPass;
extern VkExtent2D vkExtent2D_swapchain;
extern FILE* gpFile;
extern int winWidth, winHeight;


class UIProperty
{
    public:
        virtual ~UIProperty() = default;

        virtual void draw() = 0;

        std::string category;
        std::string label;
        bool readOnly = false;
        std::function<void()> onChanged;

};

template<typename T>
class UIValue : public UIProperty
{
    public:
        
        T* value = nullptr;

        UIValue(
            const std::string& category, 
            const std::string& label, 
            T* value, 
            bool readOnly = false, 
            std::function<void()> callback = nullptr
        )
        {
            this->category = category;
            this->label = label;
            this->value = value;
            this->readOnly = readOnly;
            this->onChanged = callback;
        }

        virtual void draw() override = 0;
};


template<typename T>
class UISlider : public UIValue<T>
{
    public:
        T min;
        T max;

        UISlider(
            const std::string& category, 
            const std::string& label,
            T* value,
            T min,
            T max,
            bool readOnly = false,
            std::function<void()> callback = nullptr
        )
        : UIValue<T>(category, label, value, readOnly, callback), 
          min(min),
          max(max)
        {
        }

        void draw() override;
};

template<>
inline void UISlider<float>::draw()
{
    bool changed = ImGui::SliderFloat(this->label.c_str(), this->value, min, max);
    if (changed && this->onChanged)
        this->onChanged();
}

template<>
inline void UISlider<int>::draw()
{
    bool changed = ImGui::SliderInt(this->label.c_str(), this->value, min, max);
    if (changed && this->onChanged)
        this->onChanged();
}

class Overlay
{   
    private:

        //* Overlay Structured Map
        std::map<std::string, std::vector<std::unique_ptr<UIProperty>>> properties;

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

        void drawProperties();
    
    public:
        
        Overlay(float width, float height, float fontSize);
        ~Overlay();

        //* Win32 Integration
        void addMouseMoveHandler(LPARAM lParam);
        void addMouseButtonHandler(int buttonIndex, bool status);
        void addMouseWheelHandler(WPARAM wParam);
        void addKeyboardHandler(WPARAM wParam);

        //* Overlay Functions
        void updateBuffers();
        void drawFrame(VkCommandBuffer commandBuffer);
        void newFrame(bool updateFrameGraph, float deltaTime = 0.0f);

    public:

        template<typename T>
        void addSlider(
            const std::string& category, 
            const std::string& label, 
            T* value,
            T min,
            T max, 
            bool readOnly = false, 
            std::function<void()> callback = nullptr
        )
        {
            // Code
            properties[category].emplace_back(
                std::make_unique<UISlider<T>>(
                    category,
                    label,
                    value,
                    min,
                    max,
                    readOnly,
                    callback
                )
            );
        }
        
};


#endif

