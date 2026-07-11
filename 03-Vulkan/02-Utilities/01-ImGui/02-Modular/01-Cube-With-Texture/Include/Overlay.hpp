#ifndef IMGUI_HPP
#define IMGUI_HPP

#define NOMINMAX
#include <cstdio>
#include <algorithm>

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

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
    
    public:

        struct OverlayData
        {
            // Scene-wise uniforms / variables
            float cubeAnimationSpeed;
        };

        OverlayData data;
        
        Overlay();
        ~Overlay();

        void addMouseMoveHandler(LPARAM lParam);
        void addMouseButtonHandler(int buttonIndex, bool status);
        void addMouseWheelHandler(WPARAM wParam);
        void addKeyboardHandler(WPARAM wParam);

        VkResult initialize(float width, float height);
        void updateBuffers();
        void drawFrame(VkCommandBuffer commandBuffer);
        void newFrame(bool updateFrameGraph, float deltaTime = 0.0f);
};


#endif

