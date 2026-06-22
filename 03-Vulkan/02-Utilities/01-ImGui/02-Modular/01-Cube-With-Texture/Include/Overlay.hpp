#ifndef IMGUI_HPP
#define IMGUI_HPP

#include <cstdio>
#include <array>

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
extern VkViewport vkViewport;
extern VkRect2D vkRect2D_scissor;
extern VkExtent2D vkExtent2D_swapchain;
extern FILE* gpFile;
extern int winWidth, winHeight;


class ImGUI
{   
    private:

        //* Vulkan Resources for rendering the UI
        struct BufferData
        {
            VkBuffer vkBuffer;
            VkDeviceMemory vkDeviceMemory;
            void *mapped = nullptr;
        };

        VkMemoryAllocateInfo vkMemoryAllocateInfo;

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

        struct PushConstants
        {
            glm::vec2 scale;
            glm::vec2 translate;

        } pushData;
        
        ImGUI();
        ~ImGUI();

        VkResult initialize(float width, float height);
        void updateBuffers();
        void drawFrame(VkCommandBuffer commandBuffer);
        void newFrame(bool updateFrameGraph);
};


#endif

