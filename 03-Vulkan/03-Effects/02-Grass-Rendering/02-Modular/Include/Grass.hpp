#ifndef GRASS_HPP
#define GRASS_HPP

//! Vulkan Related Header Files
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//! C++ Headers
#include <vector>
#include <ctime>

//! Helper Timer
#include "helper_timer.h"


extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;
extern VkCommandPool vkCommandPool;
extern VkQueue vkQueue;
extern FILE* gpFile;
extern int winWidth, winHeight;
extern VkShaderModule vkShaderModule_vertex_shader, vkShaderModule_geometry_shader, vkShaderModule_fragment_shader;
extern VkRenderPass vkRenderPass;
extern VkViewport vkViewport;
extern VkRect2D vkRect2D_scissor;
extern VkExtent2D vkExtent2D_swapchain;

extern VkImageView vkImageView_texture_grass;
extern VkImageView vkImageView_texture_flowmap;

extern VkSampler vkSampler_texture_grass;
extern VkSampler vkSampler_texture_flowmap;

class Grass
{
    private:

        //* Vertex Buffer Related Variables
        typedef struct
        {
            VkBuffer vkBuffer;
            VkDeviceMemory vkDeviceMemory;
        } VertexData;

        //* Position Related Variables
        VertexData vertexData_grass_position;

        //* Uniform Related Variables
        typedef struct
        {
            glm::mat4 viewMatrix;
            glm::mat4 projectionMatrix;
            glm::vec4 cameraPosition;
            float time;
            float windStrength;
        } GrassUBO;

        typedef struct
        {
            VkBuffer vkBuffer;
            VkDeviceMemory vkDeviceMemory;
        } UniformData;

        UniformData uniformData;

        std::vector<glm::vec3> grassPosition;

        glm::vec3 cameraPosition = glm::vec3(0.0f, 0.0f, 3.0f);
        glm::vec3 cameraEye = glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
        float cameraSpeed = 0.0f;
        float deltaTime = 0.0f;
        float windStrength = 0.1f;

        StopWatchInterface* timer = nullptr;

        VkDescriptorSetLayout vkDescriptorSetLayout_grass;
        VkDescriptorSet vkDescriptorSet_grass;
        VkDescriptorPool vkDescriptorPool_grass;
        VkPipeline vkPipeline_grass;
        VkPipelineLayout vkPipelineLayout_grass;

    private:
        VkResult createVertexBuffer();
        VkResult createDescriptorSetLayout();
        VkResult createDescriptorSet();
        VkResult createDescriptorPool();
        VkResult createPipelineLayout();
        VkResult createPipeline();
        VkResult createUniformBuffer();
        VkResult updateUniformBuffer();

    public:
        Grass();
        ~Grass();

        VkResult initialize();
        void populateData();
        void addVertexPosition(glm::vec3 position);
        void setGrassVertices(std::vector<glm::vec3> grassVertices);
        void buildCommandBuffers(VkCommandBuffer& commandBuffer);
        void update();
        VkResult resize(int width, int height);
        void uninitialize();  
};


#endif // GRASS_HPP
