#ifndef OCEAN_HPP
#define OCEAN_HPP

#define USE_IMGUI   1

//! Vulkan Related Header Files
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>

//! ImGui Related
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "imgui_impl_win32.h"

//! CUDA Header Files
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math_constants.h>

#define _USE_MATH_DEFINES   1
#include <math.h>

extern FILE* gpFile;
extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;
extern VkRenderPass vkRenderPass;
extern VkViewport vkViewport;
extern VkRect2D vkRect2D_scissor;
extern VkExtent2D vkExtent2D_swapchain;
extern int winWidth, winHeight;
extern VkCommandPool vkCommandPool;
extern VkQueue vkQueue;


class Ocean
{
    private:

        typedef struct 
        {
            VkBuffer vkBuffer;
            VkDeviceMemory vkDeviceMemory;
        } BufferData;

        typedef struct
        {
            VkBuffer vkBuffer;
            VkDeviceMemory vkDeviceMemory;
        } UniformData;

        typedef struct
        {
            VkImage vkImage;
            VkDeviceMemory vkDeviceMemory;
            VkImageView vkImageView;
            VkSampler vkSampler;
        } Texture;

        typedef struct
        {
            glm::mat4 modelMatrix;
            glm::mat4 viewMatrix;
            glm::mat4 projectionMatrix;
        } MVP_UniformData;

        typedef struct
        {
            glm::vec4 deepColor;
            glm::vec4 shallowColor;
            glm::vec4 skyColor;
            glm::vec4 lightDirection;

            float heightScale;
            float choppiness;
            glm::vec2 size;

        } OceanSurfaceUBO;

        //* Vertex Buffers
        BufferData vertexData_position, vertexData_height, vertexData_slope;
        BufferData indexData;
        VkDeviceSize heightSize, slopeSize, indicesSize;

        //* Texture Related Variables
        Texture bubblesTexture, foamIntensityTexture;
        
        //* Host h_twiddle_0
        float2 *host_h_twiddle_0 = nullptr;

        //* Device Vertex Data Pointers
        void *heightPtr = nullptr;
        void *slopePtr = nullptr;

        //* CUDA
        cudaError_t cudaResult;
        cufftResult_t fftResult;
        cufftHandle plan2d = 0;
        cudaExternalMemory_t cudaExternalMemory_height = nullptr;
        cudaExternalMemory_t cudaExternalMemory_slope = nullptr;
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR_fnptr = nullptr;
        
        float2 *device_h_twiddle_0 = nullptr;
        float2 *device_height = nullptr;
        float2 *device_slope = nullptr;

        //* Uniform Buffers
        UniformData uniformData_mvp;
        UniformData uniformData_ocean_surface;

        //* Vulkan Related
        VkDescriptorSetLayout vkDescriptorSetLayout_ocean;
        
        VkDescriptorSet vkDescriptorSet_ocean;
        VkPipelineLayout vkPipelineLayout_ocean;
        VkPipeline vkPipeline_ocean;
        VkShaderModule vkShaderModule_vertex_shader, vkShaderModule_fragment_shader;
        VkResult vkResult;

        //* Ocean Parameters
        int MESH_SIZE = 1024;
        int SPECTRUM_SIZE_WIDTH = MESH_SIZE + 4;
        int SPECTRUM_SIZE_HEIGHT = MESH_SIZE + 1;

        const float gravitationalConstant = 9.81f;  // Gravitational Constant
        const float waveScaleFactor = 1e-7f;       // Wave Scale Factor
        const float patchSize = 100;               // Patch Size

        float windSpeed = 50.0f;
        float windDirection = CUDART_PI_F / 3.0f;
        float waveDirectionStrength = 0.07f;

        // Wave Animation
        float fTime = 0.0f;

        //* Camera Debug
        glm::mat4 cameraViewMatrix;
        bool useCamera = false;

    private:

        //* Vulkan Related
        VkResult createBuffers();
        VkResult getMemoryWin32HandleFunction();
        float* getPositionData();
        uint32_t* generateIndices(VkDeviceSize* indexCount);
        VkResult createTexture(const char* textureFileName, Texture& texture);
        VkResult createUniformBuffer();
        VkResult updateUniformBuffer();
        VkResult createDescriptorSetLayout();
        VkResult createPipelineLayout();
        VkResult createDescriptorPool();
        VkResult createDescriptorSet();
        VkResult createPipeline();
        VkResult createShaders();
        

        //* FFT Related
        bool initializeFFT();
        void generateInitialSpectrum();
        float phillipsSpectrum(float kx, float ky);
        float urand();
        float gaussianDistribution();

    public:

        //! ImGui DEBUG
        VkDescriptorPool vkDescriptorPool_ocean;
        
        float heightScale = 0.0075f;
        float waveSpeed = 0.005f;
        int meshSize = 512;

        ImVec4 deepColor = ImVec4(0.588f, 0.823f, 1.0f, 1.0f);
        ImVec4 shallowColor = ImVec4(0.274f, 0.627f, 1.0f, 1.0f);
        ImVec4 skyColor = ImVec4(0.988f, 0.878f, 0.678f, 1.0f);
        ImVec4 lightDirection = ImVec4(-0.4f, -0.1f, -2.0f, 0.0f);

        glm::mat4 glm_translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, 0.0f));
        glm::mat4 glm_scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f, 5.0f, 5.0f));

        Ocean();
        ~Ocean();

        void update();
        void update(glm::mat4 cameraMatrix);

        VkResult resize(int width, int height);

        void buildCommandBuffers(VkCommandBuffer& commandBuffer);
};


#endif  // OCEAN_HPP
