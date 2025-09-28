#ifndef OCEAN_H
#define OCEAN_H

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//! Vulkan Related Header Files
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

//! C++ Headers
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <chrono>

//! CUDA Headers
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cufft.h>

#include "Camera.hpp"

extern FILE* gpFile;
extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;
extern VkShaderModule vkShaderModule_vertex_shader, vkShaderModule_fragment_shader;
extern VkRenderPass vkRenderPass;
extern VkViewport vkViewport;
extern VkRect2D vkRect2D_scissor;
extern VkExtent2D vkExtent2D_swapchain;


class Ocean
{
    private:

        struct OceanSettings
        {
            int tileSize = 128;
            float length = tileSize * 0.75;
            float amplitude = 2.0f;
            float windSpeed = 5.0f;
            glm::vec2 windDirection = glm::vec2(1.0, 0.0);
        };

        typedef struct 
        {
            glm::vec3 position;
            glm::vec3 color;
            glm::vec3 normal;
            glm::vec2 texcoords;
        } Vertex;

        typedef struct 
        {
            VkBuffer vkBuffer;
            VkDeviceMemory vkDeviceMemory;
        } BufferData;

        //? Uniform Related Variables
        typedef struct
        {
            glm::mat4 modelMatrix;
            glm::mat4 viewProjectionMatrix;
            glm::vec4 cameraPosition;
        } UBO;

        typedef struct
        {
            float time;
            float amplitude;
            float wavelength;
            float speed;
            float steepness;
            glm::vec4 direction;
        } OceanParams;

        typedef struct
        {
            glm::vec4 sunDirection;
            glm::vec4 sunColor;
            glm::vec4 horizonColor;
            glm::vec4 deepColor;
        } LightingUBO;


        typedef struct
        {
            VkBuffer vkBuffer;
            VkDeviceMemory vkDeviceMemory;
        } UniformData;

        BufferData vertexData, indexData;
        VkDeviceSize vertexBufferSize, indexBufferSize;
        uint32_t indexCount;

        UniformData uniformData_vbo;
        UniformData uniformData_oceanParams;
        UniformData uniformData_lighting;

        OceanSettings oceanSettings;
        
        VkResult vkResult;
        cudaError_t cudaResult;
        cufftResult_t cufftResult;

        static constexpr double G = 9.82;
        const double twoPi = glm::two_pi<double>();
        double simulationTime = 0.0;
        double period = 4.0f;
        double rippleLength = 30;

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        void* vertexMappedData = nullptr;
        void* indexMappedData = nullptr;

        // Base allocation for host data
        std::complex<double>* hostData = nullptr;

        // Pointers into hostData
        std::complex<double>* h0_tk;            // h0_tilde(k)
        std::complex<double>* h0_tmk;           // h0_tilde(-k)
        std::complex<double>* h_yDisplacement;   // h~(k, x, t) -> h(k, x, t)
        std::complex<double>* h_xDisplacement;   // x-displacement of h(k, x, t)
        std::complex<double>* h_zDisplacement;   // z-displacement of h(k, x, t)
        std::complex<double>* h_xGradient;       // x-gradient of h(k, x, t)
        std::complex<double>* h_zGradient;       // z-gradient of h(k, x, t)
        
        // Base allocation for device data
        cufftDoubleComplex* deviceData = nullptr;

        // Pointers into device data
        cufftDoubleComplex* d_yDisplacement;
        cufftDoubleComplex* d_xDisplacement;
        cufftDoubleComplex* d_zDisplacement;
        cufftDoubleComplex* d_xGradient;
        cufftDoubleComplex* d_zGradient;

        cufftHandle plan;

        VkDescriptorSetLayout vkDescriptorSetLayout_ocean;
        VkDescriptorPool vkDescriptorPool_ocean;
        VkDescriptorSet vkDescriptorSet_ocean;
        VkPipelineLayout vkPipelineLayout_ocean;
        VkPipeline vkPipeline_ocean;

    public:
        int numTiles = 1;
        float vertexDistance = 5.0f;
        float simulationSpeed = 1.0f;
        float normalRoughness = 5.0f;
        float choppiness = -1.0f;
        double dt;

        // Camera camera(
        //     glm::vec3(0.0, 1.0, 1.0),
        //     0.0, 0.0, 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0, 
        //     rotationSpeed, movementSpeed
        // );

        Camera* camera = nullptr;

        glm::vec3 cameraPosition = glm::vec3(0.0f, 1.0f, 3.0f);
        glm::vec3 cameraEye = glm::vec3(0.0f, 0.0f, -1.0f);
        glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
        float cameraSpeed = 0.0f;

    private:
        // Vulkan Related
        VkResult createBuffers();
        VkResult createUniformBuffer();
        VkResult createDescriptorSetLayout();
        VkResult createPipelineLayout();
        VkResult createDescriptorPool();
        VkResult createDescriptorSet();
        VkResult createPipeline();
        VkResult updateUniformBuffer();

        void unmapMemory(VkDeviceMemory& vkDeviceMemory);

        void updateVertices();
        void reloadSettings(OceanSettings newSettings);

        // Ocean Wave Related
        double phillipsSpectrum(const glm::vec2& K) const;
        double dispersion(const glm::vec2& K);
        std::complex<double> h0_tilde(const glm::vec2& K);
        std::complex<double> h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t);

    public:
        Ocean();
        ~Ocean();

        VkResult initialize();
        void buildCommandBuffers(VkCommandBuffer& commandBuffer);
        void update();
        VkResult resize(int width, int height);
};

#endif // OCEAN_H