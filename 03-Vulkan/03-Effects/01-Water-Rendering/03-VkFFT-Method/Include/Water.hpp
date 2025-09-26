#ifndef WATER_HPP
#define WATER_HPP

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

//! Vulkan Related Header Files
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#define _USE_MATH_DEFINES 1

//! C++ Headers
#include <vector>
#include <complex>
#include <random>
#include <cmath>
#include <chrono>

#define VKFFT_BACKEND                   0   // Vulkan
#define VKFFT_DISABLE_GLSL_COMPILATION
#include "vkFFT/vkFFT.h"

extern FILE* gpFile;
extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;
extern VkPhysicalDevice vkPhysicalDevice_selected;
extern VkQueue vkQueue;
extern VkShaderModule vkShaderModule_compute_shader;
extern VkCommandPool vkCommandPool;
extern VkFence* vkFence_array;

struct OceanSettings
{
    int tileSize = 256;
    float length = tileSize * 0.75;
    float amplitude = 2.0f;
    float windSpeed = 5.0f;
    glm::vec2 windDirection = glm::vec2(1.0, 0.0);
};

typedef struct 
{
    glm::vec4 position;
    glm::vec4 color;
    glm::vec4 normal;
    glm::vec4 texcoords;

} Vertex;

//? Vertex Buffer Related Variables
typedef struct 
{
    VkBuffer vkBuffer;
    VkDeviceMemory vkDeviceMemory;
    VkDeviceSize vkDeviceSize;
}BufferData;

typedef struct 
{
    int tileSize;
    float vertexDistance;
    float choppiness;
    float normalRoughness;
    float half;
} PushData;


class Ocean
{
    private:
        OceanSettings oceanSettings;
        VkResult vkResult;

        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;

        void* vertexMappedPtr = nullptr;
        void* indexMappedPtr = nullptr;
        void* fftMappedPtr = nullptr;

        // Ocean Wave Computation Related
        static constexpr double G = 9.82;
        std::vector<std::complex<double>> h0_k; // Initial Spectrum

        // Spectrum CPU Buffer
        std::vector<float> tildeData;
        
    public:
        int numTiles = 1;
        float vertexDistance = 1.0f;
        float simulationSpeed = 1.0f;
        float normalRoughness = 5.0f;
        float choppiness = -1.0f;
        double time;

        BufferData vertexData, indexData, fftData;
        uint32_t vertexCount, indexCount;
        PushData pushData;

        //! COMPUTE PIPELINE
        VkDescriptorSetLayout vkDescriptorSetLayout_compute;
        VkDescriptorPool vkDescriptorPool_compute;
        VkDescriptorSet vkDescriptorSet_compute;
        VkPipelineLayout vkPipelineLayout_compute;
        VkPipeline vkPipeline_compute;

        //! FFT
        VkFFTConfiguration vkFFTConfiguration;
        VkFFTApplication vkFFTApplication;

    private:

        // Vulkan Related
        VkResult createBuffers();
        VkResult createComputeDescriptorSetLayout();
        VkResult createComputePipelineLayout();
        VkResult createComputeDescriptorPool();
        VkResult createComputeDescriptorSet();
        VkResult createComputePipeline();

        bool initializeFFT();

        void createGrid();

        void unmapMemory(VkDeviceMemory& vkDeviceMemory);


        // Ocean Wave Related
        double phillipsSpectrum(const glm::vec2& K) const;
        double dispersion(const glm::vec2& K);
        void generateH0();
        void generateSpectrum(double deltaTime);


    public:
        Ocean(OceanSettings settings);
        ~Ocean();

        void init();
        void update(double deltaTime);
        void reloadSettings(OceanSettings newSettings);

};

#endif // WATER_HPP