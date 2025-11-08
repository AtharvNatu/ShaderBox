#ifndef OCEAN_HPP
#define OCEAN_HPP

//! Vulkan Related Header Files
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

//! GLM Related Macros and Header Files
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/matrix_transform.hpp>

//! CUDA Header Files
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math_constants.h>

//! C++ Headers
#include <random>
#include <complex>

#define _USE_MATH_DEFINES   1
#include <math.h>

extern FILE* gpFile;
extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;
extern VkShaderModule vkShaderModule_vertex_shader, vkShaderModule_fragment_shader;
extern VkRenderPass vkRenderPass;
extern VkViewport vkViewport;
extern VkRect2D vkRect2D_scissor;
extern VkExtent2D vkExtent2D_swapchain;
extern int winWidth, winHeight;

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

        } WaterUBO;

        //* Vertex Buffers
        BufferData vertexData_height, vertexData_slope, vertexData_position;
        VkDeviceSize heightSize, slopeSize, indexSize;
        BufferData indexData;
        VkDeviceSize meshSize;
        uint32_t indexCount;

        //* Device Vertex Data Pointers
        void *heightPtr = nullptr, *slopePtr = nullptr;

        //* CUDA
        cudaError_t cudaResult;
        cufftResult_t fftResult;
        cudaExternalMemory_t cudaExternalMemory_height = NULL;
        cudaExternalMemory_t cudaExternalMemory_slope = NULL;
        PFN_vkGetMemoryWin32HandleKHR vkGetMemoryWin32HandleKHR_fnptr = NULL;

        //* Uniform Buffers
        UniformData uniformData_mvp;
        UniformData uniformData_water;

        //* Vulkan Related
        VkDescriptorSetLayout vkDescriptorSetLayout_ocean;
        VkDescriptorPool vkDescriptorPool_ocean;
        VkDescriptorSet vkDescriptorSet_ocean;
        VkPipelineLayout vkPipelineLayout_ocean;
        VkPipeline vkPipeline_ocean;
        VkResult vkResult;

        //* Ocean Parameters
        const int MESH_SIZE = 1024;
        const int SPECTRUM_SIZE_WIDTH = MESH_SIZE + 4;
        const int SPECTRUM_SIZE_HEIGHT = MESH_SIZE + 1;

        // Mesh Resolution N * M
        const int N = MESH_SIZE;        
        const int M = MESH_SIZE;

        unsigned int meshSizeLimit = 1024;
        unsigned int spectrumW = MESH_SIZE + 4;
        unsigned int spectrumH = MESH_SIZE + 1;

        float2 *device_h_twiddle_0 = nullptr;
        float2 *host_h_twiddle_0 = nullptr;
        float2 *device_height = nullptr;
        float2 *device_slope = nullptr;

        float A = 3e-7f;                     // Phillips Spectrum Amplitude       
        float V = 30.0f;                     // Wind Speed
        glm::vec2 omega_vec = {1, 1};       //  Wind Direction
        float fTime = 0.0f;
        float heightMin = 0, heightMax = 0;
        const float waveSpeed = 0.005f;
        const float PI = float(M_PI);
        const float G = 9.8f;               //  Gravitational Constant
        const float L = 0.1;
        size_t kNum;
        glm::vec2 omega_hat; 
        float lambda;

        cufftHandle plan2d = 0;


    private:

        //* Vulkan Related
        VkResult createBuffers();
        VkResult createBuffers1();
        VkResult getMemoryWin32HandleFunction();
        VkResult createUniformBuffer();
        VkResult createDescriptorSetLayout();
        VkResult createPipelineLayout();
        VkResult createDescriptorPool();
        VkResult createDescriptorSet();
        VkResult createPipeline();

        // bool initializeHostData();
        bool initializeDeviceData();

        //* FFT + Tessendorf Related
        void generate_initial_spectrum();



        inline float omega(float k) const;
        inline float phillips_spectrum(float kx, float kz) const;

        inline float phillips_spectrum(glm::vec2 k) const;
        inline std::complex<float> func_h_twiddle_0(glm::vec2 k);

        void compute_h_twiddle_0();
        void compute_h_twiddle_0_conjugate();

        void generate_fft_data(float time);

    public:
        
        Ocean();
        ~Ocean();

        VkResult initialize();
        void buildCommandBuffers(VkCommandBuffer& commandBuffer);
        void update();
        VkResult updateUniformBuffer();
        VkResult resize(int width, int height);
};


#endif  // OCEAN_HPP
