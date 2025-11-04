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

#include "Camera.hpp"

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
            //* Light Attributes
            glm::vec4 lightPosition;
            glm::vec4 lightAmbient;
            glm::vec4 lightDiffuse;
            glm::vec4 lightSpecular;

            //* Misc
            glm::vec4 viewPosition;
            glm::vec4 heightVector;  // 0 -> Height Min | 1 -> Height Max | 2, 3 -> Padding

        } WaterUBO;

        //* Vertex Buffers
        BufferData vertexData_displacement, vertexData_normals;
        BufferData indexData;
        VkDeviceSize meshSize;
        uint32_t indexCount;

        //* Device Vertex Data Pointers
        void *displacementPtr = nullptr, *normalsPtr = nullptr;

        //* CUDA
        cudaError_t cudaResult;
        cufftResult_t fftResult;
        cudaExternalMemory_t cudaExternalMemory_displacement = NULL;
        cudaExternalMemory_t cudaExternalMemory_normals = NULL;
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
        const int MESH_SIZE = 2048;

        // Mesh Resolution N * M
        const int N = MESH_SIZE;        
        const int M = MESH_SIZE;

        // Scene Size
        float x_length = 1000.0f;            
        float z_length = 1000.0f; 

        float A = 3e-7f;                     // Phillips Spectrum Amplitude       
        float V = 30.0f;                     // Wind Speed
        glm::vec2 omega_vec = {1, 1};       //  Wind Direction
        float fTime = 0.0f;
        float heightMin = 0, heightMax = 0;
        const float waveSpeed = 0.05f;
        const float PI = float(M_PI);
        const float G = 9.8f;               //  Gravitational Constant
        const float L = 0.1;
        size_t kNum;
        glm::vec2 omega_hat; 
        float lambda;
        
        //* Scene Light
        glm::vec3 lightPosition = { 0.0f, 50, 0.0 };
        glm::vec3 lightDirection = glm::normalize(glm::vec3(0, 1, -2));

        //* Host Data
        cufftComplex *host_h_twiddle_0 = nullptr;               // Host Complex Array (Size = kNum)
        cufftComplex *host_h_twiddle_0_conjugate = nullptr;     // Conjugate Mapping

        //* Device Data
        cufftComplex *device_h_twiddle_0 = nullptr;
        cufftComplex *device_h_twiddle_0_conjugate = nullptr;
        cufftComplex *device_h_twiddle = nullptr;

        cufftComplex *device_in_height = nullptr;
        cufftComplex *device_in_slope_x = nullptr;
        cufftComplex *device_in_slope_z = nullptr;
        cufftComplex *device_in_displacement_x = nullptr;
        cufftComplex *device_in_displacement_z = nullptr;

        cufftComplex *device_out_height = nullptr;
        cufftComplex *device_out_slope_x = nullptr;
        cufftComplex *device_out_slope_z = nullptr;
        cufftComplex *device_out_displacement_x = nullptr;
        cufftComplex *device_out_displacement_z = nullptr;

        //! NEW
        std::complex<float> *h_twiddle_0 = nullptr;
        std::complex<float> *h_twiddle_0_conjunction = nullptr;

        // CUDA device buffers
        cufftComplex* d_h0 = nullptr;
        cufftComplex* d_h0_conj = nullptr;

        cufftComplex* d_h = nullptr;
        cufftComplex* d_slope_x = nullptr;
        cufftComplex* d_slope_z = nullptr;
        cufftComplex* d_disp_x = nullptr;
        cufftComplex* d_disp_z = nullptr;

        cufftHandle plan2d = 0;

        unsigned int* indices = nullptr;

        glm::mat4 cameraMatrix;

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
        void update(glm::mat4 cameraViewMatrix);
        VkResult updateUniformBuffer();
        VkResult resize(int width, int height);
};


#endif  // OCEAN_HPP
