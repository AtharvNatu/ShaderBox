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

#include <omp.h>

//! C++ Headers
#include <random>
#include <complex>
#include <algorithm>

#define _USE_MATH_DEFINES   1
#include <math.h>

#include "fftw3.h"
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

        BufferData vertexData_displacement, vertexData_normals;
        BufferData indexData;
        void *displacementPtr = nullptr, *normalsPtr = nullptr;
        VkDeviceSize meshSize;
        uint32_t indexCount;

        UniformData uniformData_mvp;
        UniformData uniformData_water;

        VkResult vkResult;

        VkDescriptorSetLayout vkDescriptorSetLayout_ocean;
        VkDescriptorPool vkDescriptorPool_ocean;
        VkDescriptorSet vkDescriptorSet_ocean;
        VkPipelineLayout vkPipelineLayout_ocean;
        VkPipeline vkPipeline_ocean;

        const int MESH_SIZE = 256;
        const float waveSpeed = 0.05f;
        
        const int N = MESH_SIZE;
        const int M = MESH_SIZE;

        int x_length = 1000;
        int z_length = 1000;

        float A = 3e-7f;
        float V = 30;                   // Wind Speed
        glm::vec2 omega_vec = {1, 1};      // Wind Direction
        float fTime = 0.0f;
        float heightMin = 0, heightMax = 0;
        
        glm::vec3 lightPosition = { 0.0f, 50, 0.0 };
        glm::vec3 lightDirection = glm::normalize(glm::vec3(0, 1, -2));

        std::complex<float>* h_twiddle_0 = nullptr;
        std::complex<float>* h_twiddle_0_conjunction = nullptr;
        std::complex<float>* h_twiddle = nullptr;

        glm::vec3* displacement_map = nullptr;
        glm::vec3* normal_map = nullptr;

        unsigned int* indices = nullptr;

        glm::mat4 cameraMatrix;

        std::default_random_engine generator;
        std::normal_distribution<float> normal_distribution{0.0f, 1.0f};

        const float PI = float(M_PI);
        const float G = 9.8f;   // Gravitational Constant
        const float L = 0.1;

        int kNum;
        glm::vec2 omega_hat; 
        float lambda;

        // N, M                 ->  Resolution
        // x_length, z_length   ->  Actual grid lengths (meters)
        // omega_hat            ->  Direction of wind
        // V                    ->  Speed of wind

    private:

        //* Vulkan Related
        VkResult createBuffers();
        VkResult createUniformBuffer();
        VkResult createDescriptorSetLayout();
        VkResult createPipelineLayout();
        VkResult createDescriptorPool();
        VkResult createDescriptorSet();
        VkResult createPipeline();
        

        //* Tessendorf Related
        inline float omega(float k) const;
        inline float phillips_spectrum(glm::vec2 k) const;
        inline std::complex<float> func_h_twiddle_0(glm::vec2 k);
        inline std::complex<float> func_h_twiddle(int kn, int km, float t) const;
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
