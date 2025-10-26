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

extern FILE* gpFile;
extern VkDevice vkDevice;
extern VkPhysicalDeviceMemoryProperties vkPhysicalDeviceMemoryProperties;

struct OceanSettings
{
    int tileSize = 256;
    float length = tileSize * 0.75;
    float amplitude = 2.0f;
    float windSpeed = 5.0f;
    glm::vec2 windDirection = glm::vec2(1.0, 0.0);
};

struct Vertex
{
    glm::vec3 position;
    glm::vec3 color;
    glm::vec3 normal;
    glm::vec2 texcoords;
};

//? Vertex Buffer Related Variables
struct BufferData
{
    VkBuffer vkBuffer;
    VkDeviceMemory vkDeviceMemory;
};

class Ocean
{
    private:
        OceanSettings oceanSettings;
        
        VkResult vkResult;
        cudaError_t cudaResult;
        cufftResult_t cufftResult;

        static constexpr double g = 9.82;
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

    public:
        int numTiles = 1;
        float vertexDistance = 5.0f;
        float simulationSpeed = 1.0f;
        float normalRoughness = 5.0f;
        float choppiness = -1.0f;
        double dt;

        BufferData vertexData, indexData;
        VkDeviceSize vertexBufferSize, indexBufferSize;
        uint32_t indexCount;

    private:
        VkResult createBuffers();
        void unmapMemory(VkDeviceMemory& vkDeviceMemory);

        void updateVertices();

        double phillips(const glm::vec2& K);
        double dispersion(const glm::vec2& K);
        std::complex<double> h0_tilde(const glm::vec2& K);
        std::complex<double> h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t);

    public:
        Ocean(OceanSettings settings);
        ~Ocean();
        void render();
        void update();
        void reloadSettings(OceanSettings newSettings);

};

#endif // WATER_HPP