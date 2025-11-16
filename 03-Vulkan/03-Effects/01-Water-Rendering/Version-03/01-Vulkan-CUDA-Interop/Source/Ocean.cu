#include "Ocean.hpp"

//! Header File For Texture
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//! CUDA Math Functions and Kernels
int cudaRoundUpDivision(int a, int b)
{
  return ((a + (b - 1)) / b);
}

// Complex math Functions
__device__ float2 conjugate(float2 arg)
{
    return (make_float2(arg.x, -arg.y));
}

__device__ float2 complex_exp(float arg)
{
    return (make_float2(cosf(arg), sinf(arg)));
}

__device__ float2 complex_add(float2 a, float2 b)
{
    return (make_float2(a.x + b.x, a.y + b.y));
}

__device__ float2 complex_multiply(float2 ab, float2 cd)
{
    return (make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x));
}

__global__ void generateSpectrumKernel(
    float2 *h0, 
    float2 *height, 
    unsigned int inWidth, 
    unsigned int outWidth, 
    unsigned int outHeight, 
    float time, 
    float patchSize
)
{
    // Code
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int in_index = y * inWidth + x;
    unsigned int in_mindex = (outHeight - y) * inWidth + (outWidth - x);
    unsigned int out_index = y * outWidth + x;

    //* Wave Vector
    float2 k;
    k.x = (-(int)outWidth / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
    k.y = (-(int)outWidth / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

    //* Dispersion
    float k_len = sqrtf(k.x * k.x + k.y * k.y);
    float w = sqrtf(9.81f * k_len);

    if ((x < outWidth) && (y < outHeight))
    {
        float2 h0_k = h0[in_index];
        float2 h0_mk = h0[in_mindex];

        height[out_index] = complex_add(
            complex_multiply(h0_k, complex_exp(w * time)), 
            complex_multiply(conjugate(h0_mk), 
            complex_exp(-w * time)
        ));
    }
}

__global__ void updateHeightMapKernel(float *heightMap, float2 *height, unsigned int width)
{
    // Code
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * width + x;

    float signCorrection = 0.0f;
    if ((x + y) & 0x01)
        signCorrection = -1.0f;
    else
        signCorrection = 1.0f;

    heightMap[i] = height[i].x * signCorrection;
}

__global__ void calculateSlopeKernel(float *h, float2 *slopeOut, unsigned int width, unsigned int height)
{
    // Code
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int i = y * width + x;

    float2 slope = make_float2(0.0f, 0.0f);

    if ((x > 0) && (y > 0) && (x < width - 1) && (y < height - 1))
    {
        slope.x = h[i + 1] - h[i - 1];
        slope.y = h[i + width] - h[i - width];
    }

    slopeOut[i] = slope;
}

Ocean::Ocean()
{
    // Code
    bool status = initializeFFT();
    if (!status)
    {
        fprintf(gpFile, "%s() => initializeFFT() Failed For Ocean : %d !!!\n", __func__);
        return;
    }
    
    vkResult = createBuffers();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createBuffers() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }

    vkResult = createTexture("Assets/Images/foam_bubbles.png", bubblesTexture);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createTexture() Failed For Bubbles Texture : %d !!!\n", __func__, vkResult);
        return;
    }

    vkResult = createTexture("Assets/Images/foam_intensity.png", foamIntensityTexture);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createTexture() Failed For Foam Intensity Texture : %d !!!\n", __func__, vkResult);
        return;
    }

    vkResult = createUniformBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createUniformBuffer() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }

    vkResult = createShaders();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createShaders() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }   

    vkResult = createDescriptorSetLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }

    vkResult = createPipelineLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createPipelineLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }

    vkResult = createDescriptorPool();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createDescriptorPool() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }
        
    vkResult = createDescriptorSet();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createDescriptorSet() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }

    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createPipeline() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return;
    }    
 
}

bool Ocean::initializeFFT()
{
    // Code

    //* Create Plan 2D Complex-To-Complex
    fftResult = cufftPlan2d(&plan2d, MESH_SIZE, MESH_SIZE, CUFFT_C2C);
    if (fftResult != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "%s() => cufftPlan2d() Failed For Ocean !!!\n", __func__);
        return false;
    }

    size_t spectrumSize = SPECTRUM_SIZE_WIDTH * SPECTRUM_SIZE_HEIGHT * sizeof(float2);

    host_h_twiddle_0 = (float2*)malloc(spectrumSize);
    if (host_h_twiddle_0 == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For host_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    generateInitialSpectrum();

    cudaResult = cudaMalloc(&device_h_twiddle_0, spectrumSize);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_h_twiddle_0 !!!\n", __func__);
        return false;
    }
    
    cudaResult = cudaMemcpy(device_h_twiddle_0, host_h_twiddle_0, spectrumSize, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMemcpy() Failed For device_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    size_t outputSize = MESH_SIZE * MESH_SIZE * sizeof(float2);

    cudaResult = cudaMalloc(&device_height, outputSize);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_height !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_slope, outputSize);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_slope !!!\n", __func__);
        return false;
    }

    heightSize = MESH_SIZE * MESH_SIZE * sizeof(float);
    slopeSize = MESH_SIZE * MESH_SIZE * sizeof(float2);
    
    return true;
}

//* Vulkan Related
VkResult Ocean::createBuffers()
{
    // Code
    vkResult = getMemoryWin32HandleFunction();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => getMemoryWin32HandleFunction() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => getMemoryWin32HandleFunction() Succeeded\n", __func__);

    //! Vertex Position Data
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_position, 0, sizeof(BufferData));

    //* Step - 5
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = MESH_SIZE * MESH_SIZE * 4 * sizeof(float);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_position.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 7
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_position.vkBuffer, &vkMemoryRequirements);

    //* Step - 8
    VkMemoryAllocateInfo vkMemoryAllocateInfo;
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;

    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_position.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_position.vkBuffer, vertexData_position.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 11
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, vertexData_position.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 12
    float* positionData = getPositionData();
    memcpy(data, positionData, MESH_SIZE * MESH_SIZE * 4 * sizeof(float));

    //* Step - 13
    vkUnmapMemory(vkDevice, vertexData_position.vkDeviceMemory);

    delete[] positionData;
    positionData = nullptr;
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! Height Data
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_height, 0, sizeof(BufferData));

    VkExternalMemoryBufferCreateInfo vkExternalMemoryBufferCreateInfo;
    memset((void*)&vkExternalMemoryBufferCreateInfo, 0, sizeof(VkExternalMemoryBufferCreateInfo));
    vkExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    vkExternalMemoryBufferCreateInfo.pNext = NULL;
    vkExternalMemoryBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.size = heightSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_height.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Height Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Height Data\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_height.vkBuffer, &vkMemoryRequirements);

    VkExportMemoryAllocateInfo vkExportMemoryAllocateInfo;
    memset((void*)&vkExportMemoryAllocateInfo, 0, sizeof(VkExportMemoryAllocateInfo));
    vkExportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    vkExportMemoryAllocateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    //* Step - 8
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;
    vkMemoryAllocateInfo.pNext = &vkExportMemoryAllocateInfo;

    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_height.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Height Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Height Data\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_height.vkBuffer, vertexData_height.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Height Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Height Data\n", __func__);

    //* Export Memory For CUDA
    VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR;
    memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
    vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    vkMemoryGetWin32HandleInfoKHR.memory = vertexData_height.vkDeviceMemory;

    HANDLE vkMemoryHandle = NULL;
    vkResult = vkGetMemoryWin32HandleKHR_fnptr(vkDevice, &vkMemoryGetWin32HandleInfoKHR, &vkMemoryHandle);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Failed For Height Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Succeeded For Height Data\n", __func__);

    //* Import into CUDA
    cudaExternalMemoryHandleDesc cuExtMemoryHandleDesc;
    memset((void*)&cuExtMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
    cuExtMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuExtMemoryHandleDesc.handle.win32.handle = vkMemoryHandle;
    cuExtMemoryHandleDesc.size = heightSize;
    cuExtMemoryHandleDesc.flags = 0;
    
    cudaResult = cudaImportExternalMemory(&cudaExternalMemory_height, &cuExtMemoryHandleDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Failed For Height Data : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Succeeded For Height Data\n", __func__);

    CloseHandle(vkMemoryHandle);

    //* Map to CUDA Pointer
    cudaExternalMemoryBufferDesc cuExtMemoryBufferDesc;
    memset((void*)&cuExtMemoryBufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
    cuExtMemoryBufferDesc.offset = 0;
    cuExtMemoryBufferDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryBufferDesc.flags =0;
    
    cudaResult = cudaExternalMemoryGetMappedBuffer(&heightPtr, cudaExternalMemory_height, &cuExtMemoryBufferDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Failed For Height Data : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Succeeded For Height Data\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! Slope Data
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_slope, 0, sizeof(BufferData));

    memset((void*)&vkExternalMemoryBufferCreateInfo, 0, sizeof(VkExternalMemoryBufferCreateInfo));
    vkExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    vkExternalMemoryBufferCreateInfo.pNext = NULL;
    vkExternalMemoryBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = slopeSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_slope.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Slope Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Slope Data\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_slope.vkBuffer, &vkMemoryRequirements);

    memset((void*)&vkExportMemoryAllocateInfo, 0, sizeof(VkExportMemoryAllocateInfo));
    vkExportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    vkExportMemoryAllocateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    //* Step - 8
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;
    vkMemoryAllocateInfo.pNext = &vkExportMemoryAllocateInfo;

    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_slope.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Slope Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Slope Data\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_slope.vkBuffer, vertexData_slope.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Slope Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Slope Data\n", __func__);

    //* Export Memory For CUDA
    memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
    vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    vkMemoryGetWin32HandleInfoKHR.memory = vertexData_slope.vkDeviceMemory;

    vkMemoryHandle = NULL;
    vkResult = vkGetMemoryWin32HandleKHR_fnptr(vkDevice, &vkMemoryGetWin32HandleInfoKHR, &vkMemoryHandle);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Failed For Slope Data : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Succeeded For Slope Data\n", __func__);

    //* Import into CUDA
    memset((void*)&cuExtMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
    cuExtMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuExtMemoryHandleDesc.handle.win32.handle = vkMemoryHandle;
    cuExtMemoryHandleDesc.size = slopeSize;
    cuExtMemoryHandleDesc.flags = 0;
    
    cudaResult = cudaImportExternalMemory(&cudaExternalMemory_slope, &cuExtMemoryHandleDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Failed For Slope Data : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Succeeded For Slope Data\n", __func__);

    CloseHandle(vkMemoryHandle);

    //* Map to CUDA Pointer
    memset((void*)&cuExtMemoryBufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
    cuExtMemoryBufferDesc.offset = 0;
    cuExtMemoryBufferDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryBufferDesc.flags =0;
    
    cudaResult = cudaExternalMemoryGetMappedBuffer(&slopePtr, cudaExternalMemory_slope, &cuExtMemoryBufferDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Failed For Slope Data : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Succeeded For Slope Data\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------------

    //! Index Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //* Generate Indices
    uint32_t *indices = generateIndices(&indicesSize);

    //* Step - 4
    memset((void*)&indexData, 0, sizeof(BufferData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = indicesSize * sizeof(unsigned int);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &indexData.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Index Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Index Buffer\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, indexData.vkBuffer, &vkMemoryRequirements);

    //* Step - 8
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;

    //* Step - 8.1
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        //* Step - 8.2
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            //* Step - 8.3
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                //* Step - 8.4
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &indexData.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Index Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Index Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, indexData.vkBuffer, indexData.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Index Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Index Buffer\n", __func__);

    //* Step - 11
    data = NULL;
    vkResult = vkMapMemory(vkDevice, indexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Index Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Index Buffer\n", __func__);

    //* Step - 12
    memcpy(data, indices, indicesSize * sizeof(uint32_t));

    //* Step - 13
    vkUnmapMemory(vkDevice, indexData.vkDeviceMemory);

    delete[] indices;
    indices = nullptr;

    //! ---------------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult Ocean::getMemoryWin32HandleFunction()
{
    VkResult vkResult = VK_SUCCESS;
    
    //* Get the required function pointer
    vkGetMemoryWin32HandleKHR_fnptr = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(vkDevice, "vkGetMemoryWin32HandleKHR");
    if (vkGetMemoryWin32HandleKHR_fnptr == NULL)
    {
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        fprintf(gpFile, "%s() => vkGetDeviceProcAddr() Failed To Get Function Pointer For vkGetMemoryWin32HandleKHR !!!\n", __func__);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkGetDeviceProcAddr() Succeeded To Get Function Pointer For vkGetMemoryWin32HandleKHR\n", __func__);

    return vkResult;
}

float* Ocean::getPositionData()
{
    float *position = new float[MESH_SIZE * MESH_SIZE * 4];

    for (int y = 0; y < MESH_SIZE; y++)
    {
        for (int x = 0; x < MESH_SIZE; x++)
        {
            int idx = (y * MESH_SIZE + x) * 4;
            float u = x / (float)(MESH_SIZE - 1);
            float v = y / (float)(MESH_SIZE - 1);

            position[idx + 0] = u * 2.0f - 1.0f;  // X
            position[idx + 1] = 0.0f;             // Y
            position[idx + 2] = v * 2.0f - 1.0f;  // Z
            position[idx + 3] = 1.0f;             // W
        }
    }

    return position;
}

uint32_t* Ocean::generateIndices(VkDeviceSize* indexCount)
{
    // Variable Declarations
    const uint32_t RESTART_INDEX = 0xFFFFFFFF;

    uint32_t indicesPerStrip = MESH_SIZE * 2;
    uint32_t totalStrips = MESH_SIZE - 1;

    // Code

    // Total Indices = Strips * (Indices Per Strip + 1 Restart Index)
    *indexCount = totalStrips * (indicesPerStrip + 1);

    uint32_t *indices = new uint32_t[*indexCount];
    unsigned int *pIndices = indices;

    for (uint32_t y = 0; y < MESH_SIZE - 1; y++)
    {
        for (uint32_t x = 0; x < MESH_SIZE; x++)
        {
            // Bottom Vertex
            uint32_t bottomVertex = y * MESH_SIZE + x;
            *pIndices = bottomVertex;
            *pIndices++;

            // Top Vertex
            uint32_t topVertex = (y + 1) * MESH_SIZE + x;
            *pIndices = topVertex;
            *pIndices++;
        }

        // Add primitive restart index at the end of each strip
        *pIndices = RESTART_INDEX;
        *pIndices++;
    }

    return indices;
}

VkResult Ocean::createTexture(const char* textureFileName, Texture& texture)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;
    uint8_t* imageData = NULL;
    int imageWidth = 0, imageHeight = 0, numChannels = 0;

    VkBuffer vkBuffer_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize imageSize;

    // Code

    //! Step - 1
    imageData = stbi_load(textureFileName, &imageWidth, &imageHeight, &numChannels, STBI_rgb_alpha);
    if (imageData == NULL || imageWidth <= 0 || imageHeight <= 0 || numChannels <= 0)
    {
        fprintf(gpFile, "%s() => stbi_load() Failed For %s !!!\n", __func__, textureFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    imageSize = imageWidth * imageHeight * 4;

    //! Step - 2
    VkBufferCreateInfo vkBufferCreateInfo_stagingBuffer;
    memset((void*)&vkBufferCreateInfo_stagingBuffer, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo_stagingBuffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo_stagingBuffer.pNext = NULL;
    vkBufferCreateInfo_stagingBuffer.flags = 0;
    vkBufferCreateInfo_stagingBuffer.size = imageSize;
    vkBufferCreateInfo_stagingBuffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo_stagingBuffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;  //* This denotes source data to be transferred to VkImage

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo_stagingBuffer, NULL, &vkBuffer_stagingBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
            fprintf(gpFile, "%s() => stbi_image_free() Called For Texture : %s\n", __func__, textureFileName);
        }

        return vkResult;
    }

    VkMemoryRequirements vkMemoryRequirements_stagingBuffer;
    memset((void*)&vkMemoryRequirements_stagingBuffer, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vkBuffer_stagingBuffer, &vkMemoryRequirements_stagingBuffer);

    VkMemoryAllocateInfo vkMemoryAllocateInfo_stagingBuffer;
    memset((void*)&vkMemoryAllocateInfo_stagingBuffer, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo_stagingBuffer.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo_stagingBuffer.pNext = NULL;
    vkMemoryAllocateInfo_stagingBuffer.allocationSize = vkMemoryRequirements_stagingBuffer.size;
    vkMemoryAllocateInfo_stagingBuffer.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements_stagingBuffer.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                vkMemoryAllocateInfo_stagingBuffer.memoryTypeIndex = i;
                break;
            }
        }
        vkMemoryRequirements_stagingBuffer.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo_stagingBuffer, NULL, &vkDeviceMemory_stagingBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
        }

        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, vkBuffer_stagingBuffer, vkDeviceMemory_stagingBuffer, 0);
    if (vkResult != VK_SUCCESS)
    { 
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
        }

        return vkResult;
    }

    void* data = NULL;
    vkResult = vkMapMemory(
        vkDevice, 
        vkDeviceMemory_stagingBuffer,
        0, 
        imageSize, 
        0, 
        &data
    );
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
        }

        return vkResult;
    }

    memcpy(data, imageData, imageSize);

    vkUnmapMemory(vkDevice, vkDeviceMemory_stagingBuffer);

    //* Free the image data given by stb, as it is copied in image staging buffer
    stbi_image_free(imageData);
    imageData = NULL;

    //! Step - 3
    VkImageCreateInfo vkImageCreateInfo;
    memset((void*)&vkImageCreateInfo, 0, sizeof(VkImageCreateInfo));
    vkImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    vkImageCreateInfo.flags = 0;
    vkImageCreateInfo.pNext = NULL;
    vkImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    vkImageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    vkImageCreateInfo.extent.width = imageWidth;
    vkImageCreateInfo.extent.height = imageHeight;
    vkImageCreateInfo.extent.depth = 1;
    vkImageCreateInfo.mipLevels = 1;
    vkImageCreateInfo.arrayLayers = 1;
    vkImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    vkImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    vkImageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    vkImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkResult = vkCreateImage(vkDevice, &vkImageCreateInfo, NULL, &texture.vkImage);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateImage() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }     

    VkMemoryRequirements vkMemoryRequirements_image;
    memset((void*)&vkMemoryRequirements_image, 0, sizeof(VkMemoryRequirements));
    vkGetImageMemoryRequirements(vkDevice, texture.vkImage, &vkMemoryRequirements_image);

    VkMemoryAllocateInfo vkMemoryAllocateInfo_image;
    memset((void*)&vkMemoryAllocateInfo_image, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo_image.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo_image.pNext = NULL;
    vkMemoryAllocateInfo_image.memoryTypeIndex = 0;
    vkMemoryAllocateInfo_image.allocationSize = vkMemoryRequirements_image.size;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements_image.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            {
                vkMemoryAllocateInfo_image.memoryTypeIndex = i;
                break;
            }
        }
        vkMemoryRequirements_image.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo_image, NULL, &texture.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    vkResult = vkBindImageMemory(vkDevice, texture.vkImage, texture.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindImageMemory() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
        }

        return vkResult;
    } 

    //! Step - 4
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------

    //* Step - 4.1
    VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo_transition_image_layout;
    memset((void*)&vkCommandBufferAllocateInfo_transition_image_layout, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    vkCommandBufferAllocateInfo_transition_image_layout.pNext = NULL;
    vkCommandBufferAllocateInfo_transition_image_layout.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo_transition_image_layout.commandBufferCount = 1;
    vkCommandBufferAllocateInfo_transition_image_layout.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VkCommandBuffer vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
    vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo_transition_image_layout, &vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_transition_image_layout : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }   

    //* Step - 4.2
    VkCommandBufferBeginInfo vkCommandBufferBeginInfo_image_transition_layout;
    memset((void*)&vkCommandBufferBeginInfo_image_transition_layout, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_image_transition_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_image_transition_layout.pNext = NULL;
    vkCommandBufferBeginInfo_image_transition_layout.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_transition_image_layout, &vkCommandBufferBeginInfo_image_transition_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 4.3
    VkPipelineStageFlags vkPipelineStageFlags_source = 0;
    VkPipelineStageFlags vkPipelineStageFlags_destination = 0;

    VkImageMemoryBarrier vkImageMemoryBarrier;
    memset((void*)&vkImageMemoryBarrier, 0, sizeof(VkImageMemoryBarrier));
    vkImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    vkImageMemoryBarrier.pNext = NULL;
    vkImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.image = texture.vkImage;
    vkImageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
    vkImageMemoryBarrier.subresourceRange.baseMipLevel = 0;
    vkImageMemoryBarrier.subresourceRange.layerCount = 1;
    vkImageMemoryBarrier.subresourceRange.levelCount = 1;

    if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = 0;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        fprintf(gpFile, "ERROR : %s() => Unsupported Texture Layout Transition !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    vkCmdPipelineBarrier(
        vkCommandBuffer_transition_image_layout,
        vkPipelineStageFlags_source,
        vkPipelineStageFlags_destination,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &vkImageMemoryBarrier
    );

    //* Step - 4.4
    vkResult = vkEndCommandBuffer(vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 4.5
    VkSubmitInfo vkSubmitInfo_transition_image_layout;
    memset((void*)&vkSubmitInfo_transition_image_layout, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_transition_image_layout.pNext = NULL;
    vkSubmitInfo_transition_image_layout.commandBufferCount = 1;
    vkSubmitInfo_transition_image_layout.pCommandBuffers = &vkCommandBuffer_transition_image_layout;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_transition_image_layout, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 4.6
    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 4.7
    if (vkCommandBuffer_transition_image_layout)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
        vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
    }
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    //! Step - 5
    VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo_buffer_to_image_copy;
    memset((void*)&vkCommandBufferAllocateInfo_buffer_to_image_copy, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo_buffer_to_image_copy.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.pNext = NULL;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.commandBufferCount = 1;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VkCommandBuffer vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
    vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo_buffer_to_image_copy, &vkCommandBuffer_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_buffer_to_image_copy : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    VkCommandBufferBeginInfo vkCommandBufferBeginInfo_buffer_to_image_copy;
    memset((void*)&vkCommandBufferBeginInfo_buffer_to_image_copy, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_buffer_to_image_copy.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_buffer_to_image_copy.pNext = NULL;
    vkCommandBufferBeginInfo_buffer_to_image_copy.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_buffer_to_image_copy, &vkCommandBufferBeginInfo_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_buffer_to_image_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    VkBufferImageCopy vkBufferImageCopy;
    memset((void*)&vkBufferImageCopy, 0, sizeof(VkBufferImageCopy));
    vkBufferImageCopy.bufferOffset = 0;
    vkBufferImageCopy.bufferRowLength = 0;
    vkBufferImageCopy.bufferImageHeight = 0;
    vkBufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkBufferImageCopy.imageSubresource.mipLevel = 0;
    vkBufferImageCopy.imageSubresource.baseArrayLayer = 0;
    vkBufferImageCopy.imageSubresource.layerCount = 1;
    vkBufferImageCopy.imageOffset.x = 0; 
    vkBufferImageCopy.imageOffset.y = 0;
    vkBufferImageCopy.imageOffset.z = 0;
    vkBufferImageCopy.imageExtent.width = imageWidth;
    vkBufferImageCopy.imageExtent.height = imageHeight;
    vkBufferImageCopy.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(
        vkCommandBuffer_buffer_to_image_copy,
        vkBuffer_stagingBuffer,
        texture.vkImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &vkBufferImageCopy
    );

    vkResult = vkEndCommandBuffer(vkCommandBuffer_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_buffer_to_image_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (texture.vkImage)
        {
            vkDestroyImage(vkDevice, texture.vkImage, NULL);
            texture.vkImage = NULL;
        }
        if (texture.vkDeviceMemory)
        {
            vkFreeMemory(vkDevice, texture.vkDeviceMemory, NULL);
            texture.vkDeviceMemory = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    VkSubmitInfo vkSubmitInfo_buffer_to_copy;
    memset((void*)&vkSubmitInfo_buffer_to_copy, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_buffer_to_copy.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_buffer_to_copy.pNext = NULL;
    vkSubmitInfo_buffer_to_copy.commandBufferCount = 1;
    vkSubmitInfo_buffer_to_copy.pCommandBuffers = &vkCommandBuffer_buffer_to_image_copy;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_buffer_to_copy, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_buffer_to_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueSubmit() Succeeded For vkSubmitInfo_buffer_to_copy\n", __func__);

    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkCommandBuffer_buffer_to_image_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    if (vkCommandBuffer_buffer_to_image_copy)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
        vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
    }    

    //! Step - 6
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------

    //* Step - 6.1
    memset((void*)&vkCommandBufferAllocateInfo_transition_image_layout, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    vkCommandBufferAllocateInfo_transition_image_layout.pNext = NULL;
    vkCommandBufferAllocateInfo_transition_image_layout.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo_transition_image_layout.commandBufferCount = 1;
    vkCommandBufferAllocateInfo_transition_image_layout.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
    vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo_transition_image_layout, &vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_transition_image_layout : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }  

    //* Step - 6.2
    memset((void*)&vkCommandBufferBeginInfo_image_transition_layout, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_image_transition_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_image_transition_layout.pNext = NULL;
    vkCommandBufferBeginInfo_image_transition_layout.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_transition_image_layout, &vkCommandBufferBeginInfo_image_transition_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 6.3
    vkPipelineStageFlags_source = 0;
    vkPipelineStageFlags_destination = 0;

    memset((void*)&vkImageMemoryBarrier, 0, sizeof(VkImageMemoryBarrier));
    vkImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    vkImageMemoryBarrier.pNext = NULL;
    vkImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.image = texture.vkImage;
    vkImageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
    vkImageMemoryBarrier.subresourceRange.baseMipLevel = 0;
    vkImageMemoryBarrier.subresourceRange.layerCount = 1;
    vkImageMemoryBarrier.subresourceRange.levelCount = 1;

    if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = 0;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        fprintf(gpFile, "ERROR : %s() => Unsupported Texture Layout Transition !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    vkCmdPipelineBarrier(
        vkCommandBuffer_transition_image_layout,
        vkPipelineStageFlags_source,
        vkPipelineStageFlags_destination,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &vkImageMemoryBarrier
    );

    //* Step - 6.4
    vkResult = vkEndCommandBuffer(vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 6.5
    memset((void*)&vkSubmitInfo_transition_image_layout, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_transition_image_layout.pNext = NULL;
    vkSubmitInfo_transition_image_layout.commandBufferCount = 1;
    vkSubmitInfo_transition_image_layout.pCommandBuffers = &vkCommandBuffer_transition_image_layout;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_transition_image_layout, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 6.6
    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 6.7
    if (vkCommandBuffer_transition_image_layout)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
        vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
    }
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    //! Step - 7
    if (vkBuffer_stagingBuffer)
    {
        vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
        vkBuffer_stagingBuffer = VK_NULL_HANDLE;
    }

    if (vkDeviceMemory_stagingBuffer)
    {
        vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
        vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
    }

    //! Step - 8
    VkImageViewCreateInfo vkImageViewCreateInfo;
    memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
    vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vkImageViewCreateInfo.pNext = NULL;
    vkImageViewCreateInfo.flags = 0;
    vkImageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    vkImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    vkImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    vkImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    vkImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    vkImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    vkImageViewCreateInfo.subresourceRange.levelCount = 1;
    vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    vkImageViewCreateInfo.subresourceRange.layerCount = 1;
    vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vkImageViewCreateInfo.image = texture.vkImage;

    vkResult = vkCreateImageView(vkDevice, &vkImageViewCreateInfo, NULL, &texture.vkImageView);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateImageView() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }   

    //! Step - 9
    VkSamplerCreateInfo vkSamplerCreateInfo;
    memset((void*)&vkSamplerCreateInfo, 0, sizeof(VkSamplerCreateInfo));
    vkSamplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    vkSamplerCreateInfo.pNext = NULL;
    vkSamplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    vkSamplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    vkSamplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkSamplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkSamplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkSamplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkSamplerCreateInfo.anisotropyEnable = VK_FALSE;
    vkSamplerCreateInfo.maxAnisotropy = 16;
    vkSamplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    vkSamplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    vkSamplerCreateInfo.compareEnable = VK_FALSE;
    vkSamplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    vkResult = vkCreateSampler(vkDevice, &vkSamplerCreateInfo, NULL, &texture.vkSampler);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateSampler() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }    

    return vkResult;
}

VkResult Ocean::createUniformBuffer()
{
    // Code

    //! Vertex Uniform Buffer
    //! ---------------------------------------------------------------------------------------------------------
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(MVP_UniformData);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData_mvp, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData_mvp.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Uniform Data\n", __func__);

    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData_mvp.vkBuffer, &vkMemoryRequirements);

    VkMemoryAllocateInfo vkMemoryAllocateInfo;
    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData_mvp.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Uniform Data\n", __func__);

    vkResult = vkBindBufferMemory(vkDevice, uniformData_mvp.vkBuffer, uniformData_mvp.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Uniform Data\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------


    //! Water UBO
    //! ---------------------------------------------------------------------------------------------------------
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(OceanSurfaceUBO);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData_ocean_surface, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData_ocean_surface.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Water Surface Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Water Surface Uniform Data\n", __func__);

    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData_ocean_surface.vkBuffer, &vkMemoryRequirements);

    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData_ocean_surface.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Water Surface Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Water Surface Uniform Data\n", __func__);

    vkResult = vkBindBufferMemory(vkDevice, uniformData_ocean_surface.vkBuffer, uniformData_ocean_surface.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Water Surface Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Water Surface Uniform Data\n", __func__);


    vkResult = updateUniformBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => updateUniformBuffer() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => updateUniformBuffer() Succeeded\n", __func__);

    return vkResult;
}

VkResult Ocean::updateUniformBuffer()
{
    // Code
    MVP_UniformData mvpData;
    memset((void*)&mvpData, 0, sizeof(MVP_UniformData));

    glm::mat4 translationMatrix = glm::mat4(1.0f);
    glm::mat4 scaleMatrix = glm::mat4(1.0f);

    // translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, -1.5f));
    translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, -0.5f));
    scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f, 5.0f, 5.0f));

    // translationMatrix = glm_translationMatrix;
    // scaleMatrix = glm_scaleMatrix;


    mvpData.modelMatrix = translationMatrix * scaleMatrix;

    if (useCamera)
        mvpData.viewMatrix = cameraViewMatrix;
    else    
        mvpData.viewMatrix = glm::mat4(1.0f);
    
    glm::mat4 perspectiveProjectionMatrix = glm::mat4(1.0f);
    perspectiveProjectionMatrix = glm::perspective(
        glm::radians(45.0f),
        (float)winWidth / (float)winHeight,
        0.1f,
        1000.0f
    );
    //! 2D Matrix with Column Major (Like OpenGL)
    perspectiveProjectionMatrix[1][1] = perspectiveProjectionMatrix[1][1] * (-1.0f);
    mvpData.projectionMatrix = perspectiveProjectionMatrix;

    OceanSurfaceUBO waterUBO;
    memset((void*)&waterUBO, 0, sizeof(OceanSurfaceUBO));

    // waterUBO.heightScale = 0.1f;
    // waterUBO.choppiness = 0.2f;
    // waterUBO.size = glm::vec2((float)MESH_SIZE, (float)MESH_SIZE);
    // waterUBO.deepColor = glm::vec4(0.02f, 0.05f, 0.1f, 1.0f);
    // waterUBO.shallowColor = glm::vec4(0.0f, 0.64f, 0.68f, 1.0f);
    // waterUBO.skyColor = glm::vec4(0.65f, 0.80f, 0.95f, 1.0f);
    // waterUBO.lightDirection = glm::vec4(-0.45f, 2.1f, -3.5f, 0.0f);

    waterUBO.heightScale = heightScale;
    waterUBO.choppiness = 0.2f;
    waterUBO.size = glm::vec2((float)meshSize, (float)meshSize);
    waterUBO.deepColor = glm::vec4(deepColor.x, deepColor.y, deepColor.z, 1.0f);
    waterUBO.shallowColor = glm::vec4(shallowColor.x, shallowColor.y, shallowColor.z, 1.0f);
    waterUBO.skyColor = glm::vec4(skyColor.x, skyColor.y, skyColor.z, 1.0f);
    waterUBO.lightDirection = glm::vec4(lightDirection.x, lightDirection.y, lightDirection.z, lightDirection.w);

    //! Map Uniform Buffer
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, uniformData_mvp.vkDeviceMemory, 0, sizeof(MVP_UniformData), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer (Vertex UBO) : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &mvpData, sizeof(MVP_UniformData));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData_mvp.vkDeviceMemory);

    //! Map Uniform Buffer
    data = NULL;
    vkResult = vkMapMemory(vkDevice, uniformData_ocean_surface.vkDeviceMemory, 0, sizeof(OceanSurfaceUBO), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer (OceanSurfaceUBO) : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &waterUBO, sizeof(OceanSurfaceUBO));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData_ocean_surface.vkDeviceMemory);

    return vkResult;
}

VkResult Ocean::createDescriptorSetLayout()
{
    //! Initialize VkDescriptorSetLayoutBinding
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding_array[4];
    memset((void*)vkDescriptorSetLayoutBinding_array, 0, sizeof(VkDescriptorSetLayoutBinding) * _ARRAYSIZE(vkDescriptorSetLayoutBinding_array));

    //! Vertex UBO
    vkDescriptorSetLayoutBinding_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[0].binding = 0;
    vkDescriptorSetLayoutBinding_array[0].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    vkDescriptorSetLayoutBinding_array[0].pImmutableSamplers = NULL;

    //! Water Surface UBO
    vkDescriptorSetLayoutBinding_array[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[1].binding = 1;
    vkDescriptorSetLayoutBinding_array[1].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    vkDescriptorSetLayoutBinding_array[1].pImmutableSamplers = NULL;

    //! Bubbles Texture
    vkDescriptorSetLayoutBinding_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorSetLayoutBinding_array[2].binding = 2;
    vkDescriptorSetLayoutBinding_array[2].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkDescriptorSetLayoutBinding_array[2].pImmutableSamplers = NULL;

    //! Foam Intensity Texture
    vkDescriptorSetLayoutBinding_array[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorSetLayoutBinding_array[3].binding = 3;
    vkDescriptorSetLayoutBinding_array[3].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkDescriptorSetLayoutBinding_array[3].pImmutableSamplers = NULL;

    //* Step - 3
    VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo;
    memset((void*)&vkDescriptorSetLayoutCreateInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
    vkDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vkDescriptorSetLayoutCreateInfo.pNext = NULL;
    vkDescriptorSetLayoutCreateInfo.flags = 0;
    vkDescriptorSetLayoutCreateInfo.bindingCount = _ARRAYSIZE(vkDescriptorSetLayoutBinding_array);
    vkDescriptorSetLayoutCreateInfo.pBindings = vkDescriptorSetLayoutBinding_array;

    //* Step - 4
    vkResult = vkCreateDescriptorSetLayout(vkDevice, &vkDescriptorSetLayoutCreateInfo, NULL, &vkDescriptorSetLayout_ocean);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Succeeded For Ocean\n", __func__);

    return vkResult;
}

VkResult Ocean::createPipelineLayout()
{
    //* Step - 3
    VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo;
    memset((void*)&vkPipelineLayoutCreateInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
    vkPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkPipelineLayoutCreateInfo.pNext = NULL;
    vkPipelineLayoutCreateInfo.flags = 0;
    vkPipelineLayoutCreateInfo.setLayoutCount = 1;
    vkPipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout_ocean;
    vkPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    vkPipelineLayoutCreateInfo.pPushConstantRanges = NULL;

    //* Step - 4
    vkResult = vkCreatePipelineLayout(vkDevice, &vkPipelineLayoutCreateInfo, NULL, &vkPipelineLayout_ocean);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Succeeded For Ocean\n", __func__);

    return vkResult;
}

VkResult Ocean::createDescriptorPool()
{
    // Code

    //* Vulkan expects decriptor pool size before creating actual descriptor pool
    VkDescriptorPoolSize vkDescriptorPoolSize_array[2];
    memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * _ARRAYSIZE(vkDescriptorPoolSize_array));

    vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorPoolSize_array[0].descriptorCount = 2;

    vkDescriptorPoolSize_array[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    // vkDescriptorPoolSize_array[1].descriptorCount = 2;
    vkDescriptorPoolSize_array[1].descriptorCount = IMGUI_IMPL_VULKAN_MINIMUM_IMAGE_SAMPLER_POOL_SIZE;

    //* Create the pool
    VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
    memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkDescriptorPoolCreateInfo.pNext = NULL;
    vkDescriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    vkDescriptorPoolCreateInfo.poolSizeCount = _ARRAYSIZE(vkDescriptorPoolSize_array);
    vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSize_array;
    // vkDescriptorPoolCreateInfo.maxSets = 2;

    for (int i = 0; i < _ARRAYSIZE(vkDescriptorPoolSize_array); i++)
        vkDescriptorPoolCreateInfo.maxSets = vkDescriptorPoolCreateInfo.maxSets + vkDescriptorPoolSize_array[i].descriptorCount;

    vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool_ocean);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Succeeded For Ocean\n", __func__);

    return vkResult;
}

VkResult Ocean::createDescriptorSet()
{
    // Code

    //* Initialize DescriptorSetAllocationInfo
    VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo;
    memset((void*)&vkDescriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
    vkDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vkDescriptorSetAllocateInfo.pNext = NULL;
    vkDescriptorSetAllocateInfo.descriptorPool = vkDescriptorPool_ocean;
    vkDescriptorSetAllocateInfo.descriptorSetCount = 1;
    vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout_ocean;

    vkResult = vkAllocateDescriptorSets(vkDevice, &vkDescriptorSetAllocateInfo, &vkDescriptorSet_ocean);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Failed For Ocean : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Succeeded For Ocean\n", __func__);

    //* Describe whether we want buffer as uniform or image as uniform
    VkDescriptorBufferInfo vkDescriptorBufferInfo_array[2];
    memset((void*)vkDescriptorBufferInfo_array, 0, sizeof(VkDescriptorBufferInfo) * _ARRAYSIZE(vkDescriptorBufferInfo_array));

    //! Vertex UBO
    vkDescriptorBufferInfo_array[0].buffer = uniformData_mvp.vkBuffer;
    vkDescriptorBufferInfo_array[0].offset = 0;
    vkDescriptorBufferInfo_array[0].range = sizeof(MVP_UniformData);

    //! Water Surface UBO
    vkDescriptorBufferInfo_array[1].buffer = uniformData_ocean_surface.vkBuffer;
    vkDescriptorBufferInfo_array[1].offset = 0;
    vkDescriptorBufferInfo_array[1].range = sizeof(OceanSurfaceUBO);

    VkDescriptorImageInfo vkDescriptorImageInfo_array[2];
    memset((void*)vkDescriptorImageInfo_array, 0, sizeof(VkDescriptorImageInfo) * _ARRAYSIZE(vkDescriptorImageInfo_array));

    //! Bubbles Texture
    vkDescriptorImageInfo_array[0].imageView = bubblesTexture.vkImageView;
    vkDescriptorImageInfo_array[0].sampler = bubblesTexture.vkSampler;
    vkDescriptorImageInfo_array[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    //! Foam Intensity Texture
    vkDescriptorImageInfo_array[1].imageView = bubblesTexture.vkImageView;
    vkDescriptorImageInfo_array[1].sampler = bubblesTexture.vkSampler;
    vkDescriptorImageInfo_array[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet vkWriteDescriptorSet_array[4];
    memset((void*)vkWriteDescriptorSet_array, 0, sizeof(VkWriteDescriptorSet) * _ARRAYSIZE(vkWriteDescriptorSet_array));

    //! Vertex UBO
    vkWriteDescriptorSet_array[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[0].pNext = NULL;
    vkWriteDescriptorSet_array[0].dstSet = vkDescriptorSet_ocean;
    vkWriteDescriptorSet_array[0].dstArrayElement = 0;
    vkWriteDescriptorSet_array[0].dstBinding = 0;
    vkWriteDescriptorSet_array[0].descriptorCount = 1;
    vkWriteDescriptorSet_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkWriteDescriptorSet_array[0].pBufferInfo = &vkDescriptorBufferInfo_array[0];
    vkWriteDescriptorSet_array[0].pImageInfo = NULL;
    vkWriteDescriptorSet_array[0].pTexelBufferView = NULL;

    //! Water Surface UBO
    vkWriteDescriptorSet_array[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[1].pNext = NULL;
    vkWriteDescriptorSet_array[1].dstSet = vkDescriptorSet_ocean;
    vkWriteDescriptorSet_array[1].dstArrayElement = 0;
    vkWriteDescriptorSet_array[1].dstBinding = 1;
    vkWriteDescriptorSet_array[1].descriptorCount = 1;
    vkWriteDescriptorSet_array[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkWriteDescriptorSet_array[1].pBufferInfo = &vkDescriptorBufferInfo_array[1];
    vkWriteDescriptorSet_array[1].pImageInfo = NULL;
    vkWriteDescriptorSet_array[1].pTexelBufferView = NULL;

    //! Bubbles Texture
    vkWriteDescriptorSet_array[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[2].pNext = NULL;
    vkWriteDescriptorSet_array[2].dstSet = vkDescriptorSet_ocean;
    vkWriteDescriptorSet_array[2].dstArrayElement = 0;
    vkWriteDescriptorSet_array[2].dstBinding = 2;
    vkWriteDescriptorSet_array[2].descriptorCount = 1;
    vkWriteDescriptorSet_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkWriteDescriptorSet_array[2].pBufferInfo = NULL;
    vkWriteDescriptorSet_array[2].pImageInfo = &vkDescriptorImageInfo_array[0];
    vkWriteDescriptorSet_array[2].pTexelBufferView = NULL;

    //! Foam Intensity Texture
    vkWriteDescriptorSet_array[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[3].pNext = NULL;
    vkWriteDescriptorSet_array[3].dstSet = vkDescriptorSet_ocean;
    vkWriteDescriptorSet_array[3].dstArrayElement = 0;
    vkWriteDescriptorSet_array[3].dstBinding = 3;
    vkWriteDescriptorSet_array[3].descriptorCount = 1;
    vkWriteDescriptorSet_array[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkWriteDescriptorSet_array[3].pBufferInfo = NULL;
    vkWriteDescriptorSet_array[3].pImageInfo = &vkDescriptorImageInfo_array[1];
    vkWriteDescriptorSet_array[3].pTexelBufferView = NULL;

    vkUpdateDescriptorSets(vkDevice, _ARRAYSIZE(vkWriteDescriptorSet_array), vkWriteDescriptorSet_array, 0, NULL);

    return vkResult;
}

VkResult Ocean::createPipeline()
{
    //* Code

    //! Vertex Input State
    VkVertexInputBindingDescription vkVertexInputBindingDescription_array[3];
    memset((void*)vkVertexInputBindingDescription_array, 0, sizeof(VkVertexInputBindingDescription) * _ARRAYSIZE(vkVertexInputBindingDescription_array));

    //! Position
    vkVertexInputBindingDescription_array[0].binding = 0;
    vkVertexInputBindingDescription_array[0].stride = sizeof(glm::vec4);
    vkVertexInputBindingDescription_array[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    
    //! Height
    vkVertexInputBindingDescription_array[1].binding = 1;
    vkVertexInputBindingDescription_array[1].stride = sizeof(float);
    vkVertexInputBindingDescription_array[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    //! Slope
    vkVertexInputBindingDescription_array[2].binding = 2;
    vkVertexInputBindingDescription_array[2].stride = sizeof(glm::vec2);
    vkVertexInputBindingDescription_array[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[3];
    memset((void*)vkVertexInputAttributeDescription_array, 0, sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));

    //! Position
    vkVertexInputAttributeDescription_array[0].binding = 0;
    vkVertexInputAttributeDescription_array[0].location = 0;
    vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    vkVertexInputAttributeDescription_array[0].offset = 0;

    //! Height
    vkVertexInputAttributeDescription_array[1].binding = 1;
    vkVertexInputAttributeDescription_array[1].location = 1;
    vkVertexInputAttributeDescription_array[1].format = VK_FORMAT_R32_SFLOAT;
    vkVertexInputAttributeDescription_array[1].offset = 0;

    //! Slope
    vkVertexInputAttributeDescription_array[2].binding = 2;
    vkVertexInputAttributeDescription_array[2].location = 2;
    vkVertexInputAttributeDescription_array[2].format = VK_FORMAT_R32G32_SFLOAT;
    vkVertexInputAttributeDescription_array[2].offset = 0;

    VkPipelineVertexInputStateCreateInfo vkPipelineVertexInputStateCreateInfo;
    memset((void*)&vkPipelineVertexInputStateCreateInfo, 0, sizeof(VkPipelineVertexInputStateCreateInfo));
    vkPipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vkPipelineVertexInputStateCreateInfo.pNext = NULL;
    vkPipelineVertexInputStateCreateInfo.flags = 0;
    vkPipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = _ARRAYSIZE(vkVertexInputBindingDescription_array);
    vkPipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = vkVertexInputBindingDescription_array;
    vkPipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = _ARRAYSIZE(vkVertexInputAttributeDescription_array);
    vkPipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = vkVertexInputAttributeDescription_array;

    //! Input Assembly State
    VkPipelineInputAssemblyStateCreateInfo vkPipelineInputAssemblyStateCreateInfo;
    memset((void*)&vkPipelineInputAssemblyStateCreateInfo, 0, sizeof(VkPipelineInputAssemblyStateCreateInfo));
    vkPipelineInputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    vkPipelineInputAssemblyStateCreateInfo.pNext = NULL;
    vkPipelineInputAssemblyStateCreateInfo.flags = 0;
    vkPipelineInputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    vkPipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = VK_TRUE;

    //! Rasterization State
    VkPipelineRasterizationStateCreateInfo vkPipelineRasterizationStateCreateInfo;
    memset((void*)&vkPipelineRasterizationStateCreateInfo, 0, sizeof(VkPipelineRasterizationStateCreateInfo));
    vkPipelineRasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    vkPipelineRasterizationStateCreateInfo.pNext = NULL;
    vkPipelineRasterizationStateCreateInfo.flags = 0;
    vkPipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    vkPipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_NONE;
    vkPipelineRasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    vkPipelineRasterizationStateCreateInfo.lineWidth = 1.0f;

    //! Color Blend State
    VkPipelineColorBlendAttachmentState vkPipelineColorBlendAttachmentState_array[1];
    memset((void*)vkPipelineColorBlendAttachmentState_array, 0, sizeof(VkPipelineColorBlendAttachmentState) * _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array));
    vkPipelineColorBlendAttachmentState_array[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    vkPipelineColorBlendAttachmentState_array[0].blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo vkPipelineColorBlendStateCreateInfo;
    memset((void*)&vkPipelineColorBlendStateCreateInfo, 0, sizeof(VkPipelineColorBlendStateCreateInfo));
    vkPipelineColorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    vkPipelineColorBlendStateCreateInfo.pNext = NULL;
    vkPipelineColorBlendStateCreateInfo.flags = 0;
    vkPipelineColorBlendStateCreateInfo.attachmentCount = _ARRAYSIZE(vkPipelineColorBlendAttachmentState_array);
    vkPipelineColorBlendStateCreateInfo.pAttachments = vkPipelineColorBlendAttachmentState_array;

    //! Viewport Scissor State
    VkPipelineViewportStateCreateInfo vkPipelineViewportStateCreateInfo;
    memset((void*)&vkPipelineViewportStateCreateInfo, 0, sizeof(VkPipelineViewportStateCreateInfo));
    vkPipelineViewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vkPipelineViewportStateCreateInfo.pNext = NULL;
    vkPipelineViewportStateCreateInfo.flags = 0;
    vkPipelineViewportStateCreateInfo.viewportCount = 1;    //* We can specify multiple viewports here
    vkPipelineViewportStateCreateInfo.scissorCount = 1;

    //! Viewport Info     
    memset((void*)&vkViewport, 0, sizeof(VkViewport));
    vkViewport.x = 0;
    vkViewport.y = 0;
    vkViewport.width = (float)vkExtent2D_swapchain.width;
    vkViewport.height = (float)vkExtent2D_swapchain.height;
    vkViewport.minDepth = 0.0f;
    vkViewport.maxDepth = 1.0f;

    vkPipelineViewportStateCreateInfo.pViewports = &vkViewport;

    //! Scissor Info
    memset((void*)&vkRect2D_scissor, 0, sizeof(VkRect2D));
    vkRect2D_scissor.offset.x = 0;
    vkRect2D_scissor.offset.y = 0;
    vkRect2D_scissor.extent.width = vkExtent2D_swapchain.width;
    vkRect2D_scissor.extent.height = vkExtent2D_swapchain.height;

    vkPipelineViewportStateCreateInfo.pScissors = &vkRect2D_scissor;

    //! Depth Stencil State !//
    VkPipelineDepthStencilStateCreateInfo vkPipelineDepthStencilCreateInfo;
    memset((void*)&vkPipelineDepthStencilCreateInfo, 0, sizeof(VkPipelineDepthStencilStateCreateInfo));
    vkPipelineDepthStencilCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    vkPipelineDepthStencilCreateInfo.flags = 0;
    vkPipelineDepthStencilCreateInfo.pNext = NULL;
    vkPipelineDepthStencilCreateInfo.depthTestEnable = VK_TRUE;
    vkPipelineDepthStencilCreateInfo.depthWriteEnable = VK_TRUE;
    vkPipelineDepthStencilCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    vkPipelineDepthStencilCreateInfo.depthBoundsTestEnable = VK_FALSE;
    vkPipelineDepthStencilCreateInfo.back.failOp = VK_STENCIL_OP_KEEP;
    vkPipelineDepthStencilCreateInfo.back.passOp = VK_STENCIL_OP_KEEP;
    vkPipelineDepthStencilCreateInfo.back.compareOp = VK_COMPARE_OP_ALWAYS;
    vkPipelineDepthStencilCreateInfo.stencilTestEnable = VK_FALSE;
    vkPipelineDepthStencilCreateInfo.front = vkPipelineDepthStencilCreateInfo.back;

    //! Dynamic State !//

    //! Multi-Sample State
    VkPipelineMultisampleStateCreateInfo vkPipelineMultisampleStateCreateInfo;
    memset((void*)&vkPipelineMultisampleStateCreateInfo, 0, sizeof(VkPipelineMultisampleStateCreateInfo));
    vkPipelineMultisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    vkPipelineMultisampleStateCreateInfo.pNext = NULL;
    vkPipelineMultisampleStateCreateInfo.flags = 0;
    vkPipelineMultisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    //! Shader Stage State
    VkPipelineShaderStageCreateInfo vkPipelineShaderStageCreateInfo_array[2];
    memset((void*)vkPipelineShaderStageCreateInfo_array, 0, sizeof(VkPipelineShaderStageCreateInfo) * _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array));

    //* Vertex Shader
    vkPipelineShaderStageCreateInfo_array[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[0].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[0].flags = 0;
    vkPipelineShaderStageCreateInfo_array[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    vkPipelineShaderStageCreateInfo_array[0].module = vkShaderModule_vertex_shader;
    vkPipelineShaderStageCreateInfo_array[0].pName = "main";
    vkPipelineShaderStageCreateInfo_array[0].pSpecializationInfo = NULL;

    //* Fragment Shader
    vkPipelineShaderStageCreateInfo_array[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[1].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[1].flags = 0;
    vkPipelineShaderStageCreateInfo_array[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkPipelineShaderStageCreateInfo_array[1].module = vkShaderModule_fragment_shader;
    vkPipelineShaderStageCreateInfo_array[1].pName = "main";
    vkPipelineShaderStageCreateInfo_array[1].pSpecializationInfo = NULL;

    //! Tessellation State !//

    //! As pipelines are created from pipeline caches, we will create VkPipelineCache Object
    VkPipelineCacheCreateInfo vkPipelineCacheCreateInfo;
    memset((void*)&vkPipelineCacheCreateInfo, 0, sizeof(VkPipelineCacheCreateInfo));
    vkPipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkPipelineCacheCreateInfo.pNext = NULL;
    vkPipelineCacheCreateInfo.flags = 0;

    VkPipelineCache vkPipelineCache = VK_NULL_HANDLE;
    vkResult = vkCreatePipelineCache(vkDevice, &vkPipelineCacheCreateInfo, NULL, &vkPipelineCache);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Succeeded For Ocean\n", __func__);

    //! Create actual Graphics Pipeline
    VkGraphicsPipelineCreateInfo vkGraphicsPipelineCreateInfo;
    memset((void*)&vkGraphicsPipelineCreateInfo, 0, sizeof(VkGraphicsPipelineCreateInfo));
    vkGraphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    vkGraphicsPipelineCreateInfo.pNext = NULL;
    vkGraphicsPipelineCreateInfo.flags = 0;
    vkGraphicsPipelineCreateInfo.pVertexInputState = &vkPipelineVertexInputStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pInputAssemblyState = &vkPipelineInputAssemblyStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pRasterizationState = &vkPipelineRasterizationStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pColorBlendState = &vkPipelineColorBlendStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pViewportState = &vkPipelineViewportStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pDepthStencilState = &vkPipelineDepthStencilCreateInfo;
    vkGraphicsPipelineCreateInfo.pDynamicState = NULL;
    vkGraphicsPipelineCreateInfo.pMultisampleState = &vkPipelineMultisampleStateCreateInfo;
    vkGraphicsPipelineCreateInfo.stageCount = _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array);
    vkGraphicsPipelineCreateInfo.pStages = vkPipelineShaderStageCreateInfo_array;
    vkGraphicsPipelineCreateInfo.pTessellationState = NULL;
    vkGraphicsPipelineCreateInfo.layout = vkPipelineLayout_ocean;
    vkGraphicsPipelineCreateInfo.renderPass = vkRenderPass;
    vkGraphicsPipelineCreateInfo.subpass = 0;
    vkGraphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    vkGraphicsPipelineCreateInfo.basePipelineIndex = 0;

    vkResult = vkCreateGraphicsPipelines(vkDevice, vkPipelineCache, 1, &vkGraphicsPipelineCreateInfo, NULL, &vkPipeline_ocean);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateGraphicsPipelines() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateGraphicsPipelines() Succeeded For Ocean\n", __func__);

    //* Destroy Pipeline Cache
    if (vkPipelineCache)
    {
        vkDestroyPipelineCache(vkDevice, vkPipelineCache, NULL);
        vkPipelineCache = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyPipelineCache() Succeeded For Ocean\n", __func__);
    }

    return vkResult;
}

VkResult Ocean::createShaders()
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Vertex Shader
    //! ---------------------------------------------------------------------------------------------------------------------------
    //* Step - 6
    const char* szFileName = "Bin/Ocean.vert.spv";
    FILE* fp = NULL;
    size_t size;

    fp = fopen(szFileName, "rb");
    if (fp == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Open SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Succeeded In Opening SPIR-V Shader File : %s\n", __func__, szFileName);

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    if (size == 0)
    {
        fprintf(gpFile, "%s() => Empty SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    fseek(fp, 0L, SEEK_SET);

    char* shaderData = (char*)malloc(size * sizeof(char));
    if (shaderData == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For shaderData !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    size_t retVal = fread(shaderData, size, 1, fp);
    if (retVal != 1)
    {
        fprintf(gpFile, "%s() => Failed To Read From SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Successfully Read Shader From SPIR-V Shader File : %s\n", __func__, szFileName);

    if (fp)
    {
        fclose(fp);
        fp = NULL;
        fprintf(gpFile, "%s() => Closed SPIR-V File : %s\n", __func__, szFileName);
    }

    //* Step - 7
    VkShaderModuleCreateInfo vkShaderModuleCreateInfo;
    memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
    vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0; //! Reserved, must be 0
    vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
    vkShaderModuleCreateInfo.codeSize = size;

    //* Step - 8
    vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_vertex_shader);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateShaderModule() Failed For Vertex Shader : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateShaderModule() Succeeded For Vertex Shader\n", __func__);

    //* Step - 9
    if (shaderData)
    {
        free(shaderData);
        shaderData = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For shaderData\n", __func__);
    }

    fprintf(gpFile, "%s() => Vertex Shader Module Successfully Created\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------

    //! Fragment Shader
    //! ---------------------------------------------------------------------------------------------------------------------------
    szFileName = "Bin/Ocean.frag.spv";

    fp = fopen(szFileName, "rb");
    if (fp == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Open SPIR-V Shader File :  %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Succeeded In Opening SPIR-V Shader File : %s\n", __func__, szFileName);

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    if (size == 0)
    {
        fprintf(gpFile, "%s() => Empty SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    fseek(fp, 0L, SEEK_SET);

    shaderData = (char*)malloc(size * sizeof(char));
    if (shaderData == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For shaderData !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    retVal = fread(shaderData, size, 1, fp);
    if (retVal != 1)
    {
        fprintf(gpFile, "%s() => Failed To Read From SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Successfully Read Shader From SPIR-V Shader File : %s\n", __func__, szFileName);

    if (fp)
    {
        fclose(fp);
        fp = NULL;
        fprintf(gpFile, "%s() => Closed SPIR-V File : %s\n", __func__, szFileName);
    }

    //* Step - 7
    memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
    vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0; //! Reserved, must be 0
    vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
    vkShaderModuleCreateInfo.codeSize = size;

    //* Step - 8
    vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_fragment_shader);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateShaderModule() Failed For Fragment Shader : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateShaderModule() Succeeded For Fragment Shader\n", __func__);

    //* Step - 9
    if (shaderData)
    {
        free(shaderData);
        shaderData = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For shaderData\n", __func__);
    }

    fprintf(gpFile, "%s() => Fragment Shader Module Successfully Created\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult Ocean::resize(int width, int height)
{
    //? DESTROY
    //?--------------------------------------------------------------------------------------------------
    //* Wait for device to complete in-hand tasks
    if (vkDevice)
        vkDeviceWaitIdle(vkDevice);

    //* Destroy PipelineLayout
    if (vkPipelineLayout_ocean)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_ocean, NULL);
        vkPipelineLayout_ocean = VK_NULL_HANDLE;
    }

    //* Destroy Pipeline
    if (vkPipeline_ocean)
    {
        vkDestroyPipeline(vkDevice, vkPipeline_ocean, NULL);
        vkPipeline_ocean = VK_NULL_HANDLE;
    }
    //?--------------------------------------------------------------------------------------------------

    //? RECREATE FOR RESIZE
    //?--------------------------------------------------------------------------------------------------
    //* Create Pipeline Layout
    vkResult = createPipelineLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "OCEAN::%s() => createPipelineLayout() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //* Create Pipeline
    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "OCEAN::%s() => createPipeline() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    //?--------------------------------------------------------------------------------------------------

    return vkResult;
}

void Ocean::buildCommandBuffers(VkCommandBuffer& commandBuffer)
{
    //! Bind with Pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline_ocean);

    //! Bind the Descriptor Set to the Pipeline
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        vkPipelineLayout_ocean,
        0,
        1,
        &vkDescriptorSet_ocean,
        0,
        NULL
    );

    //! Bind with Position Data
    VkDeviceSize vkDeviceSize_offset_array[1];
    memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));
    vkCmdBindVertexBuffers(
        commandBuffer,
        0,
        1,
        &vertexData_position.vkBuffer,
        vkDeviceSize_offset_array
    );

    //! Bind with Height Data
    memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));
    vkCmdBindVertexBuffers(
        commandBuffer,
        1,
        1,
        &vertexData_height.vkBuffer,
        vkDeviceSize_offset_array
    );

    //! Bind with Slope Data
    memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));
    vkCmdBindVertexBuffers(
        commandBuffer,
        2,
        1,
        &vertexData_slope.vkBuffer,
        vkDeviceSize_offset_array
    );

    //! Bind with Index Buffer
    vkCmdBindIndexBuffer(
        commandBuffer,
        indexData.vkBuffer,
        0,
        VK_INDEX_TYPE_UINT32
    );

    //! Vulkan Drawing Function
    vkCmdDrawIndexed(
        commandBuffer,
        indicesSize,
        1,              //* Count of geometry instances
        0,              //* Starting offset of index buffer
        0,              //* Starting offset of vertex buffer
        0               //* Nth instance
    );
}

void Ocean::update()
{
    fTime += waveSpeed;

    dim3 block(32, 32, 1);
    dim3 grid(cudaRoundUpDivision(MESH_SIZE, block.x), cudaRoundUpDivision(MESH_SIZE, block.y), 1);

    generateSpectrumKernel<<<grid, block>>>(
        device_h_twiddle_0, 
        device_height, 
        SPECTRUM_SIZE_WIDTH, 
        MESH_SIZE, 
        MESH_SIZE, 
        fTime, 
        patchSize
    );

    cudaDeviceSynchronize();

    cufftResult fftResult = cufftExecC2C(plan2d, device_height, device_height, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For device_height : %d !!!\n", __func__, fftResult);
    }

    updateHeightMapKernel<<<grid, block>>>(
        (float*)heightPtr, 
        device_height, 
        MESH_SIZE
    );

    cudaDeviceSynchronize();

    calculateSlopeKernel<<<grid, block>>>(
        (float*)heightPtr, 
        (float2*)slopePtr, 
        MESH_SIZE, 
        MESH_SIZE
    );

    cudaDeviceSynchronize();

    updateUniformBuffer();
}

void Ocean::update(glm::mat4 cameraMatrix)
{
    useCamera = true;
    cameraViewMatrix = cameraMatrix;

    fTime += waveSpeed;
    
    dim3 block(32, 32, 1);
    dim3 grid(cudaRoundUpDivision(MESH_SIZE, block.x), cudaRoundUpDivision(MESH_SIZE, block.y), 1);

    generateSpectrumKernel<<<grid, block>>>(
        device_h_twiddle_0, 
        device_height, 
        SPECTRUM_SIZE_WIDTH, 
        MESH_SIZE, 
        MESH_SIZE, 
        fTime, 
        patchSize
    );

    cudaDeviceSynchronize();

    cufftResult fftResult = cufftExecC2C(plan2d, device_height, device_height, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For device_height : %d !!!\n", __func__, fftResult);
    }

    updateHeightMapKernel<<<grid, block>>>(
        (float*)heightPtr, 
        device_height, 
        MESH_SIZE
    );

    cudaDeviceSynchronize();

    calculateSlopeKernel<<<grid, block>>>(
        (float*)heightPtr, 
        (float2*)slopePtr, 
        MESH_SIZE, 
        MESH_SIZE
    );

    cudaDeviceSynchronize();

    updateUniformBuffer();

    updateUniformBuffer();
}

Ocean::~Ocean()
{
    if (vkDevice)
        vkDeviceWaitIdle(vkDevice);

    if (vkPipelineLayout_ocean)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_ocean, NULL);
        vkPipelineLayout_ocean = VK_NULL_HANDLE;
    }

    if (vkPipeline_ocean)
    {
        vkDestroyPipeline(vkDevice, vkPipeline_ocean, NULL);
        vkPipeline_ocean = VK_NULL_HANDLE;
    }

    if (vkDescriptorPool_ocean)
    {
        vkDestroyDescriptorPool(vkDevice, vkDescriptorPool_ocean, NULL);
        vkDescriptorPool_ocean = VK_NULL_HANDLE;
        vkDescriptorSet_ocean = VK_NULL_HANDLE;
    }

    if (vkPipelineLayout_ocean)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_ocean, NULL);
        vkPipelineLayout_ocean = VK_NULL_HANDLE;
    }

    if (vkDescriptorSetLayout_ocean)
    {
        vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout_ocean, NULL);
        vkDescriptorSetLayout_ocean = VK_NULL_HANDLE;
    }

    if (vkShaderModule_fragment_shader)
    {
        vkDestroyShaderModule(vkDevice, vkShaderModule_fragment_shader, NULL);
        vkShaderModule_fragment_shader = VK_NULL_HANDLE;
    }

    if (vkShaderModule_vertex_shader)
    {
        vkDestroyShaderModule(vkDevice, vkShaderModule_vertex_shader, NULL);
        vkShaderModule_vertex_shader = VK_NULL_HANDLE;
    }

    //* Destroy Uniform Buffer
    if (uniformData_ocean_surface.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData_ocean_surface.vkDeviceMemory, NULL);
        uniformData_ocean_surface.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (uniformData_ocean_surface.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData_ocean_surface.vkBuffer, NULL);
        uniformData_ocean_surface.vkBuffer = VK_NULL_HANDLE;
    }

    if (uniformData_mvp.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData_mvp.vkDeviceMemory, NULL);
        uniformData_mvp.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (uniformData_mvp.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData_mvp.vkBuffer, NULL);
        uniformData_mvp.vkBuffer = VK_NULL_HANDLE;
    }

    //* Texture Related
    if (foamIntensityTexture.vkSampler)
    {
        vkDestroySampler(vkDevice, foamIntensityTexture.vkSampler, NULL);
        foamIntensityTexture.vkSampler = VK_NULL_HANDLE;
    }

    if (foamIntensityTexture.vkImageView)
    {
        vkDestroyImageView(vkDevice, foamIntensityTexture.vkImageView, NULL);
        foamIntensityTexture.vkImageView = NULL;
    }

    if (foamIntensityTexture.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, foamIntensityTexture.vkDeviceMemory, NULL);
        foamIntensityTexture.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (foamIntensityTexture.vkImage)
    {
        vkDestroyImage(vkDevice, foamIntensityTexture.vkImage, NULL);
        foamIntensityTexture.vkImage = NULL;
    }

    if (bubblesTexture.vkSampler)
    {
        vkDestroySampler(vkDevice, bubblesTexture.vkSampler, NULL);
        bubblesTexture.vkSampler = VK_NULL_HANDLE;
    }

    if (bubblesTexture.vkImageView)
    {
        vkDestroyImageView(vkDevice, bubblesTexture.vkImageView, NULL);
        bubblesTexture.vkImageView = NULL;
    }

    if (bubblesTexture.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, bubblesTexture.vkDeviceMemory, NULL);
        bubblesTexture.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (bubblesTexture.vkImage)
    {
        vkDestroyImage(vkDevice, bubblesTexture.vkImage, NULL);
        bubblesTexture.vkImage = NULL;
    }

    if (indexData.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, indexData.vkDeviceMemory, NULL);
        indexData.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (indexData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, indexData.vkBuffer, NULL);
        indexData.vkBuffer = VK_NULL_HANDLE;
    }

    if (cudaExternalMemory_slope)
    {
        cudaDestroyExternalMemory(cudaExternalMemory_slope);
        cudaExternalMemory_slope = nullptr;
    }

    if (slopePtr)
    {
        cudaFree(slopePtr);
        slopePtr = nullptr;
    }

    if (vertexData_slope.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_slope.vkDeviceMemory, NULL);
        vertexData_slope.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (vertexData_slope.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_slope.vkBuffer, NULL);
        vertexData_slope.vkBuffer = VK_NULL_HANDLE;
    }

    if (cudaExternalMemory_height)
    {
        cudaDestroyExternalMemory(cudaExternalMemory_height);
        cudaExternalMemory_height = NULL;
    }

    if (heightPtr)
    {
        cudaFree(heightPtr);
        heightPtr = nullptr;
    }

    if (vertexData_height.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_height.vkDeviceMemory, NULL);
        vertexData_height.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (vertexData_height.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_height.vkBuffer, NULL);
        vertexData_height.vkBuffer = VK_NULL_HANDLE;
    }

    if (vertexData_position.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_position.vkDeviceMemory, NULL);
        vertexData_position.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (vertexData_position.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_position.vkBuffer, NULL);
        vertexData_position.vkBuffer = VK_NULL_HANDLE;
    }

    if (device_slope)
    {
        cudaFree(device_slope);
        device_slope = nullptr;
    }

    if (device_height)
    {
        cudaFree(device_height);
        device_height = nullptr;
    }

    if (device_h_twiddle_0)
    {
        cudaFree(device_h_twiddle_0);
        device_h_twiddle_0 = nullptr;
    }

    if (host_h_twiddle_0)
    {
        free(host_h_twiddle_0);
        host_h_twiddle_0 = nullptr;
    }

    if (plan2d)
    {
        cufftDestroy(plan2d);
        plan2d = 0;
    }
}


//! FFT Tessendorf Related Functions
float Ocean::phillipsSpectrum(float kx, float ky)
{
    // Code
    float k_squared = kx * kx + ky * ky;
    if (k_squared == 0.0f)
        return (0.0f);

    float L = windSpeed * windSpeed / gravitationalConstant;

    float k_x = kx / sqrtf(k_squared);
    float k_y = ky / sqrtf(k_squared);
    float w_dot_k = k_x * cosf(windDirection) + k_y * sinf(windDirection);

    float phillips = waveScaleFactor * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

    if (w_dot_k < 0.0f)
        phillips *= waveDirectionStrength;

    return phillips;
}

float Ocean::urand()
{
    return rand() / (float)RAND_MAX;
}

float Ocean::gaussianDistribution()
{
    float u1 = urand();
    float u2 = urand();

    if (u1 < 1e-6f)
        u1 = 1e-6f;

    return sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2);
}

void Ocean::generateInitialSpectrum()
{
    for (unsigned int y = 0; y <= MESH_SIZE; y++)
    {
        for (unsigned int x = 0; x <= MESH_SIZE; x++)
        {
            float kx = (-(int)MESH_SIZE / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
            float ky = (-(int)MESH_SIZE / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

            float phillips = sqrtf(phillipsSpectrum(kx, ky));

            if (kx == 0.0f && ky == 0.0f)
                phillips = 0.0f;

            float Er = gaussianDistribution();
            float Ei = gaussianDistribution();

            float h0_re = Er * phillips * CUDART_SQRT_HALF_F;
            float h0_im = Ei * phillips * CUDART_SQRT_HALF_F;

            int i = y * SPECTRUM_SIZE_WIDTH + x;
            host_h_twiddle_0[i].x = h0_re;
            host_h_twiddle_0[i].y = h0_im;
        }
    }
}
