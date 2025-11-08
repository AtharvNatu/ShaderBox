#include "Ocean.hpp"

const float gravitationalConstant = 9.81f; // gravitational constant
const float waveScaleFactor = 1e-7f;       // wave scale factor
const float patchSize = 100;               // patch size
float windSpeed = 100.0f;
float windDir = CUDART_PI_F / 3.0f;
float dirDepend = 0.07f;

//! CUDA Kernels
#include <cufft.h>
#include <math_constants.h>

int cuda_iDivUp(int a, int b)
{
  return ((a + (b - 1)) / b);
}

// complex math functions
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

__device__ float2 complex_mult(float2 ab, float2 cd)
{
  return (make_float2(ab.x * cd.x - ab.y * cd.y, ab.x * cd.y + ab.y * cd.x));
}

__global__ void generateSpectrumKernel(float2 *h0, float2 *ht, unsigned int in_width, unsigned int out_width, unsigned int out_height, float t, float patchSize)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int in_index = y * in_width + x;
  unsigned int in_mindex = (out_height - y) * in_width + (out_width - x);
  unsigned int out_index = y * out_width + x;

  // calculate wave vector
  float2 k;
  k.x = (-(int)out_width / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
  k.y = (-(int)out_width / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

  // calculate dispersion w(k)
  float k_len = sqrtf(k.x * k.x + k.y * k.y);
  float w = sqrtf(9.81f * k_len);

  if ((x < out_width) && (y < out_height))
  {
    float2 h0_k = h0[in_index];
    float2 h0_mk = h0[in_mindex];

    // output frequency-space complex values
    ht[out_index] = complex_add(complex_mult(h0_k, complex_exp(w * t)), complex_mult(conjugate(h0_mk), complex_exp(-w * t)));
  }
}

__global__ void updateHeightmapKernel(float *heightMap, float2 *ht, unsigned int width)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].x * sign_correction;
}

__global__ void updateHeightmapKernel_y(float *heightMap, float2 *ht, unsigned int width)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int i = y * width + x;

  float sign_correction = ((x + y) & 0x01) ? -1.0f : 1.0f;

  heightMap[i] = ht[i].y * sign_correction;
}

__global__ void calculateSlopeKernel(float *h, float2 *slopeOut, unsigned int width, unsigned int height)
{
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

// wrapper functions
extern "C" void cudaGenerateSpectrumKernel(float2 *d_h0, float2 *d_ht, unsigned int in_width, unsigned int out_width, unsigned int out_height, float animTime, float patchSize)
{
  dim3 block(8, 8, 1);
  dim3 grid(cuda_iDivUp(out_width, block.x), cuda_iDivUp(out_height, block.y), 1);

  generateSpectrumKernel<<<grid, block>>>(d_h0, d_ht, in_width, out_width, out_height, animTime, patchSize);
}

extern "C" void cudaUpdateHeightmapKernel(float *d_heightMap, float2 *d_ht, unsigned int width, unsigned int height)
{
  dim3 block(8, 8, 1);
  dim3 grid(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);

  updateHeightmapKernel<<<grid, block>>>(d_heightMap, d_ht, width);
}

extern "C" void cudaCalculateSlopeKernel(float *hptr, float2 *slopeOut, unsigned int width, unsigned int height)
{
  dim3 block(8, 8, 1);
  dim3 grid2(cuda_iDivUp(width, block.x), cuda_iDivUp(height, block.y), 1);

  calculateSlopeKernel<<<grid2, block>>>(hptr, slopeOut, width, height);
}


Ocean::Ocean()
{
    // Code
    omega_hat = glm::normalize(omega_vec);
    meshSize = sizeof(glm::vec3) * N * M;
    kNum = N * M;
}

VkResult Ocean::initialize()
{

    bool status = initializeDeviceData();
    if (!status)
        fprintf(gpFile, "%s() => initializeDeviceData() Failed For Ocean : %d !!!\n", __func__);
    else
        fprintf(gpFile, "%s() => initializeDeviceData() Succeeded For Ocean\n", __func__);
        
    vkResult = createBuffers();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createBuffers() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createBuffers() Succeeded For Ocean\n", __func__);

    vkResult = createUniformBuffer();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createUniformBuffer() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createUniformBuffer() Succeeded For Ocean\n", __func__);

    vkResult = createDescriptorSetLayout();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Succeeded For Ocean\n", __func__);

    vkResult = createPipelineLayout();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createPipelineLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createPipelineLayout() Succeeded For Ocean\n", __func__);

    vkResult = createDescriptorPool();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorPool() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createDescriptorPool() Succeeded For Ocean\n", __func__);

    vkResult = createDescriptorSet();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorSet() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createDescriptorSet() Succeeded For Ocean\n", __func__);

    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createPipeline() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createPipeline() Succeeded For Ocean\n", __func__);

    return vkResult;
}

bool Ocean::initializeDeviceData()
{
    // Code

    //* Create Plan 2D Complex-To-Complex
    fftResult = cufftPlan2d(&plan2d, meshSizeLimit, meshSizeLimit, CUFFT_C2C);
    if (fftResult != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "%s() => cufftPlan2d() Failed For Ocean !!!\n", __func__);
        return false;
    }

    size_t spectrumSize = SPECTRUM_SIZE_WIDTH * SPECTRUM_SIZE_HEIGHT * sizeof(float2);

    cudaResult = cudaMalloc(&device_h_twiddle_0, spectrumSize);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    host_h_twiddle_0 = (float2*)malloc(spectrumSize);
    if (host_h_twiddle_0 == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For host_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    generate_initial_spectrum();
    
    cudaResult = cudaMemcpy(device_h_twiddle_0, host_h_twiddle_0, spectrumSize, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMemcpy() Failed For device_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    size_t outputSize = meshSizeLimit * meshSizeLimit * sizeof(float2);

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

    heightSize = meshSizeLimit * meshSizeLimit * sizeof(float);
    slopeSize = meshSizeLimit * meshSizeLimit * sizeof(float2);
    
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
    VkBufferCreateInfo vkBufferCreateInfo;
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
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_height.vkBuffer, &vkMemoryRequirements);

    VkExportMemoryAllocateInfo vkExportMemoryAllocateInfo;
    memset((void*)&vkExportMemoryAllocateInfo, 0, sizeof(VkExportMemoryAllocateInfo));
    vkExportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    vkExportMemoryAllocateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    //* Step - 8
    VkMemoryAllocateInfo vkMemoryAllocateInfo;
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
    // cuExtMemoryHandleDesc.size = (size_t)vkMemoryRequirements.size;
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
    // cuExtMemoryHandleDesc.size = (size_t)vkMemoryRequirements.size;
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

    //! Vertex Position Data
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_position, 0, sizeof(BufferData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = meshSizeLimit * meshSizeLimit * 4 * sizeof(float);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_position.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_position.vkBuffer, &vkMemoryRequirements);

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

    float *positionBase = new float[meshSizeLimit * meshSizeLimit * 4];
    float *position = positionBase;

    for (int y = 0; y < meshSizeLimit; y++)
    {
        for (int x = 0; x < meshSizeLimit; x++)
        {
            float u = x / (float)(meshSizeLimit - 1);
            float v = y / (float)(meshSizeLimit - 1);

            *position++ = u * 2.0f - 1.0f;
            *position++ = 0.0f;

            *position++ = v * 2.0f - 1.0f;
            *position++ = 1.0f;
        }
    }

    //* Step - 12
    memcpy(data, positionBase, meshSizeLimit * meshSizeLimit * 4 * sizeof(float));

    //* Step - 13
    vkUnmapMemory(vkDevice, vertexData_position.vkDeviceMemory);

    delete[] positionBase;
    positionBase = nullptr;
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! Index Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //* Generate Indices
    indexSize = ((meshSizeLimit * 2) + 2) * (meshSizeLimit - 1);
    unsigned int* indices = new unsigned int[indexSize];
    unsigned int* pIndex = indices;

    // for (int y = 0; y < meshSizeLimit - 1; y++)
    // {
    //     for (int x = 0; x < meshSizeLimit; x++)
    //     {
    //         *pIndex++ = y * meshSizeLimit + x;
    //         *pIndex++ = (y + 1) * meshSizeLimit + x;
    //     }
    //     *pIndex++ = (y + 1) * meshSizeLimit + (meshSizeLimit - 1);
    //     *pIndex++ = (y + 1) * meshSizeLimit;
    // }
    
    for (int y = 0; y < meshSizeLimit - 1; y++) {
    if (y > 0) {
        // Degenerate start
        *pIndex++ = y * meshSizeLimit;
    }
    for (int x = 0; x < meshSizeLimit; x++) {
        *pIndex++ = y * meshSizeLimit + x;
        *pIndex++ = (y + 1) * meshSizeLimit + x;
    }
    if (y < meshSizeLimit - 2) {
        // Degenerate end
        *pIndex++ = (y + 1) * meshSizeLimit + (meshSizeLimit - 1);
    }
}


    
    //* Step - 4
    memset((void*)&indexData, 0, sizeof(BufferData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = indexSize * sizeof(unsigned int);
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
    memcpy(data, indices, indexSize * sizeof(unsigned int));
    delete[] indices;
    indices = nullptr;

    //* Step - 13
    vkUnmapMemory(vkDevice, indexData.vkDeviceMemory);

    //! ---------------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult Ocean::getMemoryWin32HandleFunction(void)
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

VkResult Ocean::createUniformBuffer(void)
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
    vkBufferCreateInfo.size = sizeof(WaterUBO);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData_water, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData_water.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Water Surface Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Water Surface Uniform Data\n", __func__);

    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData_water.vkBuffer, &vkMemoryRequirements);

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

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData_water.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Water Surface Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Water Surface Uniform Data\n", __func__);

    vkResult = vkBindBufferMemory(vkDevice, uniformData_water.vkBuffer, uniformData_water.vkDeviceMemory, 0);
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

    translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.5f, -3.5f));
    scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 1.0f));
    mvpData.modelMatrix = translationMatrix * scaleMatrix;
    mvpData.viewMatrix = glm::mat4(1.0f);
    
    glm::mat4 perspectiveProjectionMatrix = glm::mat4(1.0f);
    perspectiveProjectionMatrix = glm::perspective(
        glm::radians(45.0f),
        (float)winWidth / (float)winHeight,
        0.1f,
        100.0f
    );
    //! 2D Matrix with Column Major (Like OpenGL)
    perspectiveProjectionMatrix[1][1] = perspectiveProjectionMatrix[1][1] * (-1.0f);
    mvpData.projectionMatrix = perspectiveProjectionMatrix;

    WaterUBO waterUBO;
    memset((void*)&waterUBO, 0, sizeof(WaterUBO));

    waterUBO.heightScale = 0.25f;
    waterUBO.choppiness = 1.0f;
    waterUBO.size = glm::vec2((float)meshSizeLimit, (float)meshSizeLimit);
    waterUBO.deepColor = glm::vec4(0.0f, 0.1f, 0.5f, 1.0f);
    waterUBO.shallowColor = glm::vec4(0.1f, 0.3f, 0.3f, 1.0f);
    waterUBO.skyColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    waterUBO.lightDirection = glm::vec4(-0.45f, 2.1f, -3.5f, 0.0f);

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
    vkResult = vkMapMemory(vkDevice, uniformData_water.vkDeviceMemory, 0, sizeof(WaterUBO), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer (WaterUBO) : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &waterUBO, sizeof(WaterUBO));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData_water.vkDeviceMemory);

    return vkResult;
}

VkResult Ocean::createDescriptorSetLayout(void)
{
    //! Initialize VkDescriptorSetLayoutBinding
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding_array[2];
    memset((void*)vkDescriptorSetLayoutBinding_array, 0, sizeof(VkDescriptorSetLayoutBinding) * _ARRAYSIZE(vkDescriptorSetLayoutBinding_array));

    //! Vertex UBO
    vkDescriptorSetLayoutBinding_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[0].binding = 0;   //! Mapped with layout(binding = 0) in vertex shader
    vkDescriptorSetLayoutBinding_array[0].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    vkDescriptorSetLayoutBinding_array[0].pImmutableSamplers = NULL;

    //! Water Surface UBO
    vkDescriptorSetLayoutBinding_array[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[1].binding = 1;   //! Mapped with layout(binding = 1) in fragment shader
    vkDescriptorSetLayoutBinding_array[1].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    vkDescriptorSetLayoutBinding_array[1].pImmutableSamplers = NULL;

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

VkResult Ocean::createPipelineLayout(void)
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

VkResult Ocean::createDescriptorPool(void)
{
    // Code

    //* Vulkan expects decriptor pool size before creating actual descriptor pool
    VkDescriptorPoolSize vkDescriptorPoolSize_array[1];
    memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * _ARRAYSIZE(vkDescriptorPoolSize_array));

    vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorPoolSize_array[0].descriptorCount = 2;

    //* Create the pool
    VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
    memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkDescriptorPoolCreateInfo.pNext = NULL;
    vkDescriptorPoolCreateInfo.flags = 0;
    vkDescriptorPoolCreateInfo.poolSizeCount = _ARRAYSIZE(vkDescriptorPoolSize_array);
    vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSize_array;
    vkDescriptorPoolCreateInfo.maxSets = 2;

    vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool_ocean);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Succeeded For Ocean\n", __func__);

    return vkResult;
}

VkResult Ocean::createDescriptorSet(void)
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
    vkDescriptorBufferInfo_array[1].buffer = uniformData_water.vkBuffer;
    vkDescriptorBufferInfo_array[1].offset = 0;
    vkDescriptorBufferInfo_array[1].range = sizeof(WaterUBO);

    /* Update above descriptor set directly to the shader
    There are 2 ways :-
        1) Writing directly to the shader
        2) Copying from one shader to another shader
    */
    VkWriteDescriptorSet vkWriteDescriptorSet_array[2];
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

    vkUpdateDescriptorSets(vkDevice, _ARRAYSIZE(vkWriteDescriptorSet_array), vkWriteDescriptorSet_array, 0, NULL);

    return vkResult;
}

VkResult Ocean::createPipeline(void)
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
    vkPipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

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
        indexSize,
        1,              //* Count of geometry instances
        0,              //* Starting offset of index buffer
        0,              //* Starting offset of vertex buffer
        0               //* Nth instance
    );
}

void Ocean::update()
{
    fTime += waveSpeed;
    
    cudaGenerateSpectrumKernel(device_h_twiddle_0, device_height, spectrumW, meshSizeLimit, meshSizeLimit, fTime, patchSize);

    cudaDeviceSynchronize();

    cufftResult fftResult = cufftExecC2C(plan2d, device_height, device_height, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For device_height : %d !!!\n", __func__, fftResult);
    }

    cudaUpdateHeightmapKernel((float*)heightPtr, device_height, meshSizeLimit, meshSizeLimit);

    cudaDeviceSynchronize();

    cudaCalculateSlopeKernel((float*)heightPtr, (float2*)slopePtr, meshSizeLimit, meshSizeLimit);

    cudaDeviceSynchronize();

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

    //* Destroy Uniform Buffer
    if (uniformData_water.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData_water.vkDeviceMemory, NULL);
        uniformData_water.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For uniformData_water.vkDeviceMemory\n", __func__);
    }

    if (uniformData_water.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData_water.vkBuffer, NULL);
        uniformData_water.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succedded For uniformData_water.vkBuffer\n", __func__);
    }

    if (uniformData_mvp.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData_mvp.vkDeviceMemory, NULL);
        uniformData_mvp.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For uniformData_mvp.vkDeviceMemory\n", __func__);
    }

    if (uniformData_mvp.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData_mvp.vkBuffer, NULL);
        uniformData_mvp.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succedded For uniformData_mvp.vkBuffer\n", __func__);
    }

    if (cudaExternalMemory_slope)
    {
        if (cudaDestroyExternalMemory(cudaExternalMemory_slope) == cudaSuccess)
            fprintf(gpFile, "%s() => cudaDestroyExternalMemory() Succedded For cudaExternalMemory_slope\n", __func__);
        cudaExternalMemory_slope = NULL;
        slopePtr = NULL;
    }

    if (cudaExternalMemory_height)
    {
        if (cudaDestroyExternalMemory(cudaExternalMemory_height) == cudaSuccess)
            fprintf(gpFile, "%s() => cudaDestroyExternalMemory() Succedded For cudaExternalMemory_height\n", __func__);
        cudaExternalMemory_height = NULL;
        heightPtr = NULL;
    }

    //* Step - 14 of Vertex Buffer
    if (indexData.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, indexData.vkDeviceMemory, NULL);
        indexData.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For indexData.vkDeviceMemory\n", __func__);
    }

    if (indexData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, indexData.vkBuffer, NULL);
        indexData.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For indexData.vkBuffer\n", __func__);
    }

    if (vertexData_position.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_position.vkDeviceMemory, NULL);
        vertexData_position.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_position.vkDeviceMemory\n", __func__);
    }

    if (vertexData_position.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_position.vkBuffer, NULL);
        vertexData_position.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_position.vkBuffer\n", __func__);
    }

    if (vertexData_slope.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_slope.vkDeviceMemory, NULL);
        vertexData_slope.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_slope.vkDeviceMemory\n", __func__);
    }

    if (vertexData_slope.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_slope.vkBuffer, NULL);
        vertexData_slope.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_slope.vkBuffer\n", __func__);
    }

    if (vertexData_height.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_height.vkDeviceMemory, NULL);
        vertexData_height.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_height.vkDeviceMemory\n", __func__);
    }

    if (vertexData_height.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_height.vkBuffer, NULL);
        vertexData_height.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_height.vkBuffer\n", __func__);
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

    if (host_h_twiddle_0)
    {
        free(host_h_twiddle_0);
        host_h_twiddle_0 = nullptr;
    }

    if (device_h_twiddle_0)
    {
        cudaFree(device_h_twiddle_0);
        device_h_twiddle_0 = nullptr;
    }

    if (plan2d)
    {
        cufftDestroy(plan2d);
        plan2d = 0;
    }

}



// Phillips spectrum
// (Kx, Ky) - normalized wave vector
// Vdir - wind angle in radians
// V - wind speed
// waveScaleFactor - constant
float phillips(float Kx, float Ky, float Vdir, float V, float waveScaleFactor, float dir_depend)
{
    float k_squared = Kx * Kx + Ky * Ky;

    if (k_squared == 0.0f)
    {
        return (0.0f);
    }

    // largest possible wave from constant wind of velocity v
    float L = V * V / gravitationalConstant;

    float k_x = Kx / sqrtf(k_squared);
    float k_y = Ky / sqrtf(k_squared);
    float w_dot_k = k_x * cosf(Vdir) + k_y * sinf(Vdir);

    float phillips = waveScaleFactor * expf(-1.0f / (k_squared * L * L)) / (k_squared * k_squared) * w_dot_k * w_dot_k;

    // filter out waves moving opposite to wind
    if (w_dot_k < 0.0f)
    {
        phillips *= dir_depend;
    }

    // damp out waves with very small length w << l
    // float w = L / 10000;
    // phillips *= expf(-k_squared * w * w);

    return (phillips);
}

float urand(void)
{
    return (rand() / (float)RAND_MAX);
}

float gauss()
{
    float u1 = urand();
    float u2 = urand();

    if (u1 < 1e-6f)
    {
        u1 = 1e-6f;
    }

    return (sqrtf(-2 * logf(u1)) * cosf(2 * CUDART_PI_F * u2));
}


void Ocean::generate_initial_spectrum()
{
    for (unsigned int y = 0; y <= meshSizeLimit; y++)
    {
        for (unsigned int x = 0; x <= meshSizeLimit; x++)
        {
            float kx = (-(int)meshSizeLimit / 2.0f + x) * (2.0f * CUDART_PI_F / patchSize);
            float ky = (-(int)meshSizeLimit / 2.0f + y) * (2.0f * CUDART_PI_F / patchSize);

            float P = sqrtf(phillips(kx, ky, windDir, windSpeed, waveScaleFactor, dirDepend));

            if (kx == 0.0f && ky == 0.0f)
            {
                P = 0.0f;
            }

            // float Er = urand()*2.0f-1.0f;
            // float Ei = urand()*2.0f-1.0f;
            float Er = gauss();
            float Ei = gauss();

            float h0_re = Er * P * CUDART_SQRT_HALF_F;
            float h0_im = Ei * P * CUDART_SQRT_HALF_F;

            int i = y * spectrumW + x;
            host_h_twiddle_0[i].x = h0_re;
            host_h_twiddle_0[i].y = h0_im;
        }
    }
}