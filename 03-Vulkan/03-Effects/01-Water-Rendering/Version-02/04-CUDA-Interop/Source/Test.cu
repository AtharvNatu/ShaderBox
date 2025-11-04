#include "Ocean.hpp"


#define K_VEC(n, m) glm::vec2(2 * M_PI * (n - N / 2) / x_length, 2 * M_PI * (m - M / 2) / z_length)

#define CUDA_CHECK(call) \
    do { cudaError_t err = (call); if (err != cudaSuccess) { fprintf(gpFile, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); return (VkResult)VK_ERROR_DEVICE_LOST; } } while(0)

#define CUFFT_CHECK(call) \
    do { cufftResult r = (call); if (r != CUFFT_SUCCESS) { fprintf(gpFile, "cuFFT error %s:%d code=%d\n", __FILE__, __LINE__, (int)r); return (VkResult)VK_ERROR_DEVICE_LOST; } } while(0)


//! CUDA Kernels
__global__ void buildSpectrumKernel(
    cuComplex* h_twiddle,
    const cuComplex* h0,
    const cuComplex* h0_conj,
    int N, int M, float t, float G)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;

    int idx = m * N + n;

    float kx = 2.0f * M_PI * (n - N / 2.0f);
    float kz = 2.0f * M_PI * (m - M / 2.0f);
    float k_len = sqrtf(kx * kx + kz * kz);
    if (k_len < 1e-6f) { h_twiddle[idx] = make_cuComplex(0,0); return; }

    float omega = sqrtf(G * k_len);
    cuComplex e_pos = make_cuComplex(cosf(omega * t), sinf(omega * t));
    cuComplex e_neg = make_cuComplex(cosf(-omega * t), sinf(-omega * t));

    cuComplex term1 = cuCmulf(h0[idx], e_pos);
    cuComplex term2 = cuCmulf(h0_conj[idx], e_neg);
    h_twiddle[idx] = cuCaddf(term1, term2);
}

// Custom float3 normalize for CUDA
__device__ inline float3 normalize3(const float3& v)
{
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-6f)
        return make_float3(v.x / len, v.y / len, v.z / len);
    else
        return make_float3(0.0f, 1.0f, 0.0f);
}

// ------------------------------
// buildSpatialSpectraKernel
// ------------------------------
__global__ void buildSpatialSpectraKernel(
    const cuComplex* h_twiddle,
    cuComplex* dispX, cuComplex* dispZ,
    cuComplex* slopeX, cuComplex* slopeZ,
    int N, int M, float lambda)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;

    int idx = m * N + n;
    float kx = 2.0f * CUDART_PI_F * (n - N / 2.0f);
    float kz = 2.0f * CUDART_PI_F * (m - M / 2.0f);
    float k_len = sqrtf(kx * kx + kz * kz);

    if (k_len < 1e-6f)
    {
        dispX[idx] = make_cuComplex(0, 0);
        dispZ[idx] = make_cuComplex(0, 0);
        slopeX[idx] = make_cuComplex(0, 0);
        slopeZ[idx] = make_cuComplex(0, 0);
        return;
    }

    float nx = kx / k_len;
    float nz = kz / k_len;

    // Use cuCmulf for complex multiplication
    slopeX[idx] = cuCmulf(make_cuComplex(0.0f, kx), h_twiddle[idx]);
    slopeZ[idx] = cuCmulf(make_cuComplex(0.0f, kz), h_twiddle[idx]);

    dispX[idx] = cuCmulf(make_cuComplex(0.0f, -nx * lambda), h_twiddle[idx]);
    dispZ[idx] = cuCmulf(make_cuComplex(0.0f, -nz * lambda), h_twiddle[idx]);
}

// ------------------------------
// finalizeOceanKernel
// ------------------------------
__global__ void finalizeOceanKernel(
    const cuComplex* outH,
    const cuComplex* outDispX,
    const cuComplex* outDispZ,
    const cuComplex* outSlopeX,
    const cuComplex* outSlopeZ,
    float3* outDisplacement,
    float3* outNormal,
    int N, int M, float x_len, float z_len)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;

    int idx = m * N + n;
    float sign = ((n + m) & 1) ? -1.0f : 1.0f;

    float y = sign * outH[idx].x;

    float x = (n - N / 2.0f) * x_len / N - sign * outDispX[idx].x;
    float z = (m - M / 2.0f) * z_len / M - sign * outDispZ[idx].x;

    float sx = sign * outSlopeX[idx].x;
    float sz = sign * outSlopeZ[idx].x;

    float3 normal = normalize3(make_float3(-sx, 1.0f, -sz));

    outDisplacement[idx] = make_float3(x, y, z);
    outNormal[idx] = normal;
}




Ocean::Ocean()
{
    // Code
    omega_hat = glm::normalize(omega_vec);
    meshSize = sizeof(glm::vec3) * N * M;
    kNum = N * M;

    std::mt19937 rng(1337);
    std::normal_distribution<float> normalDist(0.0f, 1.0f);

    const float A = 0.00002f;   // amplitude constant (tune for wave size)
    const float g = 9.81f;      // gravity
    const float windSpeed = 30.0f;  // m/s
    const glm::vec2 windDir = glm::normalize(glm::vec2(1.0f, 1.0f));
    const float L = (windSpeed * windSpeed) / g;

    const float damp = 0.001f;
    const float l = L * damp;

    h_twiddle_0 = new std::complex<float>[N * M];
    h_twiddle_0_conjunction = new std::complex<float>[N * M];

    for (int m = 0; m < M; ++m)
    {
        float kz = 2.0f * M_PI * (m - M / 2.0f) / z_length;
        for (int n = 0; n < N; ++n)
        {
            float kx = 2.0f * M_PI * (n - N / 2.0f) / x_length;

            glm::vec2 k(kx, kz);
            float k_len = glm::length(k);

            if (k_len < 1e-6f)
            {
                h_twiddle_0[m * N + n] = {0.0f, 0.0f};
                h_twiddle_0_conjunction[m * N + n] = {0.0f, 0.0f};
                continue;
            }

            // Phillips spectrum
            float k_dot_w = glm::dot(glm::normalize(k), glm::normalize(windDir));
            float phillips = A * expf(-1.0f / (k_len * L * k_len * L)) /
                             (k_len * k_len * k_len * k_len) *
                             (k_dot_w * k_dot_w);

            // suppress waves against wind direction
            if (k_dot_w < 0.0f)
                phillips *= 0.07f;

            // dampen small waves
            phillips *= expf(-k_len * k_len * l * l);

            float Er = normalDist(rng);
            float Ei = normalDist(rng);

            std::complex<float> h0 = std::complex<float>(Er, Ei) * sqrtf(phillips * 0.5f);
            h_twiddle_0[m * N + n] = h0;
            h_twiddle_0_conjunction[m * N + n] = std::conj(h0);
        }
    }
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
    size_t kNumBytes = sizeof(cufftComplex) * kNum;
    size_t float3Bytes = sizeof(float3) * kNum;

    // allocate device arrays
    CUDA_CHECK(cudaMalloc((void**)&d_h0, kNumBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_h0_conj, kNumBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_h, kNumBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_slope_x, kNumBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_slope_z, kNumBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_disp_x, kNumBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_disp_z, kNumBytes));

    cudaMemcpy(d_h0, h_twiddle_0, sizeof(cuComplex) * kNum, cudaMemcpyHostToDevice);
    cudaMemcpy(d_h0_conj, h_twiddle_0_conjunction, sizeof(cuComplex) * kNum, cudaMemcpyHostToDevice);

    // create a 2D plan (in-place)
    CUFFT_CHECK(cufftPlan2d(&plan2d, N, M, CUFFT_C2C));

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

    //! Vertex Displacement Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_displacement, 0, sizeof(BufferData));

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
    vkBufferCreateInfo.size = meshSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_displacement.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Displacement Buffer\n", __func__);

    //* Step - 7
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_displacement.vkBuffer, &vkMemoryRequirements);

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
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_displacement.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Displacement Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_displacement.vkBuffer, vertexData_displacement.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Displacement Buffer\n", __func__);

    //* Export Memory For CUDA
    VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR;
    memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
    vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    vkMemoryGetWin32HandleInfoKHR.memory = vertexData_displacement.vkDeviceMemory;

    HANDLE vkMemoryHandle = NULL;
    vkResult = vkGetMemoryWin32HandleKHR_fnptr(vkDevice, &vkMemoryGetWin32HandleInfoKHR, &vkMemoryHandle);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Succeeded For Vertex Displacement Buffer\n", __func__);

    //* Import into CUDA
    cudaExternalMemoryHandleDesc cuExtMemoryHandleDesc;
    memset((void*)&cuExtMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
    cuExtMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuExtMemoryHandleDesc.handle.win32.handle = vkMemoryHandle;
    cuExtMemoryHandleDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryHandleDesc.flags = 0;
    
    cudaResult = cudaImportExternalMemory(&cudaExternalMemory_displacement, &cuExtMemoryHandleDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Succeeded For Vertex Displacement Buffer\n", __func__);

    CloseHandle(vkMemoryHandle);

    //* Map to CUDA Pointer
    cudaExternalMemoryBufferDesc cuExtMemoryBufferDesc;
    memset((void*)&cuExtMemoryBufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
    cuExtMemoryBufferDesc.offset = 0;
    cuExtMemoryBufferDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryBufferDesc.flags =0;
    
    cudaResult = cudaExternalMemoryGetMappedBuffer(&displacementPtr, cudaExternalMemory_displacement, &cuExtMemoryBufferDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Succeeded For Vertex Displacement Buffer\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! Vertex Normals Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_normals, 0, sizeof(BufferData));

    memset((void*)&vkExternalMemoryBufferCreateInfo, 0, sizeof(VkExternalMemoryBufferCreateInfo));
    vkExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    vkExternalMemoryBufferCreateInfo.pNext = NULL;
    vkExternalMemoryBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = meshSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_normals.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Normals Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Normals Buffer\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_normals.vkBuffer, &vkMemoryRequirements);

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
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_normals.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Normals Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Normals Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_normals.vkBuffer, vertexData_normals.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Normals Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Normals Buffer\n", __func__);

    //* Export Memory For CUDA
    memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
    vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    vkMemoryGetWin32HandleInfoKHR.memory = vertexData_normals.vkDeviceMemory;

    vkMemoryHandle = NULL;
    vkResult = vkGetMemoryWin32HandleKHR_fnptr(vkDevice, &vkMemoryGetWin32HandleInfoKHR, &vkMemoryHandle);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Failed For Vertex Normals Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Succeeded For Vertex Normals Buffer\n", __func__);

    //* Import into CUDA
    memset((void*)&cuExtMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
    cuExtMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuExtMemoryHandleDesc.handle.win32.handle = vkMemoryHandle;
    cuExtMemoryHandleDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryHandleDesc.flags = 0;
    
    cudaResult = cudaImportExternalMemory(&cudaExternalMemory_normals, &cuExtMemoryHandleDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Failed For Vertex Normals Buffer : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Succeeded For Vertex Normals Buffer\n", __func__);

    CloseHandle(vkMemoryHandle);

    //* Map to CUDA Pointer
    memset((void*)&cuExtMemoryBufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
    cuExtMemoryBufferDesc.offset = 0;
    cuExtMemoryBufferDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryBufferDesc.flags =0;
    
    cudaResult = cudaExternalMemoryGetMappedBuffer(&normalsPtr, cudaExternalMemory_normals, &cuExtMemoryBufferDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Failed For Vertex Normals Buffer : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Succeeded For Vertex Normals Buffer\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! Index Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //* Generate Indices
    int p = 0;
    indexCount = (N - 1) * (M - 1) * 6;
    indices = new unsigned int[indexCount];

    for (int j = 0; j < N - 1; j++)
    {
        for (int i = 0; i < M - 1; i++)
        {
            indices[p++] = i + j * N;
            indices[p++] = (i + 1) + j * N;
            indices[p++] = i + (j + 1) * N;

            indices[p++] = (i + 1) + j * N;
            indices[p++] = (i + 1) + (j + 1) * N;
            indices[p++] = i + (j + 1) * N;
        }
    }
    
    //* Step - 4
    memset((void*)&indexData, 0, sizeof(BufferData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = indexCount * sizeof(unsigned int);
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
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, indexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Index Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Index Buffer\n", __func__);

    //* Step - 12
    memcpy(data, indices, indexCount * sizeof(unsigned int));

    //* Step - 13
    vkUnmapMemory(vkDevice, indexData.vkDeviceMemory);

    delete[] indices;
    indices = nullptr;
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

    translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -2.0f, -50.0f));
    scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.05f, 0.05f, 0.05f));
    mvpData.modelMatrix = translationMatrix * scaleMatrix;
    mvpData.viewMatrix = cameraMatrix;
    
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

    lightPosition = lightDirection * 50.0f;

    waterUBO.lightPosition = glm::vec4(lightPosition, 0.0f);
    waterUBO.lightAmbient = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    waterUBO.lightDiffuse = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    waterUBO.lightSpecular = glm::vec4(1.0f, 0.9f, 0.7f, 0.0f);
    waterUBO.viewPosition = glm::vec4(30.0f, 30.0f, 60.0f, 0.0f);
    waterUBO.heightVector = glm::vec4(heightMin * 0.1, heightMax * 0.1, 0.0f, 0.0f);

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
    vkDescriptorSetLayoutBinding_array[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
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
    VkVertexInputBindingDescription vkVertexInputBindingDescription_array[2];
    memset((void*)vkVertexInputBindingDescription_array, 0, sizeof(VkVertexInputBindingDescription) * _ARRAYSIZE(vkVertexInputBindingDescription_array));

    //! Position
    vkVertexInputBindingDescription_array[0].binding = 0;
    vkVertexInputBindingDescription_array[0].stride = sizeof(glm::vec3);
    vkVertexInputBindingDescription_array[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    
    //! Normals
    vkVertexInputBindingDescription_array[1].binding = 1;
    vkVertexInputBindingDescription_array[1].stride = sizeof(glm::vec3);
    vkVertexInputBindingDescription_array[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[2];
    memset((void*)vkVertexInputAttributeDescription_array, 0, sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));

    //! Position
    vkVertexInputAttributeDescription_array[0].binding = 0;
    vkVertexInputAttributeDescription_array[0].location = 0;
    vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[0].offset = 0;

    //! Normals
    vkVertexInputAttributeDescription_array[1].binding = 1;
    vkVertexInputAttributeDescription_array[1].location = 1;
    vkVertexInputAttributeDescription_array[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[1].offset = 0;

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
    vkPipelineInputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

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

    //! Bind with Vertex Displacement Buffer
    VkDeviceSize vkDeviceSize_offset_array[1];
    memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));
    vkCmdBindVertexBuffers(
        commandBuffer,
        0,
        1,
        &vertexData_displacement.vkBuffer,
        vkDeviceSize_offset_array
    );

    //! Bind with Vertex Normals Buffer
    memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));
    vkCmdBindVertexBuffers(
        commandBuffer,
        1,
        1,
        &vertexData_normals.vkBuffer,
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
        indexCount,
        1,              //* Count of geometry instances
        0,              //* Starting offset of index buffer
        0,              //* Starting offset of vertex buffer
        0               //* Nth instance
    );
}

void Ocean::update(glm::mat4 cameraViewMatrix)
{
    fTime += waveSpeed;

    cameraMatrix = cameraViewMatrix;
    
    //* Build Tessendorf Mesh
    generate_fft_data(fTime);

    
    updateUniformBuffer();
}

void Ocean::generate_fft_data(float time)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    // Build h_twiddle(k, t)
    buildSpectrumKernel<<<grid, block>>>(d_h, d_h0, d_h0_conj, N, M, time, G);
    cudaDeviceSynchronize();

    // Build slope/displacement frequency domain
    buildSpatialSpectraKernel<<<grid, block>>>(d_h, d_disp_x, d_disp_z, d_slope_x, d_slope_z, N, M, lambda);
    cudaDeviceSynchronize();

    // Run CUFFT (frequency -> spatial domain)
    cufftExecC2C(plan2d, (cufftComplex*)d_h, (cufftComplex*)d_h, CUFFT_INVERSE);
    cufftExecC2C(plan2d, (cufftComplex*)d_disp_x, (cufftComplex*)d_disp_x, CUFFT_INVERSE);
    cufftExecC2C(plan2d, (cufftComplex*)d_disp_z, (cufftComplex*)d_disp_z, CUFFT_INVERSE);
    cufftExecC2C(plan2d, (cufftComplex*)d_slope_x, (cufftComplex*)d_slope_x, CUFFT_INVERSE);
    cufftExecC2C(plan2d, (cufftComplex*)d_slope_z, (cufftComplex*)d_slope_z, CUFFT_INVERSE);
    cudaDeviceSynchronize();

    // Final pass — write directly to shared Vulkan buffers
    finalizeOceanKernel<<<grid, block>>>(
        d_h,
        d_disp_x, d_disp_z,
        d_slope_x, d_slope_z,
        (float3*)displacementPtr,
        (float3*)normalsPtr,
        N, M, x_length, z_length
    );
    cudaDeviceSynchronize();
    


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

    if (cudaExternalMemory_normals)
    {
        if (cudaDestroyExternalMemory(cudaExternalMemory_normals) == cudaSuccess)
            fprintf(gpFile, "%s() => cudaDestroyExternalMemory() Succedded For cudaExternalMemory_normals\n", __func__);
        cudaExternalMemory_normals = NULL;
        normalsPtr = NULL;
    }

    if (cudaExternalMemory_displacement)
    {
        if (cudaDestroyExternalMemory(cudaExternalMemory_displacement) == cudaSuccess)
            fprintf(gpFile, "%s() => cudaDestroyExternalMemory() Succedded For cudaExternalMemory_displacement\n", __func__);
        cudaExternalMemory_displacement = NULL;
        displacementPtr = NULL;
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

    if (vertexData_normals.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_normals.vkDeviceMemory, NULL);
        vertexData_normals.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_normals.vkDeviceMemory\n", __func__);
    }

    if (vertexData_normals.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_normals.vkBuffer, NULL);
        vertexData_normals.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_normals.vkBuffer\n", __func__);
    }

    if (vertexData_displacement.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_displacement.vkDeviceMemory, NULL);
        vertexData_displacement.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_displacement.vkDeviceMemory\n", __func__);
    }

    if (vertexData_displacement.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_displacement.vkBuffer, NULL);
        vertexData_displacement.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_displacement.vkBuffer\n", __func__);
    }

    if (plan2d)
    {
        cufftDestroy(plan2d);
        plan2d = 0;
    }

    if (device_out_displacement_z)
    {
        cudaFree(device_out_displacement_z);
        device_out_displacement_z = nullptr;
    }

    if (device_out_displacement_x)
    {
        cudaFree(device_out_displacement_x);
        device_out_displacement_x = nullptr;
    }

    if (device_out_slope_z)
    {
        cudaFree(device_out_slope_z);
        device_out_slope_z = nullptr;
    }

    if (device_out_slope_x)
    {
        cudaFree(device_out_slope_x);
        device_out_slope_x = nullptr;
    }

    if (device_out_height)
    {
        cudaFree(device_out_height);
        device_out_height = nullptr;
    }

    if (device_out_height)
    {
        cudaFree(device_out_height);
        device_out_height = nullptr;
    }

    if (device_in_displacement_z)
    {
        cudaFree(device_in_displacement_z);
        device_in_displacement_z = nullptr;
    }

    if (device_in_displacement_x)
    {
        cudaFree(device_in_displacement_x);
        device_in_displacement_x = nullptr;
    }

    if (device_in_slope_z)
    {
        cudaFree(device_in_slope_z);
        device_in_slope_z = nullptr;
    }

    if (device_in_slope_x)
    {
        cudaFree(device_in_slope_x);
        device_in_slope_x = nullptr;
    }

    if (device_in_height)
    {
        cudaFree(device_in_height);
        device_in_height = nullptr;
    }

    if (device_h_twiddle)
    {
        cudaFree(device_h_twiddle);
        device_h_twiddle = nullptr;
    }

    if (device_h_twiddle_0_conjugate)
    {
        cudaFree(device_h_twiddle_0_conjugate);
        device_h_twiddle_0_conjugate = nullptr;
    }

    if (device_h_twiddle_0)
    {
        cudaFree(device_h_twiddle_0);
        device_h_twiddle_0 = nullptr;
    }

    cudaFree(d_h0);
    cudaFree(d_h0_conj);
    cudaFree(d_h);
    cudaFree(d_disp_x);
    cudaFree(d_disp_z);
    cudaFree(d_slope_x);
    cudaFree(d_slope_z);

    if (host_h_twiddle_0_conjugate)
    {
        free(host_h_twiddle_0_conjugate);
        host_h_twiddle_0_conjugate = nullptr;
    }

    if (host_h_twiddle_0)
    {
        free(host_h_twiddle_0);
        host_h_twiddle_0 = nullptr;
    }
}
