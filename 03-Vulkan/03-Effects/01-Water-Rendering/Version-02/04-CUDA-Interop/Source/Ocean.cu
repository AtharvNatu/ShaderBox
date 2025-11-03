#include "Ocean.hpp"

//! CUDA Kernels

__global__ void build_fft(
    const cufftComplex* __restrict__ h_twiddle_0,
    const cufftComplex* __restrict__ h_twiddle_0_conjugate,
    cufftComplex* __restrict__ out_h_twiddle,
    cufftComplex* __restrict__ slope_x,
    cufftComplex* __restrict__ slope_z,
    cufftComplex* __restrict__ displacement_x,
    cufftComplex* __restrict__ displacement_z,
    int N,
    int M,
    float time,
    float lambda,
    float x_length,
    float z_length
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= N || iy >= M) return;

    int idx = iy * N + ix;

    // Compute wave vector
    float kx = 2.0f * M_PI * (float(ix) - float(N) * 0.5f) / x_length;
    float kz = 2.0f * M_PI * (float(iy) - float(M) * 0.5f) / z_length;
    float k_len = sqrtf(kx*kx + kz*kz);
    if (k_len < 1e-6f) k_len = 1e-6f;

    cufftComplex h0 = h_twiddle_0[idx];
    cufftComplex h0_conj = h_twiddle_0_conjugate[idx];

    // Angular frequency
    float omega = sqrtf(9.8f * k_len);

    float cosw = cosf(omega * time);
    float sinw = sinf(omega * time);

    // h_t(k, t) = h0 * e^{iwt} + h0* * e^{-iwt}
    cufftComplex term1 = { h0.x * cosw - h0.y * sinw, h0.x * sinw + h0.y * cosw };
    cufftComplex term2 = { h0_conj.x * cosw + h0_conj.y * sinw, -h0_conj.x * sinw + h0_conj.y * cosw };
    cufftComplex h_t = { term1.x + term2.x, term1.y + term2.y };

    // Checkerboard sign correction
    int sign = ((ix + iy) & 1) ? -1 : 1;
    h_t.x *= sign;
    h_t.y *= sign;

    // Write height spectrum
    out_h_twiddle[idx] = h_t;

    // Slopes
    slope_x[idx].x = -kx * h_t.y;
    slope_x[idx].y =  kx * h_t.x;

    slope_z[idx].x = -kz * h_t.y;
    slope_z[idx].y =  kz * h_t.x;

    // Displacement
    float nx = kx / k_len;
    float nz = kz / k_len;

    displacement_x[idx].x = -nx * h_t.y;
    displacement_x[idx].y =  nx * h_t.x;

    displacement_z[idx].x = -nz * h_t.y;
    displacement_z[idx].y =  nz * h_t.x;
}

__global__ void copy_to_maps(
    const cufftComplex* __restrict__ out_height,
    const cufftComplex* __restrict__ out_slope_x,
    const cufftComplex* __restrict__ out_slope_z,
    const cufftComplex* __restrict__ out_displacement_x,
    const cufftComplex* __restrict__ out_displacement_z,
    float3* __restrict__ displacement_map,
    float3* __restrict__ normal_map,
    int N,
    int M,
    float lambda,
    float x_length,
    float z_length
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= N || iy >= M) return;

    int idx = iy * N + ix;

    float scale = 1.0f / (float)(N * M);

    // Real part only (height field)
    float h = out_height[idx].x * scale;
    float sx = out_slope_x[idx].x * scale;
    float sz = out_slope_z[idx].x * scale;
    float dx = out_displacement_x[idx].x * scale;
    float dz = out_displacement_z[idx].x * scale;

    int sign = ((ix + iy) & 1) ? -1 : 1;

    // Normals
    float nx = -sx;
    float ny = 1.0f;
    float nz = -sz;
    float invlen = rsqrtf(nx*nx + ny*ny + nz*nz + 1e-12f);
    normal_map[idx] = make_float3(nx * invlen, ny * invlen, nz * invlen);

    // Displacement (world-space)
    float wx = (float(ix) - N * 0.5f) * x_length / float(N) - sign * lambda * dx;
    float wy = sign * h;
    float wz = (float(iy) - M * 0.5f) * z_length / float(M) - sign * lambda * dz;

    displacement_map[idx] = make_float3(wx, wy, wz);
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
    bool status = initializeHostData();
    if (!status)
        fprintf(gpFile, "%s() => initializeHostData() Failed For Ocean : %d !!!\n", __func__);
    else
        fprintf(gpFile, "%s() => initializeHostData() Succeeded For Ocean\n", __func__);

    status = initializeDeviceData();
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

bool Ocean::initializeHostData()
{
    // Code

    //* Allocate Host Arrays
    host_h_twiddle_0 = (cufftComplex*)malloc(sizeof(cufftComplex) * kNum);
    if (host_h_twiddle_0 == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For host_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    host_h_twiddle_0_conjugate = (cufftComplex*)malloc(sizeof(cufftComplex) * kNum);
    if (host_h_twiddle_0_conjugate == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For host_h_twiddle_0_conjunction !!!\n", __func__);
        return false;
    }

    //! Compute host_h_twiddle_0 and host_h_twiddle_0_conjugate
    compute_h_twiddle_0();
    compute_h_twiddle_0_conjugate();

    return true;
}

bool Ocean::initializeDeviceData()
{
    // Code

    //* Allocate Device Arrays
    cudaResult = cudaMalloc(&device_h_twiddle_0, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_h_twiddle_0_conjugate, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_h_twiddle_0_conjugate !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMemcpy(device_h_twiddle_0, host_h_twiddle_0, sizeof(cufftComplex) * kNum, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMemcpy() Failed For device_h_twiddle_0 !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMemcpy(device_h_twiddle_0_conjugate, host_h_twiddle_0_conjugate, sizeof(cufftComplex) * kNum, cudaMemcpyHostToDevice);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMemcpy() Failed For device_h_twiddle_0_conjugate !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_h_twiddle, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_h_twiddle !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_in_height, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_in_height !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_in_slope_x, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_in_slope_x !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_in_slope_z, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_in_slope_z !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_in_displacement_x, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_in_displacement_x !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_in_displacement_z, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_in_displacement_z !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_out_height, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_out_height !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_out_slope_x, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_out_slope_x !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_out_slope_z, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_out_slope_z !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_out_displacement_x, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_out_displacement_x !!!\n", __func__);
        return false;
    }

    cudaResult = cudaMalloc(&device_out_displacement_z, sizeof(cufftComplex) * kNum);
    if (cudaResult != cudaSuccess)
    {
        fprintf(gpFile, "%s() => cudaMalloc() Failed For device_out_displacement_z !!!\n", __func__);
        return false;
    }

    //* Create Plan 2D Complex-To-Complex
    fftResult = cufftPlan2d(&plan2d, M, N, CUFFT_C2C);
    if (fftResult != CUFFT_SUCCESS)
    {
        fprintf(gpFile, "%s() => cufftPlan2d() Failed For Ocean !!!\n", __func__);
        return false;
    }

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

//* Tessendorf Related
inline float Ocean::phillips_spectrum(float kx, float kz) const
{
    // Code
    float k_length = sqrtf((kx * kx) + (kz * kz));
    if (k_length < 1e-6f)
        return 0.0f;

    // Largest possible waves from continuous wind of speed V
    float wave_length = (V * V) / G;

    float kx_normalized = kx / k_length;
    float kz_normalized = kz / k_length;

    float dot = kx_normalized * omega_hat.x + kz_normalized * omega_hat.y;
    float dot_term = dot * dot;

    float exp_term = expf(-1.0f / (k_length * k_length * wave_length * wave_length));
    float result = A * exp_term * dot_term / powf(k_length, 4.0f);

    // Small-wave damping (Eq. 24) — uses constant L
    result *= expf(-k_length * k_length * L * L);

    return result;
}

void Ocean::compute_h_twiddle_0()
{
    // Code
    std::default_random_engine generator((unsigned)time(nullptr));
    std::normal_distribution<float> normal_distribution{0.0f, 1.0f}; 

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;
            
            // K Vector (kx, kz)
            float kx = 2.0f * M_PI * (float(n) - float(N) / 2.0f) / x_length;
            float kz = 2.0f * M_PI * (float(m) - float(M) / 2.0f) / z_length;

            float xi_real = normal_distribution(generator);
            float xi_imag = normal_distribution(generator);
            float spectrum = phillips_spectrum(kx, kz);
            float mag_sqrt = sqrtf(0.5f * spectrum);

            host_h_twiddle_0[index].x = mag_sqrt * xi_real;
            host_h_twiddle_0[index].y = mag_sqrt * xi_imag;
        }
    }
}

void Ocean::compute_h_twiddle_0_conjugate()
{
    // Code
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;
            
            int kn_neg = (N - n) % N;
            int km_neg = (M - m) % M;

            int index_neg = km_neg * N + kn_neg;

            host_h_twiddle_0_conjugate[index].x = host_h_twiddle_0_conjugate[index_neg].x;
            host_h_twiddle_0_conjugate[index].y = -host_h_twiddle_0_conjugate[index_neg].y;
        }
    }
}

//! Eqn. 19
void Ocean::generate_fft_data(float time)
{
    // Variable Declarations
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    float3* pDisplacement = (float3*)displacementPtr;
    float3* pNormals = (float3*)normalsPtr;

    build_fft<<<grid, block>>>(
        device_h_twiddle_0,
        device_h_twiddle_0_conjugate,
        device_h_twiddle,
        device_in_slope_x,
        device_in_slope_z,
        device_in_displacement_x,
        device_in_displacement_z,
        N,
        M,
        time,
        lambda,
        x_length,
        z_length
    );

    //* Inverse FFTs
    fftResult = cufftExecC2C(plan2d, device_h_twiddle, device_out_height, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For h_twiddle !!!\n", __func__);

    fftResult = cufftExecC2C(plan2d, device_in_slope_x, device_out_slope_x, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For slope_x !!!\n", __func__);

    fftResult = cufftExecC2C(plan2d, device_in_slope_z, device_out_slope_z, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For slope_z !!!\n", __func__);

    fftResult = cufftExecC2C(plan2d, device_in_displacement_x, device_out_displacement_x, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For displacement_x !!!\n", __func__);

    fftResult = cufftExecC2C(plan2d, device_in_displacement_z, device_out_displacement_z, CUFFT_INVERSE);
    if (fftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecC2C() Failed For displacement_z !!!\n", __func__);

    copy_to_maps<<<grid, block>>>(
        device_out_height,
        device_out_slope_x,
        device_out_slope_z,
        device_out_displacement_x,
        device_out_displacement_z,
        pDisplacement,
        pNormals,
        N,
        M,
        lambda,
        x_length,
        z_length
    );

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(gpFile, "CUDA Kernel Error: %s\n", cudaGetErrorString(err));

    float3 test;
    cudaMemcpy(&test, displacementPtr, sizeof(float3), cudaMemcpyDeviceToHost);
    fprintf(gpFile, "Sample vertex: %.3f %.3f %.3f\n", test.x, test.y, test.z);

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
