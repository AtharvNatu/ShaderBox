#include "Water.hpp"

static std::default_random_engine generator;
static std::normal_distribution<double> distribution(0.0, 1.0);
static auto lastFrame = std::chrono::high_resolution_clock::now();

// Vertex struct must match the one Vulkan expects (tight-packed).

__global__ void updateVerticesKernel(
    Vertex* vertices,
    cuDoubleComplex* h_y,
    cuDoubleComplex* h_x,
    cuDoubleComplex* h_z,
    cuDoubleComplex* grad_x,
    cuDoubleComplex* grad_z,
    int N,
    float vertexDistance,
    float choppiness,
    float normalRoughness)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Nplus1 = N + 1;
    int vertexCount = Nplus1 * Nplus1;
    if (idx >= vertexCount)
        return;

    int z = idx / Nplus1;
    int x = idx % Nplus1;
    int i_d = (z % N) * N + (x % N);
    int sign = ((z % N) + (x % N)) % 2 == 0 ? 1 : -1;
    double scale = 1.0 / (double)(N * N) * (double)sign;

    // Normalize FFT results
    double dx = h_x[i_d].x * scale;
    double dy = h_y[i_d].x * scale;
    double dz = h_z[i_d].x * scale;
    double gx = grad_x[i_d].x * scale;
    double gz = grad_z[i_d].x * scale;

    glm::vec3 origin(
        -0.5 + x * vertexDistance / float(N),
        0.0,
        -0.5 + z * vertexDistance / float(N)
    );

    glm::vec3 disp(
        choppiness * (float)dx,
        (float)dy,
        choppiness * (float)dz
    );

    vertices[idx].position = origin + disp;
    vertices[idx].normal   = glm::vec3(-gx * normalRoughness, 1.0f, -gz * normalRoughness);
}

__global__ void scaleKernel(
    cuDoubleComplex* dy, cuDoubleComplex* dx, cuDoubleComplex* dz,
    cuDoubleComplex* gx, cuDoubleComplex* gz,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int N2 = N * N;
    if (i >= N2) return;

    int row = i / N;
    int col = i % N;
    int sign = ((row + col) % 2 == 0) ? 1 : -1;
    double scale = (double)sign / (double)N2;

    dy[i].x *= scale;
    dy[i].y *= scale;
    dx[i].x *= scale;
    dx[i].y *= scale;
    dz[i].x *= scale;
    dz[i].y *= scale;
    gx[i].x *= scale;
    gx[i].y *= scale;
    gz[i].x *= scale;
    gz[i].y *= scale;
}



__global__ void computeSpectrumKernel(
    cuDoubleComplex* d_y,
    cuDoubleComplex* d_x,
    cuDoubleComplex* d_z,
    cuDoubleComplex* grad_x,
    cuDoubleComplex* grad_z,
    const cuDoubleComplex* h0_tk,
    const cuDoubleComplex* h0_tmk,
    int N, float L, float t)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= N) return;

    int i = m * N + n;

    float kx = (n - N / 2.f) * (2.0f * M_PI / L);
    float kz = (m - N / 2.f) * (2.0f * M_PI / L);
    float k_len = sqrtf(kx * kx + kz * kz);

    float w = sqrtf(9.81f * k_len);
    float coswt = cosf(w * t);
    float sinwt = sinf(w * t);

    cuDoubleComplex e = make_cuDoubleComplex(coswt, sinwt);
    cuDoubleComplex e_conj = make_cuDoubleComplex(coswt, -sinwt);

    cuDoubleComplex h_tk  = cuCadd(cuCmul(h0_tk[i], e), cuCmul(h0_tmk[i], e_conj));

    d_y[i] = h_tk;
    grad_x[i] = cuCmul(h_tk, make_cuDoubleComplex(0.0, kx));
    grad_z[i] = cuCmul(h_tk, make_cuDoubleComplex(0.0, kz));

    if (k_len > 1e-5f) {
        double inv_k = -1.0 / k_len;
        d_x[i] = cuCmul(h_tk, make_cuDoubleComplex(0.0, kx * inv_k));
        d_z[i] = cuCmul(h_tk, make_cuDoubleComplex(0.0, kz * inv_k));
    } else {
        d_x[i] = make_cuDoubleComplex(0.0, 0.0);
        d_z[i] = make_cuDoubleComplex(0.0, 0.0);
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(gpFile, "%s:%d CUDA error: %s (%d)\n", __func__, __LINE__, cudaGetErrorString(_e), _e); \
        /* optionally abort or return */ \
    } \
} while(0)

#define CHECK_CUFFT(call) do { \
    cufftResult _r = (call); \
    if (_r != CUFFT_SUCCESS) { \
        fprintf(gpFile, "%s:%d CUFFT error: %d\n", __func__, __LINE__, _r); \
        /* optionally abort or return */ \
    } \
} while(0)



Ocean::Ocean(OceanSettings settings) : oceanSettings(settings)
{
    const uint32_t tileSize = settings.tileSize;
    int nPlus1 = tileSize + 1;
    const uint32_t vertexCount = (tileSize + 1) * (tileSize + 1);
    const uint32_t indexCount = tileSize * tileSize * 6;

    vertexBufferSize = vertexCount * sizeof(Vertex);
    indexBufferSize = indexCount * sizeof(uint32_t);
    fprintf(gpFile, "vertexBufferSize = %lld\n", vertexBufferSize);
    fprintf(gpFile, "indexBufferSize = %lld\n", indexBufferSize);

    vertices.resize(vertexCount);
    indices.resize(indexCount);

    for (int z = 0; z < nPlus1; z++)
    {
        for (int x = 0; x < nPlus1; x++)
        {
            int i0 = z * nPlus1 + x;

            Vertex vertex;
            vertex.color = glm::normalize(glm::vec3(29, 162, 216));
            vertices[i0] = vertex;

            if (x < settings.tileSize && z < settings.tileSize)
            {
                int i1 = (z + 1) * nPlus1 + x;
                int i2 = (z + 1) * nPlus1 + (x + 1);
                int i3 = z * nPlus1 + (x + 1);

                indices.push_back(i3);
                indices.push_back(i0);
                indices.push_back(i1);
                indices.push_back(i1);
                indices.push_back(i2);
                indices.push_back(i3);
            }
        }
    }

    vkResult = createBuffers();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createBuffers() Failed For Water : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createBuffers() Succeeded For Water\n", __func__);

    reloadSettings(settings);
}

VkResult Ocean::createBuffers()
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

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

    //! VERTEX BUFFER
    memset((void*)&vertexData, 0, sizeof(BufferData));

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
    vkBufferCreateInfo.size = this->vertexBufferSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo.pNext = &vkExternalMemoryBufferCreateInfo;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Buffer\n", __func__);

    //* Step - 7
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData.vkBuffer, &vkMemoryRequirements);

    //* Step - 8
    VkExportMemoryAllocateInfo vkExportMemoryAllocateInfo;
    memset((void*)&vkExportMemoryAllocateInfo, 0, sizeof(VkExportMemoryAllocateInfo));
    vkExportMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
    vkExportMemoryAllocateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    
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
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        //* Step - 8.5
        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    //* Step - 9
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData.vkBuffer, vertexData.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Buffer\n", __func__);

    //* Export Memory For CUDA
    VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR;
    memset((void*)&vkMemoryGetWin32HandleInfoKHR, 0, sizeof(VkMemoryGetWin32HandleInfoKHR));
    vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
    vkMemoryGetWin32HandleInfoKHR.pNext = NULL;
    vkMemoryGetWin32HandleInfoKHR.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    vkMemoryGetWin32HandleInfoKHR.memory = vertexData.vkDeviceMemory;

    HANDLE vkMemoryHandle = NULL;
    vkResult = vkGetMemoryWin32HandleKHR_fnptr(vkDevice, &vkMemoryGetWin32HandleInfoKHR, &vkMemoryHandle);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Failed For Vertex Position GPU Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkGetMemoryWin32HandleKHR_fnptr() Succeeded For Vertex Position GPU Buffer\n", __func__);

    //* Import into CUDA
    cudaExternalMemoryHandleDesc cuExtMemoryHandleDesc;
    memset((void*)&cuExtMemoryHandleDesc, 0, sizeof(cudaExternalMemoryHandleDesc));
    cuExtMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    cuExtMemoryHandleDesc.handle.win32.handle = vkMemoryHandle;
    cuExtMemoryHandleDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryHandleDesc.flags = 0;
    
    cudaResult = cudaImportExternalMemory(&cudaExternalMemory, &cuExtMemoryHandleDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Failed For Vertex Position GPU Buffer : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaImportExternalMemory() Succeeded For Vertex Position GPU Buffer\n", __func__);

    CloseHandle(vkMemoryHandle);

    //* Map to CUDA Pointer
    cudaExternalMemoryBufferDesc cuExtMemoryBufferDesc;
    memset((void*)&cuExtMemoryBufferDesc, 0, sizeof(cudaExternalMemoryBufferDesc));
    cuExtMemoryBufferDesc.offset = 0;
    cuExtMemoryBufferDesc.size = (size_t)vkMemoryRequirements.size;
    cuExtMemoryBufferDesc.flags =0;
    
    cudaResult = cudaExternalMemoryGetMappedBuffer(&cudaDevicePtr, cudaExternalMemory, &cuExtMemoryBufferDesc);
    if (cudaResult != cudaSuccess)
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Failed For Vertex Position GPU Buffer : %d !!!\n", __func__, cudaResult);
    else
        fprintf(gpFile, "%s() => cudaExternalMemoryGetMappedBuffer() Succeeded For Vertex Position GPU Buffer\n", __func__);


    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! INDEX BUFFER
    memset((void*)&indexData, 0, sizeof(BufferData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = indexBufferSize;
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
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
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

    vkResult = vkMapMemory(vkDevice, indexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &indexMappedData);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For indexMappedData : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For indexMappedData\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

void Ocean::unmapMemory(VkDeviceMemory& vkDeviceMemory)
{
    vkUnmapMemory(vkDevice, vkDeviceMemory);
}

void Ocean::update()
{
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - lastFrame;
    lastFrame = now;
    this->dt = elapsed.count();

    simulationTime += simulationSpeed * dt;
    const int N = oceanSettings.tileSize;
    const float length = oceanSettings.length;

    dim3 threads2d(16, 16);
    dim3 blocks2d((N + threads2d.x - 1) / threads2d.x, (N + threads2d.y - 1) / threads2d.y);

    // 1) compute h_tilde(k,t) on GPU — pass device d_h0 (packed: h0_tk then h0_tmk)
    computeSpectrumKernel<<<blocks2d, threads2d>>>(
        d_yDisplacement, d_xDisplacement, d_zDisplacement,
        d_xGradient, d_zGradient,
        d_h0,                  // d_h0 points to h0_tk at offset 0
        d_h0 + (size_t)N * (size_t)N, // d_h0 + N*N is h0_tmk
        N, length, simulationTime
    );
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    // 2) inverse FFTs (in-place)
    cufftResult = cufftExecZ2Z(plan, d_yDisplacement, d_yDisplacement, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_yDisplacement : %d !!!\n", __func__, cufftResult);

    cufftResult = cufftExecZ2Z(plan, d_xDisplacement, d_xDisplacement, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_xDisplacement : %d !!!\n", __func__, cufftResult);

    cufftResult = cufftExecZ2Z(plan, d_zDisplacement, d_zDisplacement, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_zDisplacement : %d !!!\n", __func__, cufftResult);

    cufftResult = cufftExecZ2Z(plan, d_xGradient,     d_xGradient,     CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_xGradient : %d !!!\n", __func__, cufftResult);

    cufftResult = cufftExecZ2Z(plan, d_zGradient,     d_zGradient,     CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_zGradient : %d !!!\n", __func__, cufftResult);
    cudaDeviceSynchronize();

    // 3) Normalize (sign + divide by N*N) — kernel-based scale for all N*N elements
    {
        int num = N * N;
        int t = 256;
        int b = (num + t - 1) / t;
        scaleKernel<<<b, t>>>(d_yDisplacement, d_xDisplacement, d_zDisplacement, d_xGradient, d_zGradient, N);
        CHECK_CUDA(cudaGetLastError());
        cudaDeviceSynchronize();
    }

    // 4) Build vertex buffer directly in mapped Vulkan memory
    int vertexCount = (N + 1) * (N + 1);
    int threadsPerBlock = 256;
    int blocksPerGrid = (vertexCount + threadsPerBlock - 1) / threadsPerBlock;
    Vertex* cudaVertexBufferPtr = (Vertex*)cudaDevicePtr;

    updateVerticesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        cudaVertexBufferPtr,
        d_yDisplacement, d_xDisplacement, d_zDisplacement,
        d_xGradient, d_zGradient,
        N, vertexDistance, choppiness, normalRoughness
    );
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    // --- optional debug readback for first vertex (disable in release) ---
    #if 1
    Vertex first;
    CHECK_CUDA(cudaMemcpy(&first, cudaVertexBufferPtr, sizeof(Vertex), cudaMemcpyDeviceToHost));
    fprintf(gpFile, "DEBUG First vertex: (%.6f, %.6f, %.6f)\n", first.position.x, first.position.y, first.position.z);
    #endif
}


void Ocean::reloadSettings(OceanSettings newSettings)
{
    oceanSettings = newSettings;
    int N = newSettings.tileSize;
    float length = newSettings.length;

    // cleanup old host/device allocations
    if (hostData) { delete[] hostData; hostData = nullptr; }
    if (deviceData) { cudaFree(deviceData); deviceData = nullptr; }
    if (d_h0) { cudaFree(d_h0); d_h0 = nullptr; }

    // Host-side init for initial spectrum: store 2 * N*N complex values
    hostData = new std::complex<double>[2 * N * N];
    h0_tk  = hostData + 0 * N * N;
    h0_tmk = hostData + 1 * N * N;

    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            int i = m * N + n;
            float kx = (n - N / 2.f) * twoPi / length;
            float kz = (m - N / 2.f) * twoPi / length;
            glm::vec2 k(kx, kz);
            h0_tk[i]  = h0_tilde(k);
            h0_tmk[i] = h0_tilde(-k);
        }
    }

    // Allocate GPU buffers for all 5 displacement fields (as before)
    size_t complexBytes = sizeof(cufftDoubleComplex);
    size_t fieldSize = complexBytes * N * N;
    CHECK_CUDA(cudaMalloc((void**)&deviceData, 5 * fieldSize)); // deviceData is cufftDoubleComplex*
    // Assign typed pointers
    d_yDisplacement = deviceData + 0 * (size_t)N * (size_t)N;
    d_xDisplacement = deviceData + 1 * (size_t)N * (size_t)N;
    d_zDisplacement = deviceData + 2 * (size_t)N * (size_t)N;
    d_xGradient     = deviceData + 3 * (size_t)N * (size_t)N;
    d_zGradient     = deviceData + 4 * (size_t)N * (size_t)N;

    CHECK_CUDA(cudaMemset(deviceData, 0, 5 * fieldSize)); // clear device arrays

    // Allocate and copy h0 seeds to device (packed: first N*N = h0_tk, next N*N = h0_tmk)
    CHECK_CUDA(cudaMalloc((void**)&d_h0, 2 * fieldSize));
    CHECK_CUDA(cudaMemcpy(d_h0, h0_tk, 2 * fieldSize, cudaMemcpyHostToDevice));

    // Create FFT plan
    if (plan) { cufftDestroy(plan); plan = 0; }
    cufftResult = cufftPlan2d(&plan, N, N, CUFFT_Z2Z);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftPlan2d() Failed : %d !!!\n", __func__, cufftResult);

    // Build index buffer (same as before)
    std::vector<uint32_t> indices;
    indices.reserve(N * N * 6);
    int Nplus1 = N + 1;
    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            if (x < N && z < N) {
                int i0 = z * Nplus1 + x;
                int i1 = (z + 1) * Nplus1 + x;
                int i2 = (z + 1) * Nplus1 + (x + 1);
                int i3 = z * Nplus1 + (x + 1);
                indices.push_back(i3);
                indices.push_back(i0);
                indices.push_back(i1);
                indices.push_back(i1);
                indices.push_back(i2);
                indices.push_back(i3);
            }
        }
    }

    if (indexMappedData && indexBufferSize >= indices.size() * sizeof(uint32_t))
        memcpy(indexMappedData, indices.data(), indexBufferSize);

    indexCount = static_cast<uint32_t>(indices.size());
}



/**
* "A useful model for wind-driven waves larger than
* capillary waves in a fully developed sea is the Phillips spectrum"
*
* Equation 40 in Tessendorf (2001)
*/
double Ocean::phillips(const glm::vec2& k) {
    double L = oceanSettings.windSpeed * oceanSettings.windSpeed / Ocean::g;
    double k_len = glm::length(k);
    k_len = (k_len < 0.0001) ? 0.0001 : k_len; // to avoid divide by 0
    double k2 = k_len * k_len;
    double k4 = k2 * k2;

    double kw = 0.0;
    if (k.x || k.y) {
        kw = glm::dot(glm::normalize(k), glm::normalize(oceanSettings.windDirection));
    }

    double res = oceanSettings.amplitude * kw * kw * exp(-1 / (k2 * L * L)) / k4;

    return res;
}

/**
* Dispersion relation suggested with regard to depth d: 
*   sqrt(k * g * tanh(k * d))
* Notice: for large d, tanh = 1, so formula equals.
*   sqrt(k * g)
*/
double Ocean::dispersion(const glm::vec2& K) {
    return sqrt(glm::length(K) * Ocean::g);
}

/**
* Equation 42 in Tessendorf (2001)
*/
std::complex<double> Ocean::h0_tilde(const glm::vec2& K) {
    double er = distribution(generator);
    double ei = distribution(generator);

    return sqrt(phillips(K)) * (std::complex(er, ei)) / sqrt(2.0);
}

/**
* Equation 43 in Tessendorf (2001)
*/
std::complex<double> Ocean::h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t) {
    double wkt = dispersion(K) * t;
    return h0_tk * exp(std::complex(0.0, wkt)) + std::conj(h0_tmk) * exp(std::complex(0.0, -wkt));
}

Ocean::~Ocean()
{
    if (indexData.vkDeviceMemory)
    {
        unmapMemory(indexData.vkDeviceMemory);
        vkFreeMemory(vkDevice, indexData.vkDeviceMemory, NULL);
        indexData.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For indexData.vkDeviceMemory\n", __func__);
    }

    if (indexData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, indexData.vkBuffer, NULL);
        indexData.vkBuffer = VK_NULL_HANDLE;
        indexMappedData = NULL;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For indexData.vkBuffer\n", __func__);
    }

    if (vertexData.vkDeviceMemory)
    {
        unmapMemory(vertexData.vkDeviceMemory);
        vkFreeMemory(vkDevice, vertexData.vkDeviceMemory, NULL);
        vertexData.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData.vkDeviceMemory\n", __func__);
    }

    if (vertexData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData.vkBuffer, NULL);
        vertexData.vkBuffer = VK_NULL_HANDLE;
        vertexMappedData = NULL;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData.vkBuffer\n", __func__);
    }


    if (hostData)
    {
        delete[] hostData;
        hostData = nullptr;
    }
}
