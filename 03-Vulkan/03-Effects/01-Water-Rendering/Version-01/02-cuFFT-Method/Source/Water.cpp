#include "Water.hpp"

static std::default_random_engine generator;
static std::normal_distribution<double> distribution(0.0, 1.0);
static auto lastFrame = std::chrono::high_resolution_clock::now();


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

    //! VERTEX BUFFER
    memset((void*)&vertexData, 0, sizeof(BufferData));

    //* Step - 5
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = this->vertexBufferSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

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

    vkResult = vkMapMemory(vkDevice, vertexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &vertexMappedData);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For vertexMappedData : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For vertexMappedData\n", __func__);
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
    // Code
    auto now = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = now - lastFrame;
    lastFrame = now;

    this->dt = elapsed.count();

    simulationTime += simulationSpeed * dt;
    float length = oceanSettings.length;
    int N = oceanSettings.tileSize;

    // Setup h_tk + device
    for (int m = 0; m < N; m++) 
    {
        for (int n = 0; n < N; n++) 
        {
            int i = m * N + n;
            float kx = (n - N / 2.f) * twoPi / length;
            float kz = (m - N / 2.f) * twoPi / length;
            glm::vec2 K(kx, kz);

            h_yDisplacement[i] = h_tilde(h0_tk[i], h0_tmk[i], K, simulationTime); // Initial displacement h_tilde(k, x, t)
            h_xGradient[i] = h_yDisplacement[i] * std::complex<double>(0.0, kx);
            h_zGradient[i] = h_yDisplacement[i] * std::complex<double>(0.0, kz);
            
            double k_length = glm::length(K);

            if (k_length > 0.00001) 
            {
                h_xDisplacement[i] = h_yDisplacement[i] * std::complex<double>(0.0, -kx / k_length);
                h_zDisplacement[i] = h_yDisplacement[i] * std::complex<double>(0.0, -kz / k_length);
            } 
            else 
            {
                h_xDisplacement[i] = h_yDisplacement[i] * std::complex<double>(0.0, 0.0);
                h_zDisplacement[i] = h_yDisplacement[i] * std::complex<double>(0.0, 0.0);
            }
        }
    }

    // Copy displacement_y, displacement_x, displacement_z, gradient_x, gradient_z simultaneously to device
    cudaResult = cudaMemcpy(d_yDisplacement, h_yDisplacement, 5 * sizeof(std::complex<double>) * N * N, cudaMemcpyHostToDevice);
    if (cudaResult != CUDA_SUCCESS)
        fprintf(gpFile, "%s() => cudaMemcpy() Failed (Host To Device): %s !!!\n", __func__, cudaGetErrorString(cudaResult));

    // In place transforms
    // Inverse FFT: h(k, x, t) = sum(h_tilde(k, x, t) * e^(2pikn / N))
    cufftResult = cufftExecZ2Z(plan, d_yDisplacement, d_yDisplacement, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_yDisplacement : %d !!!\n", __func__, cufftResult);

    cufftResult = cufftExecZ2Z(plan, d_xDisplacement, d_xDisplacement, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_xDisplacement : %d !!!\n", __func__, cufftResult); 

    cufftResult = cufftExecZ2Z(plan, d_zDisplacement, d_zDisplacement, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed d_zDisplacement : %d !!!\n", __func__, cufftResult); 

    cufftResult = cufftExecZ2Z(plan, d_xGradient, d_xGradient, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_xGradient : %d !!!\n", __func__, cufftResult); 

    cufftResult = cufftExecZ2Z(plan, d_zGradient, d_zGradient, CUFFT_INVERSE);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftExecZ2Z() Failed For d_zGradient : %d !!!\n", __func__, cufftResult); 

    // Copy displacement_y, displacement_x, displacement_z, gradient_x, gradient_z simultaneously from device
    cudaResult = cudaMemcpy(h_yDisplacement, d_yDisplacement, 5 * sizeof(std::complex<double>) * N * N, cudaMemcpyDeviceToHost);
    if (cudaResult != CUDA_SUCCESS)
        fprintf(gpFile, "%s() => cudaMemcpy() Failed (Device To Host): %s !!!\n", __func__, cudaGetErrorString(cudaResult));
    
    for (int m = 0; m < N; m++) {
      for (int n = 0; n < N; n++) {
        int i = m * N + n;
        int sign = (m + n) % 2 == 0 ? 1 : -1; // Larsson (2012), Equation 4.6
        h_yDisplacement[i] /= sign * (N * N);
        h_xDisplacement[i] /= sign * (N * N);
        h_zDisplacement[i] /= sign * (N * N);
        h_xGradient[i] /= sign * (N * N);
        h_zGradient[i] /= sign * (N * N);
      }
    }

    float base_amplitude = 0.004f;
    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            int i = m * N + n;
            float k = twoPi / rippleLength;
            float x = (n - N / 2.f);
            float z = (m - N / 2.f);
            glm::vec2 X(x, z);
            glm::vec2 K(k);

            float prop_dist = rippleLength * simulationTime / period;
            float dist_to_prop = abs(prop_dist - glm::length(X));
            int max_waves_in_flight = 1;
            float max_wave_dist = rippleLength * max_waves_in_flight;
            if (dist_to_prop < max_wave_dist / 2.0f) {
                float t = max_wave_dist / 2.0 - dist_to_prop;
                float amplitude = base_amplitude * sin (glm::half_pi<float>() * t);

                float disp = dispersion(K) / glm::length(K);
                float value = glm::length(X) * k + disp * -simulationTime;
                h_yDisplacement[i] += amplitude * (-1 + 2 * sin(value));
                h_xGradient[i] += amplitude * cos(value);
                h_zGradient[i] += amplitude * cos(value);
            }
        }
    }
    
    updateVertices();
}

void Ocean::updateVertices()
{
    // Code
    const uint32_t N = oceanSettings.tileSize;
    int Nplus1 = N + 1;

    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            int i_v = z * Nplus1 + x;
            int i_d = (z % N) * N + x % N;

            glm::vec3 origin_position = glm::vec3(-0.5 + x * vertexDistance / float(N), 0, -0.5 + z * vertexDistance / float(N));
            glm::vec3 displacement(
                choppiness * h_xDisplacement[i_d].real(), 
                h_yDisplacement[i_d].real(), 
                choppiness * h_zDisplacement[i_d].real()
            );
            vertices[i_v].position = origin_position + displacement;
        }
    }

    for (int z = 0; z < Nplus1; z++) {
        for (int x = 0; x < Nplus1; x++) {
            int i_v = z * Nplus1 + x;
            int i_d = (z % N) * N + x % N;
            double ex = h_xGradient[i_d].real();
            double ez = h_zGradient[i_d].real();
            vertices[i_v].normal = glm::vec3(-ex * normalRoughness, 1.0, -ez * normalRoughness);
        }
    }

    //! Update Vertex Buffer Data
    memcpy(vertexMappedData, vertices.data(), vertexBufferSize); 
}

void Ocean::render()
{

}

void Ocean::reloadSettings(OceanSettings newSettings)
{
    if (hostData)
        delete[] hostData;

    oceanSettings = newSettings;
    int N = newSettings.tileSize;
    float length = newSettings.length;
    
    hostData = new std::complex<double>[7 * N * N];
    h0_tk = hostData + 0 * N * N; // h0_tilde(k)
    h0_tmk = hostData + 1 * N * N; // h0_tilde(-k)

    for (int m = 0; m < N; m++) {
        for (int n = 0; n < N; n++) {
            int i = m * N + n;
            float kx = (n - N / 2.f) * twoPi / length;
            float kz = (m - N / 2.f) * twoPi / length;
            glm::vec2 k(kx, kz);
            h0_tk[i] = h0_tilde(k);
            h0_tmk[i] = h0_tilde(-k);
        }
    }

    h_yDisplacement = hostData + 2 * N * N; // h(k, x, t)
    h_xDisplacement = hostData + 3 * N * N; // x-displacement of h(k, x, t)
    h_zDisplacement = hostData + 4 * N * N; // z-displacement of h(k, x, t)
    h_xGradient = hostData + 5 * N * N; // x-gradient of h(k, x, t)
    h_zGradient = hostData + 6 * N * N; // z-gradient of h(k, x, t)

    if (deviceData)
    {
        cudaResult = cudaFree(deviceData);
        if (cudaResult != CUDA_SUCCESS)
            fprintf(gpFile, "%s() => cudaFree() Failed : %s !!!\n", __func__, cudaGetErrorString(cudaResult));
        deviceData = nullptr;
    }
    
    cudaResult = cudaMalloc((void**)&deviceData, 5 * sizeof(std::complex<double>) * N * N);
    if (cudaResult != CUDA_SUCCESS)
        fprintf(gpFile, "%s() => cudaMalloc() Failed : %s !!!\n", __func__, cudaGetErrorString(cudaResult));

    d_yDisplacement = deviceData + 0 * N * N;
    d_xDisplacement = deviceData + 1 * N * N;
    d_zDisplacement = deviceData + 2 * N * N;
    d_xGradient = deviceData + 3 * N * N;
    d_zGradient = deviceData + 4 * N * N;

    cufftResult = cufftPlan2d(&plan, N, N, CUFFT_Z2Z);
    if (cufftResult != CUFFT_SUCCESS)
        fprintf(gpFile, "%s() => cufftPlan2d() Failed : %d !!!\n", __func__, cufftResult);

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

    //! Update Index Buffer Data
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

    if (deviceData)
    {
        cudaFree(deviceData);
        if (cudaResult != CUDA_SUCCESS)
            fprintf(gpFile, "%s() => cudaFree() Failed : %s !!!\n", __func__, cudaGetErrorString(cudaResult));
        deviceData = nullptr;
    }

    if (hostData)
    {
        delete[] hostData;
        hostData = nullptr;
    }
}
