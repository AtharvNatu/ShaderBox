#include "Water.hpp"

static std::default_random_engine rng;
static std::normal_distribution<double> gaussianDistribution(0.0, 1.0);

Ocean::Ocean(OceanSettings settings) : oceanSettings(settings)
{
    const uint32_t tileSize = settings.tileSize;

    this->vertexCount = (tileSize + 1) * (tileSize + 1);
    this->indexCount = tileSize * tileSize * 6;

    VkDeviceSize complexSize = sizeof(float) * 2;
    VkDeviceSize planeSize = tileSize * tileSize * complexSize;

    memset((void*)&vertexData, 0, sizeof(BufferData));
    memset((void*)&indexData, 0, sizeof(BufferData));
    memset((void*)&fftData, 0, sizeof(BufferData));

    vertexData.vkDeviceSize = vertexCount * sizeof(Vertex);
    indexData.vkDeviceSize = indexCount * sizeof(uint32_t);
    fftData.vkDeviceSize = planeSize * 5; // For 5 buffers (Displacement - x,y,z | Gradient - x,y)

    pushData.tileSize = settings.tileSize;
    pushData.vertexDistance = vertexDistance;
    pushData.choppiness = choppiness;
    pushData.normalRoughness = normalRoughness;
    pushData.half = float(tileSize) * 0.5f;

    init();
    reloadSettings(settings);
    
}

void Ocean::init()
{
    // Code
    bool status = initializeFFT();
    if (!status)
    {
        fprintf(gpFile, "%s() => initializeFFT() Failed : %d\n", __func__, vkResult);
        return;
    }

    vkResult = createBuffers();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createBuffers() Failed For Water : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createBuffers() Succeeded For Water\n", __func__);

    vkResult = createComputeDescriptorSetLayout();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputeDescriptorSetLayout() Failed For Water : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputeDescriptorSetLayout() Succeeded For Water\n", __func__);

    vkResult = createComputePipelineLayout();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputePipelineLayout() Failed For Water : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputePipelineLayout() Succeeded For Water\n", __func__);

    vkResult = createComputeDescriptorPool();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputeDescriptorPool() Failed For Water : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputeDescriptorPool() Succeeded For Water\n", __func__);

    vkResult = createComputeDescriptorSet();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputeDescriptorSet() Failed For Water : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputeDescriptorSet() Succeeded For Water\n", __func__);

    vkResult = createComputePipeline();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputePipeline() Failed For Water : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputePipeline() Succeeded For Water\n", __func__);

}

void Ocean::createGrid()
{
    // Code

    // VERTEX DATA
    vertices.resize(vertexCount);

    int N = oceanSettings.tileSize;

    for (int z = 0; z <= N; z++)
    {
        for (int x = 0; x <= N; x++)
        {
            int idx = z * (N + 1) + x;

            // Initial State => X, Z, Y = 0
            // Compute Shader will overwrite position.y, displacement, normals per frame
            vertices[idx].position[0] = (float)x / float(N) - 0.5f; // Center Grid
            vertices[idx].position[1] = 0.0f; // Height = 0
            vertices[idx].position[2] = (float)z / float(N) - 0.5f;
            vertices[idx].position[3] = 1.0f;   // Padding

            // Constant Color Values
            vertices[idx].color[0] = 29.0f / 255.0f;
            vertices[idx].color[1] = 162.0f / 255.0f;
            vertices[idx].color[2] = 216.0f / 255.0f;
            vertices[idx].color[3] = 1.0f;  // Padding

            // Initial State => 0, 1, 0
            vertices[idx].normal[0] = 0.0f;
            vertices[idx].normal[1] = 1.0f;
            vertices[idx].normal[2] = 0.0f;
            vertices[idx].normal[3] = 0.0f; // Padding

            // U, V
            vertices[idx].texcoords[0] = (float)x / float(N);
            vertices[idx].texcoords[1] = (float)z / float(N);
            vertices[idx].texcoords[2] = 0.0f; // Padding
            vertices[idx].texcoords[3] = 0.0f; // Padding
        }
    }

    // INDEX DATA
    indices.resize(indexCount);

    for (int z = 0; z <= N; z++)
    {
        for (int x = 0; x <= N; x++)
        {
            uint32_t topLeft = z * (N + 1) + x;
            uint32_t topRight = topLeft + 1;
            uint32_t bottomLeft = (z + 1) * (N + 1) + x;
            uint32_t bottomRight = bottomLeft + 1;

            // 1st Triangle
            indices.push_back(topLeft);
            indices.push_back(bottomLeft);
            indices.push_back(topRight);

            // 2nd Triangle
            indices.push_back(topRight);
            indices.push_back(bottomLeft);
            indices.push_back(bottomRight);
        }
    }

    indexCount = static_cast<uint32_t>(indices.size());

}

VkResult Ocean::createBuffers()
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //! VERTEX BUFFER
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 5
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = vertexData.vkDeviceSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; //* Storage for compute shader

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

    vkResult = vkMapMemory(vkDevice, vertexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &vertexMappedPtr);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Buffer\n", __func__);

    memcpy(vertexMappedPtr, vertices.data(), vertices.size() * sizeof(Vertex));
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! INDEX BUFFER
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = indexData.vkDeviceSize;
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

    vkResult = vkMapMemory(vkDevice, indexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &indexMappedPtr);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Index Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Index Buffer\n", __func__);
    
    memcpy(indexMappedPtr, indices.data(), indices.size() * sizeof(uint32_t));

    //! ---------------------------------------------------------------------------------------------------------------------------------

    //! FFT Displacement Interleaved Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = fftData.vkDeviceSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &fftData.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For FFT Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For FFT Buffer\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, fftData.vkBuffer, &vkMemoryRequirements);

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
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &fftData.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For FFT Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For FFT Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, fftData.vkBuffer, fftData.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For FFT Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For FFT Buffer\n", __func__);

    vkResult = vkMapMemory(vkDevice, fftData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &fftMappedPtr);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For FFT Displacement Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For FFT Displacement Buffer\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

bool Ocean::initializeFFT()
{
    // Code
    memset((void*)&vkFFTConfiguration, 0, sizeof(VkFFTConfiguration));
    vkFFTConfiguration.FFTdim = 2;  // 2D FFT
    vkFFTConfiguration.size[0] = oceanSettings.tileSize;
    vkFFTConfiguration.size[1] = oceanSettings.tileSize;

    vkFFTConfiguration.device = &vkDevice;
    vkFFTConfiguration.physicalDevice = &vkPhysicalDevice_selected;
    vkFFTConfiguration.queue = &vkQueue;
    vkFFTConfiguration.commandPool = &vkCommandPool;
    vkFFTConfiguration.fence = &vkFence_array[0];

    vkFFTConfiguration.bufferSize = &fftData.vkDeviceSize;
    vkFFTConfiguration.buffer = &fftData.vkBuffer;

    //* Batch of 5 transforms
    vkFFTConfiguration.performR2C = 0;
    vkFFTConfiguration.isInputFormatted = 0;
    vkFFTConfiguration.numberBatches = 5;

    VkFFTResult vkFFTResult = initializeVkFFT(&vkFFTApplication, vkFFTConfiguration);
    if (vkFFTResult != VKFFT_SUCCESS)
    {
        fprintf(gpFile, "initializeVkFFT() Failed : %d\n", vkFFTResult);
        return false;
    }

    return true;
}

VkResult Ocean::createComputeDescriptorSetLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Initialize VkDescriptorSetLayoutBinding
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding_array[2];
    memset((void*)vkDescriptorSetLayoutBinding_array, 0, sizeof(VkDescriptorSetLayoutBinding) * _ARRAYSIZE(vkDescriptorSetLayoutBinding_array));

    vkDescriptorSetLayoutBinding_array[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkDescriptorSetLayoutBinding_array[0].binding = 0;   //! Mapped with layout(binding = 0) in vertex shader
    vkDescriptorSetLayoutBinding_array[0].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    vkDescriptorSetLayoutBinding_array[0].pImmutableSamplers = NULL;

    vkDescriptorSetLayoutBinding_array[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkDescriptorSetLayoutBinding_array[1].binding = 1;   //! Mapped with layout(binding = 1) in fragment shader
    vkDescriptorSetLayoutBinding_array[1].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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
    vkResult = vkCreateDescriptorSetLayout(vkDevice, &vkDescriptorSetLayoutCreateInfo, NULL, &vkDescriptorSetLayout_compute);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Failed For Compute : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Succeeded For Compute\n", __func__);

    return vkResult;
}

VkResult Ocean::createComputePipelineLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    VkPushConstantRange vkPushConstantRange;
    memset((void*)&vkPushConstantRange, 0, sizeof(VkPushConstantRange));
    vkPushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    vkPushConstantRange.offset = 0;
    vkPushConstantRange.size = sizeof(PushData);

    //* Step - 3
    VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo;
    memset((void*)&vkPipelineLayoutCreateInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
    vkPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkPipelineLayoutCreateInfo.pNext = NULL;
    vkPipelineLayoutCreateInfo.flags = 0;
    vkPipelineLayoutCreateInfo.setLayoutCount = 1;
    vkPipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout_compute;
    vkPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    vkPipelineLayoutCreateInfo.pPushConstantRanges = &vkPushConstantRange;

    //* Step - 4
    vkResult = vkCreatePipelineLayout(vkDevice, &vkPipelineLayoutCreateInfo, NULL, &vkPipelineLayout_compute);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Failed For Compute : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Succeeded For Compute\n", __func__);

    return vkResult;
}

VkResult Ocean::createComputeDescriptorPool(void)
{
    // Variable Declarations
    VkResult vkResult;

    // Code

    //* Vulkan expects decriptor pool size before creating actual descriptor pool
    VkDescriptorPoolSize vkDescriptorPoolSize_array[1];
    memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * _ARRAYSIZE(vkDescriptorPoolSize_array));

    vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkDescriptorPoolSize_array[0].descriptorCount = 2;

    //* Create the pool
    VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
    memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkDescriptorPoolCreateInfo.pNext = NULL;
    vkDescriptorPoolCreateInfo.flags = 0;
    vkDescriptorPoolCreateInfo.poolSizeCount = 1;
    vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSize_array;
    vkDescriptorPoolCreateInfo.maxSets = 1;

    vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool_compute);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Failed For Compute: %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Succeeded For Compute\n", __func__);

    return vkResult;
}

VkResult Ocean::createComputeDescriptorSet(void)
{
    // Variable Declarations
    VkResult vkResult;

    // Code

    //* Initialize DescriptorSetAllocationInfo
    VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo;
    memset((void*)&vkDescriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
    vkDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vkDescriptorSetAllocateInfo.pNext = NULL;
    vkDescriptorSetAllocateInfo.descriptorPool = vkDescriptorPool_compute;
    vkDescriptorSetAllocateInfo.descriptorSetCount = 1;
    vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout_compute;

    vkResult = vkAllocateDescriptorSets(vkDevice, &vkDescriptorSetAllocateInfo, &vkDescriptorSet_compute);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Failed For Compute : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Succeeded For Compute\n", __func__);

    //* Describe whether we want buffer as uniform or image as uniform
    VkDescriptorBufferInfo vkDescriptorBufferInfo_array[2];
    memset((void*)vkDescriptorBufferInfo_array, 0, sizeof(VkDescriptorBufferInfo) * _ARRAYSIZE(vkDescriptorBufferInfo_array));

    //! FFT Displacement Buffer
    vkDescriptorBufferInfo_array[0].buffer = fftData.vkBuffer;
    vkDescriptorBufferInfo_array[0].offset = 0;
    vkDescriptorBufferInfo_array[0].range = fftData.vkDeviceSize;

    //! Vertex Buffer
    vkDescriptorBufferInfo_array[1].buffer = vertexData.vkBuffer;
    vkDescriptorBufferInfo_array[1].offset = 0;
    vkDescriptorBufferInfo_array[1].range = vertexData.vkDeviceSize;

    VkWriteDescriptorSet vkWriteDescriptorSet_array[2];
    memset((void*)vkWriteDescriptorSet_array, 0, sizeof(VkWriteDescriptorSet) * _ARRAYSIZE(vkWriteDescriptorSet_array));

    //! FFT Displacement Buffer
    vkWriteDescriptorSet_array[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[0].pNext = NULL;
    vkWriteDescriptorSet_array[0].dstSet = vkDescriptorSet_compute;
    vkWriteDescriptorSet_array[0].dstArrayElement = 0;
    vkWriteDescriptorSet_array[0].dstBinding = 0;
    vkWriteDescriptorSet_array[0].descriptorCount = 1;
    vkWriteDescriptorSet_array[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkWriteDescriptorSet_array[0].pBufferInfo = &vkDescriptorBufferInfo_array[0];
    vkWriteDescriptorSet_array[0].pImageInfo = NULL;
    vkWriteDescriptorSet_array[0].pTexelBufferView = NULL;

    //! Vertex Buffer
    vkWriteDescriptorSet_array[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[1].pNext = NULL;
    vkWriteDescriptorSet_array[1].dstSet = vkDescriptorSet_compute;
    vkWriteDescriptorSet_array[1].dstArrayElement = 0;
    vkWriteDescriptorSet_array[1].dstBinding = 1;
    vkWriteDescriptorSet_array[1].descriptorCount = 1;
    vkWriteDescriptorSet_array[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkWriteDescriptorSet_array[1].pBufferInfo = &vkDescriptorBufferInfo_array[1];
    vkWriteDescriptorSet_array[1].pImageInfo = NULL;
    vkWriteDescriptorSet_array[1].pTexelBufferView = NULL;

    vkUpdateDescriptorSets(vkDevice, _ARRAYSIZE(vkWriteDescriptorSet_array), vkWriteDescriptorSet_array, 0, NULL);

    return vkResult;
}

VkResult Ocean::createComputePipeline(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Code

    //! Shader Stage State
    VkPipelineShaderStageCreateInfo vkPipelineShaderStageCreateInfo;
    memset((void*)&vkPipelineShaderStageCreateInfo, 0, sizeof(VkPipelineShaderStageCreateInfo));

    //* Compute Shader
    vkPipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo.pNext = NULL;
    vkPipelineShaderStageCreateInfo.flags = 0;
    vkPipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    vkPipelineShaderStageCreateInfo.module = vkShaderModule_compute_shader;
    vkPipelineShaderStageCreateInfo.pName = "main";
    vkPipelineShaderStageCreateInfo.pSpecializationInfo = NULL;

    //* Pipeline Cache
    VkPipelineCacheCreateInfo vkPipelineCacheCreateInfo;
    memset((void*)&vkPipelineCacheCreateInfo, 0, sizeof(VkPipelineCacheCreateInfo));
    vkPipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkPipelineCacheCreateInfo.pNext = NULL;
    vkPipelineCacheCreateInfo.flags = 0;
    
    VkPipelineCache vkPipelineCache_compute = VK_NULL_HANDLE;
    vkResult = vkCreatePipelineCache(vkDevice, &vkPipelineCacheCreateInfo, NULL, &vkPipelineCache_compute);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Failed For Compute : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Succeeded For Compute\n", __func__);

    //! Create actual Compute Pipeline
    VkComputePipelineCreateInfo vkComputePipelineCreateInfo;
    memset((void*)&vkComputePipelineCreateInfo, 0, sizeof(VkComputePipelineCreateInfo));
    vkComputePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    vkComputePipelineCreateInfo.pNext = NULL;
    vkComputePipelineCreateInfo.flags = 0;
    vkComputePipelineCreateInfo.stage = vkPipelineShaderStageCreateInfo;
    vkComputePipelineCreateInfo.layout = vkPipelineLayout_compute;

    vkResult = vkCreateComputePipelines(vkDevice, vkPipelineCache_compute, 1, &vkComputePipelineCreateInfo, NULL, &vkPipeline_compute);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateComputePipelines() Failed For Compute : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateComputePipelines() Succeeded For Compute\n", __func__);


    return vkResult;
}

void Ocean::update(double deltaTime)
{
    // Code
    time += deltaTime * simulationSpeed;

    int N = oceanSettings.tileSize;

    // Build Fourier Spectrum
    generateSpectrum(time);

    // Copy Spectrum into mapped FFT Displacement Buffer
    memcpy(fftMappedPtr, tildeData.data(), fftData.vkDeviceSize);


    
}

void Ocean::reloadSettings(OceanSettings newSettings)
{
    // Code
    oceanSettings = newSettings;

    int N = oceanSettings.tileSize;

    h0_k.resize(N * N);
    tildeData.resize(5 * N * N * 2);

    //vkResult = vkMapMemory(vkDevice, fftData.vkDeviceMemory, 0, fftData.vkDeviceSize, 0, &fftMappedPtr);
    //if (vkResult != VK_SUCCESS)
    //    fprintf(gpFile, "%s() => vkMapMemory() Failed For FFT Displacement Buffer : %d !!!\n", __func__, vkResult);
    //vkResult = vkMapMemory(vkDevice, vertexData.vkDeviceMemory, 0, VK_WHOLE_SIZE, 0, &vertexMappedPtr);
    //if (vkResult != VK_SUCCESS)
    //    fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Buffer : %d !!!\n", __func__, vkResult);
    //vkResult = vkMapMemory(vkDevice, indexData.vkDeviceMemory, 0, VK_WHOLE_SIZE, 0, &indexMappedPtr);
    //if (vkResult != VK_SUCCESS)
    //    fprintf(gpFile, "%s() => vkMapMemory() Failed For Index Buffer : %d !!!\n", __func__, vkResult);

    createGrid();

    generateH0();
}

// Helper to map (m,n) to linear index
inline static int idx2(int m, int n, int N)
{
    return m * N + n;
}

// Phillips spectrum
double Ocean::phillipsSpectrum(const glm::vec2& K) const
{
    // Code
    double kLength = glm::length(K);
    if (kLength < 1e-6)
        return 0.0;

    double k2 = kLength * kLength;
    double k4 = k2 * k2;

    double kw = glm::dot(glm::normalize(K), glm::normalize(oceanSettings.windDirection));
    double lw = oceanSettings.windSpeed * oceanSettings.windSpeed / Ocean::G;
    double kw2 = kw * kw;
    double result = oceanSettings.amplitude * kw2 * std::exp(-1.0f / (k2 * lw * lw)) / k4;

    double damp = 0.001;
    result = result * std::exp(-k2 * damp * damp);

    return result;
}

// Dispersion Relation (Deep Water)
double Ocean::dispersion(const glm::vec2& K)
{
    return std::sqrt(glm::length(K) * Ocean::G);
}

void Ocean::generateH0()
{
    // Code
    int N = oceanSettings.tileSize;
    double L = float(N) * 1.0;

    for (int m = 0; m < N; m++)
    {
        for (int n = 0; n < N; n++)
        {
            // Wave vector => kx, kz 
            double kx = (n - N / 2.0) * (2.0 * M_PI / L);
            double kz = (m - N / 2.0) * (2.0 * M_PI / L);
            glm::vec2 K(kx, kz);

            double p = phillipsSpectrum(K);

            // Gaussian Noise
            double er = gaussianDistribution(rng);
            double ei = gaussianDistribution(rng);

            // Scale by sqrt(p / 2)
            double scale = std::sqrt(p * 0.5);
            std::complex<double> h0 = std::complex<double>(er * scale, ei * scale);
            h0_k[idx2(m, n, N)] = h0;
        }
    }
}

void Ocean::generateSpectrum(double deltaTime)
{
    // Code
    int N = oceanSettings.tileSize;

    // Domain Length
    double L = float(N) * 1.0;

    for (int m = 0; m < N; m++)
    {
        for (int n = 0; n < N; n++)
        {
            int i = idx2(m, n, N);

            // Wave vector => kx, kz 
            double kx = (n - N / 2.0) * (2.0 * M_PI / L);
            double kz = (m - N / 2.0) * (2.0 * M_PI / L);
            glm::vec2 K(kx, kz);

            double kLength = glm::length(K);
            if (kLength < 1e-6)
                kLength = 1e-6;

            // Angular Frequency
            double omega = dispersion(K);

            // Index of -k => map (m,n) => (-m % N, -n % N)
            int negM = (N - m) % N;
            int negN = (N - n) % N;
            int negI = idx2(negM, negN, N);

            std::complex<double> h0 = h0_k[i];
            std::complex<double> h0_neg = h0_k[negI];

            // h(k,t) = h0(k) e^{i ω t} + conj(h0(-k)) e^{-i ω t}
            std::complex<double> posExp = std::exp(std::complex<double>(0.0, omega * deltaTime));
            std::complex<double> negExp = std::exp(std::complex<double>(0.0, -omega * deltaTime));
            std::complex<double> h = h0 * posExp + std::conj(h0_neg) * negExp;

            // Height => h
            // Gradients => h_xGradient = i * kx * h
            // Displacements => h_xDisplacement = i * (-kx / k_len) * h  (if k_len == 0 -> 0)
            std::complex<double> I(0.0, 1.0);

            std::complex<double> h_y = h;
            std::complex<double> h_xGradient = I * kx * h;
            std::complex<double> h_zGradient = I * kz * h;

            std::complex<double> h_xDisplacement, h_zDisplacement;
            if (kLength > 1e-6)
            {
                h_xDisplacement = I * (-kx / kLength) * h;
                h_zDisplacement = I * (-kz / kLength) * h;
            }
            else
            {
                h_xDisplacement = std::complex<double>(0.0, 0.0);
                h_zDisplacement = std::complex<double>(0.0, 0.0);
            }

            size_t planeSize = size_t(N) * size_t(N) * 2;

            // Plane 0 : h_y
            size_t base0 = size_t(0) * planeSize + size_t(i) * 2;
            tildeData[base0 + 0] = h_y.real();
            tildeData[base0 + 1] = h_y.imag();

            // Plane 1 : h_xDisplacement
            size_t base1 = size_t(1) * planeSize + size_t(i) * 2;
            tildeData[base1 + 0] = h_xDisplacement.real();
            tildeData[base1 + 1] = h_xDisplacement.imag();

            // Plane 2 : h_zDisplacement
            size_t base2 = size_t(2) * planeSize + size_t(i) * 2;
            tildeData[base2 + 0] = h_zDisplacement.real();
            tildeData[base2 + 1] = h_zDisplacement.imag();

            // Plane 3 : h_xGradient
            size_t base3 = size_t(3) * planeSize + size_t(i) * 2;
            tildeData[base3 + 0] = h_xGradient.real();
            tildeData[base3 + 1] = h_xGradient.imag();

            // Plane 4 : h_zGradient
            size_t base4 = size_t(4) * planeSize + size_t(i) * 2;
            tildeData[base4 + 0] = h_zGradient.real();
            tildeData[base4 + 1] = h_zGradient.imag();

        }
    }
}

void Ocean::unmapMemory(VkDeviceMemory& vkDeviceMemory)
{
    vkUnmapMemory(vkDevice, vkDeviceMemory);
}

Ocean::~Ocean()
{
    if (vkDevice)
        vkDeviceWaitIdle(vkDevice);

    if (vkPipelineLayout_compute)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_compute, NULL);
        vkPipelineLayout_compute = VK_NULL_HANDLE;
    }

    if (vkPipeline_compute)
    {
        vkDestroyPipeline(vkDevice, vkPipeline_compute, NULL);
        vkPipeline_compute = VK_NULL_HANDLE;
    }

    if (vkDescriptorPool_compute)
    {
        vkDestroyDescriptorPool(vkDevice, vkDescriptorPool_compute, NULL);
        vkDescriptorPool_compute = VK_NULL_HANDLE;
        vkDescriptorSet_compute = VK_NULL_HANDLE;
    }

    if (vkDescriptorSetLayout_compute)
    {
        vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout_compute, NULL);
        vkDescriptorSetLayout_compute = VK_NULL_HANDLE;
    }

    deleteVkFFT(&vkFFTApplication);

    if (fftData.vkDeviceMemory)
    {
        if (fftMappedPtr)
        {
            unmapMemory(fftData.vkDeviceMemory);
            fftMappedPtr = NULL;
        }
        
        vkFreeMemory(vkDevice, fftData.vkDeviceMemory, NULL);
        fftData.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For fftData.vkDeviceMemory\n", __func__);
    }

    if (fftData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, fftData.vkBuffer, NULL);
        fftData.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For fftData.vkBuffer\n", __func__);
    }
    
    if (indexData.vkDeviceMemory)
    {
        if (indexMappedPtr)
        {
            unmapMemory(indexData.vkDeviceMemory);
            indexMappedPtr = NULL;
        }

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

    if (vertexData.vkDeviceMemory)
    {
        if (vertexMappedPtr)
        {
            unmapMemory(vertexData.vkDeviceMemory);
            vertexMappedPtr = NULL;
        }

        vkFreeMemory(vkDevice, vertexData.vkDeviceMemory, NULL);
        vertexData.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData.vkDeviceMemory\n", __func__);
    }

    if (vertexData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData.vkBuffer, NULL);
        vertexData.vkBuffer = VK_NULL_HANDLE;
        
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData.vkBuffer\n", __func__);
    }
}
