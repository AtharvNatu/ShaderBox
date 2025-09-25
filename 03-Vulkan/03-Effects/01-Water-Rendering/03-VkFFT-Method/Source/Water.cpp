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

    VkDeviceSize complexSize = sizeof(float) * 2;
    VkDeviceSize planeSize = tileSize * tileSize * complexSize;

    memset((void*)&vertexData, 0, sizeof(BufferData));
    memset((void*)&indexData, 0, sizeof(BufferData));
    memset((void*)&fftData, 0, sizeof(BufferData));

    vertexData.vkDeviceSize = vertexCount * sizeof(Vertex);
    indexData.vkDeviceSize = indexCount * sizeof(uint32_t);
    fftData.vkDeviceSize = planeSize * 5; // For 5 buffers (Displacement - x,y,z | Gradient - x,y)
    fprintf(gpFile, "vertexData.vkDeviceSize = %lld\n", vertexData.vkDeviceSize);
    fprintf(gpFile, "indexData.vkDeviceSize = %lld\n", indexData.vkDeviceSize);
    fprintf(gpFile, "fftData.vkDeviceSize = %lld\n", fftData.vkDeviceSize);
    
    pushData.tileSize = settings.tileSize;
    pushData.vertexDistance = vertexDistance;
    pushData.choppiness = choppiness;
    pushData.normalRoughness = normalRoughness;
    pushData.half = float(tileSize) * 0.5f;

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

    // reloadSettings(settings);
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

    vkResult = vkMapMemory(vkDevice, vertexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &vertexMappedData);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For vertexMappedData : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For vertexMappedData\n", __func__);
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

    vkResult = vkMapMemory(vkDevice, indexData.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &indexMappedData);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For indexMappedData : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For indexMappedData\n", __func__);
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
    //! ---------------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

bool Ocean::initializeFFT(VkCommandBuffer& commandBuffer)
{
    // Code
    memset((void*)&vkFFTConfiguration, 0, sizeof(VkFFTConfiguration));
    vkFFTConfiguration.FFTdim = 2;  // 2D FFT
    vkFFTConfiguration.size[0] = oceanSettings.tileSize;
    vkFFTConfiguration.size[1] = oceanSettings.tileSize;
    vkFFTConfiguration.device = &vkDevice;
    vkFFTConfiguration.physicalDevice = &vkPhysicalDevice_selected;
    vkFFTConfiguration.queue = &vkQueue;
    vkFFTConfiguration.commandBuffer = &commandBuffer;
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

    // 0 -> Vertex UBO
    // 1 -> Water Surface UBO
    // 2 -> Displacement Map
    // 3 -> Normal Map
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
    memcpy(vertexMappedData, vertices.data(), vertexData.vkDeviceSize); 
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
    memcpy(indexMappedData, indices.data(), indexData.vkDeviceSize); 

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

    if (fftData.vkDeviceMemory)
    {
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
