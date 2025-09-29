#include "Ocean.h"

static std::default_random_engine rng;
static std::normal_distribution<double> gaussianDistribution(0.0, 1.0);
static auto lastFrame = std::chrono::high_resolution_clock::now();

Ocean::Ocean() 
{
    float movementSpeed = 1.0;
    float rotationSpeed = 30.0;

    camera = new Camera(
        glm::vec3(0.0, 1.0, 1.0),
        0.0, 0.0, 45.0f, 1260.0f / 1080.0f, 0.01, 1000.0, 
        rotationSpeed, movementSpeed
    );

}

VkResult Ocean::initialize()
{
    // Code
    const uint32_t tileSize = oceanSettings.tileSize;
    int nPlus1 = tileSize + 1;
    const uint32_t vertexCount = (tileSize + 1) * (tileSize + 1);
    const uint32_t indexCount = tileSize * tileSize * 6;

    vertexBufferSize = vertexCount * sizeof(Vertex);
    indexBufferSize = indexCount * sizeof(uint32_t);

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

            if (x < oceanSettings.tileSize && z < oceanSettings.tileSize)
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
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Succeeded For Ocean\n", __func__);

    vkResult = createDescriptorPool();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Succeeded For Ocean\n", __func__);

    vkResult = createDescriptorSet();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Succeeded For Ocean\n", __func__);

    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createPipeline() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createPipeline() Succeeded For Ocean\n", __func__);

    reloadSettings(oceanSettings);

    return vkResult;
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

VkResult Ocean::createUniformBuffer(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //! VBO
    //! ---------------------------------------------------------------------------------------------------------
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(UBO);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData_vbo, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData_vbo.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData_vbo.vkBuffer, &vkMemoryRequirements);

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
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData_vbo.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, uniformData_vbo.vkBuffer, uniformData_vbo.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    //! ---------------------------------------------------------------------------------------------------------

    //! Ocean Parameters
    //! ---------------------------------------------------------------------------------------------------------
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(OceanParams);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData_oceanParams, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData_oceanParams.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData_oceanParams.vkBuffer, &vkMemoryRequirements);

    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData_oceanParams.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, uniformData_oceanParams.vkBuffer, uniformData_oceanParams.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    //! ---------------------------------------------------------------------------------------------------------
    
    //! Lighting UBO
    //! ---------------------------------------------------------------------------------------------------------
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(LightingUBO);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData_lighting, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData_lighting.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData_lighting.vkBuffer, &vkMemoryRequirements);

    memset((void*)&vkMemoryAllocateInfo, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo.pNext = NULL;
    vkMemoryAllocateInfo.allocationSize = vkMemoryRequirements.size;
    vkMemoryAllocateInfo.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
            {
                vkMemoryAllocateInfo.memoryTypeIndex = i;
                break;
            }
        }

        vkMemoryRequirements.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData_lighting.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, uniformData_lighting.vkBuffer, uniformData_lighting.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    //! ---------------------------------------------------------------------------------------------------------

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

VkResult Ocean::createDescriptorSetLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Initialize VkDescriptorSetLayoutBinding

    // 0 -> Vertex UBO
    // 1 -> Ocean Parameters UBO
    // 2 -> Lighting UBO
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding_array[3];
    memset((void*)vkDescriptorSetLayoutBinding_array, 0, sizeof(VkDescriptorSetLayoutBinding) * _ARRAYSIZE(vkDescriptorSetLayoutBinding_array));

    vkDescriptorSetLayoutBinding_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[0].binding = 0;   //! Mapped with layout(binding = 0) in vertex shader
    vkDescriptorSetLayoutBinding_array[0].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    vkDescriptorSetLayoutBinding_array[0].pImmutableSamplers = NULL;

    vkDescriptorSetLayoutBinding_array[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[1].binding = 1;   //! Mapped with layout(binding = 1) in fragment shader
    vkDescriptorSetLayoutBinding_array[1].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    vkDescriptorSetLayoutBinding_array[1].pImmutableSamplers = NULL;

    vkDescriptorSetLayoutBinding_array[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[2].binding = 2;   //! Mapped with layout(binding = 2) in vertex shader
    vkDescriptorSetLayoutBinding_array[2].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkDescriptorSetLayoutBinding_array[2].pImmutableSamplers = NULL;

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

    return vkResult;
}

VkResult Ocean::createPipelineLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

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

    return vkResult;
}

VkResult Ocean::createDescriptorPool(void)
{
    // Variable Declarations
    VkResult vkResult;

    // Code

    //* Vulkan expects decriptor pool size before creating actual descriptor pool
    VkDescriptorPoolSize vkDescriptorPoolSize_array[1];
    memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * _ARRAYSIZE(vkDescriptorPoolSize_array));

    vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorPoolSize_array[0].descriptorCount = 1;

    //* Create the pool
    VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
    memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkDescriptorPoolCreateInfo.pNext = NULL;
    vkDescriptorPoolCreateInfo.flags = 0;
    vkDescriptorPoolCreateInfo.poolSizeCount = _ARRAYSIZE(vkDescriptorPoolSize_array);
    vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSize_array;
    vkDescriptorPoolCreateInfo.maxSets = 1;

    vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool_ocean);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Failed For Ocean : %d !!!\n", __func__, vkResult);

    return vkResult;
}

VkResult Ocean::createDescriptorSet(void)
{
    // Variable Declarations
    VkResult vkResult;

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

    //* Describe whether we want buffer as uniform or image as uniform
    VkDescriptorBufferInfo vkDescriptorBufferInfo_array[3];
    memset((void*)vkDescriptorBufferInfo_array, 0, sizeof(VkDescriptorBufferInfo) * _ARRAYSIZE(vkDescriptorBufferInfo_array));

    //! Vertex UBO
    vkDescriptorBufferInfo_array[0].buffer = uniformData_vbo.vkBuffer;
    vkDescriptorBufferInfo_array[0].offset = 0;
    vkDescriptorBufferInfo_array[0].range = sizeof(UBO);

    //! Ocean Parameters
    vkDescriptorBufferInfo_array[1].buffer = uniformData_oceanParams.vkBuffer;
    vkDescriptorBufferInfo_array[1].offset = 0;
    vkDescriptorBufferInfo_array[1].range = sizeof(OceanParams);

    //! Lighting UBO
    vkDescriptorBufferInfo_array[2].buffer = uniformData_lighting.vkBuffer;
    vkDescriptorBufferInfo_array[2].offset = 0;
    vkDescriptorBufferInfo_array[2].range = sizeof(LightingUBO);

    /* Update above descriptor set directly to the shader
    There are 2 ways :-
        1) Writing directly to the shader
        2) Copying from one shader to another shader
    */
    VkWriteDescriptorSet vkWriteDescriptorSet_array[3];
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

    //! Ocean Parameters
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

    //! Lighting UBO
    vkWriteDescriptorSet_array[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[2].pNext = NULL;
    vkWriteDescriptorSet_array[2].dstSet = vkDescriptorSet_ocean;
    vkWriteDescriptorSet_array[2].dstArrayElement = 0;
    vkWriteDescriptorSet_array[2].descriptorCount = 1;
    vkWriteDescriptorSet_array[2].dstBinding = 2;
    vkWriteDescriptorSet_array[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkWriteDescriptorSet_array[2].pBufferInfo = &vkDescriptorBufferInfo_array[2];
    vkWriteDescriptorSet_array[2].pImageInfo = NULL;
    vkWriteDescriptorSet_array[2].pTexelBufferView = NULL;

    vkUpdateDescriptorSets(vkDevice, _ARRAYSIZE(vkWriteDescriptorSet_array), vkWriteDescriptorSet_array, 0, NULL);

    return vkResult;
}

VkResult Ocean::createPipeline(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Code

    //! Vertex Input State
    VkVertexInputBindingDescription vkVertexInputBindingDescription;
    memset((void*)&vkVertexInputBindingDescription, 0, sizeof(VkVertexInputBindingDescription));
    vkVertexInputBindingDescription.binding = 0;
    vkVertexInputBindingDescription.stride = sizeof(Vertex);
    vkVertexInputBindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[4];
    memset((void*)vkVertexInputAttributeDescription_array, 0, sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));

    //! Position
    vkVertexInputAttributeDescription_array[0].binding = 0;
    vkVertexInputAttributeDescription_array[0].location = 0;
    vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[0].offset = offsetof(Vertex, position);

    //! Color
    vkVertexInputAttributeDescription_array[1].binding = 0;
    vkVertexInputAttributeDescription_array[1].location = 1;
    vkVertexInputAttributeDescription_array[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[1].offset = offsetof(Vertex, color);

    //! Normals
    vkVertexInputAttributeDescription_array[2].binding = 0;
    vkVertexInputAttributeDescription_array[2].location = 2;
    vkVertexInputAttributeDescription_array[2].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[2].offset = offsetof(Vertex, normal);

    //! Texture
    vkVertexInputAttributeDescription_array[3].binding = 0;
    vkVertexInputAttributeDescription_array[3].location = 3;
    vkVertexInputAttributeDescription_array[3].format = VK_FORMAT_R32G32_SFLOAT;
    vkVertexInputAttributeDescription_array[3].offset = offsetof(Vertex, texcoords);

    VkPipelineVertexInputStateCreateInfo vkPipelineVertexInputStateCreateInfo;
    memset((void*)&vkPipelineVertexInputStateCreateInfo, 0, sizeof(VkPipelineVertexInputStateCreateInfo));
    vkPipelineVertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vkPipelineVertexInputStateCreateInfo.pNext = NULL;
    vkPipelineVertexInputStateCreateInfo.flags = 0;
    vkPipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = 1;
    vkPipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = &vkVertexInputBindingDescription;
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
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineCache() Succeeded\n", __func__);

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

    //* Destroy Pipeline Cache
    if (vkPipelineCache)
    {
        vkDestroyPipelineCache(vkDevice, vkPipelineCache, NULL);
        vkPipelineCache = VK_NULL_HANDLE;
    }

    return vkResult;
}

VkResult Ocean::updateUniformBuffer(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    UBO ubo;
    memset((void*)&ubo, 0, sizeof(UBO));

    //! Update Matrices
    glm::mat4 waveViewProjectionMatrix = camera->getViewProjection(true);
    glm::mat4 waterMatrix = glm::mat4(1.0f);
    waterMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -6.0f));

    ubo.viewProjectionMatrix = waveViewProjectionMatrix;
    ubo.modelMatrix = waterMatrix;
    ubo.cameraPosition = glm::vec4(camera->getPosition(), 1.0f);

    //! Map Uniform Buffer
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, uniformData_vbo.vkDeviceMemory, 0, sizeof(UBO), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer (Vertex UBO) : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &ubo, sizeof(UBO));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData_vbo.vkDeviceMemory);

    // Ocean Parameters
    OceanParams oceanParams;
    memset((void*)&oceanParams, 0, sizeof(OceanParams));

    oceanParams.time = this->dt;
    oceanParams.amplitude = 0.05f;
    oceanParams.wavelength = 20.0f;
    oceanParams.speed = 1.0f;
    oceanParams.steepness = 0.5f;
    glm::vec2 dirData = glm::normalize(glm::vec2(1.0f, 0.3f));
    oceanParams.direction = glm::vec4(dirData, 1.0f, 1.0f);

    //! Map Uniform Buffer
    data = NULL;
    vkResult = vkMapMemory(vkDevice, uniformData_oceanParams.vkDeviceMemory, 0, sizeof(OceanParams), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer (Vertex UBO) : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &oceanParams, sizeof(OceanParams));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData_oceanParams.vkDeviceMemory);

    // Lighting UBO
    LightingUBO lightingUBO;
    memset((void*)&lightingUBO, 0, sizeof(LightingUBO));

    glm::vec3 sunData = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f));
    lightingUBO.sunDirection = glm::vec4(sunData, 1.0f);
    lightingUBO.sunColor = glm::vec4(1.0f, 0.95f, 0.8f, 1.0f);
    lightingUBO.horizonColor = glm::vec4(0.2f, 0.4f, 0.7f, 1.0f);
    lightingUBO.deepColor = glm::vec4(0.0f, 0.05f, 0.15f, 1.0f);

    //! Map Uniform Buffer
    data = NULL;
    vkResult = vkMapMemory(vkDevice, uniformData_oceanParams.vkDeviceMemory, 0, sizeof(LightingUBO), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer (Vertex UBO) : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &lightingUBO, sizeof(LightingUBO));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData_oceanParams.vkDeviceMemory);

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

    //! Bind with Vertex Buffer
    VkDeviceSize vkDeviceSize_offset_array[1];
    memset((void*)vkDeviceSize_offset_array, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_array));
    vkCmdBindVertexBuffers(
        commandBuffer,
        0,
        1,
        &vertexData.vkBuffer,
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
    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
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

    // Larsson Equation
    for (int m = 0; m < N; m++)
    {
        for (int n = 0; n < N; n++)
        {
            int index = m * N + n;
            int sign = (m + n) % 2 == 0 ? 1 : -1;
            h_yDisplacement[index] /= sign * (N * N);
            h_xDisplacement[index] /= sign * (N * N);
            h_zDisplacement[index] /= sign * (N * N);
            h_xGradient[index] /= sign * (N * N);
            h_zGradient[index] /= sign * (N * N);
        }
    }

    float baseAmplitude = 0.004f;
    int maxWavesInFlight = 1;
    
    for (int m = 0; m < N; m++)
    {
        for (int n = 0; n < N; n++)
        {
            int index = m * N + n;
            float k = twoPi / rippleLength;
            float x = (n - N / 2.0f);
            float z = (m - N / 2.0f);
            glm::vec2 X(x, z);
            glm::vec2 K(k);

            float dist1 = rippleLength * simulationTime / period;
            float dist2 = abs(dist1 - glm::length(X));

            float maxWaveDistance = rippleLength * maxWavesInFlight;
            if (dist2 < maxWaveDistance / 2.0f)
            {
                float t = maxWaveDistance / 2.0 - dist2;
                float amplitude = baseAmplitude * sin(glm::half_pi<float>() * t);

                float fDispersion = dispersion(K) / glm::length(K);
                float value = glm::length(X) * k + fDispersion * -simulationTime;

                h_yDisplacement[index] += amplitude * (-1 + 2 * sin(value));
                h_xGradient[index] += amplitude * cos(value);
                h_zGradient[index] += amplitude * cos(value);
            }

        }
    }
    
    updateVertices();

    VkResult vkResult = updateUniformBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "OCEAN::%s() => updateUniformBuffer() Failed : %d !!!\n", __func__, vkResult);
    }
}

void Ocean::updateVertices()
{
    // Code
    const uint32_t N = oceanSettings.tileSize;
    int Nplus1 = N + 1;

    for (int z = 0; z < Nplus1; z++)
    {
        for (int x = 0; x < Nplus1; x++)
        {
            int iV = z * Nplus1 + x;
            int iD = (z % N) * N + x % N;

            glm::vec3 originPosition = glm::vec3(
                -0.5 + x * vertexDistance / float(N),
                0,
                -0.5 + z * vertexDistance / float(N)
            );

            glm::vec3 displacementData = glm::vec3(
                choppiness * h_xDisplacement[iD].real(), 
                h_yDisplacement[iD].real(), 
                choppiness * h_zDisplacement[iD].real()
            );

            vertices[iV].position = originPosition + displacementData;

            double ex = h_xGradient[iD].real();
            double ez = h_zGradient[iD].real();

            vertices[iV].normal = glm::vec3(
                -ex * normalRoughness,
                1.0,
                -ez * normalRoughness
            );
        }
    }

    // for (int z = 0; z < Nplus1; z++)
    // {
    //     for (int x = 0; x < Nplus1; x++)
    //     {
    //         int iV = z * Nplus1 + x;
    //         int iD = (z % N) * N + x % N;

    //         double ex = h_xGradient[iD].real();
    //         double ez = h_zGradient[iD].real();

    //         vertices[iV].normal = glm::vec3(
    //             -ex * normalRoughness,
    //             1.0,
    //             -ez * normalRoughness
    //         );
    //     }
    // }

    //! Update Vertex Buffer Data
    memcpy(vertexMappedData, vertices.data(), vertexBufferSize); 
}

void Ocean::reloadSettings(OceanSettings newSettings)
{
    if (hostData)
        delete[] hostData;

    oceanSettings = newSettings;
    int N = newSettings.tileSize;
    float length = newSettings.length;
    
    hostData = new std::complex<double>[7 * N * N];
    h0_tk = hostData + 0 * N * N;                   // h0_tilde(k)
    h0_tmk = hostData + 1 * N * N;                  // h0_tilde(-k)
    h_yDisplacement = hostData + 2 * N * N;         // h(k, x, t)
    h_xDisplacement = hostData + 3 * N * N;         // x-displacement of h(k, x, t)
    h_zDisplacement = hostData + 4 * N * N;         // z-displacement of h(k, x, t)
    h_xGradient = hostData + 5 * N * N;             // x-gradient of h(k, x, t)
    h_zGradient = hostData + 6 * N * N;             // z-gradient of h(k, x, t)

    for (int m = 0; m < N; m++) 
    {
        for (int n = 0; n < N; n++) 
        {
            int index = m * N + n;

            float kx = (n - N / 2.f) * twoPi / length;
            float kz = (m - N / 2.f) * twoPi / length;
            glm::vec2 k(kx, kz);

            h0_tk[index] = h0_tilde(k);
            h0_tmk[index] = h0_tilde(-k);
        }
    }

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

    for (int z = 0; z < Nplus1; z++) 
    {
        for (int x = 0; x < Nplus1; x++) 
        {
            if (x < N && z < N) 
            {
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

void Ocean::unmapMemory(VkDeviceMemory& vkDeviceMemory)
{
    vkUnmapMemory(vkDevice, vkDeviceMemory);
}

VkResult Ocean::resize(int width, int height)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

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

//! OCEAN WAVE RELATED (Tessendorf Method)

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

std::complex<double> Ocean::h0_tilde(const glm::vec2& K) 
{
    double er = gaussianDistribution(rng);
    double ei = gaussianDistribution(rng);

    return sqrt(phillipsSpectrum(K)) * (std::complex(er, ei)) / sqrt(2.0);
}

std::complex<double> Ocean::h_tilde(const std::complex<double>& h0_tk, const std::complex<double>& h0_tmk, const glm::vec2& K, double t) 
{
    double wkt = dispersion(K) * t;
    return h0_tk * exp(std::complex(0.0, wkt)) + std::conj(h0_tmk) * exp(std::complex(0.0, -wkt));
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
    if (uniformData_lighting.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData_lighting.vkDeviceMemory, NULL);
        uniformData_lighting.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (uniformData_lighting.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData_lighting.vkBuffer, NULL);
        uniformData_lighting.vkBuffer = VK_NULL_HANDLE;
    }
    
    if (uniformData_oceanParams.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData_oceanParams.vkDeviceMemory, NULL);
        uniformData_oceanParams.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (uniformData_oceanParams.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData_oceanParams.vkBuffer, NULL);
        uniformData_oceanParams.vkBuffer = VK_NULL_HANDLE;
    }
    
    if (uniformData_vbo.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData_vbo.vkDeviceMemory, NULL);
        uniformData_vbo.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (uniformData_vbo.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData_vbo.vkBuffer, NULL);
        uniformData_vbo.vkBuffer = VK_NULL_HANDLE;
    }
    
    if (indexData.vkDeviceMemory)
    {
        if (indexMappedData)
        {
            unmapMemory(indexData.vkDeviceMemory);
            indexMappedData = NULL;
        }

        vkFreeMemory(vkDevice, indexData.vkDeviceMemory, NULL);
        indexData.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (indexData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, indexData.vkBuffer, NULL);
        indexData.vkBuffer = VK_NULL_HANDLE;
        indexMappedData = NULL;
    }

    if (vertexData.vkDeviceMemory)
    {
        if (vertexMappedData)
        {
            unmapMemory(vertexData.vkDeviceMemory);
            vertexMappedData = NULL;
        }

        vkFreeMemory(vkDevice, vertexData.vkDeviceMemory, NULL);
        vertexData.vkDeviceMemory = VK_NULL_HANDLE;
    }

    if (vertexData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData.vkBuffer, NULL);
        vertexData.vkBuffer = VK_NULL_HANDLE;
        vertexMappedData = NULL;
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
