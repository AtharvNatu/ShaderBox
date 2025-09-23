#include "Grass.hpp"

Grass::Grass()
{

}

VkResult Grass::initialize()
{
    // Code
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    vkResult = createVertexBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS : %s() => createVertexBuffer() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    vkResult = createUniformBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS : %s() => createUniformBuffer() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    vkResult = createDescriptorSetLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS : %s() => createDescriptorSetLayout() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    vkResult = createPipelineLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS : %s() => createPipelineLayout() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    vkResult = createDescriptorPool();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS : %s() => createDescriptorPool() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    vkResult = createDescriptorSet();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS : %s() => createDescriptorSet() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS : %s() => createPipeline() Failed : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    return vkResult;
}

void Grass::populateData()
{
    // Grass Position
    srand(time(NULL));

    for (float x = -10.0f; x < 10.0f; x = x + 0.1f)
    {
        for (float z = -10.0f; z < 10.0f; z = z + 0.1f)
        {
            int randomNumberX = rand() % 10 + 1;
            int randomNumberZ = rand() % 10 + 1;

            glm::vec3 temp = glm::vec3(x + (float)randomNumberX / 10.0f, 0, z + (float)randomNumberZ / 10.0f);

            grassPosition.push_back(temp);
        }
    }
}

void Grass::addVertexPosition(glm::vec3 position)
{
    grassPosition.push_back(position);
}

void Grass::setGrassVertices(std::vector<glm::vec3> grassVertices)
{
    grassPosition = grassVertices;
}

VkResult Grass::createVertexBuffer()
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;
    VertexData vertexData_stagingBuffer;

    // Code 

    //! Staging Buffer => To Copy Grass Position Data To GPU
    //------------------------------------------------------------------------------------------------------------------------------
    memset((void*)&vertexData_stagingBuffer, 0, sizeof(VertexData));

    //* Step - 5
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = grassPosition.size() * sizeof(glm::vec3);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_stagingBuffer.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Staging Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Staging Vertex Position Buffer\n", __func__);

    //* Step - 7
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_stagingBuffer.vkBuffer, &vkMemoryRequirements);

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
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_stagingBuffer.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_stagingBuffer.vkBuffer, vertexData_stagingBuffer.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 11
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, vertexData_stagingBuffer.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &data);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Position Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Position Buffer\n", __func__);

    //* Step - 12
    memcpy(data, grassPosition.data(), grassPosition.size() * sizeof(glm::vec3));

    //* Step - 13
    vkUnmapMemory(vkDevice, vertexData_stagingBuffer.vkDeviceMemory);
    //------------------------------------------------------------------------------------------------------------------------------

    //! Device Local Buffer
    //------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_grass_position, 0, sizeof(VertexData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = grassPosition.size() * sizeof(glm::vec3);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    vkBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_grass_position.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Position Device-Local Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Position Device-Local Buffer\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_grass_position.vkBuffer, &vkMemoryRequirements);

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
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
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
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &vertexData_grass_position.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Vertex Position Device-Local Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Vertex Position Device-Local Buffer\n", __func__);

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, vertexData_grass_position.vkBuffer, vertexData_grass_position.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Vertex Position Device-Local Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Vertex Position Device-Local Buffer\n", __func__);
    //------------------------------------------------------------------------------------------------------------------------------

    //! Command Buffer
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo;
    memset((void*)&vkCommandBufferAllocateInfo, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO; 
    vkCommandBufferAllocateInfo.pNext = NULL;
    vkCommandBufferAllocateInfo.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    vkCommandBufferAllocateInfo.commandBufferCount = 1;
    
    VkCommandBuffer vkCommandBuffer = VK_NULL_HANDLE;
    vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo, &vkCommandBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed To Allocate Command Buffer For Buffer-Copy: %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Succeeded In Allocating Command Buffer For Buffer-Copy\n", __func__);
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    //! Step - 4 (Build Command Buffer)
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    VkCommandBufferBeginInfo vkCommandBufferBeginInfo;
    memset((void*)&vkCommandBufferBeginInfo, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo.pNext = NULL;
    vkCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer, &vkCommandBufferBeginInfo);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For Buffer-Copy: %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer);
            vkCommandBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For Buffer-Copy\n", __func__);
        }
        if (vertexData_stagingBuffer.vkDeviceMemory)
        {
            vkFreeMemory(vkDevice, vertexData_stagingBuffer.vkDeviceMemory, NULL);
            vertexData_stagingBuffer.vkDeviceMemory = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_stagingBuffer.vkDeviceMemory\n", __func__);
        }
        if (vertexData_stagingBuffer.vkBuffer)
        {
            vkDestroyBuffer(vkDevice, vertexData_stagingBuffer.vkBuffer, NULL);
            vertexData_stagingBuffer.vkBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_stagingBuffer.vkBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Succeeded For Buffer-Copy\n", __func__);
    
    VkBufferCopy vkBufferCopy;
    memset((void*)&vkBufferCopy, 0, sizeof(VkBufferCopy));
    vkBufferCopy.srcOffset = 0; //? SRC = 0, DST = 0 => Entire buffer
    vkBufferCopy.dstOffset = 0;
    vkBufferCopy.size = grassPosition.size() * sizeof(glm::vec3);

    vkCmdCopyBuffer(
        vkCommandBuffer,
        vertexData_stagingBuffer.vkBuffer,
        vertexData_grass_position.vkBuffer,
        1,                   //* Count of regions
        &vkBufferCopy        //* Array of VkBufferCopy structure instances
    );
    
    vkResult = vkEndCommandBuffer(vkCommandBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkEndCommandBuffer() Failed For Buffer-Copy: %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer);
            vkCommandBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For Buffer-Copy\n", __func__);
        }
        if (vertexData_stagingBuffer.vkDeviceMemory)
        {
            vkFreeMemory(vkDevice, vertexData_stagingBuffer.vkDeviceMemory, NULL);
            vertexData_stagingBuffer.vkDeviceMemory = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_stagingBuffer.vkDeviceMemory\n", __func__);
        }
        if (vertexData_stagingBuffer.vkBuffer)
        {
            vkDestroyBuffer(vkDevice, vertexData_stagingBuffer.vkBuffer, NULL);
            vertexData_stagingBuffer.vkBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_stagingBuffer.vkBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkEndCommandBuffer() Succeeded For Buffer-Copy\n", __func__);

    //* Submit above command buffer to queue
    VkSubmitInfo vkSubmitInfo;
    memset((void*)&vkSubmitInfo, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo.pNext = NULL;
    vkSubmitInfo.commandBufferCount = 1;
    vkSubmitInfo.pCommandBuffers = &vkCommandBuffer;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkQueueSubmit() Failed For Buffer-Copy : %d\n", __func__, vkResult);

        //* Cleanup Code
        if (vkCommandBuffer)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer);
            vkCommandBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For Buffer-Copy\n", __func__);
        }
        if (vertexData_stagingBuffer.vkDeviceMemory)
        {
            vkFreeMemory(vkDevice, vertexData_stagingBuffer.vkDeviceMemory, NULL);
            vertexData_stagingBuffer.vkDeviceMemory = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_stagingBuffer.vkDeviceMemory\n", __func__);
        }
        if (vertexData_stagingBuffer.vkBuffer)
        {
            vkDestroyBuffer(vkDevice, vertexData_stagingBuffer.vkBuffer, NULL);
            vertexData_stagingBuffer.vkBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_stagingBuffer.vkBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueSubmit() Succeeded For Buffer-Copy\n", __func__);

    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkQueueWaitIdle() Failed For Buffer-Copy : %d\n", __func__, vkResult);
        
        //* Cleanup Code
        if (vkCommandBuffer)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer);
            vkCommandBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For Buffer-Copy\n", __func__);
        }
        if (vertexData_stagingBuffer.vkDeviceMemory)
        {
            vkFreeMemory(vkDevice, vertexData_stagingBuffer.vkDeviceMemory, NULL);
            vertexData_stagingBuffer.vkDeviceMemory = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_stagingBuffer.vkDeviceMemory\n", __func__);
        }
        if (vertexData_stagingBuffer.vkBuffer)
        {
            vkDestroyBuffer(vkDevice, vertexData_stagingBuffer.vkBuffer, NULL);
            vertexData_stagingBuffer.vkBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_stagingBuffer.vkBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueWaitIdle() Succeeded For Buffer-Copy\n", __func__);
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    //! Step - 5
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (vkCommandBuffer)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer);
        vkCommandBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For Buffer-Copy\n", __func__);
    }
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    //! Step - 6
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if (vertexData_stagingBuffer.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_stagingBuffer.vkDeviceMemory, NULL);
        vertexData_stagingBuffer.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_stagingBuffer.vkDeviceMemory\n", __func__);
    }

    if (vertexData_stagingBuffer.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_stagingBuffer.vkBuffer, NULL);
        vertexData_stagingBuffer.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_stagingBuffer.vkBuffer\n", __func__);
    }
    //*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult Grass::createUniformBuffer(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = sizeof(GrassUBO);
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

    memset((void*)&uniformData, 0, sizeof(UniformData));

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &uniformData.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Uniform Data\n", __func__);

    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, uniformData.vkBuffer, &vkMemoryRequirements);

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

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &uniformData.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Uniform Data\n", __func__);

    vkResult = vkBindBufferMemory(vkDevice, uniformData.vkBuffer, uniformData.vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Uniform Data : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Uniform Data\n", __func__);

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

VkResult Grass::createPipelineLayout(void)
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
    vkPipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout_grass;
    vkPipelineLayoutCreateInfo.pushConstantRangeCount = 0;
    vkPipelineLayoutCreateInfo.pPushConstantRanges = NULL;

    //* Step - 4
    vkResult = vkCreatePipelineLayout(vkDevice, &vkPipelineLayoutCreateInfo, NULL, &vkPipelineLayout_grass);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Succeeded\n", __func__);

    return vkResult;
}

VkResult Grass::createDescriptorSetLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Initialize VkDescriptorSetLayoutBinding
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding_array[3]; // 0 -> Uniform, 1 -> Wind Image, 2 -> Grass Texture
    memset((void*)vkDescriptorSetLayoutBinding_array, 0, sizeof(VkDescriptorSetLayoutBinding) * _ARRAYSIZE(vkDescriptorSetLayoutBinding_array));

    vkDescriptorSetLayoutBinding_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorSetLayoutBinding_array[0].binding = 0;   //! Mapped with layout(binding = 0) in vertex shader
    vkDescriptorSetLayoutBinding_array[0].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[0].stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;
    vkDescriptorSetLayoutBinding_array[0].pImmutableSamplers = NULL;

    vkDescriptorSetLayoutBinding_array[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorSetLayoutBinding_array[1].binding = 1;   //! Mapped with layout(binding = 1) in geometry shader
    vkDescriptorSetLayoutBinding_array[1].descriptorCount = 1;
    vkDescriptorSetLayoutBinding_array[1].stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;
    vkDescriptorSetLayoutBinding_array[1].pImmutableSamplers = NULL;

    vkDescriptorSetLayoutBinding_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorSetLayoutBinding_array[2].binding = 2;   //! Mapped with layout(binding = 2) in fragment shader
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
    vkResult = vkCreateDescriptorSetLayout(vkDevice, &vkDescriptorSetLayoutCreateInfo, NULL, &vkDescriptorSetLayout_grass);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Succeeded\n", __func__);

    return vkResult;
}

VkResult Grass::createDescriptorSet(void)
{
    // Variable Declarations
    VkResult vkResult;

    // Code

    //* Initialize DescriptorSetAllocationInfo
    VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo;
    memset((void*)&vkDescriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
    vkDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vkDescriptorSetAllocateInfo.pNext = NULL;
    vkDescriptorSetAllocateInfo.descriptorPool = vkDescriptorPool_grass;
    vkDescriptorSetAllocateInfo.descriptorSetCount = 1;
    vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout_grass;

    vkResult = vkAllocateDescriptorSets(vkDevice, &vkDescriptorSetAllocateInfo, &vkDescriptorSet_grass);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Failed For vkDescriptorSet : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Succeeded For vkDescriptorSet\n", __func__);

    //* Describe whether we want buffer as uniform or image as uniform
    VkDescriptorBufferInfo vkDescriptorBufferInfo;
    memset((void*)&vkDescriptorBufferInfo, 0, sizeof(VkDescriptorBufferInfo));
    vkDescriptorBufferInfo.buffer = uniformData.vkBuffer;
    vkDescriptorBufferInfo.offset = 0;
    vkDescriptorBufferInfo.range = sizeof(GrassUBO);

    //! Descriptor Image Info
    VkDescriptorImageInfo vkDescriptorImageInfo_array[2];
    memset((void*)vkDescriptorImageInfo_array, 0, _ARRAYSIZE(vkDescriptorImageInfo_array) * sizeof(VkDescriptorImageInfo));

    //! Wind Image
    vkDescriptorImageInfo_array[0].imageView = vkImageView_texture_flowmap;
    vkDescriptorImageInfo_array[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkDescriptorImageInfo_array[0].sampler = vkSampler_texture_flowmap;

    //! Grass Image
    vkDescriptorImageInfo_array[1].imageView = vkImageView_texture_grass;
    vkDescriptorImageInfo_array[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkDescriptorImageInfo_array[1].sampler = vkSampler_texture_grass;

    /* Update above descriptor set directly to the shader
    There are 2 ways :-
        1) Writing directly to the shader
        2) Copying from one shader to another shader
    */
    VkWriteDescriptorSet vkWriteDescriptorSet_array[3];
    memset((void*)vkWriteDescriptorSet_array, 0, sizeof(VkWriteDescriptorSet) * _ARRAYSIZE(vkWriteDescriptorSet_array));

    //! Uniform Buffer Object
    vkWriteDescriptorSet_array[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[0].pNext = NULL;
    vkWriteDescriptorSet_array[0].dstSet = vkDescriptorSet_grass;
    vkWriteDescriptorSet_array[0].dstArrayElement = 0;
    vkWriteDescriptorSet_array[0].dstBinding = 0;
    vkWriteDescriptorSet_array[0].descriptorCount = 1;
    vkWriteDescriptorSet_array[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkWriteDescriptorSet_array[0].pBufferInfo = &vkDescriptorBufferInfo;
    vkWriteDescriptorSet_array[0].pImageInfo = NULL;
    vkWriteDescriptorSet_array[0].pTexelBufferView = NULL;

    //! Wind Image
    vkWriteDescriptorSet_array[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[1].pNext = NULL;
    vkWriteDescriptorSet_array[1].dstSet = vkDescriptorSet_grass;
    vkWriteDescriptorSet_array[1].dstArrayElement = 0;
    vkWriteDescriptorSet_array[1].descriptorCount = 1;
    vkWriteDescriptorSet_array[1].dstBinding = 1;
    vkWriteDescriptorSet_array[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkWriteDescriptorSet_array[1].pBufferInfo = NULL;
    vkWriteDescriptorSet_array[1].pImageInfo = &vkDescriptorImageInfo_array[0];
    vkWriteDescriptorSet_array[1].pTexelBufferView = NULL;

    //! Grass Image
    vkWriteDescriptorSet_array[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[2].pNext = NULL;
    vkWriteDescriptorSet_array[2].dstSet = vkDescriptorSet_grass;
    vkWriteDescriptorSet_array[2].dstArrayElement = 0;
    vkWriteDescriptorSet_array[2].descriptorCount = 1;
    vkWriteDescriptorSet_array[2].dstBinding = 2;
    vkWriteDescriptorSet_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkWriteDescriptorSet_array[2].pBufferInfo = NULL;
    vkWriteDescriptorSet_array[2].pImageInfo = &vkDescriptorImageInfo_array[1];
    vkWriteDescriptorSet_array[2].pTexelBufferView = NULL;

    vkUpdateDescriptorSets(vkDevice, _ARRAYSIZE(vkWriteDescriptorSet_array), vkWriteDescriptorSet_array, 0, NULL);
    //! --------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult Grass::createDescriptorPool(void)
{
    // Variable Declarations
    VkResult vkResult;

    // Code

    //* Vulkan expects decriptor pool size before creating actual descriptor pool
    VkDescriptorPoolSize vkDescriptorPoolSize_array[2];
    memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * _ARRAYSIZE(vkDescriptorPoolSize_array));

    vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorPoolSize_array[0].descriptorCount = 1;

    vkDescriptorPoolSize_array[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorPoolSize_array[1].descriptorCount = 1;

    //* Create the pool
    VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
    memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkDescriptorPoolCreateInfo.pNext = NULL;
    vkDescriptorPoolCreateInfo.flags = 0;
    vkDescriptorPoolCreateInfo.poolSizeCount = _ARRAYSIZE(vkDescriptorPoolSize_array);
    vkDescriptorPoolCreateInfo.pPoolSizes = vkDescriptorPoolSize_array;
    vkDescriptorPoolCreateInfo.maxSets = 2;

    vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool_grass);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Succeeded\n", __func__);

    return vkResult;
}

VkResult Grass::createPipeline(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Code

    //! Vertex Input State
    VkVertexInputBindingDescription vkVertexInputBindingDescription_array[1];
    memset((void*)vkVertexInputBindingDescription_array, 0, sizeof(VkVertexInputBindingDescription) * _ARRAYSIZE(vkVertexInputBindingDescription_array));

    //! Position
    vkVertexInputBindingDescription_array[0].binding = 0;
    vkVertexInputBindingDescription_array[0].stride = sizeof(float) * 3;
    vkVertexInputBindingDescription_array[0].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[1];
    memset((void*)vkVertexInputAttributeDescription_array, 0, sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));

    //! Position
    vkVertexInputAttributeDescription_array[0].binding = 0;
    vkVertexInputAttributeDescription_array[0].location = 0;
    vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    vkVertexInputAttributeDescription_array[0].offset = 0;

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
    vkPipelineInputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

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
    VkPipelineShaderStageCreateInfo vkPipelineShaderStageCreateInfo_array[3];
    memset((void*)vkPipelineShaderStageCreateInfo_array, 0, sizeof(VkPipelineShaderStageCreateInfo) * _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array));

    //* Vertex Shader
    vkPipelineShaderStageCreateInfo_array[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[0].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[0].flags = 0;
    vkPipelineShaderStageCreateInfo_array[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    vkPipelineShaderStageCreateInfo_array[0].module = vkShaderModule_vertex_shader;
    vkPipelineShaderStageCreateInfo_array[0].pName = "main";
    vkPipelineShaderStageCreateInfo_array[0].pSpecializationInfo = NULL;

    //* Geometry Shader
    vkPipelineShaderStageCreateInfo_array[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[1].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[1].flags = 0;
    vkPipelineShaderStageCreateInfo_array[1].stage = VK_SHADER_STAGE_GEOMETRY_BIT;
    vkPipelineShaderStageCreateInfo_array[1].module = vkShaderModule_geometry_shader;
    vkPipelineShaderStageCreateInfo_array[1].pName = "main";
    vkPipelineShaderStageCreateInfo_array[1].pSpecializationInfo = NULL;

    //* Fragment Shader
    vkPipelineShaderStageCreateInfo_array[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[2].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[2].flags = 0;
    vkPipelineShaderStageCreateInfo_array[2].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkPipelineShaderStageCreateInfo_array[2].module = vkShaderModule_fragment_shader;
    vkPipelineShaderStageCreateInfo_array[2].pName = "main";
    vkPipelineShaderStageCreateInfo_array[2].pSpecializationInfo = NULL;

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
    vkGraphicsPipelineCreateInfo.layout = vkPipelineLayout_grass;
    vkGraphicsPipelineCreateInfo.renderPass = vkRenderPass;
    vkGraphicsPipelineCreateInfo.subpass = 0;
    vkGraphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    vkGraphicsPipelineCreateInfo.basePipelineIndex = 0;

    vkResult = vkCreateGraphicsPipelines(vkDevice, vkPipelineCache, 1, &vkGraphicsPipelineCreateInfo, NULL, &vkPipeline_grass);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateGraphicsPipelines() Failed : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateGraphicsPipelines() Succeeded\n", __func__);

    //* Destroy Pipeline Cache
    if (vkPipelineCache)
    {
        vkDestroyPipelineCache(vkDevice, vkPipelineCache, NULL);
        vkPipelineCache = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyPipelineCache() Succeeded\n", __func__);
    }

    return vkResult;
}

void Grass::buildCommandBuffers(VkCommandBuffer& commandBuffer)
{
    //! Bind with Pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline_grass);

    //! Bind the Descriptor Set to the Pipeline
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        vkPipelineLayout_grass,
        0,
        1,
        &vkDescriptorSet_grass,
        0,
        NULL
    );

    //! Bind with Vertex Position Buffer
    VkDeviceSize vkDeviceSize_offset_position[1];
    memset((void*)vkDeviceSize_offset_position, 0, sizeof(VkDeviceSize) * _ARRAYSIZE(vkDeviceSize_offset_position));
    vkCmdBindVertexBuffers(
        commandBuffer,
        0,
        1,
        &vertexData_grass_position.vkBuffer,
        vkDeviceSize_offset_position
    );

    //! Vulkan Drawing Function (Instanced Drawing)
    vkCmdDraw(commandBuffer, 1, grassPosition.size(), 0, 0);
}

VkResult Grass::updateUniformBuffer(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    GrassUBO grassUBO;
    memset((void*)&grassUBO, 0, sizeof(GrassUBO));

    float elapsedTime = sdkGetTimerValue(&timer) / 1000.0f;

    //! Update Matrices
    glm::mat4 perspectiveProjectionMatrix = glm::mat4(1.0f);
    perspectiveProjectionMatrix = glm::perspective(
        glm::radians(45.0f),
        (float)winWidth / (float)winHeight,
        0.1f,
        100.0f
    );
    //! 2D Matrix with Column Major (Like OpenGL)
    perspectiveProjectionMatrix[1][1] = perspectiveProjectionMatrix[1][1] * (-1.0f);

    glm::mat4 translationMatrix = glm::mat4(1.0f);
    translationMatrix = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -1.0f, -6.0f));

    grassUBO.modelMatrix = translationMatrix;
    grassUBO.time = elapsedTime;
    grassUBO.windStrength = windStrength;
    grassUBO.projectionMatrix = perspectiveProjectionMatrix;

    //! Map Uniform Buffer
    void* data = NULL;
    vkResult = vkMapMemory(vkDevice, uniformData.vkDeviceMemory, 0, sizeof(GrassUBO), 0, &data);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Uniform Buffer : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //! Copy the data to the mapped buffer (present on device memory)
    memcpy(data, &grassUBO, sizeof(GrassUBO));

    //! Unmap memory
    vkUnmapMemory(vkDevice, uniformData.vkDeviceMemory);

    return vkResult;
}

void Grass::update()
{
    VkResult vkResult = updateUniformBuffer();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS::%s() => updateUniformBuffer() Failed : %d !!!\n", __func__, vkResult);
    }
}

VkResult Grass::resize(int width, int height)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //? DESTROY
    //?--------------------------------------------------------------------------------------------------
    //* Wait for device to complete in-hand tasks
    if (vkDevice)
        vkDeviceWaitIdle(vkDevice);

    //* Destroy PipelineLayout
    if (vkPipelineLayout_grass)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_grass, NULL);
        vkPipelineLayout_grass = VK_NULL_HANDLE;
    }

    //* Destroy Pipeline
    if (vkPipeline_grass)
    {
        vkDestroyPipeline(vkDevice, vkPipeline_grass, NULL);
        vkPipeline_grass = VK_NULL_HANDLE;
    }
    //?--------------------------------------------------------------------------------------------------

    //? RECREATE FOR RESIZE
    //?--------------------------------------------------------------------------------------------------
    //* Create Pipeline Layout
    vkResult = createPipelineLayout();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS::%s() => createPipelineLayout() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }

    //* Create Pipeline
    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "GRASS::%s() => createPipeline() Failed : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    //?--------------------------------------------------------------------------------------------------

    return vkResult;
}

Grass::~Grass()
{
    if (timer)
    {
        sdkStopTimer(&timer);
        sdkDeleteTimer(&timer);
        timer = nullptr;
    }

    if (vkDevice)
        vkDeviceWaitIdle(vkDevice);

    if (vkPipelineLayout_grass)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_grass, NULL);
        vkPipelineLayout_grass = VK_NULL_HANDLE;
    }

    if (vkPipeline_grass)
    {
        vkDestroyPipeline(vkDevice, vkPipeline_grass, NULL);
        vkPipeline_grass = VK_NULL_HANDLE;
    }

    if (vkDescriptorPool_grass)
    {
        vkDestroyDescriptorPool(vkDevice, vkDescriptorPool_grass, NULL);
        vkDescriptorPool_grass = VK_NULL_HANDLE;
        vkDescriptorSet_grass = VK_NULL_HANDLE;
    }

    if (vkPipelineLayout_grass)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_grass, NULL);
        vkPipelineLayout_grass = VK_NULL_HANDLE;
    }

    if (vkDescriptorSetLayout_grass)
    {
        vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout_grass, NULL);
        vkDescriptorSetLayout_grass = VK_NULL_HANDLE;
    }

    //* Destroy Uniform Buffer
    if (uniformData.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, uniformData.vkDeviceMemory, NULL);
        uniformData.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For uniformData.vkDeviceMemory\n", __func__);
    }

    if (uniformData.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, uniformData.vkBuffer, NULL);
        uniformData.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succedded For uniformData.vkBuffer\n", __func__);
    }

    //* Destroy Vertex Buffer
    if (vertexData_grass_position.vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, vertexData_grass_position.vkDeviceMemory, NULL);
        vertexData_grass_position.vkDeviceMemory = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vertexData_grass_position.vkDeviceMemory\n", __func__);
    }

    if (vertexData_grass_position.vkBuffer)
    {
        vkDestroyBuffer(vkDevice, vertexData_grass_position.vkBuffer, NULL);
        vertexData_grass_position.vkBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vertexData_grass_position.vkBuffer\n", __func__);
    }
}

