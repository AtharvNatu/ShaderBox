#include "Ocean.hpp"

//* Get K Vector From Mesh Grid (n, m)
#define K_VEC(n, m) glm::vec2(2 * M_PI * (n - N / 2) / x_length, 2 * M_PI * (m - M / 2) / z_length)

Ocean::Ocean()
{
    omega_hat = glm::normalize(omega_vec);

    meshSize = sizeof(glm::vec3) * N * M;

    generator.seed(time(nullptr));
    kNum = N * M;

    displacement_map = new glm::vec3[kNum];
    normal_map = new glm::vec3[kNum];

    h_twiddle_0 = new std::complex<float>[kNum];
    h_twiddle_0_conjunction = new std::complex<float>[kNum];
    h_twiddle = new std::complex<float>[kNum];

    //! Initialize h_twiddle_0 and h_twiddle_0_conjunction in Eqn. 26
    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;
            glm::vec2 k = K_VEC(n, m);
            h_twiddle_0[index] = func_h_twiddle_0(k);
            // h_twiddle_0_conjunction[index] = std::conj(func_h_twiddle_0(k));

            int kn_neg = (N - n) % N;
            int km_neg = (M - m) % M;
            int index_neg = km_neg * N + kn_neg;

            h_twiddle_0_conjunction[index] = std::conj(h_twiddle_0[index_neg]);
        }
    }
}

VkResult Ocean::initialize()
{
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

    return vkResult;
}

//* Vulkan Related
VkResult Ocean::createBuffers()
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //! Vertex Displacement Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_displacement, 0, sizeof(BufferData));

    //* Step - 5
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = meshSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

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

    //* Step - 11
    vkResult = vkMapMemory(vkDevice, vertexData_displacement.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &displacementPtr);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Displacement Buffer\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------------
    
    //! Vertex Normals Buffer
    //! ---------------------------------------------------------------------------------------------------------------------------------
    //* Step - 4
    memset((void*)&vertexData_normals, 0, sizeof(BufferData));

    //* Step - 5
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;   //! Valid Flags are used in sparse(scattered) buffers
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = meshSize;
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &vertexData_normals.vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Vertex Normals Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Vertex Normals Buffer\n", __func__);

    //* Step - 7
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vertexData_normals.vkBuffer, &vkMemoryRequirements);

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

    //* Step - 11
    vkResult = vkMapMemory(vkDevice, vertexData_normals.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &normalsPtr);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Normals Buffer : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Normals Buffer\n", __func__);
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

VkResult Ocean::createUniformBuffer(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

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
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

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
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

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
    else
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Succeeded For Ocean\n", __func__);

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
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

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

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            int index = j * N + i;

            if (displacement_map[index][1] > heightMax)
                heightMax = displacement_map[index][1];
            else if (displacement_map[index][1] < heightMin)
                heightMin = displacement_map[index][1];
        }
    }

    //* Copy Displacement Data
    memcpy(displacementPtr, displacement_map, meshSize);
    memcpy(normalsPtr, normal_map, meshSize);

    updateUniformBuffer();
}

//* Tessendorf Related

//! Eqn. 14
inline float Ocean::omega(float k) const
{
    return sqrt(G * k);
}

//! Eqn. 23 Phillips Spectrum
inline float Ocean::phillips_spectrum(glm::vec2 k) const
{
    // Code
    float k_length = glm::length(k);
    if (k_length < 1e-6f)
        return 0.0f;

    // Largest possible waves from continuous wind of speed V
    float wave_length = (V * V) / G;
    glm::vec2 k_hat = glm::normalize(k);

    float dot_k_hat_omega_hat = glm::dot(k_hat, omega_hat);
    float dot_term = dot_k_hat_omega_hat * dot_k_hat_omega_hat;

    float exp_term = expf(-1.0f / (k_length * k_length * wave_length * wave_length));
    float result = A * exp_term * dot_term / powf(k_length, 4.0f);

    // Small-wave damping (Eq. 24) — uses constant L
    result *= expf(-k_length * k_length * L * L);

    return result;
}

//! Eqn. 25
inline std::complex<float> Ocean::func_h_twiddle_0(glm::vec2 k)
{
    // Code
    float xi_r = normal_distribution(generator);
    float xi_i = normal_distribution(generator);

    return sqrt(0.5f) * std::complex<float>(xi_r, xi_i) * sqrt(phillips_spectrum(k));
}

//! Eqn. 26
inline std::complex<float> Ocean::func_h_twiddle(int kn, int km, float t) const
{
    // Code
    int index = km * N + kn;

    float k = glm::length(K_VEC(kn, km));

    std::complex<float> term1 = h_twiddle_0[index] * exp(std::complex<float>(0.0f, omega(k) * t));
    std::complex<float> term2 = h_twiddle_0_conjunction[index] * exp(std::complex<float>(0.0f, -omega(k) * t));

    return term1 + term2;
}

//! Eqn. 19
void Ocean::generate_fft_data(float time)
{
    // Variable Declarations
    fftwf_complex *in_height = nullptr;
    fftwf_complex *in_slope_x = nullptr, *in_slope_z = nullptr;
    fftwf_complex *in_displacement_x = nullptr, *in_displacement_z = nullptr;

    fftwf_complex *out_height = nullptr;
    fftwf_complex *out_slope_x = nullptr, *out_slope_z = nullptr;
    fftwf_complex *out_displacement_x = nullptr, *out_displacement_z = nullptr;

    fftwf_plan p_height, p_slope_x, p_slope_z, p_displacement_x, p_displacement_z;

    // Code

    //! Eqn. 20 ikh_twiddle
    std::complex<float>* slope_x_term = new std::complex<float>[kNum];
    std::complex<float>* slope_z_term = new std::complex<float>[kNum];

    //! Eqn. 29
    std::complex<float>* displacement_x_term = new std::complex<float>[kNum];
    std::complex<float>* displacement_z_term = new std::complex<float>[kNum];

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;

            h_twiddle[index] = func_h_twiddle(n, m, time);

            glm::vec2 k_vec = K_VEC(n, m);
            float k_length = glm::length(k_vec);
            glm::vec2 k_vec_normalized = k_length == 0 ? k_vec : glm::normalize(k_vec);

            slope_x_term[index] = std::complex<float>(0, k_vec[0]) * h_twiddle[index];
            slope_z_term[index] = std::complex<float>(0, k_vec[1]) * h_twiddle[index];
            
            displacement_x_term[index] = std::complex<float>(0, -k_vec_normalized[0]) * h_twiddle[index];
            displacement_z_term[index] = std::complex<float>(0, -k_vec_normalized[1]) * h_twiddle[index];
        }
    }

    //* Prepare FFT Input and Output
    in_height = (fftwf_complex*)h_twiddle;
    in_slope_x = (fftwf_complex*)slope_x_term;
    in_slope_z = (fftwf_complex*)slope_z_term;
    in_displacement_x = (fftwf_complex*)displacement_x_term;
    in_displacement_z = (fftwf_complex*)displacement_z_term;

    out_height = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_slope_x = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_slope_z = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_displacement_x = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    out_displacement_z = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * kNum);
    
    p_height = fftwf_plan_dft_2d(
        N, 
        M, 
        in_height, 
        out_height, 
        FFTW_BACKWARD, 
        FFTW_ESTIMATE
    );

    p_slope_x = fftwf_plan_dft_2d(
        N,
        M,
        in_slope_x,
        out_slope_x,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    p_slope_z = fftwf_plan_dft_2d(
        N,
        M,
        in_slope_z,
        out_slope_z,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    p_displacement_x = fftwf_plan_dft_2d(
        N,
        M,
        in_displacement_x,
        out_displacement_x,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    p_displacement_z = fftwf_plan_dft_2d(
        N,
        M,
        in_displacement_z,
        out_displacement_z,
        FFTW_BACKWARD,
        FFTW_ESTIMATE
    );

    fftwf_execute(p_height);
    fftwf_execute(p_slope_x);
    fftwf_execute(p_slope_z);
    fftwf_execute(p_displacement_x);
    fftwf_execute(p_displacement_z);

    for (int n = 0; n < N; n++)
    {
        for (int m = 0; m < M; m++)
        {
            int index = m * N + n;
            float sign = 1;

            // Flip the sign
            if ((m + n) % 2)
                sign = -1;

            normal_map[index] = glm::normalize(glm::vec3(
                sign * out_slope_x[index][0],
                -1,
                sign * out_slope_z[index][0]
            ));

            displacement_map[index] = glm::vec3(
                (n - N / 2) * x_length / N - sign * lambda * out_displacement_x[index][0],
                sign * out_height[index][0],
                (m - M / 2) * z_length / M - sign * lambda * out_displacement_z[index][0]
            );
        }
    }

    fftwf_destroy_plan(p_displacement_z);
    fftwf_destroy_plan(p_displacement_x);
    fftwf_destroy_plan(p_slope_z);
    fftwf_destroy_plan(p_slope_x);
    fftwf_destroy_plan(p_height);

    fftwf_free(out_displacement_z);
    out_displacement_z = nullptr;

    fftwf_free(out_displacement_x);
    out_displacement_x = nullptr;

    fftwf_free(out_slope_z);
    out_slope_z = nullptr;

    fftwf_free(out_slope_x);
    out_slope_x = nullptr;

    fftwf_free(out_height);
    out_height = nullptr;

    delete[] displacement_z_term;
    displacement_z_term = nullptr;

    delete[] displacement_x_term;
    displacement_x_term = nullptr;

    delete[] slope_z_term;
    slope_z_term = nullptr;

    delete[] slope_x_term;
    slope_x_term = nullptr;

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
        if (normalsPtr)
        {
            vkUnmapMemory(vkDevice, vertexData_normals.vkDeviceMemory);
            normalsPtr = nullptr;
        }
        
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
        if (displacementPtr)
        {
            vkUnmapMemory(vkDevice, vertexData_displacement.vkDeviceMemory);
            displacementPtr = nullptr;
        }
        
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

    if (h_twiddle)
    {
        delete[] h_twiddle;
        h_twiddle = nullptr;
    }

    if (h_twiddle_0_conjunction)
    {
        delete[] h_twiddle_0_conjunction;
        h_twiddle_0_conjunction = nullptr;
    }

    if (h_twiddle_0)
    {
        delete[] h_twiddle_0;
        h_twiddle_0 = nullptr;
    }

    if (normal_map)
    {
        delete[] normal_map;
        normal_map = nullptr;
    }

    if (displacement_map)
    {
        delete[] displacement_map;
        displacement_map = nullptr;
    }
}
