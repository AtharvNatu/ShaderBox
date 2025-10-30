#include "Ocean.hpp"

//! Header File For Texture
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

    vkResult = createTexture(
        "Assets/Images/Sun.png", 
        &vkImage_texture_sun, 
        &vkDeviceMemory_texture_sun, 
        &vkImageView_texture_sun, 
        &vkSampler_texture_sun
    );
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => createTexture() Failed For Sun.png : %d !!!\n", __func__, vkResult);
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => createTexture() Succeeded For Sun.png\n", __func__);

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

    //! Compute
    vkResult = createFFTBuffer();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createFFTBuffer() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createFFTBuffer() Succeeded For Ocean\n", __func__);

    vkResult = createComputeDescriptorSetLayout();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputeDescriptorSetLayout() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputeDescriptorSetLayout() Succeeded For Ocean\n", __func__);
    
    vkResult = createComputePipeline();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputePipeline() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputePipeline() Succeeded For Ocean\n", __func__);
    
    vkResult = createComputeDescriptorSet();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createComputeDescriptorSet() Failed For Ocean : %d !!!\n", __func__, vkResult);
    else
        fprintf(gpFile, "%s() => createComputeDescriptorSet() Succeeded For Ocean\n", __func__);

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
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

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

    // //* Step - 11
    // vkResult = vkMapMemory(vkDevice, vertexData_displacement.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &displacementPtr);
    // if (vkResult != VK_SUCCESS)
    //     fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Displacement Buffer : %d !!!\n", __func__, vkResult);
    // else
    //     fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Displacement Buffer\n", __func__);
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
    vkBufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

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

    //* Step - 11
    // vkResult = vkMapMemory(vkDevice, vertexData_normals.vkDeviceMemory, 0, vkMemoryAllocateInfo.allocationSize, 0, &normalsPtr);
    // if (vkResult != VK_SUCCESS)
    //     fprintf(gpFile, "%s() => vkMapMemory() Failed For Vertex Normals Buffer : %d !!!\n", __func__, vkResult);
    // else
    //     fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Vertex Normals Buffer\n", __func__);
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

VkResult Ocean::createTexture(const char* textureFileName, VkImage* vkImage_texture, VkDeviceMemory* vkDeviceMemory_texture, VkImageView* vkImageView_texture, VkSampler* vkSampler_texture)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;
    uint8_t* imageData = NULL;
    int imageWidth = 0, imageHeight = 0, numChannels = 0;

    VkBuffer vkBuffer_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize imageSize;

    // Code

    //! Step - 1
    stbi_set_flip_vertically_on_load(TRUE);
    imageData = stbi_load(textureFileName, &imageWidth, &imageHeight, &numChannels, STBI_rgb_alpha);
    if (imageData == NULL || imageWidth <= 0 || imageHeight <= 0 || numChannels <= 0)
    {
        fprintf(gpFile, "%s() => stbi_load() Failed For %s !!!\n", __func__, textureFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

    imageSize = imageWidth * imageHeight * 4;

    fprintf(gpFile, "\n%s Properties\n", textureFileName);
    fprintf(gpFile, "-------------------------------------------\n");
    fprintf(gpFile, "Image Width = %d\n", imageWidth);
    fprintf(gpFile, "Image Height = %d\n", imageHeight);
    fprintf(gpFile, "Image Size = %lld\n", imageSize);
    fprintf(gpFile, "-------------------------------------------\n\n");

    //! Step - 2
    VkBufferCreateInfo vkBufferCreateInfo_stagingBuffer;
    memset((void*)&vkBufferCreateInfo_stagingBuffer, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo_stagingBuffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo_stagingBuffer.pNext = NULL;
    vkBufferCreateInfo_stagingBuffer.flags = 0;
    vkBufferCreateInfo_stagingBuffer.size = imageSize;
    vkBufferCreateInfo_stagingBuffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo_stagingBuffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;  //* This denotes source data to be transferred to VkImage

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo_stagingBuffer, NULL, &vkBuffer_stagingBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
            fprintf(gpFile, "%s() => stbi_image_free() Called For Texture : %s\n", __func__, textureFileName);
        }

        return vkResult;
    }

    else
        fprintf(gpFile, "%s() => vkCreateBuffer() Succeeded For Staging Buffer : %s\n", __func__, textureFileName);

    VkMemoryRequirements vkMemoryRequirements_stagingBuffer;
    memset((void*)&vkMemoryRequirements_stagingBuffer, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, vkBuffer_stagingBuffer, &vkMemoryRequirements_stagingBuffer);

    VkMemoryAllocateInfo vkMemoryAllocateInfo_stagingBuffer;
    memset((void*)&vkMemoryAllocateInfo_stagingBuffer, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo_stagingBuffer.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo_stagingBuffer.pNext = NULL;
    vkMemoryAllocateInfo_stagingBuffer.allocationSize = vkMemoryRequirements_stagingBuffer.size;
    vkMemoryAllocateInfo_stagingBuffer.memoryTypeIndex = 0;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements_stagingBuffer.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                vkMemoryAllocateInfo_stagingBuffer.memoryTypeIndex = i;
                break;
            }
        }
        vkMemoryRequirements_stagingBuffer.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo_stagingBuffer, NULL, &vkDeviceMemory_stagingBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
            fprintf(gpFile, "%s() => stbi_image_free() Called For Texture : %s\n", __func__, textureFileName);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Staging Buffer : %s\n", __func__, textureFileName);

    vkResult = vkBindBufferMemory(vkDevice, vkBuffer_stagingBuffer, vkDeviceMemory_stagingBuffer, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBindBufferMemory() Succeeded For Staging Buffer : %s\n", __func__, textureFileName);

    void* data = NULL;
    vkResult = vkMapMemory(
        vkDevice,
        vkDeviceMemory_stagingBuffer,
        0,
        imageSize,
        0,
        &data
    );
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Staging Buffer : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
            fprintf(gpFile, "%s() => stbi_image_free() Called For Texture : %s\n", __func__, textureFileName);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkMapMemory() Succeeded For Staging Buffer : %s\n", __func__, textureFileName);

    memcpy(data, imageData, imageSize);

    vkUnmapMemory(vkDevice, vkDeviceMemory_stagingBuffer);

    //* Free the image data given by stb, as it is copied in image staging buffer
    stbi_image_free(imageData);
    imageData = NULL;
    fprintf(gpFile, "%s() => stbi_image_free() Called For Texture : %s\n", __func__, textureFileName);

    //! Step - 3
    VkImageCreateInfo vkImageCreateInfo;
    memset((void*)&vkImageCreateInfo, 0, sizeof(VkImageCreateInfo));
    vkImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    vkImageCreateInfo.flags = 0;
    vkImageCreateInfo.pNext = NULL;
    vkImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    vkImageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    vkImageCreateInfo.extent.width = imageWidth;
    vkImageCreateInfo.extent.height = imageHeight;
    vkImageCreateInfo.extent.depth = 1;
    vkImageCreateInfo.mipLevels = 1;
    vkImageCreateInfo.arrayLayers = 1;
    vkImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    vkImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    vkImageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    vkImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkResult = vkCreateImage(vkDevice, &vkImageCreateInfo, NULL, vkImage_texture);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateImage() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateImage() Succeeded For Texture : %s\n", __func__, textureFileName);

    VkMemoryRequirements vkMemoryRequirements_image;
    memset((void*)&vkMemoryRequirements_image, 0, sizeof(VkMemoryRequirements));
    vkGetImageMemoryRequirements(vkDevice, *vkImage_texture, &vkMemoryRequirements_image);

    VkMemoryAllocateInfo vkMemoryAllocateInfo_image;
    memset((void*)&vkMemoryAllocateInfo_image, 0, sizeof(VkMemoryAllocateInfo));
    vkMemoryAllocateInfo_image.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    vkMemoryAllocateInfo_image.pNext = NULL;
    vkMemoryAllocateInfo_image.memoryTypeIndex = 0;
    vkMemoryAllocateInfo_image.allocationSize = vkMemoryRequirements_image.size;

    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((vkMemoryRequirements_image.memoryTypeBits & 1) == 1)
        {
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            {
                vkMemoryAllocateInfo_image.memoryTypeIndex = i;
                break;
            }
        }
        vkMemoryRequirements_image.memoryTypeBits >>= 1;
    }

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo_image, NULL, vkDeviceMemory_texture);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateMemory() Succeeded For Texture : %s\n", __func__, textureFileName);

    vkResult = vkBindImageMemory(vkDevice, *vkImage_texture, *vkDeviceMemory_texture, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindImageMemory() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }
        if (imageData)
        {
            stbi_image_free(imageData);
            imageData = NULL;
            fprintf(gpFile, "%s() => stbi_image_free() Called For Texture : %s\n", __func__, textureFileName);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBindImageMemory() Succeeded For Texture : %s\n", __func__, textureFileName);

    //! Step - 4
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------

    //* Step - 4.1
    VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo_transition_image_layout;
    memset((void*)&vkCommandBufferAllocateInfo_transition_image_layout, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    vkCommandBufferAllocateInfo_transition_image_layout.pNext = NULL;
    vkCommandBufferAllocateInfo_transition_image_layout.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo_transition_image_layout.commandBufferCount = 1;
    vkCommandBufferAllocateInfo_transition_image_layout.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VkCommandBuffer vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
    vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo_transition_image_layout, &vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_transition_image_layout : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 4.2
    VkCommandBufferBeginInfo vkCommandBufferBeginInfo_image_transition_layout;
    memset((void*)&vkCommandBufferBeginInfo_image_transition_layout, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_image_transition_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_image_transition_layout.pNext = NULL;
    vkCommandBufferBeginInfo_image_transition_layout.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_transition_image_layout, &vkCommandBufferBeginInfo_image_transition_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 4.3
    VkPipelineStageFlags vkPipelineStageFlags_source = 0;
    VkPipelineStageFlags vkPipelineStageFlags_destination = 0;

    VkImageMemoryBarrier vkImageMemoryBarrier;
    memset((void*)&vkImageMemoryBarrier, 0, sizeof(VkImageMemoryBarrier));
    vkImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    vkImageMemoryBarrier.pNext = NULL;
    vkImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    vkImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.image = *vkImage_texture;
    vkImageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
    vkImageMemoryBarrier.subresourceRange.baseMipLevel = 0;
    vkImageMemoryBarrier.subresourceRange.layerCount = 1;
    vkImageMemoryBarrier.subresourceRange.levelCount = 1;

    if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = 0;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        fprintf(gpFile, "ERROR : %s() => Unsupported Texture Layout Transition !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }

    vkCmdPipelineBarrier(
        vkCommandBuffer_transition_image_layout,
        vkPipelineStageFlags_source,
        vkPipelineStageFlags_destination,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &vkImageMemoryBarrier
    );

    //* Step - 4.4
    vkResult = vkEndCommandBuffer(vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkEndCommandBuffer() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 4.5
    VkSubmitInfo vkSubmitInfo_transition_image_layout;
    memset((void*)&vkSubmitInfo_transition_image_layout, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_transition_image_layout.pNext = NULL;
    vkSubmitInfo_transition_image_layout.commandBufferCount = 1;
    vkSubmitInfo_transition_image_layout.pCommandBuffers = &vkCommandBuffer_transition_image_layout;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_transition_image_layout, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueSubmit() Succeeded For vkSubmitInfo_transition_image_layout\n", __func__);

    //* Step - 4.6
    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueWaitIdle() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 4.7
    if (vkCommandBuffer_transition_image_layout)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
        vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
    }
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------

    //! Step - 5
    VkCommandBufferAllocateInfo vkCommandBufferAllocateInfo_buffer_to_image_copy;
    memset((void*)&vkCommandBufferAllocateInfo_buffer_to_image_copy, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo_buffer_to_image_copy.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.pNext = NULL;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.commandBufferCount = 1;
    vkCommandBufferAllocateInfo_buffer_to_image_copy.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    VkCommandBuffer vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
    vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo_buffer_to_image_copy, &vkCommandBuffer_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_buffer_to_image_copy : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);

    VkCommandBufferBeginInfo vkCommandBufferBeginInfo_buffer_to_image_copy;
    memset((void*)&vkCommandBufferBeginInfo_buffer_to_image_copy, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_buffer_to_image_copy.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_buffer_to_image_copy.pNext = NULL;
    vkCommandBufferBeginInfo_buffer_to_image_copy.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_buffer_to_image_copy, &vkCommandBufferBeginInfo_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_buffer_to_image_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
            fprintf(gpFile, "ERROR : %s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);

    VkBufferImageCopy vkBufferImageCopy;
    memset((void*)&vkBufferImageCopy, 0, sizeof(VkBufferImageCopy));
    vkBufferImageCopy.bufferOffset = 0;
    vkBufferImageCopy.bufferRowLength = 0;
    vkBufferImageCopy.bufferImageHeight = 0;
    vkBufferImageCopy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkBufferImageCopy.imageSubresource.mipLevel = 0;
    vkBufferImageCopy.imageSubresource.baseArrayLayer = 0;
    vkBufferImageCopy.imageSubresource.layerCount = 1;
    vkBufferImageCopy.imageOffset.x = 0;
    vkBufferImageCopy.imageOffset.y = 0;
    vkBufferImageCopy.imageOffset.z = 0;
    vkBufferImageCopy.imageExtent.width = imageWidth;
    vkBufferImageCopy.imageExtent.height = imageHeight;
    vkBufferImageCopy.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(
        vkCommandBuffer_buffer_to_image_copy,
        vkBuffer_stagingBuffer,
        *vkImage_texture,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &vkBufferImageCopy
    );

    vkResult = vkEndCommandBuffer(vkCommandBuffer_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_buffer_to_image_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);
        }
        if (*vkImage_texture)
        {
            vkDestroyImage(vkDevice, *vkImage_texture, NULL);
            *vkImage_texture = NULL;
            fprintf(gpFile, "%s() => vkDestroyImage() Succeeded For vkImage_texture\n", __func__);
        }
        if (*vkDeviceMemory_texture)
        {
            vkFreeMemory(vkDevice, *vkDeviceMemory_texture, NULL);
            *vkDeviceMemory_texture = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_texture\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkEndCommandBuffer() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);

    VkSubmitInfo vkSubmitInfo_buffer_to_copy;
    memset((void*)&vkSubmitInfo_buffer_to_copy, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_buffer_to_copy.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_buffer_to_copy.pNext = NULL;
    vkSubmitInfo_buffer_to_copy.commandBufferCount = 1;
    vkSubmitInfo_buffer_to_copy.pCommandBuffers = &vkCommandBuffer_buffer_to_image_copy;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_buffer_to_copy, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_buffer_to_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
            fprintf(gpFile, "ERROR : %s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueSubmit() Succeeded For vkSubmitInfo_buffer_to_copy\n", __func__);

    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkCommandBuffer_buffer_to_image_copy : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
            fprintf(gpFile, "ERROR : %s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueWaitIdle() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);

    if (vkCommandBuffer_buffer_to_image_copy)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
        vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_buffer_to_image_copy\n", __func__);
    }

    //! Step - 6
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------

    //* Step - 6.1
    memset((void*)&vkCommandBufferAllocateInfo_transition_image_layout, 0, sizeof(VkCommandBufferAllocateInfo));
    vkCommandBufferAllocateInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    vkCommandBufferAllocateInfo_transition_image_layout.pNext = NULL;
    vkCommandBufferAllocateInfo_transition_image_layout.commandPool = vkCommandPool;
    vkCommandBufferAllocateInfo_transition_image_layout.commandBufferCount = 1;
    vkCommandBufferAllocateInfo_transition_image_layout.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
    vkResult = vkAllocateCommandBuffers(vkDevice, &vkCommandBufferAllocateInfo_transition_image_layout, &vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_transition_image_layout : %d !!!\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 6.2
    memset((void*)&vkCommandBufferBeginInfo_image_transition_layout, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_image_transition_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_image_transition_layout.pNext = NULL;
    vkCommandBufferBeginInfo_image_transition_layout.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_transition_image_layout, &vkCommandBufferBeginInfo_image_transition_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 6.3
    vkPipelineStageFlags_source = 0;
    vkPipelineStageFlags_destination = 0;

    memset((void*)&vkImageMemoryBarrier, 0, sizeof(VkImageMemoryBarrier));
    vkImageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    vkImageMemoryBarrier.pNext = NULL;
    vkImageMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    vkImageMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkImageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    vkImageMemoryBarrier.image = *vkImage_texture;
    vkImageMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageMemoryBarrier.subresourceRange.baseArrayLayer = 0;
    vkImageMemoryBarrier.subresourceRange.baseMipLevel = 0;
    vkImageMemoryBarrier.subresourceRange.layerCount = 1;
    vkImageMemoryBarrier.subresourceRange.levelCount = 1;

    if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = 0;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (vkImageMemoryBarrier.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && vkImageMemoryBarrier.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
    {
        vkImageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkImageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkPipelineStageFlags_source = VK_PIPELINE_STAGE_TRANSFER_BIT;
        vkPipelineStageFlags_destination = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else
    {
        fprintf(gpFile, "ERROR : %s() => Unsupported Texture Layout Transition !!!\n", __func__);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }

    vkCmdPipelineBarrier(
        vkCommandBuffer_transition_image_layout,
        vkPipelineStageFlags_source,
        vkPipelineStageFlags_destination,
        0,
        0,
        NULL,
        0,
        NULL,
        1,
        &vkImageMemoryBarrier
    );

    //* Step - 6.4
    vkResult = vkEndCommandBuffer(vkCommandBuffer_transition_image_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkEndCommandBuffer() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 6.5
    memset((void*)&vkSubmitInfo_transition_image_layout, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_transition_image_layout.pNext = NULL;
    vkSubmitInfo_transition_image_layout.commandBufferCount = 1;
    vkSubmitInfo_transition_image_layout.pCommandBuffers = &vkCommandBuffer_transition_image_layout;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_transition_image_layout, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueSubmit() Succeeded For vkSubmitInfo_transition_image_layout\n", __func__);

    //* Step - 6.6
    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkSubmitInfo_transition_image_layout : %d\n", __func__, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
            fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueWaitIdle() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);

    //* Step - 6.7
    if (vkCommandBuffer_transition_image_layout)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
        vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeCommandBuffers() Succeeded For vkCommandBuffer_transition_image_layout\n", __func__);
    }
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------

    //! Step - 7
    if (vkBuffer_stagingBuffer)
    {
        vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
        vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroyBuffer() Succeeded For vkBuffer_stagingBuffer\n", __func__);
    }

    if (vkDeviceMemory_stagingBuffer)
    {
        vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
        vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_stagingBuffer\n", __func__);
    }

    //! Step - 8
    VkImageViewCreateInfo vkImageViewCreateInfo;
    memset((void*)&vkImageViewCreateInfo, 0, sizeof(VkImageViewCreateInfo));
    vkImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vkImageViewCreateInfo.pNext = NULL;
    vkImageViewCreateInfo.flags = 0;
    vkImageViewCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    vkImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_R;
    vkImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_G;
    vkImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_B;
    vkImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    vkImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vkImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
    vkImageViewCreateInfo.subresourceRange.levelCount = 1;
    vkImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
    vkImageViewCreateInfo.subresourceRange.layerCount = 1;
    vkImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vkImageViewCreateInfo.image = *vkImage_texture;

    vkResult = vkCreateImageView(vkDevice, &vkImageViewCreateInfo, NULL, vkImageView_texture);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateImageView() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateImageView() Succeeded For Texture : %s\n", __func__, textureFileName);

    //! Step - 9
    VkSamplerCreateInfo vkSamplerCreateInfo;
    memset((void*)&vkSamplerCreateInfo, 0, sizeof(VkSamplerCreateInfo));
    vkSamplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    vkSamplerCreateInfo.pNext = NULL;
    vkSamplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    vkSamplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    vkSamplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkSamplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkSamplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkSamplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    vkSamplerCreateInfo.anisotropyEnable = VK_FALSE;
    vkSamplerCreateInfo.maxAnisotropy = 16;
    vkSamplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    vkSamplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    vkSamplerCreateInfo.compareEnable = VK_FALSE;
    vkSamplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    vkResult = vkCreateSampler(vkDevice, &vkSamplerCreateInfo, NULL, vkSampler_texture);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateSampler() Failed For Texture : %s, Error Code : %d !!!\n", __func__, textureFileName, vkResult);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkCreateSampler() Succeeded For Texture : %s\n", __func__, textureFileName);

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
    // scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(5.0f, 5.0f, 5.0f));

    mvpData.modelMatrix = translationMatrix * scaleMatrix;
    mvpData.viewMatrix = cameraMatrix;
    
    glm::mat4 perspectiveProjectionMatrix = glm::mat4(1.0f);
    perspectiveProjectionMatrix = glm::perspective(
        glm::radians(45.0f),
        (float)winWidth / (float)winHeight,
        0.1f,
        1000.0f
    );
    //! 2D Matrix with Column Major (Like OpenGL)
    perspectiveProjectionMatrix[1][1] = perspectiveProjectionMatrix[1][1] * (-1.0f);
    mvpData.projectionMatrix = perspectiveProjectionMatrix;

    WaterUBO waterUBO;
    memset((void*)&waterUBO, 0, sizeof(WaterUBO));

    // lightPosition = lightDirection * 50.0f;

    // waterUBO.lightPosition = glm::vec4(lightPosition, 0.0f);
    // waterUBO.lightAmbient = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    // waterUBO.lightDiffuse = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
    // waterUBO.lightSpecular = glm::vec4(1.0f, 0.9f, 0.7f, 0.0f);
    // waterUBO.viewPosition = glm::vec4(30.0f, 30.0f, 60.0f, 0.0f);
    // waterUBO.heightVector = glm::vec4(heightMin * 0.1, heightMax * 0.1, 0.0f, 0.0f);

    lightPosition = lightDirection * 50.0f;
    glm::vec3 sunDir = glm::normalize(lightDirection);

    waterUBO.lightPosition = glm::vec4(sunDir, 0.0f);
    waterUBO.lightAmbient  = glm::vec4(0.15f, 0.15f, 0.15f, 0.0f);
    waterUBO.lightDiffuse = glm::vec4(0.8f, 0.9f, 1.0f, 0.0f);
    waterUBO.lightSpecular = glm::vec4(1.0f, 0.9f, 0.7f, 0.0f);
    waterUBO.viewPosition = glm::vec4(30.0f, 30.0f, 60.0f, 0.0f);

    glm::mat4 model = mvpData.modelMatrix;
    float scaleY = glm::length(glm::vec3(model[1]));     // approximate Y scale (works for orthonormal+scale)
    float translateY = model[3].y;                       // translation.y

    float worldHeightMin = heightMin * scaleY + translateY;
    float worldHeightMax = heightMax * scaleY + translateY;

    waterUBO.heightVector = glm::vec4(worldHeightMin, worldHeightMax, 0.0f, 0.0f);

    // waterUBO.heightVector = glm::vec4(heightMin * 0.1, heightMax * 0.1, 0.0f, 0.0f);

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
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding_array[3];
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

    //! Sun Texture
    vkDescriptorSetLayoutBinding_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorSetLayoutBinding_array[2].binding = 2;   //! Mapped with layout(binding = 1) in fragment shader
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
    VkDescriptorPoolSize vkDescriptorPoolSize_array[2];
    memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * _ARRAYSIZE(vkDescriptorPoolSize_array));

    vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    vkDescriptorPoolSize_array[0].descriptorCount = 2;

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

    VkDescriptorImageInfo vkDescriptorImageInfo_array[1];
    memset((void*)vkDescriptorImageInfo_array, 0, sizeof(VkDescriptorImageInfo) * _ARRAYSIZE(vkDescriptorImageInfo_array));
    vkDescriptorImageInfo_array[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkDescriptorImageInfo_array[0].imageView = vkImageView_texture_sun;
    vkDescriptorImageInfo_array[0].sampler = vkSampler_texture_sun;


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

    //! Sun Texture
    vkWriteDescriptorSet_array[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet_array[2].pNext = NULL;
    vkWriteDescriptorSet_array[2].dstSet = vkDescriptorSet_ocean;
    vkWriteDescriptorSet_array[2].dstArrayElement = 0;
    vkWriteDescriptorSet_array[2].dstBinding = 2;
    vkWriteDescriptorSet_array[2].descriptorCount = 1;
    vkWriteDescriptorSet_array[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkWriteDescriptorSet_array[2].pBufferInfo = NULL;
    vkWriteDescriptorSet_array[2].pImageInfo = &vkDescriptorImageInfo_array[0];
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
    // fTime += waveSpeed;

    // cameraMatrix = cameraViewMatrix;
    
    // //* Build Tessendorf Mesh
    // generate_fft_data(fTime);

    // for (int i = 0; i < N; i++)
    // {
    //     for (int j = 0; j < M; j++)
    //     {
    //         int index = j * N + i;

    //         if (displacement_map[index][1] > heightMax)
    //             heightMax = displacement_map[index][1];
    //         else if (displacement_map[index][1] < heightMin)
    //             heightMin = displacement_map[index][1];
    //     }
    // }

    // //* Copy Displacement Data
    // memcpy(displacementPtr, displacement_map, meshSize);
    // memcpy(normalsPtr, normal_map, meshSize);

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

    //* Texture Related
    if (vkSampler_texture_sun)
    {
        vkDestroySampler(vkDevice, vkSampler_texture_sun, NULL);
        vkSampler_texture_sun = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkDestroySampler() Succeeded For vkSampler_texture_sun\n", __func__);
    }

    if (vkImageView_texture_sun)
    {
        vkDestroyImageView(vkDevice, vkImageView_texture_sun, NULL);
        vkImageView_texture_sun = NULL;
        fprintf(gpFile, "%s() => vkDestroyImageView() Succeeded For vkImage_texture_sun\n", __func__);
    }

    if (vkDeviceMemory_texture_sun)
    {
        vkFreeMemory(vkDevice, vkDeviceMemory_texture_sun, NULL);
        vkDeviceMemory_texture_sun = VK_NULL_HANDLE;
        fprintf(gpFile, "%s() => vkFreeMemory() Succeeded For vkDeviceMemory_texture_sun\n", __func__);
    }

    if (vkImage_texture_sun)
    {
        vkDestroyImage(vkDevice, vkImage_texture_sun, NULL);
        vkImage_texture_sun = NULL;
        fprintf(gpFile, "%s() => vkDestroyImage() Succeeded For vkImage_texture_sun\n", __func__);
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

VkResult Ocean::createFFTBuffer()
{
    VkResult vkResult = VK_SUCCESS;

    // FFT data count
    size_t fftCount = N * M;

    // Each entry is a vec2<float> (real + imag)
    size_t singleSectionSize = fftCount * sizeof(glm::vec2);

    // We'll pack: h_twiddle_0, h_twiddle_0_conj, h_twiddle
    fftDataSectionOffsets.h0 = 0;
    fftDataSectionOffsets.h0_conj = singleSectionSize;
    fftDataSectionOffsets.h_twiddle = singleSectionSize * 2;
    fftDataTotalSize = singleSectionSize * 3;

    // -------------------------------------------------------------------------
    // Create unified FFT buffer
    // -------------------------------------------------------------------------
    memset(&fftData, 0, sizeof(BufferData));

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = fftDataTotalSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    vkResult = vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &fftData.vkBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "Failed to create FFT buffer: %d\n", vkResult);
        return vkResult;
    }

    // Query memory requirements
    VkMemoryRequirements memReqs{};
    vkGetBufferMemoryRequirements(vkDevice, fftData.vkBuffer, &memReqs);

    // Allocate memory
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;

    uint32_t memoryTypeIndex = 0;
    for (uint32_t i = 0; i < vkPhysicalDeviceMemoryProperties.memoryTypeCount; i++)
    {
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))
        {
            memoryTypeIndex = i;
            break;
        }
    }

    allocInfo.memoryTypeIndex = memoryTypeIndex;

    vkResult = vkAllocateMemory(vkDevice, &allocInfo, nullptr, &fftData.vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "Failed to allocate FFT buffer memory: %d\n", vkResult);
        return vkResult;
    }

    vkBindBufferMemory(vkDevice, fftData.vkBuffer, fftData.vkDeviceMemory, 0);

    // Optional: map for CPU updates if needed
    vkMapMemory(vkDevice, fftData.vkDeviceMemory, 0, fftDataTotalSize, 0, &fftDataMapped);

    fprintf(gpFile, "Created interleaved FFT buffer of size %.2f MB\n", fftDataTotalSize / (1024.0 * 1024.0));

    return VK_SUCCESS;
}


VkResult Ocean::createComputeDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(3);

    for (uint32_t i = 0; i < 3; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    return vkCreateDescriptorSetLayout(vkDevice, &layoutInfo, nullptr, &vkDescriptorSetLayout_compute);
}

VkResult Ocean::createComputePipeline()
{
    VkResult result;

    VkPipelineLayoutCreateInfo layoutInfo{};

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(OceanPushConstants);

    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &vkDescriptorSetLayout_compute;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;

    result = vkCreatePipelineLayout(vkDevice, &layoutInfo, nullptr, &vkPipelineLayout_compute);
    if (result != VK_SUCCESS) return result;

    VkPipelineShaderStageCreateInfo shaderStage{};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStage.module = vkShaderModule_compute_shader;
    shaderStage.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStage;
    pipelineInfo.layout = vkPipelineLayout_compute;

    result = vkCreateComputePipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vkPipeline_compute);
    return result;
}


VkResult Ocean::createComputeDescriptorSet()
{
    VkDescriptorPoolSize vkDescriptorPoolSize_array[1];
    memset((void*)vkDescriptorPoolSize_array, 0, sizeof(VkDescriptorPoolSize) * _ARRAYSIZE(vkDescriptorPoolSize_array));

    vkDescriptorPoolSize_array[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    vkDescriptorPoolSize_array[0].descriptorCount = 3;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = _ARRAYSIZE(vkDescriptorPoolSize_array);
    poolInfo.pPoolSizes = vkDescriptorPoolSize_array;
    poolInfo.maxSets = 1;

    vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &vkDescriptorPool_compute);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = vkDescriptorPool_compute;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &vkDescriptorSetLayout_compute;

    vkAllocateDescriptorSets(vkDevice, &allocInfo, &vkDescriptorSet_compute);

    //* Describe whether we want buffer as uniform or image as uniform
    VkDescriptorBufferInfo vkDescriptorBufferInfo_array[3];
    memset((void*)vkDescriptorBufferInfo_array, 0, sizeof(VkDescriptorBufferInfo) * _ARRAYSIZE(vkDescriptorBufferInfo_array));

    vkDescriptorBufferInfo_array[0].buffer = fftData.vkBuffer;
    vkDescriptorBufferInfo_array[0].offset = 0;
    vkDescriptorBufferInfo_array[0].range = VK_WHOLE_SIZE;

    vkDescriptorBufferInfo_array[1].buffer = vertexData_displacement.vkBuffer;
    vkDescriptorBufferInfo_array[1].offset = 0;
    vkDescriptorBufferInfo_array[1].range = VK_WHOLE_SIZE;

    vkDescriptorBufferInfo_array[2].buffer = vertexData_normals.vkBuffer;
    vkDescriptorBufferInfo_array[2].offset = 0;
    vkDescriptorBufferInfo_array[2].range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet vkWriteDescriptorSet_array[3];
    memset((void*)vkWriteDescriptorSet_array, 0, sizeof(VkWriteDescriptorSet) * _ARRAYSIZE(vkWriteDescriptorSet_array));

    for (uint32_t i = 0; i < 3; i++)
    {
        vkWriteDescriptorSet_array[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        vkWriteDescriptorSet_array[i].pNext = NULL;
        vkWriteDescriptorSet_array[i].dstSet = vkDescriptorSet_compute;
        vkWriteDescriptorSet_array[i].dstArrayElement = 0;
        vkWriteDescriptorSet_array[i].dstBinding = i;
        vkWriteDescriptorSet_array[i].descriptorCount = 1;
        vkWriteDescriptorSet_array[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        vkWriteDescriptorSet_array[i].pBufferInfo = &vkDescriptorBufferInfo_array[i];
        vkWriteDescriptorSet_array[i].pImageInfo = NULL;
        vkWriteDescriptorSet_array[i].pTexelBufferView = NULL;
    }

    vkUpdateDescriptorSets(vkDevice, _ARRAYSIZE(vkWriteDescriptorSet_array), vkWriteDescriptorSet_array, 0, NULL);

    return VK_SUCCESS;
}

