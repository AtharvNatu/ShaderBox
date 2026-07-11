#include "Overlay.hpp"
#include "Utils.h"
#include "imgui.h"

#include <windowsx.h>

Overlay::Overlay()
{
    // Code

    //! Setup ImGui Context
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    io.Fonts->AddFontFromFileTTF("ImGui\\Poppins-Regular.ttf", 24.0f, NULL, io.Fonts->GetGlyphRangesDefault());
}

void Overlay::addMouseMoveHandler(LPARAM lParam)
{
    // Code
    ImGui::GetIO().AddMousePosEvent((float)GET_X_LPARAM(lParam), (float)GET_Y_LPARAM(lParam));
}

void Overlay::addMouseButtonHandler(int buttonIndex, bool status)
{
    // Code
    ImGui::GetIO().AddMouseButtonEvent(buttonIndex, status);
}

void Overlay::addMouseWheelHandler(WPARAM wParam)
{
    // Code
    ImGui::GetIO().AddMouseWheelEvent(0.0f, GET_WHEEL_DELTA_WPARAM(wParam) / (float)WHEEL_DELTA);
}

void Overlay::addKeyboardHandler(WPARAM wParam)
{
    // Code
    ImGuiIO& io = ImGui::GetIO();

    io.AddKeyEvent(ImGuiKey_ModCtrl,  (GetKeyState(VK_CONTROL) & 0x8000) != 0);
    io.AddKeyEvent(ImGuiKey_ModShift, (GetKeyState(VK_SHIFT) & 0x8000) != 0);
    io.AddKeyEvent(ImGuiKey_ModAlt,   (GetKeyState(VK_MENU) & 0x8000) != 0);

    ImGuiKey key = ImGuiKey_None;

    switch (wParam)
    {
        case VK_TAB:    key = ImGuiKey_Tab; break;
        case VK_LEFT:   key = ImGuiKey_LeftArrow; break;
        case VK_RIGHT:  key = ImGuiKey_RightArrow; break;
        case VK_UP:     key = ImGuiKey_UpArrow; break;
        case VK_DOWN:   key = ImGuiKey_DownArrow; break;
        case VK_ESCAPE: key = ImGuiKey_Escape; break;
        case VK_RETURN: key = ImGuiKey_Enter; break;
        case VK_SPACE:  key = ImGuiKey_Space; break;
    }

    if (key != ImGuiKey_None)
        io.AddKeyEvent(key, true);
}

VkResult Overlay::initialize(float width, float height)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    overlayWidth = width;
    overlayHeight = height;

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(overlayWidth, overlayHeight);
    io.DisplayFramebufferScale = ImVec2(1.0f, 1.0f);

    ImGui::StyleColorsDark();
    
    vkResult = createFontTexture();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createFontTexture() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else 
        fprintf(gpFile, "%s() => createFontTexture() Succeeded\n", __func__);

    vkResult = createShaders();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createShaders() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else 
        fprintf(gpFile, "%s() => createShaders() Succeeded\n", __func__);

    vkResult = createDescriptorPool();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorPool() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else 
        fprintf(gpFile, "%s() => createDescriptorPool() Succeeded\n", __func__);

    vkResult = createDescriptorSetLayout();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else 
        fprintf(gpFile, "%s() => createDescriptorSetLayout() Succeeded\n", __func__);

    vkResult = createDescriptorSet();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createDescriptorSet() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else 
        fprintf(gpFile, "%s() => createDescriptorSet() Succeeded\n", __func__);

    vkResult = createPipelineLayout();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createPipelineLayout() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else 
        fprintf(gpFile, "%s() => createPipelineLayout() Succeeded\n", __func__);

    vkResult = createPipeline();
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => createPipeline() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else 
        fprintf(gpFile, "%s() => createPipeline() Succeeded\n", __func__);

    return vkResult;
}

VkResult Overlay::createBuffer(BufferData* bufferData, VkBufferUsageFlagBits bufferUsageFlagBits, VkDeviceSize bufferSize)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    memset((void*)bufferData, 0, sizeof(BufferData));

    //* Step - 5
    VkBufferCreateInfo vkBufferCreateInfo;
    memset((void*)&vkBufferCreateInfo, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo.flags = 0;
    vkBufferCreateInfo.pNext = NULL;
    vkBufferCreateInfo.size = bufferSize;
    vkBufferCreateInfo.usage = bufferUsageFlagBits;

    //* Step - 6
    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo, NULL, &bufferData->vkBuffer);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed : %s !!!\n", __func__, getVkResultString(vkResult));

    //* Step - 7
    VkMemoryRequirements vkMemoryRequirements;
    memset((void*)&vkMemoryRequirements, 0, sizeof(VkMemoryRequirements));
    vkGetBufferMemoryRequirements(vkDevice, bufferData->vkBuffer, &vkMemoryRequirements);

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
            if (vkPhysicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
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
    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo, NULL, &bufferData->vkDeviceMemory);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed : %s !!!\n", __func__, getVkResultString(vkResult));

    //* Step - 10
    //! Binds Vulkan Device Memory Object Handle with the Vulkan Buffer Object Handle
    vkResult = vkBindBufferMemory(vkDevice, bufferData->vkBuffer, bufferData->vkDeviceMemory, 0);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed : %s !!!\n", __func__, getVkResultString(vkResult));

    return vkResult;
}

VkResult Overlay::mapBufferMemory(BufferData *bufferData)
{
    // Code
    return vkMapMemory(vkDevice, bufferData->vkDeviceMemory, 0, VK_WHOLE_SIZE, 0, &bufferData->mapped);
}

void Overlay::unmapBufferMemory(BufferData *bufferData)
{
    // Code
    if (bufferData->mapped)
    {
        vkUnmapMemory(vkDevice, bufferData->vkDeviceMemory);
        bufferData->mapped = nullptr;
    }
    
}

void Overlay::destroyBuffer(BufferData *bufferData)
{
    // Code
    if (bufferData->mapped)
    {
        vkUnmapMemory(vkDevice, bufferData->vkDeviceMemory);
        bufferData->mapped = nullptr;
    }

    if (bufferData->vkBuffer)
    {
        vkDestroyBuffer(vkDevice, bufferData->vkBuffer, NULL);
        bufferData->vkBuffer = NULL;
    }

    if (bufferData->vkDeviceMemory)
    {
        vkFreeMemory(vkDevice, bufferData->vkDeviceMemory, NULL);
        bufferData->vkDeviceMemory = NULL;
    }
}

VkResult Overlay::createFontTexture()
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;
    unsigned char* fontData = NULL;
    int textureWidth = 0, textureHeight = 0;

    VkBuffer vkBuffer_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
    VkDeviceSize fontUploadSize;

    // Code
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->GetTexDataAsRGBA32(&fontData, &textureWidth, &textureHeight);
    fontUploadSize = textureWidth * textureHeight * 4 * sizeof(char);

    //! Step - 2
    VkBufferCreateInfo vkBufferCreateInfo_stagingBuffer;
    memset((void*)&vkBufferCreateInfo_stagingBuffer, 0, sizeof(VkBufferCreateInfo));
    vkBufferCreateInfo_stagingBuffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    vkBufferCreateInfo_stagingBuffer.pNext = NULL;
    vkBufferCreateInfo_stagingBuffer.flags = 0;
    vkBufferCreateInfo_stagingBuffer.size = fontUploadSize;
    vkBufferCreateInfo_stagingBuffer.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkBufferCreateInfo_stagingBuffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;  //* This denotes source data to be transferred to VkImage

    vkResult = vkCreateBuffer(vkDevice, &vkBufferCreateInfo_stagingBuffer, NULL, &vkBuffer_stagingBuffer);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateBuffer() Failed For Staging Buffer : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }

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
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Staging Buffer : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    vkResult = vkBindBufferMemory(vkDevice, vkBuffer_stagingBuffer, vkDeviceMemory_stagingBuffer, 0);
    if (vkResult != VK_SUCCESS)
    { 
        fprintf(gpFile, "%s() => vkBindBufferMemory() Failed For Staging Buffer : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    void* data = NULL;
    vkResult = vkMapMemory(
        vkDevice, 
        vkDeviceMemory_stagingBuffer,
        0, 
        fontUploadSize, 
        0, 
        &data
    );
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkMapMemory() Failed For Staging Buffer : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }
        return vkResult;
    }

    memcpy(data, fontData, fontUploadSize);

    vkUnmapMemory(vkDevice, vkDeviceMemory_stagingBuffer);

    //! Step - 3
    VkImageCreateInfo vkImageCreateInfo;
    memset((void*)&vkImageCreateInfo, 0, sizeof(VkImageCreateInfo));
    vkImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    vkImageCreateInfo.flags = 0;
    vkImageCreateInfo.pNext = NULL;
    vkImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    vkImageCreateInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    vkImageCreateInfo.extent.width = textureWidth;
    vkImageCreateInfo.extent.height = textureHeight;
    vkImageCreateInfo.extent.depth = 1;
    vkImageCreateInfo.mipLevels = 1;
    vkImageCreateInfo.arrayLayers = 1;
    vkImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    vkImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    vkImageCreateInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    vkImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    vkResult = vkCreateImage(vkDevice, &vkImageCreateInfo, NULL, &fontImage);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateImage() Failed For Font Texture : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }     

    VkMemoryRequirements vkMemoryRequirements_image;
    memset((void*)&vkMemoryRequirements_image, 0, sizeof(VkMemoryRequirements));
    vkGetImageMemoryRequirements(vkDevice, fontImage, &vkMemoryRequirements_image);

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

    vkResult = vkAllocateMemory(vkDevice, &vkMemoryAllocateInfo_image, NULL, &fontMemory);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateMemory() Failed For Font Texture : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    vkResult = vkBindImageMemory(vkDevice, fontImage, fontMemory, 0);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBindImageMemory() Failed For Font Texture : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    } 

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
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }   

    //* Step - 4.2
    VkCommandBufferBeginInfo vkCommandBufferBeginInfo_image_transition_layout;
    memset((void*)&vkCommandBufferBeginInfo_image_transition_layout, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_image_transition_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_image_transition_layout.pNext = NULL;
    vkCommandBufferBeginInfo_image_transition_layout.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_transition_image_layout, &vkCommandBufferBeginInfo_image_transition_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

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
    vkImageMemoryBarrier.image = fontImage;
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
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
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
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

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
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 4.6
    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkSubmitInfo_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 4.7
    if (vkCommandBuffer_transition_image_layout)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
        vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
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
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_buffer_to_image_copy : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    VkCommandBufferBeginInfo vkCommandBufferBeginInfo_buffer_to_image_copy;
    memset((void*)&vkCommandBufferBeginInfo_buffer_to_image_copy, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_buffer_to_image_copy.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_buffer_to_image_copy.pNext = NULL;
    vkCommandBufferBeginInfo_buffer_to_image_copy.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_buffer_to_image_copy, &vkCommandBufferBeginInfo_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_buffer_to_image_copy : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

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
    vkBufferImageCopy.imageExtent.width = textureWidth;
    vkBufferImageCopy.imageExtent.height = textureHeight;
    vkBufferImageCopy.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(
        vkCommandBuffer_buffer_to_image_copy,
        vkBuffer_stagingBuffer,
        fontImage,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &vkBufferImageCopy
    );

    vkResult = vkEndCommandBuffer(vkCommandBuffer_buffer_to_image_copy);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_buffer_to_image_copy : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (fontImage)
        {
            vkDestroyImage(vkDevice, fontImage, NULL);
            fontImage = NULL;
        }
        if (fontMemory)
        {
            vkFreeMemory(vkDevice, fontMemory, NULL);
            fontMemory = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    VkSubmitInfo vkSubmitInfo_buffer_to_copy;
    memset((void*)&vkSubmitInfo_buffer_to_copy, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_buffer_to_copy.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_buffer_to_copy.pNext = NULL;
    vkSubmitInfo_buffer_to_copy.commandBufferCount = 1;
    vkSubmitInfo_buffer_to_copy.pCommandBuffers = &vkCommandBuffer_buffer_to_image_copy;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_buffer_to_copy, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_buffer_to_copy : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkQueueSubmit() Succeeded For vkSubmitInfo_buffer_to_copy\n", __func__);

    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkCommandBuffer_buffer_to_image_copy : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_buffer_to_image_copy)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
            vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    if (vkCommandBuffer_buffer_to_image_copy)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_buffer_to_image_copy);
        vkCommandBuffer_buffer_to_image_copy = VK_NULL_HANDLE;
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
        fprintf(gpFile, "%s() => vkAllocateCommandBuffers() Failed For vkCommandBuffer_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }  

    //* Step - 6.2
    memset((void*)&vkCommandBufferBeginInfo_image_transition_layout, 0, sizeof(VkCommandBufferBeginInfo));
    vkCommandBufferBeginInfo_image_transition_layout.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkCommandBufferBeginInfo_image_transition_layout.pNext = NULL;
    vkCommandBufferBeginInfo_image_transition_layout.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkResult = vkBeginCommandBuffer(vkCommandBuffer_transition_image_layout, &vkCommandBufferBeginInfo_image_transition_layout);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkBeginCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;

        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

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
    vkImageMemoryBarrier.image = fontImage;
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
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
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
        fprintf(gpFile, "ERROR : %s() => vkEndCommandBuffer() Failed For vkCommandBuffer_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 6.5
    memset((void*)&vkSubmitInfo_transition_image_layout, 0, sizeof(VkSubmitInfo));
    vkSubmitInfo_transition_image_layout.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    vkSubmitInfo_transition_image_layout.pNext = NULL;
    vkSubmitInfo_transition_image_layout.commandBufferCount = 1;
    vkSubmitInfo_transition_image_layout.pCommandBuffers = &vkCommandBuffer_transition_image_layout;

    vkResult = vkQueueSubmit(vkQueue, 1, &vkSubmitInfo_transition_image_layout, VK_NULL_HANDLE);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueSubmit() Failed For vkSubmitInfo_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 6.6
    vkResult = vkQueueWaitIdle(vkQueue);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "ERROR : %s() => vkQueueWaitIdle() Failed For vkSubmitInfo_transition_image_layout : %s\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        
        //* Cleanup Code
        if (vkCommandBuffer_transition_image_layout)
        {
            vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
            vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
        }
        if (vkDeviceMemory_stagingBuffer)
        {
            vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
            vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
        }
        if (vkBuffer_stagingBuffer)
        {
            vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
            vkBuffer_stagingBuffer = VK_NULL_HANDLE;
        }

        return vkResult;
    }

    //* Step - 6.7
    if (vkCommandBuffer_transition_image_layout)
    {
        vkFreeCommandBuffers(vkDevice, vkCommandPool, 1, &vkCommandBuffer_transition_image_layout);
        vkCommandBuffer_transition_image_layout = VK_NULL_HANDLE;
    }
    //! ----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    //! Step - 7
    if (vkBuffer_stagingBuffer)
    {
        vkDestroyBuffer(vkDevice, vkBuffer_stagingBuffer, NULL);
        vkBuffer_stagingBuffer = VK_NULL_HANDLE;
    }

    if (vkDeviceMemory_stagingBuffer)
    {
        vkFreeMemory(vkDevice, vkDeviceMemory_stagingBuffer, NULL);
        vkDeviceMemory_stagingBuffer = VK_NULL_HANDLE;
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
    vkImageViewCreateInfo.image = fontImage;

    vkResult = vkCreateImageView(vkDevice, &vkImageViewCreateInfo, NULL, &fontImageView);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateImageView() Failed For Font Texture : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }   

    //! Step - 9
    VkSamplerCreateInfo vkSamplerCreateInfo;
    memset((void*)&vkSamplerCreateInfo, 0, sizeof(VkSamplerCreateInfo));
    vkSamplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    vkSamplerCreateInfo.pNext = NULL;
    vkSamplerCreateInfo.magFilter = VK_FILTER_LINEAR;
    vkSamplerCreateInfo.minFilter = VK_FILTER_LINEAR;
    vkSamplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    vkSamplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkSamplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkSamplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkSamplerCreateInfo.anisotropyEnable = VK_FALSE;
    vkSamplerCreateInfo.maxAnisotropy = 1.0f;
    vkSamplerCreateInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    vkSamplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
    vkSamplerCreateInfo.compareEnable = VK_FALSE;
    vkSamplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;

    vkResult = vkCreateSampler(vkDevice, &vkSamplerCreateInfo, NULL, &fontSampler);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkCreateSampler() Failed For Font Texture : %s !!!\n", __func__, getVkResultString(vkResult));
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }    

    return vkResult;

}

VkResult Overlay::createShaders(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Vertex Shader
    //! ---------------------------------------------------------------------------------------------------------------------------
    //* Step - 6
    const char* szFileName = "Bin/Overlay.vert.spv";
    FILE *fp = NULL;
    size_t size;

    fp = fopen(szFileName, "rb");
    if (fp == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Open SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Succeeded In Opening SPIR-V Shader File : %s\n", __func__, szFileName);

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    if (size == 0)
    {
        fprintf(gpFile, "%s() => Empty SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    fseek(fp, 0L, SEEK_SET);

    char* shaderData = (char*)malloc(size * sizeof(char));
    if (shaderData == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For shaderData !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    size_t retVal = fread(shaderData, size, 1, fp);
    if (retVal != 1)
    {
        fprintf(gpFile, "%s() => Failed To Read From SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Successfully Read Shader From SPIR-V Shader File : %s\n", __func__, szFileName);
    
    if (fp)
    {
        fclose(fp);
        fp = NULL;
        fprintf(gpFile, "%s() => Closed SPIR-V File : %s\n", __func__, szFileName);
    }

    //* Step - 7
    VkShaderModuleCreateInfo vkShaderModuleCreateInfo;
    memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
    vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0; //! Reserved, must be 0
    vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
    vkShaderModuleCreateInfo.codeSize = size;

    //* Step - 8
    vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_vertex_shader_imgui);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateShaderModule() Failed For Vertex Shader : %s !!!\n", __func__, getVkResultString(vkResult));
    else
        fprintf(gpFile, "%s() => vkCreateShaderModule() Succeeded For Vertex Shader\n", __func__);

    //* Step - 9
    if (shaderData)
    {
        free(shaderData);
        shaderData = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For shaderData\n", __func__);
    }

    fprintf(gpFile, "%s() => Vertex Shader Module Successfully Created\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------

    //! Fragment Shader
    //! ---------------------------------------------------------------------------------------------------------------------------
    szFileName = "Bin/Overlay.frag.spv";

    fp = fopen(szFileName, "rb");
    if (fp == NULL)
    {
        fprintf(gpFile, "%s() => Failed To Open SPIR-V Shader File :  %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Succeeded In Opening SPIR-V Shader File : %s\n", __func__, szFileName);

    fseek(fp, 0L, SEEK_END);
    size = ftell(fp);
    if (size == 0)
    {
        fprintf(gpFile, "%s() => Empty SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    fseek(fp, 0L, SEEK_SET);

    shaderData = (char*)malloc(size * sizeof(char));
    if (shaderData == NULL)
    {
        fprintf(gpFile, "%s() => malloc() Failed For shaderData !!!\n", __func__);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    retVal = fread(shaderData, size, 1, fp);
    if (retVal != 1)
    {
        fprintf(gpFile, "%s() => Failed To Read From SPIR-V Shader File : %s !!!", __func__, szFileName);
        vkResult = VK_ERROR_INITIALIZATION_FAILED;
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => Successfully Read Shader From SPIR-V Shader File : %s\n", __func__, szFileName);
    
    if (fp)
    {
        fclose(fp);
        fp = NULL;
        fprintf(gpFile, "%s() => Closed SPIR-V File : %s\n", __func__, szFileName);
    }

    //* Step - 7
    memset((void*)&vkShaderModuleCreateInfo, 0, sizeof(VkShaderModuleCreateInfo));
    vkShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    vkShaderModuleCreateInfo.pNext = NULL;
    vkShaderModuleCreateInfo.flags = 0; //! Reserved, must be 0
    vkShaderModuleCreateInfo.pCode = (uint32_t*)shaderData;
    vkShaderModuleCreateInfo.codeSize = size;

    //* Step - 8
    vkResult = vkCreateShaderModule(vkDevice, &vkShaderModuleCreateInfo, NULL, &vkShaderModule_fragment_shader_imgui);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateShaderModule() Failed For Fragment Shader : %s !!!\n", __func__, getVkResultString(vkResult));
    else
        fprintf(gpFile, "%s() => vkCreateShaderModule() Succeeded For Fragment Shader\n", __func__);

    //* Step - 9
    if (shaderData)
    {
        free(shaderData);
        shaderData = NULL;
        fprintf(gpFile, "%s() => free() Succeeded For shaderData\n", __func__);
    }

    fprintf(gpFile, "%s() => Fragment Shader Module Successfully Created\n", __func__);
    //! ---------------------------------------------------------------------------------------------------------------------------

    return vkResult;
}

VkResult Overlay::createDescriptorPool(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Vulkan expects decriptor pool size before creating actual descriptor pool
    VkDescriptorPoolSize vkDescriptorPoolSize;
    memset((void*)&vkDescriptorPoolSize, 0, sizeof(VkDescriptorPoolSize));
    vkDescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorPoolSize.descriptorCount = 1;

    //* Create the pool
    VkDescriptorPoolCreateInfo vkDescriptorPoolCreateInfo;
    memset((void*)&vkDescriptorPoolCreateInfo, 0, sizeof(VkDescriptorPoolCreateInfo));
    vkDescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    vkDescriptorPoolCreateInfo.pNext = NULL;
    vkDescriptorPoolCreateInfo.flags = 0;
    vkDescriptorPoolCreateInfo.poolSizeCount = 1;
    vkDescriptorPoolCreateInfo.pPoolSizes = &vkDescriptorPoolSize;
    vkDescriptorPoolCreateInfo.maxSets = 2;

    vkResult = vkCreateDescriptorPool(vkDevice, &vkDescriptorPoolCreateInfo, NULL, &vkDescriptorPool_imgui);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorPool() Succeeded\n", __func__);

    return vkResult;
}

VkResult Overlay::createDescriptorSetLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //! Initialize VkDescriptorSetLayoutBinding
    VkDescriptorSetLayoutBinding vkDescriptorSetLayoutBinding;
    memset((void*)&vkDescriptorSetLayoutBinding, 0, sizeof(VkDescriptorSetLayoutBinding));
    vkDescriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkDescriptorSetLayoutBinding.binding = 0;
    vkDescriptorSetLayoutBinding.descriptorCount = 1;
    vkDescriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkDescriptorSetLayoutBinding.pImmutableSamplers = NULL;

    //* Step - 3
    VkDescriptorSetLayoutCreateInfo vkDescriptorSetLayoutCreateInfo;
    memset((void*)&vkDescriptorSetLayoutCreateInfo, 0, sizeof(VkDescriptorSetLayoutCreateInfo));
    vkDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    vkDescriptorSetLayoutCreateInfo.pNext = NULL;
    vkDescriptorSetLayoutCreateInfo.flags = 0;
    vkDescriptorSetLayoutCreateInfo.bindingCount = 1;
    vkDescriptorSetLayoutCreateInfo.pBindings = &vkDescriptorSetLayoutBinding;

    //* Step - 4
    vkResult = vkCreateDescriptorSetLayout(vkDevice, &vkDescriptorSetLayoutCreateInfo, NULL, &vkDescriptorSetLayout_imgui);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else
        fprintf(gpFile, "%s() => vkCreateDescriptorSetLayout() Succeeded\n", __func__);

    return vkResult;
}

VkResult Overlay::createDescriptorSet(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code

    //* Initialize DescriptorSetAllocationInfo
    VkDescriptorSetAllocateInfo vkDescriptorSetAllocateInfo;
    memset((void*)&vkDescriptorSetAllocateInfo, 0, sizeof(VkDescriptorSetAllocateInfo));
    vkDescriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    vkDescriptorSetAllocateInfo.pNext = NULL;
    vkDescriptorSetAllocateInfo.descriptorPool = vkDescriptorPool_imgui;
    vkDescriptorSetAllocateInfo.descriptorSetCount = 1;
    vkDescriptorSetAllocateInfo.pSetLayouts = &vkDescriptorSetLayout_imgui;

    vkResult = vkAllocateDescriptorSets(vkDevice, &vkDescriptorSetAllocateInfo, &vkDescriptorSet_imgui);
    if (vkResult != VK_SUCCESS)
    {
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Failed For vkDescriptorSet_imgui : %s !!!\n", __func__, getVkResultString(vkResult));
        return vkResult;
    }
    else
        fprintf(gpFile, "%s() => vkAllocateDescriptorSets() Succeeded For vkDescriptorSet_imgui\n", __func__);

    //! Descriptor Image Info
    VkDescriptorImageInfo vkDescriptorImageInfo;
    memset((void*)&vkDescriptorImageInfo, 0, sizeof(VkDescriptorImageInfo));
    vkDescriptorImageInfo.imageView = fontImageView;
    vkDescriptorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkDescriptorImageInfo.sampler = fontSampler;

    /* Update above descriptor set directly to the shader
    There are 2 ways :-
        1) Writing directly to the shader
        2) Copying from one shader to another shader
    */
    VkWriteDescriptorSet vkWriteDescriptorSet;
    memset((void*)&vkWriteDescriptorSet, 0, sizeof(VkWriteDescriptorSet));
    vkWriteDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    vkWriteDescriptorSet.pNext = NULL;
    vkWriteDescriptorSet.dstSet = vkDescriptorSet_imgui;
    vkWriteDescriptorSet.dstArrayElement = 0;
    vkWriteDescriptorSet.descriptorCount = 1;
    vkWriteDescriptorSet.dstBinding = 0;
    vkWriteDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    vkWriteDescriptorSet.pBufferInfo = NULL;
    vkWriteDescriptorSet.pImageInfo = &vkDescriptorImageInfo;
    vkWriteDescriptorSet.pTexelBufferView = NULL;

    vkUpdateDescriptorSets(vkDevice, 1, &vkWriteDescriptorSet, 0, NULL);

    return vkResult;
}

VkResult Overlay::createPipelineLayout(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    VkPushConstantRange vkPushConstantRange;
    memset((void*)&vkPushConstantRange, 0, sizeof(VkPushConstantRange));
    vkPushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    vkPushConstantRange.offset = 0;
    vkPushConstantRange.size = sizeof(PushConstants);

    //* Step - 3
    VkPipelineLayoutCreateInfo vkPipelineLayoutCreateInfo;
    memset((void*)&vkPipelineLayoutCreateInfo, 0, sizeof(VkPipelineLayoutCreateInfo));
    vkPipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    vkPipelineLayoutCreateInfo.pNext = NULL;
    vkPipelineLayoutCreateInfo.flags = 0;
    vkPipelineLayoutCreateInfo.setLayoutCount = 1;
    vkPipelineLayoutCreateInfo.pSetLayouts = &vkDescriptorSetLayout_imgui;
    vkPipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    vkPipelineLayoutCreateInfo.pPushConstantRanges = &vkPushConstantRange;

    //* Step - 4
    vkResult = vkCreatePipelineLayout(vkDevice, &vkPipelineLayoutCreateInfo, NULL, &vkPipelineLayout_imgui);
    if (vkResult != VK_SUCCESS)
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Failed : %s !!!\n", __func__, getVkResultString(vkResult));
    else
        fprintf(gpFile, "%s() => vkCreatePipelineLayout() Succeeded\n", __func__);

    return vkResult;
}

VkResult Overlay::createPipeline(void)
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    //* Code

    //! Vertex Input State
    VkVertexInputBindingDescription vkVertexInputBindingDescription_array[1];
    memset((void*)vkVertexInputBindingDescription_array, 0, sizeof(VkVertexInputBindingDescription) * _ARRAYSIZE(vkVertexInputBindingDescription_array));
    vkVertexInputBindingDescription_array[0].binding = 0;
    vkVertexInputBindingDescription_array[0].stride = sizeof(ImDrawVert);
    vkVertexInputBindingDescription_array[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vkVertexInputAttributeDescription_array[3];
    memset((void*)vkVertexInputAttributeDescription_array, 0, sizeof(VkVertexInputAttributeDescription) * _ARRAYSIZE(vkVertexInputAttributeDescription_array));

    //! Location 0 : Position
    vkVertexInputAttributeDescription_array[0].binding = 0;
    vkVertexInputAttributeDescription_array[0].location = 0;
    vkVertexInputAttributeDescription_array[0].format = VK_FORMAT_R32G32_SFLOAT;
    vkVertexInputAttributeDescription_array[0].offset = offsetof(ImDrawVert, pos);

    //! Location 1 : Texcoord
    vkVertexInputAttributeDescription_array[1].binding = 0;
    vkVertexInputAttributeDescription_array[1].location = 1;
    vkVertexInputAttributeDescription_array[1].format = VK_FORMAT_R32G32_SFLOAT;
    vkVertexInputAttributeDescription_array[1].offset = offsetof(ImDrawVert, uv);

    //! Location 2 : Color
    vkVertexInputAttributeDescription_array[2].binding = 0;
    vkVertexInputAttributeDescription_array[2].location = 2;
    vkVertexInputAttributeDescription_array[2].format = VK_FORMAT_R8G8B8A8_UNORM;
    vkVertexInputAttributeDescription_array[2].offset = offsetof(ImDrawVert, col);

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
    vkPipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

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
    vkPipelineColorBlendAttachmentState_array[0].blendEnable = VK_TRUE;
    vkPipelineColorBlendAttachmentState_array[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    vkPipelineColorBlendAttachmentState_array[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    vkPipelineColorBlendAttachmentState_array[0].colorBlendOp = VK_BLEND_OP_ADD;
    vkPipelineColorBlendAttachmentState_array[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    vkPipelineColorBlendAttachmentState_array[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    vkPipelineColorBlendAttachmentState_array[0].alphaBlendOp = VK_BLEND_OP_ADD;

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
    VkDynamicState vkDynamicState_array[2];
    vkDynamicState_array[0] = VK_DYNAMIC_STATE_VIEWPORT;
    vkDynamicState_array[1] = VK_DYNAMIC_STATE_SCISSOR;
    
    VkPipelineDynamicStateCreateInfo vkPipelineDynamicStateCreateInfo;
    memset((void*)&vkPipelineDynamicStateCreateInfo, 0, sizeof(VkPipelineDynamicStateCreateInfo));
    vkPipelineDynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    vkPipelineDynamicStateCreateInfo.pNext = NULL;
    vkPipelineDynamicStateCreateInfo.flags = 0;
    vkPipelineDynamicStateCreateInfo.dynamicStateCount = _ARRAYSIZE(vkDynamicState_array);
    vkPipelineDynamicStateCreateInfo.pDynamicStates = vkDynamicState_array;

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
    vkPipelineShaderStageCreateInfo_array[0].module = vkShaderModule_vertex_shader_imgui;
    vkPipelineShaderStageCreateInfo_array[0].pName = "main";
    vkPipelineShaderStageCreateInfo_array[0].pSpecializationInfo = NULL;

    //* Fragment Shader
    vkPipelineShaderStageCreateInfo_array[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vkPipelineShaderStageCreateInfo_array[1].pNext = NULL;
    vkPipelineShaderStageCreateInfo_array[1].flags = 0;
    vkPipelineShaderStageCreateInfo_array[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    vkPipelineShaderStageCreateInfo_array[1].module = vkShaderModule_fragment_shader_imgui;
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
    vkGraphicsPipelineCreateInfo.pDynamicState = &vkPipelineDynamicStateCreateInfo;
    vkGraphicsPipelineCreateInfo.pMultisampleState = &vkPipelineMultisampleStateCreateInfo;
    vkGraphicsPipelineCreateInfo.stageCount = _ARRAYSIZE(vkPipelineShaderStageCreateInfo_array);
    vkGraphicsPipelineCreateInfo.pStages = vkPipelineShaderStageCreateInfo_array;
    vkGraphicsPipelineCreateInfo.pTessellationState = NULL;
    vkGraphicsPipelineCreateInfo.layout = vkPipelineLayout_imgui;
    vkGraphicsPipelineCreateInfo.renderPass = vkRenderPass;
    vkGraphicsPipelineCreateInfo.subpass = 0;
    vkGraphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    vkGraphicsPipelineCreateInfo.basePipelineIndex = 0;

    vkResult = vkCreateGraphicsPipelines(vkDevice, vkPipelineCache, 1, &vkGraphicsPipelineCreateInfo, NULL, &vkPipeline_imgui);
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

void Overlay::newFrame(bool updateFrameGraph, float deltaTime)
{
    // Code
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(overlayWidth, overlayHeight);

    if (updateFrameGraph)
        io.DeltaTime = deltaTime;

    ImGui::NewFrame();

    ImGui::SetWindowSize(ImVec2(overlayWidth, overlayHeight));

    ImGui::Begin("Vulkan : ImGui");
    {
        ImGui::Text("Cube Rotation Speed");
        ImGui::SliderFloat("##", (float*)&data.cubeAnimationSpeed, 0.001f, 0.1f);
    }
    ImGui::End();

    ImGui::Render();
}

void Overlay::updateBuffers()
{
    // Variable Declarations
    VkResult vkResult = VK_SUCCESS;

    // Code
    ImDrawData* imDrawData = ImGui::GetDrawData();

    VkDeviceSize vertexBufferSize = imDrawData->TotalVtxCount * sizeof(ImDrawVert);
    VkDeviceSize indexBufferSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);

    if ((vertexBufferSize == 0) || (indexBufferSize == 0))
        return;

    //! Update Vertex and Index Buffers containing the ImGui elements only when vertex or index count has changed compared to current buffer

    //! Vertex Buffer
    if ((vertexBuffer.vkBuffer == VK_NULL_HANDLE) || (vertexCount != imDrawData->TotalVtxCount))
    {
        unmapBufferMemory(&vertexBuffer);
        
        destroyBuffer(&vertexBuffer);

        vkResult = createBuffer(&vertexBuffer, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBufferSize);
        if (vkResult != VK_SUCCESS)
            fprintf(gpFile, "%s() => createBuffer() Failed For Vertex Buffer: %s !!!", __func__, getVkResultString(vkResult));

        vertexCount = imDrawData->TotalVtxCount;

        vkResult = mapBufferMemory(&vertexBuffer);
        if (vkResult != VK_SUCCESS)
            fprintf(gpFile, "%s() => mapBufferMemory() Failed For Vertex Buffer: %s !!!", __func__, getVkResultString(vkResult));
        
    }

    //! Index Buffer
    if ((indexBuffer.vkBuffer == VK_NULL_HANDLE) || (indexCount != imDrawData->TotalIdxCount))
    {
        unmapBufferMemory(&indexBuffer);
        
        destroyBuffer(&indexBuffer);

        vkResult = createBuffer(&indexBuffer, VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBufferSize);
        if (vkResult != VK_SUCCESS)
            fprintf(gpFile, "%s() => createBuffer() Failed For Index Buffer : %s !!!", __func__, getVkResultString(vkResult));

        indexCount = imDrawData->TotalIdxCount;

        vkResult = mapBufferMemory(&indexBuffer);
        if (vkResult != VK_SUCCESS)
            fprintf(gpFile, "%s() => mapBufferMemory() Failed For Index Buffer: %s !!!", __func__, getVkResultString(vkResult));
    }

    //* Upload Data
    ImDrawVert* vertexDst = (ImDrawVert*)vertexBuffer.mapped;
    ImDrawIdx* indexDst = (ImDrawIdx*)indexBuffer.mapped;

    for (int i = 0; i < imDrawData->CmdListsCount; i++)
    {
        const ImDrawList* drawCmdList = imDrawData->CmdLists[i];

        memcpy(vertexDst, drawCmdList->VtxBuffer.Data, drawCmdList->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(indexDst, drawCmdList->IdxBuffer.Data, drawCmdList->IdxBuffer.Size * sizeof(ImDrawIdx));

        vertexDst += drawCmdList->VtxBuffer.Size;
        indexDst += drawCmdList->IdxBuffer.Size;
    }
}

void Overlay::drawFrame(VkCommandBuffer commandBuffer)
{
    // Code
    ImGuiIO& io = ImGui::GetIO();

    //! Bind with Pipeline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, vkPipeline_imgui);

    //! Bind the Descriptor Set to the Pipeline
    vkCmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        vkPipelineLayout_imgui,
        0,
        1,
        &vkDescriptorSet_imgui,
        0,
        NULL
    );

    VkViewport viewport;
    memset((void*)&viewport, 0, sizeof(VkViewport));
    viewport.width = ImGui::GetIO().DisplaySize.x;
    viewport.height = ImGui::GetIO().DisplaySize.y;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    //* UI scale and translate via push constants
    pushData.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
    pushData.translate = glm::vec2(-1.0f);
    vkCmdPushConstants(commandBuffer, vkPipelineLayout_imgui, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstants), &pushData);

    //! Render Commands
    ImDrawData* imDrawData = ImGui::GetDrawData();
    int32_t vertexOffset = 0;
    int32_t indexOffset = 0;

    if (imDrawData->CmdListsCount > 0)
    {
        //! Bind with Vertex and Index Buffer
        VkDeviceSize offsets[1] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.vkBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer.vkBuffer, 0, VK_INDEX_TYPE_UINT16);

        for (int32_t i = 0; i < imDrawData->CmdListsCount; i++)
        {
            const ImDrawList* cmdList = imDrawData->CmdLists[i];

            for (int32_t j = 0; j < cmdList->CmdBuffer.Size; j++)
            {
                const ImDrawCmd* pCmd = &cmdList->CmdBuffer[j];
                
                VkRect2D scissorRect;
                scissorRect.offset.x = std::max((int32_t)(pCmd->ClipRect.x), 0);
                scissorRect.offset.y = std::max((int32_t)(pCmd->ClipRect.y), 0);
                scissorRect.extent.width = (uint32_t)(pCmd->ClipRect.z - pCmd->ClipRect.x);
                scissorRect.extent.height = (uint32_t)(pCmd->ClipRect.w - pCmd->ClipRect.y);
                vkCmdSetScissor(commandBuffer, 0, 1, &scissorRect);


                vkCmdDrawIndexed(
                    commandBuffer, 
                    pCmd->ElemCount, 
                    1, 
                    indexOffset + pCmd->IdxOffset, 
                    vertexOffset + pCmd->VtxOffset, 
                    0);
            }

            indexOffset += cmdList->IdxBuffer.Size;
            vertexOffset += cmdList->VtxBuffer.Size;
        }
    }
}

Overlay::~Overlay()
{
    // Code
    ImGui::DestroyContext();

    if (vkDevice)
        vkDeviceWaitIdle(vkDevice);

    if (vkPipelineLayout_imgui)
    {
        vkDestroyPipelineLayout(vkDevice, vkPipelineLayout_imgui, NULL);
        vkPipelineLayout_imgui = VK_NULL_HANDLE;
    }

    if (vkPipeline_imgui)
    {
        vkDestroyPipeline(vkDevice, vkPipeline_imgui, NULL);
        vkPipeline_imgui = VK_NULL_HANDLE;
    }

    if (vkDescriptorPool_imgui)
    {
        vkDestroyDescriptorPool(vkDevice, vkDescriptorPool_imgui, NULL);
        vkDescriptorPool_imgui = VK_NULL_HANDLE;
        vkDescriptorSet_imgui = VK_NULL_HANDLE;
    }

    if (vkDescriptorSetLayout_imgui)
    {
        vkDestroyDescriptorSetLayout(vkDevice, vkDescriptorSetLayout_imgui, NULL);
        vkDescriptorSetLayout_imgui = VK_NULL_HANDLE;
    }

    if (vkShaderModule_fragment_shader_imgui)
    {
        vkDestroyShaderModule(vkDevice, vkShaderModule_fragment_shader_imgui, NULL);
        vkShaderModule_fragment_shader_imgui = VK_NULL_HANDLE;
    }

    if (vkShaderModule_vertex_shader_imgui)
    {
        vkDestroyShaderModule(vkDevice, vkShaderModule_vertex_shader_imgui, NULL);
        vkShaderModule_vertex_shader_imgui = VK_NULL_HANDLE;
    }

    //* Texture Related
    if (fontSampler)
    {
        vkDestroySampler(vkDevice, fontSampler, NULL);
        fontSampler = VK_NULL_HANDLE;
    }

    if (fontImageView)
    {
        vkDestroyImageView(vkDevice, fontImageView, NULL);
        fontImageView = NULL;
    }

    if (fontMemory)
    {
        vkFreeMemory(vkDevice, fontMemory, NULL);
        fontMemory = VK_NULL_HANDLE;
    }

    if (fontImage)
    {
        vkDestroyImage(vkDevice, fontImage, NULL);
        fontImage = NULL;
    }

    destroyBuffer(&indexBuffer);
    destroyBuffer(&vertexBuffer);
}
