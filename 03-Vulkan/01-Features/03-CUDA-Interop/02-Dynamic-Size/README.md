This directory contains shaders used in Orthographic Projection

Changes made for enabling Depth

1) createRenderPass()

 -  //* Step - 3
    VkSubpassDescription vkSubpassDescription;
    memset((void*)&vkSubpassDescription, 0, sizeof(VkSubpassDescription));
    vkSubpassDescription.flags = 0;
    vkSubpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    vkSubpassDescription.inputAttachmentCount = 0;
    vkSubpassDescription.pInputAttachments = NULL;
    vkSubpassDescription.colorAttachmentCount = 1;  //! This should be the count of vkAttachmentReference used for color
    vkSubpassDescription.pColorAttachments = &vkAttachmentReference;
    vkSubpassDescription.pDepthStencilAttachment = NULL;
    vkSubpassDescription.pPreserveAttachments = NULL;
    vkSubpassDescription.pResolveAttachments = NULL;

    Following line is changed to
    vkSubpassDescription.colorAttachmentCount = _ARRAYSIZE(vkAttachmentDescription_array);

    vkSubpassDescription.colorAttachmentCount = 1;  //! This should be the count of vkAttachmentReference used for color


2) createFramebuffers()

 - //* Step - 1
    VkImageView vkImageView_attachments_array[1];
    memset((void*)vkImageView_attachments_array, 0, sizeof(VkImageView) * _ARRAYSIZE(vkImageView_attachments_array));

    //* Step - 2
    VkFramebufferCreateInfo vkFramebufferCreateInfo;
    memset((void*)&vkFramebufferCreateInfo, 0, sizeof(VkFramebufferCreateInfo));
    vkFramebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    vkFramebufferCreateInfo.flags = 0;
    vkFramebufferCreateInfo.pNext = NULL;
    vkFramebufferCreateInfo.attachmentCount = _ARRAYSIZE(vkImageView_attachments_array);
    vkFramebufferCreateInfo.pAttachments = vkImageView_attachments_array;
    vkFramebufferCreateInfo.renderPass = vkRenderPass;
    vkFramebufferCreateInfo.width = vkExtent2D_swapchain.width;
    vkFramebufferCreateInfo.height = vkExtent2D_swapchain.height;
    vkFramebufferCreateInfo.layers = 1;

- Both above steps are moved inside the loop