const { mat4, mat3 } = glMatrix;

//* Global Variables
var canvas = null;
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

//! WebGPU Related
var clearColor;
let device = null;
let context = null;
let queue = null;
let canvasFormat = null;
let animationFrameId = null;

let buffer_position = null;
let buffer_texcoord = null;

let buffer_hostUniform = null;
let bindGroup_hostUniform = null;

let render_pipeline = null;
let perspectiveProjectionMatrix = null;
let depthTexture = null;

let texture_smiley = null;
let sampler_smiley = null;
let bindGroup_texture_sampler = null;

const hostUniformData =
{
    modelMatrix: mat4.create(),
    viewMatrix: mat4.create(),
    projectionMatrix: mat4.create(),
    keyPressedUniform: new Uint32Array([0, 0, 0, 0])
};

// MVP Matrices         : 64 + 64 + 64       =  192 +
// Key Pressed          : 16                 =  16  =   208
const hostUniformBufferSize = new ArrayBuffer(
    Float32Array.BYTES_PER_ELEMENT * 16 +
    Float32Array.BYTES_PER_ELEMENT * 16 +
    Float32Array.BYTES_PER_ELEMENT * 16 +
    Float32Array.BYTES_PER_ELEMENT * 4
).byteLength;

let keyPressed = -1;

//* Animation Related
var requestAnimationFrame = window.requestAnimationFrame;

var cancelAnimationFrame = window.cancelAnimationFrame;

async function main()
{
    // Code

    // Get Canvas
    canvas = document.getElementById("ADN");
    if (!canvas)
        console.log("Failed To Obtain Canvas !!!");
    else 
        console.log("Canvas Obtained");

    // Backup Canvas Dimensions
    canvas_original_width = canvas.width;
    canvas_original_height = canvas.height;

    // Register Event Listeners
    window.addEventListener("keydown", keyDown, false);
    window.addEventListener("click", mousedown, false);
    window.addEventListener("resize", resize, false);

    //* Best practices for WebGPU during full screen mode
    document.addEventListener("fullscreenchange", onFullScreenChange);
    document.addEventListener("webkitfullscreenchange", onFullScreenChange);

    //! Initialize WebGPU

    //* Step - 1 : Get GPU Interface
    const gpu = navigator.gpu;
    if (gpu == null)
    {
        console.log("WebGPU Is Not Supported By This Browser !!!");
        throw Error("WebGPU Is Not Supported By This Browser !!!");
    }
    else
        console.log("WebGPU Supported");

    //* Step - 2 : Get GPUAdapter Object From GPU Interface
    const adapter = await gpu.requestAdapter();
    if (adapter == null)
    {
        console.log("Failed to get GPUAdapter object !!!");
        throw Error("Failed to get GPUAdapter object !!!");
    }
    else
        console.log("GPUAdapter Instance Successfully Acquired");

    //* Step - 3 : Get GPU Device Object From Adapter
    device = await adapter.requestDevice();
    if (device == null)
    {
        console.log("Failed To Get GPU Device From Adapter !!!");
        throw Error("Failed To Get GPU Device From Adapter !!!");
    }
    else
        console.log("GPU Device successfully initialized");

    const adapterInfo = adapter.info;
    console.log("WebGPU Device Information");
    console.log("---------------------------------------");
    console.log(`Vendor : ${adapterInfo.vendor}`);
    console.log(`Architecture : ${adapterInfo.architecture}`);
    console.log(`Description : ${adapterInfo.description}`);
    console.log("---------------------------------------");

    //* Step - 4 : Listener To Handle Lost Devices
    device.addEventListener("uncapturederror", onUncapturedError);
    device.lost.then(onDeviceLost);

    // Initialize
    await initialize();

    // Warmup Resize
    resize();

    // Display
    display();

}

function onUncapturedError(event)
{
    // Code
    console.error("WebGPU Uncaptured Error : ", event.error.message);
}

function onDeviceLost(info)
{
    // Code
    console.warn("WebGPU Device Lost : ", info.reason, ", Message : ", info.message);

    if (animationFrameId != null)
    {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    device = null;
    queue = null;

    buffer_position = null;
    buffer_texcoord = null;

    buffer_hostUniform = null;
    bindGroup_hostUniform = null;

    render_pipeline = null;
    
    perspectiveProjectionMatrix = null;
    
    depthTexture = null;

    texture_smiley = null;
    sampler_smiley = null;
    bindGroup_texture_sampler = null;
}

function toggleFullScreen()
{
    // Code
    var fullscreen_element = document.fullscreenElement ||       // Generic
                             document.mozFullScreenElement ||    // Mozilla
                             document.webkitFullscreenElement || // Apple
                             document.msFullscreenElement ||     // Edge
                             null;                               
    
    if (fullscreen_element == null)
    {
        if (canvas.requestFullscreen)
            canvas.requestFullscreen();
        else if (canvas.mozRequestFullScreen)
            canvas.mozRequestFullScreen();
        else if (canvas.webkitRequestFullscreen)
            canvas.webkitRequestFullscreen();
        else if (canvas.msRequestFullscreen)
            canvas.msRequestFullscreen();
    }
    else
    {
        if (document.exitFullscreen)
            document.exitFullscreen();
        else if (document.mozExitFullscreen)
            document.mozExitFullscreen();
        else if (document.webkitExitFullscreen)
            document.webkitExitFullscreen();
        else if (document.msExitFullscreen)  
            document.msExitFullscreen();
    }                    
}

function onFullScreenChange()
{
    // Code
    var fullscreen_element = document.fullscreenElement ||       // Generic
                             document.mozFullScreenElement ||    // Mozilla
                             document.webkitFullscreenElement || // Apple
                             document.msFullscreenElement ||     // Edge
                             null;                               
    
    if (fullscreen_element == null)
        bFullscreen = false;
    else
        bFullscreen = true;

    resize();
}

async function initialize()
{
    // Code

    //* Step - 5 : Get GPUQueue Object (Synchronous)
    queue = device.queue;
    console.log("GPUQueue Acquired Successfully");

    //* Step - 6 : Get WebGPU Context
    context = canvas.getContext("webgpu");
    if (context == null)
    {
        console.log("Failed To Get WebGPU Context From Canvas !!!");
        throw Error("Failed To Get WebGPU Context From Canvas !!!");
    }
    else
        console.log("WebGPU Context Successfully Initialized");

    //* Step - 7 : Get Preferred WebGPU Color Format For Canvas
    canvasFormat = navigator.gpu.getPreferredCanvasFormat();

    //* Step - 8 : Configure Canvas Using The Format
    const canvasConfiguration = 
    {
        device: device,
        format: canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        alphaMode: "opaque"
    };

    context.configure(canvasConfiguration);

    //* Step - 9 : Define the clear color
    clearColor = 
    { 
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0
    };

    //* Load Shaders From File
    const vertexShaderWGSL = await loadShader("../Shaders/Shader.vert.wgsl");
    const fragmentShaderWGSL = await loadShader("../Shaders/Shader.frag.wgsl");

    //* Vertex Shader Module
    const shaderModuleDescriptor_vertexShader = 
    {
        code: vertexShaderWGSL
    };

    const shaderModule_vertexShader = device.createShaderModule(shaderModuleDescriptor_vertexShader);
    if (shaderModule_vertexShader == null)
    {
        console.log("Failed To Create Vertex Shader Module !!!");
        throw Error("Failed To Create Vertex Shader Module !!!");
    }
    else
        console.log("Vertex Shader Module Successfully Created");

    //* Fragment Shader Module
    const shaderModuleDescriptor_fragmentShader = 
    {
        code: fragmentShaderWGSL
    };

    const shaderModule_fragmentShader = device.createShaderModule(shaderModuleDescriptor_fragmentShader);
    if (shaderModule_fragmentShader == null)
    {
        console.log("Failed To Create Fragment Shader Module !!!");
        throw Error("Failed To Create Fragment Shader Module !!!");
    }
    else
        console.log("Fragment Shader Module Successfully Created");

    const vertex_position_rectangle = new Float32Array([
        1.0,    1.0,    0.0,    1.0,
        -1.0,   1.0,    0.0,    1.0,
        -1.0,   -1.0,   0.0,    1.0,

        -1.0,   -1.0,   0.0,    1.0,
        1.0,    -1.0,   0.0,    1.0,
        1.0,    1.0,    0.0,    1.0
    ]);

    const vertex_texcoords_rectangle = new Float32Array([
        // Triangle 1
        1.0,  0.0,             // Top Right
        0.0,  0.0,             // Top Left
        0.0,  1.0,             // Bottom Left

        // Triangle 2
        0.0,  1.0,             // Bottom Left
        1.0,  1.0,             // Bottom Right
        1.0,  0.0              // Top Right
    ]);

    //! Rectangle - Position Buffer, Texcoord Buffer, Uniform Buffer, Bind Group
    //* ---------------------------------------------------------------------------------------------------------------------------------
    buffer_position = createVertexBuffer(vertex_position_rectangle);
    if (buffer_position == null)
    {
        console.log("Failed To Create Vertex Buffer For Position !!!");
        throw Error("Failed To Create Vertex Buffer For Position !!!");
    }
    else
        console.log("Vertex Buffer For Position Successfully Created");

    buffer_texcoord = createVertexBuffer(vertex_texcoords_rectangle);
    if (buffer_texcoord == null)
    {
        console.log("Failed To Create Vertex Buffer For Texcoords !!!");
        throw Error("Failed To Create Vertex Buffer For Texcoords !!!");
    }
    else
        console.log("Vertex Buffer For Texcoords Successfully Created");

    //* Host Uniform Buffer
    buffer_hostUniform = createUniformBuffer(hostUniformBufferSize);
    if (buffer_hostUniform == null)
    {
        console.log("Failed To Create Host Uniform Buffer !!!");
        throw Error("Failed To Create Host Uniform Buffer !!!");
    }
    else
        console.log("Host Uniform Buffer Successfully Created");

    const bindGroupLayout_hostUniform = createBindGroupLayout(0, GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, "uniform");
    if (bindGroupLayout_hostUniform == null)
    {
        console.log("Failed To Create Bind Group Layout For Host Uniform !!!");
        throw Error("Failed To Create Bind Group Layout For Host Uniform !!!");
    }
    else
        console.log("Bind Group Layout For Host Uniform Successfully Created");
    
    //* Bind Group For Host Uniform
    bindGroup_hostUniform = createBindGroup(buffer_hostUniform, 0, hostUniformBufferSize, 0, bindGroupLayout_hostUniform);
    if (bindGroup_hostUniform == null)
    {
        console.log("Failed To Create Bind Group For Host Uniform !!!");
        throw Error("Failed To Create Bind Group For Host Uniform !!!");
    }
    else
        console.log("Bind Group For Host Uniform Successfully Created");
    //* ---------------------------------------------------------------------------------------------------------------------------------

    //! Load Texture
    //! ------------------------------------------------------------------------------------------------------------------
    texture_smiley = await loadTexture("../Assets/Smiley.png");
    if (texture_smiley == null)
    {
        console.log("Failed To Load Smiley Texture !!!");
        throw Error("Failed To Load Smiley Texture !!!");
    }
    else
        console.log("Smiley Texture Loaded");

    //* Texture Sampler Descriptor
    const samplerDescriptor = 
    {
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat"
    };

    //* Create Texture Sampler
    sampler_smiley = device.createSampler(samplerDescriptor);
    if (sampler_smiley == null)
    {
        console.log("Failed To Create Sampler !!!");
        throw Error("Failed To Create Sampler !!!");
    }
    else
        console.log("Sampler Created Successfully");

    //* Texture-Sampler Bind Group Layout
    const bindGroupLayout_texture_sampler = createBindGroupLayout_texture_sampler(
        "float", 
        "2d",
        false,
        0,
        GPUShaderStage.FRAGMENT,
        "filtering",
        1,
        GPUShaderStage.FRAGMENT
    );
    if (bindGroupLayout_texture_sampler == null)
    {
        console.log("Failed To Create Bind Group Layout For Texture and Sampler !!!");
        throw Error("Failed To Create Bind Group Layout For Texture and Sampler !!!");
    }
    else
        console.log("Bind Group Layout Successfully Created For Texture and Sampler");


    //* Texture-Sampler Bind Group
    bindGroup_texture_sampler = createBindGroup_texture_sampler(0, texture_smiley, 1, sampler_smiley, bindGroupLayout_texture_sampler);
    if (bindGroup_texture_sampler == null)
    {
        console.log("Failed To Create Bind Group For Texture and Sampler !!!");
        throw Error("Failed To Create Bind Group For Texture and Sampler !!!");
    }
    else
        console.log("Bind Group Successfully Created For Texture and Sampler");
    //! ------------------------------------------------------------------------------------------------------------------

    //* Step - 2: Pipeline Layout for Host Uniform

    //* Step - 2A: Pipeline Layout Descriptor
    const pipelineLayoutDescriptor = 
    {
        bindGroupLayouts:
        [
            bindGroupLayout_hostUniform,
            bindGroupLayout_texture_sampler
        ]
    };

    //* Step - 2B: Pipeline Layout
    const pipelineLayout = device.createPipelineLayout(pipelineLayoutDescriptor);
    if (pipelineLayout == null)
    {
        console.log("Failed To Create Pipeline Layout For Host Uniform !!!");
        throw Error("Failed To Create Pipeline Layout For Host Uniform !!!");
    }
    else
        console.log("Pipeline Layout For Host Uniform Successfully Created");

    //! Render Pipeline

    //* Step - 1 : Pipeline Descriptor / PSO

    //* Step - 1A: Vertex Buffer Layout

    //! Position Attribute
    const positionVertexAttribute = 
    {
        shaderLocation: 0,  //* Maps to location(0) in Vertex Shader
        offset: 0,
        format: "float32x4"
    };

    const positionVertexBufferLayout = 
    {
        attributes: 
        [
            positionVertexAttribute
        ],
        arrayStride: 4 * 4,
        stepMode: "vertex"  // Jump vertex by vertex, not instance by instance
    };

    //! Texcoord Attribute
    const texcoordVertexAttribute = 
    {
        shaderLocation: 1,  //* Maps to location(1) in Vertex Shader
        offset: 0,
        format: "float32x2"
    };

    const texcoordVertexBufferLayout = 
    {
        attributes: 
        [
            texcoordVertexAttribute
        ],
        arrayStride: 4 * 2,
        stepMode: "vertex"  // Jump vertex by vertex, not instance by instance
    };

    //* Step - 1B: Vertex Shader State
    const vertexShaderState = 
    {
        module: shaderModule_vertexShader,
        entryPoint: "main",
        buffers: 
        [
            positionVertexBufferLayout,
            texcoordVertexBufferLayout
        ]
    };

    //* Step - 1C: Fragment Shader State
    const colorTargetState = 
    {
        format: canvasFormat
    };

    const fragmentShaderState = 
    {
        module: shaderModule_fragmentShader,
        entryPoint: "main",
        targets:
        [
            colorTargetState
        ]
    };

    //* Step - 1D: Primitive State
    const primitiveState = 
    {
        frontFace: "ccw",    // Counter-Clockwise
        cullMode: "none",    // No Culling
        topology: "triangle-list"
    };

    //* Depth-Stencil State
    const depthStencilState = 
    {
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        format: "depth24plus-stencil8"
    };

    //* Step - 1E: PSO
    const pipelineDescriptor = 
    {
        layout: pipelineLayout,
        vertex: vertexShaderState,
        fragment: fragmentShaderState,
        primitive: primitiveState,
        depthStencil: depthStencilState
    };

    render_pipeline = device.createRenderPipeline(pipelineDescriptor);
    if (render_pipeline == null)
    {
        console.log("Failed To Create Render Pipeline !!!");
        throw Error("Failed To Create Render Pipeline !!!");
    }
    else
        console.log("Render Pipeline Successfully Created");

    //* Initialize Projection Matrix
    perspectiveProjectionMatrix = mat4.create();
}

async function loadShader(path)
{
    // Code
    const response = await fetch(path);
    if (!response.ok)
       throw Error(`Failed To Load Shader ${path} : ${response.status}`);

    return await response.text();
}

function createVertexBuffer(_data)
{
    // Code
    const bufferDescriptor = 
    {
        size: _data.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    };

    const buffer = device.createBuffer(bufferDescriptor);
    if (buffer == null)
        return null;

    queue.writeBuffer(buffer, 0, _data);

    return buffer;
}

function createUniformBuffer(_uniformBufferSize)
{
    // Code
    const bufferDescriptor = 
    {
        size: _uniformBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    };

    return device.createBuffer(bufferDescriptor);
}

function createBindGroupLayout(_bindingIndex, _shaderStageVisibility, _bufferType)
{
    // Code

    //* Step - 1A: Bind Group Layout Entry
    const bindGroupLayoutEntry = 
    {
        binding: _bindingIndex,
        visibility: _shaderStageVisibility,
        buffer: 
        {
            type: _bufferType
        }
    };

    //* Step - 1B: Bind Group Layout Descriptor
    const bindGroupLayoutDescriptor = 
    {
        entries: 
        [
            bindGroupLayoutEntry
        ]
    };

    //* Step - 1C: Bind Group Layout
    return device.createBindGroupLayout(bindGroupLayoutDescriptor);
}

function createBindGroup(_buffer, _offset, _bufferSize, _binding, _bindGroupLayout)
{
    // Code

    //* Step - 1A: Buffer Binding Property
    const bufferBinding = 
    {
        buffer: _buffer,
        offset: _offset,
        size: _bufferSize
    };

    //* Step - 1B: Buffer Binding Entry
    const bindGroupEntry = 
    {
        binding: _binding,
        resource: bufferBinding
    };

    //* Step - 1C: Buffer Binding Descriptor
    const bindGroupDescriptor = 
    {
        layout: _bindGroupLayout,
        entries:
        [
            bindGroupEntry
        ]
    };

    //* Step - 1D: Bind Group
    return device.createBindGroup(bindGroupDescriptor);
}

async function loadTexture(_texturePath)
{
    // Code
    const image = new Image();
    image.src = _texturePath;
    await image.decode();

    const imageBitmap = await createImageBitmap(image);
    if (imageBitmap == null)
    {
        console.log("Failed To Create Image Bitmap !!!");
        throw Error("Failed To Create Image Bitmap !!!");
    }
    else
        console.log("Image Bitmap Created Successfully");
    
    let textureDescriptor = 
    {
        size: [imageBitmap.width, imageBitmap.height, 1],
        dimension: "2d",
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
    };

    let _texture = device.createTexture(textureDescriptor);
    if (_texture == null)
    {
        console.log("Failed To Create Texture !!!");
        throw Error("Failed To Create Texture !!!");
    }
    else
        console.log("Texture Created Successfully");

    queue.copyExternalImageToTexture(
        { source: imageBitmap } ,
        { texture: _texture },
        textureDescriptor.size
    )

    return _texture;
}

function createBindGroupLayout_texture_sampler(
    _textureSampleType, 
    _textureDimension, 
    _isTextureMultiSampled, 
    _textureBindingIndex, 
    _textureShaderStageVisibility, 
    _samplerType,
    _samplerBindingIndex,
    _samplerShaderStageVisibility 
)
{
    // Code

    //* Create Binding Layout For Texture
    const bindingLayout_texture = 
    {
        sampleType: _textureSampleType,
        viewDimension: _textureDimension,
        multisampled: _isTextureMultiSampled
    };

    //* Create Bind Group Layout Entry For Texture
    const bindGroupLayoutEntry_texture = 
    {
        binding: _textureBindingIndex,
        visibility: _textureShaderStageVisibility,
        texture: bindingLayout_texture
    };
    
    //* Create Binding Layout For Sampler
    const bindingLayout_sampler = 
    {
        type: _samplerType
    };

    //* Create Bind Group Layout Entry For Sampler
    const bindGroupLayoutEntry_sampler = 
    {
        binding: _samplerBindingIndex,
        visibility: _samplerShaderStageVisibility,
        sampler: bindingLayout_sampler
    };

    //* Bind Group Layout Descriptor
    const bindGroupLayoutDescriptor = 
    {
        entries: 
        [
            bindGroupLayoutEntry_texture,
            bindGroupLayoutEntry_sampler
        ]
    };

    //* Create Bind Group Layout
    return device.createBindGroupLayout(bindGroupLayoutDescriptor);
}

function createBindGroup_texture_sampler(_textureBindingIndex, _textureId, _samplerBindingIndex, _samplerId, _bindGroupLayout)
{
    // Code

    //* Buffer Binding Entry For Texture
    const bindGroupEntry_texture = 
    {
        binding: _textureBindingIndex,
        resource: _textureId.createView()
    };

    //* Buffer Binding Entry For Sampler
    const bindGroupEntry_sampler = 
    {
        binding: _samplerBindingIndex,
        resource: _samplerId
    };

    //* Buffer Binding Descriptor
    const bindGroupDescriptor = 
    {
        layout: _bindGroupLayout,
        entries:
        [
            bindGroupEntry_texture,
            bindGroupEntry_sampler
        ]
    };

    //* Create Bind Group
    return device.createBindGroup(bindGroupDescriptor);
}

function updateTexcoords()
{
    // Code
    if (queue == null || buffer_texcoord == null)
        return;

    const texcoords = new Float32Array(12);

    switch (keyPressed)
    {
        case 1:
            texcoords.set([
                0.5, 0.5,
                0.0, 0.5,
                0.0, 1.0,
                0.0, 1.0,
                0.5, 1.0,
                0.5, 0.5
            ]);
            hostUniformData.keyPressedUniform[0] = 1;
        break;

        case 2:
            texcoords.set([
                1.0, 0.0,
                0.0, 0.0,
                0.0, 1.0,
                0.0, 1.0,
                1.0, 1.0,
                1.0, 0.0
            ]);
            hostUniformData.keyPressedUniform[0] = 1;
        break;

        case 3:
            texcoords.set([
                2.0, 0.0,
                0.0, 0.0,
                0.0, 2.0,
                0.0, 2.0,
                2.0, 2.0,
                2.0, 0.0
            ]);
            hostUniformData.keyPressedUniform[0] = 1;
        break;

        case 4:
            texcoords.fill(0.5);
            hostUniformData.keyPressedUniform[0] = 1;
        break;

        default:
            hostUniformData.keyPressedUniform[0] = 0;
        break;
    }

    queue.writeBuffer(buffer_texcoord, 0, texcoords);
}

function resize()
{
    // Code
    if (bFullscreen)
    {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    else
    {
        canvas.width = canvas_original_width;
        canvas.height = canvas_original_height;
    }

    //* Depth Texture
    if (device != null)
    {
        if (depthTexture != null)
        {
            depthTexture.destroy();
            depthTexture = null; 
        }

        // Depth Texture Descriptor
        const depthTextureDescriptor = 
        {
            size: [canvas.width, canvas.height, 1],
            dimension: "2d",
            format: "depth24plus-stencil8",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
        };

        // Create Depth Texture
        depthTexture = device.createTexture(depthTextureDescriptor);
        if (depthTexture == null)
        {
            console.log("Failed To Create Depth Texture !!!");
            throw Error("Failed To Create Depth Texture !!!");
        }
    }

    if (perspectiveProjectionMatrix != null)
        mat4.perspectiveZO(perspectiveProjectionMatrix, degreeToRadians(45.0), parseFloat(canvas.width) / parseFloat(canvas.height), 0.1, 100.0);
}

function display()
{
    // Code
    if (device == null)
        return;

    //* Step - 10 : Get Command Encoder Of The Device (Created per frame due to async nature)
    const commandEncoder = device.createCommandEncoder();
    if (commandEncoder == null)
    {
        console.log("Failed To Get Command Encoder From Device !!!");
        throw Error("Failed To Get Command Encoder From Device !!!");
    }

    //* Step - 11 : Create Renderpass Color Attachment
    const renderPassColorAttachment = 
    {
        view: context.getCurrentTexture().createView(),
        clearValue: clearColor,
        loadOp: "clear",
        storeOp: "store"
    };

    //* Create Renderpass Depth Attachment
    const renderPassDepthAttachment = 
    {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
        stencilClearValue: 0,
        stencilLoadOp: "clear",
        stencilStoreOp: "store"
    };

    //* Step - 12 : Create Renderpass Descriptor (GPURenderPassDescriptor)
    const renderPassDescriptor = 
    {
        colorAttachments: [renderPassColorAttachment],
        depthStencilAttachment:  renderPassDepthAttachment
    };
    
    //! Transformations
    hostUniformData.modelMatrix = mat4.create();
    hostUniformData.modelMatrix = mat4.translate(hostUniformData.modelMatrix, hostUniformData.modelMatrix, [0.0, 0.0, -3.0]);
    hostUniformData.viewMatrix = mat4.create();
    hostUniformData.projectionMatrix = perspectiveProjectionMatrix;

    //! Update Uniform Buffer

    // MVP Matrices         : 64 + 64 + 64       =  192
    queue.writeBuffer(
        buffer_hostUniform, 
        0, 
        hostUniformData.modelMatrix
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16, 
        hostUniformData.viewMatrix
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 + 
        Float32Array.BYTES_PER_ELEMENT * 16, 
        hostUniformData.projectionMatrix
    );

    // Key Pressed : 192 + 16 = 208
    queue.writeBuffer(buffer_hostUniform, Float32Array.BYTES_PER_ELEMENT * 16 * 3, hostUniformData.keyPressedUniform);

    //* Step - 13 : Begin The Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    {
        renderPassEncoder.setPipeline(render_pipeline);
        renderPassEncoder.setViewport(0, 0, canvas.width, canvas.height, 0.0, 1.0);
        renderPassEncoder.setScissorRect(0, 0, canvas.width, canvas.height);
        renderPassEncoder.setVertexBuffer(0, buffer_position);
        renderPassEncoder.setVertexBuffer(1, buffer_texcoord);
        renderPassEncoder.setBindGroup(0, bindGroup_hostUniform);
        renderPassEncoder.setBindGroup(1, bindGroup_texture_sampler);
        renderPassEncoder.draw(6);
    }
    renderPassEncoder.end();

    //* Step - 14 : Finish The Command Encoder And Submit To Queue
    queue.submit([commandEncoder.finish()]);

    //! Animation Loop
    animationFrameId = requestAnimationFrame(display);
}

function update()
{
    // Code
}

function degreeToRadians(degrees)
{
    return (degrees * (Math.PI / 180.0));
}

function uninitialize()
{
    // Code
    if (animationFrameId != null)
    {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    //* Unregister handlers
    window.removeEventListener("keydown", keyDown, false);
    window.removeEventListener("click", mousedown, false);
    window.removeEventListener("resize", resize, false);
    document.removeEventListener("fullscreenchange", onFullScreenChange);
    document.removeEventListener("webkitfullscreenchange", onFullScreenChange);

    if (depthTexture)
    {
        depthTexture.destroy();
        depthTexture = null;
    }

    if (texture_smiley)
    {
        texture_smiley.destroy();
        texture_smiley = null;
    }

    //* Unconfigure/Destroy Context
    if (context != null)
    {
        context.unconfigure();
        context = null;
    }

    //* Step - 15 : Destroy The Device
    if (device != null)
    {
        device.destroy();
        device = null;
        queue = null;
        buffer_position = null;
        buffer_texcoord = null;
        render_pipeline = null;
        buffer_hostUniform = null;
        bindGroup_hostUniform = null;
        sampler_smiley = null;
        bindGroup_texture_sampler = null;
    }

    perspectiveProjectionMatrix = null;
}

function keyDown(event)
{
    // Code
    switch(event.key)
    {
        case 'f':
        case 'F':
            toggleFullScreen();
        break;

        case 'q':
        case 'Q':
            uninitialize();
            window.close(); // Not Applicable For All Browsers
        break;

        case "1":
            keyPressed = 1;
            updateTexcoords();
        break;

        case "2":
            keyPressed = 2;
            updateTexcoords();
        break;

        case "3":
            keyPressed = 3;
            updateTexcoords();
        break;

        case "4":
            keyPressed = 4;
            updateTexcoords();
        break;

        default:
            keyPressed = 0;
            updateTexcoords();
        break;
            
    }
}

function mousedown()
{
    // Code
}

