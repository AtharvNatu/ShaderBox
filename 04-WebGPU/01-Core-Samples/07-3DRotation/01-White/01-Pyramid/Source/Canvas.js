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

let buffer_position_pyramid = null;
let buffer_mvpUniform_pyramid = null;
let bindGroup_mvpUniform_pyramid = null;

let render_pipeline = null;
let perspectiveProjectionMatrix = null;

let depthTexture = null;

var angle = 0.0;
const animationSpeed = 0.5;

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

    buffer_position_pyramid = null;
    buffer_mvpUniform_pyramid = null;
    bindGroup_mvpUniform_pyramid = null;

    render_pipeline = null;
    perspectiveProjectionMatrix = null;
    depthTexture = null;
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

    const vertex_position_pyramid = new Float32Array([  
        // Front
        0.0,   1.0,  0.0,  1.0,
       -1.0,  -1.0,  1.0,  1.0,
        1.0,  -1.0,  1.0,  1.0,

        // Right
        0.0,   1.0,   0.0,  1.0,
        1.0,  -1.0,   1.0,  1.0,
        1.0,  -1.0,  -1.0,  1.0,

        // Back
        0.0,   1.0,   0.0,   1.0,
        1.0,  -1.0,  -1.0,   1.0,
       -1.0,  -1.0,  -1.0,   1.0,

        // Left
        0.0,   1.0,   0.0,   1.0,
       -1.0,  -1.0,  -1.0,   1.0,
       -1.0,  -1.0,   1.0,   1.0
    ]);

    const bindGroupLayout_mvpUniform = createBindGroupLayout(0, GPUShaderStage.VERTEX, "uniform");
    if (bindGroupLayout_mvpUniform == null)
    {
        console.log("Failed To Create Bind Group Layout For MVP Uniform !!!");
        throw Error("Failed To Create Bind Group Layout For MVP Uniform !!!");
    }
    else
        console.log("Bind Group Layout For MVP Uniform Successfully Created");

    //! Pyramid - Position Buffer, Uniform Buffer, Bind Group
    //* ---------------------------------------------------------------------------------------------------------------------------------

    //* Position Buffer
    buffer_position_pyramid = createVertexBuffer(vertex_position_pyramid);
    if (buffer_position_pyramid == null)
    {
        console.log("Failed To Create Vertex Buffer For Pyramid Position !!!");
        throw Error("Failed To Create Vertex Buffer For Pyramid Position !!!");
    }
    else
        console.log("Vertex Buffer For Pyramid Position Successfully Created");

    //* MVP Uniform Buffer
    const mvpUniformBufferSize = 4 * 16;
    buffer_mvpUniform_pyramid = createUniformBuffer(mvpUniformBufferSize);
    if (buffer_mvpUniform_pyramid == null)
    {
        console.log("Failed To Create MVP Uniform Buffer For Pyramid !!!");
        throw Error("Failed To Create MVP Uniform Buffer For Pyramid !!!");
    }
    else
        console.log("MVP Uniform Buffer Successfully Created For Pyramid");
    
    //* Bind Group For MVP Uniform
    bindGroup_mvpUniform_pyramid = createBindGroup(buffer_mvpUniform_pyramid, 0, mvpUniformBufferSize, 0, bindGroupLayout_mvpUniform);
    //* ---------------------------------------------------------------------------------------------------------------------------------

    //* Step - 2: Pipeline Layout for MVP Uniform

    //* Step - 2A: Pipeline Layout Descriptor
    const pipelineLayoutDescriptor = 
    {
        bindGroupLayouts:
        [
            bindGroupLayout_mvpUniform
        ]
    };

    //* Step - 2B: Pipeline Layout
    const pipelineLayout = device.createPipelineLayout(pipelineLayoutDescriptor);
    if (pipelineLayout == null)
    {
        console.log("Failed To Create Pipeline Layout For MVP Uniform !!!");
        throw Error("Failed To Create Pipeline Layout For MVP Uniform !!!");
    }
    else
        console.log("Pipeline Layout For MVP Uniform Successfully Created");

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

    //* Step - 1B: Vertex Shader State
    const vertexShaderState = 
    {
        module: shaderModule_vertexShader,
        entryPoint: "main",
        buffers: 
        [
            positionVertexBufferLayout
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

function createBindGroup(_uniformBuffer, _offset, _uniformBufferSize, _binding, _bindGroupLayout)
{
    // Code

    //* Step - 1A: Buffer Binding Property
    const bufferBinding = 
    {
        buffer: _uniformBuffer,
        offset: _offset,
        size: _uniformBufferSize
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
    
    //* Pyramid
    //* -----------------------------------------------------------------------------------------------------------------------------
    //! Transformations
    var modelViewMatrix = mat4.create();
    var modelViewProjectionMatrix = mat4.create();
    var translationMatrix = mat4.create();
    var rotationMatrix = mat4.create();

    mat4.translate(translationMatrix, translationMatrix, [0.0, 0.0, -6.0]);
    mat4.rotate(rotationMatrix, rotationMatrix, degreeToRadians(angle), [0.0, 1.0, 0.0]);

    mat4.multiply(modelViewMatrix, translationMatrix, rotationMatrix);
    mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

    //! Update Uniform Buffer
    queue.writeBuffer(buffer_mvpUniform_pyramid, 0, modelViewProjectionMatrix, 0, modelViewProjectionMatrix.length);
    //* -----------------------------------------------------------------------------------------------------------------------------
    

    //* Step - 13 : Begin The Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    {
        renderPassEncoder.setPipeline(render_pipeline);
        renderPassEncoder.setViewport(0, 0, canvas.width, canvas.height, 0.0, 1.0);
        renderPassEncoder.setScissorRect(0, 0, canvas.width, canvas.height);

        //* Pyramid
        renderPassEncoder.setVertexBuffer(0, buffer_position_pyramid);
        renderPassEncoder.setBindGroup(0, bindGroup_mvpUniform_pyramid);
        renderPassEncoder.draw(12);
    }
    renderPassEncoder.end();

    //* Step - 14 : Finish The Command Encoder And Submit To Queue
    queue.submit([commandEncoder.finish()]);

    update();

    //! Animation Loop
    animationFrameId = requestAnimationFrame(display);
}

function update()
{
    // Code
    angle += animationSpeed;
    if (angle > 360.0)
        angle = angle - 360.0;
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

        buffer_position_pyramid = null;
        buffer_mvpUniform_pyramid = null;
        bindGroup_mvpUniform_pyramid = null;
        
        render_pipeline = null;
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
            
    }
}

function mousedown()
{
    // Code
}

