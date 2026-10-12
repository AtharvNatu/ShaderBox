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

let render_pipeline = null;
let compute_pipeline = null;

let buffer_position_cpu = null;
let buffer_position_gpu = null;

let buffer_hostUniform = null;

let bindGroup_ubo = null;
let bindGroup_compute = null;

let perspectiveProjectionMatrix = null;
let depthTexture = null;

//* Sinewave Related Variables
let MAX_MESH_SIZE = 4096;
let WORKGROUP_SIZE = 32;

let position_data = null;
let meshSize = 256;
let bUseCompute = false;
let fTime = 0.0;

const uniformData =
{
    modelMatrix: mat4.create(),
    viewMatrix: mat4.create(),
    projectionMatrix: mat4.create(),

    color: new Float32Array([1.0, 0.5, 0.0, 1.0]),                      // Orange Color
    computeData: new Float32Array([meshSize, meshSize, fTime, 0.0]),    // Width, Height, Time
};

// MVP Matrices         : 64 + 64 + 64       =  192 +
// Color Vector         : 16                 =  16  +
// Compute Vector       : 16                 =  16  =   224
const uniformBufferSize =
    Float32Array.BYTES_PER_ELEMENT * 16 +   // Model Matrix
    Float32Array.BYTES_PER_ELEMENT * 16 +   // View Matrix
    Float32Array.BYTES_PER_ELEMENT * 16 +   // Projection Matrix
    Float32Array.BYTES_PER_ELEMENT * 4  +   // Color Vector
    Float32Array.BYTES_PER_ELEMENT * 4;     // Compute Vector


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

    const adapterInfo = adapter.info;
    console.log("WebGPU Device Information");
    console.log("---------------------------------------");
    console.log(`Vendor : ${adapterInfo.vendor}`);
    console.log(`Architecture : ${adapterInfo.architecture}`);
    console.log(`Description : ${adapterInfo.description}`);
    console.log(`Max Compute Invocations Per Workgroup : ${adapter.limits.maxComputeInvocationsPerWorkgroup}`);
    console.log(`Max Workgroup Size X/Y/Z : ${adapter.limits.maxComputeWorkgroupSizeX}, ${adapter.limits.maxComputeWorkgroupSizeY}, ${adapter.limits.maxComputeWorkgroupSizeZ}`);
    console.log(`Max Storage Buffer Binding Size : ${adapter.limits.maxStorageBufferBindingSize}`);
    console.log(`Max Buffer Size : ${adapter.limits.maxBufferSize}`);
    console.log("---------------------------------------");

    const _requiredLimits = 
    {
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize
    };

    //* Step - 3 : Get GPU Device Object From Adapter
    try
    {
        device = await adapter.requestDevice({requiredLimits: _requiredLimits});
    }
    catch (error) 
    {
        console.log("GPUDevice requestDevice(requiredLimits) Failed !!! Retrying With Defaults", error);
        WORKGROUP_SIZE = 16;
        MAX_MESH_SIZE = 2048;
        device = await adapter.requestDevice();
    }

    if (device == null)
    {
        console.log("Failed To Get GPU Device From Adapter !!!");
        throw Error("Failed To Get GPU Device From Adapter !!!");
    }
    else
    {
        console.log("GPU Device Successfully Initialized");

        const requiredBufferSize = MAX_MESH_SIZE * MAX_MESH_SIZE * 4 * 4;
        if (requiredBufferSize > device.limits.maxStorageBufferBindingSize || requiredBufferSize > device.limits.maxBufferSize)
        {
            MAX_MESH_SIZE = 2048;
            console.log(`MAX_MESH_SIZE Capped To ${MAX_MESH_SIZE} Due To Device Buffer Limits !`);
        }

        if (device.limits.maxComputeInvocationsPerWorkgroup < (WORKGROUP_SIZE * WORKGROUP_SIZE) ||
            device.limits.maxComputeWorkgroupSizeX < WORKGROUP_SIZE ||
            device.limits.maxComputeWorkgroupSizeY < WORKGROUP_SIZE)
        {
            WORKGROUP_SIZE = 16;
            console.log(`WORKGROUP_SIZE Capped To ${WORKGROUP_SIZE} Due To Device Compute Limits !`);
        }

        console.log(`WORKGROUP_SIZE = ${WORKGROUP_SIZE}, MAX_MESH_SIZE = ${MAX_MESH_SIZE}`);
    }
        
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

    buffer_position_cpu = null;
    buffer_position_gpu = null;

    compute_pipeline = null;
    render_pipeline = null;

    buffer_hostUniform = null;

    bindGroup_ubo = null;
    bindGroup_compute = null;

    position_data = null;

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
    const computeShaderWGSL = await loadShader("../Shaders/Shader.comp.wgsl");

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

    //* Compute Shader Module
    const shaderModuleDescriptor_computeShader = 
    {
        code: computeShaderWGSL
    };

    const shaderModule_computeShader = device.createShaderModule(shaderModuleDescriptor_computeShader);
    if (shaderModule_computeShader == null)
    {
        console.log("Failed To Create Compute Shader Module !!!");
        throw Error("Failed To Create Compute Shader Module !!!");
    }
    else
        console.log("Compute Shader Module Successfully Created");

    const meshBufferSize = MAX_MESH_SIZE * MAX_MESH_SIZE * 4 * 4;

    //* CPU Buffer
    buffer_position_cpu = createBuffer(meshBufferSize, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    if (buffer_position_cpu == null)
    {
        console.log("Failed To Create Vertex Buffer For CPU !!!");
        throw Error("Failed To Create Vertex Buffer For CPU !!!");
    }
    else
        console.log("Vertex Buffer For CPU Successfully Created");

    //* GPU Buffer
    buffer_position_gpu = createBuffer(meshBufferSize, GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST);
    if (buffer_position_gpu == null)
    {
        console.log("Failed To Create Storage | Vertex Buffer For GPU !!!");
        throw Error("Failed To Create Storage | Vertex Buffer For GPU !!!");
    }
    else
        console.log("Storage | Vertex Buffer For GPU Successfully Created");

    //* Bind Group Layouts
    const bindGroupLayout_ubo = createBindGroupLayout(0, GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE, "uniform");
    if (bindGroupLayout_ubo == null)
    {
        console.log("Failed To Create Bind Group Layout For Graphics !!!");
        throw Error("Failed To Create Bind Group Layout For Graphics !!!");
    }
    else
        console.log("Bind Group Layout For Graphics Successfully Created");

    const bindGroupLayout_compute = createBindGroupLayout(0, GPUShaderStage.COMPUTE, "storage");
    if (bindGroupLayout_compute == null)
    {
        console.log("Failed To Create Bind Group Layout For Compute !!!");
        throw Error("Failed To Create Bind Group Layout For Compute !!!");
    }
    else
        console.log("Bind Group Layout For Compute Successfully Created");

    //* Uniform Buffer
    buffer_hostUniform = createUniformBuffer(uniformBufferSize);
    if (buffer_hostUniform == null)
    {
        console.log("Failed To Create Uniform Buffer !!!");
        throw Error("Failed To Create Uniform Buffer !!!");
    }
    else
        console.log("Uniform Buffer Successfully Created");

    //* Bind Group For Graphics
    bindGroup_ubo = createBindGroup(buffer_hostUniform, 0, uniformBufferSize, 0, bindGroupLayout_ubo);
    if (bindGroup_ubo == null)
    {
        console.log("Failed To Create Graphics Bind Group !!!");
        throw Error("Failed To Create Graphics Bind Group !!!");
    }
    else
        console.log("Graphics Bind Group Successfully Created");

    //* Bind Group For Compute
    bindGroup_compute = createBindGroup(buffer_position_gpu, 0, meshBufferSize, 0, bindGroupLayout_compute);
    if (bindGroup_compute == null)
    {
        console.log("Failed To Create Compute Bind Group !!!");
        throw Error("Failed To Create Compute Bind Group !!!");
    }
    else
        console.log("Compute Bind Group Successfully Created");
    //* ---------------------------------------------------------------------------------------------------------------------------------

    //* Step - 2: Pipeline Layout for Graphics

    //* Step - 2A: Pipeline Layout Descriptor
    const pipelineLayoutDescriptor_graphics = 
    {
        bindGroupLayouts:
        [
            bindGroupLayout_ubo
        ]
    };

    //* Step - 2B: Pipeline Layout
    const pipelineLayout_graphics = device.createPipelineLayout(pipelineLayoutDescriptor_graphics);
    if (pipelineLayout_graphics == null)
    {
        console.log("Failed To Create Pipeline Layout For Graphics !!!");
        throw Error("Failed To Create Pipeline Layout For Graphics !!!");
    }
    else
        console.log("Pipeline Layout For Graphics Successfully Created");

    const pipelineLayoutDescriptor_compute = 
    {
        bindGroupLayouts:
        [
            bindGroupLayout_ubo,
            bindGroupLayout_compute
        ]
    };

    //* Step - 2B: Pipeline Layout
    const pipelineLayout_compute = device.createPipelineLayout(pipelineLayoutDescriptor_compute);
    if (pipelineLayout_compute == null)
    {
        console.log("Failed To Create Pipeline Layout For Compute !!!");
        throw Error("Failed To Create Pipeline Layout For Compute !!!");
    }
    else
        console.log("Pipeline Layout For Compute Successfully Created");

    //! Render Pipeline

    //* Step - 1 : Pipeline Descriptor / PSO

    //* Step - 1A: Vertex Buffer Layout
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
        topology: "point-list"
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
        layout: pipelineLayout_graphics,
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

    //! Compute Pipeline
    const computePipelineDescriptor = 
    {
        layout: pipelineLayout_compute,
        compute: 
        {
            module: shaderModule_computeShader,
            entryPoint: "main",
            constants: 
            {
                WORKGROUP_SIZE: WORKGROUP_SIZE
            }
        }
    };

    compute_pipeline = device.createComputePipeline(computePipelineDescriptor);
    if (compute_pipeline == null)
    {
        console.log("Failed To Create Compute Pipeline !!!");
        throw Error("Failed To Create Compute Pipeline !!!");
    }
    else
        console.log("Compute Pipeline Successfully Created");

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

//* Compute Shader uses a different version of createVertexBuffer() compared to other samples
function createBuffer(_dataSize, _usageFlags)
{
    // Code
    const bufferDescriptor = 
    {
        size: _dataSize,
        usage: _usageFlags
    };

    return device.createBuffer(bufferDescriptor);
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

    if (!bUseCompute)
        prepareCpuBuffer(meshSize, meshSize, fTime);
    else
    {
        const computePassEncoder = commandEncoder.beginComputePass();
        {
            computePassEncoder.setPipeline(compute_pipeline);
            computePassEncoder.setBindGroup(0, bindGroup_ubo);
            computePassEncoder.setBindGroup(1, bindGroup_compute);
            const workGroups = Math.ceil(meshSize / WORKGROUP_SIZE);
            computePassEncoder.dispatchWorkgroups(workGroups, workGroups, 1);
        }
        computePassEncoder.end();
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
    let translationMatrix = mat4.create();
    uniformData.modelMatrix = mat4.create();

    mat4.translate(translationMatrix, translationMatrix, [0.0, 0.0, -3.0]);
    uniformData.modelMatrix = translationMatrix;
    uniformData.viewMatrix = mat4.create();
    uniformData.projectionMatrix = perspectiveProjectionMatrix;

    uniformData.computeData[0] = meshSize;
    uniformData.computeData[1] = meshSize;
    uniformData.computeData[2] = fTime;

    //! Update Uniform Buffer

    // MVP Matrices         : 64 + 64 + 64       =  192
    queue.writeBuffer(
        buffer_hostUniform, 
        0, 
        uniformData.modelMatrix
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16, 
        uniformData.viewMatrix
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 + 
        Float32Array.BYTES_PER_ELEMENT * 16, 
        uniformData.projectionMatrix
    );

    // Color Vector        : 16    =  16
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3,
        uniformData.color
    );

    // Compute Vector        : 16    =  16
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 1, 
        uniformData.computeData
    );

    //* Step - 13 : Begin The Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    {
        renderPassEncoder.setPipeline(render_pipeline);
        renderPassEncoder.setViewport(0, 0, canvas.width, canvas.height, 0.0, 1.0);
        renderPassEncoder.setScissorRect(0, 0, canvas.width, canvas.height);
        if (bUseCompute)
            renderPassEncoder.setVertexBuffer(0, buffer_position_gpu);
        else    
            renderPassEncoder.setVertexBuffer(0, buffer_position_cpu);
        renderPassEncoder.setBindGroup(0, bindGroup_ubo);
        renderPassEncoder.draw(meshSize * meshSize);
    }
    renderPassEncoder.end();

    //* Step - 14 : Finish The Command Encoder And Submit To Queue
    queue.submit([commandEncoder.finish()]);

    update();

    //! Animation Loop
    animationFrameId = requestAnimationFrame(display);
}

function prepareCpuBuffer(_width, _height, _fTime)
{
    // Code
    const size = _width * _height;
    if (position_data == null || position_data.length != size * 4)
        position_data = new Float32Array(size * 4);

    const frequency = 5.0;

    //* Mirrors The Compute Shader : u From x, v From y, idx = y * width + x
    for (let y = 0; y < _height; y++)
    {
        for (let x = 0; x < _width; x++)
        {
            let u = x / _width;
            let v = y / _height;

            u = u * 2.0 - 1.0;
            v = v * 2.0 - 1.0;

            const w = Math.cos(u * frequency + _fTime) * Math.sin(v * frequency + _fTime) * 0.5;
            const index = (y * _width + x) * 4;
            
            position_data[index + 0] = u;
            position_data[index + 1] = w;
            position_data[index + 2] = v;
            position_data[index + 3] = 1.0;
        }
    }

    queue.writeBuffer(buffer_position_cpu, 0, position_data);

}

function update()
{
    // Code
    fTime += 0.01;
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
        if (buffer_position_cpu != null)
            buffer_position_cpu.destroy();
        if (buffer_position_gpu != null)
            buffer_position_gpu.destroy();
        if (buffer_hostUniform != null)
            buffer_hostUniform.destroy();

        device.destroy();
        device = null;
        queue = null;
        buffer_position_cpu = null;
        buffer_position_gpu = null;

        compute_pipeline = null;
        render_pipeline = null;

        buffer_hostUniform = null;

        bindGroup_ubo = null;
        bindGroup_compute = null;

        position_data = null;
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

        case 't':
        case 'T':
            bUseCompute = !bUseCompute;
        break;

        case "1":
            meshSize = 256;
        break;

        case "2":
            meshSize = 512;
        break;

        case "3":
            meshSize = 1024;
        break;

        case "4":
            meshSize = 2048;
        break;

        case "5":
            if (MAX_MESH_SIZE == 4096)
                meshSize = MAX_MESH_SIZE;
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

