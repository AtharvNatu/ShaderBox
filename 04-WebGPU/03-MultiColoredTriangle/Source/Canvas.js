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
let buffer_color = null;
let render_pipeline = null;
let buffer_mvpUniform = null;
let bindGroup_mvpUniform = null;
let perspectiveProjectionMatrix = null;

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
    buffer_color = null;
    render_pipeline = null;
    buffer_mvpUniform = null;
    bindGroup_mvpUniform = null;
    perspectiveProjectionMatrix = null;
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

    const vertex_position = new Float32Array([
        0.0,    1.0,    0.0,    1.0,
        -1.0,   -1.0,   0.0,    1.0,
        1.0,    -1.0,   0.0,    1.0
    ]);

    const vertex_color = new Float32Array([
        1.0,    0.0,    0.0,    1.0,
        0.0,    1.0,    0.0,    1.0,
        0.0,    0.0,    1.0,    1.0
    ]);

    //* Vertex Buffer For Position
    const bufferDescriptor_position = 
    {
        size: vertex_position.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    };

    buffer_position = device.createBuffer(bufferDescriptor_position);
    if (buffer_position == null)
    {
        console.log("Failed To Create Vertex Buffer For Position !!!");
        throw Error("Failed To Create Vertex Buffer For Position !!!");
    }
    else
        console.log("Vertex Buffer For Position Successfully Created");

    queue.writeBuffer(buffer_position, 0, vertex_position, 0, vertex_position.length);
    console.log("Position Data Written To Vertex Buffer");

    //* Vertex Buffer For Color
    const bufferDescriptor_color = 
    {
        size: vertex_color.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    };

    buffer_color = device.createBuffer(bufferDescriptor_color);
    if (buffer_color == null)
    {
        console.log("Failed To Create Vertex Buffer For Color !!!");
        throw Error("Failed To Create Vertex Buffer For Color !!!");
    }
    else
        console.log("Vertex Buffer For Color Successfully Created");

    queue.writeBuffer(buffer_color, 0, vertex_color, 0, vertex_color.length);
    console.log("Color Data Written To Color Buffer");

    //! Uniform Plumbing

    //* Step - 1: Bind Group Layout for MVP Uniform

    //* Step - 1A: Bind Group Layout Entry
    const bindGroupLayoutEntry_mvpUniform = 
    {
        binding: 0, // Maps to 0th group's 0th slot binding in Vertex Shader
        visibility: GPUShaderStage.VERTEX,
        buffer: 
        {
            type: "uniform"
        }
    };

    //* Step - 1B: Bind Group Layout Descriptor
    const bindGroupLayoutDescriptor = 
    {
        entries: 
        [
            bindGroupLayoutEntry_mvpUniform
        ]
    };

    //* Step - 1C: Bind Group Layout
    const bindGroupLayout_mvpUniform = device.createBindGroupLayout(bindGroupLayoutDescriptor);
    if (bindGroupLayout_mvpUniform == null)
    {
        console.log("Failed To Create Bind Group Layout For MVP Uniform !!!");
        throw Error("Failed To Create Bind Group Layout For MVP Uniform !!!");
    }
    else
        console.log("Bind Group Layout For MVP Uniform Successfully Created");

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

    //* Step - 3: Uniform Buffer for MVP Uniform

    //* Step - 3A: Buffer Descriptor
    const mvpUniformSize = 4 * 16;

    const bufferDescriptor_mvpUniform = 
    {
        size: mvpUniformSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    };

    buffer_mvpUniform = device.createBuffer(bufferDescriptor_mvpUniform);
    if (buffer_mvpUniform == null)
    {
        console.log("Failed To Create Uniform Buffer For MVPUniform !!!");
        throw Error("Failed To Create Uniform Buffer For MVPUniform !!!");
    }
    else
        console.log("Uniform Buffer For MVPUniform Successfully Created");

    //* Step - 4: Bind Group For Uniform Buffer

    //* Step - 4A: Buffer Binding Property
    const bufferBinding_mvpUniform = 
    {
        buffer: buffer_mvpUniform,
        offset: 0,
        size: mvpUniformSize
    };

    //* Step - 4B: Buffer Binding Entry
    const bindGroupEntry_mvpUniform = 
    {
        binding: 0,
        resource: bufferBinding_mvpUniform
    };

    //* Step - 4C: Buffer Binding Descriptor
    const bindGroupDescriptor_mvpUniform = 
    {
        layout: bindGroupLayout_mvpUniform,
        entries:
        [
            bindGroupEntry_mvpUniform
        ]
    };

    //* Step - 4D: Bind Group
    bindGroup_mvpUniform = device.createBindGroup(bindGroupDescriptor_mvpUniform);
    if (bindGroup_mvpUniform == null)
    {
        console.log("Failed To Create Bind Group For MVP Uniform !!!");
        throw Error("Failed To Create Bind Group For MVP Uniform !!!");
    }
    else
        console.log("Bind Group For MVP Uniform Successfully Created");


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

    //! Color Attribute
    const colorVertexAttribute = 
    {
        shaderLocation: 1,  //* Maps to location(1) in Vertex Shader
        offset: 0,
        format: "float32x4"
    };

    const colorVertexBufferLayout = 
    {
        attributes: 
        [
            colorVertexAttribute
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
            positionVertexBufferLayout,
            colorVertexBufferLayout
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

    //* Step - 1E: PSO
    const pipelineDescriptor = 
    {
        layout: pipelineLayout,
        vertex: vertexShaderState,
        fragment: fragmentShaderState,
        primitive: primitiveState
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

    //* Step - 12 : Create Renderpass Descriptor (GPURenderPassDescriptor)
    const renderPassDescriptor = 
    {
        colorAttachments: [renderPassColorAttachment]
    };
    
    //! Transformations
    var modelViewMatrix = mat4.create();
    var modelViewProjectionMatrix = mat4.create();

    mat4.translate(modelViewMatrix, modelViewMatrix, [0.0, 0.0, -3.0]);
    mat4.multiply(modelViewProjectionMatrix, perspectiveProjectionMatrix, modelViewMatrix);

    //! Update Uniform Buffer
    queue.writeBuffer(buffer_mvpUniform, 0, modelViewProjectionMatrix, 0, modelViewProjectionMatrix.length);

    //* Step - 13 : Begin The Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    {
        renderPassEncoder.setPipeline(render_pipeline);
        renderPassEncoder.setViewport(0, 0, canvas.width, canvas.height, 0.0, 1.0);
        renderPassEncoder.setScissorRect(0, 0, canvas.width, canvas.height);
        renderPassEncoder.setVertexBuffer(0, buffer_position);
        renderPassEncoder.setVertexBuffer(1, buffer_color);
        renderPassEncoder.setBindGroup(0, bindGroup_mvpUniform);
        renderPassEncoder.draw(3);
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
        buffer_color = null;
        render_pipeline = null;
        buffer_mvpUniform = null;
        bindGroup_mvpUniform = null;
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

