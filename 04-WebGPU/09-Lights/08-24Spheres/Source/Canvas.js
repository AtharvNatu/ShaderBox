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

let buffer_hostUniform = null;
let bindGroup_hostUniform = null;

let buffer_material = null;
let bindGroup_material = null;

let perspectiveProjectionMatrix = null;
let depthTexture = null;

let sphere = null;
let numMeshIndices = 0;

let buffer_position = null;
let buffer_normal = null;
let buffer_texcoords = null;
let buffer_elements = null;

const hostUniformData =
{
    modelMatrix: mat4.create(),
    viewMatrix: mat4.create(),
    projectionMatrix: mat4.create(),

    lightAmbient: new Float32Array([0.0, 0.0, 0.0, 1.0]),
    lightDiffuse: new Float32Array([1.0, 1.0, 1.0, 1.0]),
    lightSpecular: new Float32Array([1.0, 1.0, 1.0, 1.0]),
    lightPosition: new Float32Array([100.0, 100.0, 100.0, 1.0]),
    lightEnabled: new Uint32Array([0, 0, 0, 0])
};

// MVP Matrices         : 64 + 64 + 64       =  192 +
// Light Vectors        : 16 + 16 + 16 + 16  =  64  +
// Light Enabled Vector : 16                 =  16  =   272
const hostUniformBufferSize = new ArrayBuffer(
    Float32Array.BYTES_PER_ELEMENT * 16 +   // Model Matrix
    Float32Array.BYTES_PER_ELEMENT * 16 +   // View Matrix
    Float32Array.BYTES_PER_ELEMENT * 16 +   // Projection Matrix
    Float32Array.BYTES_PER_ELEMENT * 4  +   // Light Ambient
    Float32Array.BYTES_PER_ELEMENT * 4  +   // Light Diffuse
    Float32Array.BYTES_PER_ELEMENT * 4  +   // Light Specular
    Float32Array.BYTES_PER_ELEMENT * 4  +   // Light Position
    Float32Array.BYTES_PER_ELEMENT * 4      // Light Enabled
).byteLength;

var bLight = false;
const radius = 40.0;
var chosenAxis = '';
var angleLight = 0.0;

var materialTable =
[
    // row 1
    { name: "emerald",        ambient: [0.0215,    0.1745,    0.0215,    1.0], diffuse: [0.07568,    0.61424,    0.07568,    1.0], specular: [0.633,    0.727811, 0.633,    1.0], shininess: 0.6        * 128.0 },
    { name: "brass",          ambient: [0.329412,  0.223529,  0.027451,  1.0], diffuse: [0.780392,   0.568627,   0.113725,   1.0], specular: [0.992157, 0.941176, 0.807843, 1.0], shininess: 0.21794872 * 128.0 },
    { name: "black plastic",  ambient: [0.0,       0.0,       0.0,       1.0], diffuse: [0.01,       0.01,       0.01,       1.0], specular: [0.5,      0.5,      0.5,      1.0], shininess: 0.25       * 128.0 },
    { name: "black rubber",   ambient: [0.02,      0.02,      0.02,      1.0], diffuse: [0.01,       0.01,       0.01,       1.0], specular: [0.4,      0.4,      0.4,      1.0], shininess: 0.78125    * 128.0 },
    // row 2
    { name: "jade",           ambient: [0.135,     0.2225,    0.1575,    1.0], diffuse: [0.54,       0.89,       0.63,       1.0], specular: [0.316228, 0.316228, 0.316228, 1.0], shininess: 0.1        * 128.0 },
    { name: "bronze",         ambient: [0.2125,    0.1275,    0.054,     1.0], diffuse: [0.714,      0.4284,     0.18144,    1.0], specular: [0.393548, 0.271906, 0.166721, 1.0], shininess: 0.2        * 128.0 },
    { name: "cyan plastic",   ambient: [0.0,       0.1,       0.06,      1.0], diffuse: [0.0,        0.50980392, 0.50980392, 1.0], specular: [0.50196078, 0.50196078, 0.50196078, 1.0], shininess: 0.25 * 128.0 },
    { name: "cyan rubber",    ambient: [0.0,       0.05,      0.05,      1.0], diffuse: [0.4,        0.5,        0.5,        1.0], specular: [0.04,     0.7,      0.7,      1.0], shininess: 0.078125   * 128.0 },
    // row 3
    { name: "obsidian",       ambient: [0.05375,   0.05,      0.06625,   1.0], diffuse: [0.18275,    0.17,       0.22525,    1.0], specular: [0.332741, 0.328634, 0.346435, 1.0], shininess: 0.3        * 128.0 },
    { name: "chrome",         ambient: [0.25,      0.25,      0.25,      1.0], diffuse: [0.4,        0.4,        0.4,        1.0], specular: [0.774597, 0.774597, 0.774597, 1.0], shininess: 0.6        * 128.0 },
    { name: "green plastic",  ambient: [0.0,       0.0,       0.0,       1.0], diffuse: [0.1,        0.35,       0.1,        1.0], specular: [0.45,     0.55,     0.45,     1.0], shininess: 0.25       * 128.0 },
    { name: "green rubber",   ambient: [0.0,       0.05,      0.0,       1.0], diffuse: [0.4,        0.5,        0.4,        1.0], specular: [0.04,     0.7,      0.04,     1.0], shininess: 0.0788125  * 128.0 },
    // row 4
    { name: "pearl",          ambient: [0.25,      0.20725,   0.20725,   1.0], diffuse: [1.0,        0.829,      0.829,      1.0], specular: [0.296648, 0.296648, 0.296648, 1.0], shininess: 0.088      * 128.0 },
    { name: "copper",         ambient: [0.19125,   0.0735,    0.0225,    1.0], diffuse: [0.7038,     0.27048,    0.0828,     1.0], specular: [0.256777, 0.137622, 0.086014, 1.0], shininess: 0.1        * 128.0 },
    { name: "red plastic",    ambient: [0.0,       0.0,       0.0,       1.0], diffuse: [0.5,        0.0,        0.0,        1.0], specular: [0.7,      0.6,      0.6,      1.0], shininess: 0.25       * 128.0 },
    { name: "red rubber",     ambient: [0.05,      0.0,       0.0,       1.0], diffuse: [0.5,        0.4,        0.4,        1.0], specular: [0.7,      0.04,     0.04,     1.0], shininess: 0.078125   * 128.0 },
    // row 5
    { name: "ruby",           ambient: [0.1745,    0.01175,   0.01175,   1.0], diffuse: [0.61424,    0.04136,    0.04136,    1.0], specular: [0.727811, 0.626959, 0.626959, 1.0], shininess: 0.6        * 128.0 },
    { name: "gold",           ambient: [0.24725,   0.1995,    0.0745,    1.0], diffuse: [0.75164,    0.60648,    0.22648,    1.0], specular: [0.628281, 0.555802, 0.366065, 1.0], shininess: 0.4        * 128.0 },
    { name: "white plastic",  ambient: [0.0,       0.0,       0.0,       1.0], diffuse: [0.55,       0.55,       0.55,       1.0], specular: [0.7,      0.7,      0.7,      1.0], shininess: 0.25       * 128.0 },
    { name: "white rubber",   ambient: [0.05,      0.05,      0.05,      1.0], diffuse: [0.5,        0.5,        0.5,        1.0], specular: [0.7,      0.7,      0.7,      1.0], shininess: 0.078125   * 128.0 },
    // row 6
    { name: "turquoise",      ambient: [0.1,       0.18725,   0.1745,    1.0], diffuse: [0.396,      0.74151,    0.69102,    1.0], specular: [0.297254, 0.30829,  0.306678, 1.0], shininess: 0.1        * 128.0 },
    { name: "silver",         ambient: [0.19225,   0.19225,   0.19225,   1.0], diffuse: [0.50754,    0.50754,    0.50754,    1.0], specular: [0.508273, 0.508273, 0.508273, 1.0], shininess: 0.4        * 128.0 },
    { name: "yellow plastic", ambient: [0.0,       0.0,       0.0,       1.0], diffuse: [0.5,        0.5,        0.0,        1.0], specular: [0.6,      0.6,      0.5,      1.0], shininess: 0.25       * 128.0 },
    { name: "yellow rubber",  ambient: [0.05,      0.05,      0.0,       1.0], diffuse: [0.5,        0.5,        0.4,        1.0], specular: [0.7,      0.7,      0.04,     1.0], shininess: 0.078125   * 128.0 }
];

const SPHERE_X = [50, 250, 450, 650];               // 4 Columns
const SPHERE_Y = [30, 126, 222, 318, 414, 510];     // 6 Rows

//* Material related macros for dynamic offset
const MATERIAL_SIZE = 64;       // Size of 1 material struct containing 4 vec4s()
const MATERIAL_STRIDE = 256;    // Minimum byte offset alignment by hardware limit. Added with padding 192 bytes + 64 bytes of above struct per sphere
const NUM_SPHERES = 24;

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
    
    render_pipeline = null;
    buffer_hostUniform = null;
    bindGroup_hostUniform = null;
    buffer_material = null;
    bindGroup_material = null;
    perspectiveProjectionMatrix = null;
    depthTexture = null;
    sphere = null;
    buffer_position = null;
    buffer_normal = null;
    buffer_texcoords = null;
    buffer_elements = null;
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
        r: 0.25,
        g: 0.25,
        b: 0.25,
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

    //* Sphere
    sphere = new Mesh();
    makeSphere(sphere, 2.0, 50, 30);
    numMeshIndices = sphere.getIndexCount();
    console.log("Sphere Geometry = Vertex Count = ", sphere.getVertexCount(), " Index Count = ", numMeshIndices);

    const meshData = sphere.getMeshData();

    //* Position Buffer
    buffer_position = createVertexBuffer(meshData.verticesArray);
    if (buffer_position == null)
    {
        console.log("Failed To Create Vertex Position Buffer !!!");
        throw Error("Failed To Create Vertex Position Buffer !!!");
    }
    else
        console.log("Vertex Position Buffer Successfully Created");

    //* Texcoords Buffer
    buffer_texcoords = createVertexBuffer(meshData.texCoordsArray);
    if (buffer_texcoords == null)
    {
        console.log("Failed To Create Vertex Texcoords Buffer !!!");
        throw Error("Failed To Create Vertex Texcoords Buffer !!!");
    }
    else
        console.log("Vertex Texcoords Buffer Successfully Created");

    //* Normals Buffer
    buffer_normal = createVertexBuffer(meshData.normalsArray);
    if (buffer_normal == null)
    {
        console.log("Failed To Create Vertex Normal Buffer !!!");
        throw Error("Failed To Create Vertex Normal Buffer !!!");
    }
    else
        console.log("Vertex Normal Buffer Successfully Created");

    //* Elements Buffer
    buffer_elements = createIndexBuffer(meshData.indicesArray);
    if (buffer_elements == null)
    {
        console.log("Failed To Create Elements Index Buffer !!!");
        throw Error("Failed To Create Elements Index Buffer !!!");
    }
    else
        console.log("Elements Index Buffer Successfully Created");

    //* Uniform Buffer
    buffer_hostUniform = createUniformBuffer(hostUniformBufferSize);
    if (buffer_hostUniform == null)
    {
        console.log("Failed To Create Host Uniform Buffer !!!");
        throw Error("Failed To Create Host Uniform Buffer !!!");
    }
    else
        console.log("Host Uniform Buffer Successfully Created");

    const bindGroupLayout = createBindGroupLayout(0, GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, "uniform", false, 0);
    if (bindGroupLayout == null)
    {
        console.log("Failed To Create Bind Group Layout For Uniform Buffer !!!");
        throw Error("Failed To Create Bind Group Layout For Uniform Buffer !!!");
    }
    else
        console.log("Bind Group Layout For Uniform Buffer Successfully Created");

    
    //* Bind Group
    bindGroup_hostUniform = createBindGroup(buffer_hostUniform, 0, hostUniformBufferSize, 0, bindGroupLayout);
    if (bindGroup_hostUniform == null)
    {
        console.log("Failed To Create Bind Group For Host Uniform !!!");
        throw Error("Failed To Create Bind Group For Host Uniform !!!");
    }
    else
        console.log("Bind Group For Host Uniform Successfully Created");

    //* Material UBO
    const materialBufferSize = NUM_SPHERES * MATERIAL_STRIDE;

    buffer_material = createUniformBuffer(materialBufferSize, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    // Write material table into buffer_material
    const stride = MATERIAL_STRIDE / Float32Array.BYTES_PER_ELEMENT; // [256 / 4 = 64]
    const materialArray = new Float32Array(NUM_SPHERES * stride);   //  [24 * 64 = 1536]

    for (var i = 0; i < NUM_SPHERES; i++)
    {
        materialArray.set(materialTable[i].ambient, i * stride + 0);
        materialArray.set(materialTable[i].diffuse, i * stride + 4);
        materialArray.set(materialTable[i].specular, i * stride + 8);
        materialArray.set([materialTable[i].shininess, 0.0, 0.0, 0.0], i * stride + 12);
    }

    queue.writeBuffer(buffer_material, 0, materialArray, 0, materialArray.length);

    const bindGroupLayout_material = createBindGroupLayout(0, GPUShaderStage.FRAGMENT, "uniform", true, MATERIAL_SIZE);
    if (bindGroupLayout_material == null)
    {
        console.log("Failed To Create Bind Group Layout For Material UBO !!!");
        throw Error("Failed To Create Bind Group Layout For Material UBO !!!");
    }
    else
        console.log("Bind Group Layout For Material UBO Successfully Created");

    
    //* Material Bind Group
    bindGroup_material = createBindGroup(buffer_material, 0, MATERIAL_SIZE, 0, bindGroupLayout_material);
    if (bindGroup_material == null)
    {
        console.log("Failed To Create Bind Group For Material UBO !!!");
        throw Error("Failed To Create Bind Group For Material UBO !!!");
    }
    else
        console.log("Bind Group For Material UBO Successfully Created");

    //* ---------------------------------------------------------------------------------------------------------------------------------

    //* Step - 2: Pipeline Layout

    //* Step - 2A: Pipeline Layout Descriptor
    const pipelineLayoutDescriptor = 
    {
        bindGroupLayouts:
        [
            bindGroupLayout,
            bindGroupLayout_material
        ]
    };

    //* Step - 2B: Pipeline Layout
    const pipelineLayout = device.createPipelineLayout(pipelineLayoutDescriptor);
    if (pipelineLayout == null)
    {
        console.log("Failed To Create Pipeline Layout !!!");
        throw Error("Failed To Create Pipeline Layout !!!");
    }
    else
        console.log("Pipeline Layout Successfully Created");

    //! Render Pipeline

    //* Step - 1 : Pipeline Descriptor / PSO

    //* Step - 1A: Vertex Buffer Layout
    const positionVertexAttribute = 
    {
        shaderLocation: 0,  //* Maps to location(0) in Vertex Shader
        offset: 0,
        format: "float32x3"
    };

    const positionVertexBufferLayout = 
    {
        attributes: 
        [
            positionVertexAttribute
        ],
        arrayStride: Float32Array.BYTES_PER_ELEMENT * 3,
        stepMode: "vertex"  // Jump vertex by vertex, not instance by instance
    };

    //! Normal Attribute
    const normalVertexAttribute = 
    {
        shaderLocation: 1,  //* Maps to location(1) in Vertex Shader
        offset: 0,
        format: "float32x3"
    };

    const normalVertexBufferLayout = 
    {
        attributes: 
        [
            normalVertexAttribute
        ],
        arrayStride: Float32Array.BYTES_PER_ELEMENT * 3,
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
            normalVertexBufferLayout
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

function createIndexBuffer(_data)
{
    // Code
    const bufferDescriptor = 
    {
        size: _data.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST
    };

    const buffer = device.createBuffer(bufferDescriptor);
    if (buffer == null)
        return null;

    queue.writeBuffer(buffer, 0, _data);

    return buffer;
}

function createBindGroupLayout(_bindingIndex, _shaderStageVisibility, _uniformType, _hasDynamicOffset, _minBindingSize)
{
    // Code

    //* Step - 1A: Bind Group Layout Entry
    const bindGroupLayoutEntry = 
    {
        binding: _bindingIndex,
        visibility: _shaderStageVisibility,
        buffer: 
        {
            type: _uniformType,
            hasDynamicOffset: _hasDynamicOffset,
            minBindingSize: _minBindingSize
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
    const bindGroupLayout = device.createBindGroupLayout(bindGroupLayoutDescriptor);
   
    return bindGroupLayout;
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

function createBindGroup(_uniformBuffer, _offset, _uniformBufferSize, _bindGroupBinding, _bindGroupLayout)
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
        binding: _bindGroupBinding,
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
    
    //! Transformations
    hostUniformData.modelMatrix = mat4.create();
    hostUniformData.modelMatrix = mat4.translate(hostUniformData.modelMatrix, hostUniformData.modelMatrix, [0.0, 0.0, -6.0]);
    hostUniformData.viewMatrix = mat4.create();
    hostUniformData.projectionMatrix = perspectiveProjectionMatrix;

    if (bLight)
        hostUniformData.lightEnabled[0] = 1;
    else
        hostUniformData.lightEnabled[0] = 0;

    //* Reset Light Position
    hostUniformData.lightPosition[0] = 0.0;
    hostUniformData.lightPosition[1] = 0.0;
    hostUniformData.lightPosition[2] = 0.0;
    hostUniformData.lightPosition[3] = 1.0;

    switch (chosenAxis)
    {
        case 'X':
            hostUniformData.lightPosition[1] = radius * Math.sin(degreeToRadians(angleLight));
            hostUniformData.lightPosition[2] = radius * Math.cos(degreeToRadians(angleLight));
        break;

        case 'Y':
            hostUniformData.lightPosition[0] = radius * Math.cos(degreeToRadians(angleLight));
            hostUniformData.lightPosition[2] = radius * Math.sin(degreeToRadians(angleLight));
        break;

        case 'Z':
            hostUniformData.lightPosition[0] = radius * Math.cos(degreeToRadians(angleLight));
            hostUniformData.lightPosition[1] = radius * Math.sin(degreeToRadians(angleLight));
        break;
    }

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

    // Light Vectors        : 16 + 16 + 16 + 16  =  64
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3,
        hostUniformData.lightAmbient
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 1, 
        hostUniformData.lightDiffuse
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 2, 
        hostUniformData.lightSpecular
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 3, 
        hostUniformData.lightPosition
    );

    // Light Enabled Vector : 16
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 4,  
        hostUniformData.lightEnabled
    );

    //* Step - 13 : Begin The Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    {
        renderPassEncoder.setPipeline(render_pipeline);
        renderPassEncoder.setScissorRect(0, 0, canvas.width, canvas.height);
        renderPassEncoder.setVertexBuffer(0, buffer_position);
        renderPassEncoder.setVertexBuffer(1, buffer_normal);
        renderPassEncoder.setIndexBuffer(buffer_elements, "uint16");
        renderPassEncoder.setBindGroup(0, bindGroup_hostUniform);

        const viewportWidth = canvas.width / 8.0;
        const viewportHeight = canvas.height / 8.0;

        for (let row = 0; row < 6; row++)
        {
            for (let col = 0; col < 4; col++)
            {
                const X = SPHERE_X[col];
                const Y = SPHERE_Y[row];

                const viewportX = X * canvas.width / 800;
                const viewportY = Y * canvas.height / 600;

                const i = (row * 4) + col;

                renderPassEncoder.setViewport(viewportX, viewportY, viewportWidth, viewportHeight, 0.0, 1.0);
                renderPassEncoder.setBindGroup(1, bindGroup_material, [i * MATERIAL_STRIDE]);
                renderPassEncoder.drawIndexed(numMeshIndices);
            }
        }
    }
    renderPassEncoder.end();

    //* Step - 14 : Finish The Command Encoder And Submit To Queue
    queue.submit([commandEncoder.finish()]);

    update();

    //! Animation Loop
    animationFrameId = requestAnimationFrame(display);
}

function degreeToRadians(degrees)
{
    return (degrees * (Math.PI / 180.0));
}

function update()
{
    // Code
    if (chosenAxis == 'X' || chosenAxis == 'Y' || chosenAxis == 'Z')
    {
        angleLight = angleLight + 0.5;
        if (angleLight > 360.0)
            angleLight = angleLight - 360.0;
    }

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
        render_pipeline = null;
        buffer_hostUniform = null;
        bindGroup_hostUniform = null;
        buffer_material = null;
        bindGroup_material = null;
        buffer_position = null;
        buffer_normal = null;
        buffer_texcoords = null;
        buffer_elements = null;
        sphere = null;
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

        case 'l':
        case 'L':
            bLight = !bLight;
        break;

        case 'x':
        case 'X':
            chosenAxis = 'X';
        break;

        case 'y':
        case 'Y':
            chosenAxis = 'Y';
        break;

        case 'z':
        case 'Z':
            chosenAxis = 'Z';
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

