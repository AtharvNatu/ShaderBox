const { mat4, mat3 } = glMatrix;

//* Global Variables
var canvas = null;
var bFullscreen = false;
var canvas_original_width;
var canvas_original_height;

//! WebGPU Related
var clearColor;
var clearColor_fbo;

let device = null;
let context = null;
let queue = null;
let canvasFormat = null;
let animationFrameId = null;

let perspectiveProjectionMatrix = null;
let perspectiveProjectionMatrix_fbo = null;
let depthTexture = null;

//* Teapot Related Declarations
const FBO_WIDTH = 512;
const FBO_HEIGHT = 512;

let texture_teapot_color = null;
let texture_teapot_depth = null;
let texture_marble = null;
let sampler_marble = null;
let sampler_teapot = null;
let render_pipeline_teapot = null;
let buffer_uniform_teapot = null;
let bindGroup_uniform_teapot = null;
let bindGroup_marble_texture_sampler = null;
let buffer_teapot_position = null;
let buffer_teapot_normals = null;
let buffer_teapot_texcoords = null;
let buffer_teapot_elements = null;
let num_teapot_indices = null;

//* Cube Related Declarations
let render_pipeline_cube = null;
let buffer_uniform_cube = null;
let bindGroup_uniform_cube = null;
let bindGroup_teapot_texture_sampler = null;
let buffer_cube_position = null;
let buffer_cube_texcoords = null;

const teapotUniformData =
{
    modelMatrix: mat4.create(),
    viewMatrix: mat4.create(),
    projectionMatrix: mat4.create(),

    lightAmbient: new Float32Array([0.4, 0.4, 0.4, 1.0]),
    lightDiffuse: new Float32Array([1.0, 1.0, 1.0, 1.0]),
    lightSpecular: new Float32Array([1.0, 1.0, 1.0, 1.0]),
    lightPosition: new Float32Array([100.0, 100.0, 100.0, 1.0]),

    materialAmbient: new Float32Array([0.9, 0.5, 0.3, 1.0]),
    materialDiffuse: new Float32Array([0.9, 0.5, 0.3, 1.0]),
    materialSpecular: new Float32Array([0.8, 0.8, 0.8, 1.0]),
    materialShininess: new Float32Array([128.0, 0.0, 0.0, 0.0]),
    keyFlags: new Uint32Array([0, 0, 0, 0]),   // x = Light, y = Texture
};

var bLight = false;
var bTexture = false;
var bAnimate = false;

var angleTeapot = 0.0;
var angleCube = 0.0;
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
    document.addEventListener("uncapturederror", onUncapturedError);
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

    // Teapot Related
    texture_teapot_color = null;
    texture_teapot_depth = null;
    texture_marble = null;
    sampler_marble = null;
    sampler_teapot = null;
    render_pipeline_teapot = null;
    buffer_uniform_teapot = null;
    bindGroup_uniform_teapot = null;
    bindGroup_marble_texture_sampler = null;
    buffer_teapot_position = null;
    buffer_teapot_normals = null;
    buffer_teapot_texcoords = null;
    buffer_teapot_elements = null;

    // Cube Related
    render_pipeline_cube = null;
    buffer_uniform_cube = null;
    bindGroup_uniform_cube = null;
    bindGroup_teapot_texture_sampler = null;
    buffer_cube_position = null;
    buffer_cube_texcoords = null;

    perspectiveProjectionMatrix = null;
    perspectiveProjectionMatrix_fbo = null;

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
            canvas.msRequestFullScreen();
    }
    else
    {
        if (document.exitFullscreen)
            document.exitFullscreen();
        else if (document.mozExitFullScreen)
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
    const teapotVertWGSL = await loadShader("../Shaders/Teapot.vert.wgsl");
    const teapotFragWGSL = await loadShader("../Shaders/Teapot.frag.wgsl");
    const cubeVertWSGL = await loadShader("../Shaders/Cube.vert.wgsl");
    const cubeFragWGSL = await loadShader("../Shaders/Cube.frag.wgsl");

    //* Vertex Shader Module
    const shaderModuleDescriptor_teapotVertexShader = 
    {
        code: teapotVertWGSL
    };

    const shaderModule_vertexShader_teapot = device.createShaderModule(shaderModuleDescriptor_teapotVertexShader);
    if (shaderModule_vertexShader_teapot == null)
    {
        console.log("Failed To Create Vertex Shader Module For Teapot !!!");
        throw Error("Failed To Create Vertex Shader Module For Teapot !!!");
    }
    else
        console.log("Vertex Shader Module Successfully Created For Teapot");

    //* Fragment Shader Module
    const shaderModuleDescriptor_teapotFragmentShader = 
    {
        code: teapotFragWGSL
    };

    const shaderModule_fragmentShader_teapot = device.createShaderModule(shaderModuleDescriptor_teapotFragmentShader);
    if (shaderModule_fragmentShader_teapot == null)
    {
        console.log("Failed To Create Fragment Shader Module For Teapot !!!");
        throw Error("Failed To Create Fragment Shader Module For Teapot !!!");
    }
    else
        console.log("Fragment Shader Module Successfully Created For Teapot");

    //* Vertex Shader Module
    const shaderModuleDescriptor_cubeVertexShader = 
    {
        code: cubeVertWSGL
    };

    const shaderModule_vertexShader_cube = device.createShaderModule(shaderModuleDescriptor_cubeVertexShader);
    if (shaderModule_vertexShader_cube == null)
    {
        console.log("Failed To Create Vertex Shader Module For Cube !!!");
        throw Error("Failed To Create Vertex Shader Module For Cube !!!");
    }
    else
        console.log("Vertex Shader Module Successfully Created For Cube");

    //* Fragment Shader Module
    const shaderModuleDescriptor_cubeFragmentShader = 
    {
        code: cubeFragWGSL
    };

    const shaderModule_fragmentShader_cube = device.createShaderModule(shaderModuleDescriptor_cubeFragmentShader);
    if (shaderModule_fragmentShader_cube == null)
    {
        console.log("Failed To Create Fragment Shader Module For Cube !!!");
        throw Error("Failed To Create Fragment Shader Module For Cube !!!");
    }
    else
        console.log("Fragment Shader Module Successfully Created For Cube");

    //* Teapot Geometry
    const teapot = new Mesh();
    teapot.allocate(teapotVertexCount);

    const startTime = performance.now();

    for (let i = 0; i < teapotVertexCount; i = i + 3)
    {
        // Positions
        const single_vertex = 
        [
            [
                teapotPositions[(i + 0) * 3 + 0], 
                teapotPositions[(i + 0) * 3 + 1], 
                teapotPositions[(i + 0) * 3 + 2]
            ],
            [
                teapotPositions[(i + 1) * 3 + 0], 
                teapotPositions[(i + 1) * 3 + 1], 
                teapotPositions[(i + 1) * 3 + 2]
            ],
            [
                teapotPositions[(i + 2) * 3 + 0], 
                teapotPositions[(i + 2) * 3 + 1], 
                teapotPositions[(i + 2) * 3 + 2] 
            ]
        ];

        // Normals
        const single_normal = 
        [
            [
                teapotNormals[(i + 0) * 3 + 0], 
                teapotNormals[(i + 0) * 3 + 1], 
                teapotNormals[(i + 0) * 3 + 2]
            ],
            [
                teapotNormals[(i + 1) * 3 + 0], 
                teapotNormals[(i + 1) * 3 + 1], 
                teapotNormals[(i + 1) * 3 + 2]
            ],
            [
                teapotNormals[(i + 2) * 3 + 0], 
                teapotNormals[(i + 2) * 3 + 1], 
                teapotNormals[(i + 2) * 3 + 2]
            ]
        ];

        // Normals
        const single_texcoord = 
        [
            [
                teapotTexcoords[(i + 0) * 2 + 0], 
                teapotTexcoords[(i + 0) * 2 + 1] 
            ],
            [
                teapotTexcoords[(i + 1) * 2 + 0], 
                teapotTexcoords[(i + 1) * 2 + 1] 
            ],
            [
                teapotTexcoords[(i + 2) * 2 + 0], 
                teapotTexcoords[(i + 2) * 2 + 1] 
            ]
        ];

        teapot.addTriangle(single_vertex, single_normal, single_texcoord);
    }

    num_teapot_indices = teapot.getIndexCount();
    
    const teapotMeshData = teapot.getMeshData();

    buffer_teapot_position = createVertexBuffer(teapotMeshData.verticesArray);
    if (buffer_teapot_position == null)
    {
        console.log("Failed To Create Teapot Position Buffer !!!");
        throw Error("Failed To Create Teapot Position Buffer !!!");
    }
    else
        console.log("Teapot Position Buffer Created");

    buffer_teapot_texcoords = createVertexBuffer(teapotMeshData.texcoordsArray);
    if (buffer_teapot_texcoords == null)
    {
        console.log("Failed To Create Teapot Texcoords Buffer !!!");
        throw Error("Failed To Create Teapot Texcoords Buffer !!!");
    }
    else
        console.log("Teapot Texcoords Buffer Created");

    buffer_teapot_normals = createVertexBuffer(teapotMeshData.normalsArray);
    if (buffer_teapot_normals == null)
    {
        console.log("Failed To Create Teapot Normals Buffer !!!");
        throw Error("Failed To Create Teapot Normals Buffer !!!");
    }
    else
        console.log("Teapot Normals Buffer Created");

    buffer_teapot_elements = createIndexBuffer(teapotMeshData.indicesArray);
    if (buffer_teapot_elements == null)
    {
        console.log("Failed To Create Teapot Index Buffer !!!");
        throw Error("Failed To Create Teapot Index Buffer !!!");
    }
    else
        console.log("Teapot Index Buffer Created");

    //! Cube
    const vertex_position_cube = new Float32Array([
        // Front
        1.0,  1.0,  1.0,  1.0,   // Top Right
       -1.0,  1.0,  1.0,  1.0,   // Top Left
        1.0, -1.0,  1.0,  1.0,   // Bottom Right

        1.0, -1.0,  1.0,  1.0,   // Bottom Right
       -1.0,  1.0,  1.0,  1.0,   // Top Left
       -1.0, -1.0,  1.0,  1.0,   // Bottom Left

        // Right
        1.0,  1.0, -1.0,  1.0,   // Top Right
        1.0,  1.0,  1.0,  1.0,   // Top Left
        1.0, -1.0, -1.0,  1.0,   // Bottom Right

        1.0, -1.0, -1.0,  1.0,   // Bottom Right
        1.0,  1.0,  1.0,  1.0,   // Top Left
        1.0, -1.0,  1.0,  1.0,   // Bottom Left

        // Back
        1.0,  1.0, -1.0,  1.0,   // Top Right
       -1.0,  1.0, -1.0,  1.0,   // Top Left
        1.0, -1.0, -1.0,  1.0,   // Bottom Right

        1.0, -1.0, -1.0,  1.0,   // Bottom Right
       -1.0,  1.0, -1.0,  1.0,   // Top Left
       -1.0, -1.0, -1.0,  1.0,   // Bottom Left

        // Left
       -1.0,  1.0,  1.0,  1.0,   // Top Right
       -1.0,  1.0, -1.0,  1.0,   // Top Left
       -1.0, -1.0,  1.0,  1.0,   // Bottom Right

       -1.0, -1.0,  1.0,  1.0,   // Bottom Right
       -1.0,  1.0, -1.0,  1.0,   // Top Left
       -1.0, -1.0, -1.0,  1.0,   // Bottom Left

        // Top
        1.0,  1.0, -1.0,  1.0,   // Top Right
       -1.0,  1.0, -1.0,  1.0,   // Top Left
        1.0,  1.0,  1.0,  1.0,   // Bottom Right

        1.0,  1.0,  1.0,  1.0,   // Bottom Right
       -1.0,  1.0, -1.0,  1.0,   // Top Left
       -1.0,  1.0,  1.0,  1.0,   // Bottom Left

        // Bottom
        1.0, -1.0,  1.0,  1.0,   // Top Right
       -1.0, -1.0,  1.0,  1.0,   // Top Left
        1.0, -1.0, -1.0,  1.0,   // Bottom Right

        1.0, -1.0, -1.0,  1.0,   // Bottom Right
       -1.0, -1.0,  1.0,  1.0,   // Top Left
       -1.0, -1.0, -1.0,  1.0   // Bottom Left
    ]);

    const vertex_texcoord_cube = new Float32Array([ 
        // Front Face
        0.0,  1.0,   // Top Right
        1.0,  1.0,   // Top Left
        0.0,  0.0,   // Bottom Right

        0.0,  0.0,   // Bottom Right
        1.0,  1.0,   // Top Left
        1.0,  0.0,   // Bottom Left

        // Right Face
        0.0,  1.0,   // Top Right
        1.0,  1.0,   // Top Left
        0.0,  0.0,   // Bottom Right

        0.0,  0.0,   // Bottom Right
        1.0,  1.0,   // Top Left
        1.0,  0.0,   // Bottom Left

        // Back Face
        0.0,  1.0,   // Top Right
        1.0,  1.0,   // Top Left
        0.0,  0.0,   // Bottom Right

        0.0,  0.0,   // Bottom Right
        1.0,  1.0,   // Top Left
        1.0,  0.0,   // Bottom Left

        // Left Face
        0.0,  1.0,   // Top Right
        1.0,  1.0,   // Top Left
        0.0,  0.0,   // Bottom Right

        0.0,  0.0,   // Bottom Right
        1.0,  1.0,   // Top Left
        1.0,  0.0,   // Bottom Left

        // Top Face
        0.0,  1.0,   // Top Right
        1.0,  1.0,   // Top Left
        0.0,  0.0,   // Bottom Right

        0.0,  0.0,   // Bottom Right
        1.0,  1.0,   // Top Left
        1.0,  0.0,   // Bottom Left

        // Bottom Face
        0.0,  1.0,   // Top Right
        1.0,  1.0,   // Top Left
        0.0,  0.0,   // Bottom Right

        0.0,  0.0,   // Bottom Right
        1.0,  1.0,   // Top Left
        1.0,  0.0,   // Bottom Left
    ]);

    //* Vertex Buffers
    buffer_cube_position = createVertexBuffer(vertex_position_cube);
    if (buffer_cube_position == null)
    {
        console.log("Failed To Create Cube Position Buffer !!!");
        throw Error("Failed To Create Cube Position Buffer !!!");
    }
    else
        console.log("Cube Position Buffer Created");
    
    buffer_cube_texcoords = createVertexBuffer(vertex_texcoord_cube);
    if (buffer_cube_texcoords == null)
    {
        console.log("Failed To Create Cube Texcoords Buffer !!!");
        throw Error("Failed To Create Cube Texcoords Buffer !!!");
    }
    else
        console.log("Cube Texcoords Buffer Created");

    //* Marble Texture
    texture_marble = await loadTexture("../Assets/marble.png");
    if (texture_marble == null)
    {
        console.log("Failed To Load Marble Texture !!!");
        throw Error("Failed To Load Marble Texture !!!");
    }
    else
        console.log("Marble Texture Loaded");

    //! FBO Texture - No image, the teapot will be rendered as texture for cube
    
    //* Color Texture
    const teapotColorTextureDescriptor = 
    {
        size: [FBO_WIDTH, FBO_HEIGHT, 1],
        dimension: "2d",
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
    };

    texture_teapot_color = device.createTexture(teapotColorTextureDescriptor);
    if (texture_teapot_color == null)
    {
        console.log("Failed To Create Texture For Teapot Color !!!");
        throw Error("Failed To Create Texture For Teapot Color !!!");
    }
    else
        console.log("Texture For Teapot Color Created Successfully");

    //* Depth Texture
    const teapotDepthTextureDescriptor = 
    {
        size: [FBO_WIDTH, FBO_HEIGHT, 1],
        dimension: "2d",
        format: "depth24plus-stencil8",
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    };

    texture_teapot_depth = device.createTexture(teapotDepthTextureDescriptor);
    if (texture_teapot_depth == null)
    {
        console.log("Failed To Create Texture For Teapot Depth !!!");
        throw Error("Failed To Create Texture For Teapot Depth !!!");
    }
    else
        console.log("Texture For Teapot Depth Created Successfully");

    //* Teapot Texture Sampler Descriptor
    const samplerDescriptor_teapot = 
    {
        magFilter: "linear",
        minFilter: "linear"
    };

    sampler_teapot = device.createSampler(samplerDescriptor_teapot);
    if (sampler_teapot == null)
    {
        console.log("Failed To Create Sampler For Teapot !!!");
        throw Error("Failed To Create Sampler For Teapot !!!");
    }
    else
        console.log("Sampler For Teapot Created Successfully");

    //! Teapot UBO

    // MVP Matrices         : 64 + 64 + 64       =  192 +
    // Light Vectors        : 16 + 16 + 16 + 16  =  64  +
    // Material Vectors     : 16 + 16 + 16 + 16  =  64  +
    // Key Flags Vector     : 16                 =  16  =   336
    const teapotUniformBufferSize = new ArrayBuffer(
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +   // MVP Matrices
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Light Ambient
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Light Diffuse
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Light Specular
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Light Position
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Material Ambient
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Material Diffuse
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Material Specular
        Float32Array.BYTES_PER_ELEMENT * 4  +       // Material Shininess
        Float32Array.BYTES_PER_ELEMENT * 4          // Key Flags
    ).byteLength;

    buffer_uniform_teapot = createUniformBuffer(teapotUniformBufferSize);
    if (buffer_uniform_teapot == null)
    {
        console.log("Failed To Create Uniform Buffer For Teapot !!!");
        throw Error("Failed To Create Uniform Buffer For Teapot !!!");
    }
    else
        console.log("Uniform Buffer For Teapot Successfully Created");

    //! Cube UBO
    const cubeUniformBufferSize = Float32Array.BYTES_PER_ELEMENT * 16;
    buffer_uniform_cube = createUniformBuffer(cubeUniformBufferSize);
    if (buffer_uniform_cube == null)
    {
        console.log("Failed To Create Uniform Buffer For Cube !!!");
        throw Error("Failed To Create Uniform Buffer For Cube !!!");
    }
    else
        console.log("Uniform Buffer For Cube Successfully Created");

    //! Bind Group Layouts
    const bindGroupLayout_uniform_teapot = createBindGroupLayout(0, GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, "uniform");
    if (bindGroupLayout_uniform_teapot == null)
    {
        console.log("Failed To Create Bind Group Layout For Teapot Uniform !!!");
        throw Error("Failed To Create Bind Group Layout For Teapot Uniform !!!");
    }
    else
        console.log("Bind Group Layout For Teapot Uniform Successfully Created");

    const bindGroupLayout_uniform_cube = createBindGroupLayout(0, GPUShaderStage.VERTEX, "uniform");
    if (bindGroupLayout_uniform_cube == null)
    {
        console.log("Failed To Create Bind Group Layout For Cube Uniform !!!");
        throw Error("Failed To Create Bind Group Layout For Cube Uniform !!!");
    }
    else
        console.log("Bind Group Layout For Cube Uniform Successfully Created");
    
    //* Teapot and Cube will share the same bind group layout as they use the same texture format
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

    //! Bind Groups
    bindGroup_uniform_teapot = createBindGroup(buffer_uniform_teapot, 0, teapotUniformBufferSize, 0, bindGroupLayout_uniform_teapot);
    if (bindGroup_uniform_teapot == null)
    {
        console.log("Failed To Create Bind Group For Teapot Uniform !!!");
        throw Error("Failed To Create Bind Group For Teapot Uniform !!!");
    }
    else
        console.log("Bind Group For Teapot Uniform Successfully Created");

    bindGroup_uniform_cube = createBindGroup(buffer_uniform_cube, 0, cubeUniformBufferSize, 0, bindGroupLayout_uniform_cube);
    
    bindGroup_marble_texture_sampler = createBindGroup_texture_sampler(0, texture_marble, 1, sampler_marble, bindGroupLayout_texture_sampler);
    if (bindGroup_marble_texture_sampler == null)
    {
        console.log("Failed To Create Bind Group For Texture and Sampler For Marble !!!");
        throw Error("Failed To Create Bind Group For Texture and Sampler For Marble !!!");
    }
    else
        console.log("Bind Group Successfully Created For Texture and Sampler For Marble");
    
    bindGroup_teapot_texture_sampler = createBindGroup_texture_sampler(0, texture_teapot_color, 1, sampler_teapot, bindGroupLayout_texture_sampler);
    if (bindGroup_teapot_texture_sampler == null)
    {
        console.log("Failed To Create Bind Group For Texture and Sampler For Teapot !!!");
        throw Error("Failed To Create Bind Group For Texture and Sampler For Teapot !!!");
    }
    else
        console.log("Bind Group Successfully Created For Texture and Sampler For Teapot");

    //* Teapot Pipeline Layout
    const pipelineLayoutDescriptor_teapot = 
    {
        bindGroupLayouts:
        [
            bindGroupLayout_uniform_teapot,     // Group 0
            bindGroupLayout_texture_sampler     // Group 1    
        ]
    };

    const pipelineLayout_teapot = device.createPipelineLayout(pipelineLayoutDescriptor_teapot);
    if (pipelineLayout_teapot == null)
    {
        console.log("Failed To Create Pipeline Layout For Teapot !!!");
        throw Error("Failed To Create Pipeline Layout For Teapot !!!");
    }
    else
        console.log("Pipeline Layout For Teapot Successfully Created");

    //* Cube Pipeline Layout
    const pipelineLayoutDescriptor_cube = 
    {
        bindGroupLayouts:
        [
            bindGroupLayout_uniform_cube,       // Group 0
            bindGroupLayout_texture_sampler     // Group 1    
        ]
    };

    const pipelineLayout_cube = device.createPipelineLayout(pipelineLayoutDescriptor_cube);
    if (pipelineLayout_cube == null)
    {
        console.log("Failed To Create Pipeline Layout For Cube !!!");
        throw Error("Failed To Create Pipeline Layout For Cube !!!");
    }
    else
        console.log("Pipeline Layout For Cube Successfully Created");

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

function createUniformBuffer(_uniformBufferSize, _uniformBufferUsage)
{
    // Code
    const bufferDescriptor = 
    {
        size: _uniformBufferSize,
        usage: _uniformBufferUsage
    };

    return device.createBuffer(bufferDescriptor);
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

    mat4.perspectiveZO(perspectiveProjectionMatrix, 45.0, parseFloat(canvas.width) / parseFloat(canvas.height), 0.1, 100.0);
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

    // Material Vectors     : 16 + 16 + 16 + 16  =  64
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 4,   
        hostUniformData.materialAmbient
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 5,  
        hostUniformData.materialDiffuse
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 6, 
        hostUniformData.materialSpecular
    );
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 7, 
        hostUniformData.materialShininess
    );

    // Light Enabled Vector : 16
    queue.writeBuffer(
        buffer_hostUniform, 
        Float32Array.BYTES_PER_ELEMENT * 16 * 3 +
        Float32Array.BYTES_PER_ELEMENT * 4 * 8,  
        hostUniformData.lightEnabled
    );

    //* Step - 13 : Begin The Render Pass
    const renderPassEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    {
        renderPassEncoder.setPipeline(render_pipeline);
        renderPassEncoder.setViewport(0, 0, canvas.width, canvas.height, 0.0, 1.0);
        renderPassEncoder.setScissorRect(0, 0, canvas.width, canvas.height);
        renderPassEncoder.setVertexBuffer(0, buffer_position);
        renderPassEncoder.setVertexBuffer(1, buffer_normal);
        renderPassEncoder.setIndexBuffer(buffer_elements, "uint16");
        renderPassEncoder.setBindGroup(0, bindGroup_hostUniform);
        renderPassEncoder.drawIndexed(numMeshIndices);
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

