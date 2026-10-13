# Steps For WebGPU Perspective Triangle

1) Add `gl-matrix-min.js` to the directory and include before `Canvas.js` in `Canvas.html`

2) Add global variables related to vertex buffer, uniform buffer, bind group, render pipeline, perspective projection matrix

3) `onDeviceLost()` Changes

    - Set all global variables to `null`

4) `initialize()` Changes

    1) Write vertex shader source code after canvas configuration

    2) Create vertex shader module:

        1) Create shader module descriptor
        2) Create actual vertex shader module

    3) Create fragment shader module:

        1) Create shader module descriptor
        2) Create actual fragment shader module

    4) Declare position array

    5) Create vertex buffer for position:

        1) Create buffer descriptor
        2) Create actual vertex buffer for position
        3) Write position array to above created buffer

    6) Uniform plumbing for MVP Uniform:

        1) Uniforms will bind to the bind group in the shader

            1) Create bind group layout for our MVP uniform:

                1) Create bind group layout entry
                2) Create bind group layout descriptor
                3) Create actual bind group layout

            2) Create pipeline layout to specify bind group layout count:

                1) Create pipeline layout descriptor
                2) Create actual pipeline layout

            3) Create uniform buffer for our MVP uniform:

                1) Create buffer descriptor
                2) Create actual uniform buffer [`Data will be written to this buffer in display()`]

            4) Create the bind group for our MVP uniform buffer

                1) Create buffer binding property
                2) Create binding group entry for above buffer binding property
                3) Create binding group descriptor to specify above binding group entry
                4) Create the actual binding group

    7) Create render pipeline:

        1) Create render pipeline descriptor [`Pipeline State Object (PSO)`]:

            1) Create vertex buffer layout:

                1) Vertex attributes like position, color, normal, texcoords
                2) Create actual vertex buffer layout using 1 or more vertex attributes

            2) Vertex Shader State

            3) Fragment Shader State

                1) Color Target State
                2) Create actual fragment shader state

            4) Primitive State

            5) Create a PSO / Render Pipeline Descriptor
            
        2) Create the actual render pipeline using the above PSO

    8) Initialize projection matrix

5) `resize()` Changes

    - Set projection matrix

6) `display()` Changes

    1) Transformations:

        1) Create and initialize required matrices - modelView and modelViewProjection 

        2) Perform needed transformations

        3) Perform matrix multiplication to calculate modelViewProjection Matrix

    2) Write this MVP to the uniform buffer created in `initialize()`
        
    3) Start the render pass

    4) Set following properties of Renderpass Encoder:

        1) Pipeline
        2) Viewport
        3) Scissor Rectangle
        4) Vertex Buffer(s)
        5) Bind Group

    5) End the render pass

7) `uninitialize()` Changes

    1) Destroy the newly added global variables in `device` destruction block
