struct HostUniform
{
    modelMatrix : mat4x4<f32>,
    viewMatrix : mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    keyPressed: vec4<u32>
};

struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0) texcoord : vec2<f32>
};

@group(0) @binding(0) var<uniform> hostUniform : HostUniform;

@vertex
fn main(@location(0) positionIn : vec4<f32>, @location(1) texcoordIn : vec2<f32>) -> VertexOutput
{
    // Code
    var output : VertexOutput; 
    output.position = hostUniform.projectionMatrix * hostUniform.viewMatrix * hostUniform.modelMatrix * positionIn;
    output.texcoord = texcoordIn;
    return output;
}

