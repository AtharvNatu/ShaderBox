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

@group(1) @binding(0) var texture_binding : texture_2d<f32>;
@group(1) @binding(1) var sampler_binding : sampler;

@fragment
fn main(output : VertexOutput) -> @location(0) vec4<f32>
{
    var color = vec4(1.0);

    if (hostUniform.keyPressed.x == 1)
    {
        color = textureSample(texture_binding, sampler_binding, output.texcoord);
    }
    else
    {
        color = vec4(1.0, 1.0, 1.0, 1.0);
    }
    
    return color;
}

