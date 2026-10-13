struct MVPUniform
{
    mvpMatrix : mat4x4<f32>,
};

struct VertexInput
{
    @location(0) position : vec3<f32>,
    @location(1) texcoord : vec2<f32>,
    @location(2) normal : vec3<f32>,
};

struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0) texcoord : vec2<f32>,
};

@group(0) @binding(0) var<uniform> mvpUniform : MVPUniform;

@vertex
fn main(input : VertexInput) -> VertexOutput
{
    var output : VertexOutput;

    output.position = mvpUniform.mvpMatrix * vec4<f32>(input.position, 1.0);
    output.texcoord = input.texcoord;

    return output;
}



