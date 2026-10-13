struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0) texcoord : vec2<f32>
};

@group(1) @binding(0) var texture_binding : texture_2d<f32>;
@group(1) @binding(1) var sampler_binding : sampler;

@fragment
fn main(output : VertexOutput) -> @location(0) vec4<f32>
{
    var color = textureSample(texture_binding, sampler_binding, output.texcoord);
    return color;
}

