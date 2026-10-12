struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0) texcoord : vec2<f32>
};

@group(1) @binding(0) var uTeapotTexture : texture_2d<f32>;
@group(1) @binding(1) var uTeapotSampler : sampler;

@fragment
fn main(output : VertexOutput) -> @location(0) vec4<f32>
{
    var color = textureSample(uTeapotTexture, uTeapotSampler, output.texcoord);
    return color;
}

