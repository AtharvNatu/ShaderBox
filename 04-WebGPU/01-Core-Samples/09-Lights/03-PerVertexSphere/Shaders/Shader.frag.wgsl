struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0) phong_ads_light : vec3<f32>
};

@fragment
fn main(output : VertexOutput) -> @location(0) vec4<f32>
{
    var color = vec4(output.phong_ads_light, 1.0);
    return color;
}
