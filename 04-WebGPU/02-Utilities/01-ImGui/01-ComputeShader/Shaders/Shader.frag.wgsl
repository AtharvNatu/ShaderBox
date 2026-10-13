struct UniformBuffer
{
    modelMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,

    color: vec4<f32>,
    computeData: vec4<f32>
};

@group(0) @binding(0) var<uniform> ubo : UniformBuffer;

@fragment
fn main() -> @location(0) vec4<f32>
{
    return ubo.color;
}
