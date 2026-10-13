struct UniformBuffer
{
    modelMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,

    color: vec4<f32>,
    computeData: vec4<f32>
};

@group(0) @binding(0) var<uniform> ubo : UniformBuffer;

@vertex
fn main(@location(0) vPosition : vec4<f32>) -> @builtin(position) vec4<f32>
{
    return ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * vPosition;
}



