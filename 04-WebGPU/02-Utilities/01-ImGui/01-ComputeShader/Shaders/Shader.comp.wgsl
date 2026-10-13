struct UniformBuffer
{
    modelMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,

    color: vec4<f32>,
    computeData: vec4<f32>  // 0 - Width, 1 - Height, 2 - Time, 3 - Padding
};

struct PositionBuffer
{
    position: array<vec4<f32>>
};

@group(0) @binding(0) var<uniform> ubo : UniformBuffer;
@group(1) @binding(0) var<storage, read_write> position_buffer : PositionBuffer;

// Overridable Constant
override WORKGROUP_SIZE: u32 = 32;

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let index = global_id.xy;
    if (index.x >= u32(ubo.computeData[0]) || index.y >= u32(ubo.computeData[1]))
    {
        return;
    }

    var u = f32(index.x) / ubo.computeData[0];
    var v = f32(index.y) / ubo.computeData[1];

    u = u * 2.0 - 1.0;
    v = v * 2.0 - 1.0;

    let frequency = 5.0;
    let w = cos(u * frequency + ubo.computeData[2]) * sin(v * frequency + ubo.computeData[2]) * 0.5;

    let idx = index.y * u32(ubo.computeData[0]) + index.x;
    position_buffer.position[idx] = vec4<f32>(u, w, v, 1.0);
}

