struct MVPUniform
{
    mvpMatrix : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> mvpUniform : MVPUniform;

@vertex
fn main(@location(0) positionIn : vec3<f32>) -> @builtin(position) vec4<f32>
{
    let vPosition = mvpUniform.mvpMatrix * vec4(positionIn, 1.0);
    return vPosition;
}



