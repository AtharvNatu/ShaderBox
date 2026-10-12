struct HostUniform
{
    modelMatrix : mat4x4<f32>,
    viewMatrix : mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    lightDiffuse: vec4<f32>,
    lightPosition: vec4<f32>,
    materialDiffuse: vec4<f32>,
    lightEnabled: vec4<u32>
};

struct VertexOutput
{
    @builtin(position) position : vec4<f32>,
    @location(0) diffused_light_color : vec3<f32>
};

@group(0) @binding(0) var<uniform> ubo : HostUniform;

// Extract upper 3x3 from a mat4x4
fn mat3FromMat4(m : mat4x4<f32>) -> mat3x3<f32>
{
    return(mat3x3<f32>(m[0].xyz, m[1].xyz, m[2].xyz));
}

// Inverse
fn inverse3x3(m : mat3x3<f32>) -> mat3x3<f32>
{
    let determinant = m[0][0] * (m[1][1]*m[2][2] - m[2][1]*m[1][2]) - 
                      m[1][0] * (m[0][1]*m[2][2] - m[2][1]*m[0][2]) + 
                      m[2][0] * (m[0][1]*m[1][2] - m[1][1]*m[0][2]);

    let inverse_determinant = 1.0 / determinant;

    let column0 = vec3<f32>
    (
        (m[1][1]*m[2][2] - m[2][1]*m[1][2]) * inverse_determinant,
        (m[2][1]*m[0][2] - m[0][1]*m[2][2]) * inverse_determinant,
        (m[0][1]*m[1][2] - m[1][1]*m[0][2]) * inverse_determinant
    );
    let column1 = vec3<f32>
    (
        (m[2][0]*m[1][2] - m[1][0]*m[2][2]) * inverse_determinant,
        (m[0][0]*m[2][2] - m[2][0]*m[0][2]) * inverse_determinant,
        (m[1][0]*m[0][2] - m[0][0]*m[1][2]) * inverse_determinant
    );
    let column2 = vec3<f32>
    (
        (m[1][0]*m[2][1] - m[2][0]*m[1][1]) * inverse_determinant,
        (m[2][0]*m[0][1] - m[0][0]*m[2][1]) * inverse_determinant,
        (m[0][0]*m[1][1] - m[1][0]*m[0][1]) * inverse_determinant
    );

    return(mat3x3<f32>(column0, column1, column2));
};

@vertex
fn main(@location(0) positionIn : vec4<f32>, @location(1) normalIn : vec3<f32>) -> VertexOutput
{
    // Code
    var output : VertexOutput; 

    if (ubo.lightEnabled.x == 1u)
    {
        let eyeCoordinates:vec4<f32> = ubo.viewMatrix * ubo.modelMatrix * positionIn;
        let modelViewMatrix:mat3x3<f32> = mat3FromMat4(ubo.viewMatrix * ubo.modelMatrix);
        let normalMatrix:mat3x3<f32> = transpose(inverse3x3(modelViewMatrix));
        let transformedNormals:vec3<f32> = normalize(normalMatrix * normalIn);
        let lightDirection:vec3<f32> = normalize(ubo.lightPosition.xyz - eyeCoordinates.xyz);

        output.diffused_light_color =  ubo.lightDiffuse.xyz * ubo.materialDiffuse.xyz * max(dot(lightDirection, transformedNormals), 0.0);
    }
    else
    {
        output.diffused_light_color = vec3(1.0, 1.0, 1.0);
    }

    output.position = ubo.projectionMatrix * ubo.viewMatrix * ubo.modelMatrix * positionIn;
    
    return output;
}



