struct HostUniform
{
    modelMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,
    lightAmbient: vec4<f32>,
    lightDiffuse: vec4<f32>,
    lightSpecular: vec4<f32>,
    lightPosition: vec4<f32>,
    materialAmbient: vec4<f32>,
    materialDiffuse: vec4<f32>,
    materialSpecular: vec4<f32>,
    materialShininess: vec4<f32>,
    lightEnabled: vec4<u32>
};

struct VertexOutput
{
    @builtin(position) position: vec4<f32>,
    @location(0) transformedNormals: vec3<f32>,
    @location(1) lightDirection: vec3<f32>,
    @location(2) viewerVector: vec3<f32>
};

@group(0) @binding(0) var<uniform> ubo : HostUniform;

@fragment
fn main(output:VertexOutput) -> @location(0) vec4<f32>
{
    var phong_ads_light:vec3<f32> = vec3<f32>(1.0);

    if (ubo.lightEnabled.x == 1u)
    {
        let normalized_transformed_normals:vec3<f32> = normalize(output.transformedNormals);
        let normalized_light_direction:vec3<f32> = normalize(output.lightDirection);
        let normalized_viewer_vector:vec3<f32> = normalize(output.viewerVector);
        let reflectionVector:vec3<f32> = reflect(-normalized_light_direction, normalized_transformed_normals);

        let ambient:vec3<f32> = ubo.lightAmbient.xyz * ubo.materialAmbient.xyz;
        let diffuse:vec3<f32> = ubo.lightDiffuse.xyz * ubo.materialDiffuse.xyz * max(dot(normalized_light_direction, normalized_transformed_normals), 0.0);
        let specular:vec3<f32> = ubo.lightSpecular.xyz * ubo.materialSpecular.xyz * pow(max(dot(reflectionVector, normalized_viewer_vector), 0.0), ubo.materialShininess.x);
    
        phong_ads_light = ambient + diffuse + specular;
    }
    else
    {
        phong_ads_light = vec3(1.0, 1.0, 1.0);
    }

    var color = vec4(phong_ads_light, 1.0);

    return color;
}
