struct HostUniform
{
    modelMatrix: mat4x4<f32>,
    viewMatrix: mat4x4<f32>,
    projectionMatrix: mat4x4<f32>,

    lightAmbient: array<vec4<f32>, 2>,
    lightDiffuse: array<vec4<f32>, 2>,
    lightSpecular: array<vec4<f32>, 2>,
    lightPosition: array<vec4<f32>, 2>,

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
    @location(1) lightDirection0: vec3<f32>,
    @location(2) lightDirection1: vec3<f32>,
    @location(3) viewerVector: vec3<f32>
};

@group(0) @binding(0) var<uniform> ubo : HostUniform;

@fragment
fn main(output:VertexOutput) -> @location(0) vec4<f32>
{
    var phong_ads_light:vec3<f32> = vec3<f32>(0.0);

    if (ubo.lightEnabled.x == 1u)
    {
        var ambient : array<vec3<f32>, 2>;
        var diffuse : array<vec3<f32>, 2>;
        var specular : array<vec3<f32>, 2>;
        var normalized_light_direction : array<vec3<f32>, 2>;
        var reflection_vector : array<vec3<f32>, 2>;

        let normalized_transformed_normals:vec3<f32> = normalize(output.transformedNormals);
        let normalized_viewer_vector:vec3<f32> = normalize(output.viewerVector);

        let lightDirection : array<vec3<f32>, 2> = array<vec3<f32>, 2>(output.lightDirection0, output.lightDirection1);

        for (var i = 0; i < 2; i++)
        {
            normalized_light_direction[i] = normalize(lightDirection[i]);
            reflection_vector[i] = reflect(-normalized_light_direction[i], normalized_transformed_normals);

            ambient[i] = ubo.lightAmbient[i].xyz * ubo.materialAmbient.xyz;
            diffuse[i] = ubo.lightDiffuse[i].xyz * ubo.materialDiffuse.xyz * max(dot(normalized_light_direction[i], normalized_transformed_normals), 0.0);
            specular[i] = ubo.lightSpecular[i].xyz * ubo.materialSpecular.xyz * pow(max(dot(reflection_vector[i], normalized_viewer_vector), 0.0), ubo.materialShininess.x);
        
            phong_ads_light += ambient[i] + diffuse[i] + specular[i];
        }
    }
    else
    {
        phong_ads_light = vec3(1.0, 1.0, 1.0);
    }

    var color = vec4(phong_ads_light, 1.0);

    return color;
}
