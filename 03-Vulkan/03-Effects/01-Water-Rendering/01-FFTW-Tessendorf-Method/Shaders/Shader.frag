#version 460 core
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec4 out_position;
layout(location = 1) in vec3 out_normal;
layout(location = 2) in vec2 out_texcoord;

layout(location = 0) out vec4 FragColor;

layout(binding = 1) uniform WaterSurfaceUBO
{
    vec4 cameraPosition;
    vec4 absorptionCoefficient;
    vec4 scatterCoefficient;
    vec4 backScatterCoefficient;
    vec4 terrainColor;
    float height;
    float skyIntensity;
    float specularIntensity;
    float specularHighlights;
} waterUbo;


#define M_PI 3.14159265358979323846
#define ONE_OVER_PI (1.0 / M_PI)

#define WATER_IOR 1.33477
#define AIR_IOR 1.0
#define IOR_AIR2WATER (AIR_IOR / WATER_IOR)

#define NORMAL_WORLD_UP vec3(0.0, 1.0, 0.0)

#define TERRAIN_HEIGHT 0.0f
vec4 kTerrainBoundPlane = vec4(NORMAL_WORLD_UP, TERRAIN_HEIGHT);

struct Ray
{
    vec3 org;
    vec3 dir;
};

/**
 * @brief Finds an intersection point of ray with the terrain
 * @return Distance along the ray; positive if intersects, else negative
 */
float IntersectTerrain(const in Ray ray);

vec3 TerrainNormal(const in vec2 p);

vec3 TerrainColor(const in vec2 p);


float FresnelFull(in float theta_i, in float theta_t)
{
    if (theta_i <= 0.00001)
    {
        const float at0 = (WATER_IOR-1.0) / (WATER_IOR+1.0);
        return at0*at0;
    }

    const float t1 = sin(theta_i-theta_t) / sin(theta_i+theta_t);
    const float t2 = tan(theta_i-theta_t) / tan(theta_i+theta_t);
    return 0.5 * (t1*t1 + t2*t2);
}

vec3 Attenuate(const in float kDistance, const in float kDepth)
{
    const float kScale = 0.1;
    return exp(-waterUbo.absorptionCoefficient.xyz * kScale * kDistance 
               -waterUbo.scatterCoefficient.xyz * kScale * kDepth);
}

vec3 RefractAirIncident(const in vec3 kIncident, const in vec3 kNormal)
{
    return refract(kIncident, kNormal, IOR_AIR2WATER);
}

vec3 ComputeTerrainRadiance(const in vec3 p_g, const in vec3 kIncidentDir)
{
    return TerrainColor(p_g.xz) * dot(TerrainNormal(p_g.xz), -kIncidentDir);
}

vec3 ComputeWaterSurfaceColor(
    const in Ray ray,
    const in vec3 p_w,
    const in vec3 kNormal)
{
    // Reflection of camera ray
    const vec3 kReflectDir = reflect(ray.dir, kNormal);

    // Diffuse lighting (removed sky and sun-related terms)
    vec3 L_a = waterUbo.absorptionCoefficient.xyz * max(dot(kNormal, normalize(ray.dir)), 0.0);

    // Direction to camera
    const vec3 kViewDir = vec3(-ray.dir);

    // Blinn-Phong specular reflection (removed sun-related reflection)
    const vec3 kHalfWayDir = normalize(kViewDir);
    const float specular = waterUbo.absorptionCoefficient.x *
            clamp(
                pow(
                    max(dot(kNormal, kHalfWayDir), 0.0),
                1.0)
            , 0.0, 1.0);

    const vec3 L_s = vec3(specular, specular, specular);

    const vec3 kRefractDir = RefractAirIncident(-kViewDir, kNormal);

    // Light just below the waterUbo transmitted through into the air
    vec3 L_u;
    {
        // Downwelling irradiance just below the water waterUbo
        const vec3 E_d0 = M_PI * L_a;

        // Constant diffuse radiance just below the waterUbo
        const vec3 L_df0 = (0.33 * waterUbo.backScatterCoefficient.xyz) /
                            waterUbo.absorptionCoefficient.xyz * (E_d0 * ONE_OVER_PI);

        const Ray kRefractRay = Ray(p_w, kRefractDir);
        const float t_g = IntersectTerrain(kRefractRay);
        const vec3 p_g = kRefractRay.org + t_g * kRefractRay.dir;
        const vec3 L_g = ComputeTerrainRadiance(p_g, kRefractRay.dir);

        L_u = Attenuate(t_g, 0.0) * L_g +
              (1.0 - Attenuate(t_g, abs(p_w.y - p_g.y))) * L_df0;
    }

    // Fresnel reflectivity to the camera
    float F_r = FresnelFull(dot(kNormal, kViewDir), dot(-kNormal, kRefractDir));

    return F_r * (L_s + L_a) + (1.0 - F_r) * L_u;
}

// =============================================================================
// Terrain functions

float Fbm4Noise2D(in vec2 p);

float TerrainHeight(const in vec2 p)
{
    return TERRAIN_HEIGHT - 8.f * Fbm4Noise2D(p.yx * 0.02f);
}

vec3 TerrainNormal(const in vec2 p)
{
    const vec2 kEpsilon = vec2(0.0001, 0.0);

    return normalize(
        vec3(
            // x + offset
            TerrainHeight(p - kEpsilon.xy) - TerrainHeight(p + kEpsilon.xy), 
            10.0f * kEpsilon.x,
            // z + offset
            TerrainHeight(p - kEpsilon.yx) - TerrainHeight(p + kEpsilon.yx)
        )
    );
}

vec3 TerrainColor(const in vec2 p)
{
    float n = clamp(Fbm4Noise2D(p.yx * 0.02 * 2.), 0.6, 0.9);

    return n * waterUbo.terrainColor.xyz;
}

float IntersectTerrain(const in Ray ray)
{
    return -( dot(ray.org, kTerrainBoundPlane.xyz) - kTerrainBoundPlane.w ) /
           dot(ray.dir, kTerrainBoundPlane.xyz);
}

// =============================================================================
// Noise functions

float hash1(in vec2 i)
{
    i = 50.0 * fract( i * ONE_OVER_PI );
    return fract( i.x * i.y * (i.x + i.y) );
}

float Noise2D(const in vec2 p)
{
    vec2 i = floor(p);
    vec2 f = fract(p);

    #ifdef INTERPOLATION_CUBIC
    vec2 u = f*f * (3.0-2.0*f);
    #else
    vec2 u = f*f*f * (f * (f*6.0-15.0) +10.0);
    #endif
    float a = hash1(i + vec2(0,0) );
    float b = hash1(i + vec2(1,0) );
    float c = hash1(i + vec2(0,1) );
    float d = hash1(i + vec2(1,1) );

    return -1.0 + 2.0 * (a + 
                         (b - a) * u.x + 
                         (c - a) * u.y + 
                         (a - b - c + d) * u.x * u.y);
}

const mat2 MAT345 = mat2( 4./5., -3./5.,
                          3./5.,  4./5. );

float Fbm4Noise2D(in vec2 p)
{
    const float kFreq = 2.0;
    const float kGain = 0.5;
    float amplitude = 0.5;
    float value = 0.0;

    for (int i = 0; i < 4; ++i)
    {
        value += amplitude * Noise2D(p);
        amplitude *= kGain;
        p = kFreq * MAT345 * p;
    }
    return value;
}



void main(void)
{
    // Code
    // const Ray ray = Ray(waterUbo.cameraPosition.xyz, normalize(out_position.xyz - waterUbo.cameraPosition.xyz));
    // const vec3 kNormal = normalize(out_normal);

    // const vec3 pw = vec3(out_position.x, out_position.y + waterUbo.height, out_position.z);
    // vec3 color = ComputeWaterSurfaceColor(ray, pw, kNormal);

    // if (out_position.w < 0.0)
    //     color = vec3(1.0);

    // FragColor = vec4(color, 1.0);

    // // Tone mapping
    // FragColor = vec4(1.0) - exp(-FragColor * 2.0f);

    FragColor = vec4(1.0, 1.0, 1.0, 1.0);
}
