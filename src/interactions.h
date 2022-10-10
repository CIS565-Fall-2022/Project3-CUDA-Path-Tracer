#pragma once

#include "intersections.h"
#include "cuda_runtime.h"
#include <math.h>

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */


__device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */


//Schilick
__device__ 
float fresnelReflectance(double cosTheta,double ref_index)
{
    auto r0 = (1 - ref_index) / (1 + ref_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosTheta), 5);
}

__device__ double lengthSquared(glm::vec3 vec)
{
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
}

//refract
__device__ glm::vec3 refractEvaluation(const glm::vec3&dir,const glm::vec3& n,float etai_over_etat)
{
    float cosTheta = fmin(glm::dot(-dir, n), 1.0f);
    glm::vec3 r_out_perp = etai_over_etat * (dir + cosTheta * n);
    glm::vec3 r_out_parallel = (float)-sqrt(fabs(1.0 - lengthSquared(r_out_perp))) * n;
    return r_out_perp + r_out_parallel;
}

__device__ glm::vec3 dielectricEvaluation
(PathSegment& pathSegment, glm::vec3 intersect, 
    glm::vec3 normal, const Material& m, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);
    normal = glm::normalize(normal);
    //dot product smaller than 0, means ray intersects with out face
    //else: inner face
    bool outsideFace = glm::dot(pathSegment.ray.direction, normal) < 0.0f;
    double refract_ratio = outsideFace ? (1.0 / m.indexOfRefraction) : m.indexOfRefraction;

    glm::vec3 unit_rayDir = glm::normalize(pathSegment.ray.direction);
    double cosTheta = fmin(glm::dot(-unit_rayDir, normal), 0.0f);
    double sinTheta = sqrt(1.0f - pow(cosTheta, 2));

    bool not_refractable = (refract_ratio * sinTheta) > 1.0f;
    glm::vec3 direction;
    if (not_refractable || fresnelReflectance(cosTheta, refract_ratio) > u01(rng))
    {
        float scale = m.hasRefractive <= 0.0 ? 0.0 : 1.0/(1-m.hasRefractive);
        direction = glm::reflect(unit_rayDir, normal);
        pathSegment.color *= m.color;
    }
    else
    {
        //refract
        float scale = m.hasRefractive <= 0.0 ? 0.0 : 1.0 / (1 - m.hasRefractive);
        direction = refractEvaluation(unit_rayDir,normal,u01(rng));
        if (m.specular.exponent > 0)
        {
            pathSegment.color *= m.specular.color;
        }
    }
    return glm::normalize(direction);
}

__device__ void normalMapping(glm::vec3& n, glm::vec3& normalMap,glm::vec4 tangent)
{
    glm::vec3 t = glm::vec3(tangent);
    glm::vec3 b = glm::cross(n, t) * tangent.w;
    glm::mat3 TBN = glm::mat3(t, b, n);
    n = glm::normalize(TBN*normalMap);
}


//sampling the texture object with the given UV coordinate

__device__ glm::vec3 sampleTexture(cudaTextureObject_t texObj, const glm::vec2 uv) {
    float4 rgba = tex2D<float4>(texObj, uv.x, uv.y);
    return glm::vec3(rgba.x, rgba.y, rgba.z);
}


__device__
void scatterRay(
        PathSegment & pathSegment,
        const ShadeableIntersection& i,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        cudaTextureObject_t* textures, 
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    thrust::uniform_real_distribution<float> u01(0, 1);
    //use this act as pdf
    glm::vec3 nextRayDir;;
    float eta;
    // Specular BRDF/BTDF here

    glm::vec3 albedo;
    glm::vec3 color ;
    float metallic;
    float roughness;

    normal = i.surfaceNormal;
    //Add Support for gltf attribute shading
    int textureID = -1;
    textureID = m.texOffset + m.pbrShadingAttribute.baseColorTexture.index;


    //printf("texture Offset: %d \n", m.pbrShadingAttribute.baseColorTexture.index);

    if (textureID < 0)
    {
        albedo = m.pbrShadingAttribute.baseColor;
    }
    else
    {
        albedo = sampleTexture(textures[textureID], i.uv);
    }
    //Metallic and Roughness Texture
    textureID = m.texOffset + m.pbrShadingAttribute.metallicRoughnessTexture.index;


    if (textureID < 0)
    {
        metallic = m.pbrShadingAttribute.metallicFactor;
        roughness = m.pbrShadingAttribute.roughnessFactor;
    }
    else
    {
        glm::vec3 pbrCol = sampleTexture(textures[textureID], i.uv);
        metallic = pbrCol.b * m.pbrShadingAttribute.metallicFactor;
        roughness = pbrCol.g * m.pbrShadingAttribute.roughnessFactor;
    }

    //Apply normal Map if have
    textureID = m.texOffset + m.normalTexture.index;
    if (textureID >= 0)
    {
        //have normal texture
        glm::vec3 normalMap = sampleTexture(textures[textureID], i.uv);
        normalMap = glm::normalize(normal*2.f-1.f);
        normalMapping(normal, normalMap, i.tangent);
    }

   // printf("Debug msg albedo: %f %f %f \n", albedo.x, albedo.y, albedo.z);

    float random = u01(rng);
        if (random <= m.hasReflective)
        {
            nextRayDir = glm::reflect(pathSegment.ray.direction, normal);
            if (m.specular.exponent > 0 && random > 0.5)
            {
                pathSegment.color *= m.specular.color;
            }
            else
            {
                pathSegment.color *= m.color;
            }
        }
        else if (random <= m.hasReflective + m.hasRefractive)
        {
            nextRayDir = dielectricEvaluation(pathSegment, intersect, normal, m, rng);
        }
        else
        {
            nextRayDir = calculateRandomDirectionInHemisphere(normal, rng);
            pathSegment.color *= m.color ;
        }



    pathSegment.ray.origin = intersect + 0.0001f * nextRayDir;
    pathSegment.ray.direction = nextRayDir;
}


