#pragma once

#include "intersections.h"
#include "cuda_runtime.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */

#define PROCEDURAL 1

//for sampling for cudaObject
__device__ glm::vec3 sample(cudaTextureObject_t tex, glm::vec2 const& uv) {
    auto color = tex2D<float4>(tex, uv.x, uv.y);
    return glm::vec3(color.x, color.y, color.z);
}
__host__ __device__
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

__host__ __device__ glm::vec3 checkerBoard(const glm::vec2 uv) {
    glm::vec2 v = 5.0f * uv;
    return (int(v.x) + int(v.y)) % 2 == 0 ? glm::vec3(0.f) : glm::vec3(1.f);
}

__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        glm::vec2 uv,
        const Material &m,
        thrust::default_random_engine &rng,
        const Texture* textures,
        glm::vec3* texData,
        cudaTextureObject_t* cudaTex) {
    
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (pathSegment.remainingBounces == 0) return;
    thrust::uniform_real_distribution<float> u01(0, 1);


    if (!m.hasReflective && !m.hasRefractive) {
        pathSegment.color *= m.color;
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
    }
    else if (m.hasReflective && m.hasRefractive) {//both and we will use the equation
        float n1, n2;
        glm::vec3 incident = pathSegment.ray.direction;//incident vector
        float cosineR = glm::dot(incident, normal);//outside or inside
        if (cosineR > 0) {
            normal = -normal;//reverse the normal if outside
            n1 = 1.0f;
            n2 = m.indexOfRefraction;
        }
        else {//inside
            n1 = m.indexOfRefraction;
            n2 = 1.0f;
        }

        float R0 = glm::pow((n1 - n2) / (n1 + n2), 2.0f);
        float Rtheta = R0 + (1 - R0) * glm::pow(1 - glm::dot(-incident, normal), 5.0f);
        //now I use R value from Schlink approximation and fresnel equations to decide whether reflect or refract
        // > Rtheta then refraction, else reflection
        float index = n1 / n2;
        if (u01(rng) >= Rtheta) {//refraction
            glm::vec3 refract_dir = glm::refract(incident, normal, 1.f / index);
            pathSegment.ray.direction = glm::normalize(refract_dir);
            pathSegment.ray.origin = intersect + 0.002f * pathSegment.ray.direction;
            pathSegment.color *= m.specular.color;
        }
        else {//reflection
            pathSegment.color *= m.specular.color;
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect;
        }
    }
    else if (m.hasReflective) {//reflection only
        pathSegment.color *= m.specular.color;
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect;
    }
    //texture mapping
    if (m.tex.TexIndex >= 0) {
        int w = m.tex.width;
        int x = uv.x * (w - 1);
        int y = uv.y * (m.tex.height - 1);
        //pathSegment.color *= textures->image[m.tex.TexIndex + y * w + x];
        pathSegment.color *= texData[m.tex.TexIndex + y * w + x];
    }
    //procedural for test
#if PROCEDURAL
    if (m.tex.TexIndex >= 0) {
        pathSegment.color *= checkerBoard(uv);
    }
#endif


}

//Based on: https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
//referenced from Wayne Wu
__host__ __device__
glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, glm::vec3 reflect, glm::vec4 tangent,
    thrust::default_random_engine& rng,
    float roughness) {

    thrust::uniform_real_distribution<float> u01(0, 1);
    float x1 = u01(rng);
    float x2 = u01(rng);

    float theta = atan(roughness * sqrt(x1) / sqrt(1 - x1));
    float phi = 2 * PI * x2;

    glm::vec3 dir;
    dir.x = cos(phi) * sin(theta);
    dir.y = sin(phi) * sin(theta);
    dir.z = cos(theta);

    glm::mat3 worldToLocal;
    worldToLocal[2] = normal;
    worldToLocal[1] = glm::vec3(tangent);  // t
    worldToLocal[0] = glm::cross(normal, worldToLocal[1]) * tangent.w;  // b

    glm::vec3 r = glm::normalize(worldToLocal * reflect);

    /// construct an under-constrained coordinate using reflection as up axis
    glm::mat3 sampleToLocal;
    sampleToLocal[2] = r;
    sampleToLocal[0] = glm::normalize(glm::vec3(0, r.z, -r.y));
    sampleToLocal[1] = glm::cross(sampleToLocal[2], sampleToLocal[1]);

    glm::mat3 localToWorld = glm::inverse(worldToLocal);
    glm::mat3 sampleToWorld = localToWorld * sampleToLocal;

    dir = glm::normalize(sampleToWorld * dir);

    return dir;
}

__device__
void scatterRayGLTF(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    glm::vec2 uv,
    glm::vec4 tangent,
    const Material& m,
    thrust::default_random_engine& rng,
    cudaTextureObject_t* textures,
    glm::vec3* texData) {//this function has help and is referenced from our TA Wayne Wu

    if (pathSegment.remainingBounces == 0) return;
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 color;
    glm::vec3 newDir;

    // PBR Material Properties
    glm::vec3 baseColor; 
    float metal, roughness;

    //Get Basic Color
    int txId = -1;
    txId = m.texIndex + m.pbrMetallicRoughness.baseColorTexture.index;
    if (txId < 0) {
        baseColor = m.pbrMetallicRoughness.baseColorFactor;
    }
    else {
        baseColor = sample(textures[txId], uv);
    }

    //Check Metallic and Roughness
    txId = m.texIndex + m.pbrMetallicRoughness.metallicRoughnessTexture.index;
    if (txId < 0) {
        metal = m.pbrMetallicRoughness.metallicFactor;
        roughness = m.pbrMetallicRoughness.roughnessFactor;
    }
    else {
        glm::vec3 pbr = sample(textures[txId], uv);
        metal = pbr.b * m.pbrMetallicRoughness.metallicFactor;
        roughness = pbr.g * m.pbrMetallicRoughness.roughnessFactor;
    }

    //Check Normal Map
    txId = m.texIndex + m.normalTexture.index;
    if (txId >= 0) {
        glm::vec3 n = sample(textures[txId], uv);
        n = glm::normalize(n * 2.f - 1.f);
        glm::vec3 tan = glm::vec3(tangent);
        glm::vec3 bitan = glm::cross(normal, tan) * tangent.w;
        glm::mat3 tbn = glm::mat3(tan, bitan, normal);
        normal = glm::normalize(tbn * normal);
    }

    //texture mapping
    //if (m.tex.TexIndex >= 0) {
    //    int w = m.tex.width;
    //    int x = uv.x * (w - 1);
    //    int y = uv.y * (m.tex.height - 1);
    //    //pathSegment.color *= textures->image[m.tex.TexIndex + y * w + x];
    //    pathSegment.color *= texData[m.tex.TexIndex + y * w + x];
    //}

    if (u01(rng) < metal) {
        // Specular
        glm::vec3 reflect = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.direction = calculateImperfectSpecularDirection(normal, reflect, tangent, rng, roughness);
        color = metal * baseColor;
    }
    else {
        // Diffuse
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        color = (1.f - metal) * baseColor;
    }

    pathSegment.ray.origin = intersect;
    pathSegment.color *= color;
}
