#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */

#define PROCEDURAL 1

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
        glm::vec3* texData) {
    
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
        }//what if both of the surface is not air?(TODO)

        float R0 = glm::pow((n1 - n2) / (n1 + n2), 2.0f);
        float Rtheta = R0 + (1 - R0) * glm::pow(1 - glm::dot(-incident, normal), 5.0f);
        //now I use R value from Schlink approximation and fresnel equations to decide whether reflect or refract
        // > Rtheta then refraction, else reflection
        float index = n1 / n2;
        if (u01(rng) >= Rtheta) {//refraction
            glm::vec3 refract_dir = glm::refract(incident, normal, 1.f / index);
            //if (glm::length(refract_dir) == 0.f) {
            //    pathSegment.ray.direction = glm::reflect(incident, normal);
            //    pathSegment.ray.origin = intersect;
            //    pathSegment.color *= m.specular.color;
            //}
            //else {//refraction
                pathSegment.ray.direction = glm::normalize(refract_dir);
                pathSegment.ray.origin = intersect + 0.002f * pathSegment.ray.direction;
                pathSegment.color *= m.specular.color;
            //}
        }
        else {
            pathSegment.color *= m.specular.color;
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect;
        }
        //

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
    //procedural
#if PROCEDURAL
    if (m.tex.TexIndex >= 0) {
        pathSegment.color *= checkerBoard(uv);
    }
#endif // 


}
