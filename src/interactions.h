#pragma once

#include "intersections.h"

// schlick's law
__host__ __device__ 
float schlickRefraction(float eta1, float eta2, float cosTheta) {
    float r0 = (eta1 - eta2) / (eta1 + eta2);
    r0 = powf(r0, 2.0f);
    return r0 + (1.0f - r0) * powf(1.0f - cosTheta, 5.0f);
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
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
// 
//__host__ __device__
//void scatterRay(
//        PathSegment & pathSegment,
//        glm::vec3 intersect,
//        glm::vec3 normal,
//        const Material &m,
//        thrust::default_random_engine &rng) {
//    // TODO: implement this.
//    // A basic implementation of pure-diffuse shading will just call the
//    // calculateRandomDirectionInHemisphere defined above.
//        // base on the material type
//    glm::vec3 newRayDir = glm::vec3(0.0f);
//    if (m.hasReflective > 0.0f) {
//       newRayDir = glm::reflect(pathSegment.ray.direction, normal);
//    }
//    else if (m.hasRefractive > 0.0f) {
//        // newRayDir =
//    }
//    else { // diffuse
//        newRayDir = calculateRandomDirectionInHemisphere(normal, rng);
//    }
//    // update origin to intersection point
//    pathSegment.ray.origin = intersect + 0.001f * normal;
//    // update rayDir
//    pathSegment.ray.direction = newRayDir;
//}


__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    // base on the material type
    glm::vec3 newRayDir = glm::vec3(0.0f);
    // if reflective
    if (m.hasReflective > 0.0f) {
        newRayDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + 0.001f * normal;
    }
    // if reflective
    else if (m.hasRefractive > 0.0f) {
        glm::vec3 normalizedRay = glm::normalize(pathSegment.ray.direction);
        // eta1 (incident ior)
        float eta1 = 1.0f;
        // eta2 (transmited ior)
        float eta2 = m.indexOfRefraction;
        float incidenceFromMaterial = glm::dot(normalizedRay, normal);
        // if ray and normal are not at the same side dot > 0
        if (incidenceFromMaterial > 0.0f) {
            // normal and incident ray should at the same side
            normal = -normal;
            eta1 = m.indexOfRefraction;
            eta2 = 1.0f;
        }
        float cosTheta = fmin(glm::dot(-normalizedRay, normal), 1.0f);
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
        float refractDetermination = eta1 / eta2 * sinTheta;
        // need a random number to decide refract or reflect
        thrust::uniform_real_distribution<float>u01(0, 1);
        float randomNum = u01(rng);
        float schlickRule = schlickRefraction(eta1, eta2, cosTheta);
        if ((refractDetermination > 1.0f) || schlickRule > randomNum) {
            // update ray dir with reflect
            newRayDir = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect + 0.001f * normal;
        }
        else {
            // update ray dir with refract
            newRayDir = glm::refract(pathSegment.ray.direction, normal, eta1 / eta2);
            pathSegment.ray.origin = intersect + 0.001f * glm::normalize(newRayDir);
        }
        pathSegment.color *= m.specular.color;
    }
    // if diffuse
    else { 
        newRayDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + 0.001f * normal;
    }
    pathSegment.ray.direction = glm::normalize(newRayDir);
}