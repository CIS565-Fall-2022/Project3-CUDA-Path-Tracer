#pragma once

#include "intersections.h"

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

__host__ __device__
float schlick_approx(float eta, float cosAngle)
{
    float R_0 = (1.f - eta) / (1.f + eta);
    R_0 = R_0 * R_0;
    float p = (1 - cosAngle);
    p = p * p * p * p * p;
    return R_0 + (1 - R_0) * p;
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
__host__ __device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        bool outside,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.


    thrust::uniform_real_distribution<float> u01(0, 1);
    float probForRefl = u01(rng);
    thrust::default_random_engine rng_second(utilhash(probForRefl * 1000));
    float probForRefr = u01(rng_second);

    //test for refract
    if (m.hasRefractive > 0)
    {
        if (m.hasRefractive > probForRefr) // if we gives a value between 0-1 to REFR, we can blend refract with extra reflec or diffuse
        {
            glm::vec3 wo = glm::normalize(pathSegment.ray.direction); // just in case
            glm::vec3 nor = normal;

            bool side = glm::dot(wo, nor) < 0.0f;
            nor = side ? nor : -nor;// if ray comes out from inside, reverse the normal
            float ratio = side ? 1.0f / m.indexOfRefraction : m.indexOfRefraction;
            float cos_theta = glm::dot(-wo, nor);
            float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

            float approx = schlick_approx(ratio, cos_theta);

            if (approx > probForRefr || sin_theta * ratio > 1.0f)
            {
                // reflect due to schlick or total reflection
                pathSegment.ray.direction = glm::normalize(glm::reflect(wo, nor));
                pathSegment.color *= m.specular.color;
                pathSegment.ray.origin = intersect + 0.001f * nor;
            }
            else
            {
                //refract
                pathSegment.ray.direction = glm::normalize(glm::refract(wo, nor, ratio));
                pathSegment.color *= m.color;
                pathSegment.ray.origin = intersect - 0.001f * nor;
            }

            return;
        }
    }

    //test for reflect
    if (m.hasReflective > 0)
    {
        if (m.hasReflective > probForRefl)
        {
            // do reflect ray
            pathSegment.color *= m.specular.color;
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.origin = intersect;
            return;
        }
    }
    //diffuse
    pathSegment.color *= m.color;
    pathSegment.ray.origin = intersect;
    pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    
}


