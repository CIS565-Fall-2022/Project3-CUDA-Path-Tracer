#pragma once

#include "intersections.h"
#include "utilities.h"

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
__host__ __device__
void scatterRay(
    PathSegment& pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    glm::vec3 textureColor,
    thrust::default_random_engine& rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    pathSegment.isRefrectiveRay = false;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float chance = u01(rng);

    float reflectChance = m.hasReflective + u01(rng) > 1.0001f;

    pathSegment.ray.origin = intersect; // Default origin

    glm::vec3 col = textureColor;

    if (reflectChance && m.hasRefractive)
    {
        // Pick between reflection or refratcion

        thrust::uniform_real_distribution<float> u01(0, 1);
        float chance = u01(rng);

        glm::vec3 inDir = pathSegment.ray.direction;
        float cosTheta = glm::dot(-inDir, normal);  // View dir is the opposite of incident dir

        float n1, n2;
        if (cosTheta > 0)
        {
            // Ray shoots from air
            n1 = m.indexOfRefraction;
            n2 = 1.0f;
        }
        else
        {
            // Ray shoots from material
            normal = -normal;
            n1 = 1.0f;
            n2 = m.indexOfRefraction;
        }

        float R0 = pow((n1 - n2) / (n1 + n2), 2);
        float R = R0 + (1 - R0) * pow(1 - cosTheta, 5); // Reflection coefficient

        glm::vec3 refractDir = glm::refract(inDir, normal, n2 / n1);
        glm::vec3 reflectDir = glm::reflect(inDir, normal);

        if (R > chance || glm::length(refractDir) < EPSILON)
        {
            pathSegment.ray.direction = reflectDir;
            col *= m.specular.color;
            pathSegment.isRefrectiveRay = true;
        }
        else
        {
            pathSegment.ray.direction = refractDir;
            pathSegment.ray.origin += 0.002f * refractDir;
            col *= m.specular.color;
        }
    }
    else if (reflectChance) // Pure reflection
    {
        //printf("reflect!");
        glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.direction = reflectDir;
        col *= m.specular.color;
        pathSegment.isRefrectiveRay = true;
    }
    else // Pure diffuse
    {
        glm::vec3 newDir = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.ray.direction = newDir;
        col *= m.color;
    }

    if (reflectChance && m.hasMetallic > 0.0001f + u01(rng))
    {
        col *= m.color;
    }

    pathSegment.color *= col;
    pathSegment.remainingBounces -= 1;
}

