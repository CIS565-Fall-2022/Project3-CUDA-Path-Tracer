﻿#pragma once

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
float schlickEquation(float n1, float n2, float costheta) {
    float r0 = (n1 - n2) / (n1 + n2);
    r0 = r0 * r0;
    return r0 + (1.f - r0) * glm::pow(1.f - costheta, 5.f);
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
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 newDir = glm::vec3(0, 0, 0);
    if (m.hasReflective) {
        newDir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = newDir;
    }
    else if (m.hasRefractive) {
        thrust::uniform_real_distribution<float> u01(0, 1);
        float ior;
        float etaA = 1.0, etaB = m.indexOfRefraction;
        // Check whether the ray is entering or leaving the object with which it has intersected by
        // comparing the direction of ωi to the direction of the normal
        float costheta = glm::clamp(glm::dot(pathSegment.ray.direction, normal), -1.f, 1.f);
        float sintheta = glm::sqrt(glm::max(0.f, 1.f - costheta * costheta));
        bool entering = costheta < 0.f;
        if (!entering) {
            // here, m.idexOfRefraction is set as etaB, which is the eta of material
            // we assume etaA = 1.0, which is the eta of air
            // if entering, ior = etaA / etaB = 1.0 / m.indexOfRefraction;
            // if leaving, ior = etaB / etaA = m.indexOfRefraction;
            etaB = 1.0;
            etaA = m.indexOfRefraction;
            ior = etaA / etaB;
            normal = -normal;
        }
        else {
            ior = etaA / etaB;
            costheta = glm::abs(costheta);
        }

        glm::vec3 reflectDir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        glm::vec3 refractDir = glm::normalize(glm::refract(pathSegment.ray.direction, normal, ior));
        
        // if the incoming angle is larger than a value, the ray can only be reflected
        bool canRefract = ior * sintheta < 1.f;

        if (canRefract) {
            if (u01(rng) < schlickEquation(etaB, etaA, costheta)) {
                pathSegment.ray.direction = reflectDir;
                pathSegment.ray.origin = intersect + 0.001f * glm::normalize(normal);
            }
            else {
                pathSegment.ray.direction = refractDir;
                pathSegment.ray.origin = intersect + 0.001f * glm::normalize(refractDir);
            }
        }
        else {
            pathSegment.ray.direction = reflectDir;
            pathSegment.ray.origin = intersect + 0.001f * glm::normalize(normal);
        }
    }
    else {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = newDir;
    }
}
