#pragma once

#include "intersections.h"

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

// Scatter a ray with some probabilities according to the material properties
//__host__
__device__
void scatterRay(
        PathSegment& pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        bool outside, // ray coming from outside? 
        const Material &m,
        thrust::default_random_engine &rng) {
    const float ACNE_BIAS = 3e-3f; //TODO: this is a very annoying parameter to tune. solutions?
    thrust::uniform_real_distribution<float> u01(0, 1); // [a, b)

    // normalizes might not be necesary (?)
    pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
    normal = glm::normalize(normal);

    if (m.matType == MatType::DIFFUSE) {
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
    }
    else if (m.matType == MatType::SPECULAR) {
        // from gpu gems 3 ch 20, eq 7, 8 & 9
        float theta = acos(powf(u01(rng), 1.0 / (m.specular.exponent + 1)));
        float phi = 2.f * PI * u01(rng);
        glm::vec3 untransformedRef(
            cos(phi) * sin(theta),
            sin(phi) * sin(theta),
            cos(theta));

        glm::vec3 tangent = (normal.z == 0 && normal.y == 0) ? 
            glm::vec3(-normal.z, 0.f, normal.x) : 
            glm::vec3(0.f, normal.z, -normal.y);
        glm::mat3 transform(tangent, glm::cross(normal, tangent), normal);
        pathSegment.ray.direction = transform * untransformedRef;
    }
    else if (m.matType == MatType::DIELECTRIC) { // Fresnel eqns
        // use below to check if normal is against ray direction; traps if not.
        // intersections.h should take care of all need to align normals and whatnot
        // if (glm::dot(normal, pathSegment.ray.direction) >= 0.f) { __trap(); }
        const float sceneIOR = 1.0f; // assume the scenes will be in AIR ie IOR = 1.0
        float cosThetaI = glm::dot(normal, -pathSegment.ray.direction);

        // Schlick approximation
        float reflectivityAtNormal =
            powf((m.indexOfRefraction - sceneIOR) / (m.indexOfRefraction + sceneIOR), 2.f);
        float reflectivityAtRay = reflectivityAtNormal + (1.f - reflectivityAtNormal) * powf(1.f - cosThetaI, 5.f);

        if (u01(rng) < reflectivityAtRay) { // reflect
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else { // refraction. code adapted from PBRT refract bsdf
            float eta = outside ? sceneIOR / m.indexOfRefraction : m.indexOfRefraction / sceneIOR;

            float sin2ThetaI = glm::max(0.f, 1.f - cosThetaI * cosThetaI);
            float sin2ThetaT = eta * eta * sin2ThetaI;
            if (sin2ThetaT >= 1.0) { // total internal reflection
                pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            } else {
                float cosThetaT = std::sqrt(1 - sin2ThetaT);
                pathSegment.ray.direction = eta * pathSegment.ray.direction +
                    (eta * cosThetaI - cosThetaT) * normal;
            }
        }
    }
    // TODO: more types

    // bias origin to avoid shadow acne.
    // use ray direction over normal, since some rays are trying to "pass" the material barrier
    pathSegment.ray.origin = intersect + ACNE_BIAS * pathSegment.ray.direction;
    pathSegment.color *= m.color;
}
