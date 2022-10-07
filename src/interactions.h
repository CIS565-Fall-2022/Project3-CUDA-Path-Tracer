#pragma once

#include "intersections.h"
#include "stb_image.h"
#include "cuda_runtime.h"

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
 * - Pick the split based on the intensity of each material color that you hit, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
#if USE_UV
__device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        int texid,
        glm::vec2 uv,
        const Material &m,
        // cudaArray_t &tex,
        cudaTextureObject_t &texObject,
        int numChannels,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    // treat the rest as perfectly specular.
    // assuming there's only one light

    // based on float value of reflective and refractive
    // with float percent likelihood the ray goes reflective vs. refractive

    thrust::uniform_real_distribution<float> u01(0, 1);

    float randGen = u01(rng);

    glm::vec3 pointColor;

    if (texid != -1) {
        float u = uv[0];
        float v = uv[1];


        float4 finalcol = tex2D<float4>(texObject, u, v);
        pointColor = glm::vec3(finalcol.x, finalcol.y, finalcol.z);

    }
    else {
        pointColor = m.color;
    }

    if (randGen <= m.hasReflective) {
        // take a reflective ray
        glm::vec3 newDirection = glm::reflect(pathSegment.ray.direction, normal);
        Ray newRay = {
            intersect,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            m.specular.color * pointColor * pathSegment.color * m.hasReflective,
            pathSegment.pixelIndex,
            pathSegment.remainingBounces
        };

        pathSegment = newPath;
    }
    else if (randGen <= m.hasReflective + m.hasRefractive) {
        // take a refractive ray
        float airIOR = 1.0f;
        float eta = airIOR / m.indexOfRefraction;

        float cosTheta = glm::dot(-1.f * pathSegment.ray.direction, normal);

        // then entering
        bool entering = cosTheta > 0;

        if (!entering) {
            eta = 1.0f / eta; // invert eta
        }

        float sinThetaI = sqrt(1.0 - cosTheta * cosTheta);
        float sinThetaT = eta * sinThetaI;

        glm::vec3 newDirection = pathSegment.ray.direction;

        // if total internal reflection
        if (sinThetaT >= 1) {
            newDirection = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        }
        else {
            newDirection = glm::normalize(glm::refract(pathSegment.ray.direction, normal, eta));
        }

        glm::vec3 newColor = pathSegment.color * pointColor * m.specular.color;

        Ray newRay = {
            intersect + 0.001f * pathSegment.ray.direction,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            newColor,
            pathSegment.pixelIndex,
            pathSegment.remainingBounces
        };

        pathSegment = newPath;
    }
    else {
        // only diffuse
        glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        Ray newRay = {
            intersect,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            pointColor * pathSegment.color,
            pathSegment.pixelIndex,
            pathSegment.remainingBounces
        };

        pathSegment = newPath;
    }
}

#else
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
    // treat the rest as perfectly specular.
    // assuming there's only one light

    // based on float value of reflective and refractive
    // with float percent likelihood the ray goes reflective vs. refractive

    thrust::uniform_real_distribution<float> u01(0, 1);

    float randGen = u01(rng);

    if (randGen <= m.hasReflective) {
        // take a reflective ray
        glm::vec3 newDirection = glm::reflect(pathSegment.ray.direction, normal);
        Ray newRay = {
            intersect,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            m.specular.color * m.color * pathSegment.color * m.hasReflective,
            pathSegment.pixelIndex,
            pathSegment.remainingBounces
        };

        pathSegment = newPath;
    }
    else if (randGen <= m.hasReflective + m.hasRefractive) {
        // take a refractive ray
        float airIOR = 1.0f;
        float eta = airIOR / m.indexOfRefraction;

        float cosTheta = glm::dot(-1.f * pathSegment.ray.direction, normal);

        // then entering
        bool entering = cosTheta > 0;

        if (!entering) {
            eta = 1.0f / eta; // invert eta
        }

        float sinThetaI = sqrt(1.0 - cosTheta * cosTheta);
        float sinThetaT = eta * sinThetaI;

        glm::vec3 newDirection = pathSegment.ray.direction;

        // if total internal reflection
        if (sinThetaT >= 1) {
            newDirection = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
        }
        else {
            newDirection = glm::normalize(glm::refract(pathSegment.ray.direction, normal, eta));
        }

        glm::vec3 newColor = pathSegment.color * m.color * m.specular.color;

        Ray newRay = {
            intersect + 0.001f * pathSegment.ray.direction,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            newColor,
            pathSegment.pixelIndex,
            pathSegment.remainingBounces
        };

        pathSegment = newPath;
    }
    else {
        // only diffuse
        glm::vec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);
        Ray newRay = {
            intersect,
            newDirection
        };

        glm::vec3 pointColor;

        pointColor = m.color;

        PathSegment newPath = {
            newRay,
            pointColor * pathSegment.color,
            pathSegment.pixelIndex,
            pathSegment.remainingBounces
        };

        pathSegment = newPath;
    }
}
#endif