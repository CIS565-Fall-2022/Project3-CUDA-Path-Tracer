#pragma once

#include "intersections.h"

#define STRAT_SAMP 1


// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

#if STRAT_SAMP
    const float sampleSize = 32;
    int squareIdx = u01(rng) * sampleSize * sampleSize;
    float gridLen = (1 / sampleSize);
    
    float stratX = ((squareIdx / sampleSize) + u01(rng)) * gridLen;
    float stratY = ((squareIdx % (int)sampleSize) + u01(rng)) * gridLen;
    
    // CosineWarp
    float up = sqrt(stratX); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = stratY * TWO_PI;
#else
    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;
#endif

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
float FresnelMoment(float eta) {
    float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
        eta5 = eta4 * eta;
    return (eta < 1) ?
        0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
        2.49277f * eta4 - 0.68441f * eta5 :
        -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
        1.27198f * eta4 + 0.12746f * eta5;
}

__host__ __device__
glm::vec3 imperfectSpecularScatter(
    glm::vec3& normal,
    thrust::default_random_engine& rng,
    thrust::uniform_real_distribution<float>& u01,
    float exp) {
    float theta = acos(powf(u01(rng), 1.0 / (exp + 1)));
    float phi = 2.f * PI * u01(rng);
    glm::vec3 untransformedRef(
        cos(phi) * sin(theta),
        sin(phi) * sin(theta),
        cos(theta));

    glm::vec3 tangent = (normal.z == 0 && normal.y == 0) ?
        glm::vec3(-normal.z, 0.f, normal.x) :
        glm::vec3(0.f, normal.z, -normal.y);

    glm::mat3 transform(tangent, glm::cross(normal, tangent), normal);

    return transform * untransformedRef;
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
    
    thrust::uniform_real_distribution<float> uniform(0, 1);

    if (m.absorption > 0) {
        float sceneIOR = 1.0f;
        float matIOR = m.indexOfRefraction;

        bool entering = glm::dot(normal, pathSegment.ray.direction) < 0;

        if (!entering) {
            float dist = glm::length(intersect - pathSegment.ray.origin);
            float rand = uniform(rng);
            if (rand == 0 || m.absorption == 0 || (-log(rand)) / m.absorption > dist) {
                pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
            }
            else {
                intersect = getPointOnRay(pathSegment.ray, rand);
                glm::vec3 newNorm = -pathSegment.ray.direction;
                pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(newNorm, rng));
            }
            pathSegment.color *= dist / (1.f + dist) * m.color;
        }
        else {
            float cosTheta = glm::dot(normal, -pathSegment.ray.direction);
            float c = 1 - 2 * FresnelMoment(sceneIOR / m.indexOfRefraction);

            float n1 = m.indexOfRefraction;
            float n2 = sceneIOR;

            float refAtNorm = powf((n1 - n2) / (n1 + n2), 2.f);
            refAtNorm += (1.f - refAtNorm) * powf(1.f - cosTheta, 5.f);

            if (uniform(rng) < refAtNorm) {
                if (m.specular.exponent < 30 || uniform(rng)) {
                    pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
                }
                else {
                    pathSegment.ray.direction = glm::normalize(imperfectSpecularScatter(normal, rng, uniform, m.specular.exponent));
                }
            }
            else {
                pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(-normal, rng));
            }

        }


        //pathSegment.remainingBounces = 1;
        //pathSegment.color = glm::vec3(0, 1, 1);

        pathSegment.color *= m.color;
        
    }
    else if (m.hasReflective && m.hasRefractive) {
        float sceneIOR = 1.0f;
        float matIOR = m.indexOfRefraction;

        bool entering = glm::dot(normal, pathSegment.ray.direction) < 0;

        float eta = entering ? matIOR / sceneIOR : sceneIOR / matIOR;
        normal = entering ? normal : -normal;
        float n1 = entering ? matIOR : sceneIOR;
        float n2 = entering ? matIOR : sceneIOR;

        float cosTheta = glm::dot(normal, -pathSegment.ray.direction);

        float refAtNorm = powf((n1 - n2) / (n1 + n2), 2.f);
        refAtNorm += (1.f - refAtNorm) * powf(1.f - cosTheta, 5.f);
        //float index = n1 / n2;
        if (uniform(rng) >= refAtNorm) {//refraction

            pathSegment.ray.direction = glm::normalize(glm::refract(glm::normalize(pathSegment.ray.direction), glm::normalize(normal), 1 / eta));
            pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
            pathSegment.color *= m.specular.color;
        }
        else {
            pathSegment.color *= m.specular.color;
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            (glm::normalize(pathSegment.ray.direction) * .0001f);
        }
        
    }
    else if (m.hasReflective) {
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect + (glm::normalize(pathSegment.ray.direction) * .0001f);
    }
    else {
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.color *= m.color;
        pathSegment.ray.origin = intersect + (glm::normalize(pathSegment.ray.direction) * .0001f);
    }
    pathSegment.remainingBounces--;
    
}
