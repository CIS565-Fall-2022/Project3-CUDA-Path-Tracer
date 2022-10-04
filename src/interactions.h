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

__device__ float FresnelMoment(float eta) {
    float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
        eta5 = eta4 * eta;
    return (eta < 1) ?
        0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
        2.49277f * eta4 - 0.68441f * eta5 :
        -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
        1.27198f * eta4 + 0.12746f * eta5;
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

    //if m.material is diffuse do the following
    if (m.hasRefractive != 0) {
        float IOR = 1.0f; 

        float cosTheta = glm::dot(normal, -pathSegment.ray.direction);

        float refAtNorm = powf((m.indexOfRefraction - IOR) / (m.indexOfRefraction + IOR), 2.f);
        refAtNorm += (1.f - refAtNorm) * powf(1.f - cosTheta, 5.f);

        /*if (uniform(rng) < refAtNorm) {
            pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
            pathSegment.color *= m.specular.color;
        }
        else {*/
            float eta = cosTheta > 0 ? IOR / m.indexOfRefraction : m.indexOfRefraction / IOR; 
            normal = cosTheta < 0 ? -normal : normal;
            float sinThetaI = glm::max(0.f, 1.f - cosTheta * cosTheta);
            float sinThetaT = eta * sinThetaI;
            //if (sinThetaT > 1.0) { //internal reflection
            //    pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
            //    pathSegment.color *= m.color;
            //}
            //else { //refraction
                pathSegment.ray.direction = glm::normalize(glm::refract(glm::normalize(pathSegment.ray.direction), glm::normalize(normal), uniform(rng)));
                pathSegment.color *= m.specular.color;
            //}
        //}
        
    }
    else if (m.hasReflective != 0) {
        pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.color *= m.specular.color;
    }
    else {
        pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.color *= m.color;
    }
    pathSegment.ray.origin = intersect + (pathSegment.ray.direction * .0001f);
    pathSegment.remainingBounces--;
    
    

}
