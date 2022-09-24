#pragma once

#include "intersections.h"
#include "utilities.h"

__host__ __device__
bool feq(float a, float b) {
    a -= b;
    return a <= EPSILON && a >= -EPSILON;
}

__host__ __device__
glm::vec3 randomVecInUnitSphere(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 ret(u01(rng), u01(rng), u01(rng));
    return ret / glm::length(ret);
}

__host__ __device__
float reflectance(float cosine, float eta) {
    // Use Schlick's approximation for reflectance.
    // Taken from https://raytracing.github.io/books/RayTracingInOneWeekend.html
    auto r0 = (1 - eta) / (1 + eta);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
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


#define OFFSET_EPS 0.0001f
__host__ __device__
void scatterRay(
    PathSegment& path,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material& m,
    thrust::default_random_engine& rng) {
    
    /*
    ** Heavily based on https://raytracing.github.io/books/RayTracingInOneWeekend.html
    */


    assert(path.remainingBounces > 0);
    assert(feq(m.emittance, 0));
    assert(feq(glm::length(normal), 1));
    assert(feq(glm::length(path.ray.direction), 1));

    Ray& ray = path.ray;
    glm::vec3 dir;
    color_t color = m.color;
    thrust::uniform_real_distribution<float> u01(0,1);

    MaterialType type;    
    
    if (m.hasReflective > 0) {
        if (m.hasRefractive > 0 && m.indexOfRefraction > 0) {
            type = DIELECTIC;
        } else {
            type = METAL;
        }
    } else {
        type = LAMBERT;
    }

    bool finish;
    bool leaving = glm::dot(ray.direction, normal) > 0;
    do {
        finish = true;
        float eta, cos_theta, sin_theta;

        switch (type) {
        case METAL:
            dir = glm::reflect(ray.direction, leaving ? -normal : normal);
            dir += fmax(1 - m.hasReflective, 0.0f) * randomVecInUnitSphere(rng);
            color *= m.specular.color;
            break;
        case DIELECTIC:
            // eta = source / dest
            // so entering ==> air / idx of refraction
            //    leaving ==> idx of refraction / air = idx of refraction
            eta = leaving ?  m.indexOfRefraction : 1 / m.indexOfRefraction;
            cos_theta = fmin(glm::dot(-ray.direction, normal), 1.0f);
            sin_theta = sqrtf(1 - cos_theta * cos_theta);

            // divide reflection/refraction based on how significant the factor is
            if (eta * sin_theta > 1 || reflectance(cos_theta, eta) > m.hasRefractive * u01(rng)) {
                type = METAL;
                finish = false;
            } else {
                dir = glm::refract(ray.direction, leaving ? -normal : normal, eta);
            }
            break;
        case LAMBERT:
        default:
            dir = calculateRandomDirectionInHemisphere(normal, rng);
            break;
        }
    } while (!finish);

    // update path segment
    path.color *= color;
    ray.origin = intersect + OFFSET_EPS * normal;
    ray.direction = glm::normalize(dir);
    --path.remainingBounces;
}