#pragma once

#include "intersections.h"
#include "utilities.h"

// #define GLM_REFRACT

__device__
bool feq(float a, float b) {
    a -= b;
    return a <= EPSILON && a >= -EPSILON;
}

__device__
glm::vec3 randomVecInUnitSphere(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 ret(u01(rng), u01(rng), u01(rng));
    return ret / glm::length(ret);
}


// BRDF interface
#define DECL_BRDF(name)\
__device__ float name(glm::vec3 p, glm::vec3 wi, glm::vec3 wo, glm::vec3 normal, Material const& m)


/// <summary>
/// classic phong reflectance model
/// </summary>
/// <returns></returns>
DECL_BRDF(phong_brdf) {
    float kd = 1 - m.hasReflective;
    float ks = m.hasReflective;

    float specular = 0;
    if (ks > 0) {
        specular = ks * powf(glm::dot(wi, glm::reflect(wo, normal)), m.specular.exponent);
    }
    return glm::clamp(kd + specular / glm::dot(wi, normal), 0.0f, 1.0f);
}

//TODO: http://www.codinglabs.net/article_physically_based_rendering_cook_torrance.aspx

__device__
float reflectance(float cosine, float eta) {
    // Use Schlick's approximation for reflectance.
    // Taken from https://raytracing.github.io/books/RayTracingInOneWeekend.html
    auto r0 = (1 - eta) / (1 + eta);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}
__device__
glm::vec3 refract(const glm::vec3& I, const glm::vec3& N, float eta) {
#ifdef GLM_REFRACT
    return glm::refract(I, N, eta);
#else
    // Based on https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
    float cosi = glm::clamp(glm::dot(I, N), -1.0f, 1.0f);
    if (cosi < 0) {
        cosi = -cosi;
    }
    float k = 1 - eta * eta * (1 - cosi * cosi);
    return k < 0 ? glm::vec3(0) : (eta * I + (eta * cosi - sqrtf(k)) * N);
#endif
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__
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
__device__
void scatterRay(
    PathSegment& path,
    glm::vec3 intersect,
    glm::vec3 normal,
    Material const& m,
    Span<Light> const& lights,
    thrust::default_random_engine& rng) {

    /*
    ** Heavily based on
    *  https://raytracing.github.io/books/RayTracingInOneWeekend.html
    *  https://www.pbr-book.org/3ed-2018/Light_Transport_I_Surface_Reflection/Path_Tracing
    */

    assert(path.remainingBounces > 0);
    assert(feq(m.emittance, 0));
    assert(feq(glm::length(normal), 1));
    assert(feq(glm::length(path.ray.direction), 1));

    Ray& ray = path.ray;
    glm::vec3 wi;
    glm::vec3 wo = -ray.origin;
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

        // NOTE: m.hasReflective controls how "fuzzy" the reflection is
        //       m.hasRefractive controls how likely the ray is refracted

        switch (type) {
        case METAL:
            wi = glm::reflect(ray.direction, leaving ? -normal : normal);
            wi += fmax(1 - m.hasReflective, 0.0f) * randomVecInUnitSphere(rng);
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
                wi = refract(ray.direction, leaving ? -normal : normal, eta);
            }
            break;
        case LAMBERT:
        default:
            wi = calculateRandomDirectionInHemisphere(normal, rng);
            break;
        }
    } while (!finish);

    // update path segment
    --path.remainingBounces;
    wi = glm::normalize(wi);
    //* fabs(glm::dot(wi, normal))
    path.color *= color * phong_brdf(ray.origin, wi, wo, normal, m);
    ray.origin = intersect + OFFSET_EPS * normal;
    ray.direction = wi;
}