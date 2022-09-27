#pragma once

#include "intersections.h"
#include "utilities.h"
#include <thrust/random.h>
#include <glm/gtc/epsilon.hpp>
#include "scene.h"

#define PBRT

// Reference: https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models#MicrofacetDistribution::D
// BeckmannDistribution for microfacet modeling
struct BeckmannDistribution {
    __device__
    BeckmannDistribution(float alphax, float alphay) {

    }
    __device__
    float roughness_to_alpha(float roughness) {
        roughness = glm::max(roughness, 1e-3f);
        float x = glm::log(roughness);
        return 1.62142f + 0.819955f * x + 0.1734f * x * x +
            0.0171201f * x * x * x + 0.000640711f * x * x * x * x;
    }
    __device__
    float D(glm::vec3 const& wh) const {
        return;
    }
    __device__
    glm::vec3 sample_wh(glm::vec3 const& wo, glm::vec2& u) const {
        return glm::vec3(0);
    }
};

__device__
bool feq(float a, float b) {
    a -= b;
    return a <= EPSILON && a >= -EPSILON;
}

__device__
bool is_zero(glm::vec3 const& vec) {
    return glm::length2(vec) <= glm::epsilon<float>() * glm::epsilon<float>();
}

__device__
glm::vec3 randomVecInUnitSphere(thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 ret(u01(rng), u01(rng), u01(rng));
    return ret / glm::length(ret);
}
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng) {
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


__device__
float schlick_frensnel(glm::vec3 const& wo, glm::vec3 const& up_norm, float eta) {
    // Schlick's approximation
    // Taken from: https://raytracing.github.io/books/RayTracingInOneWeekend.html
    float cosine = glm::clamp(glm::dot(wo, up_norm), 0.0f, 1.0f);
    float r0 = (1 - eta) / (1 + eta);
    r0 = r0 * r0;
    return r0 + (1 - r0) * powf((1 - cosine), 5);
}
__device__
float fresnel_dielectric(glm::vec3 const& wo, glm::vec3 const& up_norm, float eta) {
    // Reference:
    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    // https://www.pbr-book.org/3ed-2018/Reflection_Models/Specular_Reflection_and_Transmission#FresnelReflectance
    
    float cosine = glm::clamp(glm::dot(wo, up_norm), 0.0f, 1.0f);
    float sin2 = 1 - cosine * cosine;
    float eta2 = eta * eta;
    float T0 = 1 - sin2 / (eta * eta);
    if (T0 < 0) return 1;
    T0 = sqrtf(T0);
    float T1 = eta * T0;
    float T2 = eta * cosine;

    float Rs = (cosine - T1) / (cosine + T1);
    float Rp = (T0 - T2) / (T0 + T2);
    return 0.5f * (Rs * Rs + Rp * Rp);
}
__device__
glm::vec3 get_microfacet_vector(float roughness, glm::vec3 const& normal, float rng1, float rng2) {
    // Reference: 
    // https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
    // https://www.pbr-book.org/3ed-2018/Reflection_Models/Microfacet_Models
    // Uses the GGX normal distribution function 
    float phi = 2 * PI * rng1;
    float theta = atan(roughness * sqrt(rng2 / (1 - rng2)));
    // TODO
    return glm::vec3(0);
}

#ifdef PBRT
/// <summary>
/// calculates wi given wo for a material
/// </summary>
/// <returns> wi </returns>
__device__
glm::vec3 calc_wi(
    bool perfect_smooth,
    Material const& m, 
    bool leaving,
    glm::vec3 const& normal,
    glm::vec3 const& wo, // - ray.dir
    thrust::default_random_engine& rng
) {
    glm::vec3 up_norm = leaving ? -normal : normal;
    thrust::uniform_real_distribution<float> u01;
    float eta;

    if (perfect_smooth) {
        switch (m.type) {
            case Material::Type::REFL:
                return glm::reflect(-wo, up_norm);

            case Material::Type::REFR:
                if (feq(m.ior, 1)) {
                    return -wo;
                }
                eta = leaving ? m.ior : 1 / m.ior;
                if (u01(rng) < fresnel_dielectric(wo, up_norm, eta)) {
                    return glm::reflect(-wo, up_norm);
                } else {
                    return glm::refract(-wo, up_norm, eta);
                }

            case Material::Type::TRANSPARENT:
            
            default:
                return glm::vec3(0);
        }
    } else {
        switch (m.type) {
        case Material::Type::DIFFUSE:
            return calculateRandomDirectionInHemisphere(normal, rng);

        case Material::Type::GLOSSY:
            glm::vec3 rands(u01(rng), u01(rng), u01(rng));
            eta = leaving ? m.ior : 1 / m.ior;
            if (rands[0] < fresnel_dielectric(wo, up_norm, eta)) {
                return glm::reflect(-wo, get_microfacet_vector(m.roughness, up_norm, rands[1], rands[2]));
            } else {
                return calculateRandomDirectionInHemisphere(normal, rng);
            }
        case Material::Type::REFL:
        case Material::Type::REFR:
        case Material::Type::SUBSURFACE:
        case Material::Type::TRANSPARENT:
        
        default:
            return glm::vec3(0);
        }
    }
}

/// <summary>
/// samples the probability density function for the bsdf of a material
/// </summary>
/// <returns>probability</returns>
__device__
float calc_bsdf_pdf(
    bool perfect_smooth,
    Material const& m,
    bool leaving,
    glm::vec3 const& normal,
    glm::vec3 const& wo,
    glm::vec3 const& wi
) {
    glm::vec3 up_norm = leaving ? -normal : normal;
    float eta;
    float refr;

    if (perfect_smooth) {
        switch (m.type) {
        case Material::Type::REFL:
            return 1;
        case Material::Type::REFR:
            refr = glm::dot(normal, wi) * glm::dot(normal, wo);
            if (feq(m.ior, 1)) {
                return refr >= 0 ? 0 : 1;
            }
            eta = leaving ? m.ior : 1 / m.ior;
            if (refr >= 0) {
                return fresnel_dielectric(wo, up_norm, eta);
            } else {
                return 1 - fresnel_dielectric(wo, up_norm, eta);
            }
        case Material::Type::TRANSPARENT:
        default:
            return 0;
        }
    } else {
        switch (m.type) {
        case Material::Type::DIFFUSE:
            return glm::max(glm::dot(up_norm, wi) * INV_PI, 0.0f);
        case Material::Type::GLOSSY:
        case Material::Type::REFL:
        case Material::Type::REFR:
        case Material::Type::SUBSURFACE:
        case Material::Type::TRANSPARENT:
        default:
            return 0;
        }
    }
}

/// <summary>
/// evaluates radiance given a material
/// </summary>
/// <returns> sampled radiance </returns>
__device__
glm::vec3 cacl_bsdf_factor(
    bool perfect_smooth,
    Material const& m,
    bool leaving,
    glm::vec3 const& normal,
    glm::vec3 const& wo,
    glm::vec3 const& wi
){
    glm::vec3 up_norm = leaving ? -normal : normal;
    float eta;
    float refr;
    glm::vec3 ret;

    if (perfect_smooth) {
        switch (m.type) {
        case Material::Type::REFL:
#pragma unroll
            for (int i = 0; i < 3; ++i) {
                ret[i] = schlick_frensnel(wo, up_norm, (1 + sqrt(m.diffuse[i]) / (1 - sqrt(m.diffuse[i]))));
            }
            break;
        case Material::Type::REFR:
            refr = glm::dot(normal, wi) * glm::dot(normal, wo);
            if (feq(m.ior, 1)) {
                ret = refr >= 0 ? glm::vec3(0) : glm::vec3(1);
            } else {
                eta = leaving ? m.ior : 1 / m.ior;
                if (refr >= 0) {
                    ret = glm::vec3(1, 1, 1) * fresnel_dielectric(wo, up_norm, eta);
                } else {
                    //TODO fix this case
                    ret = glm::vec3(1, 1, 1) * (1 / (eta * eta)) * (1 - fresnel_dielectric(wo, up_norm, eta));
                }
            }

            break;
        case Material::Type::TRANSPARENT:
        default:
            return glm::vec3(0);
        }
    } else {
        switch (m.type) {
        case Material::Type::DIFFUSE:
            ret = m.diffuse * INV_PI * glm::abs(glm::dot(normal, wi));
            break;
        case Material::Type::GLOSSY:
        case Material::Type::REFL:
        case Material::Type::REFR:
        case Material::Type::SUBSURFACE:
        case Material::Type::TRANSPARENT:
        default:
            return glm::vec3(0);
        }
    }
    if (is_zero(ret)) {
        return ret;
    }
    return ret / calc_bsdf_pdf(perfect_smooth, m, leaving, normal, wo, wi);
}

#else

/// <summary>
/// classic phong reflectance model
/// </summary>
/// <returns></returns>
__device__
float phong(glm::vec3 p, glm::vec3 wi, glm::vec3 wo, glm::vec3 normal, Material const& m) {
    float kd = 1 - m.hasReflective;
    float ks = m.hasReflective;

    float specular = 0;
    if (ks > 0) {
        specular = ks * powf(glm::dot(wi, glm::reflect(-wo, normal)), m.specular.exponent);
    }
    return glm::clamp(kd + specular / glm::dot(wi, normal), 0.0f, 1.0f);
}
#endif


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
    ShadeableIntersection const& inters,
    Material const& m,
    Span<Light> const& lights,
    thrust::default_random_engine& rng) {

    glm::vec3 normal = inters.surfaceNormal;
    glm::vec3 intersect = inters.hitPoint;
    Ray& ray = path.ray;

    glm::vec3 wi;
    glm::vec3 wo = -ray.direction;
    bool leaving = glm::dot(wo, normal) < 0;
    glm::vec3 up_norm = leaving ? -normal : normal;

    assert(path.remainingBounces > 0);
    assert(feq(m.emittance, 0));
    assert(feq(glm::length(normal), 1));
    assert(feq(glm::length(path.ray.direction), 1));
    if (glm::dot(wo, up_norm) < 0) {
#ifndef NDEBUG
        printf("glm::dot(wo, up_norm) < 0\n");
#endif // !NDEBUG
        path.terminate();
        return;
    }

#ifdef PBRT
    color_t color (1,1,1);
    --path.remainingBounces;

    bool is_smooth = feq(m.roughness, 0);
    wi = calc_wi(is_smooth, m, leaving, normal, wo, rng);
    if (is_zero(wi)) {
        path.terminate();
    } else {
        color = cacl_bsdf_factor(is_smooth, m, leaving, normal, wo, wi);
    }

    if (is_zero(color)) {
        path.terminate();
    }

    // update path segment
    path.color *= color;
    ray.origin = intersect + OFFSET_EPS * up_norm;
    ray.direction = glm::normalize(wi);
#else // not strictly PBRT but looks alright
    /*
    ** Heavily based on
    *  https://raytracing.github.io/books/RayTracingInOneWeekend.html
    */

    color_t color = m.diffuse;
    thrust::uniform_real_distribution<float> u01(0, 1);


    enum {
        LAMBERT,
        METAL,
        DIELECTRIC,
    } type;

    if (m.hasReflective > 0) {
        if (m.hasRefractive > 0 && m.ior > 0) {
            type = DIELECTRIC;
        } else {
            type = METAL;
        }
    } else {
        type = LAMBERT;
    }

    bool finish;
    do {
        finish = true;
        float eta, cos_theta, sin_theta;

        // NOTE: m.hasReflective controls how "fuzzy" the reflection is
        //       m.hasRefractive controls how likely the ray is refracted

        switch (type) {
        case METAL:
            wi = glm::reflect(ray.direction, up_norm);
            wi += fmax(1 - m.hasReflective, 0.0f) * randomVecInUnitSphere(rng);
            break;
        case DIELECTRIC:
            // eta = source / dest
            // so entering ==> air / idx of refraction
            //    leaving ==> idx of refraction / air = idx of refraction
            eta = leaving ? m.ior : 1 / m.ior;
            cos_theta = fmin(glm::dot(wo, up_norm), 1.0f);
            sin_theta = sqrtf(1 - cos_theta * cos_theta);

            // divide reflection/refraction based on how significant the factor is
            if (eta * sin_theta > 1 || schlick_frensnel(wo, up_norm, eta) > u01(rng)) {
                type = METAL;
                finish = false;
            } else {
                wi = glm::refract(ray.direction, up_norm, eta);
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
    path.color *= color * phong(ray.origin, wi, wo, normal, m);
    ray.origin = intersect + OFFSET_EPS * normal;
    ray.direction = wi;
#endif // PBRT
}