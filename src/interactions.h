#pragma once


#include "intersections.h"
#include "utilities.h"
#include <thrust/random.h>
#include <glm/gtc/epsilon.hpp>
#include "scene.h"

__device__ bool feq(float a, float b) {
    a -= b;
    return a <= EPSILON && a >= -EPSILON;
}

__device__ bool is_zero(glm::vec3 const& vec) {
    return glm::length2(vec) <= glm::epsilon<float>() * glm::epsilon<float>();
}

__device__ __forceinline__
glm::vec3 face_forward(glm::vec3 const& N, glm::vec3 const& dir) {
    return glm::dot(N, dir) < 0 ? -N : N;
}

template<typename T>
__device__ __forceinline__
void dev_swap(T& a, T& b) {
    T tmp = a;
    a = b;
    b = tmp;
}

/**
 * Computes t20 cosine-weighted random direction in t20 hemisphere.
 * Used for diffuse lighting.
 */
__device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find t20 direction that is not the normal based off of whether or not the
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

// reference: https://raytracing.github.io/books/RayTracingInOneWeekend.html
__device__ float schlick_frensnel(float cosine, float eta) {
    // Schlick's approximation
    float t0 = 1 - glm::min(cosine, 1.0f);
    float r0 = (1 - eta) / (1 + eta);
    r0 = r0 * r0;
    return r0 + (1 - r0) * t0 * t0 * t0 * t0 * t0; // faster than pow?
}

// reference: https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
__device__ color_t fresnel_conductor(float cosine, color_t const& eta) {
    cosine = glm::clamp(cosine, -1.0f, 1.0f);

    float c2 = cosine * cosine;
    float s2 = 1. - c2;
    color_t t0 = eta * eta - s2;
    color_t t22 = t0 * t0;
    color_t t20 = glm::sqrt(0.5f * (t22 + t0));
    color_t t1 = t22 + c2;
    color_t t2 = 2.0f * cosine * t20;
    color_t t3 = c2 * t22 + s2 * s2;
    color_t Rs = (t1 - t2) / (t1 + t2);
    color_t Rp = Rs * (t3 - (t2 * s2)) / (t3 + (t2 * s2));

    return 0.5f * (Rp + Rs);
}
// reference: https://github.com/mmp/pbrt-v3/blob/master/src/core/reflection.cpp
__device__ float fresnel_dielectric(float cosine, float etaI, float etaT) {
    cosine = glm::clamp(cosine, -1.0f, 1.0f);
    if (cosine < 0.f) {
        dev_swap(etaI, etaT);
        cosine = glm::abs(cosine);
    }
    float st = etaI / etaT * glm::sqrt(glm::max((float)0, 1 - cosine * cosine));
    // total internal reflection
    if (st >= 1) return 1;
    float ct = glm::sqrt(glm::max((float)0, 1 - st * st));
    float t0 = etaI * ct;
    float t1 = etaT * ct;
    float t2 = etaI * cosine;
    float t3 = etaT * cosine;
    float Rs = (t3 - t0) / (t3 + t0);
    float Rp = (t2 - t1) / (t2 + t1);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

struct SamplePointSpace {
    glm::mat3x3 l2w, w2l; // local to world

    __device__ SamplePointSpace(glm::vec3 const& n) {
        // constructs t20 world to local matrix
        glm::vec3 h = n;
        if (fabs(h.x) <= fabs(h.y) && fabs(h.x) <= fabs(h.z)) {
            h.x = 1.0f;
        } else if (fabs(h.y) <= fabs(h.x) && fabs(h.y) <= fabs(h.z)) {
            h.y = 1.0f;
        } else {
            h.z = 1.0f;
        }
        glm::vec3 y = glm::cross(h, n);
        l2w = glm::mat3x3(
            glm::normalize(glm::cross(n, y)),
            glm::normalize(glm::cross(h, n)),
            glm::normalize(n)
        );
        // inverse same as transpose
        w2l = glm::transpose(l2w);
    }
    __device__ __forceinline__ glm::vec3 world_to_local(glm::vec3 const& v) {
        return w2l * v;
    }
    __device__ __forceinline__ glm::vec3 local_to_world(glm::vec3 const& v) {
        return l2w * v;
    }
};

// encapsulates the BSDF functions for all sort of materials
// note: all operations in local space
struct BSDF {
    Material::Type type;
    bool is_delta;
    Material const& m;
    thrust::default_random_engine& rng;
    color_t reflectance;

    __device__ BSDF(Material const& m, ShadeableIntersection const& inters, thrust::default_random_engine& rng)
        : m(m), type(m.type), is_delta(feq(m.roughness, 0)), rng(rng) {
        if (m.textures.diffuse != -1) {
            reflectance = inters.tex_color;
        } else {
            reflectance = m.diffuse;
        }
    }

    __device__ __forceinline__ float abs_cos_theta(glm::vec3 const& wi) const {
        return glm::abs(wi.z);
    }
    __device__ __forceinline__ float cos_theta(glm::vec3 const& wi) const {
        return wi.z;
    }
    __device__ __forceinline__ float same_hemisphere(glm::vec3 const& w, glm::vec3 const& wp) const {
        return w.z * wp.z > 0;
    }
    __device__ color_t sample_f(glm::vec3 const& wo, glm::vec3& wi, float& pdf) const {
        float etaI = 1, etaT = m.ior, eta, F;
        thrust::uniform_real_distribution<float> u01;
        if (is_delta) {
            switch (type) {
            case Material::Type::REFL: // Perfect Reflection
                wi = glm::vec3(-wo.x, -wo.y, wo.z);
                pdf = 1;
                return fresnel_conductor(abs_cos_theta(wi), (1.0f + sqrt(reflectance)) / (1.0f - sqrt(reflectance)))
                    * reflectance / abs_cos_theta(wi);

            case Material::Type::REFR: // Fresnel-Modulated Specular Reflection and Transmission
                F = fresnel_dielectric(wo.z, etaI, etaT);

                if (u01(rng) < F) {
                    wi = glm::vec3(-wo.x, -wo.y, wo.z);
                    pdf = F;
                    return F * reflectance / abs_cos_theta(wi);
                } else {
                    if (wo.z < 0) {
                        dev_swap(etaI, etaT);
                    }
                    wi = glm::refract(-wo, face_forward({0,0,1}, wo), etaI / etaT);
                    if (feq(glm::length(wi), 0)) {
                        break;
                    }

                    pdf = 1 - F;
                    return (1 - F) * reflectance / abs_cos_theta(wi);
                }
                
            case Material::Type::TRANSPARENT:
                break;
            }
        } else {
            switch (m.type) {
            case Material::Type::DIFFUSE:
                wi = calculateRandomDirectionInHemisphere(glm::vec3(0, 0, 1), rng);
                pdf = same_hemisphere(wo, wi) ? abs_cos_theta(wi) * INV_PI : 0;
                return reflectance * INV_PI;

            case Material::Type::GLOSSY:
            case Material::Type::REFL:
            case Material::Type::REFR:
            case Material::Type::SUBSURFACE:
            case Material::Type::TRANSPARENT:
                break;
            }
        }

        wi = glm::vec3(0);
        pdf = 0;
        return glm::vec3(1);
    }
};


/**
 * Scatter t20 ray with some probabilities according to the material properties.
 * For example, t20 diffuse surface scatters in t20 cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in t20 few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between t20 each effect (t20 diffuse bounce
 *   and t20 specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as t20 good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

#define OFFSET_EPS 0.001f
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

    bool ray_hit_front_face = glm::dot(ray.direction, normal) < 0;
    glm::vec3 up_norm = ray_hit_front_face ? normal : -normal;

    assert(path.remainingBounces > 0);
    assert(feq(m.emittance, 0));
    assert(feq(glm::length(normal), 1));
    assert(feq(glm::length(path.ray.direction), 1));
    if (glm::dot(-ray.direction, up_norm) < 0) {
#ifndef NDEBUG
        printf("glm::dot(wo, up_norm) < 0\n");
#endif // !NDEBUG
        path.terminate();
        return;
    }

    color_t color(1, 1, 1);

    float pdf;
    glm::vec3 wi;

    SamplePointSpace space(normal);
    {
        glm::vec3 wo = space.world_to_local(-ray.direction);
        BSDF bsdf(m, inters, rng);
        color = bsdf.sample_f(wo, wi, pdf);

        if (is_zero(wi) || feq(pdf, 0)) {
            color = glm::vec3(1);
            path.terminate();
            return;
        } else {
            color = color * bsdf.cos_theta(wi) / pdf;
        }

        // terminate low energy paths
        if (is_zero(color)) {
            path.terminate();
            return;
        }
    }
    wi = space.local_to_world(wi);

    // update path segment

    --path.remainingBounces;
    path.color *= color;
    ray.origin = intersect + OFFSET_EPS * wi;
    ray.direction = glm::normalize(wi);
}