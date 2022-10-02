#pragma once

#include "intersections.h"

/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__ void diffuseScatter(
    glm::vec3* ray,
    glm::vec3 normal,
    thrust::default_random_engine &rng,
    thrust::uniform_real_distribution<float> &u01) {
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

    *ray = up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ float schlick(float cos, float mIOR, float sceneIOR) {
    float reflectivityAtNormal = powf((mIOR - sceneIOR) / (mIOR + sceneIOR), 2.f);
    return reflectivityAtNormal + (1.f - reflectivityAtNormal) * powf(1.f - cos, 5.f);
}

__device__ void imperfectSpecularScatter(
    glm::vec3* ray,
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
    *ray = transform * untransformedRef;
}

// more or less copied from PBRT, 
// https://github.com/mmp/pbrt-v3/blob/master/src/core/bssrdf.cpp
__device__ float FresnelMoment1(float eta) {
    float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta,
        eta5 = eta4 * eta;
    return (eta < 1) ? 
        0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
        2.49277f * eta4 - 0.68441f * eta5 :
        -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
        1.27198f * eta4 + 0.12746f * eta5;
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
    const float ACNE_BIAS = 5e-3f; //TODO: this is a very annoying parameter to tune. solutions?
    // assume the scenes will be in AIR ie IOR = 1.0. o/w will need to refactor some things
    const float sceneIOR = 1.0f; 

    thrust::uniform_real_distribution<float> u01(0, 1); // [a, b)

    // normalizes might not be necesary (?)
    pathSegment.ray.direction = glm::normalize(pathSegment.ray.direction);
    normal = glm::normalize(normal);

    float altColor = false;
    if (m.matType == MatType::DIFFUSE) {
        diffuseScatter(&pathSegment.ray.direction, normal, rng, u01);
    }
    else if (m.matType == MatType::SPECULAR) { // from gpu gems 3 ch 20, eq 7, 8 & 9
        imperfectSpecularScatter(&pathSegment.ray.direction, normal, rng, u01, m.specular.exponent);
    }
    else if (m.matType == MatType::DIELECTRIC) { // Fresnel eqns
        // use below to check if normal is against ray direction; traps if not.
        // intersections.h should take care of all need to align normals and whatnot
        // if (glm::dot(normal, pathSegment.ray.direction) >= 0.f) { __trap(); }
        // Schlick approximation
        float cosThetaI = glm::dot(normal, -pathSegment.ray.direction);
        if (u01(rng) < schlick(cosThetaI, m.indexOfRefraction, sceneIOR)) { // reflect
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
    else if (m.matType == MatType::TRANSLUCENT) { 
        // roughly based on the PBR v3 SeparableBSSRDF,
        // also the paper Extending the Disney BRDF to a BSDF with Integrated Subsurface Scattering
        // BSSRDF has three components 1 - fr(costheta), Sp, Sw respectively:
        //      1. Fresnel term for spectral radiance through material, (see dielectric above)
        //      2. Fresnel term " " " entering the material (degree of light refracted into material),
        //      3. Profile term for how far radiance actually gets through the material
        if (outside) { // if outside, select ray for transmission, with shlick approx. (Sw, 2nd term)
            float cosThetaI = glm::dot(normal, -pathSegment.ray.direction);
            float c = 1 - 2 * FresnelMoment1(sceneIOR / m.indexOfRefraction);
            if (u01(rng) < (1 - schlick(cosThetaI, m.indexOfRefraction, sceneIOR)) / (PI * c)) {
                if (m.specular.exponent < 30 || u01(rng) < 0.9) { // ... tuning this is weirdo
                    diffuseScatter(&pathSegment.ray.direction, normal, rng, u01);
                } else {
                    imperfectSpecularScatter(&pathSegment.ray.direction, normal, rng, u01, m.specular.exponent);
                }
            } else {
                // simply diffuse into the material; c.f. 2.4 in the disney paper
                diffuseScatter(&pathSegment.ray.direction, -normal, rng, u01);
            }
        }
        else { // ray is inside material, leaving the 2nd and 3rd terms to deal with
            // PBR calculation is complicated and slower; 
            // instead make a brutal simplication: assume the whole geometry is pure diffuse
            // thus simple approximate absorption (beer-lambert) & single scattering:
            float dist = glm::length(intersect - pathSegment.ray.origin);
            float rand = u01(rng);
            if (rand == 0 || m.absorption == 0 || (-log(rand) / m.absorption > dist)) {
                // diffuse out of the object (opposite of normal)
                diffuseScatter(&pathSegment.ray.direction, -normal, rng, u01);
            } else {
                // redefine the "intersect" as a collision basically, so new ray will originate from the intersect
                intersect = getPointOnRay(pathSegment.ray, rand);
                diffuseScatter(&pathSegment.ray.direction, -pathSegment.ray.direction, rng, u01);
            }
            // very crude approx. for light transport
            pathSegment.color *= dist / (1.f + dist) * m.transColor;
            altColor = true;
        }
    }
    // bias origin to avoid shadow acne.
    // use ray direction over normal, since some rays are trying to "pass" the material barrier
    pathSegment.ray.origin = intersect + ACNE_BIAS * pathSegment.ray.direction;

    if (!altColor) {
        pathSegment.color *= m.color;
    }
}
