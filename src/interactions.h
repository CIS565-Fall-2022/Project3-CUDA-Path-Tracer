#pragma once

#include "intersections.h"

__host__ __device__
glm::vec3 squareToDiskUniform(const glm::vec2& sample)
{
    float phi, r, u, v;
    r = sqrt(sample.x);
    phi = 2 * PI * sample.y;
    u = r * cos(phi);
    v = r * sin(phi);
    return glm::vec3(u, v, 0);
}

__host__ __device__
glm::vec3 squareToDiskConcentric(const glm::vec2& sample)
{
    float phi, r, u, v;
    float a = 2 * sample.x - 1; 
    float b = 2 * sample.y - 1;
    if (a > -b) { // region 1 or 2
        if (a > b) {// region 1, also |a| > |b|
            r = a;
            phi = (PI / 4) * (b / a);
        }
        else {// region 2, also |b| > |a|
            r = b;
            phi = (PI / 4) * (2 - (a / b));
        }
    }
    else {// region 3 or 4
        if (a < b) { // region 3, also |a| >= |b|, a != 0
            r = -a;
            phi = (PI / 4) * (4 + (b / a));
        }
        else {// region 4, |b| >= |a|, but a==0 and b==0 could occur.
            r = -b;
            if (b != 0)
                phi = (PI / 4) * (6 - (a / b));
            else
                phi = 0;
        }
    }
    u = r * cos(phi);
    v = r * sin(phi);
    return glm::vec3(u, v, 0);
}

__host__ __device__
glm::vec3 localToWorldWithNormal(glm::vec3 pos, glm::vec3 normal) {
    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return pos.x * perpendicularDirection1 + pos.y * perpendicularDirection2 + pos.z * normal;
}

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng, float& pdf) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    glm::vec3 pos = squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
    pos.z = sqrt(1 - pos.x * pos.x - pos.y * pos.y);
    pdf = pos.z * INV_PI;
    return localToWorldWithNormal(pos, normal);

    //float up = sqrt(u01(rng)); // cos(theta)
    //float over = sqrt(1 - up * up); // sin(theta)
    //float around = u01(rng) * TWO_PI;

    //// Find a direction that is not the normal based off of whether or not the
    //// normal's components are all equal to sqrt(1/3) or whether or not at
    //// least one component is less than sqrt(1/3). Learned this trick from
    //// Peter Kutz.

    //glm::vec3 directionNotNormal;
    //if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
    //    directionNotNormal = glm::vec3(1, 0, 0);
    //} else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
    //    directionNotNormal = glm::vec3(0, 1, 0);
    //} else {
    //    directionNotNormal = glm::vec3(0, 0, 1);
    //}

    //// Use not-normal direction to generate two perpendicular directions
    //glm::vec3 perpendicularDirection1 =
    //    glm::normalize(glm::cross(normal, directionNotNormal));
    //glm::vec3 perpendicularDirection2 =
    //    glm::normalize(glm::cross(normal, perpendicularDirection1));

    //pdf = up * INV_PI;
    //return up * normal
    //    + cos(around) * over * perpendicularDirection1
    //    + sin(around) * over * perpendicularDirection2;
}

__host__ __device__
glm::vec3 calculateRandomDirectionInSpecularLobe(
    glm::vec3 wiCenter, float specex, thrust::default_random_engine& rng, float& pdf) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = powf(u01(rng), 1.f / (specex + 1.f)); // cos(alpha)
    float over = sqrt(1.f - up * up); // sin(alpha)
    float around = u01(rng) * TWO_PI;

    pdf = (specex + 1) * powf(up, specex) * over / TWO_PI;
    return localToWorldWithNormal(glm::vec3(cos(around) * over, sin(around) * over, up), wiCenter);

}

__host__ __device__
float FrDielectric(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
    if (cosThetaI <= 0.f)
    {
        float tmp = etaI;
        etaI = etaT;
        etaT = tmp;
        cosThetaI = abs(cosThetaI);
    }

    float sinThetaI = sqrt(glm::max((float)0, 1 - cosThetaI * cosThetaI));
    float sinThetaT = etaI / etaT * sinThetaI;
    if (sinThetaT >= 1) {
        return 1.f;
    }

    float cosThetaT = sqrt(glm::max((float)0, 1 - sinThetaT * sinThetaT));
    float Rparl = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
    float Rperp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__host__ __device__
bool Refract(const glm::vec3& wi, const glm::vec3& n, float eta,
    glm::vec3* wt) {
    // Compute cos theta using Snell's law
    float cosThetaI = glm::dot(n, wi);
    float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1) return false;
    float cosThetaT = sqrt(1 - sin2ThetaT);
    *wt = eta * -wi + (eta * cosThetaI - cosThetaT) * glm::vec3(n);
    return true;
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
    /*if (pathSegment.remainingBounces < 0) {
        int a = pathSegment.remainingBounces;
        return;
    }   */
    //todo

    
    glm::vec3 scatterDir;
    float pdf = 0.f;
    glm::vec3 color(0.f);

    if (!m.hasReflective && !m.hasRefractive) {     //pure diffuse
        scatterDir = calculateRandomDirectionInHemisphere(normal, rng, pdf);
        float cosine = glm::dot(normal, scatterDir);
        color = glm::max(cosine, 0.f) * m.color * INV_PI;
    }
    else if (m.hasReflective > 0 && m.hasReflective < 1){      //imperfect reflection
        thrust::uniform_real_distribution<float> u01(0, 1);
        float randNum = u01(rng);
        float frac = m.hasReflective;

        if (randNum < frac) {
            scatterDir = calculateRandomDirectionInHemisphere(normal, rng, pdf);
            float cosine = glm::dot(normal, scatterDir);
            color = glm::max(cosine, 0.f) * m.color * INV_PI * frac;
            pdf *= frac;
        }
        else {
            glm::vec3 wiCenter = glm::reflect(pathSegment.ray.direction, normal);
            scatterDir = calculateRandomDirectionInSpecularLobe(wiCenter, m.specular.exponent, rng, pdf);
            if (glm::dot(normal, scatterDir) <= 0) {
                pdf = 0;
            }
            else {
                float cosRI = glm::dot(scatterDir, wiCenter);
                color = m.specular.color * powf(cosRI, m.specular.exponent) * INV_PI * (1-frac);
                pdf *= (1 - frac);
            }
        }
    }
    else if (m.hasReflective == 1 && !m.hasRefractive) {    //perfect reflection
        scatterDir = glm::reflect(pathSegment.ray.direction, normal);
        pdf = 1;
        color = m.specular.color;
    }
    else if (m.hasReflective == 1 && m.hasRefractive == 1) {    //reflection and refraction, like glass
        thrust::uniform_real_distribution<float> u01(0, 1);
        float randNum = u01(rng);

        if (randNum < 0.5) {
            scatterDir = glm::reflect(pathSegment.ray.direction, normal);
            pdf = 0.5;
            float cosine = glm::dot(scatterDir, normal);
            float fresnel = FrDielectric(cosine, 1, m.indexOfRefraction);
            color = fresnel * m.specular.color;
        }
        else {
            float eta = m.indexOfRefraction;
            glm::vec3 trueNormal = normal;
            if (glm::dot(pathSegment.ray.direction, normal) > 0) {
                eta = 1.f / eta;
                trueNormal = -normal;
            }

            glm::vec3 refractDir;
            bool fullReflect = !Refract(-pathSegment.ray.direction, trueNormal, 1.f / eta, &refractDir);
            if (fullReflect) {
                scatterDir = glm::reflect(pathSegment.ray.direction, trueNormal);
                pdf = 0.5;
                float cosine = glm::dot(scatterDir, trueNormal);
                float fresnel = FrDielectric(cosine, 1, eta);
                color = fresnel * m.specular.color;
            }
            else {
                scatterDir = refractDir;
                pdf = 0.5;
                float cosine = glm::dot(scatterDir, trueNormal);
                float fresnel = (1 - FrDielectric(cosine, 1, eta));
                color = fresnel * m.refractionColor;
            }
        }
    }
    
    if (pdf < 0.01f) {
        pathSegment.color = glm::vec3(0.f);
    }
    else {
        pathSegment.color *= color / pdf;
    }
    pathSegment.ray.direction = scatterDir;
    pathSegment.ray.origin = intersect + scatterDir * 0.01f;
    pathSegment.remainingBounces--;


}
