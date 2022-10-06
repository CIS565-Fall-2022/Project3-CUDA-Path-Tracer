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

    //pure diffuse
    if (!m.hasReflective && !m.hasRefractive) {
        auto direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
        pathSegment.ray.direction = direction;
        pathSegment.ray.origin = intersect + 0.0001f * normal;
        pathSegment.color *= m.color;
    }
    //perfect reflective
    else if (m.hasReflective && !m.hasRefractive) {
        glm::vec3 reflection = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.direction = reflection;
        pathSegment.ray.origin = intersect + 0.0001f * normal;
        pathSegment.color *= m.color;
    }
    //both reflection and refraction
    else if (m.hasReflective && m.hasRefractive) {
        glm::vec3 incident = pathSegment.ray.direction;
        float cos_theta = glm::dot(normal,-incident );
        float n1 = 0.f;
        float n2 = 0.f;
        if (cos_theta >= 0) { //vacuum to object
            n1 = 1.f;
            n2 = m.indexOfRefraction;
        }
        else {//object to vacuum
            normal = glm::normalize(-normal); 
            n1 = m.indexOfRefraction;
            n2 = 1.f;
        }
        //schlick's approximation
        float R0 = (n1 - n2) / (n1 + n2);
        R0 = R0 * R0;
        float Fresnel_term = R0 + (1 - R0) * pow(1 - cos_theta, 5);
        thrust::uniform_real_distribution<float> u01(0, 1);
        if (u01(rng) < Fresnel_term) {//reflection
            glm::vec3 reflection = glm::reflect(incident, normal);
            pathSegment.ray.direction = reflection;
            pathSegment.ray.origin = intersect + 0.0001f * normal;
            pathSegment.color *= m.color;
        }
        else {//refraction
            glm::vec3 refraction = glm::normalize(glm::refract(incident, normal, n1 / n2));
            pathSegment.ray.direction = refraction;
            pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
            pathSegment.color *= m.color;
        }

    }

}

__host__ __device__
float Fresnel_Schlicks(float n1, float n2, float cos_theta) {
    float R0 = (n1 - n2) / (n1 + n2);
    R0 = R0 * R0;
    float Fresnel_term = R0 + (1 - R0) * pow(1 - cos_theta, 5);
    return Fresnel_term;
}

//helper functions for Microfacet Reflection Model
//try this 
__device__
glm::vec3 Fresnel(glm::vec3 R0, float cos_theta) {
    glm::vec3 Fresnel_term = R0 + (1.f - R0) * (float)pow(1 - cos_theta, 5);
    return Fresnel_term;
}


__device__
float G_Schlicks(float roughness, glm::vec3 normal, glm::vec3 view)
{
    float k = (roughness + 1) * (roughness + 1) / 8;
    glm::vec3 n = glm::normalize(normal);
    glm::vec3 v = glm::normalize(view);
    return glm::dot(n, v) / (glm::dot(n, v) * (1 - k) + k);
}

__device__
float Geometry_Smith(float roughness, glm::vec3 light, glm::vec3 view, glm::vec3 half)
{
    glm::vec3 v = glm::normalize(view);
    glm::vec3 h = glm::normalize(half);    
    glm::vec3 l = glm::normalize(light);
    return G_Schlicks(roughness, l, h) * G_Schlicks(roughness, v, h);
}



__device__
float D_GGX(float roughness, glm::vec3 normal, glm::vec3 half)
{
    float rough = roughness * roughness;
    float rough2 = rough * rough;
    rough2 = rough;
    float pi = 3.1415926;
    glm::vec3 n = glm::normalize(normal);
    glm::vec3 h = glm::normalize(half);
    float dot2 = glm::dot(n, h) * glm::dot(n, h);
    return rough2 / (pi * pow(dot2 * (rough2 - 1) + 1, 2));
}

__device__
void scatterRay(
    PathSegment& pathSegment,
    ShadeableIntersection& intersection,
    const Material& m,
    thrust::default_random_engine& rng,
    glm::vec3 camPos) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

    glm::vec3 intersect = getPointOnRay(pathSegment.ray, intersection.t);
    glm::vec3 normal = intersection.surfaceNormal;
    glm::vec2 uv = intersection.uv; 
    bool hasTexture = true;
    if (uv == glm::vec2(-1)) hasTexture = false;

    //glm::vec2 act_uv = glm::vec2(m.texture_width * uv.x, m.texture_height * (1 - uv.y));
    //int temp = act_uv.x * act_uv.y /** m.channels*/;
    //auto data = m.img;
    //glm::vec3 tex_color = glm::vec3(data[temp], data[temp+1], data[temp+2]);

    if (m.microfacet) {
        float metalness = m.metalness;
        float roughness = m.roughness;

        glm::vec3 eye_pos = camPos;
        glm::vec3 worldPos = intersect;
        auto wi = -pathSegment.ray.direction;
        auto N = glm::normalize(normal);
        auto wo = glm::normalize(eye_pos - worldPos);

        thrust::uniform_real_distribution<float> u01(0, 1);
        if(u01(rng) < metalness)
            wo = glm::reflect(pathSegment.ray.direction, normal);
        else
            wo = glm::normalize(calculateRandomDirectionInHemisphere(N, rng));

        auto H = glm::normalize((wi + wo) / 2.f);
        auto albedo = glm::vec3(1, 1, 1); //hardcode surface color to grey;
        auto F0 = glm::vec3(0.04f);
        F0 = glm::mix(F0, albedo, metalness);
        //F0 = glm::vec3(1,1,1);
        float thetaIH = glm::dot(H, wi);
        float thetaIN = glm::dot(N, wi);
        glm::vec3 fresnel = Fresnel(F0, thetaIH);

        //float roughness = m.roughness;
        

        float D = D_GGX(roughness, N, H);
        float G = Geometry_Smith(roughness, wi, wo, H);

        float NdotL = max(dot(N, wi), 0.0f);
        float NdotV = max(dot(N, wo), 0.0f);
        glm::vec3 brdf = fresnel * G * D / max(4.0f * NdotL * NdotV, 0.01f);

        auto radiance = pathSegment.color;

        //radiance = glm::vec3(1);
        glm::vec3 Lo = (brdf) * radiance * NdotL;
        glm::vec3 color = Lo;
        //color = color / (color + glm::vec3(1.0));
        color = pow(color, glm::vec3(1.0 / 2.2));

        pathSegment.color *= color;

        if (u01(rng) < metalness) {
            glm::vec3 reflection = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.direction = reflection;
            pathSegment.ray.origin = intersect + 0.0001f * normal;
        }
        else {
            auto direction = glm::normalize(calculateRandomDirectionInHemisphere(N, rng));
            pathSegment.ray.direction = direction;
            pathSegment.ray.origin = intersect + 0.0001f * normal;

        }


    }
    else {
        glm::vec3 tex_color = glm::vec3(0.8, 0.8, 0.8);
        //pure diffuse
        if (!m.hasReflective && !m.hasRefractive) {
            auto direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
            pathSegment.ray.direction = direction;
            pathSegment.ray.origin = intersect + 0.0001f * normal;
            if (hasTexture)
                pathSegment.color *= tex_color;
            else
                pathSegment.color *= m.color;
        }
        //perfect reflective
        else if (m.hasReflective && !m.hasRefractive) {
            glm::vec3 reflection = glm::reflect(pathSegment.ray.direction, normal);
            pathSegment.ray.direction = reflection;
            pathSegment.ray.origin = intersect + 0.0001f * normal;
            pathSegment.color *= m.color;
        }
        //both reflection and refraction
        else if (m.hasReflective && m.hasRefractive) {
            glm::vec3 incident = pathSegment.ray.direction;
            float cos_theta = glm::dot(normal, -incident);
            float n1 = 0.f;
            float n2 = 0.f;
            if (cos_theta >= 0) { //vacuum to object
                n1 = 1.f;
                n2 = m.indexOfRefraction;
            }
            else {//object to vacuum
                normal = glm::normalize(-normal);
                n1 = m.indexOfRefraction;
                n2 = 1.f;
            }
            //schlick's approximation
            /*float R0 = (n1 - n2) / (n1 + n2);
            R0 = R0 * R0;
            R0 + (1 - R0) * pow(1 - cos_theta, 5);*/
            float Fresnel_term = Fresnel_Schlicks(n1, n2, cos_theta);
            thrust::uniform_real_distribution<float> u01(0, 1);
            if (u01(rng) < Fresnel_term) {//reflection
                glm::vec3 reflection = glm::reflect(incident, normal);
                pathSegment.ray.direction = reflection;
                pathSegment.ray.origin = intersect + 0.0001f * normal;
                pathSegment.color *= m.color;
            }
            else {//refraction
                glm::vec3 refraction = glm::normalize(glm::refract(incident, normal, n1 / n2));
                pathSegment.ray.direction = refraction;
                pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
                pathSegment.color *= m.color;
            }

        }
    }
}
