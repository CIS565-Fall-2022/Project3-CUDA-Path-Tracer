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

#if USE_PROCEDURAL_TEXTURE
__host__ __device__
float fract(float x)
{
    return x - floorf(x);
}

__host__ __device__
float mix(float x, float y, float a) {
    return x * (1.f - a) + y * a;
}

__host__ __device__
//FBM NOISE FIRST VARIANT
float random3D(glm::vec3 p) {
    return sin(glm::length(glm::vec3(fract(glm::dot(p, glm::vec3(161.1, 121.8, 160.2))),
        fract(glm::dot(p, glm::vec3(120.5, 161.3, 160.4))),
        fract(glm::dot(p, glm::vec3(161.4, 161.2, 122.5))))) * 435.90906);
}

__host__ __device__
float interpolateNoise3D(float x, float y, float z)
{
    int intX = int(floor(x));
    float fractX = fract(x);
    int intY = int(floor(y));
    float fractY = fract(y);
    int intZ = int(floor(z));
    float fractZ = fract(z);

    float v1 = random3D(glm::vec3(intX, intY, intZ));
    float v2 = random3D(glm::vec3(intX + 1, intY, intZ));
    float v3 = random3D(glm::vec3(intX, intY + 1, intZ));
    float v4 = random3D(glm::vec3(intX + 1, intY + 1, intZ));

    float v5 = random3D(glm::vec3(intX, intY, intZ + 1));
    float v6 = random3D(glm::vec3(intX + 1, intY, intZ + 1));
    float v7 = random3D(glm::vec3(intX, intY + 1, intZ + 1));
    float v8 = random3D(glm::vec3(intX + 1, intY + 1, intZ + 1));


    float i1 = mix(v1, v2, fractX);
    float i2 = mix(v3, v4, fractX);

    //mix between i1 and i2
    float i3 = mix(i1, i2, fractY);

    float i4 = mix(v5, v6, fractX);
    float i5 = mix(v7, v8, fractX);

    //mix between i3 and i4
    float i6 = mix(i4, i5, fractY);

    //mix between i3 and i6
    float i7 = mix(i3, i6, fractZ);

    return i7;
}

__host__ __device__
float fbmNoise(float x, float y, float z)
{
    float total = 0.0;
    float persistence = 0.3;
    float frequency = 2.0;
    float amplitude = 6.0;
    int octaves = 2;

    for (int i = 1; i <= octaves; i++) {
        total += amplitude * interpolateNoise3D(frequency * x, frequency * y, frequency * z);
        frequency *= 2.0;
        amplitude *= persistence;
    }
    return total;
}

__host__ __device__
glm::vec3 proceduralTexture(glm::vec3 pos, const Material& m) {
    float noise = fbmNoise(pos.x, pos.y, pos.z);
    glm::vec3 pink = glm::vec3(217.f, 93.f, 184.f) / 255.f;
    glm::vec3 blue = glm::vec3(81.f, 187.f, 245.f) / 255.f;

    glm::vec3 surfaceColor = glm::vec3(0.f);
    if (noise > 0.5) {
        surfaceColor = blue;
    }
    else {
        surfaceColor = pink;
    }

    return surfaceColor;
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
    thrust::default_random_engine& rng
#if USE_PROCEDURAL_TEXTURE
    , bool hasHitObj
#endif
) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    // treat the rest as perfectly specular.
    // assuming there's only one light

    // based on float value of reflective and refractive
    // with float percent likelihood the ray goes reflective vs. refractive

    thrust::uniform_real_distribution<float> u01(0, 1);

    float randGen = u01(rng);

    glm::vec3 pointColor = m.color;

#if USE_PROCEDURAL_TEXTURE && LOAD_OBJ
    if (hasHitObj) {
        // if this ray has hit an obj
        pointColor = proceduralTexture(intersect, m);
    }
#endif

    // if perfectly specular
    if (m.hasReflective == 1) {
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
    else if (randGen <= m.hasReflective) {
        // if not perfectly specular

        thrust::uniform_real_distribution<float> u02(0.1, 0.2);
        float jitterX = u02(rng);
        float jitterY = u02(rng);

        glm::vec3 perfectSpecularDir = glm::reflect(pathSegment.ray.direction, normal);
        glm::vec3 newDirection = glm::normalize(glm::vec3(perfectSpecularDir[0] + jitterX, perfectSpecularDir[1] + jitterY, perfectSpecularDir[2])); // todo change this direction
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
#endif