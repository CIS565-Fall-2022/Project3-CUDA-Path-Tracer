#pragma once

#include "intersections.h"

#define PROCEDURAL_TEXTURE 1

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

__host__ __device__
glm::vec3 calculateImperfectSpecularDirection(
  glm::vec3 normal, float shininess, thrust::default_random_engine& rng) {
  thrust::uniform_real_distribution<float> u01(0, 1);

  float theta = 1.f / cos(1.f / pow(u01(rng), shininess + 1));
  float phi = u01(rng) * TWO_PI;

  // Find a direction that is not the normal based off of whether or not the
  // normal's components are all equal to sqrt(1/3) or whether or not at
  // least one component is less than sqrt(1/3). Learned this trick from
  // Peter Kutz.

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

  return cos(theta) * normal
    + cos(phi) * sin(theta) * perpendicularDirection1
    + sin(phi) * sin(theta) * perpendicularDirection2;
}

__host__ __device__
void FetchTexture(const Texture& texture, const int offset, const glm::vec3* texImage, const glm::vec2& uv, glm::vec3 &color) {
  int tx = uv.x * (texture.width - 1);
  int ty = (uv.y) * (texture.height - 1);
  int idx = (ty * texture.width + tx);

  //int offset = (isNormal) ? texture.offsetNormal : texture.offsetColor;

#if PROCEDURAL_TEXTURE == 1
  color = texImage[idx + offset];
#else
  color = texImage[idx + offset];
#endif
}

__host__ __device__
void normalMapping(glm::vec3& normalSurface, const glm::vec3& normalMap, const glm::vec4& tangent, glm::vec3 &normal) {
  glm::vec3 t = glm::vec3(tangent);
  glm::vec3 b = glm::cross(normalSurface, t) * tangent.w;
  glm::mat3 TBN = glm::mat3(t, b, normalSurface);

  normal = glm::normalize(TBN * normalMap);
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
        int idx,
        PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        glm::vec2 uv,
        const Material &m,
        const int num_lightSource,
        glm::vec4* lightSourceSampled,
        const Texture* tex,
        const glm::vec3* texImage, 
        const glm::vec3* emissiveTexture,
        const int traceDepth,
        thrust::default_random_engine &rng) {

  // Textue 
  glm::vec3 color;
  color = m.color;
  if (m.texID != -1) {
    FetchTexture(tex[m.texID], tex[m.texID].offsetColor, texImage, uv, color);
    if (m.emissiveTexID != -1) {
      glm::vec3 emissiveColor;
      FetchTexture(tex[m.emissiveTexID], tex[m.texID].offsetEmissive, emissiveTexture, uv, emissiveColor);

      if (emissiveColor != glm::vec3(0.f)) {
        color = emissiveColor;

        lightSourceSampled[idx] = glm::vec4(intersect, 1);

        if (pathSegment.remainingBounces != traceDepth) {
          color = 40.f*color;
        }
        pathSegment.remainingBounces = 0;
      }
    }
  }

  pathSegment.isDifuse = 0;

  // TODO: implement this.
  thrust::uniform_real_distribution<float> u01(0, 1);
  if (m.hasRefractive) {
    float ior = m.indexOfRefraction;
    float cosAngle = glm::dot(pathSegment.ray.direction, -normal);
    float reflectCoeff = ((1 - ior) / (1 + ior)) * ((1 - ior) / (1 + ior));
    reflectCoeff = reflectCoeff + (1.f - reflectCoeff) * powf(1.f - cosAngle, 5);

    if (u01(rng) < reflectCoeff) {
      glm::vec3 reflect = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
      glm::vec3 imperfect = glm::normalize(calculateImperfectSpecularDirection(reflect, m.hasReflective, rng));

      pathSegment.ray.direction = m.hasReflective * reflect + (1 - m.hasReflective) * imperfect;

      pathSegment.color *= (color);
      pathSegment.ray.origin = intersect;
    }
    else {
      pathSegment.ray.direction = glm::normalize(glm::refract(pathSegment.ray.direction, normal, cosAngle > 0.f ? 1.f / ior : ior));
      pathSegment.color *= (color);
      pathSegment.ray.origin = intersect + 0.001f * glm::normalize(pathSegment.ray.direction);
      // pathSegment.ray.origin = intersect;
    }
  }
  else if (m.hasReflective > 0) {
    glm::vec3 reflect = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
    glm::vec3 imperfect = glm::normalize(calculateImperfectSpecularDirection(reflect, m.hasReflective, rng));

    pathSegment.ray.direction = m.hasReflective * reflect + (1 - m.hasReflective) * imperfect;

    
    pathSegment.color *= (color);
    pathSegment.ray.origin = intersect;
  }
  else {
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
    pathSegment.color *= (color);
    pathSegment.ray.origin = intersect;
    pathSegment.isDifuse = 1;
  }
  


}
