#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}



// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ int boundingBoxIntersection(const Ray r, const glm::vec3& boundingMax, const glm::vec3& boundingMin) {
  float tmin = -1e38f;
  float tmax = 1e38f;
  for (int xyz = 0; xyz < 3; ++xyz) {
    float qdxyz = r.direction[xyz];
    /*if (glm::abs(qdxyz) > 0.00001f)*/ {
      float t1 = (boundingMin[xyz] - r.origin[xyz]) / qdxyz;
      float t2 = (boundingMax[xyz] - r.origin[xyz]) / qdxyz;
      
      tmin = max(tmin, min(t1, t2));
      tmax = min(tmax, max(t1, t2));
    }
  }

  return tmax > max(tmin, 0.f);
}

/**
 * Test intersection between a ray and a transformed mesh. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom geom, const SceneMeshesData sceneMeshesData, Ray r,
  glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, glm::vec4& tangent) {
  float t = FLT_MAX;

  glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

  Ray rt;
  rt.origin = ro;
  rt.direction = rd;

  if (!boundingBoxIntersection(rt, geom.boundingMax, geom.boundingMin)) {
    return -1;
  }

  for (int i = 0; i < geom.count/3; i++) {
    //auto f0 = sceneMeshesData.indices[3 * i + 0];
    //auto f1 = sceneMeshesData.indices[3 * i + 1];
    //auto f2 = sceneMeshesData.indices[3 * i + 2];

    int idx = i * 3 + geom.startIndex;
    // get the 3 normal vectors for that face
    glm::vec3 v0, v1, v2;
    v0 = sceneMeshesData.positions[idx + 0];
    v1 = sceneMeshesData.positions[idx + 1];
    v2 = sceneMeshesData.positions[idx + 2];

    glm::vec3 bary;
    if (glm::intersectRayTriangle(ro, rd, v0, v1, v2, bary) && bary.z < t) {
      glm::vec3 n0, n1, n2;
      n0 = sceneMeshesData.normals[idx + 0];
      n1 = sceneMeshesData.normals[idx + 1];
      n2 = sceneMeshesData.normals[idx + 2];
      normal = (1 - bary.x - bary.y) * n0 + bary.x * n1 + bary.y * n2;
      
      if (geom.useTex == 1) {
        glm::vec2 uv0, uv1, uv2;
        uv0 = sceneMeshesData.uvs[idx + 0];
        uv1 = sceneMeshesData.uvs[idx + 1];
        uv2 = sceneMeshesData.uvs[idx + 2];
        uv = (1 - bary.x - bary.y) * uv0 + bary.x * uv1 + bary.y * uv2;
      }

      if (geom.hasTangent) {
        glm::vec4 t0, t1, t2;
        t0 = sceneMeshesData.tangents[idx + 0];
        t1 = sceneMeshesData.tangents[idx + 1];
        t2 = sceneMeshesData.tangents[idx + 2];
        tangent = (1 - bary.x - bary.y) * t0 + bary.x * t1 + bary.y * t2;
      }

      t = bary.z;
    }
  }

  glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

  intersectionPoint = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.f));
  normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.f)));

  return glm::length(r.origin - intersectionPoint);
}