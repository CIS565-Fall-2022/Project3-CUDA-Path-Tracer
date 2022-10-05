#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "consts.h"

__host__ __device__  bool AABBIntersect(AABB const& a, AABB const& b);
__host__ __device__  bool AABBRayIntersect(AABB const& aabb, Ray const& r, float* t);
__host__ __device__  bool AABBPointIntersect(AABB const& aabb, glm::vec3 const& point);
template<size_t N>
__host__ __device__ __forceinline__ void project(glm::vec3 axis, glm::vec3 const(&pos)[N], float& tmin, float& tmax) {
    tmin = LARGE_FLOAT, tmax = SMALL_FLOAT;
    axis = glm::normalize(axis);

#pragma unroll
    for (int i = 0; i < N; ++i) {
        float t = glm::dot(pos[i], axis);
        tmin = fmin(tmin, t);
        tmax = fmax(tmax, t);
    }
}
// reference: https://stackoverflow.com/questions/17458562/efficient-aabb-triangle-intersection-in-c-sharp
__host__ __device__  bool AABBTriangleIntersect(AABB const& a, glm::vec3 const(&tri_verts)[3]);
/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ __forceinline__ unsigned int utilhash(unsigned int a) {
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
__host__ __device__ __forceinline__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/// <summary>
/// lerp between 3 vectors based on barycentric coord
/// </summary>
/// <param name="bary">barycentric coord</param>
/// <param name="vecs">3 vectors</param>
/// <returns>the interpolated vector</returns>
template<typename T>
__host__ __device__ __forceinline__ T lerpBarycentric(glm::vec2 bary, T const(&vecs)[3]) {
    float u = (1.0f - bary.x - bary.y), v = bary.x, w = bary.y;
    return u * vecs[0] + v * vecs[1] + w * vecs[2];
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ __forceinline__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
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
__host__ __device__  float boxIntersectionTest(Geom box, Ray r, ShadeableIntersection& inters);

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
__host__ __device__  float sphereIntersectionTest(Geom sphere, Ray r, ShadeableIntersection& inters);

// fills the ShadeableIntersection from triangle hit information
__device__  float intersFromTriangle(
    ShadeableIntersection& inters,
    Ray const& ray,
    float hit_t,
    MeshInfo const& meshInfo,
    Geom const& mesh,
    Triangle const& tri,
    glm::vec2 barycoord);

/**
 * Test intersection between a ray and a transformed mesh.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__  float meshIntersectionTest(Geom mesh, Ray r, MeshInfo meshInfo, ShadeableIntersection& inters);