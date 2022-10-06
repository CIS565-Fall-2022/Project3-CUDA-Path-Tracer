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
    /*if (!outside) {
        normal = -normal;
    }*/

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ void swapFloat(float& a, float& b)
{
    float tmp = a;
    a = b;
    b = tmp;
}

__host__ __device__ bool boundingBoxCheck(glm::vec3 origin, glm::vec3 direction, glm::vec3 min, glm::vec3 max)
{
    glm::vec3 inverse_direction = 1.0f / direction;

    float tMin = (min.x - origin.x) * inverse_direction.x;
    float tMax = (max.x - origin.x) * inverse_direction.x;

    if (tMin > tMax) { swapFloat(tMin, tMax); }

    float tyMin = (min.y - origin.y) * inverse_direction.y;
    float tyMax = (max.y - origin.y) * inverse_direction.y;
    
    if (tyMin > tyMax) { swapFloat(tyMin, tyMax); }

    if (tMin > tyMax || tyMin > tMax) { return false; }

    if (tyMin > tMin) { tMin = tyMin; }
    if (tyMax < tMax) { tMax = tyMax; }

    float tzMin = (min.z - origin.z) * inverse_direction.z;
    float tzMax = (max.z - origin.z) * inverse_direction.z;

    if (tzMin > tzMax) { swapFloat(tzMin, tzMax); }

    if (tMin > tzMax || tzMin > tMax) { return false; }

    /*if (tzMin > tMin) { tMin = tzMin; }
    if (tzMax < tMax) { tMax = tzMax; }*/

    return true;
}


__host__ __device__ float MeshIntersectionTest(Geom mesh, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, Triangle* allTriangles) {
    Ray q;
    q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    //check bounding box first
    if (!boundingBoxCheck(q.origin, q.direction, mesh.boxMin, mesh.boxMax))
    {
        return -1;
    }

    float t = FLT_MAX;
    glm::vec3 temp_normal;

    // test for tirangles belong to this mesh
    for (int i = mesh.triangleStartIndex; i < mesh.totaltriangles; i++)
    {
        glm::vec3 bary;
        if (!glm::intersectRayTriangle(q.origin, q.direction, allTriangles[i].pos1, allTriangles[i].pos2, allTriangles[i].pos3, bary))
        {
            continue;
        }
        if (bary.z > t) { continue; } // this one is closer


        t = bary.z;
        float z = 1.0f - bary.x - bary.y;

        //glm::vec3 intersection_current = bary.x * tri->pos1 + bary.y * tri->pos2 + z * tri->pos3;
        if (allTriangles[i].normal1 == glm::vec3(0))// no normal imported
        {
            temp_normal = glm::normalize(glm::cross(allTriangles[i].pos2 - allTriangles[i].pos1, allTriangles[i].pos3 - allTriangles[i].pos1));
        }
        else
        {
            temp_normal = glm::normalize(bary.x * allTriangles[i].normal1 + bary.y * allTriangles[i].normal2 + z * allTriangles[i].normal3);
        }
    }

    if (t == FLT_MAX)
    {
        //no intersection
        return -1;
    }
    // determine if outside
    if (glm::dot(temp_normal, q.direction)>0)
    {
        //it is from inside
        outside = false;
        //temp_normal = -temp_normal;
    }
    else {
        outside = true;
    }
    intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(q, t), 1.0f));
    normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(temp_normal,1.0f)));
    return glm::length(r.origin - intersectionPoint);
}


__host__ __device__ glm::vec3 randomPointOnCube(Geom cube, thrust::default_random_engine& rng)
{
    thrust::uniform_real_distribution<float> u1(-0.5, 0.5);
    glm::vec3 pos(u1(rng), u1(rng), u1(rng));
    return glm::vec3(cube.transform * glm::vec4(pos, 1.f));
}

__host__ __device__ void generateRayToCube(Ray& r, Geom cube, thrust::default_random_engine& rng)
{
    glm::vec3 target = randomPointOnCube(cube, rng);
    r.direction = glm::normalize(target - r.origin);
}