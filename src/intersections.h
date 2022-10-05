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

__host__ __device__ glm::vec3 calculateNormal(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2) {
    glm::vec3 v1 = p1 - p0;
    glm::vec3 v2 = p2 - p0;
    return glm::normalize(glm::cross(v1, v2));
}

__host__ __device__
bool rayBoundingBoxIntersect(glm::vec3 origin, glm::vec3 dir, glm::vec3 aabbMin, glm::vec3 aabbMax)
{
    float tmin = (aabbMin.x - origin.x) / dir.x;
    float tmax = (aabbMax.x - origin.x) / dir.x;

    if (tmin > tmax) swap(tmin, tmax);

    float tymin = (aabbMin.y - origin.y) / dir.y;
    float tymax = (aabbMax.y - origin.y) / dir.y;

    if (tymin > tymax) swap(tymin, tymax);

    if ((tmin > tymax) || (tymin > tmax))
        return false;

    if (tymin > tmin)
        tmin = tymin;

    if (tymax < tmax)
        tmax = tymax;

    float tzmin = (aabbMin.z - origin.z) / dir.z;
    float tzmax = (aabbMax.z - origin.z) / dir.z;

    if (tzmin > tzmax) swap(tzmin, tzmax);

    if ((tmin > tzmax) || (tzmin > tmax))
        return false;

    if (tzmin > tmin)
        tmin = tzmin;

    if (tzmax < tmax)
        tmax = tzmax;

    return true;
}

__host__ __device__ 
float meshIntersectionTest(Geom mesh, Ray r,
                                                 glm::vec3& intersectionPoint,
                                                 glm::vec3& normal, 
                                                 Triangle* tirangles,
                                                 bool& outside)
{
#if USE_BOUNDING_BOX
    //bool isIntersect = rayBoundingBoxIntersect(r.)
#endif

    glm::vec3 intersectionBaryPos;
    float tMin = FLT_MAX;
    glm::vec3 tMinNormal;
    bool intersected = false;

    // do intersection test with every triangle
    for (int i = 0; i < mesh.numTris; ++i)
    {
        Triangle tri = tirangles[i];
        // transform the triangle position from object to world coordinate
        glm::vec3 p0 = multiplyMV(mesh.transform, glm::vec4(tri.p0, 1.0f));
        glm::vec3 p1 = multiplyMV(mesh.transform, glm::vec4(tri.p1, 1.0f));
        glm::vec3 p2 = multiplyMV(mesh.transform, glm::vec4(tri.p2, 1.0f));

       /* glm::vec3 n0 = multiplyMV(mesh.transform, glm::vec4(tri.n0, 0.0f));
        glm::vec3 n1 = multiplyMV(mesh.transform, glm::vec4(tri.n1, 0.0f));
        glm::vec3 n2 = multiplyMV(mesh.transform, glm::vec4(tri.n2, 0.0f));*/

        // to intersection test
        bool intersection = glm::intersectRayTriangle(r.origin, r.direction, p0, p1, p2, intersectionBaryPos);
        if (!intersection) {
            continue;
        }
        // if intersect calculate the distance
        float t = glm::length(r.origin - intersectionBaryPos);
        // use barycentric to calculate the normal
        //glm::vec3 intersectionNormal = glm::normalize(intersectionBaryPos.x * tri.n0 + intersectionBaryPos.y * tri.n1 + intersectionBaryPos.z * tri.n2);
        //intersectionNormal = multiplyMV(mesh.transform, glm::vec4(intersectionNormal, 0.0f));

        glm::vec3 intersectionNormal = calculateNormal(p0, p1, p2);
        // record the min distance between ray and triangles as the intersection
        if (t < tMin) {
            tMin = t;
            tMinNormal = intersectionNormal;
            intersected = true;
        }
    }

    // if intersect, return intersectionPoint
    if (intersected) {
        intersectionPoint = getPointOnRay(r, tMin);
        normal = tMinNormal;
        return glm::length(r.origin - intersectionPoint);
    }
    return -1.f;    
}