#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define CHECK_ZERO(a) (a > -0.00001f && a < 0.00001f)



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

__host__ __device__ glm::vec3 calcTriangleNormal(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2)
{
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;

    return glm::normalize(glm::cross(edge1, edge2));
}

template<typename VALUETYPE>
__host__ __device__ VALUETYPE barycentricInterpolation(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 point,
    VALUETYPE value0, VALUETYPE value1, VALUETYPE value2)
{
    float S = 0.5f * glm::length(glm::cross(v0 - v1, v2 - v1));
    float S0 = 0.5f * glm::length(glm::cross(v1 - point, v2 - point));
    float S1 = 0.5f * glm::length(glm::cross(v0 - point, v2 - point));
    float S2 = 0.5f * glm::length(glm::cross(v0 - point, v1 - point));
    return (S0 / S) * value0 + (S1 / S) * value1 + (S2 / S) * value2;
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


__host__ __device__ float objIntersectionTest(Geom geom, Ray r, Triangle *triangles, 
    LinearBVHNode* bvhNodes, int *bvhArrayToUse, int pixelIndex,
        glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, int& triangleId, bool& outside)
{
    float minT = BIG_FLOAT;

    // Ray to local space
    Ray localRay;
    localRay.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    localRay.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

#if ENABLE_BVH

    glm::vec3 invRayDir;
    invRayDir.x = CHECK_ZERO(localRay.direction.x) ? 0.f : 1.f / localRay.direction.x;
    invRayDir.y = CHECK_ZERO(localRay.direction.y) ? 0.f : 1.f / localRay.direction.y;
    invRayDir.z = CHECK_ZERO(localRay.direction.z) ? 0.f : 1.f / localRay.direction.z;

    int dirIsNeg[3] = { invRayDir.x < 0, invRayDir.y < 0, invRayDir.z < 0};

    int arrayIndexSt = pixelIndex * BVH_INTERSECT_STACK_SIZE;
    int toVisitOffset = 0, currNodeIndex = geom.bvhNodeStartIndex;

    for (int i = arrayIndexSt; i < arrayIndexSt + BVH_INTERSECT_STACK_SIZE; ++i)
    {
        bvhArrayToUse[i] = -1;
    }

    while (currNodeIndex >= geom.bvhNodeStartIndex && currNodeIndex < geom.bvhNodeEndIndex)
    {
        const LinearBVHNode node = bvhNodes[currNodeIndex];    

        // Bound ray intersection
        bool hasIntersect = true;
        float t0 = 0, t1 = BIG_FLOAT;
        for (int i = 0; i < 3; ++i)
        {
            float tMin = (node.bound.pMin[i] - localRay.origin[i]) * invRayDir[i];
            float tMax = (node.bound.pMax[i] - localRay.origin[i]) * invRayDir[i];
            if (tMin > tMax)
            {
                float tmp = tMin;
                tMin = tMax;
                tMax = tmp;
            }

            t0 = tMin > t0 ? tMin : t0;
            t1 = tMax < t1 ? tMax : t1;

            if (t0 > t1)
            {
                hasIntersect = false;
                break;
            }
        }

        if(hasIntersect)
        {    
            if (node.nPrimitives > 0)
            {
                // When BVH Node is leaf node
                for (int i = 0; i < node.nPrimitives; ++i)
                {
                    Triangle triangle = triangles[geom.triangleStartIndex + node.firstPrimOffset + i];
                    glm::vec3 baryPos;
                    glm::vec3 localIntersectionPoint;
                    if (glm::intersectRayTriangle(localRay.origin, localRay.direction,
                        triangle.v0, triangle.v1, triangle.v2, baryPos))
                    {
                        // Barycentric to normal coordinate
                        localIntersectionPoint = (1 - baryPos[0] - baryPos[1]) * triangle.v0 +
                            baryPos[0] * triangle.v1 + baryPos[1] * triangle.v2;
                   
                        float localT = glm::length(localIntersectionPoint - localRay.origin);
                        if (localT > minT)
                        {
                            continue;
                        }
                        minT = localT;

                        glm::vec3 localNormal;
                        if (geom.hasNormal)
                        {
                            localNormal = barycentricInterpolation<glm::vec3>(triangle.v0, triangle.v1, triangle.v2, localIntersectionPoint,
                                triangle.n0, triangle.n1, triangle.n2);
                        }
                        else
                        {
                            localNormal = calcTriangleNormal(triangle.v0, triangle.v1, triangle.v2);
                        }

                        // UV
                        if (geom.hasUV)
                        {
                            uv = barycentricInterpolation<glm::vec2>(triangle.v0, triangle.v1, triangle.v2, localIntersectionPoint,
                                triangle.tex0, triangle.tex1, triangle.tex2);
                        }

                        // Get value in world space
                        outside = glm::dot(localNormal, localRay.direction) < 0;
                        normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0.f)));
                        intersectionPoint = multiplyMV(geom.transform, glm::vec4(localIntersectionPoint, 1.f));
                        triangleId = geom.triangleStartIndex + node.firstPrimOffset + i;
                    }
                }

                if (toVisitOffset <= 0) break;
                currNodeIndex = bvhArrayToUse[--toVisitOffset + arrayIndexSt];
            }
            else
            {
                // When BVH Node is interior node
                if (dirIsNeg[node.axis])
                {                   
                    bvhArrayToUse[arrayIndexSt + toVisitOffset++] = currNodeIndex + 1;  // Insert left child
                    currNodeIndex = node.rightChildOffset + geom.bvhNodeStartIndex;                              // Visit right child first
                }
                else
                {
                    bvhArrayToUse[arrayIndexSt + toVisitOffset++] = node.rightChildOffset + geom.bvhNodeStartIndex;
                    currNodeIndex = currNodeIndex + 1;
                }
            }
        }
        else
        {
            if (toVisitOffset <= 0) break;
            currNodeIndex = bvhArrayToUse[--toVisitOffset + arrayIndexSt];
        }
    }

#else

    for (int i = geom.triangleStartIndex; i < geom.triangleEndIndex; ++i)
    {
        Triangle triangle = triangles[i];
        
        glm::vec3 baryPos;
        glm::vec3 localIntersectionPoint;
        if (glm::intersectRayTriangle(localRay.origin, localRay.direction, 
            triangle.v0, triangle.v1, triangle.v2, baryPos))
        {
            // Barycentric to normal coordinate
            localIntersectionPoint = (1 - baryPos[0] - baryPos[1]) * triangle.v0 +
                baryPos[0] * triangle.v1 + baryPos[1] *triangle.v2;

            float localT = glm::length(localIntersectionPoint - localRay.origin);
            if (localT > minT)
            {
                continue;
            }
            minT = localT;

            glm::vec3 localNormal;
            if (geom.hasNormal)
            {
                localNormal = barycentricInterpolation<glm::vec3>(triangle.v0, triangle.v1, triangle.v2, localIntersectionPoint,
                    triangle.n0, triangle.n1, triangle.n2);
            }
            else
            {
                localNormal = calcTriangleNormal(triangle.v0, triangle.v1, triangle.v2);
            }

            // UV
            if (geom.hasUV)
            {
                uv = barycentricInterpolation<glm::vec2>(triangle.v0, triangle.v1, triangle.v2, localIntersectionPoint,
                    triangle.tex0, triangle.tex1, triangle.tex2);
            }

            // Get value in world space
            outside = glm::dot(localNormal, localRay.direction) < 0;
            normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(localNormal, 0.f)));
            intersectionPoint = multiplyMV(geom.transform, glm::vec4(localIntersectionPoint, 1.f));
            triangleId = i;
        }
    }
#endif

    if (CHECK_ZERO(minT - BIG_FLOAT))
    {
        return -1;
    }
    else
    {
        if (!outside) {
            normal = -normal;
        }

        return minT;
    }  
}