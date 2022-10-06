#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define CULLING 1

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

__host__ __device__ inline void swap(float& a, float& b) {
    float tmp = a;
    a = b;
    b = tmp;
}

__host__ __device__ bool intersectionCheck(glm::vec3 aabb_min, glm::vec3 aabb_max, Ray q) {
    glm::vec3 inv_dir = 1.0f / q.direction;
    float xmin = (aabb_min.x - q.origin.x) * inv_dir.x;
    float xmax = (aabb_max.x - q.origin.x) * inv_dir.x;
    float ymin = (aabb_min.y - q.origin.y) * inv_dir.y;
    float ymax = (aabb_max.y - q.origin.y) * inv_dir.y;
    
    if (xmin > xmax) swap(xmin, xmax);
    if (ymin > ymax) swap(ymin, ymax);
    if (xmin > ymax || ymin > xmax) {
        return false;
    }
    if (ymin > xmin) xmin = ymin;
    if (ymax < xmax) xmax = ymax;

    float zmin = (aabb_min.z - q.origin.z) * inv_dir.z;
    float zmax = (aabb_max.z - q.origin.z) * inv_dir.z;
    if (zmin > zmax) swap(zmin, zmax);

    if (xmin > zmax || xmax < zmin) {
        return false;
    }
    if (zmin > xmin) {
        xmin = zmin;
    }
    if (zmax < xmax) {
        xmax = zmax;
    }
    return true;
}

__host__ __device__ float primitiveIntersectionTest(Geom& geom, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, glm::vec4& tangent, Primitive* prims, Material* material, glm::vec3* texData) {
#if CULLING
    if (!intersectionCheck(geom.aabb_min, geom.aabb_max, r)) {
        return -1;
    }
#endif // CULLING
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 bary;
    float t1 = FLT_MAX;
    bool intersect = false;

    for (int primId = geom.primBegin; primId < geom.primEnd; primId++) {
        const Primitive& prim = prims[primId];
        
        //TODO aabb here?
        if (glm::intersectRayTriangle(q.origin, q.direction, prim.pos[0], prim.pos[1], prim.pos[2], bary)) {
            intersect = true;
            if (bary.z > t1) {
                continue;
            }
            t1 = bary.z;
            bary.z = 1.0f - bary.x - bary.y;
            geom.matId = prim.mat_id;
            //we interpolate three vectors 
            if (prim.hasNormal) {
                normal = glm::normalize(prim.normal[0] * bary.z + prim.normal[1] * bary.x + prim.normal[2] * bary.y);
            }
            if (prim.hasUV) {
                uv = prim.uv[0] * bary.z + prim.uv[1] * bary.x + prim.uv[2] * bary.y;
            }
            else {
                uv = glm::vec2(-1);
            }
            if (prim.hasTangent) {
                tangent = prim.tangent[0] * bary.z + prim.tangent[1] * bary.x + prim.tangent[2] * bary.y;
            }
            else {//or we have to calculate Tangent for the mesh
                //https://www.cs.upc.edu/~virtual/G/1.%20Teoria/06.%20Textures/Tangent%20Space%20Calculation.pdf
                glm::vec3 q1 = prim.pos[1] - prim.pos[0];
                glm::vec3 q2 = prim.pos[2] - prim.pos[0];
                glm::vec2 st1 = prim.uv[1] - prim.uv[0];
                glm::vec2 st2 = prim.uv[2] - prim.uv[0];

                float r = 1.0f / (st1.x * st2.y - st1.y * st2.x);
                glm::vec3 sdir((st2.y * q1.x - st1.y * q2.x) * r, (st2.y * q1.y - st2.x * q2.y) * r, (st2.y * q1.z - st1.y * q2.z) * r);
                glm::vec3 tdir((st1.x * q2.x - st2.x * q1.x) * r, (st1.x * q2.y - st2.x * q1.y) * r, (st1.x * q2.z - st2.x * q1.z) * r);
                //gram-schmidt orthogonalize and calculate handedness
                tangent = glm::vec4(
                    glm::normalize(sdir - normal * glm::dot(normal, sdir)),
                    glm::dot(glm::cross(normal, sdir), tdir) < 0.0f ? -1.0f : 1.0f);
            }
            //do bump mapping if possible
            if (material[prim.mat_id].bump.TexIndex >= 0) {
                glm::vec3 tan = glm::vec3(tangent);
                glm::vec3 bitan = glm::cross(normal, tan) * tangent.w;
                glm::mat3 tbn = glm::mat3(tan, bitan, normal);
                int w = material[prim.mat_id].bump.width;
                int x = uv.x * (w - 1);
                int y = uv.y * (material[prim.mat_id].bump.height - 1);
                normal = texData[material[prim.mat_id].bump.TexIndex + +y * w + x];
                normal = normal * 2.0f - 1.0f;
                normal = glm::normalize(tbn * normal);
            }
        }
    }
    if (!intersect) {
        return -1;
    }
    intersectionPoint = multiplyMV(geom.transform, glm::vec4(getPointOnRay(q, t1), 1.0f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 1.f)));
    tangent = glm::vec4(glm::normalize(multiplyMV(geom.invTranspose, tangent)), tangent.z);
    return glm::length(r.origin - intersectionPoint);
    
}


