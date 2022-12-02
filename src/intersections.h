#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include <glm\gtc\matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/matrix_inverse.hpp>



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



//Moller-Trumbore Algorithm for ray triangle intersection
__host__ __device__
bool rayTriangleIntersectionTest(glm::vec3 &p0, glm::vec3 &p1, glm::vec3 &p2,
    glm::vec3 &O, glm::vec3 &D, glm::vec3 &result)
{
    glm::vec3 E1 = p1 - p0;
    glm::vec3 E2 = p2 - p0;
    glm::vec3 S = O - p0;
    glm::vec3 S1 = glm::cross(D, E2);
    glm::vec3 S2 = glm::cross(S, E1);
    float S2E2 = glm::dot(S2, E2);
    float S1S = glm::dot(S1, S);
    float S2D = glm::dot(S2, D);
    glm::vec3 bary = 1 / (glm::dot(S1, E1)) * glm::vec3(S2E2, S1S, S2D);
    float b1 = bary.y; float b2 = bary.z;
    if (b1 < 0 || b2 < 0 || (1 - b1 - b2) < 0)
        return false;
    else {
        result = bary;
        return true;
    }
}


__host__ __device__ 
float triangleIntersectionTest(Geom triangle, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside)
{

    Ray q;
    q.origin = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));
    
    glm::vec3 barycentric; //0: b1, 1: b2, 2: t
    bool inter = glm::intersectRayTriangle(q.origin, q.direction, triangle.pos[0], triangle.pos[1], triangle.pos[2], barycentric);
    
    if (!inter) return -1;

    float b1 = barycentric[0], b2 = barycentric[1], t = barycentric[2];
    //intersectionPoint = b1 * triangle.pos[0] + b2 * triangle.pos[1] + (1 - b1 - b2) * triangle.pos[2];
    normal = b1 * triangle.normal[0] + b2 * triangle.normal[1] + (1 - b1 - b2) * triangle.normal[2];
    uv = b1 * triangle.uv[0] + b2 * triangle.uv[1] + (1 - b1 - b2) * triangle.uv[2];

    glm::vec3 objspaceIntersection = getPointOnRay(q, t);

    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(normal, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ 
float meshIntersectionTest(Geom mesh, Ray r, Geom* triangle, int tri_size, bool aabb,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside)
{
    int start = mesh.obj_start_offset;
    int end = start + mesh.obj_end;

    if (!aabb) {
        float t_min = FLT_MAX;
        float temp_t = 0.0f;

        glm::vec3 t_intersetion;
        glm::vec3 t_normal;
        glm::vec2 t_uv;
        bool t_outside;

        for (int i = start; i < end; i++) {
            temp_t = triangleIntersectionTest(triangle[i], r, t_intersetion, t_normal, t_uv, t_outside);
            if (temp_t != -1) {
                t_min = temp_t;
                break;
            }
        }

        intersectionPoint = t_intersetion;
        normal = t_normal;
        uv = t_uv;
        outside = t_outside;
        return t_min;
    }
    else {
        if (!mesh.bbox.IntersectP(r)) return -1;
        else {
            float t_min = FLT_MAX;
            float temp_t = 0.0f;

            glm::vec3 t_intersetion;
            glm::vec3 t_normal;
            glm::vec2 t_uv;
            bool t_outside;
            for (int i = start; i < end; i++) {
                temp_t = triangleIntersectionTest(triangle[i], r, t_intersetion, t_normal, t_uv, t_outside);
                //if (temp_t < t_min)
                //    t_min = temp_t;
                if (temp_t != -1) {
                    t_min = temp_t;
                    break;
                }
            }

            intersectionPoint = t_intersetion;
            normal = t_normal;
            uv = t_uv;
            outside = t_outside;
            return t_min;
        }
    }
    return -1;
}

__host__ __device__
float triangleIntersectionTest(Tri triangle, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, bool& outside)
{

    Ray q;
    q.origin = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 barycentric; //0: b1, 1: b2, 2: t
    bool inter = glm::intersectRayTriangle(q.origin, q.direction, triangle.p0, triangle.p1, triangle.p2, barycentric);

    if (!inter) return -1;

    float b1 = barycentric[0], b2 = barycentric[1], t = barycentric[2];
    //intersectionPoint = b1 * triangle.pos[0] + b2 * triangle.pos[1] + (1 - b1 - b2) * triangle.pos[2];
    normal = b1 * triangle.n0 + b2 * triangle.n1 + (1 - b1 - b2) * triangle.n2;
    uv = b1 * triangle.t0 + b2 * triangle.t1 + (1 - b1 - b2) * triangle.t2;

    glm::vec3 objspaceIntersection = getPointOnRay(q, t);

    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(normal, 0.f)));

    return glm::length(r.origin - intersectionPoint);
}