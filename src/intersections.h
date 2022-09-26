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

/// <summary>
/// lerp between 3 vectors based on barycentric coord
/// </summary>
/// <param name="bary">barycentric coord</param>
/// <param name="vecs">3 vectors</param>
/// <returns>the interpolated vector</returns>
template<typename T>
__host__ __device__ T lerpBarycentric(glm::vec3 bary, T(*vecs)[3]) {
    return (1.0f - bary.x - bary.y) * (*vecs)[0] + bary.x * (*vecs)[1] + bary.y * (*vecs)[2];
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
__host__ __device__ float boxIntersectionTest(Geom box, Ray r, ShadeableIntersection& inters) {
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
        //outside = true;
        //if (tmin <= 0) {
        //    tmin = tmax;
        //    tmin_n = tmax_n;
        //    outside = false;
        //}
        inters.hitPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        inters.surfaceNormal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        inters.materialId = box.materialid;

        return glm::length(r.origin - inters.hitPoint);
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
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r, ShadeableIntersection& inters) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float t;
    if (!glm::intersectRaySphere(ro, rd, glm::vec3(0), radius * radius, t)) {
        return -1;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);
    inters.hitPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    inters.surfaceNormal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    inters.materialId = sphere.materialid;

    return glm::length(r.origin - inters.hitPoint);
}

/**
 * Test intersection between a ray and a transformed mesh.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__ float meshIntersectionTest(Geom mesh, Ray r, MeshInfo meshInfo, ShadeableIntersection& inters) {
    
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_min = FLT_MAX;
    bool intersect = false;
    
    auto const& meshes = meshInfo.meshes;
    auto const& tris = meshInfo.tris;
    auto const& verts = meshInfo.vertices;
    auto const& norms = meshInfo.normals;
    auto const& uvs = meshInfo.uvs;

    glm::vec3 normal;
    glm::vec2 uv;
    int mat_id = -1;

    for (int i = meshes[mesh.meshid].tri_start; i < meshes[mesh.meshid].tri_end; ++i) {
        
        glm::vec3 barycoord;
        glm::vec3 triangle_verts[3]{
            verts[tris[i].verts[0]],
            verts[tris[i].verts[1]],
            verts[tris[i].verts[2]]
        };
        glm::vec3 triangle_norms[3]{
            norms[tris[i].norms[0]],
            norms[tris[i].norms[1]],
            norms[tris[i].norms[2]]
        };
        glm::vec2 triangle_uvs[3]{
            uvs[tris[i].uvs[0]],
            uvs[tris[i].uvs[1]],
            uvs[tris[i].uvs[2]]
        };
        if (glm::intersectRayTriangle(ro, rd, triangle_verts[0], triangle_verts[1], triangle_verts[2], barycoord)) {
            intersect = true;
            if (barycoord.z > t_min) {
                continue;
            }
            
            t_min = barycoord.z;
            normal = lerpBarycentric(barycoord, &triangle_norms);
            uv = lerpBarycentric(barycoord, &triangle_uvs);
            mat_id = tris[i].mat_id;
        }
    }

    if (!intersect) {
        return -1;
    }

    // transform to world space and store results
    Ray local_ray;
    local_ray.origin = ro;
    local_ray.direction = rd;

    inters.hitPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(local_ray, t_min), 1));
    inters.surfaceNormal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(normal, 0)));
    inters.uv = uv;
    inters.materialId = mat_id;

    return glm::length(r.origin - inters.hitPoint);
}