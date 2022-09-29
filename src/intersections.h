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
#ifdef USE_GLM_RAY_SPHERE

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float t;
    if (!glm::intersectRaySphere(ro, rd, glm::vec3(0), radius * radius, t)) {
        return -1;
    }

    glm::vec3 point = getPointOnRay(rt, t);

    inters.hitPoint = multiplyMV(sphere.transform, glm::vec4(point, 1.f));
    inters.surfaceNormal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(point, 0.f)));
    inters.materialId = sphere.materialid;

#else
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
    bool outside;
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

    glm::vec3 intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    glm::vec3 normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    inters.hitPoint = intersectionPoint;
    inters.surfaceNormal = normal;
    inters.materialId = sphere.materialid;
#endif // USE_GLM_RAY_SPHERE

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
__device__ float meshIntersectionTest(Geom mesh, Ray r, Material* materials, MeshInfo meshInfo, ShadeableIntersection& inters) {
    
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

        bool has_norm = true, has_uv = true;
        glm::vec3 triangle_norms[3];
        glm::vec2 triangle_uvs[3];

        for (int x = 0; x < 3; ++x) {
            if (tris[i].norms[x] != -1) {
                triangle_norms[x] = norms[tris[i].norms[x]];
            } else {
                has_norm = false;
                break;
            }
        }
        for (int x = 0; x < 3; ++x) {
            if (tris[i].uvs[x] != -1) {
                triangle_uvs[x] = uvs[tris[i].uvs[x]];
            } else {
                has_uv = false;
                break;
            }
        }

        if (glm::intersectRayTriangle(ro, rd, triangle_verts[0], triangle_verts[1], triangle_verts[2], barycoord)) {
            intersect = true;
            if (barycoord.z > t_min) {
                continue;
            }
            
            t_min = barycoord.z;

            if (has_norm) {
                normal = lerpBarycentric(barycoord, &triangle_norms);
            } else {
                glm::vec3 v0v1 = triangle_verts[1] - triangle_verts[0];
                glm::vec3 v0v2 = triangle_verts[2] - triangle_verts[0];
                normal = glm::cross(v0v1, v0v2);
            }
            normal = glm::normalize(normal);

            if (has_uv) {
                uv = lerpBarycentric(barycoord, &triangle_uvs);
            } else {
                uv = glm::vec2(-1);
            }

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

    // use per-mesh material if per-face material is missing
    if (mat_id == -1) {
        inters.materialId = mesh.materialid;
    } else {
        inters.materialId = mat_id;
        // use texture color if applicable
        Material const& mat = materials[mat_id];
        if (mat.textures.diffuse != -1) {
            inters.tex_color = meshInfo.texs[mat.textures.diffuse].sample(uv);
        }
    }

    return glm::length(r.origin - inters.hitPoint);
}