#pragma once
#include <cuda.h>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include "Collision/AABB.h"

template<size_t N>
__host__ __device__ inline void project(glm::vec3 axis, glm::vec3 const(&pos)[N], float& tmin, float& tmax) {
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
__host__ __device__ inline bool AABBTriangleIntersect(AABB const& a, glm::vec3 const(&tri_verts)[3]);
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
__host__ __device__ inline glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/// <summary>
/// lerp between 3 vectors based on barycentric coord
/// </summary>
/// <param name="bary">barycentric coord</param>
/// <param name="vecs">3 vectors</param>
/// <returns>the interpolated vector</returns>
template<typename T>
__host__ __device__ inline T lerpBarycentric(glm::vec2 bary, T const(&vecs)[3]) {
    float u = (1.0f - bary.x - bary.y), v = bary.x, w = bary.y;
    return u * vecs[0] + v * vecs[1] + w * vecs[2];
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ inline glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

__host__ __device__ inline bool AABBIntersect(AABB const& a, AABB const& b) {
    glm::vec3 d1 = a.min() - b.max();
    glm::vec3 d2 = b.min() - a.max();
    return (d1.x <= 0 && d2.x <= 0) && (d1.y <= 0 && d2.y <= 0) && (d1.z <= 0 && d2.z <= 0);
}
__host__ __device__ inline bool AABBRayIntersect(AABB const& aabb, Ray const& r, float* t) {
    float tmin = SMALL_FLOAT;
    float tmax = LARGE_FLOAT;

    glm::vec3 tmins = (aabb.min() - r.origin) / r.direction;
    glm::vec3 tmaxs = (aabb.max() - r.origin) / r.direction;

#pragma unroll
    for (int i = 0; i < 3; ++i) {
        if (fabsf(r.direction[i]) < EPSILON) {
            if (r.origin[i] < aabb.min()[i] || r.origin[i] > aabb.max()[i]) {
                return false;
            }
        } else {
            float t0 = tmins[i];
            float t1 = tmaxs[i];
            if (t0 > t1) {
                float tmp = t0;
                t0 = t1;
                t1 = tmp;
            }
            if (tmin > t1 || tmax < t0) {
                return false;
            }
            tmin = fmax(t0, tmin);
            tmax = fmin(t1, tmax);
            if (tmin > tmax) {
                return false;
            }
        }
    }

    if (tmin < 0 && tmax < 0) {
        return false;
    }

    if (t) {
        if (tmin < 0) {
            *t = tmax;
        } else {
            *t = tmin;
        }
    }

    return true;
}
__host__ __device__ inline bool AABBPointIntersect(AABB const& aabb, glm::vec3 const& point) {
    bool contained = true;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        if (point[j] > aabb.max()[j] || point[j] < aabb.min()[j]) {
            contained = false;
            break;
        }
    }
    return contained;
}

__host__ __device__ inline bool AABBTriangleIntersect(AABB const& a, glm::vec3 const(&tri_verts)[3]) {
    bool contained = true;
    for (int i = 0; i < 3 && contained; ++i) {
        if (!AABBPointIntersect(a, tri_verts[i])) {
            contained = false;
        }
    }
    if (contained) {
        return true;
    }

    // reduce to a loose & inaccurate AABB-AABB test instead
    // because I run out of time trying to debug this
    // this function is only used by the octree anyways
    glm::vec3 min_tri(FLT_MAX), max_tri(FLT_MIN);
    for (int i = 0; i < 3; ++i) {
        min_tri = glm::min(min_tri, tri_verts[i]);
        max_tri = glm::max(max_tri, tri_verts[i]);
    }
    return AABBIntersect(a, AABB(min_tri, max_tri));

#ifdef STACK_OVERFLOW_IMPL
    // reference: https://stackoverflow.com/questions/17458562/efficient-aabb-triangle-intersection-in-c-sharp

    float tri_min, tri_max;
    float box_min, box_max;

    glm::vec3 box_verts[8];
    a.vertices(box_verts, true);
    glm::vec3 tri_edges[3] = {
        tri_verts[0] - tri_verts[1],
        tri_verts[1] - tri_verts[2],
        tri_verts[2] - tri_verts[0]
    };
    glm::vec3 tri_normal = glm::normalize(glm::cross(tri_edges[0], tri_edges[1]));
    glm::vec3 box_normals[3] = { {1,0,0}, {0,1,0}, {0,0,1} };

    // test box normal
#pragma unroll
    for (int i = 0; i < 3; i++) {
        project(box_normals[i], tri_verts, tri_min, tri_max);
        if (tri_max < a.min()[i] || tri_min > a.max()[i]) {
            return false;
        }
    }
    // test triangle normal
    float offset = glm::dot(tri_normal, tri_verts[0]);
    project(tri_normal, box_verts, box_min, box_max);
    if (box_max < offset || box_min > offset) {
        return false;
    }

    // SAT test
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            glm::vec3 axis = glm::cross(tri_edges[i], box_normals[j]);
            project(axis, box_verts, box_min, box_max);
            project(axis, tri_verts, tri_min, tri_max);
            if (box_max < tri_min || box_min > tri_max) {
                return false;
            }
        }
    }
    return true;
#endif
}

/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ inline float boxIntersectionTest(Geom box, Ray r, ShadeableIntersection& inters) {
    Ray q;
    q.origin = multiplyMV(box.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-PRIM_CUBE_EXTENT - q.origin[xyz]) / qdxyz;
            float t2 = (+PRIM_CUBE_EXTENT - q.origin[xyz]) / qdxyz;
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

        return inters.t = glm::length(r.origin - inters.hitPoint);
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
__host__ __device__ inline float sphereIntersectionTest(Geom sphere, Ray r, ShadeableIntersection& inters) {
#ifdef USE_GLM_RAY_SPHERE
    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));
    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float t;
    if (!glm::intersectRaySphere(ro, rd, glm::vec3(0), PRIM_SPHERE_RADIUS * PRIM_SPHERE_RADIUS, t)) {
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
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(PRIM_SPHERE_RADIUS, 2));
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
        t = fmin(t1, t2);
        outside = true;
    } else {
        t = fmax(t1, t2);
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

    return inters.t = glm::length(r.origin - inters.hitPoint);
}

// fills the ShadeableIntersection from triangle hit information
__device__ inline float intersFromTriangle(
    ShadeableIntersection& inters,
    Ray const& ray,
    float hit_t,
    MeshInfo const& meshInfo,
    Geom const& mesh,
    Triangle const& tri,
    glm::vec2 barycoord)
{
    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(ray.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(ray.direction, 0.0f)));
    Ray local_ray;
    local_ray.origin = ro;
    local_ray.direction = rd;

    auto const& meshes = meshInfo.meshes;
    auto const& tris = meshInfo.tris;
    auto const& verts = meshInfo.vertices;
    auto const& norms = meshInfo.normals;
    auto const& uvs = meshInfo.uvs;
    auto const& materials = meshInfo.materials;

    glm::vec3 normal;
    glm::vec2 uv;
    int mat_id = -1;

    bool has_uv = true;
    glm::vec3 triangle_verts[3]{
        verts[tri.verts[0]],
        verts[tri.verts[1]],
        verts[tri.verts[2]]
    };
    glm::vec3 triangle_norms[3]{
        norms[tri.norms[0]],
        norms[tri.norms[1]],
        norms[tri.norms[2]]
    };
    glm::vec2 triangle_uvs[3];
#pragma unroll
    for (int x = 0; x < 3; ++x) {
        if (tri.uvs[x] != -1) {
            triangle_uvs[x] = uvs[tri.uvs[x]];
        } else {
            has_uv = false;
            break;
        }
    }

    // record uv info
    if (has_uv) {
        uv = lerpBarycentric(barycoord, triangle_uvs);
    } else {
        uv = glm::vec2(-1);
    }

    // record material id
    mat_id = tri.mat_id;

    // record normal info
    // use bump mapping if applicable
    if (mat_id != -1 && has_uv &&
        materials[mat_id].textures.bump != -1) {
        glm::vec3 tans[3];
        glm::vec3 bitans[3];

#pragma unroll
        for (int x = 0; x < 3; ++x) {
            glm::vec4 const& tmp = meshInfo.tangents[tri.tangents[x]];
            tans[x] = glm::vec3(tmp);
            bitans[x] = glm::cross(triangle_norms[x], tans[x]) * tmp.w;
        }

        glm::vec3 tan = glm::normalize(lerpBarycentric(barycoord, tans));
        glm::vec3 anorm = glm::normalize(lerpBarycentric(barycoord, triangle_norms));
        glm::vec3 bitan = glm::normalize(lerpBarycentric(barycoord, bitans));

        // glm::vec3 bitan = glm::normalize(glm::cross(tan, anorm));

        Material const& mat = materials[mat_id];
        normal = meshInfo.texs[mat.textures.bump].sample(uv);
        normal = glm::normalize(normal * 2.0f - 1.0f);
        normal = glm::mat3x3(tan, bitan, anorm) * normal;
    } else {
        normal = lerpBarycentric(barycoord, triangle_norms);
    }

    normal = glm::normalize(normal);

    // transform to world space and store results
    inters.hitPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(local_ray, hit_t), 1));
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

    return inters.t = glm::length(ray.origin - inters.hitPoint);
}

/**
 * Test intersection between a ray and a transformed mesh.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__device__ inline float meshIntersectionTest(Geom mesh, Ray r, MeshInfo meshInfo, ShadeableIntersection& inters) {

    glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float t_min = FLT_MAX;

    auto const& meshes = meshInfo.meshes;
    auto const& tris = meshInfo.tris;
    auto const& verts = meshInfo.vertices;

    int idx = -1;
    glm::vec3 barycoord;

    for (int i = meshes[mesh.meshid].tri_start; i < meshes[mesh.meshid].tri_end; ++i) {
        glm::vec3 tmp_barycoord;
        glm::vec3 triangle_verts[3]{
            verts[tris[i].verts[0]],
            verts[tris[i].verts[1]],
            verts[tris[i].verts[2]]
        };

        if (glm::intersectRayTriangle(ro, rd, triangle_verts[0], triangle_verts[1], triangle_verts[2], tmp_barycoord)) {
            if (tmp_barycoord.z > t_min) {
                continue;
            }
            idx = i;
            barycoord = tmp_barycoord;
            t_min = tmp_barycoord.z;
        }
    }

    if (idx == -1) {
        return -1;
    }
    return inters.t = intersFromTriangle(inters, r, t_min, meshInfo, mesh, tris[idx], glm::vec2(barycoord));
}