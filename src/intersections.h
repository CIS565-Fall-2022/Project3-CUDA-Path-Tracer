#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define BB_CULLING

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

__host__ __device__ float aabbIntersectionTest(AABB aabb, Ray r) {
    glm::vec3 invR = glm::vec3(1.0, 1.0, 1.0) / r.direction;

    float x1 = (aabb.min.x - r.origin.x) * invR.x;
    float x2 = (aabb.max.x - r.origin.x) * invR.x;

    float tmin = min(x1, x2);
    float tmax = max(x1, x2);

    float y1 = (aabb.min.y - r.origin.y) * invR.y;
    float y2 = (aabb.max.y - r.origin.y) * invR.y;

    tmin = min(tmin, min(y1, y2));
    tmax = max(tmin, max(y1, y2));

    float z1 = (aabb.min.z - r.origin.z) * invR.z;
    float z2 = (aabb.max.z - r.origin.z) * invR.z;

    tmin = min(tmin, min(y1, y2));
    tmax = max(tmin, max(y1, y2));

    return tmin <= tmax;
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

/**
 * Test intersection between a ray and a transformed triangle.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float triangleIntersectionTest(Triangle tri, Ray r,
    glm::vec3& barycenter) {

    bool intersect = glm::intersectRayTriangle(r.origin, r.direction,
                                               tri.verts[0], tri.verts[1], tri.verts[2],
                                               barycenter);
    if (!intersect) return -1.f;

    return barycenter.z;
}

/**
 * Test intersection between a ray and a triangle mesh.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
    const Triangle* tris, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {

#ifdef BB_CULLING
    // Test ray against mesh AABB
    bool intersectAABB = aabbIntersectionTest(mesh.aabb, r);
    if (!intersectAABB) return -1.f;
#endif

    // If bounding box is intersected, then check for intersection with all triangles
    Triangle min_tri;
    glm::vec3 barycenter, min_barycenter;
    float min_t = INFINITY;
    for (int i = mesh.startIdx; i < mesh.startIdx + mesh.triangleCount; i++)
    {
        float t = triangleIntersectionTest(tris[i], r, barycenter);
        if (t < min_t && t > 0.f)
        {
            min_t = t;
            min_barycenter = barycenter;
            min_tri = tris[i];
        }
    }

    // Find intersection point and normal
    float u = min_barycenter.x;
    float v = min_barycenter.y;
    float w = 1.f - u - v;
    intersectionPoint = u * min_tri.verts[0] + v * min_tri.verts[1] + w * min_tri.verts[2];
    normal = glm::cross(min_tri.verts[1] - min_tri.verts[0], min_tri.verts[2] - min_tri.verts[0]);

    return min_t;
}

__host__ __device__ bool isLeaf(const LBVHNode* nodes, int idx) {
    return nodes[idx].left == 0xFFFFFFFF && nodes[idx].right == 0xFFFFFFFF;
}

/**
 * Test intersection between a ray and an LBVH.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float lbvhIntersectionTest(const LBVHNode* nodes, const Triangle* tris, Ray r, int triangleCount,
     glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {

    float stack[64];
    int stackPtr = -1;

    float min_t = INFINITY;
    glm::vec3 barycenter;
    
    // Push root node
    stack[++stackPtr] = triangleCount;
    int currNodeIdx = stack[stackPtr];
    while (stackPtr >= 0)
    {
        // Check intersection with left and right children
        int leftChild = nodes[currNodeIdx].left;
        int rightChild = nodes[currNodeIdx].right;
        LBVHNode left = nodes[leftChild];
        LBVHNode right = nodes[rightChild];

        bool intersectLeft = aabbIntersectionTest(nodes[leftChild].aabb, r);
        bool intersectRight = aabbIntersectionTest(nodes[rightChild].aabb, r);

        // If intersection found, and they are leaf nodes, check for triangle intersections
        if (intersectLeft && isLeaf(nodes, leftChild)) {
            float t = triangleIntersectionTest(tris[left.objectId], r, barycenter);
            min_t = glm::min(min_t, t);
        }
        if (intersectRight && isLeaf(nodes, rightChild)) {
            float t = triangleIntersectionTest(tris[right.objectId], r, barycenter);
            min_t = glm::min(min_t, t);
        }

        // If internal nodes, keep traversing
        bool traverseLeftSubtree = (intersectLeft && !isLeaf(nodes, leftChild));
        bool traverseRightSubtree = (intersectRight && !isLeaf(nodes, rightChild));

        if (!traverseLeftSubtree && !traverseRightSubtree) {
            // Pop node from stack
            currNodeIdx = stack[stackPtr--];
        }
        else {
            currNodeIdx = (traverseLeftSubtree) ? leftChild : rightChild;
            if (traverseLeftSubtree && traverseRightSubtree) {
                // Push right child onto stack
                stack[++stackPtr] = rightChild;
            }
        }
    }

    return min_t;
}
