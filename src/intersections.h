#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define BOUNDINGVOLUME 1
#define BVH 1
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

#if BVH
__host__ __device__ bool BoundingBoxIntersectionTest(const Geom& geom, const Ray& r, const BBox& box) {
#else
__host__ __device__ bool meshBoundingBoxIntersectionTest(Geom geom, Ray r) {
#endif
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        if (glm::abs(qdxyz) > 0.00001f) {
#if BVH
            float t1 = (box.minCorner[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (box.maxCorner[xyz] - q.origin[xyz]) / qdxyz;
#else
            float t1 = (geom.mesh.min[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (geom.mesh.max[xyz] - q.origin[xyz]) / qdxyz;
#endif
            
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
        return true;
    }
    return false;
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

//helper functions
__host__ __device__ float TriArea(const glm::vec3& p1, const glm::vec3& p2, const glm::vec3& p3) {
    return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
}

__host__ __device__ glm::vec3 getNormal(const Triangle& t, const glm::vec3& P) {
    float A = TriArea(t.v1, t.v2, t.v3);
    float A0 = TriArea(t.v2, t.v3, P);
    float A1 = TriArea(t.v1, t.v3, P);
    float A2 = TriArea(t.v1, t.v2, P);
    return glm::normalize(t.n1 * A0 / A + t.n2 * A1 / A + t.n3 * A2 / A);
}


//Input ray is in Local Space
//__host__ __device__ float triangleIntersectionTest(Geom geom, Triangle face, Ray r,
//    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
//
//    glm::vec3 baryPosition;
//    bool isHit = glm::intersectRayTriangle(r.origin, r.direction, face.v1, face.v2, face.v3, baryPosition);
//    if (isHit) {
//        glm::vec3 localIntersectionPoint = (1.f - baryPosition.x - baryPosition.y) * face.v1 + baryPosition.x * face.v2 + baryPosition.y * face.v3;
//        intersectionPoint = multiplyMV(geom.transform, glm::vec4(bary_position, 1.f));
//        //1. Ray-plane intersection
//        float t = glm::dot(planeNormal, (points[0] - r.origin)) / glm::dot(planeNormal, r.direction);
//        if (t < 0) return false;
//
//        glm::vec3 P = r.origin + t * r.direction;
//    }
//    
//    return -1.f;
//
//}

#if !BVH
//world space
__host__ __device__ float meshIntersectionTest(Geom geom, Triangle* tri, Ray r, 
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
#if BOUNDINGVOLUME
    if (!meshBoundingBoxIntersectionTest(geom, r)) return -1.f;
#endif 
    
    outside = true;

    
    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    
    
    float closest_t = -1;
    Triangle closestTri;
    float t = 0;
    glm::vec3 baryPosition(0.f, 0.f, 0.f);
    glm::vec3 localIntersectionPoint;
    glm::vec3 bary;
    for (int i = geom.mesh.facesIdOffset; i < geom.mesh.facesIdOffset + geom.mesh.faceCount; i++) {
        Triangle curTri = tri[i];    
        if (glm::intersectRayTriangle(rt.origin, rt.direction, curTri.v1, curTri.v2, curTri.v3, baryPosition)) {
            localIntersectionPoint = (1.f - baryPosition.x - baryPosition.y) * curTri.v1 + baryPosition.x * curTri.v2 + baryPosition.y * curTri.v3;
            t = glm::distance(localIntersectionPoint, rt.origin);
            if (t > 0 && (t < closest_t || closest_t < 0)) {
                closest_t = t;
                closestTri = curTri;
                bary = localIntersectionPoint;              
            }             
        }
    }
    

    if (closest_t > 0) {
        glm::vec3 objspaceIntersection = getPointOnRay(rt, closest_t);
        intersectionPoint = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.f));
        normal = getNormal(closestTri, bary);
        normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.f)));
    
        if (glm::dot(normal, r.direction) > 0) {
            normal = -normal;
            outside = false;
        }
        return closest_t = glm::distance(intersectionPoint, r.origin);
    }

   
    
    return -1.f;
}
#endif
#if BVH
__host__ __device__ float bvhIntersectionTest(const Geom& geom, Triangle* tri, bvhNode* bvh, const Ray& r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, int bvhSize) {
    //transformation
    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;
    float closest_t = -1;
    float t = 0;
    glm::vec3 bary;
    glm::vec3 baryPosition(0.f, 0.f, 0.f);
    Triangle closestTri;

    int stack[32];
    int top = 0, cur = 0;
    
    //push root onto the stack
    stack[top++] = 0;
    while (top) {
        cur = stack[--top];
        //pop top
        auto curNode = bvh[cur];
        
        if (BoundingBoxIntersectionTest(geom, r, curNode.box)) {
            //if hit the leaf, try intersecting with the triangle
            if (curNode.triID >= 0) {
                Triangle curTri = tri[curNode.triID];
                
                if (glm::intersectRayTriangle(rt.origin, rt.direction, curTri.v1, curTri.v2, curTri.v3, baryPosition)) {
                    glm::vec3 localIntersectionPoint = (1.f - baryPosition.x - baryPosition.y) * curTri.v1 + baryPosition.x * curTri.v2 + baryPosition.y * curTri.v3;
                    //local t
                    t = glm::distance(localIntersectionPoint, rt.origin);
                    if (t > 0 && (t < closest_t || closest_t < 0)) {
                        closest_t = t;
                        closestTri = curTri;
                        bary = localIntersectionPoint;
                    }
                }
            }
            else {
                int left = 2 * cur + 1;
                int right = 2 * cur + 2;
                
                if (right > 0 && right < bvhSize) {
                    stack[top++] = right;
                }
                if (left > 0 && left < bvhSize) {
                    stack[top++] = left;
                }
                
            }
        }
    }

     if (closest_t > 0) {
        glm::vec3 objspaceIntersection = getPointOnRay(rt, closest_t);
        intersectionPoint = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.f));
        normal = getNormal(closestTri, bary);
        normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(normal, 0.f)));
    
        if (glm::dot(normal, r.direction) > 0) {
            normal = -normal;
            outside = false;
        }
        return closest_t = glm::distance(intersectionPoint, r.origin);
    }
    return -1.f;
}
#endif
///// Float approximate-equality comparison
//template<typename T>
//inline bool fequal(T a, T b, T epsilon = 0.0001) {
//    if (a == b) {
//        // Shortcut
//        return true;
//    }
//
//    const T diff = std::abs(a - b);
//    if (a * b == 0) {
//        // a or b or both are zero; relative error is not meaningful here
//        return diff < (epsilon* epsilon);
//    }
//
//    return diff / (std::abs(a) + std::abs(b)) < epsilon;
//}
////cis561
//__host__ __device__ float triangleIntersectionTest(Triangle triangle, Ray r,
//    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
//    //1. Ray-plane intersection
//    float t = glm::dot(triangle.n1, (triangle.v1 - r.origin)) / glm::dot(triangle.n1, r.direction);
//    if (t < 0) return -1.f;
//
//    glm::vec3 P = r.origin + t * r.direction;
//    //2. Barycentric test
//    float S = 0.5f * glm::length(glm::cross(triangle.v1 - triangle.v2, triangle.v1 - triangle.v3));
//    float s1 = 0.5f * glm::length(glm::cross(P - triangle.v2, P - triangle.v3)) / S;
//    float s2 = 0.5f * glm::length(glm::cross(P - triangle.v3, P - triangle.v1)) / S;
//    float s3 = 0.5f * glm::length(glm::cross(P - triangle.v1, P - triangle.v2)) / S;
//    float sum = s1 + s2 + s3;
//
//    if (s1 >= 0 && s1 <= 1 && s2 >= 0 && s2 <= 1 && s3 >= 0 && s3 <= 1 && fequal(sum, 1.0f)) {
//        //isect->t = t;
//        return t;
//    }
//    return -1.f;
//}