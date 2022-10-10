#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

#define BOUNDINGBOX 1

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

/*
 ******************************************************
 * OBJECT INTERSECTION TEST FOR OBJ
 ******************************************************
 */

__host__ __device__ float boundingBoxIntersectionTest(Geom box, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
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
            float t1 = (box.boundingBox.min[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (box.boundingBox.max[xyz] - q.origin[xyz]) / qdxyz;
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
    return -1.f;
}

__host__ __device__ float objIntersectionTest(Geom obj, Triangle *dev_tri, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    
    bool isectFound = false;

#if BOUNDINGBOX
    if (boundingBoxIntersectionTest(obj, r, intersectionPoint, normal, outside) == -1.f) {
        return -1;
    }
#endif

    int triCount = obj.triCount;
    glm::vec3 minVal = glm::vec3(INT_MAX, INT_MAX, INT_MAX);
    glm::vec3 barycentric;
    glm::vec3 n1, n2, n3;

    Triangle* dev_triItr = dev_tri;

    for (int i = 0; i < triCount; i++, dev_triItr++) {

        //printf("\n****GPU****\n");
        ////printf("\n %f, %f, %f", dev_triItr->pos[0].x, dev_triItr->pos[0].y, dev_triItr->pos[0].z);
        //printf("\n %f, %f, %f", dev_triItr->nor[0].x, dev_triItr->nor[0].y, dev_triItr->nor[0].z);

        glm::vec3 v1_pos = glm::vec3(obj.transform * glm::vec4(dev_triItr->pos[0].x, dev_triItr->pos[0].y, dev_triItr->pos[0].z, 1.0));
        glm::vec3 v2_pos = glm::vec3(obj.transform * glm::vec4(dev_triItr->pos[1].x, dev_triItr->pos[1].y, dev_triItr->pos[1].z, 1.0));
        glm::vec3 v3_pos = glm::vec3(obj.transform * glm::vec4(dev_triItr->pos[2].x, dev_triItr->pos[2].y, dev_triItr->pos[2].z, 1.0));

        isectFound = glm::intersectRayTriangle(r.origin, r.direction,
            v1_pos, v2_pos, v3_pos, barycentric);

        if (isectFound) {

            /*printf("\n****GPU****\n");
            printf("\n %f, %f, %f", dev_triItr->nor[0].x, dev_triItr->nor[0].y, dev_triItr->nor[0].z);*/

            if (barycentric[2] >= 0 && barycentric[2] < minVal[2])
            {
                minVal = barycentric;
            }

            glm::vec3 v1 = dev_triItr->pos[0];
            glm::vec3 v2 = dev_triItr->pos[1];
            glm::vec3 v3 = dev_triItr->pos[2];

            n1 = glm::normalize(glm::cross(v2 - v1, v3 - v1));
            n2 = glm::normalize(glm::cross(v1 - v3, v2 - v3));
            n3 = glm::normalize(glm::cross(v3 - v2, v1 - v2));

            //printf("NOR1: %f, %f, %f\n", n1.x, n1.y, n1.z);
            //printf("NOR2: %f, %f, %f\n", n2.x, n2.y, n2.z);
            //printf("NOR3: %f, %f, %f\n", n3.x, n3.y, n3.z);

            break;
        }
    }

    if (isectFound)
    {
        float u = minVal[0];
        float v = minVal[1];
        float t = minVal[2];
        intersectionPoint = getPointOnRay(r, t);
        normal = glm::normalize(glm::vec3(u * n1 + v * n2 + (1 - u - v) * n3));
        /*intersectionPoint = isectPoint;
        normal = isectNor;*/
        //printf("INOR: %f, %f, %f\n", normal.x, normal.y, normal.z);
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;

    /*printf("\n***********");
    for (int i = 0; i < obj.triCount; i++)
    {
        printf("\n%f, %f, %f", dev_tri->pos[0].x, dev_tri->pos[0].y, dev_tri->pos[0].z);
        dev_tri++;
    }
    printf("\n###########\n");*/
}

/*
 ******************************************************
 * PROCEDURAL SHAPES FUNCTIONS
 ******************************************************
 */

#define MAX_STEPS 100
#define MAX_DIST 100.f
#define SURF_DIST 0.01

 /*
  ******************************************************
  * Procedural Primitive SDFs
  ******************************************************
  */

__host__ __device__ float roundedCylinderSDF(glm::vec3 queryPos, float ra, float rb, float h)
{
    glm::vec2 d = glm::vec2(glm::length(glm::vec2(queryPos.x, queryPos.z)) - 2.0 * ra + rb, abs(queryPos.y) - h);
    return min(max(d.x, d.y), 0.f) + glm::length(max(d, 0.f)) - rb;
}

__host__ __device__ float sphereSDF(glm::vec3 p) {
    glm::vec3 sphereCenter = glm::vec3(0, 0, 0);    // center.xyz,radius
    float sphereRadius = 1.f;
    //normal = glm::normalize(p - sphereCenter);
    return glm::length(p - sphereCenter) - sphereRadius; // dist from sphere = dist from center - radius
}

__host__ __device__ float torusSDF(glm::vec3 p, glm::vec2 t)
{
    glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - t.x, p.y);
    return glm::length(q) - t.y;
}

__host__ __device__ float cappedCylinderSDF(glm::vec3 queryPos, float r, float h)
{
    glm::vec2 d = abs(glm::vec2(glm::length(glm::vec2(queryPos.x, queryPos.z)), queryPos.y)) - glm::vec2(r, h);
    return min(max(d.x, d.y), 0.f) + glm::length(max(d, 0.f));
}

__host__ __device__ float boxSDF(glm::vec3 p, glm::vec3 b)
{
    glm::vec3 q = abs(p) - b;
    return glm::length(max(q, 0.f)) + min(max(q.x, max(q.y, q.z)), 0.f);
}

__host__ __device__ float roundBoxSDF(glm::vec3 p, glm::vec3 b, float r)
{
    glm::vec3 q = abs(p) - b;
    return glm::length(max(q, 0.f)) + min(max(q.x, max(q.y, q.z)), 0.f) - r;
}

__host__ __device__ float cappedConeSDF(glm::vec3 p, glm::vec3 a, glm::vec3 b, float ra, float rb)
{
    float rba = rb - ra;
    float baba = dot(b - a, b - a);
    float papa = dot(p - a, p - a);
    float paba = dot(p - a, b - a) / baba;
    float x = sqrt(papa - paba * paba * baba);
    float cax = max(0.f, x - ((paba < 0.5f) ? ra : rb));
    float cay = abs(paba - 0.5) - 0.5;
    float k = rba * rba + baba;
    float f = glm::clamp((rba * (x - ra) + paba * baba) / k, 0.f, 1.f);
    float cbx = x - ra - f * rba;
    float cby = paba - f;
    float s = (cbx < 0.0 && cay < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(cax * cax + cay * cay * baba,
        cbx * cbx + cby * cby * baba));
}

/*
 ******************************************************
 * Procedural Operators
 ******************************************************
 */

__host__ __device__ glm::mat2 rot(float a) {
    float s = sin(a);
    float c = cos(a);
    return glm::mat2(c, -s, s, c);
}

/*
 ******************************************************
 * Scene SDFs
 ******************************************************
 */

__host__ __device__ float mugSDF(glm::vec3 p) {
    // coffee mug
    float dMugOuter = roundedCylinderSDF(p, 0.6f, 0.2f, 1.f);
    float dMugInner = roundedCylinderSDF(p - glm::vec3(0.f, 0.1f, 0.f), 0.5f, 0.2f, 1.f);

    //return max(dMugOuter, -dMugInner);
    // vec3 rotatedP = rotateZ(p, 60.f * 3.14f/ 180.f);
    glm::vec3 pRotMugHandle = p + glm::vec3(-1.5f, 0.0, 0.0);
    glm::mat2 rotMugHandleTransform = rot(90.f * 3.14f / 180.f);
    glm::vec2 pRotMugHandleTemp = glm::vec2(pRotMugHandle.y, pRotMugHandle.z) * rotMugHandleTransform;
    pRotMugHandle.y = pRotMugHandleTemp.x;
    pRotMugHandle.z = pRotMugHandleTemp.y;

    float dMugHandle = torusSDF(pRotMugHandle, glm::vec2(0.7f, 0.07));
    //return dMugHandle;
    float dMugTemp = min(dMugOuter, dMugHandle);//smoothUnion(dMugOuter, dMugHandle, 0.2f);
    float dMug = max(dMugTemp, -dMugInner);
    return dMug;
}

__host__ __device__ float coffeeSDF(glm::vec3 p) {
    float dCoffee = cappedCylinderSDF(p - glm::vec3(0.f, 0.7f, 0.f), 1.f, 0.01f);
    return dCoffee; // dist from sphere = dist from center - radius
}

__host__ __device__ float bookCoverSDF(glm::vec3 p) {
    glm::vec3 pCBook = p;// +glm::vec3(2.5f, 0.80, 0.0);

    float dCBookCover = roundBoxSDF(pCBook, glm::vec3(0.8f, 0.25f, 1.1f), 0.2);
    float dCBookPagesCut = boxSDF(pCBook + glm::vec3(-0.2f, 0.f, 0.f), glm::vec3(1.1f, 0.28f, 1.8f));
    float dCBook = max(dCBookCover, -dCBookPagesCut);
    return dCBook;
}

__host__ __device__ float bookPagesSDF(glm::vec3 p) {
    glm::vec3 pCBook = p;// +glm::vec3(2.5f, 0.80, 0.0);
    float dCBookPages = boxSDF(pCBook, glm::vec3(0.9f, 0.28f, 1.2f));
    return dCBookPages;
}

__host__ __device__ float lightSDF(glm::vec3 p) {

    p -= glm::vec3(0.f, 0.f, 0.f);

    glm::vec3 pRotOLightHead = p + glm::vec3(0.3f, 1.8f, 0.f);
    glm::mat2 rotOLightHeadTransform = rot(30.f * 3.14f / 180.f);
    glm::vec2 pRotOLightHeadTemp = glm::vec2(pRotOLightHead.x, pRotOLightHead.y) * rotOLightHeadTransform;
    pRotOLightHead.x = pRotOLightHeadTemp.x;
    pRotOLightHead.y = pRotOLightHeadTemp.y;
    float dOLightHead = cappedConeSDF(pRotOLightHead, glm::vec3(-1.2f, 5.f, 0.f), glm::vec3(0.f, 5.f, 0.f), 0.2f, 1.f);

    glm::vec3 pRotILightHead = p + glm::vec3(0.3f, 1.8f, 0.f);
    glm::mat2 rotILightHeadTransform = rot(30.f * 3.14f / 180.f);
    glm::vec2 pRotILightHeadTemp = glm::vec2(pRotILightHead.x, pRotILightHead.y) * rotILightHeadTransform;
    pRotILightHead.x = pRotILightHeadTemp.x;
    pRotILightHead.y = pRotILightHeadTemp.y;
    float dILightHead = cappedConeSDF(pRotILightHead, glm::vec3(-0.8f, 5.f, 0.f), glm::vec3(0.1f, 5.f, 0.f), 0.17f, 0.9f);

    float dLightHead = max(dOLightHead, -dILightHead);

    glm::vec3 pRotULightStand = p - glm::vec3(0.6f, 3.5f, 0.f);
    glm::mat2 rotULightStandTransform = rot(-60.f * 3.14f / 180.f);
    glm::vec2 pRotULightStandTemp = glm::vec2(pRotULightStand.x, pRotULightStand.y) * rotULightStandTransform;
    pRotULightStand.x = pRotULightStandTemp.x;
    pRotULightStand.y = pRotULightStandTemp.y;
    float dULightStand = cappedCylinderSDF(pRotULightStand, 0.1f, 0.7f);

    glm::vec3 pLLightStand = p - glm::vec3(0.f, 2.f, 0.f);
    float dLLightStand = cappedCylinderSDF(pLLightStand, 0.1f, 2.3f);

    glm::vec3 pLightBase = p - glm::vec3(0.f, 0.f, 0.f);
    float dLightBase = cappedCylinderSDF(pLightBase, 1.2f, 0.2f);

    return min(min(min(dLightHead, dULightStand), dLLightStand), dLightBase);
}

__host__ __device__ float sceneSDF(glm::vec3 p, Geom impGeom) {

     glm::vec4 transP4 = impGeom.inverseTransform* glm::vec4(p.x, p.y, p.z, 1.0);
     glm::vec3 transP3 = glm::vec3(transP4.x, transP4.y, transP4.z);
     float d = 0;
     switch (impGeom.implicitobj) {
        case IMP_SPHERE:        d = sphereSDF(transP3);
                                break;
        case IMP_MUG:           d = mugSDF(transP3);
                                break;
        case IMP_COFFEE:        d = coffeeSDF(transP3);
                                break;
        case IMP_BOOKCOVER:     d = bookCoverSDF(transP3);
                                break;
        case IMP_BOOKPAGES:     d = bookPagesSDF(transP3);
                                break;
        case IMP_LIGHT:         d = lightSDF(transP3);
                                break;
     }
     return d;
}

/*
 ******************************************************
 * Estimating Normals
 ******************************************************
 */

__host__ __device__ glm::vec3 estimateNormal(glm::vec3 p, Geom geom) {
    float x = sceneSDF(glm::vec3(p.x + EPSILON, p.y, p.z), geom) - sceneSDF(glm::vec3(p.x - EPSILON, p.y, p.z), geom);
    float y = sceneSDF(glm::vec3(p.x, p.y + EPSILON, p.z), geom) - sceneSDF(glm::vec3(p.x, p.y - EPSILON, p.z), geom);
    float z = sceneSDF(glm::vec3(p.x, p.y, p.z + EPSILON), geom) - sceneSDF(glm::vec3(p.x, p.y, p.z - EPSILON), geom);

    return glm::normalize(glm::vec3(x, y, z));
}

 /*
  ******************************************************
  * Ray Marching
  ******************************************************
  */

__host__ __device__ float implicitIntersectionTest(Geom impGeom, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {

    float t;
    //printf("## 1 ##");
    glm::vec3 queryPoint = r.origin;
    for (int i = 0; i < MAX_STEPS; ++i)
    {
        //printf("## 2 ##");
        float distanceToSurface = sceneSDF(queryPoint, impGeom);

        if (distanceToSurface < EPSILON)
        {
            //printf("## 3 ##");
            //intersection.position = queryPoint;
            normal = estimateNormal(queryPoint, impGeom);
            t = glm::length(queryPoint - r.origin);
            intersectionPoint = getPointOnRay(r, t);
            return t;
        }
        //printf("## 4 ##");
        queryPoint = queryPoint + r.direction * distanceToSurface;
    }
    //printf("## 5 ##");
    t = -1.0;
    intersectionPoint = getPointOnRay(r, t);
    //normal = intersectionPoint - ;
    //printf("## 6 ##");
    return t;
}