#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

using namespace scene_structs;

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

__device__ float triangleIntersectionTest(const Triangle& triangle, const glm::vec3 &ro, const glm::vec3 &rd,
  glm::vec3& out_intersectionPoint, glm::vec3& out_normal, glm::vec2& out_uv, bool& out_outside) {

  glm::vec3 v1 = triangle.verts[0].position;
  glm::vec3 v2 = triangle.verts[1].position;
  glm::vec3 v3 = triangle.verts[2].position;

  glm::vec3 edge1 = v2 - v1; //       v1
  glm::vec3 edge2 = v3 - v2; //     /   |
  glm::vec3 edge3 = v1 - v3; //    v2---v3

  // triangle is a plane of the form ax + bx + cx = d, where (a,b,c) is the normal
  glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2)); // vertices are counter-clockwise

  // plug in arbitrary point (v0) to solve for d
  float d = glm::dot(normal, v1);

  if (glm::abs(glm::dot(normal, rd)) < 0.0001f) { // TODO: tune
    // avoid divide by 0, ray could be parallel to plane in which case we don't see it
    return -1;
  }

  // ro + t * rd = some point p that intersects with the plane
  // because p intersects, we can plug in
  // normal . (ro + t * rd) = d
  float t = (d - glm::dot(normal, ro)) / glm::dot(normal, rd);

  if (t < 0) { // Check if triangle is behind the camera
    return -1;
  }

  // Check if ray hits triangle or not (barycentric coords)
  // all signed areas must be positive (relative to normal) for point to be inside the triangle
  glm::vec3 intersectionPoint = ro + t * rd;

  glm::vec3 c1 = intersectionPoint - v1;
  glm::vec3 c2 = intersectionPoint - v2;
  glm::vec3 c3 = intersectionPoint - v3;

  float v1Area = glm::dot(normal, glm::cross(edge2, c2));
  float v2Area = glm::dot(normal, glm::cross(edge3, c3));
  float v3Area = glm::dot(normal, glm::cross(edge1, c1));
  float triangleArea = v1Area + v2Area + v3Area;

  bool isInsideTriangle = v1Area > 0 && v2Area > 0 && v3Area > 0;

  if (!isInsideTriangle) {
    return -1;
  }

  // Finally, check if we are hitting front side or back side of face by checking
  // to hit outside, normal and ray direction should point the opposite way
  out_intersectionPoint = intersectionPoint;
  if (glm::dot(rd, normal) < 0) {
    out_outside = true;
  }
  else {
    out_outside = false;
  }

  // TODO: get normals and uv using barycentric interpolation
  out_normal = glm::normalize((triangle.verts[0].normal * v1Area + triangle.verts[1].normal * v2Area
    + triangle.verts[2].normal * v3Area));
  if (out_outside) {
    out_normal = -out_normal;
  }
  //out_normal = triangle.verts[0].normal;

  out_uv = (triangle.verts[0].uv * v1Area + triangle.verts[1].uv * v2Area
    + triangle.verts[2].uv * v3Area) / triangleArea;

  return t;
}

__device__ float triangleMeshIntersectionTest(const Geom &triangleMesh, Triangle *triangles, Ray r,
  glm::vec3& out_intersectionPoint, glm::vec3& out_normal, glm::vec2 & out_uv, bool& out_outside) {

  // first apply inverse transformation to ray
  glm::vec3 ro = multiplyMV(triangleMesh.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(triangleMesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

  float t;
  float t_min = FLT_MAX;
  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;
  glm::vec2 tmp_uv;
  bool tmp_outside;

  bool hitGeom = false;

  for (int i = triangleMesh.triangleOffset; i < triangleMesh.triangleOffset + triangleMesh.numTriangles; ++i) {
    Triangle& triangle = triangles[i];

    t = triangleIntersectionTest(triangle, ro, rd, tmp_intersect, tmp_normal, tmp_uv, tmp_outside);

    if (t > 0.0f && t_min > t)
    {
      hitGeom = true;
      t_min = t;
      out_intersectionPoint = tmp_intersect;
      out_normal = tmp_normal;
      out_uv = tmp_uv;
      out_outside = tmp_outside;
    }
  }

  // Don't forget to transform back
  out_normal = glm::normalize(multiplyMV(triangleMesh.inverseTransform, glm::vec4(out_normal, 0)));

  if (!hitGeom) {
    return -1;
  }

  return t_min;
}

// slab test - clip the ray by the parallel planes
 __device__ bool rayIntersectsBounds(const glm::vec3 &ro, const glm::vec3 &rd, const Bounds &b) {
  float tx1 = (b.min.x - ro.x) / rd.x, tx2 = (b.max.x - ro.x) / rd.x;
  float tmin = min(tx1, tx2), tmax = max(tx1, tx2);
  float ty1 = (b.min.y - ro.y) / rd.y, ty2 = (b.max.y - ro.y) / rd.y;
  tmin = max(tmin, min(ty1, ty2)), tmax = min(tmax, max(ty1, ty2));
  float tz1 = (b.min.z - ro.z) / rd.z, tz2 = (b.max.z - ro.z) / rd.z;
  tmin = max(tmin, min(tz1, tz2)), tmax = min(tmax, max(tz1, tz2));
  return tmax >= tmin && tmax > 0;
}

#define BVH_STACK_SIZE 128

__device__ float bvhTriangleMeshIntersectionTest(const Geom& triangleMesh, BvhNode* bvh_list, Triangle* triangle_list, Ray r,
  glm::vec3& out_intersectionPoint, glm::vec3& out_normal, glm::vec2& out_uv, bool& out_outside) {

  glm::vec3 ro = multiplyMV(triangleMesh.inverseTransform, glm::vec4(r.origin, 1.0f));
  glm::vec3 rd = glm::normalize(multiplyMV(triangleMesh.inverseTransform, glm::vec4(r.direction, 0.0f)));
  BvhNode* mesh_bvh = bvh_list + triangleMesh.bvhOffset;
  Triangle* mesh_triangles = triangle_list + triangleMesh.triangleOffset;

  bool hitGeom = false;
  float t_min = FLT_MAX;
  glm::vec3 tmp_intersect;
  glm::vec3 tmp_normal;
  glm::vec2 tmp_uv;
  bool tmp_outside;

  // Instead of recursion, which causes a stack overflow on the gpu
  // use an actual stack to traverse bvh
  // (thanks compute-sanitizer for enlightening me after debugging "memory access violation" for 2 days fml...)
  // can't allocate dynamic memory on gpu so hardcoding length
  int bvhIndicesStack[BVH_STACK_SIZE];
  bvhIndicesStack[0] = 0; // start traversing at root node (0)
  int stackEndIndex = 0;
  
  while (stackEndIndex >= 0) {
    int currNodeIdx = bvhIndicesStack[stackEndIndex--]; // pop
    const BvhNode& curr_node = mesh_bvh[currNodeIdx];

    if (!rayIntersectsBounds(ro, rd, curr_node.bounds)) {
      continue; // no intersection with this bounding box -> move onto next
    }
    
    // Case: leaf node. Do triangle intersection tests and don't add any child nodes to stack
    if (curr_node.numTriangles > 0) {
      for (int i = curr_node.firstTriangleOffset; i < curr_node.firstTriangleOffset + curr_node.numTriangles; ++i) {
        const Triangle& triangle = mesh_triangles[i];

        float t = triangleIntersectionTest(triangle, ro, rd, tmp_intersect, tmp_normal, tmp_uv, tmp_outside);

        if (t > 0.0f && t_min > t)
        {
          t_min = t;
          out_intersectionPoint = tmp_intersect;
          out_normal = tmp_normal;
          out_uv = tmp_uv;
          out_outside = tmp_outside;
          hitGeom = true;
        }
      }
      continue;
    }

    // Case: parent node. Add left and/or right children to stack
    if (curr_node.leftChildIndex != -1) {
      bvhIndicesStack[++stackEndIndex] = curr_node.leftChildIndex;
    }
    if (curr_node.rightChildIndex != -1) {
      bvhIndicesStack[++stackEndIndex] = curr_node.rightChildIndex;
    }

    assert(stackEndIndex < BVH_STACK_SIZE);
  }

  // Don't forget to transform back the normal
  // Intersection point doesn't need to be transformed
  out_normal = glm::normalize(multiplyMV(triangleMesh.inverseTransform, glm::vec4(out_normal, 0)));
  
  return hitGeom ? t_min : -1;
}
