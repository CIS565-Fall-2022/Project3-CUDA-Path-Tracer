#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 invDirection;
    float intersectionCount;
};

struct AABB {
    glm::vec3 min;
    glm::vec3 max;

    float surfaceArea() {
        glm::vec3 e = min - max;
        return 2.f * (e.x * e.y + e.x * e.z + e.y * e.z);
    };
};

struct MortonCode {
    int objectId;
    unsigned int code;
};

struct Triangle {
    AABB aabb;
    glm::vec3 centroid;
    glm::vec3 verts[3];
    glm::vec3 norms[3];
    unsigned int mcode; // for testing, remove later
    int objectId;

    void computeAABB() {
        aabb.min = glm::min(verts[0], glm::min(verts[1], verts[2]));
        aabb.max = glm::max(verts[0], glm::max(verts[1], verts[2]));
    }

    void computeCentroid() {
        centroid = (verts[0] + verts[1] + verts[2]) / glm::vec3(3.f, 3.f, 3.f);
    }
};

struct NodeRange {
    int i;
    int j;
    int l;
    int d;
};

struct BVHNode {
    AABB aabb;
    unsigned int left, right;
    int firstTri, numTris;
};

struct LBVHNode {
    AABB aabb;
    int objectId;
    unsigned int left;
    unsigned int right;
};

struct Geom {
    enum GeomType type;
    AABB aabb;
    int startIdx; 
    int triangleCount;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    float lens_radius;
    float focal_dist;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color; // accumulated light
    glm::vec3 throughput;
    int pixelIndex;
    int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
