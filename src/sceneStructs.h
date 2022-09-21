#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

//enum GeomType {
//    Sphere,
//    Cube,
//    Mesh
//};

enum class GeomType {
    Sphere = 0,
    Cube = 1,
    Mesh = 2
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB {
    glm::vec3 pMin;
    glm::vec3 pMax;
};

struct BVHNode {
    AABB box;
    int geomIdx;
    int size;
};

struct BVHTableElement {
};

struct Geom {
    GeomType type;
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

//struct Material {
//    enum Type {
//        Lambertian = 0, MetallicWorkflow = 1, Dielectric = 2
//    };
//
//    glm::vec3 baseColor;
//    float metallic;
//    float roughness;
//    float ior;
//    float emittance;
//    int type;
//};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
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
  glm::vec2 surfaceUV;
  int materialId;
};
