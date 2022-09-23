#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

typedef glm::vec3 color_t;
struct Material {
    color_t color;
    struct {
        float exponent;
        color_t color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;


    // BSDF
    __host__ __device__ color_t f(glm::vec3 const& wo, glm::vec3 const& wi) const {
        return color * INV_PI;
    }
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
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
};


struct PathSegment {
    struct Pred {
        __host__ __device__
        bool operator()(PathSegment const& seg) const {
            return seg.remainingBounces > 0;
        }
    };

    Ray ray;
    color_t color;
    int pixelIndex;
    int remainingBounces;

    __host__ __device__ bool operator!() const {
        return !remainingBounces;
    }
    __host__ __device__ void init(int max_bounce, int pix_idx, Ray const& ray) {        
        pixelIndex = pix_idx;
        remainingBounces = max_bounce;
        this->ray = ray;
        color = glm::vec3(1,1,1);
    }
    __host__ __device__ void terminate() {
        remainingBounces = 0;
    }
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
