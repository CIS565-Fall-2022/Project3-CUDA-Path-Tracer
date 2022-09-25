#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum MaterialType {
    LAMBERT,
    METAL,
    DIELECTIC,
};

enum GeomType {
    SPHERE,
    CUBE,
    MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    int meshid;
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
    struct PartitionRule {
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
    glm::vec3 hitPoint;
    bool rayFromOutside;
    int materialId;


    __host__ __device__ friend bool operator<(ShadeableIntersection const& a, ShadeableIntersection const& b) {
        return a.materialId < b.materialId;
    }
};

// Stored in the scene structure
// automatically generated from objects with emittance > 0 
// to provide info about what lights are in the scene
struct Light {
    color_t color;
    float intensity;
    glm::vec3 position;
};

// Stored in the scene structure
struct Triangle {
    glm::ivec3 verts;
    glm::ivec3 norms;
    Triangle(int(*arr)[6]) {
        for (int i = 0; i < 3; ++i) {
            verts[i] = (*arr)[i];
        }
        for (int i = 0; i < 3; ++i) {
            norms[i] = (*arr)[i + 3];
        }
    }
};
typedef glm::vec3 Vertex;
typedef glm::vec3 Normal;

struct Mesh {
    Mesh(int tri_start, int tri_end) : tri_start(tri_start), tri_end(tri_end) { }
    int tri_start;
    int tri_end;
};

struct MeshInfo {
    Span<Vertex> vertices;
    Span<Normal> normals;
    Span<Triangle> tris;
    Span<Mesh> meshes;
};