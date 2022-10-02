#pragma once

#include <string>
#include <vector>
#include "Texture.h"
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    OBJECT
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle{
    glm::vec3 v0, v1, v2;       // Vertex
    glm::vec3 n0, n1, n2;       // Normal
    glm::vec2 tex0, tex1, tex2; // UV coordinate
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

    // For object geometry
    int triangleStartIndex = -1;
    int triangleEndIndex = -1;
    bool hasNormal = false;
    bool hasUV = false;
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

    int textureIndex = -1;
    int normalMapIndex = -1;
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
    Ray ray;
    glm::vec3 color;
    int pixelIndex;
    int remainingBounces;
    bool hitLightSource;
    bool insightMat;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  int materialId;
  int geomId;
};


// Mark texture information

struct TextureInfo{
    const char* id = "";

    int width;
    int height;
    int channels;

    int startIndex;
};
