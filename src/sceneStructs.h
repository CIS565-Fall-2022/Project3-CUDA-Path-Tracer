#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Texture {
  int width;
  int height;
  int components;

  int offsetNormal = 0;
  int offsetColor = 0;
  int offsetEmissive = 0;
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct SceneMeshesData {
  unsigned short* indices;
  glm::vec3* positions;
  glm::vec3* normals;
  glm::vec2* uvs;
  glm::vec4* tangents;
};

struct Geom {
    enum GeomType type;
    int materialid;
    int useTex = 0;
    int normalMapID = -1;
    int hasNormalMap = 0;
    int hasTangent = 0;

    int meshid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 meshTransform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int startIndex;
    int count;

    glm::vec3 boundingMin;
    glm::vec3 boundingMax;
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

    int texID = -1;
    int emissiveTexID = -1;
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

    int isDifuse = 0;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  int materialId;
  int useTex;
};
