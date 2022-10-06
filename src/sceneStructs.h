#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define OCTREE_DEPTH 4
#define OCTREE

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

struct OctreeNode {
  glm::vec3 boundingMax;
  glm::vec3 boundingMin;

  int maxDepth = 4;

  bool isLeaf = 0;
  int startIndex;
  int count;

  OctreeNode(glm::vec3 maxBB, glm::vec3 minBB) {
    this->boundingMax = maxBB;
    this->boundingMin = minBB;
  }
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Triangle {
  glm::vec3 pos[3];
  glm::vec3 normal[3];
  glm::vec2 uv[3];
  glm::vec4 tangent[3];
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
    int startTriIndex;
    int count;
    int octreeStartIndex;

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

    float focal = 0;
    float aperture = 0;
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
