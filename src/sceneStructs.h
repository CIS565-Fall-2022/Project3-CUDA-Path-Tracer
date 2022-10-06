#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "tiny_gltf.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Primitive {
    glm::vec3 pos[3];
    glm::vec3 normal[3];
    glm::vec2 uv[3];
    glm::vec4 tangent[3];

    int mat_id;
    bool hasNormal = false;
    bool hasUV = false;
    bool hasTangent = false;
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

    //for primitives only
    int primBegin;
    int primEnd;
    //for culling
    glm::vec3 aabb_min;
    glm::vec3 aabb_max;
    //for extra storing
    int matId;
};

struct TextureInfo {
    int index = -1;  // required.
    int texCoord;    // The set index of texture's TEXCOORD attribute used for
                     // texture coordinate mapping.

    TextureInfo& operator=(const tinygltf::TextureInfo t) {
        index = t.index;
        texCoord = t.texCoord;
        return *this;
    }
};

// pbrMetallicRoughness class defined in glTF 2.0 spec.
struct PbrMetallicRoughness {
    glm::vec3 baseColorFactor = glm::vec3(1.0f);  // len = 4. default [1,1,1,1]
    TextureInfo baseColorTexture;
    double metallicFactor;   // default 1
    double roughnessFactor;  // default 1
    TextureInfo metallicRoughnessTexture;

    __host__ __device__ PbrMetallicRoughness()
        : baseColorFactor(glm::vec3(1.0f)),
        metallicFactor(1.0f),
        roughnessFactor(1.0f) {}
};

struct NormalTextureInfo {
    int index = -1;  // required
    int texCoord;    // The set index of texture's TEXCOORD attribute used for
                     // texture coordinate mapping.
    float scale = 1.0f;    // scaledNormal = normalize((<sampled normal texture value>
                     // * 2.0 - 1.0) * vec3(<normal scale>, <normal scale>, 1.0))
    NormalTextureInfo& operator=(const tinygltf::NormalTextureInfo t) {
        index = t.index;
        texCoord = t.texCoord;
        scale = t.scale;
        return *this;
    }
};

struct Texture {
    int TexIndex = -1;
    int width;
    int height;
    int components;
    unsigned char* image;
    int size;
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

    Texture tex;
    Texture bump;

    //for gltf extra factor
    bool gltf = false;
    int texIndex = 0;
    PbrMetallicRoughness pbrMetallicRoughness;

    NormalTextureInfo normalTexture;
    //OcclusionTextureInfo occlusionTexture;
    TextureInfo emissiveTexture;
    //cuda cannot use std vector or double so need to switch TODO
    glm::vec3 emissiveFactor;  // length 3. default [0, 0, 0]
    //std::string alphaMode;               // default "OPAQUE"
    //double alphaCutoff;                  // default 0.5
    bool doubleSided;                    // default false;
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
    float focal_length;
    float aperture_radius;
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
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 uv;
  glm::vec4 tangent;
};



