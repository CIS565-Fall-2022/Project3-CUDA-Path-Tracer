#pragma once

#include <string>
#include <vector>
#include "Texture.h"
#include "utilities.h"
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

struct Bound3 {

    glm::vec3 pMin = glm::vec3(1.f);
    glm::vec3 pMax = glm::vec3(-1.f);

    Bound3 Union(Bound3 b) const
    {
        // If bound is not initialized, then return the other bound
        if (this->pMax.x < this->pMin.x)
        {
            return Bound3{b.pMin, b.pMax};
        }
        else if (b.pMax.x < b.pMin.x)
        {
            return Bound3{this->pMin, this->pMax};
        }

        glm::vec3 minPoint = glm::vec3(std::min(b.pMin.x, pMin.x),
            std::min(b.pMin.y, pMin.y),
            std::min(b.pMin.z, pMin.z));

        glm::vec3 maxPoint = glm::vec3(std::max(b.pMax.x, pMax.x),
            std::max(b.pMax.y, pMax.y),
            std::max(b.pMax.z, pMax.z));

        return Bound3{ minPoint, maxPoint };
    }

    Bound3 Union(glm::vec3 p) const
    {
        // If bound is not initialized
        if (this->pMax.x < this->pMin.x)
        {
            return Bound3{ p, p };
        }

        glm::vec3 minPoint = glm::vec3(std::min(p.x, pMin.x),
            std::min(p.y, pMin.y), 
            std::min(p.z, pMin.z));

        glm::vec3 maxPoint = glm::vec3(std::max(p.x, pMax.x),
            std::max(p.y, pMax.y),
            std::max(p.z, pMax.z));

        return Bound3{ minPoint, maxPoint };
    }

    glm::vec3 Offset(glm::vec3 p) const
    {
        glm::vec3 o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }

    float SurfaceArea() const
    {
        glm::vec3 p = pMax - pMin;

        return 2 * (p.x * p.y + p.y * p.z + p.z * p.x);
    }

    int MaxExtent() const
    {
        glm::vec3 p = pMax - pMin;
        if (p.x > p.y && p.x > p.z)
            return 0;
        else if (p.y > p.x && p.y > p.z)
            return 1;
        else
            return 2;
    }
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

    // For BVH
    int bvhNodeStartIndex = -1;
    int bvhNodeEndIndex = -1;
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

    float hasMetallic = 0;

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
    bool isRefrectiveRay;
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
  int triangleId;
};


// For obj loading
struct Triangle {
    glm::vec3 v0, v1, v2;       // Vertex
    glm::vec3 n0, n1, n2;       // Normal
    glm::vec2 tex0, tex1, tex2; // UV coordinate

    glm::mat3 TBN;

    void CacheTBN()
    {
        glm::vec3 deltaPos1 = v1 - v0;  // In model space
        glm::vec3 deltaPos2 = v2 - v0;
        glm::vec2 deltaUV1 = tex1 - tex0;
        glm::vec2 deltaUV2 = tex2 - tex0;

        float r = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV1.y * deltaUV2.x);
        glm::vec3 tagent = (deltaPos1 * deltaUV2.y - deltaPos2 * deltaUV1.y) * r;
        glm::vec3 bitangent = (deltaPos2 * deltaUV1.x - deltaPos1 * deltaUV2.x) * r;
        glm::vec3 nor = glm::normalize(glm::cross(tagent, bitangent));
        //glm::vec3 nor = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(normal, 0.0f)));
        
        TBN = glm::transpose(glm::mat3(tagent, bitangent, nor));
    }

    Bound3 getBound()
    {
        glm::vec3 pMin = glm::vec3(std::min(std::min(v0.x, v1.x), v2.x),
            std::min(std::min(v0.y, v1.y), v2.y),
            std::min(std::min(v0.z, v1.z), v2.z));

        glm::vec3 pMax = glm::vec3(std::max(std::max(v0.x, v1.x), v2.x),
            std::max(std::max(v0.y, v1.y), v2.y),
            std::max(std::max(v0.z, v1.z), v2.z));

        return Bound3{pMin, pMax};
    }

    glm::vec3 getCentriod()
    {
        return (v0 + v1 + v2) * 0.333333f;
    }
};

// For texture mapping
struct TextureInfo{
    const char* id = "";

    int width;
    int height;
    int channels;

    int startIndex;
};

// For BVH
struct BVHPrimitiveInfo {
    int triangleId;
    Bound3 bound;
    glm::vec3 centroid;
};

struct BVHNode {
    int splitAxis = -1;
    int firstPrimOffset = -1;
    int nPrimitives = -1;
    Bound3 bound;
    BVHNode* leftChild = nullptr;
    BVHNode* rightChild = nullptr;
};

struct LinearBVHNode{
    Bound3 bound;       // 24 bytes
    union {
        int firstPrimOffset = -1;   // For leaf node
        int rightChildOffset;       // For interior node
    };
    uint16_t nPrimitives;
    uint8_t axis = 3;
    uint8_t pad[1]; // Ensure 32 byte total size
};
