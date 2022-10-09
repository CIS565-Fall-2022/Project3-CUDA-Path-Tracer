#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MODEL
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};


//bounding box
struct BBox {
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
    BBox(): minCorner(glm::vec3(0)), maxCorner(glm::vec3(0)){}
    BBox(const glm::vec3& p1, const glm::vec3& p2)
        : minCorner(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
            std::min(p1.z, p2.z)),
        maxCorner(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
            std::max(p1.z, p2.z)) {
    }

    glm::vec3 Diagonal() const { return maxCorner - minCorner; }

    int MaximumExtent() const {
        glm::vec3 d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
    static BBox Union(const BBox& b, const glm::vec3& p) {
        return BBox(glm::vec3(std::min(b.minCorner.x, p.x),
                    std::min(b.minCorner.y, p.y),
                    std::min(b.minCorner.z, p.z)),
          glm::vec3(std::max(b.maxCorner.x, p.x),
                    std::max(b.maxCorner.y, p.y),
                    std::max(b.maxCorner.z, p.z)));
    }
    
    
};
struct Mesh {
    glm::vec3 max{ FLT_MIN };
    glm::vec3 min{ FLT_MAX };
    int faceCount;
    int facesIdOffset;
};

struct Geom {
    enum GeomType type;
    int materialid;
    Mesh mesh;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Triangle {
    BBox bbox;
    int geomID;
    glm::vec3 v1;
    glm::vec3 v2;
    glm::vec3 v3;
    glm::vec3 n1;
    glm::vec3 n2;
    glm::vec3 n3;

    void setGlobalBBox(const Geom& geom) {
        glm::vec3 globalV1 = glm::vec3(geom.transform * glm::vec4(v1, 1.f));
        glm::vec3 globalV2 = glm::vec3(geom.transform * glm::vec4(v2, 1.f));
        glm::vec3 globalV3 = glm::vec3(geom.transform * glm::vec4(v3, 1.f));
        bbox.minCorner = glm::min(globalV1, glm::min(globalV2, globalV3));
        bbox.maxCorner = glm::max(globalV1, glm::max(globalV2, globalV3));
    }
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
    float lensRadius;
    float focalDistance;

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
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
};
