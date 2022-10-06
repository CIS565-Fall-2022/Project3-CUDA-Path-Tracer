#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include <array>

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
    TRIANGLE
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct AABB {
    glm::vec3 pMin, pMax;

    AABB()
    {
        float minNum = FLT_MIN;
        float maxNum = FLT_MAX;
        pMin = glm::vec3(minNum);
        pMax = glm::vec3(maxNum);
    }

    AABB(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
        pMin = glm::vec3(fmin(p1.x, p2.x), fmin(p1.y, p2.y), fmin(p1.z, p2.z));
        pMax = glm::vec3(fmax(p1.x, p2.x), fmax(p1.y, p2.y), fmax(p1.z, p2.z));

        pMin = glm::vec3(fmin(pMin.x, p3.x), fmin(pMin.y, p3.y), fmin(pMin.z, p3.z));
        pMax = glm::vec3(fmax(pMax.x, p3.x), fmax(pMax.y, p3.y), fmax(pMax.z, p3.z));

    }

    bool IntersectP(const Ray& ray) const
    {

        //auto temp1 = glm::vec3(pMax.x - ray.origin.x, pMax.y - ray.origin.y, pMax.z - ray.origin.z);
        //auto temp2 = glm::vec3(pMin.x - ray.origin.x, pMin.y - ray.origin.y, pMin.z - ray.origin.z);
        /*glm::vec3 ttop = glm::vec3((float)temp1.x * (float)invDir.x, (float)temp1.y * (float)invDir.y, (float)temp1.z * (float)invDir.z);
        glm::vec3 tbot = glm::vec3((float)temp2.x * (float)invDir.x, (float)temp2.y * (float)invDir.y, (float)temp2.z * (float)invDir.z);*/

        glm::vec3 invDir = glm::vec3(1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z);

        auto temp1 = pMax - ray.origin;
        auto temp2 = pMin - ray.origin;
        glm::vec3 ttop = temp1 * invDir;
        glm::vec3 tbot = temp2 * invDir;

        auto tmin = glm::vec3(std::min(ttop.x, tbot.x), std::min(ttop.y, tbot.y), std::min(ttop.z, tbot.z));
        auto tmax = glm::vec3(std::max(ttop.x, tbot.x), std::max(ttop.y, tbot.y), std::max(ttop.z, tbot.z));

        float t0 = std::max(tmin.x, (std::max)(tmin.y, tmin.z));
        float t1 = std::min(tmax.x, (std::min)(tmax.y, tmax.z));
        return t0 <= t1 && t1 >= 0;
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

    glm::vec3 pos[3];
    glm::vec3 normal[3];
    glm::vec2 uv[3];
    bool isObj{ false };

    const char* textureName;
    unsigned char* img;
    int texture_width;
    int texture_height;
    int channels;

    AABB bbox;
    int obj_start_offset;
    int obj_end;

    glm::vec3 endPos;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective{0};
    float hasRefractive{0};
    float indexOfRefraction{0};
    float emittance{0};
    float microfacet{0};
    float roughness{0};
    float metalness{ 0 };

    const char* textureName;
    unsigned char* img;
    int texture_width;
    int texture_height;
    int channels;
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
    float lensRadius;
    float focalDistance;
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
};

struct Triangle {
    glm::vec3 pos[3];
    glm::vec3 normal[3];
    glm::vec2 uv[3];
    int materialId;
};


static AABB Union(const AABB& b1, const AABB& b2) {
    AABB ret;
    ret.pMin = glm::vec3((std::min)(b1.pMin.x, b2.pMin.x),
        (std::min)(b1.pMin.y, b2.pMin.y),
        (std::min)(b1.pMin.z, b2.pMin.z));
    ret.pMax = glm::vec3((std::max)(b1.pMax.x, b2.pMax.x),
        (std::max)(b1.pMax.y, b2.pMax.y),
        (std::max)(b1.pMax.z, b2.pMax.z));
    return ret;
}

struct Obj {
    AABB box;
    Geom* data;
};