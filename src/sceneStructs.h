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
    MESH
};

typedef glm::vec3 color_t;


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

struct Texture {
    __host__ __device__ 
    Texture() : pixel_width(0), pixel_height(0), pixels(nullptr) {}
    __host__ __device__
    Texture(Texture const& o) = default;
    __host__ __device__
    Texture(Texture&& o) = default;
    __host__
    Texture(int pixel_width, int pixel_height, int channel, unsigned char* raw_pixel) : 
        pixel_width(pixel_width), pixel_height(pixel_height) {

        assert(channel > 0);

        size_t tot = (size_t) pixel_width * pixel_height;
        pixels = new color_t[tot];

        for (size_t i = 0; i < tot; ++i) {
            pixels[i] = { 
                raw_pixel[i * channel],
                channel > 1 ? raw_pixel[i * channel + 1] : 0,
                channel > 2 ? raw_pixel[i * channel + 2] : 0,
            };
        }
    }
    __host__
    ~Texture() { delete[] pixels; }
    
    Texture to_device() const {
        Texture ret;
        ret.pixel_width = pixel_width;
        ret.pixel_height = pixel_height;

        size_t tot = (size_t)pixel_width * pixel_height;
        ALLOC(ret.pixels, tot);
        H2D(ret.pixels, pixels, tot);
    }

    int pixel_width;
    int pixel_height;
    color_t* pixels;
};

struct Material {
    color_t diffuse;
    struct {
        float exponent;
        color_t color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float ior;
    float emittance;

    // below fields are used by PBRT
    enum Type {
        DIFFUSE,
        GLOSSY,
        REFL,
        TRANSPARENT,
        REFR,
        SUBSURFACE,
        INVALID
    };
    
    struct {
        int tex_idx;
    } textures;      // optional: invalid = -1

    Type type;       // optional: default = DIFFUSE
    float roughness; // optional: default = 1
    static Type str_to_mat_type(std::string str) {
#define CHK(x) if(str == #x) return x
        CHK(DIFFUSE);
        CHK(GLOSSY);
        CHK(REFL);
        CHK(TRANSPARENT);
        CHK(REFR);
        CHK(SUBSURFACE);

        assert(!"invalid material");
        return INVALID;
#undef CHK
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
    int materialId;

    // only used by mesh
    glm::vec2 uv;

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
    glm::ivec3 uvs;
    int mat_id;

    Triangle(glm::ivec3 verts, glm::ivec3 norms, glm::ivec3 uvs, int mat_id) 
        : verts(verts), norms(norms), uvs(uvs), mat_id(mat_id) {}
};
typedef glm::vec3 Vertex;
typedef glm::vec3 Normal;
typedef glm::vec2 TexCoord;

struct Mesh {
    Mesh(int tri_start, int tri_end) : tri_start(tri_start), tri_end(tri_end) { }
    int tri_start;
    int tri_end;
};


/// <summary>
/// this is basically GPU counterpart of the mesh vectors
/// </summary>
struct MeshInfo {
    Span<Vertex> vertices;
    Span<Normal> normals;
    Span<TexCoord> uvs;
    Span<Texture> texs;
    Span<Triangle> tris;
    Span<Mesh> meshes;
};