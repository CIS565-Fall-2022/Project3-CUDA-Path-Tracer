#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"
#include "texture_types.h"
#include <texture_fetch_functions.h>

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define NUM_TEX_CHANNEL 4


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
    __host__
    Texture(int pixel_width, int pixel_height, unsigned char* raw_pixel) : 
        pixel_width(pixel_width), pixel_height(pixel_height), channel(NUM_TEX_CHANNEL) {
        size_t tot = (size_t) pixel_width * pixel_height * channel;
        pixels.assign(raw_pixel, raw_pixel + tot);
    }
    int channel;
    int pixel_width;
    int pixel_height;
    std::vector<unsigned char> pixels;
};

struct TextureGPU {
    __host__
    TextureGPU(Texture const& hst_tex) :
        pixel_width(hst_tex.pixel_width), pixel_height(hst_tex.pixel_height), dev_arr(nullptr), tex(0) {
        //reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api

        cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
        CHECK_CUDA(cudaMallocArray(&dev_arr, &desc, pixel_width, pixel_height));
        CHECK_CUDA(cudaMemcpyToArray(dev_arr, 0, 0, hst_tex.pixels.data(), (size_t)pixel_width * pixel_height * hst_tex.channel * sizeof(unsigned char), cudaMemcpyHostToDevice));
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = dev_arr;

        cudaTextureDesc tex_desc;
        memset(&tex_desc, 0, sizeof(tex_desc));
        tex_desc.addressMode[0] = cudaAddressModeWrap;
        tex_desc.addressMode[1] = cudaAddressModeWrap;
        tex_desc.filterMode = cudaFilterModeLinear;
        tex_desc.readMode = cudaReadModeNormalizedFloat;
        tex_desc.normalizedCoords = 1;

        CHECK_CUDA(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, 0));
    }
    __host__
    void free() {
        CHECK_CUDA(cudaFreeArray(dev_arr));
        CHECK_CUDA(cudaDestroyTextureObject(tex));
    }

    __device__ color_t sample(glm::vec2 const& uv) {
#ifndef __CUDA_ARCH__
#define f(x,y,z) color_t(1,0,0)
#else
#define f(x,y,z) tex2D<float4>(x,y,z)
#endif
        auto col = f(tex, uv.x, uv.y);
        return color_t(col.x, col.y, col.z);
    }

    int pixel_width;
    int pixel_height;
    cudaArray_t dev_arr;
    cudaTextureObject_t tex;
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
        int diffuse;
        int bump;
    } textures;      // optional: invalid = -1

    Type type;       // optional: default = DIFFUSE
    float roughness; // optional: default = 1

    __host__ __device__ Material() : 
        diffuse(BACKGROUND_COLOR),
        hasReflective(0),
        hasRefractive(0),
        ior(1),
        emittance(0),
        type(DIFFUSE),
        roughness(1)
    {
        specular.color = color_t(0);
        specular.exponent = 0;
        textures.diffuse = textures.bump = -1;
    }

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

// the result of 0 sanity
struct JunksFromMain {
    float zoom, theta, phi;
    glm::vec3 cameraPosition;
    glm::vec3 ogLookAt;
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

    // only used by textured points
    glm::vec2 uv;
    color_t tex_color;

    __host__ __device__ ShadeableIntersection() 
        : t(-1), surfaceNormal(0), hitPoint(0), materialId(-1), uv(-1), tex_color(-1) { }

    __host__ __device__ friend bool operator<(ShadeableIntersection const& a, ShadeableIntersection const& b) {
        return a.materialId < b.materialId;
    }
};

// Stored in the scene structure
// automatically generated from objects with emittance > 0 
// to provide info about lights in the scene
struct Light {
    color_t color;
    float intensity;
    glm::vec3 position;
};




// --------------------------------------------
// MESH STRUCTURE:
// Mesh --> a range of triangles
// Triangle --> Material
// Material --> Textures
// --------------------------------------------
// 
// Stored in the scene structure
struct Triangle {
    glm::ivec3 verts;
    glm::ivec3 norms;
    
    // only used by textured faces
    glm::ivec3 uvs;
    int mat_id;

    // only used by normal-mapped faces
    glm::ivec3 tangents;

    Triangle(glm::ivec3 verts, glm::ivec3 norms, glm::ivec3 uvs, int mat_id) 
        : verts(verts), norms(norms), uvs(uvs), mat_id(mat_id), tangents(-1) {}
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
    Vertex* vertices;
    Normal* normals;
    TexCoord* uvs;
    TextureGPU* texs; //array of textures pointers
    Triangle* tris;
    glm::vec4* tangents;
    Mesh* meshes;
};