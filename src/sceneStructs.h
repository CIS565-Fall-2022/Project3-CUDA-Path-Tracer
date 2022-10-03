#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <tiny_gltf.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

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
    int mesh_id;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};
//Added here
struct Primitive
{
    int count;
    int index_Offset;
    int vertex_Offset;
    int normal_Offset=-1;
    int uv_Offset=-1;
    int tangent_Offset = -1;
    int mat_id;
    glm::vec3 boundingBoxMax;
    glm::vec3 boundingBoxMin;
    glm::mat4 pivotTransform;
};


struct PrimitiveData
{
    Primitive* primitives;
    glm::vec3* vertices;
    glm::vec2* texCoords;
    glm::vec3* normals;
    glm::vec4* tangents;
    uint16_t* indices;
    void free()
    {
        cudaFree(primitives);
        cudaFree(vertices);
        cudaFree(texCoords);
        cudaFree(normals);
        cudaFree(tangents);
        cudaFree(indices);
    }
};

struct Texture
{
    int size;
    int width;
    int height;
    int component;
    unsigned char* image;
};

struct TextureInfo
{
    int index;
    int texCoord;
    TextureInfo& operator=(const tinygltf::TextureInfo temp)
    {
        index = temp.index;
        texCoord = temp.texCoord;
        return *this;
    }
};

struct NormalTextureInfo
{
    float scale = 1.0f;
    int index = -1;  // required
    int texCoord;
    NormalTextureInfo& operator=(const tinygltf::NormalTextureInfo temp)
    {
        index = temp.index;
        scale = temp.scale;
        texCoord = temp.texCoord;
        return *this;
    }
};
struct Mesh
{
    int prim_count;
    int prim_offset;
};

struct PBRShadingAttribute
{
    glm::vec4 baseColor = glm::vec4(1.0);

    TextureInfo baseColorTexture;
    double metallicFactor;   // default 1
    double roughnessFactor;  // default 1
    TextureInfo metallicRoughnessTexture;

    __host__ __device__ PBRShadingAttribute()
    {
        baseColor = glm::vec4(1.0);
        metallicFactor = 1.0f;
        roughnessFactor = 1.0f;
    }

};

struct Material {

    int texOffset = 0;
    bool gltf = false;


    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    TextureInfo emissiveTexture;
    NormalTextureInfo normalTexture;
    PBRShadingAttribute pbrVal;

    glm::vec3 emissiveFactor;

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
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 intersectionPoint;
  glm::vec3 surfaceNormal;
  bool outSide;
  int materialId;
};

//Add for string compaction

struct isPathCompleted
{
    __host__ __device__
    bool operator()(const PathSegment& path)
    {
        return path.remainingBounces <= 0;
    }
};

struct compareIntersection
{
    __host__ __device__
    bool operator()(const ShadeableIntersection& a,const ShadeableIntersection& b)
    {
       return a.materialId > b.materialId;
    }
};