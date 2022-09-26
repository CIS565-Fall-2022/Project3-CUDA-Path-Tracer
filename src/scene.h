#pragma once

#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "tinyobjloader/tiny_obj_loader.h"
#include "glm/glm.hpp"
#include "utilities.h"
#include "cudaUtil.h"
#include "sceneStructs.h"
#include "material.h"
#include "image.h"
#include "bvh.h"

#define MESH_DATA_STRUCT_OF_ARRAY false
#define MESH_DATA_INDEXED false

struct Triangle {
    glm::vec3 vertex[3];
    glm::vec3 normal[3];
    glm::vec2 texcoord[3];
};

struct MeshData {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
#if MESH_DATA_INDEXED
    std::vector<glm::ivec3> indices;
#endif
};

struct ModelInstance {
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    glm::mat4 transform;
    glm::mat4 transfInv;
    glm::mat3 normalMat;

    int materialId;
    MeshData* meshData;
};

class Resource {
public:
    static MeshData* loadOBJMesh(const std::string& filename);
    static MeshData* loadGLTFMesh(const std::string& filename);
    static MeshData* loadModelMeshData(const std::string& filename);
    static Image* loadTexture(const std::string& filename);

    static void clear();

private:
    static std::map<std::string, MeshData*> meshDataPool;
    static std::map<std::string, Image*> texturePool;
};

class Scene;

struct DevScene {
    void createDevData(Scene& scene);
    void freeDevData();

    __device__ int getMTBVHId(glm::vec3 dir) {
        glm::vec3 absDir = glm::abs(dir);
        if (absDir.x > absDir.y) {
            if (absDir.x > absDir.z) {
                return dir.x > 0 ? 0 : 1;
            }
            else {
                return dir.z > 0 ? 4 : 5;
            }
        }
        else {
            if (absDir.y > absDir.z) {
                return dir.y > 0 ? 2 : 3;
            }
            else {
                return dir.z > 0 ? 4 : 5;
            }
        }
    }

    __device__ void getIntersecGeomInfo(int primId, glm::vec3 bary, Intersection& intersec) {
        glm::vec3 va = devVertices[primId * 3 + 0];
        glm::vec3 vb = devVertices[primId * 3 + 1];
        glm::vec3 vc = devVertices[primId * 3 + 2];

        glm::vec3 na = devNormals[primId * 3 + 0];
        glm::vec3 nb = devNormals[primId * 3 + 1];
        glm::vec3 nc = devNormals[primId * 3 + 2];

        glm::vec2 ta = devTexcoords[primId * 3 + 0];
        glm::vec2 tb = devTexcoords[primId * 3 + 1];
        glm::vec2 tc = devTexcoords[primId * 3 + 2];

        intersec.position = va * bary.x + vb * bary.y + vc * bary.z;
        intersec.normal = na * bary.x + nb * bary.y + nc * bary.z;
        intersec.texcoord = ta * bary.x + tb * bary.y + tc * bary.z;
    }

    __device__ bool intersectPrimitive(int primId, Ray ray, float& dist, glm::vec3& bary) {
        glm::vec3 va = devVertices[primId * 3 + 0];
        glm::vec3 vb = devVertices[primId * 3 + 1];
        glm::vec3 vc = devVertices[primId * 3 + 2];

        if (!glm::intersectRayTriangle(ray.origin, ray.direction, va, vb, vc, bary)) {
            return false;
        }
        glm::vec3 hitPoint = va * bary.x + vb * bary.y + vc * bary.z;
        dist = glm::length(hitPoint - ray.origin);
        return true;
    }

    __device__ bool intersectPrimitiveDetailed(int primId, Ray ray, Intersection& intersec) {
        glm::vec3 va = devVertices[primId * 3 + 0];
        glm::vec3 vb = devVertices[primId * 3 + 1];
        glm::vec3 vc = devVertices[primId * 3 + 2];
        glm::vec3 bary;

        if (!glm::intersectRayTriangle(ray.origin, ray.direction, va, vb, vc, bary)) {
            return false;
        }

        glm::vec3 na = devNormals[primId * 3 + 0];
        glm::vec3 nb = devNormals[primId * 3 + 1];
        glm::vec3 nc = devNormals[primId * 3 + 2];

        glm::vec2 ta = devTexcoords[primId * 3 + 0];
        glm::vec2 tb = devTexcoords[primId * 3 + 1];
        glm::vec2 tc = devTexcoords[primId * 3 + 2];

        intersec.position = va * bary.x + vb * bary.y + vc * bary.z;
        intersec.normal = na * bary.x + nb * bary.y + nc * bary.z;
        intersec.texcoord = ta * bary.x + tb * bary.y + tc * bary.z;
        intersec.dist = glm::length(intersec.position - ray.origin);
        return true;
    }

    __device__ int intersect(Ray ray, Intersection& intersec) {
        float closestDist = FLT_MAX;
        int closestPrimId = NullPrimitive;
        glm::vec3 closestBary;

        MTBVHNode* nodes = devBVHNodes[getMTBVHId(-ray.direction)];
        int node = 0;

        while (node != BVHSize) {
            AABB& bound = devBoundingBoxes[nodes[node].boundingBoxId];
            float boundDist;
            bool boundHit = bound.intersect(ray, boundDist);

            // Only intersect a primitive if its bounding box is hit and
            // that box is closer than previous hit record
            if (boundHit && boundDist < closestDist) {
                int primId = nodes[node].primitiveId;
                if (primId != NullPrimitive) {
                    float dist;
                    glm::vec3 bary;
                    bool hit = intersectPrimitive(primId, ray, dist, bary);

                    if (hit && dist < closestDist) {
                        closestDist = dist;
                        closestBary = bary;
                        closestPrimId = primId;
                    }
                }
                node++;
            }
            else {
                node = nodes[node].nextNodeIfMiss;
            }
        }
        if (closestPrimId != NullPrimitive) {
            getIntersecGeomInfo(closestPrimId, closestBary, intersec);
            intersec.dist = closestDist;
            intersec.incomingDir = -ray.direction;
            intersec.materialId = devMaterialIds[closestPrimId];
        }
        else {
            intersec.dist = -1.f;
        }
        return closestPrimId;
    }

    glm::vec3* devVertices = nullptr;
    glm::vec3* devNormals = nullptr;
    glm::vec2* devTexcoords = nullptr;
    AABB* devBoundingBoxes = nullptr;
    MTBVHNode* devBVHNodes[6] = { nullptr };
    int BVHSize;

    int* devMaterialIds = nullptr;
    Material* devMaterials = nullptr;
    glm::vec3* devTextureData = nullptr;
    DevTextureObj* devTextureObjs = nullptr;
};

class Scene {
public:
    friend struct DevScene;

    Scene(const std::string& filename);
    ~Scene();

    void buildDevData();
    void clear();

private:
    void loadModel(const std::string& objectId);
    void loadMaterial(const std::string& materialId);
    void loadCamera();

public:
    RenderState state;
    std::vector<ModelInstance> modelInstances;
    std::vector<Image*> textures;
    std::vector<Material> materials;
    std::map<std::string, int> materialMap;
    std::vector<int> materialIds;
    int BVHSize;
    std::vector<AABB> boundingBoxes;
    std::vector<std::vector<MTBVHNode>> BVHNodes;
    MeshData meshData;

    DevScene devScene;

private:
    std::ifstream fpIn;
};
