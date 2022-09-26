#pragma once

#include <map>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "tinyobjloader/tiny_obj_loader.h"
#include "glm/glm.hpp"
#include "utilities.h"
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

struct DevResource {
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
    std::vector<Geom> geoms;
    std::vector<ModelInstance> modelInstances;
    std::vector<Image*> textures;
    std::vector<Material> materials;
    std::map<std::string, int> materialMap;
    std::vector<int> materialIds;
    int BVHSize;
    std::vector<AABB> boundingBoxes;
    std::vector<std::vector<MTBVHNode>> BVHNodes;
    MeshData meshData;

    DevResource devResources;

private:
    std::ifstream fpIn;
};
