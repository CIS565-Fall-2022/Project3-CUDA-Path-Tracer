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

#define INDEXED_MESH_DATA false

struct Model {
    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
#if INDEXED_MESH_DATA
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
    Model* meshData;
};

class Resource {
public:
    static Model* loadModel(const std::string& filename);
    static Image* loadTexture(const std::string& filename);

    static void clear();

private:
    static std::map<std::string, Model*> modelPool;
    static std::map<std::string, Image*> texturePool;
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

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> texcoords;
#if INDEXED_MESH_DATA
    std::vector<glm::ivec3> indices;
#endif

    glm::vec3* devVertices = nullptr;
    glm::vec3* devNormals = nullptr;
    glm::vec3* devTexcoords = nullptr;
#if INDEXED_MESH_DATA
    glm::ivec3* devIndices = nullptr;
#endif
    AABB* devBoundingBoxes = nullptr;
    glm::vec3* devMaterials = nullptr;
    glm::vec3* devTexture = nullptr;

private:
    std::ifstream fpIn;
};
