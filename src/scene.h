#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#include "tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadGLTF(const string &filename, Geom &newGeom);
    int loadNodes(const vector<tinygltf::Node>& nodes, int index, glm::mat4& transform);
    void addChild(const int parentIdx);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    SceneMeshesData sceneMeshesData;
    RenderState state;

    std::unordered_map<int, glm::mat4> nodeToTransform;
    std::unordered_map<int, int> geomIdMap;

    std::vector<Triangle> triangles;
    std::vector<OctreeNode> octree;
    const int treeDepth = 5;
    std::vector<int> trianglesIndices;

    std::vector<unsigned short> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> tangents;

    int colorTextureID = -1;
    int normalTextureID = -1;
    int emissiveTextureID = -1;
    std::vector<Texture> textures;
    std::vector<glm::vec3> colorTexture;
    std::vector<glm::vec3> normalTexture;
    std::vector<glm::vec3> emissiveTexture;
    glm::vec3 emissiveFactor;
};
