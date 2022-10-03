#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>


#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE
#include <tiny_gltf.h>

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadGLTF(const string filename);
    int loadGLTFNodes(const std::vector<tinygltf::Node>& nodes, const tinygltf::Node& node, bool* isLoaded,glm::mat4& transformMat);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    //Added here
    std::vector<Mesh> meshes;
    std::vector<Texture> textures;
    std::vector<Primitive> primitives;

    //Primitive data
    std::vector<uint16_t> mesh_indices;
    std::vector<glm::vec3> mesh_vertices;
    std::vector<glm::vec3> mesh_normals;
    std::vector<glm::vec4> mesh_tangents;
    std::vector<glm::vec2> mesh_uvs;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
