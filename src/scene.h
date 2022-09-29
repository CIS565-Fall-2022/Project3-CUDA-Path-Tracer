#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
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
    int loadGLTF(string filename, Geom& geom);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Primitive> primitives;
    std::vector<int> mesh_indices;
    const float* mesh_vertices;
    const float* mesh_uvs = nullptr;
    const float* mesh_tangents = nullptr;
    const float* mesh_normal = nullptr;

    std::vector<Material> materials;
    std::vector<Texture> textures;
    RenderState state;
};
