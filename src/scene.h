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
    int loadMaterial(tinygltf::Material, int id);
    int loadGeom(string objectid);
    int loadGeom(tinygltf::Mesh mesh, int id);
    int loadCamera();
public:
    Scene(string filename);
    Scene(tinygltf::Model model);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
