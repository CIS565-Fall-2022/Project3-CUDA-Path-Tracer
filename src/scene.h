#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;
using namespace scene_structs;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    void loadDefaultCamera();
    int loadGeom(string objectid);
    int loadCamera();
    int loadTinyGltf(std::string filename);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Image> images;
    std::vector<Triangle> triangles; // all triangles in the scene from all meshes

    RenderState state;
};
