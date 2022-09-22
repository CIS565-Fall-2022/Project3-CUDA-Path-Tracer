#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "material.h"

class Scene {
private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialId);
    int loadGeom(std::string objectid);
    int loadCamera();
public:
    Scene(std::string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
