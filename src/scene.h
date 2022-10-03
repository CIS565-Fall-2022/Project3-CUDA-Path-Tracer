#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadObj(const char* filepath, 
                glm::mat4 transform, 
                glm::vec3 trans, 
                glm::vec3 rot,
                glm::vec3 scale,
                int matId,
                std::vector<Triangle>* triangleArray,
                const char* basepath,
                bool triangulate);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
