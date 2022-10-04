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
    int loadObj(string filePath, Geom &geom, int geomId);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Triangle> faces;
    std::vector<Material> materials;
    RenderState state;
};
