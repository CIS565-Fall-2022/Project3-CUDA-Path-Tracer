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
    int loadGeom(string objectid, int& geomId, int& lightId);
    int loadCamera();
    
public:
    Scene(string filename);
    ~Scene();

    int loadMesh(string fileName);

    std::vector<Geom> geoms;
    std::vector<Geom> lights;
    std::vector<Material> materials;
    std::vector<TriMesh> meshes;
    RenderState state;
};
