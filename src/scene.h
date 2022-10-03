#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "lbvh.h"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadOBJ(string filename, int objectid);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    //std::vector<MortonCode> mcodes;
    std::vector<unsigned int> mcodes;
    std::vector<LBVHNode> lbvh;
    AABB sceneAABB;
    RenderState state;
};
