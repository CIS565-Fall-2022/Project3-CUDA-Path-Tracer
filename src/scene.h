#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tinyobj/tiny_obj_loader.h"
#include "bvh.h"

#define DEV_TRI 1
using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadObj(Geom& geom,int id, string path);
public:
    Scene(string filename);
    ~Scene();

    int globalTriOffset = 0;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;
    RenderState state;
    bvhTree sceneBVH;
};
