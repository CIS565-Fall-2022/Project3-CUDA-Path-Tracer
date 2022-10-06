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
    int loadTrianglesForMesh(string filename, Geom& thisMesh);
public:
    Scene(string filename);
    ~Scene();
    void freeObjs();

    std::vector<unsigned int> lightIndices;
    std::vector<Geom> geoms;
    std::vector<Material> materials;

    std::vector<Triangle>* globalTriangles;
    unsigned int globalTriangleCount = 0;
    RenderState state;
};
