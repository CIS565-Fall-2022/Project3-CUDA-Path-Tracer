#pragma once

#include <vector>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "Texture.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadGeomTriangles(Geom& geom, string filePath);

public:
    Scene(string filename);
    ~Scene();

    int markTexture(string filePath);
    int markTextureNormal(string filePath);

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Triangle> triangles;

    std::vector<string> textureIds;
    std::vector<string> textureNormalIds;

    RenderState state;
};
