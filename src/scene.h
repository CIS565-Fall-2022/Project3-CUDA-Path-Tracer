#pragma once

#include <vector>
#include <set>
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
    bool hasTexture(const char* texName);
    int loadObj(const char* filename, std::vector<Triangle>* triangleArray, const char* basepath, bool triangulate);
    int loadMaterial(string materialid);
    int loadTexture(string textureid);
    Texture createTexture(string texturePath);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;

    // for textures
    std::vector<Texture> textures;
    // std::vector<int> textureChannels;

    RenderState state;

    std::vector<Geom> lights;
    int numLights = 0;
};
