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
    int loadObj(Geom&, const char*);
    int loadTexture(string textureID);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Triangle> triangles;
    std::vector<Texture> textures;
    std::vector<glm::vec3> textureColors;
    std::vector<Material> materials;
    RenderState state;

    std::vector<Geom> lights;
    int lightCount = 0;
};
