#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <map>
#include <unordered_map>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

#define MAX_EMITTANCE 100.0f

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom();
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;

    // buffers
    std::vector<Mesh> meshes;
    std::vector<Normal> normals;
    std::vector<Vertex> vertices;
    std::vector<TexCoord> uvs;
    
    std::vector<Texture> textures;
    std::unordered_map<string, int> tex_name_to_id;

    // all triangles, untransformed, in model space
    std::vector<Triangle> triangles;

    RenderState state;
};