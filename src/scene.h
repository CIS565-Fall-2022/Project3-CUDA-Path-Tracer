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
#include "consts.h"

class Scene {
private:
    std::ifstream fp_in;
    int loadMaterial(std::string materialid);
    bool loadGeom();
    void loadCamera();
public:
    Scene(std::string filename, bool load_render_state = true);
    ~Scene();
    
    std::string filename;

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;

    // buffers
    std::vector<Mesh> meshes;
    std::vector<Normal> normals;
    std::vector<Vertex> vertices;
    std::vector<TexCoord> uvs;
    std::vector<glm::vec4> tangents;
    
    std::vector<Texture> textures;

    // all triangles, untransformed, in model space
    std::vector<Triangle> triangles;

    // caches
    std::unordered_map<std::string, int> tex_name_to_id;
    std::unordered_map<std::string, int> mtl_to_id;

    RenderState state;
    AABB world_AABB;
};