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
    int loadObj(const char* fileName);
    int loadMesh(const char* fileName);
    glm::vec3 loadTexture(Geom &geo, const char* fileName);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;

    //std::vector<Material> m_materials;

    std::vector<Material> OBJ_materials;

    RenderState state;

    std::vector<Geom> Obj_geoms;
    
    BVHNode* buildBVH(int start_index, int end_index);
    void reformatBVHToGPU();

    int num_tris = 0;
    int num_geoms = 0;
    std::vector<Tri> mesh_tris;
    std::vector<Tri> mesh_tris_sorted;
    BVHNode* root_node;
    int num_nodes = 0;
    std::vector<BVHNode_GPU> bvh_nodes_gpu;
    std::vector<TriBounds> tri_bounds;

};
