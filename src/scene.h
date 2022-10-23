#pragma once

#include <vector>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <iostream>
#include "setting.h"
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

    // Texture related
    int markTexture(string filePath);
    int markTextureNormal(string filePath);

    // BVH related
    BVHNode* recursiveBuildBVH(int st, int ed, int& count,
        std::vector<BVHPrimitiveInfo>& primInfos, std::vector<Triangle>& reorderPrims);
    int flattenBVHNode(BVHNode* node, int *offset, std::vector<LinearBVHNode>& vec);
    void deleteBVHNode(BVHNode* node);

    // Skybox
    int markSkyBox(string filePath);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    
    // Object loading and texture 
    std::vector<Triangle> triangles;
    std::vector<string> textureIds;
    std::vector<string> textureNormalIds;

    // BVH
    std::vector<LinearBVHNode> bvhNodes;
    
    // Skybox
    string skyboxId;

    RenderState state;
};
