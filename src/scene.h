#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

#define MAX_EMITTANCE 100.0f

// Stored in the scene structure
// automatically generated from objects with emittance > 0 
// to provide info about what lights are in the scene
struct Light {
    color_t color;
    float intensity;
    glm::vec3 position;
};

// Stored in the scene structure
struct Triangle {
    glm::ivec3 verts;
    glm::ivec3 norms;
    Triangle(int(*arr)[6]) {
        for (int i = 0; i < 3; ++i) {
            verts[i] = (*arr)[i];
        }
        for (int i = 0; i < 3; ++i) {
            norms[i] = (*arr)[i + 3];
        }
    }
};
typedef glm::vec3 Vertex;
typedef glm::vec3 Normal;

struct Mesh {
    Mesh(int tri_start, int tri_end) : tri_start(tri_start), tri_end(tri_end) { }
    int tri_start;
    int tri_end;
};

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Light> lights;

    // index buffer
    std::vector<Mesh> meshes;
    std::vector<Normal> normals;
    std::vector<Vertex> vertices;
    // all triangles, untransformed, in model space
    std::vector<Triangle> triangles;

    RenderState state;
};