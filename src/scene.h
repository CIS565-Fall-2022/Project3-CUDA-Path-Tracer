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
	int loadGLTF(string fileDir, Geom& geom);
	int loadGeom(string objectid);
	int loadCamera();
	

public:

	Scene(string filename);
	~Scene();

	std::vector<Geom> geoms;
	std::vector<Material> materials;
	RenderState state;
	vector<glm::vec3> maps;
	std::vector<Triangle> triangles;
	
};
