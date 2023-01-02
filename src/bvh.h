#pragma once

#include "sceneStructs.h"

using namespace scene_structs;

struct Bounds {
  glm::vec3 min;
  glm::vec3 max;

  Bounds(glm::vec3& min, glm::vec3& max)
    :min(min), max(max) {}
};

struct BvhNode {
  Bounds bounds;
  // points to original triangle array by index. numTriangles > 0 means node is leaf
  int firstTriangleOffset = -1;
  int numTriangles = -1;
  // points to Bvh tree's flattened array structure
  int leftChildIndex = -1;
  int rightChildIndex = -1;

  BvhNode() :bounds(Bounds(glm::vec3(FLT_MAX), glm::vec3(FLT_MIN))) {}
};

class Bvh
{
private:
  int buildBvh(std::vector<Triangle>& triangles, int startIdx, int numTriangles, int &maxLeafSize);
public:
  std::vector<BvhNode> allBvhNodes;

  Bvh(std::vector<Triangle>& triangles, std::vector<Geom> &geoms);
  Bvh();
  //~Bvh();
};

