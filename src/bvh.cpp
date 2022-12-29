#include "bvh.h"

// referred to https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

struct Bounds {
  glm::vec3 min;
  glm::vec3 max;

  Bounds(glm::vec3& min, glm::vec3& max)
    :min(min), max(max) {}
};

Bounds boundsUnion(const Bounds &b1, const Bounds &b2) {
  glm::vec3 newMin = glm::min(b1.min, b2.min);
  glm::vec3 newMax = glm::max(b1.max, b2.max);
  return Bounds(newMin, newMax);
}

//struct BvhGeomData {
//  
//};
//
//struct BvhNode {
//  Bounds bounds;
//  // points to original geoms array by index. numGeoms > 0 means node is leaf
//  int firstGeomOffset, numGeoms;
//  // points to Bvh tree's flattened array structure
//  int leftChildIndex, rightChildIndex;
//};
//
//Bvh::Bvh(const std::vector<Geom>& geoms)
//{
//  // Preprocessing - get centroids and bounding boxes of primitives
//
//  int numNodes = 2 * geoms.size() - 1;
//  std::vector<BvhNode> bvhNodes(numNodes);
//}
