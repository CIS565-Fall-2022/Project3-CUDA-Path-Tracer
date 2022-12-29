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

struct BvhPrimitiveData {
  int index;
  glm::vec3 centroid;
  Bounds bounds;

  BvhPrimitiveData(int index, glm::vec3 centroid, Bounds bounds)
    :index(index), centroid(centroid), bounds(bounds) {}
};

struct BvhNode {
  Bounds bounds;
  // points to original triangle array by index. numTriangles > 0 means node is leaf
  int firstTriangleOffset = -1;
  int numTriangles = -1;
  // points to Bvh tree's flattened array structure
  int leftChildIndex = -1;
  int rightChildIndex = -1;

  BvhNode():bounds(Bounds(glm::vec3(FLT_MAX), glm::vec3(FLT_MIN))) {}
};

void updateNodeBounds(const std::vector<BvhPrimitiveData> &primitiveData, std::vector<BvhNode> &bvhNodes, int nodeIndex) {
  BvhNode& node = bvhNodes[nodeIndex];
  node.bounds = Bounds(glm::vec3(FLT_MAX), glm::vec3(FLT_MIN));

  for (int i = node.firstTriangleOffset; i < node.firstTriangleOffset + node.numTriangles; ++i) {
    node.bounds = boundsUnion(node.bounds, primitiveData[i].bounds);
  }
}

Bvh::Bvh(const std::vector<Triangle> &triangles)
{
  std::vector<BvhPrimitiveData> primitiveData;

  // Preprocessing - get centroids and bounding boxes of primitives
  for (int i = 0; i < triangles.size(); ++i) {
    const glm::vec3& v1 = triangles[i].verts[0].position;
    const glm::vec3& v2 = triangles[i].verts[1].position;
    const glm::vec3& v3 = triangles[i].verts[2].position;

    glm::vec3 centroid = (v1 + v2 + v3) * 0.333333f;
    Bounds bounds(glm::min(v1, glm::min(v2, v3)), glm::max(v1, glm::max(v2, v3)));

    BvhPrimitiveData data(i, centroid, bounds);
    primitiveData.push_back(data);
  }

  int rootNodeIndex = 0;
  int nodesUsed = 1;

  int numNodes = 2 * triangles.size() - 1;
  std::vector<BvhNode> bvhNodes(numNodes);

  BvhNode& root = bvhNodes[rootNodeIndex];
  root.leftChildIndex = 0;
  root.rightChildIndex = 0;
  root.firstTriangleOffset = 0;
  root.numTriangles = triangles.size();
}
