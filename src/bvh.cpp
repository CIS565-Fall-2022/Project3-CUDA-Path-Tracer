#include "bvh.h"
#include <iostream>

// referred to https://jacco.ompf2.com/2022/04/13/how-to-build-a-bvh-part-1-basics/

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

void updateNodeBounds(const std::vector<BvhPrimitiveData> &primitiveData, std::vector<BvhNode> &bvhNodes, int nodeIndex) {
  BvhNode& node = bvhNodes[nodeIndex];
  node.bounds = Bounds(glm::vec3(FLT_MAX), glm::vec3(FLT_MIN));

  for (int i = node.firstTriangleOffset; i < node.firstTriangleOffset + node.numTriangles; ++i) {
    node.bounds = boundsUnion(node.bounds, primitiveData[i].bounds);
  }
}

// split bounding box along longest axis by half
// x = 0, y = 1, z = 2
int getSplitAxis(const Bounds &bounds, float &out_splitPos) {
  glm::vec3 extent = bounds.max - bounds.min;
  int axis = 0;
  if (extent.y > extent.x) axis = 1;
  if (extent.z > extent[axis]) axis = 2;

  out_splitPos = bounds.min[axis] + extent[axis] * 0.5f;
  return axis;
}

void subdivide(std::vector<BvhPrimitiveData>& primitiveData, std::vector<BvhNode>& bvhNodes,
  int nodeIndex, int &nodesUsed) {
  BvhNode& node = bvhNodes[nodeIndex];
  float splitPos;
  int axis = getSplitAxis(node.bounds, splitPos);

  // Partition elements on smaller side of split to lower end of array
  // By end of loop, i is the index of the first triangle to be on upper end of split
  // i could be out of bounds if no triangles are on the right side
  int i = node.firstTriangleOffset;
  int j = i + node.numTriangles - 1;
  while (i <= j) {
    if (primitiveData[i].centroid[axis] < splitPos) {
      i++;
    }
    else {
      std::swap(primitiveData[i], primitiveData[j--]);
    }
  }

  // Base case: let node be a leaf if next split is empty
  int leftCount = i - node.firstTriangleOffset;
  if (leftCount == 0 || leftCount == node.numTriangles) { 
    return;
  }

  // otherwise, create child nodes
  int leftChildIdx = nodesUsed++; // use 2 more nodes, serially, for left and right children
  int rightChildIdx = nodesUsed++;
  node.leftChildIndex = leftChildIdx;
  node.rightChildIndex = rightChildIdx;

  bvhNodes[leftChildIdx].firstTriangleOffset = node.firstTriangleOffset;
  bvhNodes[leftChildIdx].numTriangles = leftCount;
  bvhNodes[rightChildIdx].firstTriangleOffset = i;
  bvhNodes[rightChildIdx].numTriangles = node.numTriangles - leftCount;
  node.numTriangles = 0;

  //recurse
  updateNodeBounds(primitiveData, bvhNodes, leftChildIdx);
  updateNodeBounds(primitiveData, bvhNodes, rightChildIdx);
  subdivide(primitiveData, bvhNodes, leftChildIdx, nodesUsed);
  subdivide(primitiveData, bvhNodes, rightChildIdx, nodesUsed);
}

void Bvh::buildBvh(std::vector<Triangle> &triangles, int startIdx, int numTriangles)
{
  std::vector<BvhPrimitiveData> primitiveData;

  // Preprocessing - get centroids and bounding boxes of primitives
  for (int i = startIdx; i < numTriangles; ++i) {
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

  updateNodeBounds(primitiveData, bvhNodes, rootNodeIndex);
  subdivide(primitiveData, bvhNodes, rootNodeIndex, nodesUsed);

  // reorder the triangles, taking offset into account
  // aka. triangle at position offset + 2, would have index 2 in the bvh primitive array
  // in the bvh this doesn't get updated.
  // The intersection test will use the triangle offset + idx from bvh when it gets to leaves
  // TODO: do it in place to save memory
  if (primitiveData.size() != numTriangles) {
    std::cout << "Error: primitives number should be same as triangle number" << std::endl;
  }
  std::vector<Triangle> newTriangles(numTriangles);
  for (int i = 0; i < numTriangles; ++i) {
    newTriangles[i] = triangles[startIdx + primitiveData[i].index];
  }
  for (int i = 0; i < numTriangles; ++i) {
    triangles[startIdx + i] = newTriangles[i];
  }

  // stack bvh array into big array
  allBvhNodes.insert(allBvhNodes.end(), bvhNodes.begin(), bvhNodes.begin() + nodesUsed);
}

Bvh::Bvh(std::vector<Triangle>& triangles, std::vector<Geom>& geoms) {

  // Need to - build bvh for each triangle mesh (done inside buildBvh func)
  // - reorder the triangles for each geom according to their bvh's (done inside buildBvh func)
  // - stack each geom's bvh array into 1 big array (allBvhNodes)
  // - update each geom's pointers to their bvh arrays

  for (auto& geom : geoms) {
    if (geom.type == TRIANGLE_MESH) {
      int triangleStartIdx = geom.triangleOffset;
      int numTriangles = geom.numTriangles;
      geom.bvhOffset = allBvhNodes.size();
      std::cout << "Added a bvh" << std::endl;
      buildBvh(triangles, triangleStartIdx, numTriangles);
    }
  }
}

Bvh::Bvh()
{}
