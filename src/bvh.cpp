#include "bvh.h"

using namespace std;
__host__ void bvhTree::buildTree(const std::vector<Triangle>& triangles) {
	if (triangles.size() == 0)
		return;
	std::vector<BVHPrimitiveInfo> primitiveInfo(triangles.size());
	for (size_t i = 0; i < triangles.size(); ++i)
		primitiveInfo[i] = {i, triangles[i].bbox};
	int depth = glm::ceil(glm::log2(float(triangles.size())));
	int allocSize = glm::pow(2, depth+1) - 1;
	//allocate memory for nodes
	bvhNodes.resize(allocSize);
	recursiveBuild(primitiveInfo, 0, triangles.size(), 0);
	nodeCount = allocSize;
}

__host__ void bvhTree::recursiveBuild(std::vector<BVHPrimitiveInfo>& primitiveInfo, int start, int end, int cur) {
	if (start + 1 > end) {
		return;
	}
	//create leaf node
	if (end - start == 1) {
		initLeaf(primitiveInfo, cur, start);
		return;
	}
	else {
		BBox centroidBounds = primitiveInfo[start].bounds;
		for (int i = start + 1; i < end; ++i) {
			centroidBounds.minCorner = glm::min(centroidBounds.minCorner, primitiveInfo[i].bounds.minCorner);
			centroidBounds.maxCorner = glm::max(centroidBounds.maxCorner, primitiveInfo[i].bounds.maxCorner);
		}
			
		int dim = centroidBounds.MaximumExtent();
		int mid = (start + end) / 2;

		//preorder 
		int left = cur * 2 + 1;
		int right = cur * 2 + 2;
		
		//split
		std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
			&primitiveInfo[(size_t)end - 1] + 1,
			[dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
				return a.centroid[dim] < b.centroid[dim];
			});
		bvhNodes[cur].box = centroidBounds;
		bvhNodes[cur].left = left;
		bvhNodes[cur].right = right;
		
		recursiveBuild(primitiveInfo, start, mid, left);
		recursiveBuild(primitiveInfo, mid, end, right);
	}
}

__host__ void bvhTree::initLeaf(std::vector<BVHPrimitiveInfo>& primitiveInfo, int cur, int triIdx) {
	bvhNodes[cur].box = primitiveInfo[triIdx].bounds;
	bvhNodes[cur].triID = primitiveInfo[triIdx].primitiveNumber;
	bvhNodes[cur].left = -1;
	bvhNodes[cur].right = -1;
}


