#include "bvh.h"

using namespace std;
__host__ void bvhTree::buildTree(std::vector<Triangle>& triangles) {
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
	//cout << "NodeCount: " << nodeCount << endl;

}

__host__ void bvhTree::recursiveBuild(std::vector<BVHPrimitiveInfo>& primitiveInfo, int start, int end, int cur) {
	if (start + 1 > end) {
		return;
	}
	//create leaf node
	if (end - start == 1) {
		initLeaf(primitiveInfo, cur, start);
		//BBox box = bvhNodes[cur].box;
		/*printf("==Leaf: nodes[%d].box={<%f,%f,%f>, <%f,%f,%f>} with geoms leaf[%d] at depth[%d]\n",
			cur,
			box.minCorner.x, box.minCorner.y, box.minCorner.z,
			box.maxCorner.x, box.maxCorner.y, box.maxCorner.z,
			start, depth);*/
		return;
	}
	else {
		BBox centroidBounds = primitiveInfo[start].bounds;
		//centroidBounds.isValid = 1;
		/*centroidBounds.minCorner = glm::vec3{ FLT_MAX };
		centroidBounds.maxCorner = glm::vec3{ FLT_MIN };*/
		for (int i = start + 1; i < end; ++i) {
			//BBox tmp = primitiveInfo[i].bounds;
			//centroidBounds += tmp;
			centroidBounds.minCorner = glm::min(centroidBounds.minCorner, primitiveInfo[i].bounds.minCorner);
			centroidBounds.maxCorner = glm::max(centroidBounds.maxCorner, primitiveInfo[i].bounds.maxCorner);
		}
			

		int dim = centroidBounds.MaximumExtent();

		int mid = (start + end) / 2;

		//preorder 
		int left = cur * 2 + 1;
		int right = cur * 2 + 2;
		/*if (centroidBounds.maxCorner[dim] == centroidBounds.minCorner[dim]) {
			initLeaf(primitiveInfo, cur, start);
			cout << "im here ! " << endl;
		}
		else {*/
			//split
		std::nth_element(&primitiveInfo[start], &primitiveInfo[mid],
			&primitiveInfo[(size_t)end - 1] + 1,
			[dim](const BVHPrimitiveInfo& a, const BVHPrimitiveInfo& b) {
				return a.centroid[dim] < b.centroid[dim];
			});
		bvhNodes[cur].box = centroidBounds;
		bvhNodes[cur].left = left;
		bvhNodes[cur].right = right;

		/*printf("==NODE: nodes[%d].box={<%f,%f,%f>, <%f,%f,%f>} with left: [%d], right: [%d], at depth [%d], geomID is [%d]\n",
			cur,
			centroidBounds.minCorner.x, centroidBounds.minCorner.y, centroidBounds.minCorner.z,
			centroidBounds.maxCorner.x, centroidBounds.maxCorner.y, centroidBounds.maxCorner.z,
			left, right, depth, bvhNodes[cur].triID);*/
		recursiveBuild(primitiveInfo, start, mid, left);
		recursiveBuild(primitiveInfo, mid, end, right);
		//}


	}
}

__host__ void bvhTree::initLeaf(std::vector<BVHPrimitiveInfo>& primitiveInfo, int cur, int triIdx) {
	bvhNodes[cur].box = primitiveInfo[triIdx].bounds;
	bvhNodes[cur].triID = primitiveInfo[triIdx].primitiveNumber;
	bvhNodes[cur].left = -1;
	bvhNodes[cur].right = -1;
}


