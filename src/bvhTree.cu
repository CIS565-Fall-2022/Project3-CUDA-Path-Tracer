#include "bvhTree.h"
#if USE_BVH_FOR_INTERSECTION
#include <algorithm>

__host__
BVHNode::BVHNode()
	: idx(-1), minCorner(glm::vec3(FLT_MAX)), maxCorner(glm::vec3(FLT_MIN)), isLeaf(false), hasFace(false), axis(0)
{
}

__host__
BVHNode::~BVHNode()
{
}

__host__ __device__
int BVHNode::getLeftChildIdx() const
{
	return idx * 2 + 1;
}

__host__ __device__
int BVHNode::getRightChildIdx() const
{
	return idx * 2 + 2;
}

__host__
BVHTree::BVHTree()
{
}

__host__
BVHTree::~BVHTree()
{
}

// this function modifies the order of faces, so Geom.faceStartIdx may become invalid
__host__
void BVHTree::build(std::vector<Triangle>& faces)
{
	if (faces.size() == 0) return;
	// The maximum depth of nodes if we use equal counts methods to build the tree.
	int depth = glm::ceil(glm::log2((float)faces.size())) + 1;
	bvhNodes.resize(glm::pow(2, depth) - 1);

	recursiveBuild(0, faces);
}

__host__
void BVHTree::recursiveBuild(int nodeIdx, std::vector<Triangle>& faces)
{
	if (nodeIdx >= bvhNodes.size() * 2 - 1) return;
	bvhNodes[nodeIdx].idx = nodeIdx;

	if (faces.size() == 0)
	{
		bvhNodes[nodeIdx].isLeaf = true;
	}

	// find the bounding box of the node
	bvhNodes[nodeIdx].minCorner = faces[0].minCorner;
	bvhNodes[nodeIdx].maxCorner = faces[0].maxCorner;
	for (int i = 1; i < faces.size(); i++)
	{
		bvhNodes[nodeIdx].minCorner = glm::min(bvhNodes[nodeIdx].minCorner, faces[i].minCorner);
		bvhNodes[nodeIdx].maxCorner = glm::max(bvhNodes[nodeIdx].maxCorner, faces[i].maxCorner);
	}

	if (/*nodeIdx >= (bvhNodes.size() + 1) / 2 - 1 && */faces.size() == 1)
	{
		bvhNodes[nodeIdx].face = faces[0];
		bvhNodes[nodeIdx].isLeaf = true;
		bvhNodes[nodeIdx].hasFace = true;
		return;
	}

	// estimate split axis by calculating maximum extent
	int& axis = bvhNodes[nodeIdx].axis;
	axis = 0;
	glm::vec3 diagonal = bvhNodes[nodeIdx].maxCorner - bvhNodes[nodeIdx].minCorner;
	if (diagonal.x > diagonal.y && diagonal.x > diagonal.z) axis = 0;
	else if (diagonal.y > diagonal.z) axis = 1;
	else axis = 2;

	// Partition primitives into equally sized subsets
	int midIdx = faces.size() / 2;
	std::nth_element(&faces[0], &(faces[midIdx]), &(faces[faces.size() - 1]) + 1, 
		[axis](const Triangle& a, const Triangle& b)
		{
			return a.centroid[axis] < b.centroid[axis];
		});

	std::vector<Triangle> leftFaces(faces.begin(), faces.begin() + midIdx);
	recursiveBuild(bvhNodes[nodeIdx].getLeftChildIdx(), leftFaces);

	std::vector<Triangle> rightFaces(faces.begin() + midIdx, faces.end());
	recursiveBuild(bvhNodes[nodeIdx].getRightChildIdx(), rightFaces);
}

#endif