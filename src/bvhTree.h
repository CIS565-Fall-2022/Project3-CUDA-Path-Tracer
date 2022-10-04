#pragma once

#include <vector>
#include "glm/glm.hpp"
#include "sceneStructs.h"

#define USE_BVH_FOR_INTERSECTION 1

#if USE_BVH_FOR_INTERSECTION

class BVHNode
{
public:
    int idx;
    glm::vec3 minCorner, maxCorner; // The world-space bounds of this node
    //glm::vec3 centroid;
    bool isLeaf;
    bool hasFace;
    Triangle face;

    int axis; // split axis

    __host__ BVHNode();
    __host__ ~BVHNode();

    __host__ __device__ int getLeftChildIdx() const;
    __host__ __device__ int getRightChildIdx() const;

};

class BVHTree
{
public:
    //BVHNode* root;
    std::vector<BVHNode> bvhNodes;

    __host__ BVHTree();
    __host__ ~BVHTree();
    __host__ void build(std::vector<Triangle>& faces);

private:
    __host__ void recursiveBuild(int nodeIdx, std::vector<Triangle>& faces);

};

#endif