#pragma once
#include <iostream>
#include <vector>
#include "sceneStructs.h"
#include "utilities.h"
struct bvhNode {
    BBox box;
    int left = -1;
    int right = -1;
    int triID = -1;
    // int splitAxis, firstPrimOffset, nPrimitives;
};

struct bvhTree {
    
    struct BVHPrimitiveInfo {
        BVHPrimitiveInfo(): primitiveNumber(0){}
        BVHPrimitiveInfo(size_t primitiveNumber, const BBox& bounds)
            : primitiveNumber(primitiveNumber), bounds(bounds),
            centroid(.5f * bounds.minCorner + .5f * bounds.maxCorner) {}
      
        size_t primitiveNumber;
        BBox bounds;
       // int triID;
        glm::vec3 centroid;
    };

    bvhTree(){}
    ~bvhTree(){}

	__host__ void buildTree(const std::vector<Triangle>& triangles);
    __host__ void recursiveBuild(std::vector<BVHPrimitiveInfo>& primitiveInfo, int start, int end, int curIdx);
    __host__ void initLeaf(std::vector<BVHPrimitiveInfo>& primitiveInfo, int cur, int triIdx);
    //linear form of the tree
    std::vector<bvhNode> bvhNodes;
    int nodeCount = 0;
    int treeHeight = 0;
};

