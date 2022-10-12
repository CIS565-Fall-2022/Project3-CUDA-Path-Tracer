#pragma once

#include <cuda.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "glm/glm.hpp"
#include "utilities.h"
#include "scene.h"
#include "sceneStructs.h"

class Scene;

/// LBVH FUNCTIONS /// 

// Morton code generation and sorting
unsigned int expand(unsigned int n);
unsigned int mortonCode3D(const glm::vec3& centroid);
void computeMortonCodes(Scene* scene, const AABB& sceneAABB);
void sortMortonCodes(Scene* scene);

// Tree building
bool isLeaf(const LBVHNode* node);
int delta(unsigned int* sortedMCodes, int N, int i, int j);
int sign(unsigned int* sortedMCodes, int N, int i);

NodeRange determineRange(unsigned int* sortedMCodes, int triangleCount, int idx);
int findSplit(unsigned int* sortedMCodes, int triangleCount, NodeRange range);
void assignBoundingBoxes(Scene* scene);
void buildLBVH(Scene* scene, int leafStart, int triangleCount);

// Construct LBVH
void generateLBVH(Scene* scene);

/// BVH FUNCTIONS /// 

void updateBounds(Scene* scene, const int idx);
int maxExtent(glm::vec3 extent);
float evalSAH(Scene* scene, BVHNode* node, float queryPos, int axis);
void calculateCost(Scene* scene, BVHNode* node, float& split, int& axis);
void chooseSplit(Scene* scene, BVHNode* node, float& split, int& axis);
void addChildren(Scene* scene, BVHNode* node);

void generateBVH(Scene* scene);
