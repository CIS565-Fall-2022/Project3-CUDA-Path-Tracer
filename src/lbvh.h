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

// Morton code generation and sorting
unsigned int expandBits(unsigned int v);
unsigned int mortonCode3D(glm::vec3 centroid);
void computeMortonCodes(Scene* scene, const AABB& sceneAABB);
void sortMortonCodes(Scene* scene);

// Tree building
bool isLeaf(const LBVHNode* node);
int delta(unsigned int* sortedMCodes, int N, int i, int j);
int sign(unsigned int* sortedMCodes, int N, int i);

NodeRange determineRange(unsigned int* sortedMCodes, int triangleCount, int idx);
int findSplit(unsigned int* sortedMCodes, int triangleCount, NodeRange range);
void assignBoundingBoxes(Scene* scene);
void buildLBVH(Scene* scene, int triangleCount);
void traverseLBVH(Scene* scene);

bool bvhIsLeaf(const BVHNode* node);
void test_bvhIntersectionTestRecursive(const BVHNode* nodes, const Triangle* tris, Ray r, int idx, Triangle& min_tri, glm::vec3& min_barycenter, float& min_t);
float test_bvhIntersectionTest(const BVHNode* nodes, const Triangle* tris, Ray r, int triangleCount, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside);

// Construct entire LBVH
void generateLBVH(Scene* scene);
void generateBVH(Scene* scene, int triangleCount);
void flattenBVH(Scene* scene, int triangleCount);

// Small test for LBVH
void unitTest(Scene* scene);
void unitTestBVH(Scene* scene, int triangleCount);
