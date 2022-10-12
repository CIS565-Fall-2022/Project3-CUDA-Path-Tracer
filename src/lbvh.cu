#include "lbvh.h"

/// LBVH FUNCTIONS ///

// This optimized LBVH is based on the paper "Maximizing Parallelism in the Construction of BVHs,
// Octrees, and k-d Trees" by Tero Karras of NVIDIA Research

bool morton_sort(const MortonCode& a, const MortonCode& b) {
    return a.code < b.code;
}

bool isLeaf(const LBVHNode* node) {
    return node->left == 0xFFFFFFFF && node->right == 0xFFFFFFFF;
}

AABB Union(AABB left, AABB right) {
    glm::vec3 umin = glm::min(left.min, right.min);
    glm::vec3 umax = glm::max(left.max, right.max);
    return AABB{ umin, umax }; 
}

// Expand 10-bit integer into 30-bit integer
unsigned int expand(unsigned int n)
{
    n = (n | (n << 16)) & 0b00000011000000000000000011111111;
    n = (n | (n << 8)) & 0b00000011000000001111000000001111;
    n = (n | (n << 4)) & 0b00000011000011000011000011000011;
    n = (n | (n << 2)) & 0b00001001001001001001001001001001;
    return n;
}

// Based on PBRT 4.3.3. and Tero Karras version at https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
unsigned int mortonCode3D(const glm::vec3& centroid) {
    // Convert centroid coordinates to value between 0 and 1024
    float x = min(max(centroid.x * 1024.0f, 0.0f), 1023.0f);
    float y = min(max(centroid.y * 1024.0f, 0.0f), 1023.0f);
    float z = min(max(centroid.z * 1024.0f, 0.0f), 1023.0f);

    // Expand each 10 bit value so that ith value is at 3 * ith position
    unsigned int xx = expand((unsigned int)x);
    unsigned int yy = expand((unsigned int)y);
    unsigned int zz = expand((unsigned int)z);

    // Interleave the bits
    return (xx << 2) | (yy << 1) | zz;
}

void computeMortonCodes(Scene* scene, const AABB& sceneAABB) {
    for (int i = 0; i < scene->triangles.size(); i++) {
        // Find centroid of triangle's bounding box
        glm::vec3 centroid = 0.5f * scene->triangles[i].aabb.min + 0.5f * scene->triangles[i].aabb.max;

        // Normalize centroid w.r.t. scene bounding box
        glm::vec3 norm_centroid = (centroid - sceneAABB.min) / (sceneAABB.max - sceneAABB.min);

        // Calculate Morton code and add to list
        MortonCode mcode;
        mcode.objectId = i;
        mcode.code = mortonCode3D(norm_centroid);
        scene->mcodes.push_back(mcode);
    }
}

void sortMortonCodes(Scene* scene) {
    std::vector<MortonCode> mcodes_copy = scene->mcodes;
    std::sort(mcodes_copy.begin(), mcodes_copy.end(), morton_sort);
    scene->mcodes = mcodes_copy;
}

// Determines the number of common bits between two numbers 
int delta(MortonCode* sortedMCodes, int N, int i, int j) {
    // Range check
    if (j < 0 || j >= N) {
        return -1;
    }

    if (sortedMCodes[i].code == sortedMCodes[j].code)
    {
        return 32 + __lzcnt(i ^ j);
    }
    
    return __lzcnt(sortedMCodes[i].code ^ sortedMCodes[j].code);
}

// Determines in which direction the node's range will grow
int sign(MortonCode* sortedMCodes, int N, int i) {
    int diff = delta(sortedMCodes, N, i, i + 1) - delta(sortedMCodes, N, i, i - 1);
    return (diff >= 0) ? 1 : -1;
}

NodeRange determineRange(MortonCode* sortedMCodes, int triangleCount, int i) {
    // Determine direction of range (+1 or -1)
    int d = sign(sortedMCodes, triangleCount, i);

    // Compute upper bound of range
    int deltaMin = delta(sortedMCodes, triangleCount, i, i - d);
    int lMax = 2;
    while (delta(sortedMCodes, triangleCount, i, i + lMax * d) > deltaMin) {
        lMax = lMax * 2;
    }

    // Find the other end with binary search
    int l = 0;
    for (int t = lMax / 2; t >= 1; t /= 2) {
        if (delta(sortedMCodes, triangleCount, i, i + (l + t) * d) > deltaMin) {
            l = l + t;
        }
    }
    int j = i + l * d;

    return NodeRange{ i, j, l, d };
}

int findSplit(MortonCode* sortedMCodes, int triangleCount, NodeRange range) {
    int i = range.i;
    int j = range.j;
    int l = range.l;
    int d = range.d;    
    
    // Find split position with binary search
    int deltaNode = delta(sortedMCodes, triangleCount, range.i, range.j);
    int s = 0;
    int t = l;
    do {
        t = ceil(t / 2.f);
        if (delta(sortedMCodes, triangleCount, i, i + (s + t) * d) > deltaNode) {
            s = s + t;
        }
    } while (t > 1);

    int gamma = i + s * d + min(d, 0);
    
    return gamma;
}

// Recursively assigns bounding boxes to each node, start from the leaf nodes and recursing upwards
AABB assignBoundingBoxes(Scene* scene, LBVHNode* node) {

    if (!isLeaf(node)) {
        AABB leftAABB = assignBoundingBoxes(scene, &scene->lbvh[node->left]);
        AABB rightAABB = assignBoundingBoxes(scene, &scene->lbvh[node->right]);
        node->aabb = Union(leftAABB, rightAABB);
    }

    return node->aabb;
}

// Tree-building functions
void buildLBVH(Scene* scene, int leafStart, int triangleCount, int meshNum) {
    // Resize LBVH
    int numLeaf = triangleCount;
    int numInternal = triangleCount - 1;
    int internalStart = leafStart + numLeaf;
    scene->lbvh.resize(numLeaf + numInternal);
    scene->sorted_triangles.resize(numLeaf);

    // Initialize leaf nodes
    for (int i = leafStart; i < numLeaf; ++i) {
        LBVHNode leafNode;
        leafNode.objectId = scene->mcodes[i - leafStart].objectId; 
        leafNode.aabb = scene->triangles[leafNode.objectId].aabb;
        leafNode.left = 0xFFFFFFFF;
        leafNode.right = 0xFFFFFFFF;
        scene->lbvh[i] = leafNode;

        scene->sorted_triangles[i] = scene->triangles[leafNode.objectId];
    }
    scene->triangles = scene->sorted_triangles;

    // Initialize internal nodes
    for (int j = internalStart; j < internalStart + numInternal; ++j) {
        LBVHNode internalNode;

        // Determine range
        NodeRange range = determineRange(scene->mcodes.data(), triangleCount, j - triangleCount);

        // Find split position
        int split = findSplit(scene->mcodes.data(), triangleCount, range);
    
        int leftChild = -1;
        int rightChild = -1;
        if (min(range.i, range.j) == split) {
            leftChild = split;
        }
        else {
            leftChild = triangleCount + split;
        }

        if (max(range.i, range.j) == split + 1) {
            rightChild = split + 1;
        }
        else {
            rightChild = triangleCount + split + 1;
        }

        internalNode.objectId = -1;
        internalNode.left = leftChild;
        internalNode.right = rightChild;
        scene->lbvh[j] = internalNode;
    }
    // Assign bounding boxes here
    assignBoundingBoxes(scene, &scene->lbvh[triangleCount]);
}

void generateLBVH(Scene* scene)
{
    for (int i = 0; i < scene->meshCount; i++) {
        // Morton code computation
        computeMortonCodes(scene, scene->mesh_aabbs[i]);

        // Sort Morton codes
        sortMortonCodes(scene);

        // Build tree from sorted Morton codes
        buildLBVH(scene, scene->lbvh.size(), scene->mcodes.size(), i);

        scene->mcodes.clear();
    }
}

/// BASIC BVH FUNCTIONS ///

// Counter to keep track of the current available node in the tree
int idx = 1;

// Finds the new bounds of the aabb
void updateBounds(Scene* scene, const int idx)
{
    BVHNode& node = scene->bvh[idx];
    for (int i = node.firstTri; i < node.firstTri + node.numTris; ++i)
    {
        node.aabb = Union(node.aabb, scene->triangles[i].aabb);
    }
}

int maxExtent(glm::vec3 extent) {
    if (extent.x > extent.y && extent.x > extent.z) {
        return 0;
    }
    else if (extent.y > extent.z) {
        return 1;
    }
    else {
        return 2;
    }
}

AABB Union(AABB aabb, glm::vec3 p) {
    glm::vec3 umin = glm::min(aabb.min, p);
    glm::vec3 umax = glm::max(aabb.max, p);
    return AABB{ umin, umax };
}

// SAH cost = num_triangles_left * left_box_area + num_triangles_right * right_box_area
// Determines bounding boxes that result from splitting at this position and how many
// triangles to place in each box. Once these are determined, we can calculate SAH cost
float evalSAH(Scene* scene, BVHNode* node, float queryPos, int axis)
{
    AABB leftChild = { glm::vec3{INFINITY, INFINITY, INFINITY}, glm::vec3{-INFINITY, -INFINITY, -INFINITY} };
    AABB rightChild = { glm::vec3{INFINITY, INFINITY, INFINITY}, glm::vec3{-INFINITY, -INFINITY, -INFINITY} };
    int leftCount = 0;
    int rightCount = 0;

    for (int i = node->firstTri; i < node->firstTri + node->numTris; ++i) {
        glm::vec3 centroid = scene->triangles[i].centroid;
        if (centroid[axis] < queryPos) {
            leftCount++;
            leftChild = Union(leftChild, scene->triangles[i].aabb);
        }
        else {
            rightCount++;
            rightChild = Union(rightChild, scene->triangles[i].aabb);
        }
    }
    // Calculate cost
    float cost = leftCount * leftChild.surfaceArea() + rightCount * rightChild.surfaceArea();

    return cost;
}

void calculateSAHSplit(Scene* scene, BVHNode* node, float& split, int& axis)
{
    // To find the optimal cost, we must calculate the cost of splitting along each
    // axis for each triangle contained within this node
    float optimalCost = INFINITY;
    for (int i = 0; i < 3; ++i) {
        for (int j = node->firstTri; j < node->firstTri + node->numTris; ++j) {
            float centroidPos = scene->triangles[j].centroid[i];
            float cost = evalSAH(scene, node, centroidPos, i);
            if (cost < optimalCost) {
                optimalCost = cost;
                split = centroidPos;
                axis = i;
            }
        }
    }
}

void chooseSplit(Scene* scene, BVHNode* node, float& split, int& axis)
{

#if USE_BVH_MIDPOINT
    // Find bounding box of centroids 
    AABB centroidAABB = { glm::vec3{INFINITY, INFINITY, INFINITY}, glm::vec3{-INFINITY, -INFINITY, -INFINITY} };
    for (int i = node->firstTri; i < node->firstTri + node->numTris; ++i)
        centroidAABB = Union(centroidAABB, scene->triangles[i].centroid);
    axis = maxExtent(centroidAABB.max - centroidAABB.min);
    split = (centroidAABB.min[axis] + centroidAABB.max[axis]) * 0.5f;

#elif USE_BVH_SAH
    calculateSAHSplit(scene, node, split, axis);
#endif

}

void addChildren(Scene* scene, BVHNode* node)
{
    if (node->numTris <= 2)
    {
        return;
    }

    // Choose split axis and position
    float split = 0.f;
    int axis = 0;
    chooseSplit(scene, node, split, axis);

    // Partition primitives (in-place sorting)
    int start = node->firstTri;
    int end = node->firstTri + node->numTris - 1;
    while(start <= end) {
        if (scene->triangles[start].centroid[axis] < split) {
            start++;
        }
        else {
            std::swap(scene->triangles[start], scene->triangles[end]);
            end--;
        }
    }

    // Make sure there is no empty side on partition
    int count = start - node->firstTri;
    if (count == 0 || count == node->numTris) return;

    // Set children nodes
    node->left = idx++;
    node->right = idx++;
    scene->bvh[node->left].firstTri = node->firstTri;
    scene->bvh[node->left].numTris = start - node->firstTri;
    scene->bvh[node->right].firstTri = start;
    scene->bvh[node->right].numTris = node->numTris - scene->bvh[node->left].numTris;
    node->numTris = 0;

    updateBounds(scene, node->left);
    updateBounds(scene, node->right);

    addChildren(scene, &scene->bvh[node->left]);
    addChildren(scene, &scene->bvh[node->right]);
}

void generateBVH(Scene* scene)
{
    // Resize BVH
    scene->bvh.resize(2 * scene->triangles.size() - 1);

    // Initialize root node
    BVHNode* root = &scene->bvh[0];
    root->aabb = scene->mesh_aabbs[0];
    root->firstTri = 0;
    root->numTris = scene->triangles.size();

    // Construct hierarchy
    addChildren(scene, root);
}