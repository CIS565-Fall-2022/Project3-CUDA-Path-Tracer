#include "lbvh.h"

bool morton_sort(MortonCode a, MortonCode b) {
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

// Helper functions to calculate a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1] (credit: Tero Karras)
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

unsigned int mortonCode3D(glm::vec3 centroid)
{
    float x = min(max(centroid.x * 1024.0f, 0.0f), 1023.0f);
    float y = min(max(centroid.y * 1024.0f, 0.0f), 1023.0f);
    float z = min(max(centroid.z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

void computeMortonCodes(Scene* scene, const AABB& sceneAABB) {
    for (int i = 0; i < scene->triangles.size(); i++) {
        // Find centroid of triangle's bounding box
        glm::vec3 centroid = 0.5f * scene->triangles[i].aabb.min + 0.5f * scene->triangles[i].aabb.max;

        // Normalize centroid w.r.t. scene bounding box
        glm::vec3 norm_centroid = (centroid - sceneAABB.min) / (sceneAABB.max - sceneAABB.min);

        // Calculate Morton code and add to list
        unsigned int mcode = mortonCode3D(norm_centroid);
        scene->triangles[i].mcode = mcode;
        scene->mcodes.push_back(mcode);
    }
}

void sortMortonCodes(Scene* scene) {
    std::vector<unsigned int> mcodes_copy = scene->mcodes;
    thrust::sort(thrust::host, mcodes_copy.begin(), mcodes_copy.end());
    scene->mcodes = mcodes_copy;
}

// Determine range functions
int delta(unsigned int* sortedMCodes, int N, int i, int j) {
    // Range check
    if (i < 0 || j < 0 || i >= N || j >= N) {
        return -1;
    }

    // if same - return 32 + lzcnt(i ^ j) ?
    // Is this 31 - lzcnt?
    return __lzcnt(sortedMCodes[i] ^ sortedMCodes[j]);
}

int sign(unsigned int* sortedMCodes, int N, int i) {
    int diff = delta(sortedMCodes, N, i, i + 1) - delta(sortedMCodes, N, i, i - 1);
    return (diff >= 0) ? 1 : -1;
}

NodeRange determineRange(unsigned int* sortedMCodes, int triangleCount, int i) {
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

int findSplit(unsigned int* sortedMCodes, int triangleCount, NodeRange range) {
    int i = range.i;
    int j = range.j;
    int l = range.l;
    int d = range.d;
    
    // Find split position with binary search
    int deltaNode = delta(sortedMCodes, triangleCount, range.i, range.j);
    int s = 0;
    for (int t = l / 2; t >= 1; t /= 2) {
        if (delta(sortedMCodes, triangleCount, i, i + (s + t) * d) > deltaNode) {
            s = s + t;
        }
    }
    int gamma = i + s * d + min(d, 0);
    
    return gamma;
}

// TODO: make sure leaf bounding boxes are assigned in buildLBVH function
AABB assignBoundingBoxes(Scene* scene, LBVHNode* node) {

    if (!isLeaf(node)) {
        AABB leftAABB = assignBoundingBoxes(scene, &scene->lbvh[node->left]);
        AABB rightAABB = assignBoundingBoxes(scene, &scene->lbvh[node->right]);
        node->aabb = Union(leftAABB, rightAABB);
    }

    return node->aabb;
}

// Tree-building functions
void buildLBVH(Scene* scene, int triangleCount) {
    // Resize LBVH
    int numLeaf = triangleCount;
    int numInternal = triangleCount - 1;
    scene->lbvh.resize(numLeaf + numInternal);

    // Initialize leaf nodes
    for (int i = 0; i < numLeaf; ++i) {
        LBVHNode leafNode;
        leafNode.objectId = i; // TODO: this points to the wrong data
        leafNode.left = 0xFFFFFFFF;
        leafNode.right = 0xFFFFFFFF;
        scene->lbvh[i] = leafNode;
    }

    // Initialize internal nodes
    for (int j = numLeaf; j < numLeaf + numInternal; ++j) {
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

        internalNode.objectId = j;
        internalNode.left = leftChild;
        internalNode.right = rightChild;
        scene->lbvh[j] = internalNode;
    }
}

void unitTest(Scene* scene)
{
    // Morton code computation
    computeMortonCodes(scene, scene->sceneAABB);

    // Sort Morton codes
    sortMortonCodes(scene);
    
    for (int i = 0; i < scene->mcodes.size(); i++) {
        std::cout << scene->mcodes[i] << std::endl;
    }

    std::vector<unsigned int> test_sorted_mcodes = { 0b00001, 0b00010, 0b00100, 0b00101, 0b10011, 0b11000, 0b11001, 0b11110 };
    scene->mcodes = test_sorted_mcodes;
    // Test common prefix function
    int commonPrefix = delta(test_sorted_mcodes.data(), test_sorted_mcodes.size(), 5, 6);

    // Test tree building
    buildLBVH(scene, test_sorted_mcodes.size());
    int j = 0;
    return;
}