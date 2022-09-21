#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

struct AABB {
    glm::vec3 pMin;
    glm::vec3 pMax;
};

struct BVHNode {
    AABB box;
    int geomIdx;
    int size;
};

struct BVHTableElement {
};