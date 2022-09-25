#pragma once

#include <iostream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/type_ptr.hpp>

struct AABB {
    AABB() : pMin(FLT_MAX), pMax(-FLT_MAX) {}

    AABB(glm::vec3 pMin, glm::vec3 pMax) : pMin(pMin), pMax(pMax) {}

    AABB(glm::vec3 va, glm::vec3 vb, glm::vec3 vc) :
        pMin(glm::min(glm::min(va, vb), vc)), pMax(glm::max(glm::max(va, vb), vc)) {}

    AABB(const AABB& a, const AABB& b) :
        pMin(glm::min(a.pMin, b.pMin)), pMax(glm::min(a.pMax, b.pMax)) {}

    glm::vec3 center() const {
        return (pMin + pMax) * .5f;
    }

    float surfaceArea() const {
        glm::vec3 size = pMax - pMin;
        return 2.f * (size.x * size.y + size.y * size.z + size.z * size.x);
    }

    /**
    * Returns 0 for X, 1 for Y, 2 for Z
    */
    int longestAxis() const {
        glm::vec3 size = pMax - pMin;
        if (size.x < size.y) {
            return size.y > size.z ? 1 : 2;
        }
        else {
            return size.x > size.z ? 0 : 2;
        }
    }

    glm::vec3 pMin;
    glm::vec3 pMax;
};

class BVH {
public:
};