#pragma once
#include <glm/glm.hpp>
#include "../utilities.h"

class Ray;
class Geom;
class Camera;

class AABB {
	glm::vec3 _min;
	glm::vec3 _max;

public:
	HOST_DEVICE AABB() : _min(0), _max(0) {}
	HOST_DEVICE AABB(glm::vec3 const& min, glm::vec3 const& max) : _min(min), _max(max) { }
	HOST_DEVICE glm::vec3 const& min() const { return _min; }
	HOST_DEVICE glm::vec3 const& max() const { return _max; }
};

AABB computeAABB(Geom const& geom, Camera const& cam);