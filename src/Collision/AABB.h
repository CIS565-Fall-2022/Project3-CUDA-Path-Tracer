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
	HOST_DEVICE glm::vec3 center() const { return (_min + _max) * 0.5f; }
	HOST_DEVICE glm::vec3 extent() const { return (_max - _min) * 0.5f; }
	HOST_DEVICE void vertices(glm::vec3(&out)[8], bool world) const {
		glm::vec3 ex = extent(), ct = center();
		int i = 0;
		for (float x : { -ex.x, ex.x }) {
			for (float y : { -ex.y, ex.y }) {
				for (float z : { -ex.z, ex.z }) {
					if (world) {
						out[i++] = glm::vec3(x, y, z) + ct;
					} else {
						out[i++] = glm::vec3(x, y, z);
					}
				}
			}
		}
	}
};