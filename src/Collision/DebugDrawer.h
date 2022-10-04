#pragma once
#include "../sceneStructs.h"

class AABB;
namespace DebugDrawer {
	// void DrawLine()
	void DrawRect(float x, float y, float w, float h, color_t color);
	void DrawRect3D(glm::vec3 center, glm::vec3 extent, color_t color);
	void DrawAABB(AABB const&, color_t color);
	void DrawLine3D(glm::vec3 from, glm::vec3 to, color_t color, float thickness = 2);
};