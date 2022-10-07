#include "DebugDrawer.h"
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../sceneStructs.h"
#include "../ImGui/imgui.h"
#include "../ImGui/imgui_impl_glfw.h"
#include "../ImGui/imgui_impl_opengl3.h"
#include "../consts.h"

extern RenderState* g_renderState;
extern JunksFromMain g_mainJunks;
extern int width;
extern int height;

// world to screen
static glm::vec2 w2s(glm::mat4 const& model, glm::vec3 const& point) {
	glm::vec2 const& pixel_len = g_renderState->camera.pixelLength;
	glm::vec3 const& look_point = g_renderState->camera.lookAt;
	glm::vec3 const& eye = g_mainJunks.cameraPosition;
	float fov = glm::radians(g_renderState->camera.fov.y * 2.0f);

	glm::mat4 view = glm::lookAt(eye, look_point, WORLD_UP);
	glm::mat4 proj = glm::perspective(fov, width / (float)height, 0.12f, 100.0f);
	glm::vec4 tmp = proj * (view * (model * glm::vec4(point, 1)));
	tmp /= tmp.w;

	float w = width * 0.5f;
	float h = height * 0.5f;
	glm::vec2 ret(tmp.x * w, tmp.y * h);
	// to viewport
	ret.x = w + ret.x;
	ret.y = h - ret.y;
	return ret;
}

void DebugDrawer::DrawRect(float x, float y, float w, float h, color_t color) {
	ImGui::GetForegroundDrawList()->AddRect(ImVec2(x, y - 1), ImVec2(x + w, y + h),
		ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1)), 0, 0);
}

void DebugDrawer::DrawAABB(AABB const& aabb, color_t color) {
#ifdef DrawAABB_IMPL1
	glm::vec3 verts[8];
	aabb.vertices(verts, false);

	glm::mat4 model = glm::translate(aabb.center());
	glm::vec2 points[8];
	for (int i = 0; i < 8; ++i) {
		points[i] = w2s(model, verts[i]);
	}
	int faces[6][4] = { {0,2,3,1}, {4,6,7,5}, {0,4,5,1}, {2,6,7,3}, {0,4,6,2}, {1,5,7,3} };
	ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 4; ++j) {
			int lst = j == 0 ? 3 : j - 1;
			glm::vec2 from = points[faces[i][lst]];
			glm::vec2 to = points[faces[i][j]];

			ImGui::GetForegroundDrawList()->AddLine(ImVec2(from.x, from.y), ImVec2(to.x, to.y), col);
		}
	}
#else
	DrawRect3D(aabb.center(), aabb.extent(), color);
#endif // IMPL1
}

void DebugDrawer::DrawRect3D(glm::vec3 center, glm::vec3 extent, color_t color) {
	glm::mat4 model = glm::translate(center);
	glm::vec2 points[8]; int i = 0;

	for (float x : { -extent.x, extent.x }) {
		for (float y : { -extent.y, extent.y }) {
			for (float z : { -extent.z, extent.z }) {
				points[i++] = w2s(model, glm::vec3(x, y, z));
			}
		}
	}
	int faces[6][4] = { {0,2,3,1}, {4,6,7,5}, {0,4,5,1}, {2,6,7,3}, {0,4,6,2}, {1,5,7,3} };
	ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 4; ++j) {
			int lst = j == 0 ? 3 : j - 1;
			glm::vec2 from = points[faces[i][lst]];
			glm::vec2 to = points[faces[i][j]];

			ImGui::GetForegroundDrawList()->AddLine(ImVec2(from.x, from.y), ImVec2(to.x, to.y), col);
		}
	}
}

void DebugDrawer::DrawLine3D(glm::vec3 from, glm::vec3 to, color_t color, float thickness) {
	glm::vec2 from2D = w2s(glm::mat4(1), from);
	glm::vec2 to2D = w2s(glm::mat4(1), to);
	ImU32 col = ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, 1));
	ImGui::GetForegroundDrawList()->AddLine(ImVec2(from2D.x, from2D.y), ImVec2(to2D.x, to2D.y), col, thickness);
}