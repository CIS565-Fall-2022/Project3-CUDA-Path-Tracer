#include "DebugDrawer.h"
#include <glm/glm.hpp>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../ImGui/imgui.h"
#include "../ImGui/imgui_impl_glfw.h"
#include "../ImGui/imgui_impl_opengl3.h"

#include "../consts.h"
#include "../sceneStructs.h"
#include "../camState.h"
#include <stb_image.h>

// world to screen
static glm::vec2 w2s(glm::mat4 const& model, glm::vec3 const& point) {
	glm::vec3 const& look_point = g_renderState->camera.lookAt;
	glm::vec3 const& eye = g_mainJunks.cameraPosition;
	float fov = glm::radians(g_renderState->camera.fov.y * 2.0f);

	glm::mat4 view = glm::lookAt(eye, look_point, WORLD_UP);
	glm::mat4 proj = glm::perspective(fov, width / (float)height, 0.12f, 100.0f);
	glm::vec4 tmp = CamState::get_proj() * (CamState::get_view() * (model * glm::vec4(point, 1)));
	return glm::vec2(CamState::clip_to_viewport(tmp));
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


// reference: https://github.com/ocornut/imgui/wiki/Image-Loading-and-Displaying-Examples
bool DebugDrawer::DrawImage(char const* filename, int width, int height) {
	static GLuint s_texture_id = ~0;
	static std::string s_filename;

	// Create a OpenGL texture identifier
	if (s_texture_id == ~0) {
		glGenTextures(1, &s_texture_id);
	}

	if (s_filename != filename) {
		s_filename = filename;

		// Load from file
		int image_width = 0;
		int image_height = 0;
		unsigned char* image_data = stbi_load(filename, &image_width, &image_height, NULL, 4);
		if (image_data == NULL)
			return false;

		glBindTexture(GL_TEXTURE_2D, s_texture_id);
		// Setup filtering parameters for display
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same

		// Upload pixels into texture
#if defined(GL_UNPACK_ROW_LENGTH) && !defined(__EMSCRIPTEN__)
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
#endif
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);

		stbi_image_free(image_data);
	}

	ImGui::Image((void*)s_texture_id, ImVec2(width, height));
	return true;
}