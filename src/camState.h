#pragma once
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "consts.h"

struct RenderState;

// the result of 0 sanity
struct JunksFromMain {
    float zoom, theta, phi;
    glm::vec3 cameraPosition;
    glm::vec3 ogLookAt;
};

extern RenderState* g_renderState;
extern JunksFromMain g_mainJunks;
extern int width;
extern int height;

namespace CamState {
	inline glm::mat4x4 get_view() {
		glm::vec3 const& look_point = g_renderState->camera.lookAt;
		glm::vec3 const& eye = g_mainJunks.cameraPosition;
		return glm::lookAt(eye, look_point, WORLD_UP);
	}

	inline glm::mat4x4 get_proj() {
		float fov = glm::radians(g_renderState->camera.fov.y * 2.0f);
		return glm::perspective(fov, width / (float)height, 0.01f, 100.0f);
	}
	inline glm::vec3 clip_to_screen(glm::vec4 clip_coord) {
		clip_coord /= clip_coord.w;

		float w = width * 0.5f;
		float h = height * 0.5f;
		glm::vec3 ret(clip_coord.x * w, clip_coord.y * h, clip_coord.z);
		return ret;
	}
	inline glm::vec4 screen_to_world(glm::mat4x4 const& view, glm::mat4x4 const& proj, glm::vec3 screen_coord) {
		glm::vec4 ret = glm::inverse(proj) * glm::vec4(screen_coord, 1.f);
		ret /= ret.w;
		return glm::inverse(view) * ret;
	}
	inline glm::vec3 clip_to_viewport(glm::vec4 clip_coord) {
		clip_coord /= clip_coord.w;

		float w = width * 0.5f;
		float h = height * 0.5f;
		glm::vec3 ret(clip_coord.x * w, clip_coord.y * h, clip_coord.z);
		// to viewport
		ret.x = w + ret.x;
		ret.y = h - ret.y;
		return ret;
	}
};