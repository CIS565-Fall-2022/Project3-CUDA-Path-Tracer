#pragma once
#include <glm/glm.hpp>

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