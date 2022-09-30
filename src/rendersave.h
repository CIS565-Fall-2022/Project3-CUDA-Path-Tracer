#pragma once
#include "glm/glm.hpp"
#include "sceneStructs.h"
#include "scene.h"

// serializes the current render state to a file
bool save_state(int iter, RenderState const& state, Scene const& scene, char const* filename);

// deserializes a scene
bool read_state(char const* filename, int& iter, Scene*& scene);