#pragma once
#include "glm/glm.hpp"
#include "sceneStructs.h"
#include "scene.h"

/// <summary>
/// serializes the current render state to a file
/// </summary>
/// <param name="state">render state</param>
/// <param name="scene">scene data</param>
/// <param name="dev_image">device radiance buffer</param>
/// <param name="filename">path to the file</param>
void save_state(RenderState const& state, Scene const& scene, glm::vec3 const* dev_image, char const* filename);

/// <summary>
/// deserializes a render state to a file
/// </summary>
/// <param name="state">render state</param>
/// <param name="dev_image">device radiance buffer</param>
/// <param name="filename">path to the file</param>
bool read_state(RenderState& state, glm::vec3* dev_image, char const* filename);